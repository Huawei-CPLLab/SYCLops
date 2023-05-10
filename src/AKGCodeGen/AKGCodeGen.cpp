//===-- AKGCodeGen.cpp - AKG CodeGen Definitions ----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "AKGCodeGen/AKGCodeGen.h"
#include "Util/Matcher.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/PatternMatch.h"
#include "llvm/Support/FileSystem.h"
#include <string>

using namespace llvm;
using namespace converter;
using namespace llvm::PatternMatch;
using std::string;

#define DEBUG_TYPE "akg-converter"

AKGCodeGen::AKGCodeGen(LLVMContext &Ctx) : CodeGen(Ctx) {}

/// Codegen specific reset for internal variables. Called just before converting
/// a function.
void AKGCodeGen::resetCodeGen() {
  CodeOutput.clear();
  StmtMap.clear();
  SkipInstrSet.clear();
  Builder.clear();
  InvertCmp = false;
}

/// Get the generated AKG IR as a result of conversion
string &AKGCodeGen::getCodeOutput() { return this->CodeOutput; }

/// Given an instruction, check to see if it needs to be handled, and generate
/// corresponding akg IR for it.
/// In case of AKG, we are only handling the following instructions:
///   - store
///   - phi
///   - call
/// This is where the actual instructions will get recursively generated
void AKGCodeGen::genInstruction(Instruction *I) {
  if (SkipInstrSet.contains(I))
    return;

  if (auto *PHI = dyn_cast<PHINode>(I)) {
    LLVM_DEBUG(dbgs() << "Generating PHI Instruction for "; PHI->dump());
    genPHINode(PHI);
  } else if (auto *SI = dyn_cast<StoreInst>(I)) {
    LLVM_DEBUG(dbgs() << "Generating Store Instruction for "; SI->dump());
    genStoreInst(SI);
  } else if (auto *II = dyn_cast<IntrinsicInst>(I)) {
    LLVM_DEBUG(dbgs() << "Generating Intrinsic Instruction for "; II->dump());
    genIntrInst(II);
  }
}

/// Helper recursive method that generates BB's so that dependant instructions
/// can be generated in order
void AKGCodeGen::genBlock(BasicBlock *BB) {
  LLVM_DEBUG(dbgs() << "Generating instructions for block ";
             BB->printAsOperand(dbgs()); dbgs() << "\n");
  for (Instruction &I : *BB)
    genInstruction(&I);
  LLVM_DEBUG(
      dbgs() << "Finish generating block "
                "========================================================\n");

  auto *DTNode = DT->getNode(BB);
  BasicBlock *LastChild = nullptr;
  for (auto *ChildNode : DTNode->children()) {
    BasicBlock *ChildBB = ChildNode->getBlock();
    if (ChildBB->getUniquePredecessor() == BB)
      genBlock(ChildBB);
    else {
      assert(!LastChild && "Block contains more than 3 children?");
      LastChild = ChildBB;
    }
  }
  if (LastChild)
    genBlock(LastChild);
}

/// Given a basic block, parses all the instructions within it and generate the
/// AKGBuilder::Block
void AKGCodeGen::parseBlocks() { genBlock(&F->getEntryBlock()); }

/// Generate loops performing the memset operation
void AKGCodeGen::genMemSetOrCpy(IntrinsicInst *II, bool IsCpy) {
  Value *PtrOperand = II->getOperand(0);
  auto *ValToSet = II->getOperand(1);
  auto *SizeToSet = dyn_cast<ConstantInt>(II->getOperand(2));

  assert(SizeToSet && "Does not support dynamic memsets yet");
  assert((IsCpy || (isa<Constant>(ValToSet) &&
                    cast<Constant>(ValToSet)->isZeroValue())) &&
         "Does not support memset to values other than 0 yet");
  Shape *S = getOrCreateShape(PtrOperand);
  const Type *ElTy = S->getElementType();
  unsigned ElSize = ElTy->getScalarSizeInBits();
  unsigned IntrElSize =
      getPointerElementType(PtrOperand)->getScalarSizeInBits();
  unsigned TripCount = SizeToSet->getZExtValue();
  TripCount /= ElSize / IntrElSize;
  unsigned DimSize = S->getDim(S->getNumDims() - 1);
  assert(DimSize >= TripCount &&
         "Does not support multi-dimensional memsets yet");
  (void)DimSize;

  // construct and insert the loop object
  auto *FL = Builder.getForLoop(II);
  Statement *IV = Builder.getStmt("i" + std::to_string(IndexCounter++));
  string LoopCond = IV->toString() + ", 0, ";
  LoopCond += std::to_string(TripCount);
  FL->setCondition(Builder.getStrRef(LoopCond));
  Builder.setInsertBlock(II->getParent());
  Builder.append(FL);

  // Get the statement for the array access to the pointer operand, and modify
  // the last subscript to include the IV of the generated loop.
  auto InsertArtificialSubscript = [&](Value *Operand) -> Statement * {
    Statement *RetVal = genOperand(Operand);
    // There are odd cases where the pointer used is directly coming from an
    // argument, thus not having any subscripts...
    Shape *OpShape = getOrCreateShape(Operand);
    while (RetVal->Subscripts.size() != OpShape->getNumDims())
      RetVal->Subscripts.push_back(Builder.getStmt("0"));

    Statement *&LastSubscript = RetVal->Subscripts.back();
    if (LastSubscript->toString() == "0")
      LastSubscript = IV;
    else {
      LastSubscript = Builder.concatStmt(LastSubscript, IV, "+");
      LastSubscript->Bracketed = true;
    }
    // Since we have made changes to the subscript of this operand, we need to
    // erase it from StmtMap to avoid getting a copy of this specific operand.
    StmtMap.erase(tracePastCastInsts(Operand));
    return RetVal;
  };

  // generate the memset assignment
  Statement *LHS = InsertArtificialSubscript(PtrOperand);
  Statement *RHS;
  if (IsCpy) {
    RHS = InsertArtificialSubscript(ValToSet);
  } else {
    if (ElTy->isFloatingPointTy())
      RHS = Builder.getStmt("0.0");
    else if (ElTy->isIntegerTy())
      RHS = Builder.getStmt("0");
    else
      llvm_unreachable("Unsupported data type in memset");
  }

  // Append the assignment into a dedicated block, and insert the block into the
  // loop.
  auto *AssignStmt = Builder.concatStmt(LHS, RHS, " = ");
  Block *Blk = Builder.getBlock(II);
  Builder.setInsertBlock(Blk);
  Builder.append(AssignStmt);
  FL->setBody(Blk);
}

/// Generate the AKG IR for some specific intrinsics, currently includes
/// memcpy and memset.
void AKGCodeGen::genIntrInst(IntrinsicInst *II) {
  switch (II->getIntrinsicID()) {
  case Intrinsic::memset:
    genMemSetOrCpy(II, false);
    break;
  case Intrinsic::memcpy:
    genMemSetOrCpy(II, true);
    break;
  default:
    llvm_unreachable("Unexpected intrinsic instruction");
  }
}

/// Generates the Statement for an operand. If the value has already been
/// generated (it exists in StmtMap), then it will return the cached result.
/// Otherwise try to generate the Value based on its type, and insert the
/// generated Statement into the StmtMap.
/// Note: There are some special cases where we fiddle with this convention of
///       mapping Values to their corresponding statements (in genPHINode and
///       genIfElseCond), where we interact with StmtMap in ways that we might
///       not expect.
Statement *AKGCodeGen::genOperand(Value *Operand) {
  Operand = tracePastCastInsts(Operand);
  auto It = StmtMap.find(Operand);
  if (It != StmtMap.end())
    return It->second;

  LLVM_DEBUG(dbgs() << "Generating as operand: "; Operand->dump());
  Statement *RetVal = nullptr;

  if (isLoopIV(Operand)) {
    RetVal = Builder.getStmt(genVar(Operand));
  } else if (auto *Inst = dyn_cast<Instruction>(Operand)) {
    RetVal = genInstructionOperand(Inst);
  } else if (isa<Constant>(Operand) || isa<Argument>(Operand)) {
    RetVal = Builder.getStmt(genVar(Operand));
  } else {
    llvm_unreachable("Unexpected operand");
  }
  LLVM_DEBUG({
    dbgs() << "\tFinished generating operand from ";
    Operand->dump();
    dbgs() << "\t" << RetVal->toString() << "\n";
  });

  // If the process of generating this operand has written to this entry inside
  // of StmtMap, we don't want to overwrite it here.
  if (!StmtMap.count(Operand))
    StmtMap[Operand] = RetVal;
  return StmtMap[Operand];
}

/// Generate the AKG IR for assignment `v1 = ...`, by using
/// genOperand/genInstructionOperand to trace back the use/def chain to generate
/// the expression being assigned
void AKGCodeGen::genStoreInst(StoreInst *SI) {
  Value *PtrVal = SI->getPointerOperand();
  Value *StoreVal = SI->getValueOperand();

  // Decide on where to store, if the store value is a LCSSA PHI node, then
  // generate the store in the definition of the values itself.
  // LLVM sunk this to outside the loop for obvious reasons, but this will
  // interfere with polyhedral analysis.
  const BasicBlock *StoreAt = SI->getParent();
  while (auto *LCSSAPhi = dyn_cast<PHINode>(tracePastCastInsts(StoreVal))) {
    if (LCSSAPhi->getNumIncomingValues() == 1) {
      StoreAt = LCSSAPhi->getIncomingBlock(0);
      StoreVal = LCSSAPhi->getIncomingValue(0);
    } else
      break;
  }

  // if LHS's shape is input, promote it to output since we are storing to it
  auto *DestShape = getOrCreateShape(PtrVal);
  auto ST = DestShape->getShapeType();
  // unset the input bit and set the output bit.
  if (ST & Shape::ShapeType::Input) {
    ST = (ST & (!Shape::Input)) | Shape::Output;
    DestShape->setShapeType(ST);
  }

  // Generate the actual assignment expression
  Statement *LHS = genOperand(PtrVal);
  Statement *RHS = genOperand(StoreVal);
  Statement *Result = Builder.concatStmt(LHS, RHS, " = ");
  Builder.setInsertBlock(StoreAt);
  Builder.append(Result);
  LLVM_DEBUG({
    string Output;
    Result->genCode(Output);
    dbgs() << "Inserting into BB:\n\t" << Output << "\n";
  });
}

/// Utility for setupCFG to avoid inserting blocks into each other multiple
/// times. For us, if a Block object already has a parent, then it means it
/// should have already been processed.
bool AKGCodeGen::hasBeenProcessed(const BasicBlock *BB) {
  auto *LB = Builder.getBlock(BB);
  return LB->Parent;
}

/// Utility for setupCFG. does what it says on the surface.
void AKGCodeGen::insertBlockIntoParent(const BasicBlock *Parent,
                                       const BasicBlock *BB) {
  auto *ChildBlk = Builder.getBlock(BB);
  Builder.setInsertBlock(Parent);
  Builder.append(ChildBlk);
}

/// Create a Loop object, inserts it into Parent, while also inserting the
/// header block of the loop into the Loop object.
void AKGCodeGen::insertLoopBlockIntoParent(const BasicBlock *Parent,
                                           const Loop *L,
                                           const BasicBlock *BB) {
  assert(BB == L->getHeader() &&
         "Expecting loop header for construction of loops");
  Builder.setInsertBlock(Parent);
  auto *LB = Builder.getBlock(BB);
  auto *FL = Builder.createForLoop(BB, this->genLoopCond(L), LB);
  // add self to parent and continue
  Builder.append(FL);
}

/// Create a IfElse object, inserts it into Parent, generates the conditional.
/// If 'IfSuccessor' is not present (nullptr), then it means that the 'if'
/// condition needs to be inverted, and the 'else' branch need to become the if.
/// If 'ElseSuccessor' is not present, then no 'else' block will be generated.
void AKGCodeGen::generateIfElse(const BasicBlock *Parent,
                                const BasicBlock *IfSuccessor,
                                const BasicBlock *ElseSuccessor, Value *Cond) {
  // If the IfSuccessor is nullptr, then the if only has an else region; thus
  // the condition will need to be inverted.
  bool Invert = false;
  if (IfSuccessor == nullptr) {
    Invert = true;
    IfSuccessor = ElseSuccessor;
    ElseSuccessor = nullptr;
  }

  Builder.setInsertBlock(Parent);
  auto *IfElse = Builder.getIfElse(Cond);
  Builder.append(IfElse);

  IfElse->setIfBody(Builder.getBlock(IfSuccessor));
  if (ElseSuccessor)
    IfElse->setElseBody(Builder.getBlock(ElseSuccessor));
  IfElse->setCondition(genIfCond(Cond, Invert));
}

/// Finalizes the IR, when it comes to AKG, this means the `attr` fields for
/// input/output and number of cores (if necessary). Also uses the Trampoline
/// Builder to build the trampoline function.
void AKGCodeGen::finalize() {
  // Intialize the Trampoline Builder
  TrampBuilder.initialize(this->F);
  // Get the wrapping block
  Block *RootBlk = Builder.getRootBlock();
  // Get a dummy attr block using the first parameter of the function
  auto *AttrBlk = Builder.getBlock(this->F->getArg(0));

  // Populate the attribute block:
  Builder.setInsertBlock(AttrBlk);
  Builder.appendStmt("attr:");

  SmallVector<std::pair<Value *, Type *>> OutputArgs, InputArgs, ParamArgs;
  SmallVector<Statement *> OutDecls, InDecls, LocalDecls, ParamDecls;

  // Generate necessary declarations shapes and populate argument mapping.
  for (auto ValShape : this->ShapeMap) {
    Shape *S = &ValShape.second;
    int32_t ST = S->getShapeType();

    // Skip range generation entirely for right now. It cannot be supported at
    // this moment.
    if (ST & Shape::Range)
      continue;

    string ShapeDecl = genShapeDecl(S);
    // if no shape declaration generated, then it is not a function parameter
    if (ShapeDecl.empty())
      continue;

    auto *DeclStmt = Builder.getStmt(ShapeDecl);
    // Argument needs to be globals followed by inputs, outputs, then scalar
    // parameters
    Value *RootVal = S->getRoot();
    Type *Ty = S->getElementType();
    // if input contains multiple dimensions, get the pointer to the element.
    if (S->getNumDims() != 0) {
      assert(isa<PointerType>(RootVal->getType()) &&
             "Expecting pointer type for pointer arguments");
      unsigned AS = S->getAddrSpace();
      Ty = S->getElementType()->getPointerTo(AS);
    }
    auto ToInsert = std::make_pair(RootVal, Ty);
    if (ST & Shape::ShapeType::Global) {
      if (S->isSimpleWrapper())
        TrampBuilder.addSyclWrapperArg(RootVal, Ty);
      else
        TrampBuilder.addPointerArg(RootVal, Ty);
      Builder.appendStmt(ShapeDecl);
    } else if (ST & Shape::ShapeType::Input) {
      InputArgs.push_back(ToInsert);
      InDecls.push_back(DeclStmt);
    } else if (ST & Shape::ShapeType::Output) {
      OutputArgs.push_back(ToInsert);
      OutDecls.push_back(DeclStmt);
    } else if (ST & Shape::ShapeType::Local) {
      LocalDecls.push_back(DeclStmt);
    } else {
      // Other parameter types are not arguments to the kernel
      continue;
    }
  } // End ShapeMap range loop

  // insert the shape declarations in order
  for (auto *Stmt : OutDecls)
    Builder.append(Stmt);
  for (auto *Stmt : InDecls)
    Builder.append(Stmt);
  for (auto *Stmt : ParamDecls)
    Builder.append(Stmt);
  for (auto *Stmt : LocalDecls)
    Builder.append(Stmt);

  // Add the trampoline arguments in the correct order
  auto addToTrampoline = [this](SmallVector<std::pair<Value *, Type *>> &Args) {
    for (auto &Pair : Args) {
      if (!Pair.first->getType()->isPointerTy()) {
        TrampBuilder.addScalarArg(Pair.first, Pair.second);
        continue;
      }
      Shape *S = getOrCreateShape(Pair.first);
      if (S->isSimpleWrapper()) {
        TrampBuilder.addSyclWrapperArg(Pair.first, Pair.second);
        continue;
      }
      TrampBuilder.addPointerArg(Pair.first, Pair.second);
    }
  };
  // Inputs
  addToTrampoline(InputArgs);
  // Then Outputs
  addToTrampoline(OutputArgs);
  // Then Parameters
  addToTrampoline(ParamArgs);

  // Single empty line required after attribute block
  Builder.appendStmt("");

  // Finally connect everything
  Builder.setInsertBlock(RootBlk);
  Builder.append(AttrBlk);
  auto *FuncBlock = Builder.getBlock(this->F);
  if (FuncBlock->Contents.empty())
    FuncBlock = Builder.getBlock(&this->F->getEntryBlock());
  Builder.append(FuncBlock);

  // emit the generated code to CodeOutput.
  Builder.dumpToString(this->CodeOutput);

  // Set the KernelName of the trampoline function to match the naming
  // convention in AKG
  std::string KernelName = "cce_func";
  if (HybridFunctionID > 0)
    KernelName = KernelName + "_" + std::to_string(HybridFunctionID);
  KernelName = KernelName + "_kernel0";
  HybridFunctionID++;
  TrampBuilder.setKernelName(KernelName);

  // Finalize the Trampoline Builder
  TrampBuilder.finalize();

  LLVM_DEBUG({
    string Debug;
    RootBlk->printStructure(Debug);
    dbgs() << "Structure:\n\n" << Debug;
    dbgs() << "\nCode Output:\n"
           << "-------------------------------------------------------------\n";
    dbgs() << CodeOutput << "\n"
           << "-------------------------------------------------------------\n";
  });
}

/// Helper for converting the types to AKG type declaration strings.
string AKGCodeGen::typeToString(const Type *Ty) const {
  if (Ty->isHalfTy())
    return "fp16";
  if (Ty->isFloatTy())
    return "fp32";
  if (Ty->isDoubleTy())
    return "fp64";
  if (Ty->isIntegerTy(8))
    return "int8";
  if (Ty->isIntegerTy(16))
    return "int16";
  if (Ty->isIntegerTy(32))
    return "int32";
  if (Ty->isIntegerTy(64))
    return "int64";
  if (auto *FVT = dyn_cast<FixedVectorType>(Ty))
    return typeToString(FVT->getElementType());

  LLVM_DEBUG(Ty->dump());
  llvm_unreachable("Unsupported data type");
}

/// Generate the shape declarations, in the form of:
/// `ShapeType` `Name`: <`Type`, `shape`>
/// e.g. `in v0: <f32, 16, 32>
string AKGCodeGen::genShapeDecl(Shape *S) const {
  int32_t ST = S->getShapeType();
  string ShapeDecl;

  // AKG expects all indices to be in i32
  if (ST & (Shape::Range | Shape::Offset))
    S->setElementType(IntegerType::getInt32Ty(F->getContext()));

  bool IsLocal = false;
  if (ST & Shape::ShapeType::Input)
    ShapeDecl = "in ";
  else if (ST & Shape::ShapeType::Output)
    ShapeDecl = "out ";
  else if (ST & Shape::ShapeType::Local) {
    ShapeDecl = "local ";
    IsLocal = true;
  } else if (ST & Shape::ShapeType::Global)
    ShapeDecl = "global ";
  else if (ST & (Shape::ShapeType::Constant | Shape::ShapeType::Index)) {
    // No need to declare constant and index vars akg
    return string();
  } else {
    LLVM_DEBUG(dbgs() << "For shape `" << S->getName()
                      << "` corresponding to Value ";
               S->getRoot()->dump());
    llvm_unreachable("Invalid ShapeType for AKG");
  }

  ShapeDecl += S->getName() + ": <";

  // Generate the type
  const Type *STy = S->getElementType();
  ShapeDecl += typeToString(STy);

  if (S->isDynamic())
    return ShapeDecl + ", ?>";

  for (unsigned Dim = 0, EndDim = S->getNumDims(); Dim < EndDim; Dim++) {
    ShapeDecl += ", " + std::to_string(S->getDim(Dim));
  }

  if (S->getNumDims() == 0) {
    if (IsLocal)
      ShapeDecl += ", 1";
    // For whatever reason, scalars also need a comma after the type decl.
    else
      ShapeDecl += ",";
  }
  ShapeDecl += ">";
  return ShapeDecl;
}

/// Generates the array accesses (i.e. square brackets) based on GEP. The
/// input value should be the pointer operand from store or load instructions
Statement *AKGCodeGen::genArrayAccess(Value *V) {
  SmallVector<Value *> Indices = {};

  // Check for the edge case where the pointer is generated from a select
  // instruction
  V = tracePastCastInsts(V);
  if (auto *SI = dyn_cast<SelectInst>(V))
    return genSelectInst(SI);

  Value *BasePtr = gatherArrayIndices(V, Indices);

  LLVM_DEBUG(dbgs() << "Generating array access for: "; BasePtr->dump());

  // For every dimension of the shape, generate an index.
  Statement *RetVal = Builder.getStmt(genVar(BasePtr));
  for (Value *Index : Indices)
    RetVal->Subscripts.push_back(genOperand(Index));

  LLVM_DEBUG({
    string Output;
    RetVal->genCode(Output);
    dbgs() << "\tCreated array access: " << Output << "\n";
  });
  return RetVal;
}

/// Handle the PHI nodes.
/// * If the PHI is an induction variable, we simply generate the 'i#'.
/// * If the PHI is a LCSSA PHI (only one incoming value), then we just use the
///   incoming value in place of PHI.
/// * If the PHI is used as a reduction, then the PHI is determined to be an
///   'alias' of the pointer value that it is being stored to. In this case we
///   generate the initialization of the PHI as a store to that pointer value,
///   and set the PHI to map to the Statement that the pointer value generates;
///   so that further uses of the PHI value from genOperand will generate the
///   array access instead.
/// * Otherwise if we failed to match the PHINode as a reduction, then we will
///   generate a local array with length 1, and generate a store to it in each
///   IncomingBlock with IncomingValue
void AKGCodeGen::genPHINode(PHINode *PHI) {
  // For Loop IV's, simply map the PHI to the induction variable generated.
  if (isLoopIV(PHI)) {
    StmtMap[PHI] = Builder.getStmt(genVar(PHI));
    return;
  }

  // If somehow this PHINode is already handled, skip
  auto StmtMapIt = StmtMap.find(PHI);
  if (StmtMapIt != StmtMap.end())
    return;

  // Special case PHI used for trip count 2 loops using this as backbranch
  // condition, where it does not have a use elsewhere.
  if (PHI->getType()->isIntegerTy(1)) {
    Loop *L = LI->getLoopFor(PHI->getParent());
    BasicBlock *Latch = L->getLoopLatch();
    assert(Latch && "Expecting loop to only have one latch");
    auto *BI = dyn_cast<BranchInst>(Latch->getTerminator());
    assert(BI && "Expecting loop latch to have branch inst as terminator");
    if (BI->getCondition() == PHI)
      return;
  }

  // LCSSA PHI, just use the incoming value
  if (PHI->getNumIncomingValues() == 1) {
    Value *IncomingVal = PHI->getIncomingValue(0);
    // Also take into account LCSSA PHI's for nested loops.
    auto *RootPHI = dyn_cast<PHINode>(IncomingVal);
    while (RootPHI && RootPHI->getNumIncomingValues() == 1) {
      IncomingVal = RootPHI->getIncomingValue(0);
      RootPHI = dyn_cast<PHINode>(IncomingVal);
    }

    // Look if this incoming value is in StmtMap (i.e. used by reduction PHI),
    // if so, then add this PHI to StmtMap as well
    StmtMapIt = StmtMap.find(IncomingVal);
    if (StmtMapIt != StmtMap.end()) {
      Statement *RetVal = StmtMapIt->second;
      StmtMap[PHI] = RetVal;
      return;
    }
    // If the incoming value is not a reduction or otherwise not generated
    // already, then just generate the value as an operand.
    StmtMap[PHI] = genOperand(IncomingVal);
    return;
  }

  Value *Alias = matchValueAliasPHI(PHI, LI, DT);

  if (Alias) {
    const Use *SingleUse = Alias->getSingleUndroppableUse();
    LLVM_DEBUG(dbgs() << "\tPHI found to alias with "; Alias->dump();
               dbgs() << "\n");
    const StoreInst *SingleUserStore = nullptr;
    // If this pointer is only used in a store, then there is no longer a need
    // to generate that store instruction.
    if (SingleUse) {
      auto *SingleUsr = SingleUse->getUser();
      if (auto *SI = dyn_cast<StoreInst>(SingleUsr))
        SingleUserStore = SI;
    }

    Statement *AliasStmt = genOperand(Alias);
    StmtMap[PHI] = AliasStmt;

    // If there's only a single store, then it should also be initialized at the
    // incoming block for initial value, outside of the loop
    if (SingleUserStore) {
      // generate a store to initialize the value.
      assert(PHI->getNumIncomingValues() == 2 &&
             "Unexpected num incoming values from PHI");
      Value *InitVal = nullptr;
      BasicBlock *Preheader = nullptr;
      Loop *L = LI->getLoopFor(PHI->getParent());
      assert(L && "Reduction PHI not a part of a loop");
      for (unsigned Idx = 0; Idx < 2; Idx++) {
        Preheader = PHI->getIncomingBlock(Idx);
        // Preheader is not a part of the loop.
        if (L->contains(Preheader))
          continue;
        InitVal = PHI->getIncomingValue(Idx);
        break;
      }
      // no need to insert init statement if the incoming value is also a PHI,
      // since that PHI should handle the initialization.
      if (isa<PHINode>(InitVal))
        return;
      Builder.setInsertBlock(Preheader);
      Statement *InitOp = genOperand(InitVal);
      Statement *InitStmt = Builder.concatStmt(AliasStmt, InitOp, " = ");
      Builder.append(InitStmt);
    }
    return;
  }

  Statement *Access = Builder.getStmt(genVar(PHI));
  Access->Subscripts.push_back(Builder.getStmt("0"));
  StmtMap[PHI] = Access;
  LLVM_DEBUG(dbgs() << "Mapping use of "; PHI->printAsOperand(dbgs());
             dbgs() << " to " << Access->toString() << "\n");

  // basically reg2mem the PHI nodes to add a temporary into each incoming
  // block, and map the incoming value to the temporary value so that the temp
  // var gets generated instead of the entire expression
  for (unsigned Idx = 0, NumIncoming = PHI->getNumIncomingValues();
       Idx < NumIncoming; Idx++) {
    Value *Val = PHI->getIncomingValue(Idx);
    BasicBlock *Block = PHI->getIncomingBlock(Idx);
    Builder.setInsertBlock(Block);
    Statement *Assignment = genOperand(Val);
    Builder.append(Builder.concatStmt(Access, Assignment, " = "));

    // Constants don't need mapping
    if (isa<Constant>(Val))
      continue;
    // if this value is already mapped to some other value, skip
    LLVM_DEBUG({
      if (StmtMap.count(Val)) {
        dbgs() << "\tTrying to map use of ";
        Val->printAsOperand(dbgs());
        dbgs() << " to " << Access->toString() << "\n\twhile already mapped to "
               << StmtMap[Val]->toString() << "\n";
      }
    });
    StmtMap[Val] = Access;
    LLVM_DEBUG(dbgs() << "Mapping use of "; Val->printAsOperand(dbgs());
               dbgs() << " to " << Access->toString() << "\n");
  }
}

/// Generates the expression of `Var1 <op> Var2`
Statement *AKGCodeGen::genBinaryOperator(const BinaryOperator *BO) {
  LLVM_DEBUG(dbgs() << "\tGenerating BinaryOperator "; BO->dump());
  Value *LHS = BO->getOperand(0);
  Value *RHS = BO->getOperand(1);

  // generate an explicit cast if any operands are casts
  // TODO: look into including this in genOperand, but could also generate
  // unwanted casts?
  auto GenOperandStmt = [&](Value *Operand) -> Statement * {
    Statement *OperandStmt = genOperand(Operand);
    auto *CI = dyn_cast<CastInst>(Operand);
    if (!CI)
      return OperandStmt;

    switch (CI->getOpcode()) {
    case Instruction::FPExt:
    case Instruction::ZExt:
    case Instruction::SExt:
    case Instruction::Trunc:
    case Instruction::FPTrunc:
      /*case Instruction::FPToSI:
      case Instruction::FPToUI:
      case Instruction::SIToFP:
      case Instruction::UIToFP:*/ // TODO: These currently crashes converter_adaptor
      {
        // can't use typeToString because of course the declaration and cast
        // uses different names (fp32 vs float32)
        Type *CastTo = Operand->getType();
        string CastStr;
        switch (CastTo->getTypeID()) {
        case Type::FloatTyID:
          CastStr = "float32";
          break;
        case Type::DoubleTyID:
          CastStr = "float64";
          break;
        case Type::HalfTyID:
          CastStr = "float16";
          break;
        default:
          return OperandStmt;
        }
        auto *CastStmt = Builder.getStmt(CastStr);
        return Builder.concatStmt(CastStmt, OperandStmt);
      }
    default:
      return OperandStmt;
    }
  };

  auto *LHSStmt = GenOperandStmt(LHS);
  auto *RHSStmt = GenOperandStmt(RHS);

  // the operator in between the statements
  string Op;
  switch (BO->getOpcode()) {
  case Instruction::BinaryOps::Add:
  case Instruction::BinaryOps::FAdd:
    Op = "+";
    break;
  case Instruction::Sub:
  case Instruction::FSub:
    Op = "-";
    break;
  case Instruction::Mul:
  case Instruction::FMul:
    Op = "*";
    break;
  case Instruction::UDiv:
  case Instruction::SDiv:
  case Instruction::FDiv:
    Op = "/";
    break;
  case Instruction::And:
    assert(BO->getType()->isIntegerTy(1) &&
           "Does not support bitwise logic operations in AKG backend");
    Op = "&&";
    break;
  case Instruction::Or:
    assert(BO->getType()->isIntegerTy(1) &&
           "Does not support bitwise logic operations in AKG backend");
    Op = "||";
    break;
  default:
    llvm_unreachable("Unexpected binary operation");
  }

  auto *Result = Builder.concatStmt(LHSStmt, RHSStmt, Op);
  // for AKG, the binary operators should be bracketed to chain into
  // compound statements
  Result->Bracketed = true;

  return Result;
}

/// Generate `<TrueValue> if <Cond> else <FalseValue>`
Statement *AKGCodeGen::genSelectInst(SelectInst *SI) {
  Statement *Cond = genOperand(SI->getCondition());
  Statement *TrueOp = genOperand(SI->getTrueValue());
  Statement *FalseOp = genOperand(SI->getFalseValue());
  Statement *IfStmt = Builder.concatStmt(TrueOp, Cond, " if ");
  return Builder.concatStmt(IfStmt, FalseOp, " else ");
}

/// Generate `(<LHS> <= <RHS>)` for icmp ule <LHS>, <RHS> etc.
/// If RHS is a constant, then we also generate a `(LHS>=0)` for the unsigned
/// comparison, as AKG has no notion of signed or unsigned integers.
Statement *AKGCodeGen::genCmpInst(const CmpInst *CI) {
  Value *LHS = CI->getOperand(0);
  Value *RHS = CI->getOperand(1);
  string Operation;
  bool IsUnsigned = false;

  switch (CI->getPredicate()) {
  case CmpInst::Predicate::ICMP_EQ:
  case CmpInst::Predicate::FCMP_OEQ:
  case CmpInst::Predicate::FCMP_UEQ:
    Operation = "==";
    break;
  case CmpInst::Predicate::ICMP_ULT:
  case CmpInst::Predicate::FCMP_ULT:
    IsUnsigned = true;
    [[fallthrough]];
  case CmpInst::Predicate::ICMP_SLT:
  case CmpInst::Predicate::FCMP_OLT:
    Operation = "<";
    break;
  case CmpInst::Predicate::ICMP_UGT:
  case CmpInst::Predicate::FCMP_UGT:
    IsUnsigned = true;
    [[fallthrough]];
  case CmpInst::Predicate::ICMP_SGT:
  case CmpInst::Predicate::FCMP_OGT:
    Operation = ">";
    break;
  case CmpInst::Predicate::ICMP_ULE:
  case CmpInst::Predicate::FCMP_ULE:
    IsUnsigned = true;
    [[fallthrough]];
  case CmpInst::Predicate::ICMP_SLE:
  case CmpInst::Predicate::FCMP_OLE:
    Operation = "<=";
    break;
  case CmpInst::Predicate::ICMP_UGE:
  case CmpInst::Predicate::FCMP_UGE:
    IsUnsigned = true;
    [[fallthrough]];
  case CmpInst::Predicate::ICMP_SGE:
  case CmpInst::Predicate::FCMP_OGE:
    Operation = ">=";
    break;
  case CmpInst::Predicate::ICMP_NE:
  case CmpInst::Predicate::FCMP_ONE:
  case CmpInst::Predicate::FCMP_UNE:
    Operation = "!=";
    break;
  default:
    llvm_unreachable("Unimplemented cmp predicate");
  }

  Statement *LOperand = genOperand(LHS);
  Statement *ROperand = genOperand(RHS);
  auto *RetVal = Builder.concatStmt(LOperand, ROperand, Operation);
  RetVal->Bracketed = true;

  if (IsUnsigned && isa<ConstantInt>(RHS)) {
    string UnsignedCmpOp = ">=";
    string UnsignedLogic = "&&";
    if (InvertCmp) {
      UnsignedCmpOp = "<";
      UnsignedLogic = "||";
    }
    Statement *UnsignedCmp =
        Builder.concatStmt(LOperand, Builder.getStmt("0"), UnsignedCmpOp);
    UnsignedCmp->Bracketed = true;
    RetVal = Builder.concatStmt(RetVal, UnsignedCmp, UnsignedLogic);
    RetVal->Bracketed = true;
  }

  return RetVal;
}

/// Generates the statement for an operand. e.g. from:
///
/// Val = fadd V0, V1
/// *Ptr = GEP *Base, idx
/// store Val, *Ptr
///
/// we can generate the `Ptr` and `Val`, such that in the resulting akg IR
/// looks like this: `Ptr[idx] = V0 + V1`
Statement *AKGCodeGen::genInstructionOperand(Instruction *Operand) {
  if (isa<GetElementPtrInst>(Operand))
    return genArrayAccess(Operand);

  // If a pointer generated by Alloca doesn't go through GEP, fill indices with
  // 0's.
  if (auto *AI = dyn_cast<AllocaInst>(Operand)) {
    Shape *S = getOrCreateShape(AI);
    Statement *RetVal = Builder.getStmt(genVar(AI));
    for (unsigned N = S->getNumDims(); N > 0; N--)
      RetVal->Subscripts.push_back(Builder.getStmt("0"));
    return RetVal;
  }

  // for load instructions, we just need to get the shape and the index.
  if (auto *LI = dyn_cast<LoadInst>(Operand))
    return genArrayAccess(LI->getPointerOperand());

  // PHIs should be handled through gen instruction, here we just handle LCSSA
  // PHIs that only have one incoming value.
  if (auto *PHI = dyn_cast<PHINode>(Operand)) {
    if (PHI->getNumIncomingValues() > 1) {
      LLVM_DEBUG(dbgs() << "[WARNING] genPHINode from genInstructionOperand\n");
      genPHINode(PHI);
      return genOperand(PHI);
    }
    return genOperand(PHI->getIncomingValue(0));
  }

  if (auto *BO = dyn_cast<BinaryOperator>(Operand))
    return genBinaryOperator(BO);

  if (auto *SI = dyn_cast<SelectInst>(Operand))
    return genSelectInst(SI);

  if (auto *CI = dyn_cast<CmpInst>(Operand))
    return genCmpInst(CI);

  if (auto *CI = dyn_cast<CallInst>(Operand)) {
    auto *FuncStmt = Builder.getStmt(CI->getCalledFunction()->getName());
    size_t NumParams = CI->arg_size();
    assert(NumParams > 0 && "Function calls with no arguments likely have side "
                            "effects that are not captured.");
    auto *ArgStmt = genOperand(CI->getArgOperand(0));
    for (unsigned Idx = 1; Idx < NumParams; Idx++) {
      auto *NewStmt = genOperand(CI->getArgOperand(Idx));
      ArgStmt = Builder.concatStmt(ArgStmt, NewStmt, ", ");
    }
    ArgStmt->Bracketed = true;
    return Builder.concatStmt(FuncStmt, ArgStmt);
  }

  if (auto *EEI = dyn_cast<ExtractElementInst>(Operand)) {
    // currently we only support extracting from the global variable used by
    // parallel_for
    unsigned Id;
    GlobalValue *GV;
    if (!matchExtractParallelId(EEI, GV, Id)) {
      llvm_unreachable("Unsupported extractelement instruction");
    }
    auto *RetVal = Builder.getStmt(genVar(GV));
    RetVal->Subscripts.push_back(Builder.getStmt(std::to_string(Id)));
    return RetVal;
  }
  llvm_unreachable("Unexpected Instruction operand");
}

/// Given a Value, find its root, construct or get a Shape from it, and simply
/// construct a statement with its name as the variable name.
/// For AKG, constant values does not need to be assigned to a variable,
/// hence we can print it out directly
string AKGCodeGen::genVar(Value *V) {
  if (auto *Const = dyn_cast<ConstantData>(V)) {
    if (auto *ConstFP = dyn_cast<ConstantFP>(Const)) {
      if (ConstFP->isZero())
        return "0.0";
      return std::to_string(ConstFP->getValue().convertToDouble());
    }
    string Printer;
    raw_string_ostream RSO(Printer);
    Const->printAsOperand(RSO, /*PrintType=*/false, this->F->getParent());
    RSO.flush();
    return Printer;
  }
  Shape *S = this->getOrCreateShape(V);
  return S->getName();
}

/// Generates the loop header (e.g. `i0, 0, 64`) for the given loop L
string AKGCodeGen::genLoopCond(const Loop *L) {
  LoopComponents &LC = LoopComponentMap[L];
  string RetVal = genOperand(LC.IV)->toString() + ", ";
  RetVal += genOperand(LC.Start)->toString() + ", ";
  RetVal += genOperand(LC.Bound)->toString();

  return RetVal;
}

/// Generates the if condition (e.g. `i0 < 16`) for an i1 condition
/// Cond, and will add the `not` operator if Inverse is true.
Statement *AKGCodeGen::genIfCond(Value *Cond, bool Inverse) {
  if (Inverse) {
    InvertCmp = true;
    Statement *RetVal = genOperand(Cond);
    auto *NotOp = Builder.getStmt("!");
    NotOp->RHS = RetVal;
    RetVal = NotOp;
    return RetVal;
  }
  InvertCmp = false;
  return genOperand(Cond);
}

/// Writes the CodeOutput to the file specified, if exists.
Error AKGCodeGen::writeToFile(const std::string &OutFile) {
  std::error_code EC;
  llvm::raw_fd_ostream FS(OutFile, EC, sys::fs::OF_Append);
  if (EC) {
    return llvm::make_error<llvm::StringError>(
        "Converter unable to write to output file", EC);
  }
  FS << CodeOutput;
  FS.close();

  return Error::success();
}