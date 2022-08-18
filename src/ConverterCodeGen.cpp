//===-- ConverterCodeGen.cpp - LLVMIR To AKGIR/MLIR -------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Contains the base class definitions for the MLIR and AKG backends. The
// CodeGen class will contain all the necessary methods to parse the LLVM IR,
// while the MLIR and AKG classes will override pure virtual methods to be
// able to generate different IR from the same information.
//
//===----------------------------------------------------------------------===//

#include "ConverterCodeGen.h"
#include "Util/Matcher.h"
#include "Util/Preprocessing.h"
#include "llvm/Analysis/AssumptionCache.h"
#include "llvm/Analysis/IVDescriptors.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/MemorySSA.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/PatternMatch.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"

#define DEBUG_TYPE "converter-codegen"

using namespace llvm;
using namespace converter;
using namespace llvm::PatternMatch;
using std::string;

// ================= CodeGen method definitions ================================

/// Registers and adds passes required by the FunctionPassManager.
CodeGen::CodeGen(LLVMContext &Ctx) : TrampBuilder(Ctx) {
  FAM.registerPass([&] { return TargetLibraryAnalysis(); });
  FAM.registerPass([&] { return TargetIRAnalysis(); });
  FAM.registerPass([&] { return LoopAnalysis(); });
  FAM.registerPass([&] { return PassInstrumentationAnalysis(); });
  FAM.registerPass([&] { return DominatorTreeAnalysis(); });
  FAM.registerPass([&] { return ScalarEvolutionAnalysis(); });
  FAM.registerPass([&] { return AssumptionAnalysis(); });
  FAM.registerPass([&] { return MemorySSAAnalysis(); });
}

/// The method that generates the arguments and the code output for a given
/// target backend. Calls specialized pure virtual functions such that MLIR
/// and AKG can emit different code with the same information. Returns
/// true if success, false otherwise.
Error CodeGen::convert(Function *F) {
  // reset the data structures
  this->reset();
  this->F = F;

  // Check the cases where the conversion of the full function does not need to
  // happen (to avoid checks for cases which we don't support but don't need to
  // convert anyway)
  if (Error E = rejectUnsupportedBuiltin(F))
    return E;

  // preprocess the function prior to conversion
  preprocess();
  LLVM_DEBUG(dbgs() << "*** IR Dump After Preprocessing ******\n"; F->dump());
  // get the required analysis results
  this->LI = &FAM.getResult<LoopAnalysis>(*F);
  this->DT = &FAM.getResult<DominatorTreeAnalysis>(*F);
  this->SE = &FAM.getResult<ScalarEvolutionAnalysis>(*F);

  // populate the LoopComponentMap
  for (Loop *L : LI->getLoopsInPreorder()) {
    if (Error E = this->parseLoop(L))
      return createError(E, "Converter failed while parsing a loop.");
  }

  // generate target representation of Blocks for every BB
  this->parseBlocks();

  // connect the Blocks in the correct order, and wrap around For/IfElse
  // operations if needed
  this->setupCFG();

  // MLIR and AKG have their own module/function/attr declarations. Call the
  // pure virtual function for that
  this->finalize();

  return Error::success();
}

/// Clears the data structures.
void CodeGen::reset() {
  LoopComponentMap.clear();
  ShapeMap.clear();
  IndexCounter = 0;
  ArgCounter = 0;
  DimCounter = 0;
  ConstCounter = 0;
  GlobalCounter = 0;
  LocalCounter = 0;

  // Run codegen specific reset for internal variables.
  resetCodeGen();
}

/// Extract necessary information from LoopInfo analysis and populates a
/// LoopComponents object. Also adds the Loop:LoopComponents pair into the
/// LoopComponentMap
Error CodeGen::parseLoop(const Loop *L) {
  LLVM_DEBUG(dbgs() << "Generating LoopComponents for loop: "
                    << L->getName().data() << "\n");

  if (!L->isLoopSimplifyForm())
    return createError("Expecting loops to be in loop simplify form");

  // Create the LoopComponents for this loop
  LoopComponents LC;

  // Find the induction variable, this is rather flimsy so we have more fallback
  // analysis if this fails
  LC.IV = L->getInductionVariable(*SE);
  InductionDescriptor ID;
  if (L->getInductionDescriptor(*SE, ID)) {
    LC.Start = ID.getStartValue();
    Value *StepSize = ID.getConstIntStepValue();
    if (!StepSize) {
      BinaryOperator *BO = ID.getInductionBinOp();
      if (BO->getOperand(0) == LC.IV)
        StepSize = BO->getOperand(1);
      else if (BO->getOperand(1) == LC.IV)
        StepSize = BO->getOperand(0);
      else
        return createError("Unhandled dynamic loop step size.");
    }
    LC.Step = StepSize;
  }

  // Edge case checking - If the failed to get the IV,
  // then it could be a special case where the loop trip count is 2
  if (!LC.IV) {
    LLVM_DEBUG(dbgs() << "Failed to find loop IV or starting value, checking "
                         "for extend 2 loop\n");
    LC.IV = findTrip2LoopIV(L);
    if (LC.IV) {
      LLVM_DEBUG(dbgs() << "Found Loop with trip count 2.\n");
      LC.Bound = ConstantInt::get(LC.IV->getType(), 2);
      LC.Start = LC.IV->getIncomingValueForBlock(L->getLoopPreheader());
      LC.IsDynamic = false;
      LC.Condition = nullptr;
      LC.Step = ConstantInt::get(LC.IV->getType(), 1);
      this->LoopComponentMap[L] = LC;
      return Error::success();
    }
    LLVM_DEBUG(dbgs() << "Failed to find loop with trip count 2.\n");
  }

  // If by this point the induction variable still has not been found, we
  // perform the final fallback analysis
  if (!LC.IV) {
    Error Err = matchLoopComponents(L, LC);
    if (Err)
      return Err;
    this->LoopComponentMap[L] = LC;
    return Err;
  }

  // Make sure the loop is in correct form
  Value *Increment = LC.IV->getIncomingValueForBlock(L->getLoopLatch());
  if (!match(Increment, m_c_Add(m_Specific(LC.IV), m_Specific(LC.Step))))
    return createError("Loop not in canonical form\n");

  // set the loop condition
  LC.Bound = matchLoopBound(L, LC.IV, LC.Step);
  if (!LC.Bound)
    return createError("Failed to match loop bound - is it in canonical form?");

  if (isa<ConstantInt>(LC.Bound) && isa<ConstantInt>(LC.Step) &&
      isa<ConstantInt>(LC.Start))
    LC.IsDynamic = false;
  else
    LC.IsDynamic = true;

  // Some assertions as a sanity check
  assert(LC.IV->getNumIncomingValues() == 2 &&
         "More than one latch/entry to loop?");
  assert(LC.IV->getIncomingValueForBlock(L->getLoopPreheader()) == LC.Start &&
         "Loop IV incoming value from preheader is not starting value");

  // Finally add the L:LC pair to the map
  this->LoopComponentMap[L] = LC;

  return Error::success();
}

/// Recursive method to iterate to the exiting block of the function, and links
/// the block tree from the bottom up.
/// Note: Does not support functions with multiple exits
void CodeGen::insertBlock(const BasicBlock *Parent, const BasicBlock *BB,
                          bool OnlyGenerateChildren) {
  LLVM_DEBUG(dbgs() << "Inserting BB `" << BB->getName() << "` into `"
                    << Parent->getName() << "`\n");

  // Check to see if this block has already been processed.
  if (!OnlyGenerateChildren && this->hasBeenProcessed(BB)) {
    LLVM_DEBUG(dbgs() << "\tBlock has already been processed, skipping\n");
    return;
  }

  // Lambda to check of the child block can be inserted into a parent block.
  auto CanInsertBlockIntoParent = [&](const BasicBlock *Parent,
                                      const BasicBlock *Child) -> bool {
    // Cannot insert a block into itself
    if (Parent == Child)
      return false;
    // If the child has already been processed, it should not be inserted again.
    if (this->hasBeenProcessed(Child))
      return false;
    return true;
  };

  const Instruction *TerInst = BB->getTerminator();
  // If this is a exiting block, insert and go back, since there are no children
  if (isa<ReturnInst>(TerInst)) {
    LLVM_DEBUG(dbgs() << "\tFound ret instruction\n");
    if (CanInsertBlockIntoParent(Parent, BB))
      this->insertBlockIntoParent(Parent, BB);
    return;
  }

  auto *Branch = dyn_cast<BranchInst>(TerInst);
  if (!Branch) {
    llvm_unreachable("Terminator of BB not branch nor return");
    return;
  }

  // Lambda to check if a BasicBlock is the header of a loop. If true then
  // return the loop that it is a header of, else return null.
  auto CheckLoopHeaderLambda = [&](const BasicBlock *MaybeHeader) -> Loop * {
    Loop *L = LI->getLoopFor(MaybeHeader);
    if (L && L->getHeader() == MaybeHeader)
      return L;
    return nullptr;
  };

  // if BB is the header of a loop, generate the loop block and insert into
  // parent
  if (Loop *L = CheckLoopHeaderLambda(BB)) {
    LLVM_DEBUG(dbgs() << "\tBB is a loop header, creating Loop\n");
    if (CanInsertBlockIntoParent(Parent, BB))
      this->insertLoopBlockIntoParent(Parent, L, BB);
  }

  // if the branch is unconditional, insert the child to the current BB, then
  // insert self into parent.
  if (Branch->isUnconditional()) {
    LLVM_DEBUG(
        dbgs() << "\tUnconditional branch, recursively inserting successor\n");
    if (CanInsertBlockIntoParent(BB, Branch->getSuccessor(0)))
      this->insertBlock(BB, Branch->getSuccessor(0));
    // Need to check for duplicate block entries
    if (CanInsertBlockIntoParent(Parent, BB))
      this->insertBlockIntoParent(Parent, BB);
    return;
  }

  // We have an if conditional branch here, check the two successors to see
  // which order to insert them in.
  BasicBlock *TrueSucc = Branch->getSuccessor(0);
  BasicBlock *FalseSucc = Branch->getSuccessor(1);

  // check if the current BB is the (unique) exiting block, in which case insert
  // the exit after the loop. The backedge could be a critial edge block that
  // would be always legal to merge, so check for that case as well
  Loop *L = LI->getLoopFor(BB);
  if (L && L->getExitingBlock() == BB) {
    LLVM_DEBUG(dbgs() << "\tAdding exiting block to loop block\n");
    // Add the children (exit block of loop) after the loop.
    auto *ExitBlock = L->getExitBlock();
    BasicBlock *BackEdgeBB = nullptr;
    if (TrueSucc == ExitBlock)
      BackEdgeBB = FalseSucc;
    else if (FalseSucc == ExitBlock)
      BackEdgeBB = TrueSucc;
    else
      llvm_unreachable("Exit block of loop not a successor of loop latch");
    // Need to check if the BB has already been processed.
    if (CanInsertBlockIntoParent(Parent, BB))
      this->insertBlockIntoParent(Parent, BB);
    if (CanInsertBlockIntoParent(BB, BackEdgeBB))
      this->insertBlock(BB, BackEdgeBB);
    this->insertBlock(L->getLoopPreheader(), ExitBlock);
    return;
  }

  // Here we handle if-else branching
  Value *Cond = Branch->getCondition();
  BasicBlock *ThenBB;
  DomTreeNode *Node = DT->getNode(BB);
  switch (Node->getNumChildren()) {
  // The case of 1 children means that the then-block of this if condition is
  // shared with another if condition higher up in the nest, hence handle like a
  // regular if-then.
  case 1:
  case 2: {
    LLVM_DEBUG(dbgs() << "\tBB is the header of if-then block.\n");
    // Find the successor that is the if branch, we do this by checking to see
    // which block has a single predecessor being the current block, and we take
    // that block as the If body.
    BasicBlock *IfBB;
    BasicBlock *ElseBB;
    if (TrueSucc->getSinglePredecessor() == BB) {
      IfBB = TrueSucc;
      ElseBB = nullptr;
      ThenBB = FalseSucc;
    } else if (FalseSucc->getSinglePredecessor() == BB) {
      // Here the if block is taken on the false branch.
      IfBB = nullptr;
      ElseBB = FalseSucc;
      ThenBB = TrueSucc;
    } else
      llvm_unreachable("Neither successor of if-else has single predecessor");

    LLVM_DEBUG(dbgs() << "\tThen block determined to be " << ThenBB->getName()
                      << "\n";);
    // Generate the if statement into the BB
    generateIfElse(BB, IfBB, ElseBB, Cond);
    // Insert the then block into the BB. This must be done after generating the
    // if statement so it comes after.
    this->insertBlock(BB, ThenBB);
    // Recursivly generate the If/ElseBB's children. This must be done after
    // inserting the then block because the If/ElseBB always has the ThenBB as a
    // child node and will generate it inside the if body if you are not
    // careful.
    if (IfBB)
      this->insertBlock(IfBB, IfBB, /*OnlyGenerateChildren*/ true);
    if (ElseBB)
      this->insertBlock(ElseBB, ElseBB, /*OnlyGenerateChildren*/ true);
    break;
  }
  case 3: {
    LLVM_DEBUG(dbgs() << "\tBB is the header of if-else-then block.\n");
    // Identify the then-block, and insert after the if-else
    for (auto *Child : Node->children()) {
      BasicBlock *ChildBB = Child->getBlock();
      if (ChildBB != TrueSucc && ChildBB != FalseSucc) {
        ThenBB = ChildBB;
        break;
      }
    }
    LLVM_DEBUG(dbgs() << "\tThen block determined to be " << ThenBB->getName()
                      << "\n");
    // Generate the if statement into the BB
    generateIfElse(BB, TrueSucc, FalseSucc, Cond);
    // Insert the then block into the BB. This must be done after generating the
    // if statement so it comes after.
    this->insertBlock(BB, ThenBB);
    // Recursivly generate the TrueSucc and FalseSucc's children. This must be
    // done after inserting the then block because the IfBB always has the
    // ThenBB as a child node and will generate it inside the if body if you are
    // not careful.
    this->insertBlock(TrueSucc, TrueSucc, /*OnlyGenerateChildren*/ true);
    this->insertBlock(FalseSucc, FalseSucc, /*OnlyGenerateChildren*/ true);
    break;
  }
  default:
    LLVM_DEBUG(dbgs() << "For node defined by BB: " << BB->getName() << "\n");
    LLVM_DEBUG(dbgs() << "If-Then branching has " << Node->getNumChildren()
                      << " children.\n");
    llvm_unreachable("Code contains unexpected control flow");
  }
  // If the BB has not been inserted to Parent by this point, insert it.
  if (CanInsertBlockIntoParent(Parent, BB))
    this->insertBlockIntoParent(Parent, BB);
}

/// Will traverse the BBs and link together the Blocks together to form the
/// proper control flow of the generated code.
void CodeGen::setupCFG() {
  LLVM_DEBUG(dbgs() << "Connecting BBs according to control flow\n"
                    << ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n");
  // The entry block of the function will be used as a wrapper for the function
  // block.
  BasicBlock *Entry = &this->F->getEntryBlock();
  // Since the function block does not necessarily have a parent block, only
  // generate its children.
  this->insertBlock(Entry, Entry, /*OnlyGenerateChildren*/ true);
}

// Specific Pattern Matching Methods=========================================

/// This method will try to get the parallel_for id from the ExtractElementInst,
/// currently only used for parallel_for.
bool CodeGen::matchExtractParallelId(const ExtractElementInst *EEI,
                                     GlobalValue *&GV, unsigned &Id) {
  Value *LoadVal;
  uint64_t ExtractIdx;
  if (!match(EEI,
             m_ExtractElt(m_Load(m_Value(LoadVal)), m_ConstantInt(ExtractIdx))))
    return false;
  LoadVal = tracePastCastInsts(LoadVal);

  // TODO: Add more specific check for global parallel_for ID
  if ((GV = dyn_cast<GlobalValue>(LoadVal))) {
    Id = ExtractIdx;
    return true;
  }
  return false;
}

/// Try to pattern match the linearization of dynamic buffers:
/// Arr [((i0 * Range[1]) + i1 * Range[2]) + i2]       =>
/// Arr[i0][i1][i2]
bool CodeGen::matchBufferAccess(Value *Index, SmallVector<Value *> &Indices) {
  Value *MulLHS = nullptr, *MulRHS = nullptr;
  Value *LastIdx;
  // Lambda: Given a multiply, figure out which operand corresponds to an
  // dimension/range, and use the other operand as the index of the previous
  // dimension.
  auto IdentifyIndex = [this, &MulLHS, &MulRHS, &Index]() -> Value * {
    // Require LHS and RHS to be actual values.
    if (!(MulLHS && MulRHS))
      return nullptr;
    Shape *LS = getOrCreateShape(MulLHS);
    int LST = LS ? LS->getShapeType() : Shape::Unset;
    Shape *RS = getOrCreateShape(MulRHS);
    int RST = RS ? RS->getShapeType() : Shape::Unset;
    int ValidType = Shape::Range;
    if (LST & ValidType)
      return MulRHS;
    if (RST & ValidType)
      return MulLHS;
    LLVM_DEBUG(dbgs() << "\tFailed to recognize linearized index from ";
               Index->dump());
    return nullptr;
  }; // End Lambda

  // Stack to reverse the indices with
  SmallVector<Value *> IdxStack;
  bool Recognized = false;
  while (matchLastLinearizedDim(Index, MulLHS, MulRHS, LastIdx)) {
    IdxStack.push_back(LastIdx);
    // Try to figure out which of Mul operands is the dim and which is the
    // previous dimension
    Index = IdentifyIndex();
    if (!Index)
      return false;
    Recognized = true;
  } // end while match success

  // Add the last dimension
  Index = IdentifyIndex();
  if (!Index)
    return false;
  Indices.push_back(Index);
  // Reverse the previous indices
  while (!IdxStack.empty())
    Indices.push_back(IdxStack.pop_back_val());
  return Recognized;
}

// Internal Helper Methods =====================================================

/// Checks if the value is an index variable of a loop:
bool CodeGen::isLoopIV(const Value *V) {
  // index variables are always phi instructions
  auto *PHI = dyn_cast<PHINode>(V);
  if (!PHI)
    return false;

  // find the loop containing this PHI instruction. Since we checked for
  // canonical loops in parseLoop(), the IV PHI should always be in the header
  Loop *L = LI->getLoopFor(PHI->getParent());
  if (!L)
    return false;

  if (LoopComponentMap[L].IV == PHI)
    return true;
  return false;
}

/// Checks for a special case where a loop has bound of 2 but
/// InductionDescriptor cannot identify IV. Returns the IV if found, otherwise
/// nullptr.
PHINode *CodeGen::findTrip2LoopIV(const Loop *L) {
  // This should have two PHINodes in the header
  PHINode *IVPhi = nullptr, *CondPhi = nullptr;
  BasicBlock *Preheader = L->getLoopPreheader();
  BasicBlock *Latch = L->getLoopLatch();
  for (Instruction &I : *L->getHeader()) {
    if (CondPhi && IVPhi)
      break;
    if (auto *PHI = dyn_cast<PHINode>(&I)) {
      if (PHI->getNumIncomingValues() != 2)
        continue;
      // Both phi nodes should have one incoming edge and a back edge
      BasicBlock *LBlk = PHI->getIncomingBlock(0);
      BasicBlock *RBlk = PHI->getIncomingBlock(1);
      Value *EntryVal, *LatchVal;
      if (LBlk == Latch && RBlk == Preheader) {
        LatchVal = PHI->getIncomingValue(0);
        EntryVal = PHI->getIncomingValue(1);
      } else if (RBlk == Latch && LBlk == Preheader) {
        LatchVal = PHI->getIncomingValue(1);
        EntryVal = PHI->getIncomingValue(0);
      } else
        continue;

      // Both incoming values should be constants
      auto *EntryConst = dyn_cast<ConstantInt>(EntryVal);
      auto *LatchConst = dyn_cast<ConstantInt>(LatchVal);
      if (!(EntryConst && LatchConst))
        continue;

      // Check for CondPhi, with additional checks just to be extra safe: if
      // this phi is used as the condition of the backbranch and the two
      // incoming values are different
      if (cast<BranchInst>(Latch->getTerminator())->getCondition() == PHI &&
          EntryConst->getZExtValue() != LatchConst->getZExtValue()) {
        CondPhi = PHI;
        LLVM_DEBUG(dbgs() << "\tFound backbranch condition phi ";
                   CondPhi->dump());
        continue;
      }

      // Currently only support going from 0 to 1:
      // TODO: Maybe unroll this loop in preprocessing to allow for arbitrary
      // IV's
      if (EntryConst->isZero() && LatchConst->isOne()) {
        IVPhi = PHI;
        LLVM_DEBUG(dbgs() << "\tFound IV phi "; IVPhi->dump());
      }
    }
  }

  if (CondPhi)
    return IVPhi;
  return nullptr;
}

/// Get the Value from which we can construct/obtain a Shape object from.
///
/// Currently all CastInst are skipped and the source Value used instead, and
/// Arguments, GlobalValues, PHINodes, AllocaInsts, SelectInsts (?) and
/// CallInsts are returned. If we encounter anything that is not expected, then
/// we return a nullptr.
///
/// Note: SelectInst probably doesn't belong here. We need to make a
/// preprocessing pass that hoists GEP->select->GEP into GEP->select
llvm::Value *CodeGen::getRoot(llvm::Value *V) {
  while (1) {
    // skip past casting operator/instructions
    V = tracePastCastInsts(V);

    if (llvm::isa<Argument, GlobalValue, PHINode, AllocaInst, SelectInst,
                  CallInst>(V))
      return V;

    // The root of any llvm::Value that is a pointer is the Value that created
    // the pointer originally (looking past any GEPs, loads, or casts).
    if (auto *GEP = dyn_cast<GetElementPtrInst>(V)) {
      V = GEP->getPointerOperand();
    } else if (auto *GEPOp = dyn_cast<GEPOperator>(V)) {
      V = GEPOp->getPointerOperand();
    } else if (auto *LI = dyn_cast<LoadInst>(V)) {
      V = LI->getPointerOperand();
    } else {
      LLVM_DEBUG(dbgs() << "No root for "; V->dump());
      return nullptr;
    }
  }
}

/// Generate a Shape from the root value of SourceVal. The root can be
/// different between MLIR and AKG.
Shape *CodeGen::getOrCreateShape(Value *SourceVal) {
  Value *RootVal = getRoot(SourceVal);
  if (!RootVal)
    return nullptr;
  Type *RootTy = RootVal->getType();
  // check if shape already exists for this value
  auto VarMapIter = ShapeMap.find(RootVal);
  if (VarMapIter != ShapeMap.end())
    return &VarMapIter->second;

  // insert the shape into the map and return the pointer to the Shape
  auto MapIter = ShapeMap.insert(std::make_pair(RootVal, Shape())).first;
  Shape *RetVal = &MapIter->second;
  RetVal->setRoot(RootVal);

  // TODO: move name related fields into AKGCodeGen since MLIR does not need
  // them.
  string Name;
  auto ST = Shape::ShapeType::Unset;

  // Create a new varshape
  if (isLoopIV(RootVal)) {
    Name = "i" + std::to_string(IndexCounter++);
    ST = Shape::ShapeType::Index;
  } else if (isa<GlobalValue>(RootVal)) {
    // globals have to come before constants since it is a subclass of constant
    Name = "g" + std::to_string(GlobalCounter++);
    ST = Shape::ShapeType::Global;
  } else if (isa<Constant>(RootVal)) {
    Name = "c" + std::to_string(ConstCounter++);
    ST = Shape::ShapeType::Constant;
  } else if (isa<Instruction>(RootVal)) {
    Name = "t" + std::to_string(LocalCounter++);
    ST = Shape::ShapeType::Local;
  } else {
    Name = "v" + std::to_string(ArgCounter++);
    // the shape type of arguments are set to input initially, and will be
    // changed to output/dim later.
    ST = Shape::ShapeType::Input;
  }
  RetVal->setName(Name);

  auto IsSyclHalf = [&](const Type *Ty) {
    if (auto *CheckHalfStruct = dyn_cast<StructType>(Ty)) {
      if (CheckHalfStruct->getNumElements() == 1 &&
          CheckHalfStruct->getElementType(0)->isHalfTy()) {
        return true;
      }
    }
    return false;
  };

  // To build the dimensions for ndbuffer and usmbuffer as we go, we need to
  // keep track of the index of the operand that GEP uses to find the
  // dimensions. This will be used in the gatherArrayIndices() method.
  unsigned GEPIdx = 0;

  // Set the type of the source value
  if (auto *PtrTy = dyn_cast<PointerType>(RootTy)) {
    RootTy = PtrTy->getElementType();
    RetVal->setAddrSpace(PtrTy->getAddressSpace());

    // Ptrs of single value type inferred to be dynamic
    if ((RootTy->isSingleValueType() && !RootTy->isVectorTy()) ||
        IsSyclHalf(RootTy)) {
      // unless its an alloca instruction...
      if (!isa<AllocaInst>(RootVal))
        RetVal->setDynamic(true);
      // Add a dummy dimension of size 1 alongside the required GEPIdx
      RetVal->addDim(1, GEPIdx);
    }
    GEPIdx++;
  }

  // Check if the root value is a simple wrapper. For now we define a simple
  // wrapper as a struct that contains a single pointer. This information will
  // be used to help generate the trampoline calls.
  if (auto *StructTy = dyn_cast<StructType>(RootTy))
    if (StructTy->getElementType(0)->isPointerTy())
      RetVal->setSimpleWrapper(true);

  // Unwrap the type structs that Sycl made to get the element type
  while (1) {
    if (auto *StructTy = dyn_cast<StructType>(RootTy)) {
      // For struct types, simply increment the GEPIdx and continue to the
      // contained types. Currently we do not support struct types with more
      // than one value inside.
      assert(StructTy->getNumElements() == 1 &&
             "More than one element in struct, not a simple wrapper, is this "
             "an sycl type?");
      RootTy = StructTy->getElementType(0);
    } else if (auto *ArrTy = dyn_cast<ArrayType>(RootTy)) {
      // For array types, add the dimension of the array to the shape and
      // continue
      RetVal->addDim(ArrTy->getNumElements(), GEPIdx);
      RootTy = ArrTy->getElementType();
    } else if (auto *FVTy = dyn_cast<FixedVectorType>(RootTy)) {
      RetVal->addDim(FVTy->getNumElements(), GEPIdx);
      RootTy = FVTy->getElementType();
      break;
    } else if (auto *PTy = dyn_cast<PointerType>(RootTy)) {
      // For pointer types we get the type pointed to by the pointer and reset
      // the GEPIdx, since the GEP will most likely use the pointer value as
      // base.
      RootTy = PTy->getElementType();
      // The address space will need to be set based on the last pointer found.
      RetVal->setAddrSpace(PTy->getAddressSpace());
      // If the pointer is contained within a wrapper struct, then it must be
      // dynamic as well.
      RetVal->setDynamic(true);
      GEPIdx = 0;
    } else if (RootTy->isSingleValueType()) {
      // Pointers with a single value root type are arrays of length 1.
      if (RetVal->getNumDims() == 0 && isa<PointerType>(RootVal->getType()))
        RetVal->addDim(1, 0);
      // Here is a superset of supported base types currently:
      if (isa<IntegerType>(RootTy) || RootTy->isFloatingPointTy())
        break;
      llvm_unreachable("CreateVar encountered unexpected type");
    } else {
      LLVM_DEBUG(dbgs() << "When trying to parse type: "; RootTy->dump());
      llvm_unreachable("CreateVar encountered unexpected type");
    }
    GEPIdx++;
  }
  RetVal->setElementType(RootTy);
  RetVal->setShapeType(ST);

  LLVM_DEBUG({
    dbgs() << "Creating new";
    if (RetVal->isDynamic())
      dbgs() << " dynamic";
    dbgs() << " Shape `" << Name << "` for `";
    SourceVal->printAsOperand(dbgs());
    dbgs() << "`\n\twith root `";
    RootVal->printAsOperand(dbgs());
    dbgs() << "`\n\tand shape ";
    for (unsigned Dim = 0; Dim < RetVal->getNumDims(); Dim++)
      dbgs() << RetVal->getDim(Dim) << "x";
    RetVal->getElementType()->dump();
  });
  return RetVal;
}

/// Given an Operand, trace back to the GEP that created it and gather its
/// indices. This function returns the pointer created by the GEP.
Value *CodeGen::gatherArrayIndices(Value *Operand,
                                   SmallVector<Value *> &Indices) {
  // Get the base shape of the operand
  Shape *BaseShape = getOrCreateShape(Operand);
  if (!BaseShape)
    return nullptr;

  // GEPs can be chained together. To handle this, we traverse backwards for
  // each GEP instruction and collect the operands into one vector to pretend
  // like it is one GEP instruction.
  SmallVector<Value *> GEPOperands = {};
  while (1) {
    // Trace past any cast instructions
    Operand = tracePastCastInsts(Operand);
    // If the current Operand is a GEP, get the operands
    User *GEP = dyn_cast<GetElementPtrInst>(Operand);
    if (!GEP)
      GEP = dyn_cast<GEPOperator>(Operand);
    if (!GEP)
      break;
    // If this is a GEP feeding into another GEP, erase the first operand of the
    // second GEP if it is zero, else crash.
    if (GEPOperands.size() > 0) {
      auto *Const = dyn_cast<Constant>(GEPOperands[0]);
      if (Const && Const->isZeroValue()) {
        GEPOperands.erase(GEPOperands.begin());
      } else {
        llvm_unreachable("When GEP A feeds into GEP B, GEP B is expected to "
                         "start with a zero value.");
      }
    }
    // GEP operands are inserted in-order, while the GEPs themselves are
    // inserted in reverse order.
    // For example: %GEP1 = getelementptr %root, %A, %B, %C
    //              %GEP2 = getelementptr %GEP1, %1, %2, %3
    // The GEPOperands would be: [%A, %B, %C, %1, %2, %3]
    // However since the second gep is expected to begin with a zero value
    // it would instead be:      [%A, %B, %C, %2, %3]
    unsigned NumGEPOperands = GEP->getNumOperands();
    for (unsigned Idx = 1; Idx < NumGEPOperands; Idx++) {
      auto *InsertIt = GEPOperands.begin() + Idx - 1;
      GEPOperands.insert(InsertIt, GEP->getOperand(Idx));
    }
    Operand = GEP->getOperand(0);

    // For now, if this is a dynamic pointer, we assume that we cannot feed a
    // GEP into a GEP; we only care about the last GEP. So break out of the
    // while loop here.
    if (BaseShape->isDynamic())
      break;
  }

  // For every dimension of the shape, get an index.
  unsigned NumDims = BaseShape->getNumDims();
  unsigned NumGEPOperands = GEPOperands.size();
  if (NumGEPOperands > 0) {
    // Dynamic shapes only contain a single linearized index that needs to be
    // delinearized.
    if (BaseShape->isDynamic()) {
      Value *GEPOperand = GEPOperands[BaseShape->getGEPIdx(0)];
      // Special case for 1D buffers:
      if (NumDims == 1) {
        Indices.push_back(GEPOperand);
        return Operand;
      }
      bool Success = this->matchBufferAccess(GEPOperand, Indices);
      assert(Success && "Buffer access matching failed");
      (void)Success;
      return Operand;
    }

    // For static shapes:
    for (unsigned Dim = 0; Dim < NumDims; Dim++) {
      unsigned GEPIdx = BaseShape->getGEPIdx(Dim);
      assert(GEPIdx < GEPOperands.size() && "Unexpected number of GEP indices");
      Value *GEPOperand = GEPOperands[GEPIdx];
      Indices.push_back(GEPOperand);
    }
  } else {
    // If the GEPoperands has no operands inside of it, it means the above code
    // did not find a GEP instruction for the array access.
    LLVM_DEBUG({
      dbgs() << "\t[WARNING] Failed to find GEP instruction from array access, "
                "filling indices with [0]: ";
      Operand->dump();
    });
    const DataLayout DL = this->F->getParent()->getDataLayout();
    Type *IndexType = DL.getIndexType(Operand->getType());
    Constant *ConstZero = ConstantInt::get(IndexType, 0);
    for (unsigned Dim = 0; Dim < NumDims; Dim++)
      Indices.push_back(ConstZero);
  }

  // Return the current operand.
  return Operand;
}

/// Preprocesses the IR for easier parsing and translation
void CodeGen::preprocess() {
  IRBuilder<> Builder(F->getContext());
  Preprocessor P(F, &Builder, &FAM);
  P.run();
  this->ArgCounter = P.parseAccessorArguments(ShapeMap);
}
