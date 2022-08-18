//===-- MLIRCodeGen.cpp - MLIR CodeGen --------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Convert SYCL LLVMIR into MLIR (affine dialect).
//
//===----------------------------------------------------------------------===//

#include "MLIRCodeGen/MLIRCodeGen.h"
#include "MLIRCodeGen/LLVMToMLIR.h"
#include "Util/Matcher.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/IntegerSet.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/Support/FileSystem.h"

using namespace mlir;
using namespace llvm::converter;
using llvm::Error;
using std::string;

#define DEBUG_TYPE "mlir-converter"

/// Constructor for the MLIRCodeGen.
///
/// Sets up the Context, Builder, and Module used for generating MLIR. Using the
/// llvm::Module, which will house the llvm::Functions that will be converted by
/// this class, generates an mlir::Module with proper attributes.
MLIRCodeGen::MLIRCodeGen(const llvm::Module &M, unsigned indexBitwidth)
    : CodeGen(M.getContext()), Ctx(), Builder(&Ctx),
      indexBitwidth(indexBitwidth) {
  // Register and load required dialects into the MLIR Context
  DialectRegistry Registry;
  Registry.insert<AffineDialect>();
  Registry.insert<arith::ArithmeticDialect>();
  Registry.insert<func::FuncDialect>();
  Registry.insert<scf::SCFDialect>();
  Registry.insert<math::MathDialect>();
  Registry.insert<memref::MemRefDialect>();
  Registry.insert<vector::VectorDialect>();
  Ctx.appendDialectRegistry(Registry);
  Ctx.loadAllAvailableDialects();

  // Create the module
  Builder.clearInsertionPoint();
  Location Loc = FileLineColLoc::get(&Ctx, M.getSourceFileName(), 0, 0);
  Module = Builder.create<ModuleOp>(Loc, M.getName());
  // Set the data layout attribute.
  StringRef DLAttrName = mlir::LLVM::LLVMDialect::getDataLayoutAttrName();
  const char *DataLayout = M.getDataLayoutStr().c_str();
  Module->setAttr(DLAttrName, Builder.getStringAttr(DataLayout));
  // Set the target triple attribute.
  StringRef TTAttrName = mlir::LLVM::LLVMDialect::getTargetTripleAttrName();
  const char *TargetTriple = M.getTargetTriple().c_str();
  Module->setAttr(TTAttrName, Builder.getStringAttr(TargetTriple));

  if (this->indexBitwidth == 0) {
    this->indexBitwidth =
        mlir::DataLayout(Module).getTypeSizeInBits(IndexType::get(&Ctx));
  }
}

/// Destructor for the MLIRCodeGen.
///
/// The Blocks in the LLVMToMLIRBlockMap were allocated with the new keyword.
/// When the MLIRCodeGen is destroyed, we must delete all of these blocks from
/// the map to prevent a memory leak.
MLIRCodeGen::~MLIRCodeGen() {
  for (auto Pair : LLVMToMLIRBlockMap)
    delete Pair->second;
}

/// Codegen specific reset for internal variables that is called just before
/// converting a function.
///
/// Resets all internal maps, sets, and vectors used throughout the conversion
/// process.
void MLIRCodeGen::resetCodeGen() {
  // Clear the value map
  LLVMToMLIRValueMap.clear();
  // Clear the processed blocks
  ProcessedBlocks.clear();
  // Clear the index maps
  IndexMap.clear();
  LLVMToMLIRIndexValueMap.clear();
  // Clear the Global Values
  GlobalValues.clear();
  // Clear the Loop Iter Args map
  LoopIterArgsMap.clear();
  // Clear the Block map safely
  for (auto Pair : LLVMToMLIRBlockMap)
    delete Pair->second;
  LLVMToMLIRBlockMap.clear();
}

/// Get the context used throughout the codegen for MLIR.
MLIRContext *MLIRCodeGen::getContext() { return &Ctx; }

/// Get the builder used to build MLIR.
OpBuilder *MLIRCodeGen::getBuilder() { return &Builder; }

/// Get the mlir::Module which will house all of the converted functions.
ModuleOp *MLIRCodeGen::getModule() { return &Module; }

/// Writes the current mlir::Module to the OutFile specified.
Error MLIRCodeGen::writeToFile(const string &OutFile) {
  std::error_code EC;
  raw_fd_ostream FS(OutFile, EC, sys::fs::OF_Text);
  if (EC)
    return createError("Unable to write to output file", EC);
  Module.print(FS);
  return Error::success();
}

/// Verifies that all of the ops within the module are legal.
///
/// This method runs MLIR's internal verifier which is provided by the
/// operations themselves. This method should throw an error if the given module
/// is not valid MLIR code.
Error MLIRCodeGen::verify() {
  if (mlir::verify(Module).failed()) {
    llvm::errs() << Module << "\n";
    return createError("MLIR Module failed verification.", std::error_code());
  }
  return Error::success();
}

/// Get or create the MLIR block associated with the given LLVM BasicBlock
Block *MLIRCodeGen::getMLIRBlock(const BasicBlock *Val) {
  auto It = LLVMToMLIRBlockMap.find(Val);
  if (It == LLVMToMLIRBlockMap.end()) {
    // The Block must be used as a pointer in this map due to a method marked as
    // destroyed. Have to be careful of memory leaks.
    Block *B = new Block();
    It = LLVMToMLIRBlockMap.insert(std::make_pair(Val, B)).first;
  }
  return It->second;
}

/// Replace the block in the LLVMToMLIRBlockMap with the given key
/// of OldBlk with the given NewBlk, safely erasing the old block. The old block
/// is then deleted and cannot be used.
void MLIRCodeGen::replaceMLIRBlock(const BasicBlock *OldBlk,
                                   mlir::Block *NewBlk) {
  // Get the current MLIR Block pointed to by the parent.
  mlir::Block *OldMLIRBlk = getMLIRBlock(OldBlk);
  assert(OldMLIRBlk != NewBlk && "Cannot replace the same block.");
  // Make the BasicBlock point to the parent
  LLVMToMLIRBlockMap[OldBlk] = NewBlk;
  // Free the memory of the old block since nothing points to it, preventing a
  // memory leak.
  delete OldMLIRBlk;
}

/// Create a dummy value of the given MLIR type.
///
/// Because we are generating the blocks before setting up the control flow
/// graph, we may need access to mlir::Values that have not been created yet
/// (for example function args, AffineForOp iter args and IVs, etc.). This
/// method will create a dummy Op which will return a Value of a given type.
/// This Op MUST be replaced by the end of the conversion or else it will fail
/// verification.
mlir::Value MLIRCodeGen::createDummyVal(mlir::Type DummyType) {
  // Set the insertion point to the beginning of the function block.
  OpBuilder::InsertionGuard Guard(Builder);
  Builder.setInsertionPointToStart(getMLIRBlock(&this->F->getEntryBlock()));
  Location Loc = Builder.getUnknownLoc();
  // Create dummy Op and return its result.
  return Builder.create<func::CallOp>(Loc, "Dummy", DummyType).getResult(0);
}

/// Set the insertion point of the internal builder to the end of the block. If
/// there is a terminator instruction, it will set the insertion point just
/// before it.
void MLIRCodeGen::setInsertionPointBeforeTerminator(Block *B) {
  // Set the insertion point before the terminator.
  // If there is no terminator, set it to the end of the block.
  if (!B->empty() && B->back().hasTrait<OpTrait::IsTerminator>())
    Builder.setInsertionPoint(B->getTerminator());
  else
    Builder.setInsertionPointToEnd(B);
}

/// Get an MLIR Location that describes the location of the given LLVM
/// Instruction.
///
/// Locations in MLIR are not required for the converter. Locations just
/// provide debug information to the user if the MLIR module fails to verify.
Location MLIRCodeGen::getInstrLoc(const Instruction *I) {
  // If an LLVM DebugLoc is provided by the instruction, use it to generate an
  // MLIR FileLineColLoc.
  if (DebugLoc LLVMLoc = I->getDebugLoc()) {
    StringRef FileName = this->F->getParent()->getSourceFileName();
    unsigned int Line = LLVMLoc.getLine();
    unsigned int Col = LLVMLoc.getCol();
    return FileLineColLoc::get(getContext(), FileName, Line, Col);
  }
  // If no debug location information is provided, for now return the unknown
  // location which would produce no debug information.
  return Builder.getUnknownLoc();
}

/// Parse the BasicBlocks of the LLVM Function we are converting.
///
/// For each basic block in the original LLVM function, generate MLIR operations
/// for required ops and put them into MLIR blocks that will be linked together
/// by the setupCFG method. Every LLVM BasicBlock will have an associated MLIR
/// Block.
void MLIRCodeGen::parseBlocks() {
  SmallVector<Loop *> Loops = {};
  SmallVector<BasicBlock *> IfLatchBlocks = {};
  // Generate the store and return instructions in each of the basic blocks of
  // the function. This will then recursivly generate any instruction that is
  // needed for a generated Op.
  for (BasicBlock &BB : *F) {
    LLVM_DEBUG(dbgs() << "Generating instructions for BB `"
                      << BB.getName().str()
                      << "`\n>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n");
    for (Instruction &I : BB) {
      if (auto *SI = dyn_cast<StoreInst>(&I)) {
        LLVM_DEBUG(dbgs() << "Generating store instruction `" << *SI << "`\n");
        genStoreInst(SI);
      } else if (auto *PHI = dyn_cast<PHINode>(&I)) {
        // PHINodes must always be generated because they act as iter args and
        // escaping values for the loops and ifs. This is a safety in case an if
        // requires a loop iter arg that is not generated until after the loop
        // is generated (which would cause a problem). A PHI should not be
        // generated here when it is only used as a latch to a branch
        // instruction. Control flow should be handled by the loop and ifs
        // themselves.
        if (PHI->hasOneUse() && llvm::isa<BranchInst>(*PHI->users().begin()))
          continue;
        LLVM_DEBUG(dbgs() << "Generating PHINode `" << *PHI << "`\n");
        genOperand(PHI);
      } else if (auto *RET = dyn_cast<ReturnInst>(&I)) {
        LLVM_DEBUG(dbgs() << "Generating return instruction `" << *RET
                          << "`\n");
        genReturnInst(RET);
      } else if (auto *MEMCPY = dyn_cast<MemCpyInst>(&I)) {
        LLVM_DEBUG(dbgs() << "Generating MemCpy instruction `" << *MEMCPY
                          << "`\n");
        genMemCpyInst(MEMCPY);
      } else if (auto *MEMSET = dyn_cast<MemSetInst>(&I)) {
        LLVM_DEBUG(dbgs() << "Generating MemSet instruction `" << *MEMSET
                          << "`\n");
        genMemSetInst(MEMSET);
      }
    }
    LLVM_DEBUG(dbgs() << "<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<\n");
    // Collect the Loops and if latch blocks for further processing.
    Loop *L = LI->getLoopFor(&BB);
    if (L && L->getLoopLatch() == &BB)
      Loops.push_back(L);
    else if (matchIfLatchBlock(&BB, LI))
      IfLatchBlocks.push_back(&BB);
  }
  // Generate the loop ops from the loops in the function.
  for (Loop *L : Loops) {
    LLVM_DEBUG(dbgs() << "Generating Loop Op for Loop `" << L->getName().str()
                      << "`\n>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n");
    genLoopOp(L);
    LLVM_DEBUG(dbgs() << "<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<\n");
  }
  // Generate the if ops from the if latch blocks in the function.
  for (BasicBlock *BB : IfLatchBlocks) {
    LLVM_DEBUG(dbgs() << "Generating If Op from block `" << BB->getName().str()
                      << "`\n>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n");
    genIfOp(BB);
    LLVM_DEBUG(dbgs() << "<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<\n");
  }
  // Finally to clean up the IR, there are IndexCastOps that operate on the
  // Induction Variables that may have not been used. They should be deleted
  // here.
  for (Loop *L : Loops) {
    LoopComponents &LC = LoopComponentMap[L];
    mlir::Value CastIV = genOperand(LC.IV);
    if (CastIV.getUsers().empty())
      CastIV.getDefiningOp()->erase();
  }
}

/// Given an LLVM Value, get the value as an MLIR Value and then cast it to an
/// index if needed.
///
/// To prevent multiple index cast ops, this method will only cast an
/// mlir::Value to an index once and all subsequent calls for that mlir::Value
/// will return the cast mlir::Value.
mlir::Value MLIRCodeGen::getOrCreateIndex(llvm::Value *LLVMVal) {
  // Check if an Index is already in the map
  auto It = IndexMap.find(LLVMVal);
  // If not map, create the op, if not already, and insert into map.
  if (It == IndexMap.end()) {
    mlir::Value NewIndex;
    // If the value is a constant int, the Index can be generated as a constant.
    if (auto *CI = dyn_cast<ConstantInt>(LLVMVal)) {
      // Create a constant op at the top of the function.
      OpBuilder::InsertionGuard Guard(Builder);
      setInsertionPointBeforeTerminator(
          getMLIRBlock(&this->F->getEntryBlock()));
      auto Attr =
          Builder.getIntegerAttr(Builder.getIndexType(), CI->getValue());
      Location Loc = Builder.getUnknownLoc();
      NewIndex = Builder.create<arith::ConstantOp>(Loc, Attr);
    } else {
      mlir::Value Val = genOperand(LLVMVal);
      // If the Val is not an index, cast it into one.
      if (Val.getType().isIndex()) {
        NewIndex = Val;
      } else {
        OpBuilder::InsertionGuard Guard(Builder);
        Builder.setInsertionPointAfterValue(Val);
        // Special case: PHINodes will use a dummy variable at the top of the
        // kernel, this cast should be in the containing block.
        if (auto PHI = dyn_cast<PHINode>(LLVMVal))
          setInsertionPointBeforeTerminator(getMLIRBlock(PHI->getParent()));
        Location Loc = Val.getLoc();
        mlir::Type DestType = Builder.getIndexType();
        NewIndex = Builder.create<arith::IndexCastOp>(Loc, DestType, Val);
      }
    }
    // Insert the index into the map.
    It = IndexMap.insert(std::make_pair(LLVMVal, NewIndex)).first;
  }
  // Return what is stored in the map
  return It->second;
}

/// Helper method for inserting a value into a vector. If the value is already
/// in the vector, it does not insert it again. This function returns the
/// position of the value in the vector.
static unsigned insertIntoVector(SmallVector<mlir::Value> &Vector,
                                 mlir::Value Val) {
  size_t I;
  size_t VectorSize = Vector.size();
  for (I = 0; I < VectorSize; I++)
    if (Vector[I] == Val)
      break;
  // At this point I holds the position in the vector the Val will be.
  // If I is the size of the vector, it was not found in the vector. Insert it.
  if (I == VectorSize)
    Vector.push_back(Val);
  // Return the position of Val in the Vector
  return I;
}

/// Recursivly generate an Affine Expression starting at the given Index.
///
/// Given an llvm::Value to be used as an index, create an Affine Expression for
/// the index. Affine Expressions are Affine's way of handling index's. Every
/// index must be declared as either a dim, an index tied to the shape
/// information of an Op, or a symbol, any other legal index. For more
/// information on dims vs symbols, see the affine dialect page on MLIR's
/// website.
///
/// When generating Affine Expressions for loads and stores, the offsets will
/// still be in the value even though the memref stores the offset information.
/// The IgnoreOffsets flag will zero-out any offsets it finds.
AffineExpr MLIRCodeGen::getIndexAffineExpr(llvm::Value *Index,
                                           SmallVector<mlir::Value> &DimArgs,
                                           SmallVector<mlir::Value> &SymArgs,
                                           bool IgnoreOffsets) {
  assert(Index->getType()->isIntegerTy() && "Index must be an integer.");
  // If the current index is a cast Op, look past it. Indexes need not be cast.
  if (auto *CI = dyn_cast<CastInst>(Index))
    return getIndexAffineExpr(CI->getOperand(0), DimArgs, SymArgs,
                              IgnoreOffsets);
  // If the current Index is a loop induction variable, generate the induction
  // variable and add it as a Dim Operand. Loop Induction Variables are always
  // dims.
  if (isLoopIV(Index)) {
    // The Index is originally casted, so we need to get the uncasted value,
    // since AffineExpressions want index type.
    mlir::Value CastVal = genOperand(Index);
    assert(mlir::isa<arith::IndexCastOp>(CastVal.getDefiningOp()) &&
           "Induction Variable expected to be the cast of the dummy index.");
    mlir::Value IndexVal = CastVal.getDefiningOp()->getOperand(0);
    unsigned Pos = insertIntoVector(DimArgs, IndexVal);
    return getAffineDimExpr(Pos, getContext());
  }
  // If the current Index is a PHINode (and not a loop induction variable), get
  // the induction variable as an index and add it as a Symbol Operand. Without
  // more information the general iter_arg or escaping scalar is always a
  // symbol.
  if (llvm::isa<PHINode>(Index)) {
    mlir::Value IndexVal = getOrCreateIndex(Index);
    unsigned Pos = insertIntoVector(SymArgs, IndexVal);
    return getAffineSymbolExpr(Pos, getContext());
  }
  // If the current Index is an argument, generate it and add it as a Symbol
  // Operand. Arguments to functions are always symbols.
  if (llvm::isa<Argument>(Index)) {
    mlir::Value IndexVal = getOrCreateIndex(Index);
    unsigned Pos = insertIntoVector(SymArgs, IndexVal);
    return getAffineSymbolExpr(Pos, getContext());
  }
  // If the current Index is an element of an array, generate it and add it as a
  // Symbol operand. Unless more information is known about the value in the
  // array, it is assumed that the values in arrays are symbols.
  if (auto *Load = dyn_cast<LoadInst>(Index)) {
    // handle if the Index is loading an offset of a memref.
    Shape *S = getOrCreateShape(Load->getPointerOperand());
    if (S->getShapeType() & Shape::ShapeType::Offset) {
      if (IgnoreOffsets)
        return getAffineConstantExpr(0, getContext());
      llvm_unreachable("TODO: Handle loading the offset as an index.");
    }
    mlir::Value IndexVal = getOrCreateIndex(Index);
    unsigned Pos = insertIntoVector(SymArgs, IndexVal);
    return getAffineSymbolExpr(Pos, getContext());
  }
  // Similar to the LoadInst above, values extracted from vectors are also
  // assumed to be symbols.
  if (llvm::isa<ExtractElementInst>(Index)) {
    mlir::Value IndexVal = getOrCreateIndex(Index);
    unsigned Pos = insertIntoVector(SymArgs, IndexVal);
    return getAffineSymbolExpr(Pos, getContext());
  }
  // If the current index is a constant, return a Constant Expression with the
  // value of the constant.
  if (llvm::isa<Constant>(Index)) {
    auto *ConstInstr = dyn_cast<ConstantInt>(Index);
    assert(ConstInstr && "Constant Index expected to be a ConstantInt.");
    APInt Val = ConstInstr->getValue();
    int64_t ConstVal = Val.getSExtValue();
    return getAffineConstantExpr(ConstVal, getContext());
  }
  // If the current index is a BinaryOperator, create the Affine Expression of
  // the LHS and RHS, then perform the Op specified and return the result. Note:
  // Affine Expressions do not support all binary operations.
  if (auto *BO = dyn_cast<BinaryOperator>(Index)) {
    AffineExpr LHS =
        getIndexAffineExpr(BO->getOperand(0), DimArgs, SymArgs, IgnoreOffsets);
    AffineExpr RHS =
        getIndexAffineExpr(BO->getOperand(1), DimArgs, SymArgs, IgnoreOffsets);
    switch (BO->getOpcode()) {
    case Instruction::BinaryOps::Add:
      return LHS + RHS;
    case Instruction::BinaryOps::Sub:
      return LHS - RHS;
    case Instruction::BinaryOps::Mul:
      return LHS * RHS;
    case Instruction::BinaryOps::UDiv:
    case Instruction::BinaryOps::SDiv:
      return LHS.floorDiv(RHS);
    case Instruction::BinaryOps::URem:
    case Instruction::BinaryOps::SRem:
      return LHS % RHS;
    default:
      llvm_unreachable("Unhandled AffineExpression BinaryOp");
    }
  }
  // If the current index is a CallInst, this may be a special index operation
  // (like AffineMinOp or AffineMaxOp) and these index ops may need to be
  // generated in place.
  if (auto *CALL = dyn_cast<CallInst>(Index)) {
    // Check to see if an index was already generated for this call
    auto It = LLVMToMLIRIndexValueMap.find(Index);
    if (It != LLVMToMLIRIndexValueMap.end()) {
      // If it was already generated, add it as a symbol; the result of an
      // operation should be a symbol.
      unsigned Pos = insertIntoVector(SymArgs, It->second);
      return getAffineSymbolExpr(Pos, getContext());
    }
    // Set the insertion point to the end of the parent block
    OpBuilder::InsertionGuard Guard(Builder);
    setInsertionPointBeforeTerminator(getMLIRBlock(CALL->getParent()));
    Location Loc = getInstrLoc(CALL);
    // Use the func name to decide what op needs to be created.
    Function *Func = CALL->getCalledFunction();
    StringRef FuncName = Func->getName();
    if (FuncName == "_Z17__spirv_ocl_s_minii") {
      // Get the Affine Expressions for the LHS and RHS args
      SmallVector<mlir::Value> CallDimArgs;
      SmallVector<mlir::Value> CallSymArgs;
      AffineExpr LHSAffineExpr = getIndexAffineExpr(
          CALL->getArgOperand(0), CallDimArgs, CallSymArgs, IgnoreOffsets);
      AffineExpr RHSAffineExpr = getIndexAffineExpr(
          CALL->getArgOperand(1), CallDimArgs, CallSymArgs, IgnoreOffsets);
      // Create a map from these affine expressions, the map would have two
      // results and the AffineMinOp will return the smaller of the two results.
      auto MinMap =
          AffineMap::get(CallDimArgs.size(), CallSymArgs.size(),
                         {LHSAffineExpr, RHSAffineExpr}, getContext());
      // Get the map operands
      SmallVector<mlir::Value> MapArgs;
      MapArgs.insert(MapArgs.end(), CallDimArgs.begin(), CallDimArgs.end());
      MapArgs.insert(MapArgs.end(), CallSymArgs.begin(), CallSymArgs.end());
      // Build the AffineMinOp and add it as a symbol operand; the result of an
      // operation should be a symbol.
      mlir::Value IndexVal = Builder.create<AffineMinOp>(Loc, MinMap, MapArgs);
      LLVMToMLIRIndexValueMap[Index] = IndexVal;
      unsigned Pos = insertIntoVector(SymArgs, IndexVal);
      return getAffineSymbolExpr(Pos, getContext());
    }
    if (FuncName == "_Z17__spirv_ocl_s_maxii") {
      // Get the Affine Expressions for the LHS and RHS args
      SmallVector<mlir::Value> CallDimArgs;
      SmallVector<mlir::Value> CallSymArgs;
      AffineExpr LHSAffineExpr = getIndexAffineExpr(
          CALL->getArgOperand(0), CallDimArgs, CallSymArgs, IgnoreOffsets);
      AffineExpr RHSAffineExpr = getIndexAffineExpr(
          CALL->getArgOperand(1), CallDimArgs, CallSymArgs, IgnoreOffsets);
      // Create a map from these affine expressions, the map would have two
      // results and the AffineMaxOp will return the larger of the two results.
      auto MaxMap =
          AffineMap::get(CallDimArgs.size(), CallSymArgs.size(),
                         {LHSAffineExpr, RHSAffineExpr}, getContext());
      SmallVector<mlir::Value> MapArgs;
      MapArgs.insert(MapArgs.end(), CallDimArgs.begin(), CallDimArgs.end());
      MapArgs.insert(MapArgs.end(), CallSymArgs.begin(), CallSymArgs.end());
      // Build the AffineMinOp and add it as a symbol operand; the result of an
      // operation should be a symbol.
      mlir::Value IndexVal = Builder.create<AffineMaxOp>(Loc, MaxMap, MapArgs);
      LLVMToMLIRIndexValueMap[Index] = IndexVal;
      unsigned Pos = insertIntoVector(SymArgs, IndexVal);
      return getAffineSymbolExpr(Pos, getContext());
    }
    llvm::errs() << "CallInst: " << *CALL << "\n";
    llvm_unreachable("Unhandled AffineExpression CallInst.");
  }

  // If we are selecting between integers, the result of the select should be
  // turned into an Index and passed as a Symbol, since we dont know enough
  // information to classify it as a Dim.
  if (llvm::isa<SelectInst>(Index)) {
    mlir::Value IndexVal = getOrCreateIndex(Index);
    unsigned Pos = insertIntoVector(SymArgs, IndexVal);
    return getAffineSymbolExpr(Pos, getContext());
  }

  llvm::errs() << "Index: " << *Index << "\n";
  llvm_unreachable("Unhandled AffineExpression.");
}

/// Generates a MemRef operand for the given LLVM Value.
///
/// Affine Load/Store requires an AffineMap and MapOperands if you want to pass
/// in more complex indices. This method gets the memref as well as populates
/// the AffineMap and MapOperands.
mlir::Value
MLIRCodeGen::genMemrefOperand(llvm::Value *Operand, AffineMap &Map,
                              SmallVector<mlir::Value> &MapOperands) {
  SmallVector<llvm::Value *> Indices = {};
  llvm::Value *BasePtr = gatherArrayIndices(Operand, Indices);
  LLVM_DEBUG(dbgs() << "Generating indices from: " << *Operand << "\n");
  // For every dimension of the shape, generate an Affine Expression.
  SmallVector<AffineExpr> Exprs = {};
  SmallVector<mlir::Value> DimArgs = {};
  SmallVector<mlir::Value> SymArgs = {};
  for (llvm::Value *Index : Indices) {
    LLVM_DEBUG(dbgs() << "\tGenerating AffineExpr for: " << *Index << "\n");
    // We want to ignore the offsets because the offsets should be handled by
    // the memref itself.
    AffineExpr Expr = getIndexAffineExpr(Index, DimArgs, SymArgs, true);
    LLVM_DEBUG(dbgs() << "\t\tAffine Expression Generated: " << Expr << "\n");
    Exprs.push_back(Expr);
  }
  // Create the AffineMap from the expressions.
  Map = AffineMap::get(DimArgs.size(), SymArgs.size(), Exprs, getContext());
  // Always insert dims then symbols into map operands.
  MapOperands.insert(MapOperands.end(), DimArgs.begin(), DimArgs.end());
  MapOperands.insert(MapOperands.end(), SymArgs.begin(), SymArgs.end());
  LLVM_DEBUG(dbgs() << "\tAffineMap: " << Map << "\n");
  // If the BasePtr is being loaded from a simple wrapper, the memref should be
  // completely described by the root of this value. This extra load should be
  // ignored since it will be handled by the trampoling function.
  if (auto *LI = dyn_cast<LoadInst>(tracePastCastAndGEP(BasePtr))) {
    llvm::Value *LoadBasePtr = LI->getPointerOperand();
    Shape *S = getOrCreateShape(LoadBasePtr);
    if (S->isSimpleWrapper()) {
      LLVM_DEBUG(
          dbgs() << "\tPtr operand found to be loaded from simple wrapper. "
                    "Using the root of the load as the memref operand.");
      BasePtr = getRoot(LoadBasePtr);
    }
  }
  // Generate the memref and return it.
  return genOperand(BasePtr);
}

/// A general method for getting an MLIR Value associated with the given LLVM
/// Value.
///
/// Given an LLVM Operand, this method will generate an MLIR Operand in the
/// associated MLIR Block. When converting an LLVM Instruction to an MLIR
/// Operation, the operands of the LLVM Instruction will need to be converted
/// beforehand, then the current Operation will be generated, thus this method
/// will likely be called recursivly.
mlir::Value MLIRCodeGen::genOperand(llvm::Value *Operand) {
  // If this is a pointer, use the root of the pointer. Any GEP's should be
  // handled when getting the indices for loads and stores.
  if (Operand->getType()->isPointerTy())
    Operand = getRoot(Operand);
  // Check if the Operand has already been generated, if so return it.
  auto It = LLVMToMLIRValueMap.find(Operand);
  if (It != LLVMToMLIRValueMap.end())
    return It->second;

  LLVM_DEBUG(dbgs() << "Generating as operand: "; Operand->dump());
  mlir::Value RetVal;
  if (auto *Arg = dyn_cast<Argument>(Operand))
    RetVal = genArgumentOperand(Arg);
  else if (auto *GO = dyn_cast<GlobalValue>(Operand))
    RetVal = genGlobalOperand(GO);
  else if (auto *CO = dyn_cast<Constant>(Operand))
    RetVal = genConstantOperand(CO);
  else if (auto *Inst = dyn_cast<Instruction>(Operand))
    RetVal = genInstructionOperand(Inst);
  else
    llvm_unreachable("Unexpected operand");
  LLVM_DEBUG(dbgs() << "\tFinished generating operand: " << RetVal << "\n");

  // Insert the value into the map, if has not done so already
  const auto &InsertPair = std::make_pair(Operand, std::move(RetVal));
  auto InsertIt = LLVMToMLIRValueMap.insert(InsertPair);
  // Return whatever is in the LLVMToMLIRValueMap.
  return InsertIt.first->second;
}

/// Generate an MLIR Function Argument given an LLVM Argument.
///
/// This method creates a placeholder MLIR Value that will be replaced in the
/// finalize method once the function (and thus its args) are generated.
mlir::Value MLIRCodeGen::genArgumentOperand(Argument *Arg) {
  // Get the type of the argument. If the arg is a pointer it wil be converted
  // to a memref, if not it will be converted to its associated MLIR type.
  mlir::Type ArgType;
  if (llvm::isa<PointerType>(Arg->getType()))
    ArgType = getMemrefType(Arg);
  else
    ArgType = LLVMTypeToMLIRType(Arg->getType(), getContext());
  // Create a dummy value for the arg, this will get replaced when the
  // function Op is generated in the finalize method.
  return createDummyVal(ArgType);
}

/// Generate an MLIR Function Argument given an LLVM GlobalValue.
///
/// Instead of treating the Value as a GlobalValue, we treat it as a function
/// argument so we can allow the trampoline function to handle the interface
/// between globals and MLIR memrefs.
///
/// This method creates a placeholder MLIR Value that will be replaced in the
/// finalize method once the function (and thus its args) are generated.
mlir::Value MLIRCodeGen::genGlobalOperand(GlobalValue *GO) {
  // Get the type of the argument. If the arg is a pointer it wil be converted
  // to a memref, if not it will be converted to its associated MLIR type.
  mlir::Type ArgType;
  if (llvm::isa<PointerType>(GO->getType()))
    ArgType = getMemrefType(GO);
  else
    ArgType = LLVMTypeToMLIRType(GO->getType(), getContext());
  // Insert the global value into the GlobalValues map so that it will be added
  // to the function args.
  GlobalValues.push_back(GO);
  // Create a dummy value for the arg, this will get replaced when the
  // function Op is generated in the finalize method.
  return createDummyVal(ArgType);
}

/// Generate an MLIR ConstantOp given an LLVM Constant.
mlir::Value MLIRCodeGen::genConstantOperand(const Constant *CO) {
  // Set the insertion point to the beginning of the function block. This is so
  // the constants can be accessed by all ops.
  OpBuilder::InsertionGuard Guard(Builder);
  Builder.setInsertionPointToStart(getMLIRBlock(&this->F->getEntryBlock()));
  Location Loc = Builder.getUnknownLoc();

  // Get the constant attr given the element type
  mlir::Attribute Attr;
  if (auto *FVTy = dyn_cast<FixedVectorType>(CO->getType())) {
    assert((llvm::isa<ConstantDataVector, ConstantVector>(CO)) &&
           "Constant vector expected to be a Constant[Data]Vector.");
    auto *ConstVec = dyn_cast<Constant>(CO);
    // Get the vec length and the aggregate elements.
    unsigned VecLength = 0;
    SmallVector<Constant *> AggrElems = {};
    for (unsigned I = 0, NumElems = FVTy->getNumElements(); I < NumElems; I++) {
      Constant *AggrElement = ConstVec->getAggregateElement(I);
      if (llvm::isa<PoisonValue>(AggrElement))
        break;
      AggrElems.push_back(AggrElement);
      VecLength++;
    }
    llvm::Type *ElemType = FVTy->getElementType();
    mlir::Type DataType = LLVMTypeToMLIRType(ElemType, getContext());
    auto VecTy = mlir::VectorType::get({VecLength}, DataType);
    // Get the dense elements attribute
    if (ElemType->isFloatingPointTy()) {
      SmallVector<APFloat> Values = {};
      for (Constant *AggrElement : AggrElems) {
        auto *ConstFP = dyn_cast<ConstantFP>(AggrElement);
        assert(ConstFP && "Expected to be a ConstantFP type.");
        Values.push_back(ConstFP->getValue());
      }
      Attr = mlir::DenseElementsAttr::get(VecTy, Values);
    } else if (ElemType->isIntegerTy()) {
      SmallVector<APInt> Values = {};
      for (Constant *AggrElement : AggrElems) {
        auto *ConstInt = dyn_cast<ConstantInt>(AggrElement);
        assert(ConstInt && "Expected to be a ConstantInt type.");
        Values.push_back(ConstInt->getValue());
      }
      Attr = mlir::DenseElementsAttr::get(VecTy, Values);
    } else {
      llvm_unreachable("Unhandled vector constant type.");
    }
  } else if (CO->getType()->isIntegerTy()) {
    auto *ConstInstr = dyn_cast<ConstantInt>(CO);
    assert(ConstInstr && "Constant Integer expected to be a ConstantInt.");
    mlir::Type DataType = LLVMTypeToMLIRType(CO->getType(), getContext());
    Attr = Builder.getIntegerAttr(DataType, ConstInstr->getValue());
  } else if (CO->getType()->isFloatingPointTy()) {
    auto *ConstInstr = dyn_cast<ConstantFP>(CO);
    assert(ConstInstr && "Constant fp expected to be a ConstantFP.");
    mlir::Type DataType = LLVMTypeToMLIRType(CO->getType(), getContext());
    Attr = Builder.getFloatAttr(DataType, ConstInstr->getValue());
  }
  assert(Attr && "Unhandled Constant for genConstantOperand.");

  // Create the Op
  return Builder.create<arith::ConstantOp>(Loc, Attr);
}

/// Helper method for the genOperand method.
///
/// If the given operand is an instruction, the instruction needs to be created
/// and the result of this Op needs to be returned.
mlir::Value MLIRCodeGen::genInstructionOperand(Instruction *Operand) {
  // The array accesses should have been handled by genMemrefOperand.
  assert(!llvm::isa<GetElementPtrInst>(Operand) &&
         "Unexpected array access here.");

  // Generate the Instruction.
  // https://llvm.org/docs/LangRef.html#llvm-language-reference-manual
  mlir::Value RetVal;
  if (auto *UO = dyn_cast<UnaryOperator>(Operand)) {
    RetVal = genUnaryOperator(UO);
  } else if (auto *BO = dyn_cast<BinaryOperator>(Operand)) {
    RetVal = genBinaryOperator(BO);
  } else if (auto *EE = dyn_cast<ExtractElementInst>(Operand)) {
    RetVal = genExtractElementInst(EE);
  } else if (auto *SV = dyn_cast<ShuffleVectorInst>(Operand)) {
    RetVal = genShuffleVectorInst(SV);
  } else if (auto *AI = dyn_cast<AllocaInst>(Operand)) {
    RetVal = genAllocaInst(AI);
  } else if (auto *LI = dyn_cast<LoadInst>(Operand)) {
    RetVal = genLoadInst(LI);
  } else if (auto *CI = dyn_cast<CastInst>(Operand)) {
    RetVal = genCastInst(CI);
  } else if (auto *CI = dyn_cast<CmpInst>(Operand)) {
    RetVal = genCmpInst(CI);
  } else if (auto *PHI = dyn_cast<PHINode>(Operand)) {
    // genPHINode internally inserts the Value into the map, it has to do this
    // to prevent infinite loops, return the value here.
    return genPHINode(PHI);
  } else if (auto *SI = dyn_cast<SelectInst>(Operand)) {
    RetVal = genSelectInst(SI);
  } else if (auto *CI = dyn_cast<CallInst>(Operand)) {
    RetVal = genCallInst(CI);
  } else {
    llvm_unreachable("Unexpected Instruction operand");
  }

  // Insert the instruction into the Value map.
  const auto &InsertPair = std::make_pair(Operand, std::move(RetVal));
  auto InsertVal = LLVMToMLIRValueMap.insert(InsertPair);
  // If it was already in the map, that means that one of the operands generated
  // this value. Erase the Op we just created. This should only happen if a
  // PHINode Calls another PHINode.
  if (!InsertVal.second) {
    assert(RetVal != InsertVal.first->second &&
           "Trying to insert an instruction operand that already exists. It "
           "should not be deleted.");
    RetVal.getDefiningOp()->erase();
  }
  // Return the value in the valueMap.
  return InsertVal.first->second;
}

/// Generate an MLIR ReturnOp for the given LLVM ReturnInst.
void MLIRCodeGen::genReturnInst(const ReturnInst *RET) {
  // Set the insertion point to the end of the parent block
  OpBuilder::InsertionGuard Guard(Builder);
  setInsertionPointBeforeTerminator(getMLIRBlock(RET->getParent()));
  Location Loc = getInstrLoc(RET);

  // Get the operands
  SmallVector<mlir::Value> Operands = {};
  for (const Use &Operand : RET->operands())
    Operands.push_back(genOperand(Operand));

  // Create the Op
  Builder.create<func::ReturnOp>(Loc, Operands);
}

/// Generate the MLIR Op relating to the given LLVM UnaryOperator.
mlir::Value MLIRCodeGen::genUnaryOperator(const UnaryOperator *UO) {
  // Set the insertion point to the end of the parent block
  OpBuilder::InsertionGuard Guard(Builder);
  setInsertionPointBeforeTerminator(getMLIRBlock(UO->getParent()));
  Location Loc = getInstrLoc(UO);

  // Get the operands
  mlir::Value Operand = genOperand(UO->getOperand(0));

  // Create the Op
  switch (UO->getOpcode()) {
  case Instruction::UnaryOps::FNeg:
    return Builder.create<arith::NegFOp>(Loc, Operand).getResult();
  default:
    llvm_unreachable("Unhandled unary operand.");
  }
}

/// Generate the MLIR Op relating to the given LLVM BinaryOperator.
mlir::Value MLIRCodeGen::genBinaryOperator(const BinaryOperator *BO) {
  // Set the insertion point to the end of the parent block
  OpBuilder::InsertionGuard Guard(Builder);
  setInsertionPointBeforeTerminator(getMLIRBlock(BO->getParent()));
  Location Loc = getInstrLoc(BO);

  // Get the operands
  mlir::Value LHS = genOperand(BO->getOperand(0));
  mlir::Value RHS = genOperand(BO->getOperand(1));

  // Create the Op
  switch (BO->getOpcode()) {
  case Instruction::BinaryOps::Add:
    return Builder.create<arith::AddIOp>(Loc, LHS, RHS).getResult();
  case Instruction::BinaryOps::FAdd:
    return Builder.create<arith::AddFOp>(Loc, LHS, RHS).getResult();
  case Instruction::BinaryOps::Sub:
    return Builder.create<arith::SubIOp>(Loc, LHS, RHS).getResult();
  case Instruction::BinaryOps::FSub:
    return Builder.create<arith::SubFOp>(Loc, LHS, RHS).getResult();
  case Instruction::BinaryOps::Mul:
    return Builder.create<arith::MulIOp>(Loc, LHS, RHS).getResult();
  case Instruction::BinaryOps::FMul:
    return Builder.create<arith::MulFOp>(Loc, LHS, RHS).getResult();
  case Instruction::BinaryOps::UDiv:
    return Builder.create<arith::DivUIOp>(Loc, LHS, RHS).getResult();
  case Instruction::BinaryOps::SDiv:
    return Builder.create<arith::DivSIOp>(Loc, LHS, RHS).getResult();
  case Instruction::BinaryOps::FDiv:
    return Builder.create<arith::DivFOp>(Loc, LHS, RHS).getResult();
  case Instruction::BinaryOps::URem:
    return Builder.create<arith::RemUIOp>(Loc, LHS, RHS).getResult();
  case Instruction::BinaryOps::SRem:
    return Builder.create<arith::RemSIOp>(Loc, LHS, RHS).getResult();
  case Instruction::BinaryOps::FRem:
    return Builder.create<arith::RemFOp>(Loc, LHS, RHS).getResult();
  case Instruction::BinaryOps::Shl:
    return Builder.create<arith::ShLIOp>(Loc, LHS, RHS).getResult();
  case Instruction::BinaryOps::LShr:
    return Builder.create<arith::ShRUIOp>(Loc, LHS, RHS).getResult();
  case Instruction::BinaryOps::AShr:
    return Builder.create<arith::ShRSIOp>(Loc, LHS, RHS).getResult();
  case Instruction::BinaryOps::And:
    return Builder.create<arith::AndIOp>(Loc, LHS, RHS).getResult();
  case Instruction::BinaryOps::Or:
    return Builder.create<arith::OrIOp>(Loc, LHS, RHS).getResult();
  case Instruction::BinaryOps::Xor:
    return Builder.create<arith::XOrIOp>(Loc, LHS, RHS).getResult();
  default:
    llvm_unreachable("Unexpected binary operation");
  }
}

/// Generate an MLIR vector::ExtractOp for the given ExtractElementInst.
mlir::Value MLIRCodeGen::genExtractElementInst(ExtractElementInst *EE) {
  // Set the insertion point to the end of the parent block
  OpBuilder::InsertionGuard Guard(Builder);
  setInsertionPointBeforeTerminator(getMLIRBlock(EE->getParent()));
  Location Loc = getInstrLoc(EE);

  // Get the operands
  mlir::Value Vec = genOperand(EE->getVectorOperand());
  auto *Index = dyn_cast<ConstantInt>(EE->getIndexOperand());
  assert(Index && "Index of ExtractElementInst must be a constant for now. "
                  "TODO: Handle dynamic index.");

  // Create the op
  return Builder.create<vector::ExtractOp>(Loc, Vec, Index->getZExtValue());
}

/// Generate the MLIR equivalent Op for the ShuffleVectorInst.
mlir::Value MLIRCodeGen::genShuffleVectorInst(const ShuffleVectorInst *SV) {
  // Get the operands
  mlir::Value FirstVector = genOperand(SV->getOperand(0));

  // Check if this is a first vector identity. If so, return the first vector.
  auto FirstVectorType = FirstVector.getType().dyn_cast<mlir::VectorType>();
  assert(FirstVectorType && "FirstVector expected to be a vector.");
  for (int I = 0, NumElems = FirstVectorType.getShape()[0]; I < NumElems; I++)
    if (SV->getMaskValue(I) != I)
      llvm_unreachable("Unhandled ShuffleVectorInstruction. Can only handle "
                       "first vector identity right now.");
  return FirstVector;
}

/// Generate an MLIR AllocaOp for the given LLVM AllocaInst.
mlir::Value MLIRCodeGen::genAllocaInst(AllocaInst *AI) {
  // Set the insertion point to the end of the parent block
  OpBuilder::InsertionGuard Guard(Builder);
  setInsertionPointBeforeTerminator(getMLIRBlock(AI->getParent()));
  Location Loc = getInstrLoc(AI);

  // Get the memref type
  MemRefType MemrefType = getMemrefType(AI);

  // Create the Op
  return Builder.create<memref::AllocaOp>(Loc, MemrefType);
}

/// Generate an MLIR AffineLoadOp for the given LLVM LoadInst.
///
/// When the loaded value is a vector, an AffineVectorLoadOp is generated
/// instead.
mlir::Value MLIRCodeGen::genLoadInst(LoadInst *LI) {
  // Set the insertion point to the end of the parent block
  OpBuilder::InsertionGuard Guard(Builder);
  setInsertionPointBeforeTerminator(getMLIRBlock(LI->getParent()));
  Location Loc = getInstrLoc(LI);

  AffineMap Map;
  SmallVector<mlir::Value> MapArgs = {};
  llvm::Value *PtrOperand = LI->getPointerOperand();

  // Handle Vector loads
  if (llvm::isa<FixedVectorType>(LI->getType())) {
    // Get the Memref to load the vector from
    mlir::Value Memref = genOperand(PtrOperand);
    MemRefType MemrefTy = Memref.getType().dyn_cast<MemRefType>();
    assert(MemrefTy && "LoadInst PtrOperand expected to be a MemRef.");
    // Get the vector type
    mlir::Type DataType = MemrefTy.getElementType();
    mlir::VectorType VTy = mlir::VectorType::get(MemrefTy.getShape(), DataType);
    // Since we are implicitly converting the Vector args into Memrefs of rank
    // 1 with the same length as the vector, we can always say the index is 0.
    Map = AffineMap::getConstantMap(0, getContext());
    // Create the vector load
    return Builder.create<AffineVectorLoadOp>(Loc, VTy, Memref, Map, MapArgs);
  }

  // Create the AffineLoadOp
  mlir::Value Memref = genMemrefOperand(PtrOperand, Map, MapArgs);
  return Builder.create<AffineLoadOp>(Loc, Memref, Map, MapArgs).getResult();
}

/// Generate an MLIR AffineStorOp for the given LLVM StoreInst.
///
/// When the value being stored is a vector, an AffineVectorStoreOp is generated
/// instead.
void MLIRCodeGen::genStoreInst(StoreInst *SI) {
  // Set the insertion point to the end of the parent block
  OpBuilder::InsertionGuard Guard(Builder);
  setInsertionPointBeforeTerminator(getMLIRBlock(SI->getParent()));
  Location Loc = getInstrLoc(SI);

  // Generate the value to store
  mlir::Value StoreVal = genOperand(SI->getValueOperand());

  AffineMap Map;
  SmallVector<mlir::Value> MapArgs = {};
  llvm::Value *PtrOperand = SI->getPointerOperand();

  // Handle vector stores
  if (StoreVal.getType().isa<mlir::VectorType>()) {
    // Generate the memref base
    mlir::Value Memref = genOperand(PtrOperand);
    // Since we are implicitly converting the Vector args into Memrefs of rank 1
    // with the same length as the vector, we can always say the index is 0.
    Map = AffineMap::get(0, 0, getAffineConstantExpr(0, getContext()));
    Builder.create<AffineVectorStoreOp>(Loc, StoreVal, Memref, Map, MapArgs);
    return;
  }

  // Create the AffineStoreOp
  mlir::Value Memref = genMemrefOperand(PtrOperand, Map, MapArgs);
  Builder.create<AffineStoreOp>(Loc, StoreVal, Memref, Map, MapArgs);
}

/// Generate the MLIR Op associated with the given LLVM CastInst.
///
/// Since MLIR does not handle pointers in the same way that LLVM does, this
/// method only generates non-pointer cast instructions. The pointer cast
/// instructions should be handled when generating MemRefs.
mlir::Value MLIRCodeGen::genCastInst(const CastInst *CI) {
  // Set the insertion point to the end of the parent block
  OpBuilder::InsertionGuard Guard(Builder);
  setInsertionPointBeforeTerminator(getMLIRBlock(CI->getParent()));
  Location Loc = getInstrLoc(CI);

  // Get the arg being casted
  mlir::Value CastArg = genOperand(CI->getOperand(0));

  // If this cast is returning a pointer, it does not make sense to turn this
  // into an MLIR Op (since it doesnt have pointers). Any pointers should be
  // handled in other methods (like genMemrefOperand).
  assert(!CI->getType()->isPointerTy() && "Unexpected pointer cast.");

  // Create the MLIR version of the cast instruction.
  mlir::Type TargetType = LLVMTypeToMLIRType(CI->getType(), getContext());
  if (llvm::isa<TruncInst>(CI))
    return Builder.create<arith::TruncIOp>(Loc, TargetType, CastArg);
  if (llvm::isa<ZExtInst>(CI))
    return Builder.create<arith::ExtUIOp>(Loc, TargetType, CastArg);
  if (llvm::isa<SExtInst>(CI))
    return Builder.create<arith::ExtSIOp>(Loc, TargetType, CastArg);
  if (llvm::isa<FPTruncInst>(CI))
    return Builder.create<arith::TruncFOp>(Loc, TargetType, CastArg);
  if (llvm::isa<FPExtInst>(CI))
    return Builder.create<arith::ExtFOp>(Loc, TargetType, CastArg);
  if (llvm::isa<FPToUIInst>(CI))
    return Builder.create<arith::FPToUIOp>(Loc, TargetType, CastArg);
  if (llvm::isa<FPToSIInst>(CI))
    return Builder.create<arith::FPToSIOp>(Loc, TargetType, CastArg);
  if (llvm::isa<UIToFPInst>(CI))
    return Builder.create<arith::UIToFPOp>(Loc, TargetType, CastArg);
  if (llvm::isa<SIToFPInst>(CI))
    return Builder.create<arith::SIToFPOp>(Loc, TargetType, CastArg);

  llvm_unreachable("Unhandled Cast instruction.");
}

/// Generate an MLIR CmpOp for the given LLVM CmpInst.
mlir::Value MLIRCodeGen::genCmpInst(const CmpInst *CI) {
  // Set the insertion point to the end of the parent block
  OpBuilder::InsertionGuard Guard(Builder);
  setInsertionPointBeforeTerminator(getMLIRBlock(CI->getParent()));
  Location Loc = getInstrLoc(CI);

  // Get the operands
  mlir::Value LHS = genOperand(CI->getOperand(0));
  mlir::Value RHS = genOperand(CI->getOperand(1));

  // Create the Op
  if (CI->getOpcode() == Instruction::OtherOps::FCmp) {
    arith::CmpFPredicate Pred = LLVMFCmpPredicateToMLIR(CI->getPredicate());
    return Builder.create<arith::CmpFOp>(Loc, Pred, LHS, RHS).getResult();
  }
  if (CI->getOpcode() == Instruction::OtherOps::ICmp) {
    arith::CmpIPredicate Pred = LLVMICmpPredicateToMLIR(CI->getPredicate());
    return Builder.create<arith::CmpIOp>(Loc, Pred, LHS, RHS).getResult();
  }
  llvm_unreachable("Unexpected Compare instruction.");
}

/// Generate the MLIR equivalent of an LLVM PHINode.
///
/// MLIR does not have PHINodes in its IR,
/// so they must be implemented as block arguments. However, in the Affine
/// dialect the user does not have direct access to the block arguments, so the
/// PHINodes will need to be handled in a special way. For example, AffineForOps
/// use iter args and escaping values to handle their PHINodes; AffineIfOps use
/// escaping values to handle their PHINodes; etc.
mlir::Value MLIRCodeGen::genPHINode(PHINode *PHI) {
  // Get the MLIRType of this PHINode
  mlir::Type PHIType = LLVMTypeToMLIRType(PHI->getType(), getContext());
  // If a PHINode is an Induction Variable for a loop, it will become an
  // induction variable for an AffineForOp. Create a dummy value of type index,
  // and then cast it to the same type as the PHINode. We will return the casted
  // value because we assume that genOperand will be using the same type as the
  // LLVMIR. Induction Variables may be used for other purposes and it is
  // important they have the same type as they do in the LLVMIR, however this
  // will be generated as an Index, so when the AffineForOp is generated this
  // will need to be done carefully.
  if (isLoopIV(PHI)) {
    mlir::Value DummyIndex = createDummyVal(Builder.getIndexType());
    // Set the insertion point right after the dummy index
    OpBuilder::InsertionGuard Guard(Builder);
    Builder.setInsertionPointAfter(DummyIndex.getDefiningOp());
    Location Loc = DummyIndex.getLoc();
    mlir::Value Dummy =
        Builder.create<arith::IndexCastOp>(Loc, PHIType, DummyIndex);
    return Dummy;
  }
  // Create a dummy Value that will be replaced the real Value once the
  // AffineForOp is generated.
  mlir::Value Dummy = createDummyVal(PHIType);
  // Insert Dummy into the LLVMToMLIRValueMap now to prevent infinite loops when
  // we call genOperand in this function. We call genOperand because a PHINode
  // may call another PHINode which may be another iter arg or escaping value
  // for a loop.
  LLVMToMLIRValueMap[PHI] = Dummy;
  // Check if this PHINode is used on an AffineIf statement. AffineIfOps handle
  // escaping scalars differently to AffineForOps iter args. A PHINode is an
  // escaping value for an AffineIf statement if the immediate dominator of the
  // block that contains the PHI is an IfLatchBlock.
  BasicBlock *IfLatchBlock = nullptr;
  DomTreeNode *Node = DT->getNode(PHI->getParent());
  BasicBlock *IDomBlock = nullptr;
  if (DomTreeNode *IDom = Node->getIDom()) {
    IDomBlock = IDom->getBlock();
    if (converter::matchIfLatchBlock(IDomBlock, LI))
      IfLatchBlock = IDomBlock;
  }
  unsigned NumIncomingValues = PHI->getNumIncomingValues();
  if (IfLatchBlock) {
    assert(
        NumIncomingValues == 2 &&
        "A PHINode with incoming values from blocks inside Ifs should always "
        "have two incoming values: The value if true and the value if false.");
    // Create a IfEscVal to hold the information on this escaping value.
    IfEscVal IfEscInfo;
    IfEscInfo.ExitPHI = PHI;
    BasicBlock *TrueSucc = IfLatchBlock->getTerminator()->getSuccessor(0);
    BasicBlock *FalseSucc = IfLatchBlock->getTerminator()->getSuccessor(1);
    // Get the yielded value if true and false. An incoming value is the yielded
    // true value if the true successor dominates the incoming block. An
    // incoming value is the yielded false value if the false successor
    // dominates the incoming block. This must be written as an if else chain
    // like this because it is possible for an if to only have a one successor
    // (think of an if that has no else block). That would mean that the "true"
    // value would be in the true block, while the false value is initialized
    // elsewhere.
    if (DT->dominates(TrueSucc, PHI->getIncomingBlock(0))) {
      IfEscInfo.TrueVal = PHI->getIncomingValue(0);
      IfEscInfo.FalseVal = PHI->getIncomingValue(1);
    } else if (DT->dominates(FalseSucc, PHI->getIncomingBlock(0))) {
      IfEscInfo.FalseVal = PHI->getIncomingValue(0);
      IfEscInfo.TrueVal = PHI->getIncomingValue(1);
    } else if (DT->dominates(TrueSucc, PHI->getIncomingBlock(1))) {
      IfEscInfo.TrueVal = PHI->getIncomingValue(1);
      IfEscInfo.FalseVal = PHI->getIncomingValue(0);
    } else if (DT->dominates(FalseSucc, PHI->getIncomingBlock(1))) {
      IfEscInfo.FalseVal = PHI->getIncomingValue(1);
      IfEscInfo.TrueVal = PHI->getIncomingValue(0);
    } else {
      llvm_unreachable("Unhandled If Escaping value PHINode.");
    }
    // Add the IfEscInfo to the map, indexed by the IfHeader block.
    IfEscValsMap[IfLatchBlock].push_back(IfEscInfo);
    // Generate the true and false values.
    genOperand(IfEscInfo.TrueVal);
    genOperand(IfEscInfo.FalseVal);
    return Dummy;
  }

  // Since we performed LCSSA, if a PHINode has only one incoming value it means
  // that it is taking a value that is created inside one loop and using it in
  // an outter loop. For example if a loop is performing a sum, and then that
  // sum is stored into an array; this PHI would take the sum generated by the
  // loop to the store Op. In MLIR this would be equivalent to taking the result
  // of the AffineForOp.
  if (NumIncomingValues == 1) {
    BasicBlock *IncBlock = PHI->getIncomingBlock(0);
    Value *IncValue = PHI->getIncomingValue(0);
    // Get the Loop that is creating the incoming value
    Loop *L = LI->getLoopFor(IncBlock);
    assert(L &&
           "TODO: Handle what to do if there is a PHINode outside of a Loop.");
    // Check if any other iter args already exist for this exit value
    for (LoopIterArg &IterArgInfo : LoopIterArgsMap[L]) {
      if (IterArgInfo.ExitVal == IncValue) {
        // If the iter arg already exists, set the ExitPHI to this PHI
        assert(!IterArgInfo.ExitPHI &&
               "LoopIterArg already has an ExitPHI, cannot currently handle "
               "multiple ExitPHIs.");
        IterArgInfo.ExitPHI = PHI;
        return Dummy;
      }
    }
    // If an iter arg for this exit value does not already exist, make one
    LoopIterArg IterArgInfo;
    IterArgInfo.ExitVal = IncValue;
    IterArgInfo.ExitPHI = PHI;
    // Set the init value. At this point we do not know what the init value will
    // be (if one already exists). It is possible for an iter arg to not have an
    // init value (because it doesnt need one), however the AffineForOp requires
    // one so initialize it to zero.
    Constant *ConstZero = nullptr;
    if (PHI->getType()->isIntegerTy())
      ConstZero = ConstantInt::get(PHI->getType(), 0);
    else if (PHI->getType()->isFloatingPointTy())
      ConstZero = ConstantFP::get(PHI->getType(), 0.0);
    else
      llvm_unreachable("Unhandled PHINode type.");
    IterArgInfo.InitVal = ConstZero;
    LoopIterArgsMap[L].push_back(IterArgInfo);
    // Gen the Incoming Value.
    genOperand(IncValue);
    return Dummy;
  }

  // If a PHINode is used in a loop and has two incoming values, that would mean
  // that it has one incoming value to initialize it and one incoming value to
  // update it after each loop trip. This is identical to iter_args in
  // AffineForOps, store the iter_arg info into a map. This will be used in the
  // insertLoopBlockIntoParent method. If a PHINode were to have more than 2
  // incoming values, then it would have multiple initial values or multiple
  // yielded values.
  assert(
      NumIncomingValues == 2 &&
      "Cannot currently lower a PHINode with more than two incoming values.");
  // Get the loop that directly contains the parent of the PHINode.
  Loop *L = LI->getLoopFor(PHI->getParent());
  assert(L &&
         "TODO: Handle what to do if there is a PHINode outside of a Loop.");
  // Load the IterArg info to be used when generating the AffineForOp.
  LoopIterArg IterArgInfo;
  // Assuming the PHINode is located inside the LoopHeader, the initvalue is the
  // value coming from the loop preheader block. Assuming this is a simple loop,
  // this PHINode should only be getting one value from outside the loop, so the
  // other value must be inside the loop and will be the exit value.
  if (PHI->getIncomingBlock(0) == L->getLoopPreheader()) {
    IterArgInfo.InitVal = PHI->getIncomingValue(0);
    IterArgInfo.ExitVal = PHI->getIncomingValue(1);
  } else {
    assert(PHI->getIncomingBlock(1) == L->getLoopPreheader() &&
           "InitVal expected to be initialized in the preheader of the loop.");
    IterArgInfo.InitVal = PHI->getIncomingValue(1);
    IterArgInfo.ExitVal = PHI->getIncomingValue(0);
  }
  // Gen the Incoming Values.
  genOperand(IterArgInfo.InitVal);
  genOperand(IterArgInfo.ExitVal);
  // Insert the IterArgInfo into the LoopIterArgsMap for this loop.
  // Check if an iter arg already exists for this ExitVal.
  for (LoopIterArg &OtherIterArgInfo : LoopIterArgsMap[L]) {
    if (OtherIterArgInfo.ExitVal == IterArgInfo.ExitVal) {
      // If so set the IntiVal and PHI of the iter arg.
      OtherIterArgInfo.InitVal = IterArgInfo.InitVal;
      OtherIterArgInfo.PHI = PHI;
      return Dummy;
    }
  }
  // If one does not exist, insert it.
  IterArgInfo.PHI = PHI;
  LoopIterArgsMap[L].push_back(IterArgInfo);
  return Dummy;
}

/// Generate an MLIR SelectOp for the given LLVM SelectInst.
mlir::Value MLIRCodeGen::genSelectInst(const SelectInst *SI) {
  // Set the insertion point to the end of the parent block
  OpBuilder::InsertionGuard Guard(Builder);
  setInsertionPointBeforeTerminator(getMLIRBlock(SI->getParent()));
  Location Loc = getInstrLoc(SI);

  // Get the operands
  mlir::Value Cond = genOperand(SI->getOperand(0));
  mlir::Value True = genOperand(SI->getOperand(1));
  mlir::Value False = genOperand(SI->getOperand(2));

  // Create the Op
  return Builder.create<arith::SelectOp>(Loc, Cond, True, False).getResult();
}

/// Generate an MLIR CallOp for the given LLVM CallInst.
///
/// Some SYCL intrinsics are not lowered to LLVM and remain as call
/// instructions, however these intrinsics can be lowered to MLIR Operations.
/// Special handling is used for these cases.
mlir::Value MLIRCodeGen::genCallInst(const CallInst *CI) {
  // Set the insertion point to the end of the parent block
  OpBuilder::InsertionGuard Guard(Builder);
  setInsertionPointBeforeTerminator(getMLIRBlock(CI->getParent()));
  Location Loc = getInstrLoc(CI);

  // Get the operands
  SmallVector<mlir::Value> Operands;
  for (const llvm::Use &Operand : CI->args())
    Operands.push_back(genOperand(Operand));

  // Check if we recognize this function
  Function *Func = CI->getCalledFunction();
  StringRef FuncName = Func->getName();
  // List of all supported spirv functions found here:
  // sycl/include/CL/sycl/detail/builtins.hpp
  // The following is not exaustive and we will add more as needed.
  if (FuncName == "_Z15__spirv_ocl_expf")
    return Builder.create<math::ExpOp>(Loc, Operands).getResult();
  if (FuncName == "_Z15__spirv_ocl_expd")
    return Builder.create<math::ExpOp>(Loc, Operands).getResult();
  if (FuncName == "_Z16__spirv_ocl_sqrtf")
    return Builder.create<math::SqrtOp>(Loc, Operands).getResult();
  if (FuncName == "_Z23__spirv_ocl_fmin_commonff")
    return Builder.create<arith::MinFOp>(Loc, Operands).getResult();
  if (FuncName == "_Z17__spirv_ocl_s_minii")
    return Builder.create<arith::MinSIOp>(Loc, Operands).getResult();
  if (FuncName == "_Z17__spirv_ocl_s_maxii")
    return Builder.create<arith::MaxSIOp>(Loc, Operands).getResult();

  // If the function is a declaration then the function will not be converted
  // and will not appear in the module. Will need to create a declaration for
  // this function, if not done so already.
  mlir::Type ResTy = LLVMTypeToMLIRType(CI->getType(), getContext());
  if (Func->isDeclaration() && !Module.lookupSymbol(FuncName)) {
    OpBuilder::InsertionGuard Guard(Builder);
    Builder.setInsertionPointToStart(&Module.getBodyRegion().front());
    Location FuncLoc = Builder.getUnknownLoc();
    auto FuncArgTypes = ValueRange(Operands).getTypes();
    auto FuncTy = mlir::FunctionType::get(getContext(), FuncArgTypes, ResTy);
    assert(GlobalValue::isExternalLinkage(Func->getLinkage()) &&
           "Function with declaration expected to have external linkage");
    mlir::StringAttr Visibility = Builder.getStringAttr("private");
    Builder.create<func::FuncOp>(FuncLoc, FuncName, FuncTy, Visibility);
  }

  // Build the Op.
  return Builder.create<func::CallOp>(Loc, FuncName, ResTy, Operands).getResult(0);
}

/// Generate the MLIR equivalent of an LLVM MemCpyInst.
///
/// MLIR does not have a MemCpyInst, at least not yet. So we generate a for loop
/// to copy elements from one memref to another, defined by the MemCpyInst.
void MLIRCodeGen::genMemCpyInst(const MemCpyInst *MEMCPY) {
  // Set the insertion point to the end of the parent block
  OpBuilder::InsertionGuard Guard(Builder);
  setInsertionPointBeforeTerminator(getMLIRBlock(MEMCPY->getParent()));
  Location Loc = getInstrLoc(MEMCPY);

  // Get the operands
  mlir::Value Src = genOperand(MEMCPY->getSource());
  mlir::Value Dest = genOperand(MEMCPY->getDest());

  // Checks
  MemRefType SrcTy = Src.getType().dyn_cast<MemRefType>();
  MemRefType DestTy = Dest.getType().dyn_cast<MemRefType>();
  assert(SrcTy && DestTy &&
         "Src and Dest of MemCpyInst expected to be memrefs.");
  assert(SrcTy.getElementType() == DestTy.getElementType() &&
         "Src and Dest of MemCpyInst expected to have same element type.");
  assert(SrcTy.getRank() == 1 && DestTy.getRank() == 1 &&
         "TODO: genMemCpyInst can only handle 1D copy right now.");
  (void)SrcTy;

  // Create a for loop to act as the memset
  AffineMap LBMap = AffineMap::getConstantMap(0, getContext());
  SmallVector<mlir::Value> LBArgs = {};
  SmallVector<mlir::Value> UBArgs = {};
  AffineMap UBMap = getLoopBound(MEMCPY->getLength(), UBArgs);
  // The MemCpy's length is given in bytes, however we need it in elements; do
  // divide the length by the byte width of the element type.
  assert(UBMap.getNumResults() == 1 && "Map expected to have one result.");
  unsigned EltTyByteWidth = DestTy.getElementTypeBitWidth() / 8;
  AffineExpr ByteUBExpr = UBMap.getResult(0).floorDiv(EltTyByteWidth);
  UBMap = AffineMap::get(UBMap.getNumDims(), UBMap.getNumSymbols(), ByteUBExpr);
  auto ForOp = Builder.create<AffineForOp>(Loc, LBArgs, LBMap, UBArgs, UBMap);

  // Get the src and dest offsets. The MemCpyInst may use GEPs to index into the
  // pointers, similar to how it is done for loads and stores, so we gather the
  // indices and add the for loop IV.
  auto CreateOffsets = [&](llvm::Value *Val,
                           SmallVector<mlir::Value> &MapOperands) {
    SmallVector<llvm::Value *> LLVMIndices = {};
    gatherArrayIndices(Val, LLVMIndices);
    SmallVector<AffineExpr> Exprs = {};
    SmallVector<mlir::Value> DimArgs = {};
    SmallVector<mlir::Value> SymArgs = {};
    for (int I = (int)LLVMIndices.size() - 1; I >= 0; I--) {
      AffineExpr Expr = getIndexAffineExpr(LLVMIndices[I], DimArgs, SymArgs);
      unsigned Pos = insertIntoVector(DimArgs, ForOp.getInductionVar());
      Expr = getAffineDimExpr(Pos, getContext()) + Expr;
      Exprs.push_back(Expr);
    }
    MapOperands.insert(MapOperands.end(), DimArgs.begin(), DimArgs.end());
    MapOperands.insert(MapOperands.end(), SymArgs.begin(), SymArgs.end());
    return AffineMap::get(DimArgs.size(), SymArgs.size(), Exprs, getContext());
  };
  SmallVector<mlir::Value> LdMapArgs = {};
  SmallVector<mlir::Value> StrMapArgs = {};
  AffineMap LdMap = CreateOffsets(MEMCPY->getSource(), LdMapArgs);
  AffineMap StrMap = CreateOffsets(MEMCPY->getDest(), StrMapArgs);

  // Create the load and store Op
  Builder.setInsertionPoint(ForOp.getLoopBody().front().getTerminator());
  auto LoadOp = Builder.create<AffineLoadOp>(Loc, Src, LdMap, LdMapArgs);
  Builder.create<AffineStoreOp>(Loc, LoadOp.getResult(), Dest, StrMap, StrMapArgs);
}

/// Generate the MLIR equivalent if an LLVM MemSetInst.
///
/// MLIR does not have an MemSetOp yet, so we generate a for loop to set
/// elements of a memref, defined by the MemSetInst.
void MLIRCodeGen::genMemSetInst(const MemSetInst *MEMSET) {
  // Set the insertion point to the end of the parent block
  OpBuilder::InsertionGuard Guard(Builder);
  setInsertionPointBeforeTerminator(getMLIRBlock(MEMSET->getParent()));
  Location Loc = getInstrLoc(MEMSET);

  // Get the operand
  mlir::Value Dest = genOperand(MEMSET->getDest());

  // Checks
  MemRefType DestTy = Dest.getType().dyn_cast<MemRefType>();
  assert(DestTy && "MemSet destination expected to be a memref.");
  assert(DestTy.getRank() == 1 &&
         "TODO: genMemSetInst can only handle 1D MemSet right now.");

  // Create a for loop to act as the memset
  AffineMap LBMap = AffineMap::getConstantMap(0, getContext());
  SmallVector<mlir::Value> LBArgs = {};
  SmallVector<mlir::Value> UBArgs = {};
  AffineMap UBMap = getLoopBound(MEMSET->getLength(), UBArgs);
  // The MemCpy's length is given in bytes, however we need it in elements; do
  // divide the length by the byte width of the element type.
  assert(UBMap.getNumResults() == 1 && "Map expected to have one result.");
  unsigned EltTyByteWidth = DestTy.getElementTypeBitWidth() / 8;
  AffineExpr ByteUBExpr = UBMap.getResult(0).floorDiv(EltTyByteWidth);
  UBMap = AffineMap::get(UBMap.getNumDims(), UBMap.getNumSymbols(), ByteUBExpr);
  auto ForOp = Builder.create<AffineForOp>(Loc, LBArgs, LBMap, UBArgs, UBMap);

  // Get the value to store. The value to store will be given as a single byte
  // that will be written into each byte of the element.
  ConstantInt *Val = dyn_cast<ConstantInt>(MEMSET->getValue());
  assert(Val && "TODO: Handle non-constant memset value.");
  assert(MEMSET->getValue()->getType()->isIntegerTy(8) &&
         "genMemSetInst is written with the assumption that the set value is a "
         "byte value.");
  uint64_t ByteVal = 0;
  for (unsigned I = 0; I < EltTyByteWidth; I++) {
    ByteVal = (ByteVal << 8) | Val->getZExtValue();
  }
  Constant *ConstVal = nullptr;
  APInt APIntVal = APInt(DestTy.getElementTypeBitWidth(), ByteVal, false);
  if (DestTy.getElementType().isIntOrIndex()) {
    ConstVal = ConstantInt::get(MEMSET->getContext(), APIntVal);
  } else {
    auto GetAPFloat = [&](const mlir::Type &ElementType, APInt &Value) {
      if (ElementType.isF32())
        return APFloat(APFloat::IEEEsingle(), Value);
      if (ElementType.isF16())
        return APFloat(APFloat::IEEEhalf(), Value);
      if (ElementType.isF64())
        return APFloat(APFloat::IEEEdouble(), Value);
      llvm_unreachable("Unhandled float semantic.");
    };
    APFloat APFloatVal = GetAPFloat(DestTy.getElementType(), APIntVal);
    ConstVal = ConstantFP::get(MEMSET->getContext(), APFloatVal);
  }
  mlir::Value StoreVal = genOperand(ConstVal);

  // Get the offsets into the memref. The MemCpyInst may use GEPs to index into
  // the pointers, similar to how it is done for loads and stores, so we gather
  // the indices and add the for loop IV.
  SmallVector<llvm::Value *> LLVMIndices = {};
  gatherArrayIndices(MEMSET->getDest(), LLVMIndices);
  SmallVector<AffineExpr> Exprs = {};
  SmallVector<mlir::Value> DimArgs = {};
  SmallVector<mlir::Value> SymArgs = {};
  for (int I = (int)LLVMIndices.size() - 1; I >= 0; I--) {
    AffineExpr Expr = getIndexAffineExpr(LLVMIndices[I], DimArgs, SymArgs);
    unsigned Pos = insertIntoVector(DimArgs, ForOp.getInductionVar());
    Expr = getAffineDimExpr(Pos, getContext()) + Expr;
    Exprs.push_back(Expr);
  }
  unsigned DimCount = DimArgs.size();
  unsigned SymCount = SymArgs.size();
  auto Map = AffineMap::get(DimCount, SymCount, Exprs, getContext());
  SmallVector<mlir::Value> MapOperands = {};
  MapOperands.insert(MapOperands.end(), DimArgs.begin(), DimArgs.end());
  MapOperands.insert(MapOperands.end(), SymArgs.begin(), SymArgs.end());

  // Create the store Op
  Builder.setInsertionPoint(ForOp.getLoopBody().front().getTerminator());
  Builder.create<AffineStoreOp>(Loc, StoreVal, Dest, Map, MapOperands);
}

/// Given an LLVM Value representing either the lower or upper bound of a loop,
/// generate an affine map for the bound and the map operands.
AffineMap MLIRCodeGen::getLoopBound(llvm::Value *Bound,
                                    SmallVector<mlir::Value> &Operands) {
  // Create an affine expression for the bound
  SmallVector<mlir::Value> Dims = {};
  SmallVector<mlir::Value> Symbols = {};
  AffineExpr Expr = getIndexAffineExpr(Bound, Dims, Symbols);
  // Insert the dim then the symbol operands into operands
  Operands.insert(Operands.end(), Dims.begin(), Dims.end());
  Operands.insert(Operands.end(), Symbols.begin(), Symbols.end());
  // Generate the affine map from the expression and the operands
  return AffineMap::get(Dims.size(), Symbols.size(), Expr, getContext());
}

/// Generate an MLIR AffineForOp for the given LLVM Loop.
///
/// If the Loop is not able to be expressed as an AffineForOp, an scf::ForOp
/// will be generated instead.
///
/// This method is intended to be
/// called after all the instructions of the function have been generated to
/// load the IterArgs, but must be called before setupCFG() so all required
/// values are generated before blocks start getting merged.
void MLIRCodeGen::genLoopOp(const Loop *L) {
  assert(LLVMToMLIRLoopMap.count(L) == 0 &&
         "Loop Op should only be generated once.");

  // Set the insertionpoint to the end of the temp block. The temp block is just
  // a place to store temporary ops. This Op will be moved to its final location
  // in setupCFG().
  OpBuilder::InsertionGuard Guard(Builder);
  Builder.setInsertionPointToEnd(&TempBlock);
  Location Loc = Builder.getUnknownLoc();

  LoopComponents &LC = LoopComponentMap[L];
  SmallVector<LoopIterArg> &LoopIterArgs = LoopIterArgsMap[L];

  // Get the iter args. these are essentially AffineForOps way of handling
  // PHINodes. The LoopIterArgsMap is populated in the genPHINode() method.
  SmallVector<mlir::Value> IterArgs = {};
  for (LoopIterArg &IterArg : LoopIterArgs)
    IterArgs.push_back(genOperand(IterArg.InitVal));

  // Get the induction variable. This is grabbed now since the reverse iteration
  // handling needs to use it.
  mlir::Value CastIV = genOperand(LC.IV);
  assert(mlir::isa<arith::IndexCastOp>(CastIV.getDefiningOp()) &&
         "Induction Variable expected to be the cast of the dummy index.");
  mlir::Value IndexIV = CastIV.getDefiningOp()->getOperand(0);

  // A loop can be represented as an AffineForOp as long as the step size is
  // static. This is assuming the start and bound values can be represented as
  // AffineExpressions.
  auto *StepValue = dyn_cast<ConstantInt>(LC.Step);
  Operation *ForOp = nullptr;
  Block::BlockArgListType RegionIterArgs;
  mlir::Value NewIV;
  mlir::Block *LoopBodyBlock = nullptr;
  if (StepValue) {
    // Generate an AffineForOp
    // Get the Lower Bound and Upper Bound Operands and Map
    SmallVector<mlir::Value> LBArgs = {};
    SmallVector<mlir::Value> UBArgs = {};
    AffineMap LBMap = getLoopBound(LC.Start, LBArgs);
    AffineMap UBMap = getLoopBound(LC.Bound, UBArgs);
    // Get the loop step size.
    int64_t Step = StepValue->getSExtValue();
    // Check if this is a reverse iteration. If so, the AffineForOp cannot
    // handle reverse iteration, so we need to iterate forward, but apply a
    // transformation on the IV so that it iterates backwards. For example, the
    // follow affine for cannot be expressed:
    //   affine.for i = 31 to 15 step -1
    //     function(i)
    // So we transform it into:
    //   affine.for i = 16 to 32 step 1 {
    //     j = (16 + 32 - 1) - i     // <-- (LB + UB - 1) - i
    //     function(j)
    //   }
    mlir::Value ReverseIV;
    if (Step < 0) {
      assert(LBMap.getNumResults() == 1 && UBMap.getNumResults() == 1 &&
             "TODO: Handle reverse iteration with multiple results, however "
             "this shouldnt happen.");
      // Reverse the step.
      Step *= -1;
      // Swap the LB and UB, adding one to both.
      AffineMap PlusOneTransform =
          AffineMap::get(1, 0, getAffineDimExpr(0, getContext()) + 1);
      AffineMap TempMap = LBMap;
      LBMap = PlusOneTransform.compose(UBMap);
      UBMap = PlusOneTransform.compose(TempMap);
      LBArgs.swap(UBArgs);
      // Replace all uses of the IV such that NewIV = LB + UB - 1 - OldIV
      // Shift the Dims and Syms of UB by the LB.
      AffineMap ShiftedUBMap = UBMap.shiftDims(LBMap.getNumDims())
                                   .shiftSymbols(LBMap.getNumSymbols());
      // Create the Reverse IV Map.
      unsigned TotalNumDims = LBMap.getNumDims() + UBMap.getNumDims();
      unsigned TotalNumSyms = LBMap.getNumSymbols() + UBMap.getNumSymbols();
      AffineExpr ReverseExpr = LBMap.getResult(0) + ShiftedUBMap.getResult(0) -
                               1 - getAffineDimExpr(TotalNumDims, getContext());
      AffineMap ReverseIVMap =
          AffineMap::get(TotalNumDims + 1, TotalNumSyms, ReverseExpr);
      // Get the args, {LBArgs, UBArgs, IV}
      SmallVector<mlir::Value> ReverseArgs = {};
      ReverseArgs.insert(ReverseArgs.end(), LBArgs.begin(), LBArgs.end());
      ReverseArgs.insert(ReverseArgs.end(), UBArgs.begin(), UBArgs.end());
      ReverseArgs.push_back(IndexIV);
      // Canonicalize and simplify the map and its operands to remove duplicate
      // args and make the affine expression simpler.
      canonicalizeMapAndOperands(&ReverseIVMap, &ReverseArgs);
      ReverseIVMap = simplifyAffineMap(ReverseIVMap);
      // Build an AffineApplyOp to perform the transformation on the IV, for now
      // store it with the cast, but this will need to be moved into the body of
      // the ForOp.
      OpBuilder::InsertionGuard Guard(Builder);
      Builder.setInsertionPoint(CastIV.getDefiningOp());
      ReverseIV = Builder.create<AffineApplyOp>(Loc, ReverseIVMap, ReverseArgs);
      // Replace all uses of the IV with this reversed IV.
      IndexIV.replaceAllUsesExcept(ReverseIV, ReverseIV.getDefiningOp());
    }
    // Get the body builder. This will create the affine yield Op which returns
    // the new value of the iter args after a loop trip.
    auto BodyBuilder = [this, &LoopIterArgs](OpBuilder &OpBuilder, Location Loc,
                                             mlir::Value Val,
                                             ValueRange ValueRange) {
      (void)Val;
      (void)ValueRange;
      SmallVector<mlir::Value> YieldOperands = {};
      for (LoopIterArg &IterArg : LoopIterArgs)
        YieldOperands.push_back(genOperand(IterArg.ExitVal));
      OpBuilder.create<AffineYieldOp>(Loc, YieldOperands);
    };
    // Create the AffineForOp
    auto AffForOp = Builder.create<AffineForOp>(
        Loc, LBArgs, LBMap, UBArgs, UBMap, Step, IterArgs, BodyBuilder);
    ForOp = AffForOp;
    RegionIterArgs = AffForOp.getRegionIterArgs();
    NewIV = AffForOp.getInductionVar();
    LoopBodyBlock = &AffForOp.getLoopBody().front();

    // If the induction variable was reversed, make sure that the AffineApplyOp
    // is the first operation in this block.
    if (ReverseIV)
      ReverseIV.getDefiningOp()->moveBefore(&LoopBodyBlock->front());
  } else {
    // Generate an scf::ForOp
    mlir::Value LB = getOrCreateIndex(LC.Start);
    mlir::Value UB = getOrCreateIndex(LC.Bound);
    mlir::Value Step = getOrCreateIndex(LC.Step);
    auto SCFForOp = Builder.create<scf::ForOp>(Loc, LB, UB, Step, IterArgs);
    ForOp = SCFForOp;
    RegionIterArgs = SCFForOp.getRegionIterArgs();
    NewIV = SCFForOp.getInductionVar();
    LoopBodyBlock = &SCFForOp.getLoopBody().front();
    // Create the yield operation if iter args are provided
    if (!LoopIterArgs.empty()) {
      SmallVector<mlir::Value> YieldOperands = {};
      for (LoopIterArg &IterArg : LoopIterArgs)
        YieldOperands.push_back(genOperand(IterArg.ExitVal));
      Builder.setInsertionPointToEnd(LoopBodyBlock);
      Builder.create<scf::YieldOp>(Loc, YieldOperands);
    }
  }

  // Remove the dummy induction variable and replace it with the new one. In the
  // map, the induction variable is the casted version of the dummy index
  // variable. The real index will be an IndexType, so we replaces all uses of
  // the dummy with the real induction variable and move the cast to the top of
  // the body of the ForOp. This cast will be removed later if it is found to
  // never be used.
  IndexIV.replaceAllUsesWith(NewIV);
  IndexIV.getDefiningOp()->erase();
  CastIV.getDefiningOp()->moveBefore(LoopBodyBlock->getTerminator());
  // Remove other PHINode Dummy values
  for (size_t I = 0, NumIterArgs = IterArgs.size(); I < NumIterArgs; I++) {
    // Remove the dummy variables used for the iter args and replace them with
    // the real iter_args.
    BlockArgument NewIterArg = RegionIterArgs[I];
    if (PHINode *PHI = LoopIterArgs[I].PHI)
      replaceMLIRValue(PHI, NewIterArg);
    // Remove the dummy variables used for the Exit Values and replace them with
    // the results of the AffineForOp.
    if (PHINode *ExitPHI = LoopIterArgs[I].ExitPHI)
      replaceMLIRValue(ExitPHI, ForOp->getResult(I));
  }
  // Insert the ForOp into the LoopMap
  LLVMToMLIRLoopMap[L] = ForOp;
}

/// Recursive helper method used to populate the Conditions of an if/else block.
/// For the conditions to be legal in affine, the condtions must be a
/// conjunction of constraints; So therefore no OR operations allowed, only AND
/// (for now at least).
static void populateConditions(llvm::Value *Root,
                               SmallVector<llvm::CmpInst *> &Conditions) {
  // Base case: If the root is a compare instruction, add to the vector of
  // conditions and return.
  if (auto *CMP = dyn_cast<llvm::CmpInst>(Root)) {
    Conditions.push_back(CMP);
    return;
  }

  if (auto *BO = dyn_cast<llvm::BinaryOperator>(Root)) {
    switch (BO->getOpcode()) {
    case llvm::Instruction::BinaryOps::And:
      // If the root is an AND Op, populate the conditions with the LHS and RHS
      // of the Op.
      populateConditions(BO->getOperand(0), Conditions);
      populateConditions(BO->getOperand(1), Conditions);
      return;
    default:
      llvm_unreachable("Unhandled Binary Constraint.");
    }
  }
  llvm_unreachable("Unhandled Constraint.");
}

/// Generate an MLIR AffineIfOp for the given LLVM If Header Block, the block
/// that constains the conditional branch (If Latch Block).
///
/// If the if is unable to be expressed as an AffineIfOp, an scf::IfOp will be
/// generated instead.
///
/// This method is intended to be called after all the
/// instructions of the function have been generated to load the EscVals, but
/// must be called before setupCFG() so all required values are generated before
/// blocks start getting merged.
void MLIRCodeGen::genIfOp(const BasicBlock *BB) {
  assert(LLVMToMLIRIfMap.count(BB) == 0 &&
         "If Op should only be generated once.");
  // Set the insertionpoint to the end of the temp block. The temp block is just
  // a place to store temporary ops. This Op will be moved to its final location
  // in setupCFG().
  OpBuilder::InsertionGuard Guard(Builder);
  Builder.setInsertionPointToEnd(&TempBlock);
  Location Loc = Builder.getUnknownLoc();
  // Get the conditional branch
  auto *BR = dyn_cast<BranchInst>(BB->getTerminator());
  assert(BR && BR->isConditional() &&
         "Terminator of If latch block expected to be a conditional branch.");
  // Check if the IfOp has any escaping scalars
  SmallVector<IfEscVal> &EscapeVals = IfEscValsMap[BB];
  bool HasEscVals = !EscapeVals.empty();
  // Check if the else has exactly one predecessor, this implies that there is
  // an else region.
  bool SingleElsePred = BR->getSuccessor(1)->hasNPredecessors(1);
  // An else region will need to be generated if there are any escaping values
  // or if the else has a single predecessor.
  bool WithElseRegion = HasEscVals || SingleElsePred;
  // Check if this can be lowered to an AffineIfOp, if it cannot, it will be
  // lowered to an scf::IfOp.
  llvm::Value *Cond = BR->getCondition();
  bool IsAffineIf = matchAffineCondition(Cond);
  // Get the result types for the IfOp. This should match any escaping values.
  SmallVector<mlir::Type> ResTypes = {};
  for (IfEscVal &EscVal : EscapeVals) {
    llvm::Type *ExitPHIType = EscVal.ExitPHI->getType();
    mlir::Type ResType = LLVMTypeToMLIRType(ExitPHIType, getContext());
    ResTypes.push_back(ResType);
  }
  // Create the IfOp
  Operation *IfOp = nullptr;
  mlir::Block *IfThenBlock = nullptr;
  mlir::Block *IfElseBlock = nullptr;
  if (IsAffineIf) {
    // Get a conjunction of conditions for the if/else
    SmallVector<CmpInst *> Conditions = {};
    populateConditions(Cond, Conditions);
    // Get Affine Expressions representing the constraints for each condition.
    SmallVector<mlir::Value> DimArgs = {};
    SmallVector<mlir::Value> SymArgs = {};
    SmallVector<bool> EqualFlags = {};
    SmallVector<AffineExpr> Constraints = {};
    for (CmpInst *CMP : Conditions) {
      // Equal flag will say if the condition is a "==" or a ">="
      bool EqualFlag = false;

      // Generate affine expressions for the LHS and RHS of the conditional.
      AffineExpr LHS = getIndexAffineExpr(CMP->getOperand(0), DimArgs, SymArgs);
      AffineExpr RHS = getIndexAffineExpr(CMP->getOperand(1), DimArgs, SymArgs);

      // Create the affine constraint given the compare predicate.
      AffineExpr AffineConstraint;
      // TODO: is there a difference between unsigned and signed comparison in
      // this case?
      switch (CMP->getPredicate()) {
      case CmpInst::Predicate::ICMP_SLT:
      case CmpInst::Predicate::ICMP_ULT:
        // LHS < RHS ====> -LHS + RHS - 1 >= 0
        AffineConstraint = -LHS + RHS - 1;
        break;
      case CmpInst::Predicate::ICMP_SLE:
      case CmpInst::Predicate::ICMP_ULE:
        // LHS <= RHS ===> -LHS + RHS >= 0
        AffineConstraint = -LHS + RHS;
        break;
      case CmpInst::Predicate::ICMP_SGT:
      case CmpInst::Predicate::ICMP_UGT:
        // LHS > RHS ====> LHS - RHS - 1 >= 0
        AffineConstraint = LHS - RHS - 1;
        break;
      case CmpInst::Predicate::ICMP_SGE:
      case CmpInst::Predicate::ICMP_UGE:
        // LHS >= RHS ===> LHS - RHS >= 0
        AffineConstraint = LHS - RHS;
        break;
      case CmpInst::Predicate::ICMP_EQ:
        // LHS == RHS ===> LHS - RHS == 0
        AffineConstraint = LHS - RHS;
        EqualFlag = true;
        break;
      default:
        // TODO: NEQ is a bit complicated, may require swapping then and else
        // blocks.
        llvm_unreachable("Unhandled if/else constraint predicate.");
      }
      // Add the constraint and the equal flags to the vector to be used to
      // generate the integer set.
      Constraints.push_back(AffineConstraint);
      EqualFlags.push_back(EqualFlag);
    }

    // Build the integer set
    unsigned DimCount = DimArgs.size();
    unsigned SymCount = SymArgs.size();
    auto Set = IntegerSet::get(DimCount, SymCount, Constraints, EqualFlags);
    // Combine the dim and symbol args into Args.
    SmallVector<mlir::Value> Args = {};
    Args.insert(Args.end(), DimArgs.begin(), DimArgs.end());
    Args.insert(Args.end(), SymArgs.begin(), SymArgs.end());
    // Create the IfOp
    auto AffIfOp =
        Builder.create<AffineIfOp>(Loc, ResTypes, Set, Args, WithElseRegion);
    IfOp = AffIfOp;
    IfThenBlock = AffIfOp.getThenBlock();
    if (WithElseRegion)
      IfElseBlock = AffIfOp.getElseBlock();
  } else {
    mlir::Value IfCond = genOperand(Cond);
    auto ScfIfOp =
        Builder.create<scf::IfOp>(Loc, ResTypes, IfCond, WithElseRegion);
    IfOp = ScfIfOp;
    IfThenBlock = ScfIfOp.thenBlock();
    if (WithElseRegion)
      IfElseBlock = ScfIfOp.elseBlock();
  }
  // Yield any escaping values in the then and else blocks.
  if (HasEscVals) {
    // Yield for then block
    Builder.setInsertionPointToEnd(IfThenBlock);
    SmallVector<mlir::Value> ThenYieldOperands = {};
    for (IfEscVal &EscVal : EscapeVals)
      ThenYieldOperands.push_back(genOperand(EscVal.TrueVal));
    if (IsAffineIf)
      Builder.create<AffineYieldOp>(Loc, ThenYieldOperands);
    else
      Builder.create<scf::YieldOp>(Loc, ThenYieldOperands);
    // Yield for else block
    Builder.setInsertionPointToEnd(IfElseBlock);
    SmallVector<mlir::Value> ElseYieldOperands = {};
    for (IfEscVal &EscVal : EscapeVals)
      ElseYieldOperands.push_back(genOperand(EscVal.FalseVal));
    if (IsAffineIf)
      Builder.create<AffineYieldOp>(Loc, ElseYieldOperands);
    else
      Builder.create<scf::YieldOp>(Loc, ElseYieldOperands);
  }
  // Replace the dummy values used for escaping values with the proper result of
  // the IfOp.
  for (size_t I = 0, NumEscVals = EscapeVals.size(); I < NumEscVals; I++) {
    PHINode *ExitPHI = EscapeVals[I].ExitPHI;
    replaceMLIRValue(ExitPHI, IfOp->getResult(I));
  }
  // Insert the IfOp into the IfMap
  LLVMToMLIRIfMap[BB] = IfOp;
}

/// Get the MemRefType from a given llvm Value.
MemRefType MLIRCodeGen::getMemrefType(llvm::Value *Val) {
  Shape *S = getOrCreateShape(Val);
  assert(S->getNumDims() != 0 && "Expected shape to have dimensions.");
  // Get the memref shape
  SmallVector<int64_t> MemRefShape = {};
  for (unsigned Dim = 0, NumDims = S->getNumDims(); Dim < NumDims; Dim++) {
    if (S->isDynamic()) {
      MemRefShape.push_back(ShapedType::kDynamicSize);
    } else {
      MemRefShape.push_back(S->getDim(Dim));
    }
  }

  // Get the memref's data type
  mlir::Type DataType = LLVMTypeToMLIRType(S->getElementType(), getContext());

  // Get the address space
  unsigned AddrSpace = S->getAddrSpace();
  switch (AddrSpace) {
  case 4:
    AddrSpace = 6;
    break;
  default:
    break;
  }
  auto ASAttr = Builder.getIntegerAttr(Builder.getI64Type(), AddrSpace);

  // Handle offsets and strides. For now assume dynamic offset and dynamic
  // strides.
  AffineMapAttr MapAttr;
  if (S->getOffset()) {
    // Offsets and strides are described by an AffineMap which will specify how
    // to delinearlize the index to properly access the memref. Since the
    // defalut AffineMap accesses memrefs the same way that LLVM accesses
    // pointers, we like to use the default unless we need to specify different
    // strides or offsets. Start building the map. Offset; if the offset was
    // constant, this would be a constant expr.
    AffineExpr MapExpr = getAffineSymbolExpr(0, getContext());
    // Strides
    for (unsigned Dim = 0, NumDims = S->getNumDims(); Dim < NumDims; Dim++) {
      AffineExpr SubMapExpr = getAffineDimExpr(Dim, getContext());
      for (unsigned StrideDim = Dim + 1; StrideDim < NumDims; StrideDim++) {
        // If the stride was constant, this would be a constant expr.
        SubMapExpr = getAffineSymbolExpr(StrideDim, getContext()) * SubMapExpr;
      }
      MapExpr = MapExpr + SubMapExpr;
    }
    AffineMap Map = AffineMap::get(S->getNumDims(), S->getNumDims(), MapExpr);
    MapAttr = AffineMapAttr::get(Map);
  }

  return MemRefType::get(MemRefShape, DataType, MapAttr, ASAttr);
}

/// Replace an LLVM Value in the MLIR CodeGen internals into a new
/// MLIR Value. The old val must be passed as an llvm::Value because we need to
/// update the LLVMToMLIRValueMap which requires the llvm Value as input.
void MLIRCodeGen::replaceMLIRValue(llvm::Value *OldVal, mlir::Value NewVal) {
  // If the OldVal is a pointer, get the root of the OldVal
  if (OldVal->getType()->isPointerTy())
    OldVal = getRoot(OldVal);
  // If the OldVal was never generated, just set the LLVMToMLIRValueMap.
  if (LLVMToMLIRValueMap.count(OldVal) == 0) {
    LLVMToMLIRValueMap[OldVal] = NewVal;
    return;
  }
  // Get the old MLIR value
  mlir::Value OldMLIRVal = LLVMToMLIRValueMap[OldVal];
  // Replace all of its uses with the new value and erase the old Op
  OldMLIRVal.replaceAllUsesWith(NewVal);
  OldMLIRVal.getDefiningOp()->erase();
  // Need to update the LLVMToMLIRValueMap.
  LLVMToMLIRValueMap[OldVal] = NewVal;
}

/// Check if a block has already been processed.
///
/// This means that MLIR Block associated with the given LLVM BasicBlock has
/// been inserted into another block, thus it should not be inserted into
/// another block.
bool MLIRCodeGen::hasBeenProcessed(const BasicBlock *BB) {
  return ProcessedBlocks.count(BB) == 1;
}

/// Merge the ChildBlock into the ParentBlock.
///
/// MLIR does not have a built-in method to merge two blocks, this is because
/// the BlockArguments are specific to the Op that contains the Block and must
/// be handled special. However, in the MLIRCodeGen we do not work with Block
/// arguments so we just merge the operations of the child block into the parent
/// block. Block args are only generated by the ForOps and IfOps and they should
/// never be the child block.
void MLIRCodeGen::mergeMLIRBlocks(mlir::Block *PB, mlir::Block *CB) {
  assert(CB->getNumArguments() == 0 &&
         "TODO: handle what to do with block arguments; however this shouldn't "
         "happen.");
  // Collect all the operations in the Child Block
  SmallVector<Operation *> OpsToMove;
  for (Operation &Op : *CB)
    OpsToMove.push_back(&Op);
  // If there is a terminator, insert all ops before it
  // else insert all ops at the end
  if (!PB->empty() && PB->back().hasTrait<OpTrait::IsTerminator>()) {
    assert((CB->empty() || !CB->back().hasTrait<OpTrait::IsTerminator>()) &&
           "TODO: Handle if both the parent and child have terminators; "
           "however this shouldn't happen.");
    Operation *Terminator = PB->getTerminator();
    for (Operation *Op : OpsToMove)
      Op->moveBefore(Terminator);
  } else {
    for (Operation *Op : OpsToMove)
      Op->moveBefore(PB, PB->end());
  }
}

/// Insert the MLIR ChildBlock pointed to by BB into the MLIR
/// ParentBlock pointed to by Parent.
void MLIRCodeGen::insertBlockIntoParent(const BasicBlock *Parent,
                                        const BasicBlock *BB) {
  mlir::Block *PB = getMLIRBlock(Parent);
  mlir::Block *CB = getMLIRBlock(BB);
  mergeMLIRBlocks(PB, CB);
  replaceMLIRBlock(BB, PB);
  ProcessedBlocks.insert(BB);
}

/// Insert the LoopBlock, pointed to by BB and defined by L, into
/// the MLIR ParentBlock pointed to by Parent.
///
/// The LoopOp was generated in the parseBlocks method, this method need only
/// move the Loop Op into the ParentBlock and insert the LoopBlock into its
/// body.
void MLIRCodeGen::insertLoopBlockIntoParent(const BasicBlock *Parent,
                                            const Loop *L,
                                            const BasicBlock *BB) {
  // Get the for Op from the LLVMToMLIRLoopMap, which was populated by the
  // genLoopOp method, and move the Op to the end of the parent block.
  assert(LLVMToMLIRLoopMap.count(L) == 1 &&
         "LoopOp not generated for LLVM Loop.");
  Operation *ForOp = LLVMToMLIRLoopMap[L];
  assert(ForOp->getBlock() == &TempBlock && "ForOp has already been moved.");
  mlir::Block *PB = getMLIRBlock(Parent);
  if (!PB->empty() && PB->back().hasTrait<OpTrait::IsTerminator>())
    ForOp->moveBefore(PB->getTerminator());
  else
    ForOp->moveBefore(PB, PB->end());

  // Get the ForOp body
  mlir::Block *LoopBody = nullptr;
  if (auto AffForOp = dyn_cast<AffineForOp>(ForOp))
    LoopBody = &AffForOp.getLoopBody().front();
  else if (auto SCFForOp = dyn_cast<scf::ForOp>(ForOp))
    LoopBody = &SCFForOp.getLoopBody().front();
  else
    llvm_unreachable("ForOp expected to be an AffineForOp or scf::ForOp.");

  // Merge the LoopBlock into the ForOp body
  mlir::Block *LoopBlock = getMLIRBlock(BB);
  mergeMLIRBlocks(LoopBody, LoopBlock);
  ProcessedBlocks.insert(BB);

  // The BB should now point to the body of the AffineFor
  replaceMLIRBlock(BB, LoopBody);
}

/// Generate the If Else logic in MLIR given the condition of the If and the If
/// and Else blocks pointed to by the If and Else Successor. The IfOp will be
/// inserted into the ParentBlock pointed to by the Parent.
///
/// The IfOp was generated in the parseBlocks method, this method need only move
/// the If Op into the ParentBlock and insert the If and Else Blocks into its
/// body.
void MLIRCodeGen::generateIfElse(const BasicBlock *Parent,
                                 const BasicBlock *IfSuccessor,
                                 const BasicBlock *ElseSuccessor,
                                 llvm::Value *Cond) {
  // The conditions are generated along with the rest of the operations in the
  // blocks.
  (void)Cond;

  // Get the if Op from the LLVMToMLIRIfMap, which was populated by the
  // genIfOp method, and move the Op to the end of the parent block.
  const BasicBlock *Pred;
  if (IfSuccessor)
    Pred = IfSuccessor->getSinglePredecessor();
  else if (ElseSuccessor)
    Pred = ElseSuccessor->getSinglePredecessor();
  else
    llvm_unreachable("Both If and Else successors cannot be nullptr.");
  assert(Pred && "IfElseSuccessor expected to have a single predecessor.");
  assert(LLVMToMLIRIfMap.count(Pred) == 1 &&
         "IfOp not generated for this block.");
  Operation *IfOp = LLVMToMLIRIfMap[Pred];
  assert(IfOp->getBlock() == &TempBlock && "IfOp has already been moved.");
  mlir::Block *PB = getMLIRBlock(Parent);
  if (!PB->empty() && PB->back().hasTrait<OpTrait::IsTerminator>())
    IfOp->moveBefore(PB->getTerminator());
  else
    IfOp->moveBefore(PB, PB->end());
  // Get the IfOp's then and else blocks
  mlir::Block *IfThenBlock = nullptr;
  mlir::Block *IfElseBlock = nullptr;
  if (auto AffIfOp = dyn_cast<AffineIfOp>(IfOp)) {
    IfThenBlock = AffIfOp.getThenBlock();
    if (AffIfOp.hasElse())
      IfElseBlock = AffIfOp.getElseBlock();
  } else if (auto ScfIfOp = dyn_cast<scf::IfOp>(IfOp)) {
    IfThenBlock = ScfIfOp.thenBlock();
    if (ScfIfOp.elseBlock())
      IfElseBlock = ScfIfOp.elseBlock();
  } else {
    llvm_unreachable("IfOp Op must be either an AffineIfOp or an scf::IfOp.");
  }
  // If there is a then, merge the then block into the IfOp
  if (IfSuccessor) {
    mlir::Block *ThenBlock = getMLIRBlock(IfSuccessor);
    mergeMLIRBlocks(IfThenBlock, ThenBlock);
    ProcessedBlocks.insert(IfSuccessor);
    // The IfSuccessor should now point to the then block of the IfOp
    replaceMLIRBlock(IfSuccessor, IfThenBlock);
  }
  // If there is an else, merge the else block into the IfOp
  if (ElseSuccessor) {
    mlir::Block *ElseBlock = getMLIRBlock(ElseSuccessor);
    mergeMLIRBlocks(IfElseBlock, ElseBlock);
    ProcessedBlocks.insert(ElseSuccessor);
    // The ElseSuccessor should now point to the else block of the IfOp
    replaceMLIRBlock(ElseSuccessor, IfElseBlock);
  }
}

/// Outlines the given mlir Block from its function into new function within the
/// same module with the given FuncName. This pass is entirely written in MLIR
/// and is designed to be run at the way end of conversion. The given block must
/// have a terminator.
/// TODO: This pass would probably fit better somewhere in the MLIR directory
/// rather than in this converter.
static void outlineMLIRBlock(Block *Block, llvm::StringRef FuncName,
                             OpBuilder &Builder) {
  // We will need to keep track of the terminator. The terminator will decide
  // what and how many results the outlined function will return.
  Operation *Terminator = Block->getTerminator();
  // We will need to walk the ops in the loop and collect any "external"
  // operands that will need to be passed into the outlined function.
  DenseSet<Operation *> InternalOps;
  DenseSet<mlir::Value> FuncArgsSet;
  Block->walk<WalkOrder::PreOrder>([&](Operation *Op) {
    for (mlir::Value Operand : Op->getOperands()) {
      // Get the op that defined the operand.
      // Block arguments are defined by the parent op of the region that
      // contains the block.
      Operation *DefOp = nullptr;
      if (auto BlkArg = Operand.dyn_cast<BlockArgument>())
        DefOp = BlkArg.getParentRegion()->getParentOp();
      else
        DefOp = Operand.getDefiningOp();
      // If the value was defined in the loop nest, it will not need to be
      // passed in.
      if (InternalOps.contains(DefOp))
        continue;
      // If the operand was not defined internally, add it to the set of
      // funcArgs.
      FuncArgsSet.insert(Operand);
    }
    // Insert this op into the internal ops.
    InternalOps.insert(Op);
  });

  // Collect the function args and their types from the set to get the function
  // type.
  SmallVector<mlir::Value> FuncArgs;
  SmallVector<mlir::Type> FuncArgTypes;
  for (auto Arg : FuncArgsSet) {
    FuncArgs.push_back(Arg);
    FuncArgTypes.push_back(Arg.getType());
  }
  auto FuncRetTypes = Terminator->getOperands().getTypes();
  auto FuncTy =
      mlir::FunctionType::get(Builder.getContext(), FuncArgTypes, FuncRetTypes);

  // Create the FuncOp and insert it directly after the Block's parent function.
  OpBuilder::InsertionGuard Guard(Builder);
  func::FuncOp Func = Block->getParentOp()->getParentOfType<func::FuncOp>();
  Builder.setInsertionPointAfter(Func);
  mlir::StringAttr Visibility = Builder.getStringAttr("private");
  Location Loc = Block->getParentOp()->getLoc();
  auto OutlinedFunc =
      Builder.create<func::FuncOp>(Loc, FuncName, FuncTy, Visibility);

  // Add the "target-cpu" attribute from the original function.
  StringRef TargetCPUAttrName = "target-cpu";
  auto TargetCPUAttr = Func->getAttr(TargetCPUAttrName);
  if (TargetCPUAttr)
    OutlinedFunc->setAttr(TargetCPUAttrName, TargetCPUAttr);

  // Use the builder's clone method to clone the ops from the block to the new
  // function. A mapper is used to map the function args to their uses in the
  // new function, as well as the new internal values.
  Builder.setInsertionPointToStart(OutlinedFunc.addEntryBlock());
  BlockAndValueMapping Mapper;
  for (size_t i = 0; i < FuncArgs.size(); i++)
    Mapper.map(FuncArgs[i], OutlinedFunc.getArgument(i));
  SmallVector<Operation *> OpsToErase;
  for (auto &Op : Block->getOperations()) {
    if (&Op == Terminator) {
      // The terminator is special, it should not be erased and, for the
      // outlined function, it should be a ReturnOp.
      Operation *TempTerminator = Builder.clone(Op, Mapper);
      Builder.create<func::ReturnOp>(Loc, TempTerminator->getOperands());
      TempTerminator->erase();
      continue;
    }
    Builder.clone(Op, Mapper);
    OpsToErase.push_back(&Op);
  }

  // Create the call to the outlined function at the start of the block
  Builder.setInsertionPointToStart(Block);
  auto OutlinedCallOp = Builder.create<func::CallOp>(Loc, OutlinedFunc, FuncArgs);

  // Set the terminator's operands to the outlined function's results.
  Terminator->setOperands(OutlinedCallOp->getResults());
  // Erase the block's ops. This must be done after the terminator and must be
  // done in reverse; this prevents us from deleting operands that have uses.
  for (int i = OpsToErase.size() - 1; i >= 0; i--) {
    OpsToErase[i]->erase();
  }
}

/// Final step in generating the IR. This is where we create the function Op and
/// insert it into the module.
void MLIRCodeGen::finalize() {
  // Intialize the Trampoline Builder
  TrampBuilder.initialize(this->F);
  // Collect all values that will become args to the kernel
  SmallVector<llvm::Value *> FuncArgs = {};
  for (auto &GlobalArg : this->GlobalValues) {
    FuncArgs.push_back(GlobalArg);
  }
  for (Argument &FuncArg : this->F->args()) {
    FuncArgs.push_back(&FuncArg);
  }
  // Populate argument mapping with any arguments from the LLVM Function that
  // were used in the MLIR Function. Also get the args for the MLIR function.
  SmallVector<mlir::Value> FuncArgVals = {};
  SmallVector<mlir::Type> FuncArgTypes = {};
  for (llvm::Value *FuncArg : FuncArgs) {
    // If an MLIR Value was not generated for an argument, it wasnt used.
    auto It = LLVMToMLIRValueMap.find(FuncArg);
    if (It == LLVMToMLIRValueMap.end())
      continue;
    // Collect the Args as MLIR Values and their types to be used to build the
    // FuncOp.
    mlir::Value MLIRArgValue = It->second;
    FuncArgVals.push_back(MLIRArgValue);
    FuncArgTypes.push_back(MLIRArgValue.getType());
    // Insert the argument and the type into the outlined argument
    auto MemRefTy = MLIRArgValue.getType().dyn_cast<MemRefType>();
    if (!MemRefTy) {
      // If the arg is not a memref, then pass the Arg and its type.
      llvm::Type *Ty = FuncArg->getType();
      TrampBuilder.addScalarArg(FuncArg, Ty);
      continue;
    }
    int64_t Rank = MemRefTy.getRank();
    // When MemRefs are lowered from std to LLVM (without bare pointer
    // convention), they are lowered into a list of arguments like so:
    // memref<16x32xf32>: float*,     float*,    i64,    i64, i64,   i64, i64
    //                 | AllocPtr | AllignPtr | Offset |   shape   |  stride  |
    // For more information see: https://mlir.llvm.org/docs/TargetLLVMIR/
    // TODO: add a option to turn bare pointer on or off in the converter, for
    // now always assume bare pointer is off.
    // Get the type of the pointer
    assert(FuncArg->getType()->isPointerTy() &&
           "MemRefs are expected to be a pointer.");
    llvm::Type *Ty = nullptr;
    Shape *S = getOrCreateShape(FuncArg);
    if (!llvm::isa<FixedVectorType>(FuncArg->getType()))
      Ty = S->getElementType()->getPointerTo(S->getAddrSpace());
    else
      Ty = S->getElementType();
    // Get the strides and offsets
    SmallVector<int64_t> Strides = {};
    int64_t Offset = 0;
    if (failed(getStridesAndOffset(MemRefTy, Strides, Offset)))
      llvm_unreachable("Failed to get strides and offsets of memref.");
    // Get the index type
    auto *IndexType =
        llvm::IntegerType::get(this->F->getContext(), this->indexBitwidth);
    // Add the list of args to the outlined arguments
    // Allocated pointer. This is only used for deallocating the memref.
    if (S->isSimpleWrapper())
      TrampBuilder.addSyclWrapperArg(FuncArg, Ty);
    else
      TrampBuilder.addPointerArg(FuncArg, Ty);
    // Alligned pointer. The pointer used to index into the memref.
    if (S->isSimpleWrapper())
      TrampBuilder.addSyclWrapperArg(FuncArg, Ty);
    else
      TrampBuilder.addPointerArg(FuncArg, Ty);
    // Offset
    if (!ShapedType::isDynamicStrideOrOffset(Offset)) {
      Value *OffsetRoot = ConstantInt::get(IndexType, Offset);
      TrampBuilder.addScalarArg(OffsetRoot, IndexType);
    } else {
      Shape *OffsetShape = S->getOffset();
      assert(OffsetShape && "Dynamic offset expected to have an offset shape.");
      Shape *RangeShape = S->getRange();
      assert(RangeShape && "Dynamic offset expected to have a range shape.");
      Value *OffsetRoot = OffsetShape->getRoot();
      Value *RangeRoot = RangeShape->getRoot();
      TrampBuilder.addSingleSyclIDArg(OffsetRoot, RangeRoot, IndexType, Rank);
    }
    // Size (shape)
    Shape *DimShape = S->getRange();
    for (int64_t i = 0; i < Rank; i++) {
      int64_t Dim = MemRefTy.getDimSize(i);
      if (!ShapedType::isDynamic(Dim)) {
        Value *DimVal = llvm::ConstantInt::get(IndexType, Dim);
        TrampBuilder.addScalarArg(DimVal, IndexType);
        continue;
      }
      assert(DimShape && "Dynamic dim expected to have a range shape.");
      Value *DimRoot = DimShape->getRoot();
      TrampBuilder.addSyclRangeArg(DimRoot, IndexType, i);
    }
    // Strides
    // First do a pass to see if strides are dynamic (except the last, which
    // should always be one.)
    unsigned NumDynamicStrides = 0;
    for (int64_t Stride : Strides) {
      if (ShapedType::isDynamicStrideOrOffset(Stride))
        NumDynamicStrides++;
    }
    // If all strides are static, then we just add then to the argument
    if (NumDynamicStrides == 0) {
      for (int64_t Stride : Strides) {
        Value *StrideVal = llvm::ConstantInt::get(IndexType, Stride);
        TrampBuilder.addScalarArg(StrideVal, IndexType);
      }
    } else {
      // Otherwise if any strides are dynamic, we use the range passed into the
      // kernel instead.
      assert(NumDynamicStrides == Strides.size() - 1 &&
             "Assuming all but the last dimension of the shape have a dynamic "
             "stride");
      TrampBuilder.addStrideArgs(DimShape->getRoot(), Strides.size(),
                                 IndexType);
    }
  }

  // Create the FuncOp
  OpBuilder::InsertionGuard Guard(Builder);
  Builder.setInsertionPointToEnd(&Module.getBodyRegion().front());
  Location Loc = Builder.getUnknownLoc();
  string FuncName = "mlir_func" + this->F->getName().str();
  auto FuncTy = mlir::FunctionType::get(getContext(), FuncArgTypes, {});
  auto Func = Builder.create<func::FuncOp>(Loc, StringRef(FuncName), FuncTy);

  // get the function block.
  mlir::Block *FuncBlock = getMLIRBlock(&this->F->getEntryBlock());
  // Merge the FuncBlock into the entry block of the Func.
  mergeMLIRBlocks(Func.addEntryBlock(), FuncBlock);

  // Replace the placeholder args with the real ones
  for (size_t I = 0, NumFuncArgs = FuncArgVals.size(); I < NumFuncArgs; I++) {
    FuncArgVals[I].replaceAllUsesWith(Func.getArgument(I));
    FuncArgVals[I].getDefiningOp()->erase();
  }

  // Set the KernelName of the trampoline function to the FuncName
  TrampBuilder.setKernelName(FuncName);
  // Finalize the Trampoline Builder
  TrampBuilder.finalize();

  // Ensure that all indices are valid dims/symbols, if not the block that
  // contains them will need to be outlined to its own function. This makes the
  // symbols valid in the eyes of the Affine dialect.
  // TODO: The block being outlined could either be the block that contains the
  // troubled op, or the affine region that is being invalidated. This first was
  // chosen for its simplicity, however the latter may be better for some
  // applications; consider using the second method.
  DenseSet<Block *> OutlineBlocksSet;
  SmallVector<Block *> OutlineBlocks;
  auto areValidAffineIndices = [&](OperandRange indices) {
    for (mlir::Value Idx : indices)
      if (!isValidSymbol(Idx) && !isValidDim(Idx))
        return false;
    return true;
  };
  Func.walk<WalkOrder::PreOrder>([&](Operation *Op) {
    Block *ParentBlock = Op->getBlock();
    if (OutlineBlocksSet.contains(ParentBlock))
      return;
    bool IsValid = true;
    if (auto StoreOp = dyn_cast<AffineStoreOp>(Op))
      IsValid = areValidAffineIndices(StoreOp.getIndices());
    else if (auto LoadOp = dyn_cast<AffineLoadOp>(Op))
      IsValid = areValidAffineIndices(LoadOp.getIndices());
    else if (auto VecStoreOp = dyn_cast<AffineVectorStoreOp>(Op))
      IsValid = areValidAffineIndices(VecStoreOp.getIndices());
    else if (auto VecLoadOp = dyn_cast<AffineVectorLoadOp>(Op))
      IsValid = areValidAffineIndices(VecLoadOp.getIndices());
    else if (auto IfOp = dyn_cast<AffineIfOp>(Op))
      IsValid = areValidAffineIndices(IfOp.getOperands());
    else if (auto ForOp = dyn_cast<AffineForOp>(Op)) {
      IsValid = areValidAffineIndices(ForOp.getUpperBoundOperands()) &&
                areValidAffineIndices(ForOp.getLowerBoundOperands());
    }

    if (!IsValid) {
      OutlineBlocks.push_back(ParentBlock);
      OutlineBlocksSet.insert(ParentBlock);
    }
  });
  // OutlineBlocks now contains the blocks that need to be outlined. The vector
  // should be traversed in such a way that blocks are outlined before the block
  // that contains them; and since the blocks are inserted in a breadth first
  // fasion, we need to traverse the vector in reverse order.
  unsigned OutlineID = 0;
  for (Block *BlockToOutline : llvm::reverse(OutlineBlocks)) {
    std::string OutlinedFuncName =
        FuncName + "_converter_outlined_" + std::to_string(OutlineID);
    outlineMLIRBlock(BlockToOutline, OutlinedFuncName, Builder);
    OutlineID++;
  }

  LLVM_DEBUG(dbgs() << "Module after func gen:\n" << Module << "\n");
}

static MemRefType removeAddressSpace(MemRefType T) {
  // For now we are ignoring the affine maps when generating the main for
  // testing. This is due solely to the fact the MLIR cannot lower memref.alloc
  // statements with non-identity AffineMaps. This may be fixed in the future
  // but for now we can just remove the AffineMaps.
  // return MemRefType::get(T.getShape(), T.getElementType(), T.getLayout());
  return MemRefType::get(T.getShape(), T.getElementType());
}

/// Generate a main function that will call all of the functions within the
/// module.
Error MLIRCodeGen::generateMainForTesting() {
  LLVM_DEBUG(dbgs() << "=============== Generating Main ===============\n");
  // Collect all the funcs we need to call
  SmallVector<func::FuncOp> FuncsToCall = {};
  Module.walk([&](func::FuncOp Op) {
    if (!Op.isDeclaration())
      FuncsToCall.push_back(Op);
  });

  // Create an insertion Guard just in case
  OpBuilder::InsertionGuard Guard(Builder);

  // Need to replace the memrefs in address spaces into the default address
  // space since we are compiling on host.
  // Memrefs are currently only generated by args and alloc statements
  for (func::FuncOp Func : FuncsToCall) {
    LLVM_DEBUG(dbgs() << "Setting the memrefs to default address space in: "
                      << Func.getName() << "\n");
    // Replace the arguments
    for (unsigned I = 0, NumArgs = Func.getNumArguments(); I < NumArgs; I++) {
      BlockArgument Arg = Func.getArgument(I);
      auto MemrefType = Arg.getType().dyn_cast<MemRefType>();
      // If the arg is not a memref type in another address sapce, ignore.
      if (!MemrefType || MemrefType.getMemorySpaceAsInt() == 0)
        continue;
      LLVM_DEBUG(dbgs() << "\tFound memref arg not in default address space: "
                        << Arg << "\n");
      // Create a memref with the same shape and type
      auto NewMemrefType = removeAddressSpace(MemrefType);
      // Insert the memref into the function in the same position as the old
      // arg. This pushes the old arg to position (I+1).
      Func.insertArgument(I, NewMemrefType, DictionaryAttr::get(getContext()),
                          Arg.getLoc());
      // Replace the old arg with the new arg and erase the old one
      BlockArgument NewArg = Func.getArgument(I);
      Arg.replaceAllUsesWith(NewArg);
      Func.eraseArgument(I + 1);
      LLVM_DEBUG(dbgs() << "\t\tReplaced arg with: " << NewArg << "\n");
    }
    // Replace locals declared inside the function
    SmallVector<Operation *> OpsToErase = {};
    Func.walk([&](Operation *Op) {
      // If there are no results, return
      if (Op->getNumResults() == 0)
        return;
      // If the result is not a memref or is in the default address space,
      // return.
      MemRefType MrfTy = Op->getResult(0).getType().dyn_cast<MemRefType>();
      if (!MrfTy || MrfTy.getMemorySpaceAsInt() == 0)
        return;

      LLVM_DEBUG(dbgs() << "\tFound local memref not in default address space: "
                        << *Op << "\n");

      // Create the new memref type in default address space
      auto NewMrfTy = removeAddressSpace(MrfTy);
      // Set the insertion point for the NewOp.
      Builder.setInsertionPoint(Op);
      Location Loc = Op->getLoc();

      Operation *NewOp;
      assert(!mlir::isa<memref::AllocOp>(Op) &&
             "TODO: Handle Alloc ops. Alloc ops may be special.");
      if (auto Alloca = dyn_cast<memref::AllocaOp>(Op)) {
        // Create a new AllocaOp and replace the old one
        NewOp = Builder.create<memref::AllocaOp>(Loc, NewMrfTy);
      } else if (auto SEL = dyn_cast<arith::SelectOp>(Op)) {
        // Create a new SelectOp and replace the old one
        NewOp =
            Builder.create<arith::SelectOp>(Loc, NewMrfTy, SEL.getCondition(),
                                     SEL.getTrueValue(), SEL.getFalseValue());
      } else {
        llvm_unreachable("Unhandled Op for stripping Memref address space.");
      }
      Op->replaceAllUsesWith(NewOp);
      OpsToErase.push_back(Op);
      LLVM_DEBUG(dbgs() << "\t\tReplaced Op with: " << NewOp << "\n");
    });
    // Delete all of the ops marked for death
    for (Operation *Op : OpsToErase) {
      Op->erase();
    }
  }

  // Create the main function
  auto FuncType = Builder.getFunctionType({}, {Builder.getI32Type()});
  Builder.setInsertionPointToEnd(&Module.getBodyRegion().front());
  Location Loc = Builder.getUnknownLoc();
  auto MainFunc = Builder.create<func::FuncOp>(Loc, StringRef("main"), FuncType);
  mlir::Block *FuncBlock = MainFunc.addEntryBlock();
  Builder.setInsertionPointToStart(FuncBlock);
  // Add the function calls
  mlir::Value OneIndex;
  for (func::FuncOp Func : FuncsToCall) {
    LLVM_DEBUG(dbgs() << "Creating call to: " << Func.getName() << "\n");
    SmallVector<mlir::Value> Operands = {};
    for (BlockArgument Arg : Func.getArguments()) {
      if (auto MemrefTy = Arg.getType().dyn_cast<MemRefType>()) {
        SmallVector<mlir::Value> dynSzs = {};
        SmallVector<mlir::Value> SymOpr = {};
        int64_t NumDyDims = MemrefTy.getNumDynamicDims();
        for (int64_t I = 0; I < NumDyDims; I++) {
          if (!OneIndex)
            OneIndex =
                Builder.create<arith::ConstantIndexOp>(Loc, 1).getResult();
          dynSzs.push_back(OneIndex);
        }
        if (auto MemrefLayout = MemrefTy.getLayout()) {
          AffineMap MemMap = MemrefLayout.getAffineMap();
          unsigned NumSymbols = MemMap.getNumSymbols();
          for (unsigned I = 0; I < NumSymbols; I++) {
            if (!OneIndex)
              OneIndex =
                  Builder.create<arith::ConstantIndexOp>(Loc, 1).getResult();
            SymOpr.push_back(OneIndex);
          }
        }
        auto Operand =
            Builder.create<memref::AllocOp>(Loc, MemrefTy, dynSzs, SymOpr);
        Operands.push_back(Operand.getResult());
      } else if (auto ZeroAttr = Builder.getZeroAttr(Arg.getType())) {
        auto ZeroVal =
            Builder.create<arith::ConstantOp>(Loc, ZeroAttr, Arg.getType());
        Operands.push_back(ZeroVal);
      } else {
        llvm_unreachable("Unhandled Func Arg when generating @main.");
      }
    }
    Builder.create<func::CallOp>(Loc, Func, Operands);
  }

  // Create the return of 0
  auto ConstZero = Builder.create<arith::ConstantIntOp>(Loc, 0, 32);
  Builder.create<func::ReturnOp>(Loc, ConstZero.getResult());

  return Error::success();
}
