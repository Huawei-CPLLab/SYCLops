//===-- MLIRCodeGen.h - MLIR CodeGen Declarations -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef SYCLOPS_INCLUDE_MLIRCODEGEN_MLIRCODEGEN_H
#define SYCLOPS_INCLUDE_MLIRCODEGEN_MLIRCODEGEN_H

#include "../ConverterCodeGen.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include <string>

namespace mlir {
class Value;
} // namespace mlir

namespace llvm {
class MemCpyInst;
class MemSetInst;
class Loop;
class Value;
class BinaryOperator;
namespace converter {

// AffineForOps have the ability to carry "escaping scalars" using something
// called "iter_args". For the construction of these iter_args, two pieces of
// information are needed: the initial value of the arg, and the update value
// (the value it should be upddated to at the end of each loop trip). The
// PHINode is also needed for more information, if needed, and handling the
// dummy instruction. The ExitPHI represents any PHINode that uses the ExitVal
// outside of the loop. This PHINode's dummy instruction will be replaced by the
// result of the AffineForOp.
struct LoopIterArg {
  // The PHI that generated this iter_arg
  PHINode *PHI = nullptr;
  // The Initial value of the iter_arg
  llvm::Value *InitVal = nullptr;
  // The update value of the iter_arg
  llvm::Value *ExitVal = nullptr;
  // The PHI that will use this iter_arg outside of the AffineForOp (escaping
  // val)
  PHINode *ExitPHI = nullptr;
};

// AffineIfOps also have the ability to carry "escaping scalars" using their
// yield ops. If one or more values are being yielded by and AffineIfOp, the
// same number and type of values need to be yielded in both the true and false
// blocks of the op. These values can be used outside of the op as the result of
// the op. To handle these, we need the yield value if true, the yield value if
// false, and the PHINode that is calling it.
struct IfEscVal {
  // The PHINode that generated this escaping val
  PHINode *ExitPHI = nullptr;
  // The value of the escaping val in the true block
  llvm::Value *TrueVal = nullptr;
  // The value of the escaping val in the false block
  llvm::Value *FalseVal = nullptr;
};

class MLIRCodeGen : public CodeGen {
public:
  explicit MLIRCodeGen(const llvm::Module &M, unsigned indexBitwidth = 0);
  ~MLIRCodeGen();
  mlir::MLIRContext *getContext();
  mlir::OpBuilder *getBuilder();
  mlir::ModuleOp *getModule();
  llvm::Error writeToFile(const std::string &OutFile) override;
  llvm::Error verify();
  llvm::Error generateMainForTesting();

private:
  // Internal Variables ========================================================
  mlir::MLIRContext Ctx;
  mlir::OpBuilder Builder;
  mlir::ModuleOp Module;
  mlir::Block TempBlock;
  unsigned indexBitwidth;
  ValueMap<const BasicBlock *, mlir::Block *> LLVMToMLIRBlockMap;
  ValueMap<const llvm::Value *, mlir::Value> LLVMToMLIRValueMap;
  DenseMap<const Loop *, mlir::Operation *> LLVMToMLIRLoopMap;
  ValueMap<const BasicBlock *, mlir::Operation *> LLVMToMLIRIfMap;
  DenseMap<const llvm::Value *, mlir::Value> IndexMap;
  ValueMap<const llvm::Value *, mlir::Value> LLVMToMLIRIndexValueMap;
  SmallVector<llvm::Value *> GlobalValues;
  DenseMap<const Loop *, SmallVector<LoopIterArg>> LoopIterArgsMap;
  ValueMap<const BasicBlock *, SmallVector<IfEscVal>> IfEscValsMap;
  DenseSet<const BasicBlock *> ProcessedBlocks;

  // Overriden virtual methods =================================================
  void resetCodeGen() override;
  bool hasBeenProcessed(const BasicBlock *BB) override;
  void parseBlocks() override;
  void insertBlockIntoParent(const BasicBlock *Parent,
                             const BasicBlock *BB) override;
  void insertLoopBlockIntoParent(const BasicBlock *Parent, const Loop *L,
                                 const BasicBlock *BB) override;
  void generateIfElse(const BasicBlock *Parent, const BasicBlock *IfSuccessor,
                      const BasicBlock *ElseSuccessor,
                      llvm::Value *Cond) override;
  void finalize() override;

  // Internal helper methods ==================================================
  void setInsertionPointBeforeTerminator(mlir::Block *B);
  mlir::Location getInstrLoc(const Instruction *I);
  mlir::Block *getMLIRBlock(const BasicBlock *Val);
  void replaceMLIRBlock(const BasicBlock *OldBlk, mlir::Block *NewBlk);
  mlir::Value createDummyVal(mlir::Type DummyType);
  mlir::MemRefType getMemrefType(llvm::Value *Val);
  void replaceMLIRValue(llvm::Value *OldVal, mlir::Value NewVal);
  mlir::Value getOrCreateIndex(llvm::Value *LLVMVal);
  mlir::AffineExpr getIndexAffineExpr(llvm::Value *Index,
                                      SmallVector<mlir::Value> &DimOperands,
                                      SmallVector<mlir::Value> &SymbolOperands,
                                      bool IgnoredOffsets = false);
  mlir::Value genMemrefOperand(llvm::Value *Operand, mlir::AffineMap &Map,
                               SmallVector<mlir::Value> &MapOperands);
  mlir::Value genOperand(Value *Operand);
  mlir::Value genArgumentOperand(Argument *Arg);
  mlir::Value genGlobalOperand(GlobalValue *GO);
  mlir::Value genConstantOperand(const Constant *CO);
  mlir::Value genInstructionOperand(Instruction *Inst);
  void genReturnInst(const ReturnInst *RET);
  mlir::Value genUnaryOperator(const UnaryOperator *UO);
  mlir::Value genBinaryOperator(const BinaryOperator *BO);
  mlir::Value genExtractElementInst(ExtractElementInst *EE);
  mlir::Value genShuffleVectorInst(const ShuffleVectorInst *SV);
  mlir::Value genAllocaInst(AllocaInst *AI);
  mlir::Value genLoadInst(LoadInst *LI);
  void genStoreInst(StoreInst *SI);
  mlir::Value genCastInst(const CastInst *CI);
  mlir::Value genCmpInst(const CmpInst *CI);
  mlir::Value genPHINode(PHINode *PHI);
  mlir::Value genSelectInst(const SelectInst *SI);
  mlir::Value genCallInst(const CallInst *CI);
  void genMemCpyInst(const MemCpyInst *MEMCPY);
  void genMemSetInst(const MemSetInst *MEMSET);
  void genLoopOp(const Loop *L);
  void genIfOp(const BasicBlock *BB);
  void mergeMLIRBlocks(mlir::Block *PB, mlir::Block *CB);
  mlir::AffineMap getLoopBound(llvm::Value *Bound,
                               SmallVector<mlir::Value> &Operands);
};

} // namespace converter
} // namespace llvm

#endif // SYCLOPS_INCLUDE_MLIRCODEGEN_MLIRCODEGEN_H
