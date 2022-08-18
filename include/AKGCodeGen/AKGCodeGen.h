//===-- AKGCodeGen.h - AKG CodeGen Declarations ---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef SYCLOPS_INCLUDE_AKGCODEGEN_AKGCODEGEN_H
#define SYCLOPS_INCLUDE_AKGCODEGEN_AKGCODEGEN_H

#include "../ConverterCodeGen.h"
#include "../Util/ConverterUtil.h"
#include "AKGBuilder.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/IR/LLVMContext.h"

namespace llvm {
class IntrinsicInst;
namespace converter {

class AKGCodeGen : public CodeGen {
private:
  void finalize() override;
  void parseBlocks() override;
  bool hasBeenProcessed(const BasicBlock *BB) override;
  void insertBlockIntoParent(const BasicBlock *Parent,
                             const BasicBlock *BB) override;
  void insertLoopBlockIntoParent(const BasicBlock *Parent, const Loop *L,
                                 const BasicBlock *BB) override;
  void generateIfElse(const BasicBlock *Parent, const BasicBlock *IfSuccessor,
                      const BasicBlock *ElseSuccessor, Value *Cond) override;
  void resetCodeGen() override;

  // Internal Variables =======================================================
  // the generated AKG kernel
  std::string CodeOutput;

  AKGBuilder Builder;

  bool InvertCmp;

  size_t HybridFunctionID = 0;

  // Containers ===============================================================
  // Whenever a value is generated as a statement, it will get mapped to this
  // container so that we don't have to re-generate the same statement multiple
  // times.
  DenseMap<const Value *, Statement *> StmtMap;

  // When a store is made redundant by generating an assignment from genPHI,
  // the store is added here so that we can skip generating them.
  // TODO: currently no longer being used due to changes in genPHI.
  SmallPtrSet<const Instruction *, 4> SkipInstrSet;

  // Internal helper methods
  // ==================================================
  std::string typeToString(const Type *Ty) const;
  std::string genShapeDecl(Shape *S) const;
  void genBlock(BasicBlock *BB);
  void genPHINode(PHINode *PHI);
  void genStoreInst(StoreInst *SI);
  void genMemSetOrCpy(IntrinsicInst *II, bool IsCpy);
  void genIntrInst(IntrinsicInst *II);
  Statement *genIfCond(Value *Cond, bool Inverse);
  void genInstruction(Instruction *I);
  Statement *genArrayAccess(Value *V);
  Statement *genBinaryOperator(const BinaryOperator *BO);
  Statement *genSelectInst(SelectInst *SI);
  Statement *genCmpInst(const CmpInst *CI);
  Statement *genInstructionOperand(Instruction *Operand);
  Statement *genOperand(Value *Operand);
  std::string genVar(Value *V);
  std::string genLoopCond(const Loop *L);

public:
  explicit AKGCodeGen(LLVMContext &Ctx);
  llvm::Error writeToFile(const std::string &OutFile) override;
  std::string &getCodeOutput();
};

} // namespace converter
} // namespace llvm

#endif // SYCLOPS_INCLUDE_AKGCODEGEN_AKGCODEGEN_H
