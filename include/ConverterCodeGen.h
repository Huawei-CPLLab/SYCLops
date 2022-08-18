//===-- ConverterCodeGen.h - LLVMIR To AKGIR/MLIR -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Contains the base class for the MLIR and AKG CodeGens.
//
//===----------------------------------------------------------------------===//

#ifndef SYCLOPS_INCLUDE_CONVERTERCODEGEN_H
#define SYCLOPS_INCLUDE_CONVERTERCODEGEN_H

#include "TrampolineBuilder/TrampolineBuilder.h"
#include "Util/ConverterUtil.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/ValueMap.h"
#include "llvm/Support/Error.h"

namespace llvm {
class Loop;
class LoopInfo;
class DominatorTree;
class ScalarEvolution;
namespace converter {

// Common base class of MLIR and AKG generators. Analyzes the LLVM IR for
// conversion.
class CodeGen {
public:
  explicit CodeGen(LLVMContext &Ctx);
  // Clears the data structures and flags
  void reset();
  // Attempts to converts the specified function F with its analysis results
  Error convert(Function *F);
  // Writes IR to file
  virtual llvm::Error writeToFile(const std::string &OutFile) = 0;

protected:
  // Builder for building the trampoline functions
  TrampolineBuilder TrampBuilder;
  // Necessary information regarding loops for each loop
  DenseMap<const Loop *, LoopComponents> LoopComponentMap;
  // Maps values to their corresponding shapes
  ValueMap<Value *, Shape> ShapeMap;
  // Number of each Shape, for name generation purposes
  unsigned ArgCounter;
  unsigned IndexCounter;
  unsigned LocalCounter;
  unsigned DimCounter;
  unsigned ConstCounter;
  unsigned GlobalCounter;

  // Function being converted
  Function *F;
  // Analysis information of the function
  LoopInfo *LI;
  DominatorTree *DT;
  ScalarEvolution *SE;
  // Pass managers
  FunctionAnalysisManager FAM;

  // Specific Pattern Matching Methods=========================================
  bool matchExtractParallelId(const ExtractElementInst *EEI, GlobalValue *&GV,
                              unsigned &Id);
  bool matchBufferAccess(Value *Index, SmallVector<Value *> &Indices);

  // Internal Helper Methods ==================================================
  bool isLoopIV(const Value *V);
  PHINode *findTrip2LoopIV(const Loop *L);
  Shape *getOrCreateShape(Value *SourceVal);
  Value *gatherArrayIndices(Value *Operand, SmallVector<Value *> &Indices);
  Value *getRoot(Value *V);
  virtual void parseBlocks() = 0;
  virtual bool hasBeenProcessed(const BasicBlock *BB) = 0;
  virtual void insertBlockIntoParent(const BasicBlock *Parent,
                                     const BasicBlock *BB) = 0;
  virtual void insertLoopBlockIntoParent(const BasicBlock *Parent,
                                         const Loop *L,
                                         const BasicBlock *BB) = 0;
  virtual void generateIfElse(const BasicBlock *Parent,
                              const BasicBlock *IfSuccessor,
                              const BasicBlock *ElseSuccessor, Value *Cond) = 0;
  virtual void preprocess();
  virtual void finalize() = 0;
  virtual void resetCodeGen() = 0;

  // Member Methods ===========================================================
  Error parseLoop(const Loop *L);
  void insertBlock(const BasicBlock *Parent, const BasicBlock *BB,
                   bool OnlyGenerateChildren = false);
  void setupCFG();
};

} // namespace converter
} // namespace llvm

#endif // SYCLOPS_INCLUDE_CONVERTERCODEGEN_H
