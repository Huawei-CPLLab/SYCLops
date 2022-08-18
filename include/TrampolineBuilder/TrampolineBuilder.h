//===-- TrampolineBuilder.h - Trampoline Func Call Builder ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Builder class intended to encapsulate the creation of the trampoline calls.
//
//===----------------------------------------------------------------------===//

#ifndef SYCLOPS_INCLUDE_TRAMPOLINEBUILDER_TRAMPOLINEBUILDER_H
#define SYCLOPS_INCLUDE_TRAMPOLINEBUILDER_TRAMPOLINEBUILDER_H

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/LLVMContext.h"

namespace llvm {
namespace converter {

class TrampolineBuilder {
public:
  explicit TrampolineBuilder(LLVMContext &Ctx);
  void initialize(Function *F);
  void addScalarArg(Value *Arg, Type *TargetTy);
  void addPointerArg(Value *PtrArg, Type *TargetTy);
  void addSyclWrapperArg(Value *Root, Type *TargetTy);
  void addSyclRangeArg(Value *Root, Type *TargetTy, uint64_t Dim);
  void addStrideArgs(Value *Root, unsigned Dim, Type *TargetTy);
  void addSyclIDArg(Value *Root, Type *TargetTy, uint64_t Dim);
  void addSingleSyclIDArg(Value *IDRoot, Value *RangeRoot, Type *TargetTy,
                          int64_t Rank);
  void addSyclDimArg(Value *Root, Type *TargetTy);
  void setKernelName(std::string KernelName);
  std::string getKernelName() const;
  void finalize();

private:
  // Internal variables
  IRBuilder<> Builder;
  BasicBlock *TrampBlock;
  Function *F = nullptr;
  Function *TrampolineFunc = nullptr;
  SmallVector<Value *> TrampolineArgs = {};
  std::string KernelName = "";
  // Internal loop-up maps
  DenseMap<std::pair<Value *, uint64_t>, Value *> RangeMap;
  DenseMap<std::pair<Value *, const Type *>, Value *> CastMap;
  // Internal methods
  Value *getSyclArrayFromWrapper(Value *Root);
  Value *getSyclRangeDim(Value *Root, uint64_t Dim);
  Value *getSyclID(Value *Root, uint64_t Dim);
  Value *getSyclDim(Value *Root);
  Value *castArg(Value *Arg, Type *TargetTy);
};

} // namespace converter
} // namespace llvm

#endif // SYCLOPS_INCLUDE_TRAMPOLINEBUILDER_TRAMPOLINEBUILDER_H
