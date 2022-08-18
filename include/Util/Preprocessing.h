//===-- Preprocessing.h - Converter Preprocessing Methods -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef SYCLOPS_INCLUDE_UTIL_PREPROCESSING_H
#define SYCLOPS_INCLUDE_UTIL_PREPROCESSING_H

#include "ConverterUtil.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/ValueMap.h"

namespace llvm {
namespace converter {

class Preprocessor {
  Function *F;
  IRBuilder<> *Builder;
  FunctionAnalysisManager *FAM;

  void mergeLocalContext();
  void eliminateRedundantIntToPtr();
  void eliminateAllocas();
  void undoURemStrReduction();
  void undoDivRemStrReduction();
  void removeRedundantInstrs();
  void removeFreezeInsts();
  void mergeGEPs();
  void undoMulDivStrReduction();
  void undoTruncAdd();
  void removeRedundantExts();
  void simplifySelectLogic();
  void removeHalfToI16();
  void removePtrSelects();
  void undoCombinedCmp();
  void removeLoopGuard();
  void ifSimplify();

public:
  Preprocessor(Function *F, IRBuilder<> *Builder, FunctionAnalysisManager *FAM);

  void run();
  unsigned parseAccessorArguments(ValueMap<Value *, Shape> &ShapeMap);
};

} // namespace converter
} // namespace llvm

#endif // SYCLOPS_INCLUDE_UTIL_PREPROCESSING_H
