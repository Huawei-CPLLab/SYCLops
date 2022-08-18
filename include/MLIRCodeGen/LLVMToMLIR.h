//===-- LLVMToMLIR.h - LLVM To MLIR Translations --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef SYCLOPS_INCLUDE_MLIRCODEGEN_LLVMTOMLIR_H
#define SYCLOPS_INCLUDE_MLIRCODEGEN_LLVMTOMLIR_H

#include "llvm/IR/InstrTypes.h"

namespace mlir {
class Type;
class MLIRContext;
namespace arith {
enum class CmpFPredicate : uint64_t;
enum class CmpIPredicate : uint64_t;
} // namespace arith
} // namespace mlir

namespace llvm {
class Type;
namespace converter {

mlir::arith::CmpFPredicate LLVMFCmpPredicateToMLIR(CmpInst::Predicate Pred);

mlir::arith::CmpIPredicate LLVMICmpPredicateToMLIR(CmpInst::Predicate Pred);

mlir::Type LLVMTypeToMLIRType(const llvm::Type *Ty, mlir::MLIRContext *Ctx);

} // namespace converter
} // namespace llvm

#endif // SYCLOPS_INCLUDE_MLIRCODEGEN_LLVMTOMLIR_H
