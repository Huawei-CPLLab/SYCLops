//===-- LLVMToMLIR.cpp - LLVM To MLIR ---------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Helper methods for converting LLVMIR to MLIR.
//
//===----------------------------------------------------------------------===//

#include "MLIRCodeGen/LLVMToMLIR.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Types.h"
#include "llvm/IR/InstrTypes.h"

using namespace mlir;

namespace llvm {
namespace converter {

/// Lookup table to match LLVM FCmp predicates to MLIR CmpF predicates
arith::CmpFPredicate LLVMFCmpPredicateToMLIR(CmpInst::Predicate Pred) {
  switch (Pred) {
  case CmpInst::Predicate::FCMP_FALSE:
    return arith::CmpFPredicate::AlwaysFalse;
  case CmpInst::Predicate::FCMP_OEQ:
    return arith::CmpFPredicate::OEQ;
  case CmpInst::Predicate::FCMP_OGT:
    return arith::CmpFPredicate::OGT;
  case CmpInst::Predicate::FCMP_OGE:
    return arith::CmpFPredicate::OGE;
  case CmpInst::Predicate::FCMP_OLT:
    return arith::CmpFPredicate::OLT;
  case CmpInst::Predicate::FCMP_OLE:
    return arith::CmpFPredicate::OLE;
  case CmpInst::Predicate::FCMP_ONE:
    return arith::CmpFPredicate::ONE;
  case CmpInst::Predicate::FCMP_ORD:
    return arith::CmpFPredicate::ORD;
  case CmpInst::Predicate::FCMP_UEQ:
    return arith::CmpFPredicate::UEQ;
  case CmpInst::Predicate::FCMP_UGT:
    return arith::CmpFPredicate::UGT;
  case CmpInst::Predicate::FCMP_UGE:
    return arith::CmpFPredicate::UGE;
  case CmpInst::Predicate::FCMP_ULT:
    return arith::CmpFPredicate::ULT;
  case CmpInst::Predicate::FCMP_ULE:
    return arith::CmpFPredicate::ULE;
  case CmpInst::Predicate::FCMP_UNE:
    return arith::CmpFPredicate::UNE;
  case CmpInst::Predicate::FCMP_UNO:
    return arith::CmpFPredicate::UNO;
  case CmpInst::Predicate::FCMP_TRUE:
    return arith::CmpFPredicate::AlwaysTrue;
  default:
    llvm_unreachable("Unexpected LLVM FCmp predicate.");
  }
}

/// Lookup table to match LLVM ICmp predicates to MLIR CmpI predicates
arith::CmpIPredicate LLVMICmpPredicateToMLIR(CmpInst::Predicate Pred) {
  switch (Pred) {
  case CmpInst::Predicate::ICMP_EQ:
    return arith::CmpIPredicate::eq;
  case CmpInst::Predicate::ICMP_NE:
    return arith::CmpIPredicate::ne;
  case CmpInst::Predicate::ICMP_UGT:
    return arith::CmpIPredicate::ugt;
  case CmpInst::Predicate::ICMP_UGE:
    return arith::CmpIPredicate::uge;
  case CmpInst::Predicate::ICMP_ULT:
    return arith::CmpIPredicate::ult;
  case CmpInst::Predicate::ICMP_ULE:
    return arith::CmpIPredicate::ule;
  case CmpInst::Predicate::ICMP_SGT:
    return arith::CmpIPredicate::sgt;
  case CmpInst::Predicate::ICMP_SGE:
    return arith::CmpIPredicate::sge;
  case CmpInst::Predicate::ICMP_SLT:
    return arith::CmpIPredicate::slt;
  case CmpInst::Predicate::ICMP_SLE:
    return arith::CmpIPredicate::sle;
  default:
    llvm_unreachable("Unexpected LLVM FCmp predicate.");
  }
}

/// Method to convert an LLVM Type to MLIR Type.
mlir::Type LLVMTypeToMLIRType(const llvm::Type *Ty, MLIRContext *Ctx) {
  // Integers and boolean
  if (Ty->isIntegerTy())
    return mlir::IntegerType::get(Ctx, Ty->getIntegerBitWidth());
  // The 6 Floating point types
  if (Ty->isFP128Ty())
    return mlir::FloatType::getF128(Ctx);
  if (Ty->isX86_FP80Ty())
    return mlir::FloatType::getF80(Ctx);
  if (Ty->isDoubleTy())
    return mlir::FloatType::getF64(Ctx);
  if (Ty->isFloatTy())
    return mlir::FloatType::getF32(Ctx);
  if (Ty->isHalfTy())
    return mlir::FloatType::getF16(Ctx);
  if (Ty->isBFloatTy())
    return mlir::FloatType::getBF16(Ctx);
  llvm_unreachable("Unhandled LLVM to MLIR Type conversion.");
}

} // namespace converter
} // namespace llvm
