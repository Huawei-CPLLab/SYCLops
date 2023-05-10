//===-- Matcher.h - Converter Matching Utility Methods --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Pattern matching functions used by the converter
//
//===----------------------------------------------------------------------===//

#ifndef SYCLOPS_INCLUDE_MATCHER_H
#define SYCLOPS_INCLUDE_MATCHER_H

#include "ConverterUtil.h"
#include <cstdint>
#include <string>

namespace llvm {
class DominatorTree;
class Instruction;
class LoopInfo;
class StoreInst;
class Value;
class Type;
class Loop;
namespace converter {

void matchLoopGuard(Loop *L, BranchInst *BI, BasicBlock *&Preheader);

Type *matchVecStore(StoreInst *SI);

Value *matchExpandedURem(Instruction *I);

Value *matchOptimizedDivRemPair(Instruction *I, uint64_t &Mod, uint64_t &Div);

Value *matchLoopBound(const Loop *L, const PHINode *IV, Value *Step);

Error matchLoopComponents(const Loop *L, LoopComponents &LC);

bool matchLastLinearizedDim(Value *V, Value *&MulLHS, Value *&MulRHS,
                            Value *&Index);

bool matchDelinearizeIndex(const Value *V, SmallVector<const Value *> &Indices);

BinaryOperator *matchTruncAdd(Instruction *I);

Value *matchValueAliasPHI(const PHINode *PHI, const LoopInfo *LI,
                          const DominatorTree *DT);

bool matchIfLatchBlock(const BasicBlock *BB, const LoopInfo *LI);

bool matchIfBodyBlock(const BasicBlock *BB, const LoopInfo *LI);

bool matchAffineCondition(const Value *Cond);

} // namespace converter
} // namespace llvm

#endif // SYCLOPS_INCLUDE_MATCHER_H
