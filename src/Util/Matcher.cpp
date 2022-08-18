//===-- Matcher.cpp - Converter Matching Utility Methods --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Keeps track of all the sycl pattern matching methods in the converter.
//
// TODO: extract/migrate the existing spaghetti pattern matching into this file
// eventually
//
//===----------------------------------------------------------------------===//

#include "Util/Matcher.h"
#include "Util/ConverterUtil.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/PatternMatch.h"

#define DEBUG_TYPE "converter-matcher"

using namespace llvm::PatternMatch;
namespace llvm {
namespace converter {

/// Recursive helper for matchLoopGuard
static bool useChainSearch(const Value *Used, const Value *Target,
                           SmallPtrSet<const Value *, 16> &Visited) {
  // ignore constant values
  if (isa<Constant>(Used))
    return false;
  if (Used == Target)
    return true;
  Visited.insert(Used);
  for (auto *Usr : Used->users()) {
    if (Visited.contains(Usr))
      continue;
    if (useChainSearch(Usr, Target, Visited))
      return true;
  }
  return false;
}

/// Verifies that the loop guard is checking for the loop bound and returns the
/// preheader of the guarded loop (loop branch of the guard conditional)
void matchLoopGuard(Loop *L, BranchInst *BI, BasicBlock *&Preheader) {
  Preheader = nullptr;
  if (L->getNumBackEdges() > 1)
    return terminate("Expecting single backedge in loop\n");
  SmallVector<BasicBlock *, 1> ExitingBlks;
  L->getExitingBlocks(ExitingBlks);
  if (ExitingBlks.size() != 1)
    return terminate("Expecting single exit from loop\n");

  BasicBlock *ExitingBlk = ExitingBlks[0];
  auto *ExitBr = dyn_cast<BranchInst>(ExitingBlk->getTerminator());
  if (!ExitBr)
    return terminate("Expecting loop exit to be a regular branch\n");
  Value *ExitCond = ExitBr->getCondition();
  // If the guard condition is not a user, then this is not a true guard against
  // trip 0 loops.
  auto *GuardCond = dyn_cast<User>(BI->getCondition());
  if (!GuardCond)
    return;

  // Trace use-def chain for operands of GuardCond (most likely an icmp), to see
  // if it ends up being used by ExitCond as well, confirming that it is indeed
  // a guard branch for this particular loop.
  SmallPtrSet<const Value *, 16> Visited; // To avoid looping use chain
  bool IsGuard = false;
  for (auto *V : GuardCond->operand_values()) {
    const Value *OriginalV = tracePastCastInsts(V);
    Visited.clear();
    if (useChainSearch(OriginalV, ExitCond, Visited)) {
      IsGuard = true;
      break;
    }
  }

  // If the compare is not actually comparing values related to the loop bound,
  // then this is not actually a loop guard branch.
  if (!IsGuard) {
    LLVM_DEBUG(
        dbgs()
        << "Found L->getLoopGuard() that is not a loop guard from rotation\n");
    return;
  }

  // If this is verified to be a guard branch, then return the preheader of the
  // loop through the preheader parameter.
  Preheader = L->getLoopPreheader();
  assert(
      Preheader &&
      (BI->getSuccessor(0) == Preheader || BI->getSuccessor(1) == Preheader) &&
      "Guard branch of a loop should lead to the preheader of the loop.");
  return;
}

const Type *matchVecStore(const StoreInst *SI) {
  const Value *StoreVal = SI->getValueOperand();
  const Value *PtrVal = SI->getPointerOperand();

  Constant *ConstVec;
  if (match(StoreVal, m_Shuffle(m_Value(), m_Constant(ConstVec)))) {
    if (ConstVec->containsUndefOrPoisonElement()) {
      // this would be the case where we are storing to a un-aligned store to a
      // vector.
      // Look past the casts
      while (auto *CI = dyn_cast<CastInst>(PtrVal))
        PtrVal = CI->getOperand(0);

      // find the original type
      Type *OriginalTy = PtrVal->getType()->getPointerElementType();
      while (OriginalTy->isStructTy())
        OriginalTy = OriginalTy->getStructElementType(0);
      if (OriginalTy->isVectorTy())
        return OriginalTy;
    }
  }
  return StoreVal->getType();
}

/// Match integer modulo expansion:
/// `n -  ( (n / A) * A)`
/// Returns the modulo RHS operand (`A` in the examplel)
Value *matchExpandedURem(Instruction *I) {
  Value *SubLHS, *DivLHS;
  Value *DivRHS, *MulRHS;
  if (match(I, m_Sub(m_Value(SubLHS),
                     m_Mul(m_UDiv(m_Value(DivLHS), m_Value(DivRHS)),
                           m_Value(MulRHS))))) {
    if (DivLHS != SubLHS)
      return nullptr;
    if (DivRHS != MulRHS)
      return nullptr;
    return DivRHS;
  }
  // could also be a multiply with negative value and add
  if (match(I, m_Add(m_Value(SubLHS),
                     m_Mul(m_UDiv(m_Value(DivLHS), m_Value(DivRHS)),
                           m_Value(MulRHS))))) {
    if (DivLHS != SubLHS)
      return nullptr;
    // be a bit conservative, and match either constantInt, or DivRHS * -1 ==
    // MulRhs
    auto *ConstDiv = dyn_cast<ConstantInt>(DivRHS);
    auto *ConstMul = dyn_cast<ConstantInt>(MulRHS);
    if (ConstDiv && ConstMul)
      if (ConstDiv->getSExtValue() * -1 == ConstMul->getSExtValue())
        return DivRHS;
    if (match(MulRHS, m_Mul(m_Specific(DivRHS), m_ConstantInt(ConstMul))))
      if (ConstMul->getSExtValue() == -1)
        return DivRHS;
  }

  return nullptr;
}

/// If a rem instruction is followed by a div by some power of 2, then it might
/// be optimized into shift and `and`s.
/// e.g.:
/// %rem = lshr %base, 5
/// %div = and %rem, 31
/// or
/// %rem = and %base, 1023
/// %div = lshr %rem, 5
/// This operation is actually performing base % 1024 (which is `(31+1)<<5`),
/// then a rem / 32; Assuming no other uses are present for %rem.
///
/// Returns the %rem value, and sets the modulo and division operands if %rem
/// has no other users, otherwise returns nullptr
Value *matchOptimizedDivRemPair(Instruction *I, uint64_t &Mod, uint64_t &Div) {
  Value *RemVal;
  ConstantInt *AndOp, *ShiftOp;
  if (match(I, m_And(m_Value(RemVal), m_ConstantInt(AndOp)))) {
    if (!match(RemVal, m_LShr(m_Value(), m_ConstantInt(ShiftOp))))
      return nullptr;
    // check to make sure no other users are using RemVal
    if (!RemVal->hasOneUser())
      return nullptr;
    uint64_t DivBy = AndOp->getZExtValue() + 1;

    // This also only works if divisor is a power of 2.
    if (!isPowerOf2_64(DivBy))
      return nullptr;

    // Now find the actual modulo operand
    Mod = DivBy << ShiftOp->getZExtValue();
    Div = DivBy;
    return RemVal;
  }

  // now match the second case
  if (!match(I, m_LShr(m_Value(RemVal), m_ConstantInt(ShiftOp))))
    return nullptr;
  if (!match(RemVal, m_And(m_Value(), m_ConstantInt(AndOp))))
    return nullptr;

  uint64_t ModBy = AndOp->getZExtValue() + 1;
  // again, this only works if modulo is power of 2
  if (!isPowerOf2_64(ModBy))
    return nullptr;
  Div = 1 << ShiftOp->getZExtValue();
  Mod = ModBy;
  return RemVal;
}

/// Matcher that tries to find the loop bound. In some edge cases this will
/// insert dead add instructions to generate the bound value from.
/// i.e. optimizations performed for `for (i=0; i<n+1; i++)` etc.
/// NOTE: Assumes the loop IV increment is always an add instruction.
Value *matchLoopBound(const Loop *L, const PHINode *IV, Value *Step) {
  // Get the LHS and RHS of the condition.
  ICmpInst *Cond = L->getLatchCmpInst();
  if (!Cond) {
    LLVM_DEBUG(dbgs() << "Invalid compare condition for loop backedge\n");
    return nullptr;
  }
  Value *CmpLHS = Cond->getOperand(0);
  Value *CmpRHS = Cond->getOperand(1);

  // helper lambda to fold constant additions to avoid generating redundant
  // instructions.
  auto FoldAdd = [&](Value *LHS, Value *RHS) -> Value * {
    auto *CL = dyn_cast<ConstantInt>(LHS);
    auto *CR = dyn_cast<ConstantInt>(RHS);
    if (CL && CR)
      return ConstantInt::get(CL->getType(), CL->getValue() + CR->getValue());
    return BinaryOperator::CreateNUWAdd(LHS, RHS, "loop.bound", Cond);
  };

  // Note: In this analysis we define the "Bound" as the value the IV gets to,
  //       but not equal to.
  //   For example:
  //                (i = 0; i < N; i++) the Bound would be N
  //               (i = 0; i <= N; i++) the Bound would be N + 1
  //           (i = N - 1; i >= 0; i--) the Bound would be -1
  //            (i = N - 1; i > 0; i--) the Bound would be 0
  // The comparison should be one of two cases:
  //   1) Pre-Step Comparison:  compare(i, BOUND)
  //        This means that the Pre-Stepped IV is compared, while the
  //        Post-Stepped is used within the loop body. This means that the IV
  //        would be equal to the BOUND in its final iteration, and thus the
  //        true Bound is BOUND + STEP.
  //   2) Post-Step Comparison: compare(IV + STEP, BOUND)
  //        This means that the Post-Stepped IV is compared and used within the
  //        loop body. This means that the IV would be equal to the BOUND - STEP
  //        in its final iteration, and thus BOUND is the true Bound.
  // Note: The LHS and RHS of the comparison may be reversed.

  Value *Bound = nullptr;
  // Pre-Step Comparison:
  if (CmpLHS == IV)
    Bound = FoldAdd(CmpRHS, Step);
  else if (CmpRHS == IV)
    Bound = FoldAdd(CmpLHS, Step);
  // Post-Step Comparison:
  else if (match(CmpLHS, m_c_Add(m_Specific(IV), m_Specific(Step))))
    Bound = CmpRHS;
  else if (match(CmpRHS, m_c_Add(m_Specific(IV), m_Specific(Step))))
    Bound = CmpLHS;
  // If it is neither of the two comparison cases:
  else
    LLVM_DEBUG(dbgs() << "Unexpected comparison for loop backedge\n");

  return Bound;
}

/// Fallback loop analysis to populate the LC being passed in.
Error matchLoopComponents(const Loop *L, LoopComponents &LC) {
  LLVM_DEBUG(
      dbgs() << "Using fallback loop analysis to generate loop components\n");
  BasicBlock *HeaderBB = L->getHeader();
  if (!HeaderBB)
    return createError("Cannot find unique header BB in loop");
  BasicBlock *PreheaderBB = L->getLoopPreheader();
  if (!PreheaderBB)
    return createError("Cannot find loop preheader");
  BasicBlock *ExitingBB = L->getExitingBlock();
  if (!ExitingBB)
    return createError("Cannot find unique exiting BB in loop");
  BranchInst *ExitingEdge = dyn_cast<BranchInst>(ExitingBB->getTerminator());
  if (!ExitingEdge)
    return createError("Cannot find unique exit in loop");
  if (!ExitingEdge->isConditional())
    return createError("Expecting exiting edge of loop to be conditional");

  // First attempt to find the exit condition and IV
  LC.Condition = dyn_cast<ICmpInst>(ExitingEdge->getCondition());
  if (!LC.Condition)
    return createError("Expecting icmp instruction for loop exit condition");

  Value *CmpLHS = LC.Condition->getOperand(0);
  Value *CmpRHS = LC.Condition->getOperand(1);

  LC.Step = nullptr;
  LC.IV = nullptr;
  for (PHINode &PHI : HeaderBB->phis()) {
    if (PHI.getType()->isIntegerTy(1) || !PHI.getType()->isIntegerTy())
      continue;
    if (CmpLHS == &PHI || CmpRHS == &PHI ||
        match(CmpLHS, m_c_Add(m_Specific(&PHI), m_Value(LC.Step))) ||
        match(CmpRHS, m_c_Add(m_Specific(&PHI), m_Value(LC.Step))))
      LC.IV = &PHI;
  }

  if (!LC.IV)
    return createError("Cannot find induction variable for loop");

  // Find the step of the IV:
  if (!LC.Step) {
    // The preprocessing step should guarantee there will only ever be one or
    // two predecessors to a block.
    BasicBlock *BackEdgeBB = LC.IV->getIncomingBlock(0);
    if (BackEdgeBB == PreheaderBB)
      BackEdgeBB = LC.IV->getIncomingBlock(1);
    Value *BackedgeVal = LC.IV->getIncomingValueForBlock(BackEdgeBB);
    if (!match(BackedgeVal, m_c_Add(m_Specific(LC.IV), m_Value(LC.Step))))
      return createError("Invalid induction variable increment");
  }

  LC.Start = LC.IV->getIncomingValueForBlock(PreheaderBB);

  LC.Bound = matchLoopBound(L, LC.IV, LC.Step);
  if (!LC.Bound)
    return createError("Cannot find loop bound");

  if (isa<ConstantInt>(LC.Bound) && isa<ConstantInt>(LC.Start) &&
      isa<ConstantInt>(LC.Step))
    LC.IsDynamic = false;
  else
    LC.IsDynamic = true;

  return Error::success();
}

/// Helper function to gather all the users into TempUsers, including if the
/// user itself is a CastInst or a casting Operator
static void gatherUsers(const Value *Root,
                        SmallVector<const User *> &UserList) {
  for (auto *Usr : Root->users()) {
    if (isa<CastInst>(Usr) || isa<ZExtOperator>(Usr))
      gatherUsers(Usr, UserList);
    else
      UserList.push_back(Usr);
  }
}

/// Given V as a linearized index, get the delinearized indices if they exist.
bool matchDelinearizeIndex(const Value *V,
                           SmallVector<const Value *> &Indices) {
  // There are two possibilities,
  // either the urem instructions are chained together:
  //    %rem0 = urem %idx, %mn  ; Index of Dim 0
  //    ...
  //    %rem1 = urem %rem0, %n  ; Use for calculation of Dim 1
  // Or the urems are all operating on the index:
  //    %rem0 = urem %idx, %mn  ; Index of Dim 0
  //    ...
  //    %rem1 = urem %idx, %n   ; Use for calculation of Dim 1
  // We need to check for both cases
  //
  SmallVector<const Value *> Rems;
  SmallVector<const User *> TempUsers;

  const Value *LastRem = V;
  bool FirstIter = true;
  // This flag will be set for the second case.
  bool GatherRem = false;
  while (1) {
    unsigned AddedRem = 0;
    const Value *Div = nullptr;
    gatherUsers(LastRem, TempUsers);
    for (const User *Usr : TempUsers) {
      auto *UsrInst = dyn_cast<Instruction>(Usr);
      assert(UsrInst && "User of linearized index not an instruction");
      unsigned OpCode = UsrInst->getOpcode();
      if (OpCode == Instruction::UDiv) {
        if (Div) {
          LLVM_DEBUG(dbgs()
                     << "TODO: Cannot distinguish between division of index "
                        "and delinearization");
          return false;
        }
        Div = UsrInst;
      } else if (OpCode == Instruction::URem) {
        Rems.push_back(UsrInst);
        AddedRem++;
      }
    }
    TempUsers.clear();

    if (GatherRem && AddedRem > 0) {
      LLVM_DEBUG(dbgs() << "TODO: Cannot distinguish between remainder "
                           "operation and delinearization of index");
      return false;
    }
    GatherRem |= FirstIter && (AddedRem > 1);

    // No more dimensions to delinearize
    if (Rems.empty()) {
      // use the last rem instruction as the final index
      Indices.push_back(LastRem);
      return true;
    }

    // add the div as the index for this dimension
    Indices.push_back(Div);

    // Now decide which Remainder to use for the next dimension
    if (GatherRem) {
      // find the remainder operation with the largest RHS and pop from list
      SmallVector<const Value *>::iterator ToErase;
      uint64_t Max = 0;
      for (auto It = Rems.begin(), End = Rems.end(); It != End; It++) {
        const Value *R = *It;
        auto *RemInst = cast<BinaryOperator>(R);
        auto *RemRHS = dyn_cast<ConstantInt>(RemInst->getOperand(1));
        if (!RemRHS) {
          llvm_unreachable(
              "TODO: Cannot delinearize dynamic shapes in this version");
          return false;
        }
        uint64_t RemBy = RemRHS->getZExtValue();
        if (Max == 0 || RemBy > Max) {
          Max = RemBy;
          LastRem = R;
          ToErase = It;
        }
      }
      Rems.erase(ToErase);
    } else {
      LastRem = Rems.pop_back_val();
      assert(Rems.empty() && "Expecting only one value in list");
    }
    FirstIter = false;
  }
}

/// Try to match the last dimension of a linearization, where:
/// V = (i * Dim) + Index
/// In this case i can be another linearized dimension, but we don't care
/// about that here. i and Dim will be returned through MulLHS and MulRHS, and
/// Index will be returned via Index. The idea is to call this function
/// through a loop to identify each of the subscripts of an array access,
/// until the last dimension, where one of the operands of Mul is either a
/// Dim, or an accessor range.
bool matchLastLinearizedDim(Value *V, Value *&MulLHS, Value *&MulRHS,
                            Value *&Index) {
  if (!V)
    return false;
  V = tracePastCastInsts(V);
  Value *LHS, *RHS;
  if (!match(V, m_Add(m_Value(LHS), m_Value(RHS))))
    return false;
  LHS = tracePastCastInsts(LHS);
  RHS = tracePastCastInsts(RHS);
  bool LHSIsMul = false, RHSIsMul = false;
  // if LHS was the mul, then RHS is the index
  if (match(LHS, m_Mul(m_Value(MulLHS), m_Value(MulRHS)))) {
    Index = RHS;
    LHSIsMul = true;
  }
  // and vice versa
  if (match(RHS, m_Mul(m_Value(MulLHS), m_Value(MulRHS)))) {
    Index = LHS;
    RHSIsMul = true;
  }
  // Only return true if only one of the add operands are multiplies,
  // otherwise we can't be sure which is the actual index TODO: Make this more
  // robust?
  if (LHSIsMul ^ RHSIsMul)
    return true;
  return false;
}

/// Recursive helper for matchTruncAdd
static BinaryOperator *findURemInUseChain(User *I) {
  auto *BI = dyn_cast<BinaryOperator>(I);
  if (!BI)
    return nullptr;
  if (BI->getOpcode() == BinaryOperator::URem)
    return BI;
  // This transformation should only be legal with add operations.
  if (BI->getOpcode() != BinaryOperator::Add)
    return nullptr;

  for (auto *Usr : BI->users()) {
    if (auto *Rem = findURemInUseChain(Usr)) {
      if (isa<ConstantInt>(Rem->getOperand(1)))
        return Rem;
    }
  }
  return nullptr;
}

/// Matcher used in Preprocessing.cpp:undoTruncAdd()
BinaryOperator *matchTruncAdd(Instruction *I) {
  // Match for x + const
  auto *AI = dyn_cast<BinaryOperator>(I);
  if (!AI)
    return nullptr;
  if (AI->getOpcode() != BinaryOperator::Add)
    return nullptr;
  auto *ConstOp = dyn_cast<ConstantInt>(AI->getOperand(1));
  if (!ConstOp)
    return nullptr;
  return findURemInUseChain(AI);
}

/// Helper recursive method to check if a LCSSA PHI chain contains a user that
/// is a store instruction
static StoreInst *checkLCSSAPhiStoreUsr(PHINode *PHI) {
  StoreInst *RetVal;
  for (User *Usr : PHI->users()) {
    RetVal = dyn_cast<StoreInst>(Usr);
    if (RetVal)
      return RetVal;
    auto *ChainPHI = dyn_cast<PHINode>(Usr);
    if (ChainPHI && ChainPHI->getNumIncomingValues() == 1)
      RetVal = checkLCSSAPhiStoreUsr(ChainPHI);
    if (RetVal)
      return RetVal;
  }
  return nullptr;
}

/// try to match the following:
///
/// store %Initial, %Alias
/// %PHI = phi [%Initial, ...], [%OP, ...]
/// %OP = add %PHI, ...
/// store %OP, %Alias
///
/// OR
///
/// loop:
///   %PHI = phi ... [%OP, %loop]
///   %OP = add %PHI, ...
/// exit:
///   %lcssa = phi [%OP, %loop]
///   store %lcssa, %Alias
///
/// For which the PHI value can be treated as the same value as the contents
/// of %Alias. Upon success, returns the %Alias pointer if not, and returns
/// nullptr on failure.
Value *matchValueAliasPHI(const PHINode *PHI, const LoopInfo *LI,
                          const DominatorTree *DT) {
  Value *InitialVal = nullptr;
  Value *ReuseValue = nullptr;
  size_t NumIncoming = PHI->getNumIncomingValues();

  // Currently only looking for PHIs with two incoming values
  if (NumIncoming != 2)
    return nullptr;

  // Requires the PHI node to be inside a loop
  Loop *L = LI->getLoopFor(PHI->getParent());
  if (!L)
    return nullptr;

  // Find the initial value and the reuse value with respective to the loop
  for (size_t Idx = 0; Idx < NumIncoming; Idx++) {
    Value *IncomingVal = PHI->getIncomingValue(Idx);
    BasicBlock *IncomingBlk = PHI->getIncomingBlock(Idx);
    if (L->contains(IncomingBlk))
      ReuseValue = IncomingVal;
    else
      InitialVal = IncomingVal;
  }

  // Require values within and outside the loop
  if (!(InitialVal && ReuseValue))
    return nullptr;

  auto TraceLCSSA = [&](Value *V) {
    while (auto *LCSSAPhi = dyn_cast<PHINode>(V)) {
      if (LCSSAPhi->getNumIncomingValues() != 1)
        break;
      V = LCSSAPhi->getIncomingValue(0);
    }
    return V;
  };
  // Trace reuse values down LCSSA chain if exists
  ReuseValue = TraceLCSSA(ReuseValue);
  LLVM_DEBUG(dbgs() << "Found potential reuse value "; ReuseValue->dump());

  // Check if Reuse actually reuses the PHI value
  auto *ReuseUsr = dyn_cast<User>(ReuseValue);
  if (!ReuseUsr)
    return nullptr;
  bool FoundUse = false;
  for (auto *Val : ReuseUsr->operand_values()) {
    if (Val == PHI)
      FoundUse = true;
    // also check if a user phi node is lcssa chain with this phi inside it
    else if (auto *PHIChain = dyn_cast<PHINode>(Val)) {
      LLVM_DEBUG(
          dbgs()
          << "\tPotential reuse has PHI operand, checking for PHI chain\n");
      while (PHIChain) {
        if (PHIChain == PHI) {
          FoundUse = true;
          break;
        }

        unsigned NumIncoming = PHIChain->getNumIncomingValues();
        // only handle two-value phi's
        if (NumIncoming > 2)
          break;

        PHINode *NewPHI = nullptr;
        for (unsigned Idx = 0; Idx < NumIncoming; Idx++) {
          Value *IncomingVal = PHIChain->getIncomingValue(Idx);
          if (TraceLCSSA(IncomingVal) == ReuseValue)
            continue;
          NewPHI = dyn_cast<PHINode>(IncomingVal);
        }
        PHIChain = NewPHI;
      } // end while PHIChain
    }   // end if PHIChain
    if (FoundUse)
      break;
  }

  if (!FoundUse)
    return nullptr;
  LLVM_DEBUG(dbgs() << "\tFound valid reuse Values\n");

  StoreInst *ReuseStore = nullptr;
  // For the reuse, check if it is stored.
  for (auto *Usr : ReuseValue->users()) {
    if (Usr == PHI)
      continue;
    if (auto *ChainPHI = dyn_cast<PHINode>(Usr)) {
      if (ChainPHI->getNumIncomingValues() == 1) {
        ReuseStore = checkLCSSAPhiStoreUsr(ChainPHI);
      }
      // TODO: Maybe also do a recursive call here to check for other phi
      // nodes?
    } else {
      ReuseStore = dyn_cast<StoreInst>(Usr);
    }
    if (ReuseStore)
      break;
  }

  if (!ReuseStore)
    return nullptr;

  if (!DT->dominates(ReuseValue, ReuseStore))
    return nullptr;

  Value *AliasPtr = ReuseStore->getPointerOperand();

  LLVM_DEBUG(dbgs() << "\tFound pointer alias for PHI: "; AliasPtr->dump());
  return AliasPtr;
}

// Matcher for checking if a BasicBlock is the latching block for an if.
bool matchIfLatchBlock(const BasicBlock *BB, const LoopInfo *LI) {
  // The terminator of the predecessor cannot be a return and must be a
  // conditional branch.
  const Instruction *Terminator = BB->getTerminator();
  if (isa<ReturnInst>(Terminator))
    return false;
  auto *Branch = dyn_cast<BranchInst>(Terminator);
  if (!Branch)
    return false;
  if (Branch->isUnconditional())
    return false;
  // The If Latch should not be a loop latch
  Loop *L = LI->getLoopFor(BB);
  if (L && L->getLoopLatch() == BB)
    return false;
  // If all else holds true, then this BB is either a true or false block for
  // an If.
  return true;
}

/// Matcher for checking if the given BasicBlock is either a true or false
/// block of an If.
bool matchIfBodyBlock(const BasicBlock *BB, const LoopInfo *LI) {
  // For now, assume that an if body block will have a single predecessor.
  const BasicBlock *Pred = BB->getSinglePredecessor();
  if (!Pred)
    return false;
  // Check if the predecessor is a latch for an if.
  return matchIfLatchBlock(Pred, LI);
}

// an Affine Condition must be a conjunction of integer conditions. It cannot
// contain any OR operations and any float inputs. The Affine dialect also
// does not handle NEQ at the current moment, however this may change in the
// future.
bool matchAffineCondition(const Value *Cond) {
  assert(Cond->getType()->isIntegerTy(1) &&
         "matchAffineCondition expected to be run on boolean value.");
  // Check if this is a Binary Operator
  if (auto *BO = dyn_cast<llvm::BinaryOperator>(Cond)) {
    if (BO->getOpcode() == llvm::Instruction::BinaryOps::And) {
      // An AND is only affine if the LHS && RHS are affine
      return matchAffineCondition(BO->getOperand(0)) &&
             matchAffineCondition(BO->getOperand(1));
    }
    // Any other binary operator is not allowed
    return false;
  }
  // Check if this is a CmpInst
  if (auto *CMP = dyn_cast<llvm::CmpInst>(Cond)) {
    // If we run into a NE cmp, return false.
    if (CMP->getPredicate() == CmpInst::Predicate::ICMP_NE)
      return false;

    // Make sure both of the operands of the compare can be represented as
    // affine expressions.
    return CMP->getOperand(0)->getType()->isIntegerTy() &&
           CMP->getOperand(1)->getType()->isIntegerTy();
  }
  // If the condition is not a binary operator or a compare instruction, it
  // cannot be handled as an affine condition.
  return false;
}

bool DemangleRange(StringRef &FuncName, SmallVector<unsigned> &RangeVec) {
  bool Success = true;
  while (FuncName.consume_front("L")) {
    int RangeVal;
    // range values are unsigned longs
    Success &= FuncName.consume_front("m");
    Success &= !FuncName.consumeInteger(10, RangeVal);
    Success &= FuncName.consume_front("E");
    assert(Success && "Unexpected parallel_for template, check implementation");
    RangeVec.push_back(RangeVal);
  }
  return Success;
}

/// Parse the usage of the global id, extract the values that correspond to the
/// dimensions of the abstracted for loops.
static void
parseGlobalIdUsage(const SmallVector<const Instruction *> &GIDUsrs,
                   DenseMap<unsigned, const Value *> &IdxToGlobalValueMap) {
  bool MatchSuccess = false;
  if (GIDUsrs.size() == 1) {
    LLVM_DEBUG(
        dbgs() << "Attempting to match Load->ExtractElement(GlobalId)\n");
    const Instruction *GlobalIdUsr = GIDUsrs[0];
    if (!isa<LoadInst>(GlobalIdUsr))
      llvm_unreachable("Unexpected usage of global id");
    for (const User *Usr : GlobalIdUsr->users()) {
      auto *EEI = dyn_cast<ExtractElementInst>(Usr);
      if (!EEI)
        break;
      auto *EERHS = dyn_cast<ConstantInt>(EEI->getOperand(1));
      assert(EERHS && "Unexpected usage of global id");
      unsigned Idx = EERHS->getZExtValue();
      IdxToGlobalValueMap[Idx] = Usr;
      MatchSuccess = true;
    }
  }
  if (!MatchSuccess) {
    LLVM_DEBUG(dbgs() << "Attempting to match Load(GEP(GlobalId, 0, Idx))\n");
    // Match load(gep(GID, 0, id)) pattern
    for (const Instruction *GlobalIdUsr : GIDUsrs) {
      if (auto *LI = dyn_cast<LoadInst>(GlobalIdUsr)) {
        const Value *PossibleGEP = LI->getPointerOperand();
        auto *GEPOp = dyn_cast<GEPOperator>(PossibleGEP);
        assert(GEPOp && "Unexpected usage of global id");
        assert(GEPOp->getNumIndices() == 2 &&
               "Expecting only two indices to this GEP operation");
        // get the index corresponding to the GID extractelement value.
        auto *ConstIdx = dyn_cast<ConstantInt>(GEPOp->getOperand(2));
        assert(ConstIdx && "Expecting constant operand in GEPOperator");
        unsigned Idx = ConstIdx->getZExtValue();
        IdxToGlobalValueMap[Idx] = LI;
      }
    }
  }
}

} // namespace converter
} // namespace llvm
