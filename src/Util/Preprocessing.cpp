//===-- Preprocessing.cpp - Converter Preprocessing Methods -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Util/Preprocessing.h"
#include "Util/ConverterUtil.h"
#include "Util/Matcher.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/PassManager.h"
#include "llvm/IR/PatternMatch.h"
#include "llvm/Transforms/Scalar/DCE.h"
#include "llvm/Transforms/Scalar/SimplifyCFG.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/LCSSA.h"
#include "llvm/Transforms/Utils/LoopSimplify.h"
#include "llvm/Transforms/Utils/LoopUtils.h"
#include <queue>

#define DEBUG_TYPE "converter"

namespace llvm {
using namespace PatternMatch;

namespace converter {

namespace {
/// Helper for promoting pass-by-value context structures, such that references
/// to the local context are converted to the function arguments instead. This
/// should be safe, as the input parameters should only ever be pointer values
/// that are never modified anyway.
struct ByvalStructPromoter {
  Function *F;
  AllocaInst *Ctx;
  SmallVector<uint64_t> SrcGEPIdx;
  SmallVector<uint64_t> DestGEPIdx;
  IRBuilder<> *Builder;
  SmallVector<MemCpyInst *> Memcpys;
  SmallVector<std::pair<LoadInst *, StoreInst *>> Copies;
  DenseSet<GetElementPtrInst *> ToReplace;

  /// Constructor of the struct finds the alloca instruction that corresponds to
  /// the local context structure.
  ByvalStructPromoter(Function *F, IRBuilder<> &Builder)
      : F(F), Builder(&Builder) {
    Ctx = nullptr;
    StructType *AllocaTy;
    // Find the local context alloca:
    for (auto &BB : *F) {
      for (auto &I : BB) {
        // First filter through the allocas.
        auto *Alloca = dyn_cast<AllocaInst>(&I);
        if (!Alloca)
          continue;
        // Then check to see if its allocating a struct
        AllocaTy = dyn_cast<StructType>(Alloca->getType()->getElementType());
        if (!AllocaTy)
          continue;
        // Then check to see if its allocating simple structs that isn't the
        // context object
        const Type *UnwrappedTy = AllocaTy;
        unwrapStructs(UnwrappedTy);
        if (!UnwrappedTy->isStructTy())
          continue;
        // At this point the only thing we have should be the context struct,
        // and we should ever only have one of these.
        assert(!Ctx &&
               "More than one complex structure types locally allocated");
        Ctx = Alloca;
      }
    }
  }

  /// Trace past one level of casting (If exists) in the use-def chain to find
  /// the copying load-from-argument and store-to-context
  StoreInst *traceLoadUsrs(LoadInst *Ld) {
    StoreInst *RetVal = nullptr;
    auto IsStoreToCtx = [&](User *U) {
      if (auto *SI = dyn_cast<StoreInst>(U)) {
        if (tracePastCastAndGEP(SI->getPointerOperand()) == Ctx) {
          assert(!RetVal && "Multiple stores to the same context location");
          RetVal = SI;
        }
      } // end if store inst
    };

    for (auto *Usr : Ld->users()) {
      if (auto *CI = dyn_cast<CastInst>(Usr)) {
        for (auto *CIUsr : CI->users()) {
          IsStoreToCtx(CIUsr);
        }
      } else
        IsStoreToCtx(Usr);
    } // end iterating through load users
    return RetVal;
  }

  /// Finds the memory instructions that are copying from the Argument into the
  /// local context structure. Also finds the GEP instructions that uses the
  /// context struct as a base, so that we may replace them with GEPs to
  /// arguments.
  void populateWorklist() {
    SmallVector<StoreInst *> ToErase;
    // Gather all the load/memcpys from the function which have function
    // arguments as source address. Find the local context alloca:
    for (auto &BB : *F) {
      for (auto &I : BB) {
        if (auto *Ld = dyn_cast<LoadInst>(&I)) {
          if (!isa<Argument>(tracePastCastAndGEP(Ld->getPointerOperand())))
            continue;
          // Check if this load is being used in a store to a context structure
          StoreInst *CopyStore = traceLoadUsrs(Ld);
          if (CopyStore)
            Copies.push_back({Ld, CopyStore});
        } else if (auto *MemCpy = dyn_cast<MemCpyInst>(&I)) {
          if (isa<Argument>(tracePastCastAndGEP(MemCpy->getRawSource())) &&
              tracePastCastAndGEP(MemCpy->getRawDest()) == Ctx)
            Memcpys.push_back(MemCpy);
        } else if (auto *ConstSt = dyn_cast<StoreInst>(&I)) {
          // Check for local variables initialized to a constant as well:
          auto *ConstVal = dyn_cast<Constant>(ConstSt->getValueOperand());
          if (!ConstVal)
            continue;
          auto *CtxGEP =
              dyn_cast<GetElementPtrInst>(ConstSt->getPointerOperand());
          if (!CtxGEP || CtxGEP->getPointerOperand() != Ctx)
            continue;
          // Check legality of our transformation
          for (Use &U : CtxGEP->indices()) {
            if (!isa<Constant>(U.get()))
              report_fatal_error("Unsupported: initialization of local array "
                                 "in device kernel");
          }

          // First check if there are any uses of this GEP, if there are none
          // apart from the constant store, then simply erase the store.
          if (CtxGEP->hasOneUse()) {
            ToErase.push_back(ConstSt);
            continue;
          }
          // Create a new local variable to store
          Builder->SetInsertPoint(ConstSt);
          Type *ValType = ConstVal->getType();
          auto *NewLocal = Builder->CreateAlloca(ValType);
          Builder->CreateStore(ConstVal, NewLocal);
          auto *LocalLd = Builder->CreateLoad(ValType, NewLocal);
          ConstSt->setOperand(0, LocalLd);
          Copies.push_back({LocalLd, ConstSt});
        } else if (auto *GEP = dyn_cast<GetElementPtrInst>(&I)) {
          if (tracePastCastInsts(GEP->getPointerOperand()) == Ctx) {
            ToReplace.insert(GEP);
          }
        }
      }
    }
    for (auto *SI : ToErase)
      SI->eraseFromParent();
  }

  /// For some reason GetElementPtrInst::getIndexedType() isn't working as I
  /// thought, so here it is...
  Type *getIndexedType(Value *Val, ArrayRef<uint64_t> Indices) {
    Type *Ty = Val->getType();
    assert(Ty->isPointerTy());
    for (uint64_t Idx : Indices) {
      if (auto *StructTy = dyn_cast<StructType>(Ty))
        Ty = StructTy->getElementType(Idx);
      else if (auto *ArrTy = dyn_cast<ArrayType>(Ty))
        Ty = ArrTy->getElementType();
      else if (auto *VecTy = dyn_cast<VectorType>(Ty))
        Ty = VecTy->getElementType();
      else if (auto *PtrTy = dyn_cast<PointerType>(Ty))
        Ty = PtrTy->getElementType();
      else
        return nullptr;
    }
    return Ty;
  }

  /// Special check for wrapper types.
  bool isWrappedType(Type *&MaybeWrapper, Type *Ty) {
    if (auto *StructTy = dyn_cast<StructType>(MaybeWrapper)) {
      assert(StructTy->getNumElements() == 1 &&
             "This method should only ever encounter basic sycl types");
      if (StructTy->getElementType(0) == Ty) {
        MaybeWrapper = Ty;
        return true;
      }
    }
    return false;
  }

  /// Given the base pointer, populates the DestGEPIdx such that it can index to
  /// a ValType. This will be used to match which GEP to replace with which
  /// argument.
  void getCtxGEPIdxFromPtr(Value *DestPtr, Type *ValType) {
    auto *DestGEP = dyn_cast<GetElementPtrInst>(tracePastCastInsts(DestPtr));
    DestGEPIdx.clear();

    // Populate DestGEPIdx
    Value *DestBase;
    if (!DestGEP) {
      DestGEPIdx = {0, 0};
      DestBase = Ctx;
    } else {
      DestBase = DestGEP->getPointerOperand();
      for (auto &U : DestGEP->indices()) {
        Type *IndexedTy = getIndexedType(DestBase, DestGEPIdx);
        if (IndexedTy == ValType)
          break;
        if (isWrappedType(ValType, IndexedTy)) {
          SrcGEPIdx.push_back(0);
          break;
        }
        auto *ConstIdx = dyn_cast<ConstantInt>(U.get());
        assert(ConstIdx && "Local context non-constant indexing");
        DestGEPIdx.push_back(ConstIdx->getZExtValue());
      }
    }

    assert(getIndexedType(DestBase, DestGEPIdx) == ValType &&
           "Failed to find matching type from Ctx GEP");
  }

  /// Finds GEPs from the ToReplace worklist and match the leading indices with
  /// DestGEPIdx and slice the indices from the GEP. Then populates NewIdx with
  /// concat(SrcGEPIdx, GEP->RemainingIndices)
  bool matchAndSwapGEPIdx(GetElementPtrInst *GEP,
                          SmallVector<Value *> &NewIdx) {
    auto *IdxIt = GEP->idx_begin();
    for (uint64_t Idx : DestGEPIdx) {
      auto *ConstIdx = dyn_cast<ConstantInt>(IdxIt->get());
      if (!ConstIdx || ConstIdx->getZExtValue() != Idx)
        return false;
      IdxIt++;
    }
    // Construct the new GEP indices
    NewIdx.clear();
    for (uint64_t Idx : SrcGEPIdx)
      NewIdx.push_back(Builder->getInt32(Idx));
    while (IdxIt != GEP->idx_end()) {
      NewIdx.push_back(IdxIt->get());
      IdxIt++;
    }
    return true;
  }

  /// Replace the GEPs in the ToReplace worklist with new GEPs with NewBase as
  /// the base, and updates the indices accordingly.
  void replaceGEPBase(Value *NewBase) {
    SmallVector<GetElementPtrInst *> ToErase;
    SmallVector<Value *> NewIndices;
    for (auto *GEP : ToReplace) {
      if (!matchAndSwapGEPIdx(GEP, NewIndices))
        continue;
      // If the NewIndices is empty, then it means that this GEP is returning a
      // double pointer, which is the pointer being used to store the argument.
      // In this case this would become dead once the store to the local context
      // is erased, and should be removed by DCE.
      if (NewIndices.empty())
        continue;
      ToErase.push_back(GEP);
      Builder->SetInsertPoint(GEP);
      unsigned AS = GEP->getAddressSpace();
      Type *SrcTy = NewBase->getType();
      Type *ElTy = SrcTy->getPointerElementType();
      if (SrcTy->getPointerAddressSpace() != AS) {
        SrcTy = PointerType::get(ElTy, AS);
        NewBase = Builder->CreateAddrSpaceCast(NewBase, SrcTy);
      }
      auto *ReplWith = Builder->CreateGEP(ElTy, NewBase, NewIndices);
      GEP->replaceAllUsesWith(ReplWith);
    }
    for (auto *Val : ToErase)
      ToReplace.erase(Val);
  }

  /// Populate both the Src and Dest GEPIdx based on the ReplWith and BaseToRepl
  /// pointer access patterns.
  Value *populateGEPIdx(Value *ReplWith, Value *BaseToRepl) {
    auto *SrcGEP = dyn_cast<GetElementPtrInst>(tracePastCastInsts(ReplWith));
    Value *SrcPtr = tracePastCastAndGEP(ReplWith);
    SrcGEPIdx.clear();
    // Populate SrcGEPIdx
    if (!SrcGEP) {
      SrcGEPIdx.push_back(0);
    } else {
      for (auto &U : SrcGEP->indices()) {
        auto *ConstIdx = dyn_cast<ConstantInt>(U.get());
        if (!ConstIdx)
          break;
        SrcGEPIdx.push_back(ConstIdx->getZExtValue());
      }
      SrcPtr = SrcGEP->getPointerOperand();
    }

    Type *SrcTy = getIndexedType(SrcPtr, SrcGEPIdx);

    // And this will populate the DestGEPIdx.
    getCtxGEPIdxFromPtr(BaseToRepl, SrcTy);

    return SrcPtr;
  }

  void rewrite() {
    if (!Ctx)
      return;

    LLVM_DEBUG(dbgs() << "\tFound local context allocation: "; Ctx->dump());

    populateWorklist();
    // For every copy into the local context, keep track of the GEP indexes used
    // for both the source and dest, then replace all GEPs to the context with
    // the dest indices with a GEP to the source with the src indices.
    for (auto *MC : Memcpys) {
      Value *NewBase = populateGEPIdx(MC->getRawSource(), MC->getRawDest());
      replaceGEPBase(NewBase);
      MC->eraseFromParent();
    }

    for (auto Pair : Copies) {
      LoadInst *Ld = Pair.first;
      StoreInst *St = Pair.second;
      Value *NewBase;
      Value *DestPtr = St->getPointerOperand();
      // If the Stored value is a pointer, then simply use this pointer as the
      // base of the new GEPs
      Value *StoreVal = St->getValueOperand();
      if (auto *PtrTy = dyn_cast<PointerType>(StoreVal->getType())) {
        NewBase = StoreVal;
        // In this case, simply truncate the leading indices matching DestGEPIdx
        // in GEPs (ToReplace) and replace the base pointer.
        SrcGEPIdx.clear();
        getCtxGEPIdxFromPtr(DestPtr, PtrTy);
      } else
        NewBase =
            populateGEPIdx(Ld->getPointerOperand(), St->getPointerOperand());
      replaceGEPBase(NewBase);
      St->eraseFromParent();
    }

    // There also could be instances where the argument is stored directly into
    // the context struct. In this case, erase the redundant store and replace
    // the GEP result with the argument directly.
    for (auto *GEP : ToReplace) {
      if (auto *Usr = GEP->getUniqueUndroppableUser()) {
        auto *St = dyn_cast<StoreInst>(Usr);
        assert(St && "Unhandled ByVal access");
        if (isa<Argument>(St->getValueOperand())) {
          St->eraseFromParent();
          continue;
        }
        LLVM_DEBUG(dbgs() << "[ERROR] Remaining context store instruction: ";
                   St->dump());
      }
      // At this point, all the GEPs remaining should only be GEPs to double
      // pointers which are dead due to the store that use them being erased.
      assert(GEP->getNumUses() == 0 &&
             "Live GEPs from ByVal context still exists!");
    }
  }
}; // end ByvalStructPromoter

/// Struct for encapsulating information needed for generating dedicated `if`
/// exits.
struct IfBlocks {
  BasicBlock *ExitBlock;
  BasicBlock *BranchExit1;
  BasicBlock *BranchExit2;

  IfBlocks() : ExitBlock(nullptr), BranchExit1(nullptr), BranchExit2(nullptr) {}

  /// Returns true if both exit branches are added
  bool addExit(BasicBlock *Exit) {
    assert(!BranchExit2 &&
           "Trying to add more than two branches to a single if condition");
    if (!BranchExit1) {
      BranchExit1 = Exit;
      return false;
    }
    BranchExit2 = Exit;
    return true;
  }
};

/// Helper for simplifying if structures so that analysis and conversion will be
/// easier.
class IfElseSimplifier {
  using DTNode = DomTreeNodeBase<BasicBlock>;
  unsigned Counter;
  Function *F;
  FunctionAnalysisManager *FAM;
  IRBuilder<> *Builder;
  LoopInfo *LI;
  DominatorTree *DT;
  SmallVector<BasicBlock *> ExitBlocks;
  MapVector<BasicBlock *, IfBlocks> Worklist;

  /// Checks if an if structure is simplified or not (i.e. have a dedicated exit
  /// block common to only the branches of the if condition).
  bool isSimplified(BasicBlock *IfHeader) {
    DTNode *Node = DT->getNode(IfHeader);
    BasicBlock *Child1 = nullptr;
    BasicBlock *Child2 = nullptr;

    // Helper to assign to the two children block variables
    auto AssignChild = [&](BasicBlock *C) {
      if (!Child1)
        Child1 = C;
      else if (!Child2)
        Child2 = C;
      else
        terminate("Unsupported: switch statements");
    };

    // Helper to check if C2 has a predecessor that is dominated by C1
    auto DominatesPred = [DomTree = DT](BasicBlock *C1,
                                        BasicBlock *C2) -> bool {
      for (auto *Pred : predecessors(C2)) {
        if (DomTree->dominates(C1, Pred))
          return true;
      }
      return false;
    };

    switch (Node->getNumChildren()) {
    case 2:
      // If the if header is simplified and has two children, then one of them
      // should be the landing pad of the if/else, meaning that it has two
      // predecessors, one being the IfHeader itself, and the other being
      // dominated by the other child of IfHeader.
      // ============  ===========
      // | IfHeader |->| Landing |
      // ============  ===========
      //       v            ^
      // ============       |
      // | ........ |-------|
      // ============
      for (auto *Succ : successors(IfHeader)) {
        if (Succ->hasNPredecessorsOrMore(3))
          return false;
        AssignChild(Succ);
      }
      return (DominatesPred(Child1, Child2) || DominatesPred(Child2, Child1));
    case 3: {
      BasicBlock *ExitBlock = nullptr;
      // Similarily to the previous case, but Landing now should have two
      // predecessors that are dominated by each of the successor of IfHeader.
      for (DTNode *C : Node->children()) {
        // The immediate successors of IfHeader should only have a single
        // predecessor.
        auto *ChildBB = C->getBlock();
        if (ChildBB->getSinglePredecessor() == IfHeader)
          AssignChild(ChildBB);
        else if (!ExitBlock)
          ExitBlock = ChildBB;
        else
          return false;
      }
      // If child2 is not populated, then this is not the case we expected.
      if (!Child2)
        return false;
      return (DominatesPred(Child1, ExitBlock) &&
              DominatesPred(Child2, ExitBlock));
    }
    default:
      return false;
    }
  }

  /// Find all the if headers (BBs with conditional branches that are not
  /// loop exiting block) in post order such that in case of nested if
  /// statements, the inner if's will come before the outer one.
  /// Also find all if/else exit blocks (BB's that have more than 2
  /// predecessors) in preorder, so if there are multiple that dominate each
  /// other, the dominating one will come first.
  /// Returns false if worklist is empty, true otherwise.
  bool findBlocks(DTNode *Node) {
    // Verify that the dominator tree is actually in a state we can handle
    if (Node->getNumChildren() > 3) {
      terminate(
          "Dominator tree node contains more than 3 children - Control flow "
          "contains unexpected jumps");
    }
    BasicBlock *BB = Node->getBlock();
    // Pre order ExitBlocks
    if (BB->hasNPredecessorsOrMore(3))
      ExitBlocks.push_back(BB);

    bool FoundIfBlock = false;
    // Recursive call
    for (auto *ChildNode : Node->children())
      FoundIfBlock |= findBlocks(ChildNode);

    // Post order search for If structure headers:
    // Block is only an IfHeader if it has a condional branch as terminator
    auto *BI = dyn_cast<BranchInst>(BB->getTerminator());
    if (!BI || !BI->isConditional())
      return FoundIfBlock;
    // Loop exiting blocks are also not considered if structures...
    if (Loop *L = LI->getLoopFor(BB)) {
      if (L->getExitingBlock() == BB)
        return FoundIfBlock;
    }
    // If block is already simplified, ignore for this iteration.
    if (isSimplified(BB))
      return FoundIfBlock;
    LLVM_DEBUG(dbgs() << ">> Attempting to simplify `if` structure "
                      << BB->getName() << "\n");
    Worklist[BB] = IfBlocks();
    return true;
  }

  /// Wrapper around findBlocks.
  bool populateWorklist() {
    LI = &FAM->getResult<LoopAnalysis>(*F);
    DT = &FAM->getResult<DominatorTreeAnalysis>(*F);
    Worklist.clear();
    ExitBlocks.clear();
    return findBlocks(DT->getRootNode());
  }

  /// Find the closet if structure that dominates this block
  IfBlocks &findNearest(BasicBlock *BB) {
    for (auto &Pair : Worklist) {
      BasicBlock *IfHeader = Pair.first;
      if (DT->dominates(IfHeader, BB)) {
        return Pair.second;
      }
    }
    llvm_unreachable("Cannot find a If Header for a predecessor of a block "
                     "with more than 2 predecessors");
  }

  /// Identify the edges going to the exit block of a if structure. Populates
  /// worklist from edges that are going straight from IfHeaders to the exit
  /// block, and populates IfExitEdges from unconditional branches.
  void findIfExitEdges() {
    for (auto *BB : ExitBlocks) {
      // Now find all predecessors of the block and add them to the worklist
      // structure
      for (auto *Pred : predecessors(BB)) {
        auto *BI = dyn_cast<BranchInst>(Pred->getTerminator());
        if (!BI)
          terminate("Cannot handle switch or goto statements");
        if (BI->isConditional()) {
          assert(Worklist.count(Pred) &&
                 "Conditional predecessor of a block with more than 2 "
                 "precessors is not an IfHeader");
          Worklist[Pred].ExitBlock = BB;
          continue;
        }
        auto &IB = findNearest(Pred);
        // If this is an if/else block (if header is not an immediate
        // predecessor to the landing pad), then add the landing pad to the
        // IfBlocks.
        if (IB.addExit(Pred)) {
          IB.ExitBlock = BB;
        }
      } // End iterating through BB predecessors
    }   // End iterating through all BB's in Function
  }

  /// Create PHINodes in NewExit based on PHIs in values from Blk.ExitBlock
  void updatePHIs(BasicBlock *BranchExit, BasicBlock *Pred, BasicBlock *OldExit,
                  BasicBlock *NewExit) {
    for (auto &PHI : OldExit->phis()) {
      // Construct new PHI node.
      auto *NewPHI = Builder->CreatePHI(PHI.getType(), 2, "split.phi");
      Value *Val = PHI.getIncomingValueForBlock(BranchExit);
      NewPHI->addIncoming(Val, BranchExit);
      Val = PHI.getIncomingValueForBlock(Pred);
      NewPHI->addIncoming(Val, Pred);

      // Update old PHI node.
      PHI.removeIncomingValue(BranchExit);
      PHI.removeIncomingValue(Pred, /*DeleteIfEmpty*/ false);
      if (PHI.getNumIncomingValues() == 0)
        PHI.replaceAllUsesWith(NewPHI);
      else
        PHI.addIncoming(NewPHI, NewExit);
    }
  }

  /// For every item in the worklist, if it is valid (i.e. both BranchExit1 and
  /// ExitBlock is valid), then create a new BB and insert in between the
  /// exiting blocks and the old exit block.
  bool insertExitBlocks() {
    bool RetVal = false;
    for (auto WorkItem : Worklist) {
      // first check if the required blocks are found for this if structure.
      auto &Blocks = WorkItem.second;
      // Since the exit block would only be inserted iff an edge from the if
      // header to the exit block was found, OR both Branchexits have been
      // found, we just need to check for Exit1 and the ExitBlock itself.
      if (!(Blocks.ExitBlock && Blocks.BranchExit1))
        continue;

      RetVal = true;
      BasicBlock *IfHeader = WorkItem.first;
      BasicBlock *SecondPred =
          Blocks.BranchExit2 ? Blocks.BranchExit2 : IfHeader;
      BasicBlock *OldExit = Blocks.ExitBlock;
      // Create the new block to insert in between the branch exits and the exit
      // block.
      auto *NewExit = BasicBlock::Create(F->getContext(), "if.exit", F);
      LLVM_DEBUG(dbgs() << ">> Creating new exit block " << NewExit->getName()
                        << " for " << IfHeader->getName() << "\n");
      Builder->SetInsertPoint(NewExit);
      updatePHIs(Blocks.BranchExit1, SecondPred, OldExit, NewExit);
      Builder->CreateBr(Blocks.ExitBlock);
      // Redirect the edges to ExitBlock to the new Exit.
      Blocks.BranchExit1->getTerminator()->replaceUsesOfWith(OldExit, NewExit);
      SecondPred->getTerminator()->replaceUsesOfWith(OldExit, NewExit);
    }
    return RetVal;
  }

  void transform() {
    Counter++;
    LLVM_DEBUG(dbgs() << "Running iteration " << Counter
                      << " of if structure simplification\n");
    // Set up the IfBlocks struct
    findIfExitEdges();
    // Insert exit blocks for every valid worklist item.
    if (!insertExitBlocks()) {
      LLVM_DEBUG(F->dump());
      terminate("Could not identify `if` structure, try simplifying if/else "
                "statements");
    }
    // NOTE: Probably a better idea to use dominator tree updater than to
    // invalide everything, but we'll do that later...
    FAM->invalidate(*F, PreservedAnalyses::none());
  }

public:
  IfElseSimplifier(Function *F, FunctionAnalysisManager *FAM,
                   IRBuilder<> *Builder)
      : Counter(0), F(F), FAM(FAM), Builder(Builder) {}

  /// Adds a block dominated by the if header that represents the closing
  /// brackets of an if statement. Returns true when complete
  void run() {
    // Keep running while there are still if structures that can be simplified.
    while (populateWorklist())
      transform();
  }
}; // end IfElseSimplifier
} // namespace

/// Sometimes Mem2Reg/SROA (I think) fails to promote pass-by-value structures,
/// where input arguments are stored within the function call stack (We'll refer
/// to them as local context). Here we try to redirect references to the local
/// context back to the function arguments.
/// NOTE: Traditionally this would not be legal, but as we are only handling
///       device kernels, we will not be modifying the input pointers in any
///       way, hence it should be safe to do this.
void Preprocessor::mergeLocalContext() {
  LLVM_DEBUG(dbgs() << "[Preprocessing] Finding and replacing local kernel "
                       "context and merging with function arguments.\n");
  // NOTE: There also are cases where local variables are allocated for each
  //       individual argument. Currently we can generate legal code from this,
  //       but maybe match those as well in the future to avoid unnecessary
  //       local allocations and copies.
  ByvalStructPromoter Rewriter(F, *Builder);
  Rewriter.rewrite();
  // NOTE: Maybe instead of invalidating everything, think about which analyses
  // are preserved.
  FAM->invalidate(*F, PreservedAnalyses::none());
}

// Stores the integer value of inttoptr instruction into IntVal, and returns
// false if PointerVal is not an inttoptr.
static bool getIntFromI2P(Value *PointerVal, uint64_t &IntVal) {
  using IntToPtrOperator = ConcreteOperator<Operator, Instruction::IntToPtr>;
  // Check for both operators and instruction version of inttoptr
  User *Int2Ptr = dyn_cast<IntToPtrOperator>(PointerVal);
  if (!Int2Ptr) {
    Int2Ptr = dyn_cast<IntToPtrInst>(tracePastCastInsts(PointerVal));
    if (!Int2Ptr)
      return false;
  }
  // Int2Ptr should only have the integer as the operand.
  auto *ConstInt = dyn_cast<ConstantInt>(Int2Ptr->getOperand(0));
  if (!ConstInt)
    llvm_unreachable("Expecting conversion from int to pointer to be constant");
  IntVal = ConstInt->getZExtValue();
  return true;
}

/// Eliminates copies from function arguments to inttoptr'ed address locations
void Preprocessor::eliminateRedundantIntToPtr() {
  LLVM_DEBUG(dbgs() << "[Preprocessing] Undoing urem strength reduction\n");
  DenseMap<uint64_t, StoreInst *> Stores;
  DenseMap<LoadInst *, uint64_t> Loads;
  // Find the load and store instructions that uses inttoptr as the pointer
  // operand.
  for (BasicBlock &BB : *F) {
    for (Instruction &I : BB) {
      uint64_t IntVal;
      if (auto *SI = dyn_cast<StoreInst>(&I)) {
        if (getIntFromI2P(SI->getPointerOperand(), IntVal))
          Stores[IntVal] = SI;
        continue;
      }
      if (auto *LI = dyn_cast<LoadInst>(&I)) {
        if (getIntFromI2P(LI->getPointerOperand(), IntVal))
          Loads[LI] = IntVal;
      }
    }
  }
  // Use the DT in the assertion check for domination
  auto &DT = FAM->getResult<DominatorTreeAnalysis>(*F);
  (void)DT;
  // Replace the loads using the same address as stores
  for (auto KeyVal : Loads) {
    LoadInst *Load = KeyVal.first;
    uint64_t Addr = KeyVal.second;
    auto It = Stores.find(Addr);
    if (It == Stores.end())
      continue;
    StoreInst *Store = It->second;
    assert(DT.dominates(Store, Load) &&
           "Expecting store to dominate the load from int2ptr address.");
    Load->replaceAllUsesWith(Store->getValueOperand());
    Load->eraseFromParent();
  }
  // Erase all the extraneous stores.
  for (auto KeyVal : Stores)
    KeyVal.second->eraseFromParent();
  FAM->invalidate(*F, PreservedAnalyses::none());
}

/// Turn `n - n / A * A` into `n % A`
void Preprocessor::undoURemStrReduction() {
  LLVM_DEBUG(dbgs() << "[Preprocessing] Undoing urem strength reduction\n");
  SmallVector<std::pair<Instruction *, Value *>> URems;
  for (BasicBlock &BB : *F) {
    for (Instruction &I : BB) {
      if (I.getOpcode() == Instruction::Sub)
        if (Value *RemRHS = matchExpandedURem(&I))
          URems.push_back(std::make_pair(&I, RemRHS));
    }
  }
  for (auto Pair : URems) {
    Instruction *I = Pair.first;
    Value *RemRHS = Pair.second;
    assert(RemRHS->getType() == I->getType());
    Builder->SetInsertPoint(I);
    Value *ReplaceWith = Builder->CreateURem(I->getOperand(0), RemRHS, "rem");
    I->replaceAllUsesWith(ReplaceWith);
  }
  FAM->invalidate(*F, PreservedAnalyses::none());
}

/// Revert the potential optimization done by instcombine, where rem + div is
/// turned into and + lshr instructions.
void Preprocessor::undoDivRemStrReduction() {
  LLVM_DEBUG(
      dbgs() << "[Preprocessing] Undoing div + rem strength reduction\n");
  SmallVector<Instruction *> MaybeDivRem;
  for (BasicBlock &BB : *F) {
    for (Instruction &I : BB) {
      if (I.getOpcode() == Instruction::And ||
          I.getOpcode() == Instruction::LShr)
        MaybeDivRem.push_back(&I);
    }
  }
  for (Instruction *I : MaybeDivRem) {
    uint64_t Mod, Div;
    Value *RemVal = matchOptimizedDivRemPair(I, Mod, Div);
    if (!RemVal)
      continue;
    auto *RemInst = dyn_cast<Instruction>(RemVal);
    if (!RemInst)
      continue;

    Builder->SetInsertPoint(I);
    Value *RemLHS = RemInst->getOperand(0);
    unsigned Bitsize = RemLHS->getType()->getIntegerBitWidth();
    Value *ReplaceRem =
        Builder->CreateURem(RemLHS, Builder->getIntN(Bitsize, Mod), "rem");
    Value *ReplaceDiv =
        Builder->CreateUDiv(ReplaceRem, Builder->getIntN(Bitsize, Div), "div");

    RemVal->replaceAllUsesWith(ReplaceRem);
    I->replaceAllUsesWith(ReplaceDiv);
    RemInst->eraseFromParent();
    I->eraseFromParent();
  }
  FAM->invalidate(*F, PreservedAnalyses::none());
}

/// Remove intrinsics that are redundant for conversion
void Preprocessor::removeRedundantInstrs() {
  LLVM_DEBUG(dbgs() << "[Preprocessing] Removing redundant intrinsics\n");
  SmallVector<IntrinsicInst *> RedundantIntrs;
  for (BasicBlock &BB : *F) {
    for (Instruction &I : BB) {
      auto *II = dyn_cast<IntrinsicInst>(&I);
      if (II) {
        switch (II->getIntrinsicID()) {
        case Intrinsic::assume:
        case Intrinsic::lifetime_start:
        case Intrinsic::lifetime_end:
        case Intrinsic::experimental_noalias_scope_decl:
          RedundantIntrs.push_back(II);
          break;
        }
      }
    }
  }

  for (IntrinsicInst *II : RedundantIntrs) {
    II->eraseFromParent();
  }
  FAM->invalidate(*F, PreservedAnalyses::none());
}

/// Remove freeze instructions and replace them with their source
void Preprocessor::removeFreezeInsts() {
  LLVM_DEBUG(dbgs() << "[Preprocessing] Removing freeze instructions\n");
  SmallVector<Instruction *> Freezes;
  for (BasicBlock &BB : *F) {
    for (Instruction &I : BB) {
      if (I.getOpcode() == Instruction::Freeze)
        Freezes.push_back(&I);
    }
  }

  for (Instruction *Freeze : Freezes) {
    Value *Operand = Freeze->getOperand(0);
    Freeze->replaceAllUsesWith(Operand);
    Freeze->eraseFromParent();
  }
  FAM->invalidate(*F, PreservedAnalyses::none());
}

/// Used by eliminateAllocas. Determines if an Alloca instruction is redundant
/// by looking at its use list. Currently only taking into account three
/// instructions: Store, Memcpy, and Load
/// > If the user is a store of memcpy, then get the source value and obtain the
///   source where it is defined.
/// > If it is a load instruction, stop looking at the use chain since we only
///   care where the alloca is copied from.
static void checkRedundantAlloca(
    AllocaInst *AI, DenseMap<AllocaInst *, Value *> &MemMap,
    SmallVector<AllocaInst *> &Deferred,
    SmallVector<std::pair<AllocaInst *, Instruction *>> &ToErase) {
  SmallVector<User *> Worklist;
  for (User *Usr : AI->users())
    Worklist.push_back(Usr);

  // Lambda to check if same value exists in MemMap
  auto InsertIfNotExist = [&MemMap, &Deferred, AI](Value *Val) {
    // We only make this assumption for allocas copied from arguments
    if ((isa<Constant>(Val) && !isa<GlobalValue>(Val)) || !isa<Argument>(Val))
      return true;
    if (auto *CopyAI = dyn_cast<AllocaInst>(Val)) {
      auto It = MemMap.find(CopyAI);
      if (It == MemMap.end()) {
        Deferred.push_back(CopyAI);
        return false;
      }
      Val = It->second;
    }
    auto It = MemMap.find(AI);
    if (It == MemMap.end()) {
      MemMap[AI] = Val;
      return true;
    }
    // If the source kept in MemMap is not the same source as this store, then
    // it is not a redundant alloca
    if (It->second != Val) {
      MemMap.erase(It);
      return false;
    }
    return true;
  };

  while (!Worklist.empty()) {
    User *Usr = Worklist.pop_back_val();
    if (auto *SI = dyn_cast<StoreInst>(Usr)) {
      Value *Src = tracePastCastAndGEP(SI->getValueOperand());
      if (auto *LI = dyn_cast<LoadInst>(Src))
        Src = tracePastCastAndGEP(LI->getPointerOperand());
      if (!InsertIfNotExist(Src))
        return;
      ToErase.push_back({AI, SI});
    } else if (auto *MC = dyn_cast<MemCpyInst>(Usr)) {
      Value *Src = MC->getSource();
      if (!InsertIfNotExist(Src))
        return;
      ToErase.push_back({AI, MC});
    } else if (!isa<LoadInst>(Usr)) {
      // We don't care what happens to the alloca'ed values past the load.
      // Trace the use chain apart from loads.
      for (auto *NewUsr : Usr->users())
        Worklist.push_back(NewUsr);
    }
  }
}

/// Some individual allocas may be missed by mergeLocalContext(), hence here we
/// try to match more individual allocas that are copis of the range values in
/// the function parameter, as these allocated values in default address space
/// will cause issues in MLIR.
void Preprocessor::eliminateAllocas() {
  LLVM_DEBUG(dbgs() << "[Preprocessing] Eliminating redundant alloca insts\n");
  DenseMap<AllocaInst *, Value *> MemMap;
  SmallVector<AllocaInst *> Deferred;
  SmallVector<std::pair<AllocaInst *, Instruction *>> ToErase;
  for (auto &BB : *F) {
    for (auto &I : BB)
      if (auto *AI = dyn_cast<AllocaInst>(&I))
        checkRedundantAlloca(AI, MemMap, Deferred, ToErase);
  }
  SmallVector<AllocaInst *> DeferredCopy(Deferred);
  unsigned NumDeferred = Deferred.size();
  while (1) {
    Deferred.clear();
    for (auto *AI : DeferredCopy)
      checkRedundantAlloca(AI, MemMap, Deferred, ToErase);
    unsigned DeferredSize = Deferred.size();
    // Break if no deferred allocas can be identified as redudant or all
    // deferred allocas are redundant.
    if (DeferredSize == NumDeferred || DeferredSize == 0)
      break;
    NumDeferred = Deferred.size();
    DeferredCopy = Deferred;
  }

  for (auto Pair : MemMap) {
    AllocaInst *AI = Pair.first;
    Value *Src = Pair.second;
    LLVM_DEBUG(dbgs() << "Found alloca: "; AI->dump();
               dbgs() << "\twith source: "; Src->dump());
    PointerType *AllocaTy = AI->getType();
    Type *SrcTy = Src->getType();
    auto *SrcPTy = dyn_cast<PointerType>(SrcTy);
    if (!SrcPTy) {
      LLVM_DEBUG(dbgs() << "No handling of non-pointer src type\n");
      continue;
    }
    if (AllocaTy->getPointerElementType() != SrcPTy->getPointerElementType()) {
      LLVM_DEBUG(dbgs() << "No handling of non-simple copies\n");
      continue;
    }
    Builder->SetInsertPoint(AI);
    Value *NewVal = Src;
    if (AllocaTy->getAddressSpace() != SrcPTy->getAddressSpace())
      NewVal = Builder->CreateAddrSpaceCast(NewVal, AllocaTy);
    for (auto Pair : ToErase) {
      if (Pair.first == AI)
        Pair.second->eraseFromParent();
    }
    AI->replaceAllUsesWith(NewVal);
  }
}

/// Merge chained GEP instructions into a single GEP.
void Preprocessor::mergeGEPs() {
  LLVM_DEBUG(dbgs() << "[Preprocessing] Merging chained GEPs\n");
  SmallVector<GetElementPtrInst *> GEPs;
  for (BasicBlock &BB : *F) {
    for (Instruction &I : BB) {
      if (auto *GEP = dyn_cast<GetElementPtrInst>(&I))
        GEPs.push_back(GEP);
    }
  }
  // Combine chained GEP instructions into a single GEP with casts when
  // needed.
  for (auto *GEP : GEPs) {
    SmallVector<Value *> Indices;
    bool Done = false;
    bool Chained = false;
    Value *Base = nullptr;
    Builder->SetInsertPoint(GEP);
    // Go thorugh the chain of GEP and AddrSpaceCast if exists
    auto *CurrGEP = GEP;
    while (!Done) {
      Value *PtrVal = CurrGEP->getPointerOperand();
      // Look past AddrSpaceCast instructions
      if (auto *ASC = dyn_cast<AddrSpaceCastInst>(PtrVal))
        PtrVal = ASC->getPointerOperand();

      auto *ChainedGEP = dyn_cast<GetElementPtrInst>(PtrVal);
      if (ChainedGEP)
        Chained = true;

      // This will check if there were previous iterations, and save all the
      // relevant information if necessary
      // This will need to be done for the last GEP in a chain as well, who
      // doesn't have another GEP as the pointer operand.
      if (Chained) {
        // The indices need to be in the front of the list in order.
        // The first index of the previous GEP (if exists) need to be merged
        // with the last index of CurGEP
        auto *It = Indices.begin();
        Value *PrevIdx = nullptr;
        if (It != Indices.end()) {
          PrevIdx = *It;
          It = Indices.erase(It);
        }

        unsigned IndexIdx = 0;
        for (unsigned NumIndex = CurrGEP->getNumIndices() - 1;
             IndexIdx < NumIndex; IndexIdx++) {
          // The first operand of GEP is always the pointer operand
          It = Indices.insert(It, CurrGEP->getOperand(IndexIdx + 1)) + 1;
        }

        // The last index of thie GEP to be merged with the previous index
        // Note: When indexing into a structure, only i32 constants are allowed
        // E.g.
        //   %struct.B = type { %struct.A }
        //   %struct.A = type { i32 }
        //   GEP1: %a = getelementptr %struct.B, %struct.B* %b, i64 0, i32 0
        //   GEP2: %v = getelementptr %struct.A, %struct.A* %a, i64 0, i32 0
        //   Merge GEP1 and GEP2, then get:
        //   %m = getelementptr %struct.B, %struct.B* %b, i64 0, i32 0, i32 0
        Value *NewIdx = CurrGEP->getOperand(IndexIdx + 1);
        if (PrevIdx) {
          auto *NewTy = NewIdx->getType();
          auto *ConstPrev = dyn_cast<Constant>(PrevIdx);

          // Merge the two indices
          if (!(ConstPrev && ConstPrev->isZeroValue())) {
            auto *PrevTy = PrevIdx->getType();
            // Make a cast if the indices are of different types
            if (PrevTy != NewTy)
              PrevIdx = Builder->CreateZExtOrTrunc(PrevIdx, NewTy);
            NewIdx = Builder->CreateNUWAdd(PrevIdx, NewIdx, "merge.idx");
          }
        }
        // Insert the new index
        Indices.insert(It, NewIdx);
      }

      // If the ptr is not a GEP, then break from the while loop
      if (!ChainedGEP) {
        Done = true;
        Base = CurrGEP->getPointerOperand();
        break;
      }
      Chained = true;
      CurrGEP = ChainedGEP;
    }

    if (Chained) {
      // Replace the last GEP with a new GEP that gathers all the indices.
      // if addrspace is different, create a cast
      unsigned NewAddrSpace = GEP->getAddressSpace();
      auto *BaseTy = cast<PointerType>(Base->getType());
      if (BaseTy->getAddressSpace() != NewAddrSpace) {
        Base = Builder->CreateAddrSpaceCast(
            Base, PointerType::get(BaseTy->getElementType(), NewAddrSpace));
      }
      auto *NewGEP = Builder->CreateGEP(
          Base->getType()->getPointerElementType(), Base, Indices);
      GEP->replaceAllUsesWith(NewGEP);
      GEP->eraseFromParent();
    }
  }
  FAM->invalidate(*F, PreservedAnalyses::none());
}

/// Turn `and`, `or`, and shift instructions back into `rem`, `add`, `div` and
/// `mul` instructions.
void Preprocessor::undoMulDivStrReduction() {
  LLVM_DEBUG(dbgs() << "[Preprocessing] Transforming valid `and` and shift "
                       "into `rem`, `div`, and `mul`\n");
  SmallVector<Instruction *> Shifts;
  SmallVector<Instruction *> Ands;
  SmallVector<Instruction *> Ors;
  for (BasicBlock &BB : *F) {
    for (Instruction &I : BB) {
      switch (I.getOpcode()) {
      case BinaryOperator::AShr:
      case BinaryOperator::LShr:
      case BinaryOperator::Shl:
        Shifts.push_back(&I);
        break;
      case BinaryOperator::And:
        Ands.push_back(&I);
        break;
      case BinaryOperator::Or:
        Ors.push_back(&I);
        break;
      default:
        break;
      }
    }
  }
  // Shift right instructions:
  for (Instruction *I : Shifts) {
    Value *BaseOp = I->getOperand(0);
    // Look for Shl's being fed into shr's with the same shift. If so they are
    // redundant and can be removed.
    if (I->getOpcode() == Instruction::Shl) {
      for (User *U : I->users()) {
        Instruction *UserInst = dyn_cast<Instruction>(U);
        // Replace if the user is a shift right and has the same shift.
        if (UserInst &&
            (UserInst->getOpcode() == Instruction::AShr ||
             UserInst->getOpcode() == Instruction::LShr) &&
            UserInst->getOperand(1) == I->getOperand(1))
          UserInst->replaceAllUsesWith(BaseOp);
      }
    }
    auto *ConstShift = dyn_cast<ConstantInt>(I->getOperand(1));
    // If the shift operand is not a constant, skip over this.
    if (!ConstShift)
      continue;
    Builder->SetInsertPoint(I);
    // calculate the multiplier based on the shift
    size_t ShiftBy = ConstShift->getZExtValue();
    size_t MulOrDivBy = (size_t)1 << ShiftBy;
    auto *NewRHS = Builder->getIntN(ConstShift->getBitWidth(), MulOrDivBy);
    Value *ReplaceWith;
    switch (I->getOpcode()) {
    case Instruction::LShr:
      ReplaceWith = Builder->CreateUDiv(BaseOp, NewRHS, "div");
      break;
    case Instruction::AShr:
      ReplaceWith = Builder->CreateSDiv(BaseOp, NewRHS, "div");
      break;
    case Instruction::Shl:
      ReplaceWith = Builder->CreateMul(BaseOp, NewRHS, "mul");
      break;
    default:
      llvm_unreachable("Not all opcode handled properly");
    }
    I->replaceAllUsesWith(ReplaceWith);
    I->eraseFromParent();
  }

  // And instructions
  for (Instruction *I : Ands) {
    Value *LHS = I->getOperand(0);
    Value *RHS = I->getOperand(1);
    auto *ConstOp = dyn_cast<ConstantInt>(LHS);
    Value *BaseOp;

    // if none of the operators are constants, skip this instruction
    if (ConstOp)
      BaseOp = RHS;
    else if ((ConstOp = dyn_cast<ConstantInt>(RHS)))
      BaseOp = LHS;
    else
      continue;

    // check if the constant op is one less than a power of two, making it a
    // `rem` operation.
    uint64_t RemOf = ConstOp->getZExtValue() + 1;
    if (!isPowerOf2_64(RemOf))
      continue;

    Builder->SetInsertPoint(I);
    Value *NewRHS = Builder->getIntN(ConstOp->getBitWidth(), RemOf);
    auto *ReplaceWith =
        cast<Instruction>(Builder->CreateURem(BaseOp, NewRHS, "rem"));
    I->replaceAllUsesWith(ReplaceWith);
    I->eraseFromParent();
  }

  // Or instructions. Now that we have changed shl's into muls, check to see
  // if one of the operands is a mul with power of 2, and the other can be
  // guarenteed to be smaller than that power of 2 (Currently only checking
  // for a PHI with two constant incoming values)

  // Lambda helper to identify if a value is a mul with a power of 2.
  auto IsMulWithPow2 = [&](Value *V) -> uint64_t {
    V = tracePastCastInsts(V);
    auto *MaybeMul = dyn_cast<BinaryOperator>(V);
    if (!MaybeMul || MaybeMul->getOpcode() != BinaryOperator::Mul)
      return 0;
    auto *ConstMult = dyn_cast<ConstantInt>(MaybeMul->getOperand(1));
    if (!ConstMult)
      return 0;
    uint64_t MulVal = ConstMult->getZExtValue();
    if (!isPowerOf2_64(MulVal))
      return 0;
    return MulVal;
  }; // end lambda

  // Vector to keep track of incoming values in case a operand is a phinode.
  // This is cleared inside the loop once we find that one of the operands is
  // actually a phinode.
  SmallVector<ConstantInt *, 2> Incomings;
  for (Instruction *I : Ors) {
    Value *LHS = I->getOperand(0);
    Value *RHS = I->getOperand(1);
    Value *CheckLessThan, *MulVar;
    Value *MaybeConstOp;
    uint64_t MulVal;
    if ((MulVal = IsMulWithPow2(LHS))) {
      MulVar = LHS;
      MaybeConstOp = RHS;
      CheckLessThan = tracePastCastInsts(RHS);
    } else if ((MulVal = IsMulWithPow2(RHS))) {
      MulVar = RHS;
      MaybeConstOp = LHS;
      CheckLessThan = tracePastCastInsts(LHS);
    } else
      continue;

    auto ReplaceOrWithAdd = [&]() {
      Builder->SetInsertPoint(I);
      auto *ReplWith = Builder->CreateAdd(MulVar, MaybeConstOp);
      I->replaceAllUsesWith(ReplWith);
      I->eraseFromParent();
    };

    auto *ConstOp = dyn_cast<ConstantInt>(CheckLessThan);
    if (ConstOp) {
      uint64_t OrVal = ConstOp->getZExtValue();
      if (OrVal < MulVal)
        ReplaceOrWithAdd();
      continue;
    }
    // At this point CheckLessThan is not a constant, check if its a phi with
    // only constant incoming values.
    auto *PHI = dyn_cast<PHINode>(CheckLessThan);
    if (!PHI)
      continue;
    bool Skip = false;
    Incomings.clear();
    for (unsigned Idx = 0, NumIncoming = PHI->getNumIncomingValues();
         Idx < NumIncoming; Idx++) {
      ConstOp = dyn_cast<ConstantInt>(PHI->getIncomingValue(Idx));
      if (!ConstOp) {
        Skip = true;
        break;
      }
      Incomings.push_back(ConstOp);
    }
    if (Skip)
      continue;

    // Check if the PHINode incoming values are less than the MulVal, if so
    // then we replace this with an add.
    uint64_t Max = 0;
    for (auto *CI : Incomings) {
      uint64_t ConstVal = CI->getZExtValue();
      if (ConstVal > Max)
        Max = ConstVal;
    }

    if (Max < MulVal)
      ReplaceOrWithAdd();
  } // End replace `or` with `add`
  FAM->invalidate(*F, PreservedAnalyses::none());
}

/// There are instances of truncation where i32 `x+y-1` is transformed into i64
/// `(x + 0xffff ffff + y) & 0xffff ffff`, this will reverse this
/// transformation.
void Preprocessor::undoTruncAdd() {
  LLVM_DEBUG(dbgs() << "Undoing truncated add instructions\n");
  SmallVector<std::pair<Instruction *, Instruction *>> Worklist;
  for (BasicBlock &BB : *F) {
    for (Instruction &I : BB) {
      BinaryOperator *RemInst = matchTruncAdd(&I);
      if (RemInst)
        Worklist.push_back({&I, RemInst});
    }
  }
  for (auto Pair : Worklist) {
    auto *Add = Pair.first;
    auto *Rem = Pair.second;
    Builder->SetInsertPoint(Rem);
    assert(Add->hasOneUse() &&
           "TODO: more than one use of transformed Trunc Add");
    // BinaryOperators should have been canonicalized to have constants as the
    // second operand
    int64_t AddOp = cast<ConstantInt>(Add->getOperand(1))->getZExtValue();
    Add->replaceAllUsesWith(Add->getOperand(0));
    Value *ResultOp = Rem->getOperand(0);
    int64_t RemOp = cast<ConstantInt>(Rem->getOperand(1))->getZExtValue();
    auto *AddBy = ConstantInt::get(ResultOp->getType(), AddOp - RemOp);
    Value *NewAdd = Builder->CreateAdd(ResultOp, AddBy, "trunc.add");
    Rem->replaceAllUsesWith(NewAdd);
  }
  FAM->invalidate(*F, PreservedAnalyses::none());
}

/// Remove cast to larger sized type only to truncate it back again.
// TODO: implement this for integer types as well.
void Preprocessor::removeRedundantExts() {
  LLVM_DEBUG(
      dbgs() << "[Preprocessing] Removing redundant extend instructions\n");
  SmallVector<FPExtInst *> FPExts;
  for (BasicBlock &BB : *F) {
    for (Instruction &I : BB) {
      if (auto *FPEXT = dyn_cast<FPExtInst>(&I)) {
        FPExts.push_back(FPEXT);
      }
    }
  }
  // The goal of this pass is to remove uneccessary extension and truncation
  // into the double type since our hardware cannot support the double type.
  // Without this pass the converter should still work, this just simplifies
  // things. Mainly trying to match the following:
  //   %conv64 = fpext float %2 to double
  //   %add = fadd double %conv64, 1.000000e+00
  //   %div = fdiv double 1.000000e+00, %add
  //   %conv65 = fptrunc double %div to float
  // And convert into
  //   %add = fadd float %2, 1.000000e+00
  //   %conv64 = fpext float %add to double
  //   %div = fdiv double 1.000000e+00, %conv64
  //   %conv65 = fptrunc double %div to float
  // The fpext would then trickle down until it is fed into the fptrunc, where
  // it can then be removed.
  // NOTE: This cannot be an range-based for loop since we may be adding more
  // FPExtInsts to the list inside this for loop, so the size is not static.
  for (size_t Idx = 0; Idx < FPExts.size(); Idx++) {
    FPExtInst *Ext = FPExts[Idx];
    // Check each of the uses of the extended value for the pattern.
    for (auto &Use : Ext->uses()) {
      Value *ExtValue = Ext->getOperand(0);
      // If the FPExt is being truncated directly back, replace all uses of
      // the trunc with the operand of the ext.
      if (auto *Trunc = dyn_cast<FPTruncInst>(Use.getUser())) {
        if (ExtValue->getType() == Trunc->getType()) {
          Trunc->replaceAllUsesWith(ExtValue);
        } else
          continue;
      }

      // If the FPExt is being fed into a expd call, replace the expd with
      // expf
      if (auto *CALL = dyn_cast<CallInst>(Use.getUser())) {
        auto *Func = CALL->getCalledFunction();
        if (Func->getName() != "_Z15__spirv_ocl_expd")
          continue;
        // For now, only handle the fp32 case.
        Type *ExpTy = ExtValue->getType();
        if (!ExpTy->isFloatTy())
          continue;
        // Get or create the expf version of the exp op.
        const char *ExpFName = "_Z15__spirv_ocl_expf";
        Module *M = F->getParent();
        Function *NewFunc = M->getFunction(ExpFName);
        if (!NewFunc) {
          // Create the expf op by copying the expd op. Note this is just a
          // safety in case this func somehow survives after the conversion.
          // It likely wont. However this assumes that the attributes and
          // metadata for the expf op is the same as expd.
          FunctionType *NewFuncTy = FunctionType::get(ExpTy, {ExpTy}, false);
          auto Linkage = Func->getLinkage();
          NewFunc = Function::Create(NewFuncTy, Linkage, ExpFName, *M);
          NewFunc->copyAttributesFrom(Func);
          NewFunc->copyMetadata(Func, 0);
        }
        // Create a call to this new function, moving the extension after
        Builder->SetInsertPoint(CALL->getNextNode());
        StringRef NewCallName = CALL->getName();
        auto *NewCall = Builder->CreateCall(NewFunc, {ExtValue}, NewCallName);
        Type *TargetTy = Ext->getType();
        auto *NewExt = Builder->CreateFPExt(NewCall, TargetTy, Ext->getName());
        // Replace the uses of the old Call with the new one
        CALL->replaceAllUsesWith(NewExt);
        // Push back the NewExt so it can be processed as well.
        FPExts.push_back(cast<FPExtInst>(NewExt));
      }

      // Check if it is being used in a Binary Operation
      auto *BO = dyn_cast<BinaryOperator>(Use.getUser());
      if (!BO)
        continue;

      // Prepare the other operand
      unsigned OtherOperandIdx = Use.getOperandNo() == 0 ? 1 : 0;
      Value *OtherOperand = BO->getOperand(OtherOperandIdx);
      Value *NewOtherOperand = nullptr;
      if (auto *ConstFP = dyn_cast<ConstantFP>(OtherOperand)) {
        // If the other operand is a constant
        double ConstVal = ConstFP->getValueAPF().convertToDouble();
        NewOtherOperand = ConstantFP::get(ExtValue->getType(), ConstVal);
      } else if (auto *OtherFPExt = dyn_cast<FPExtInst>(OtherOperand)) {
        // If the other operand is also and FPExt going from the same type
        Value *OtherFPExtArg = OtherFPExt->getOperand(0);
        if (ExtValue->getType() != OtherFPExtArg->getType())
          continue;
        NewOtherOperand = OtherFPExtArg;
      } else {
        continue;
      }

      // Create the new BO (this is to update its type)
      Builder->SetInsertPoint(BO);
      Value *LHS = OtherOperandIdx == 0 ? NewOtherOperand : ExtValue;
      Value *RHS = OtherOperandIdx == 1 ? NewOtherOperand : ExtValue;
      StringRef Name = BO->getName();
      auto *NewBO = Builder->CreateBinOp(BO->getOpcode(), LHS, RHS, Name);
      // Create the newExt
      auto *NewExt = Builder->CreateFPExt(NewBO, BO->getType());
      // Replace all uses of the BO with the newly extended value.
      BO->replaceAllUsesWith(NewExt);

      // Push back the NewExt so it can be processed as well.
      FPExts.push_back(cast<FPExtInst>(NewExt));
    }
  }
  FAM->invalidate(*F, PreservedAnalyses::none());
}

/// The way that SYCL/LLVM lowers condition logic sometimes generates boolean
/// select instructions which can be simplified into either an AND/OR
/// instruction. This preprocessing pass matches for these select instructions
/// and replaces them.
void Preprocessor::simplifySelectLogic() {
  LLVM_DEBUG(dbgs() << "[Preprocessing] Simplifying select logic.\n");
  // Collect all select ops that result in a boolean value
  SmallVector<SelectInst *> BoolSelectInsts = {};
  for (BasicBlock &BB : *F)
    for (Instruction &I : BB)
      if (auto *SEL = dyn_cast<SelectInst>(&I))
        if (SEL->getType()->isIntegerTy(1))
          BoolSelectInsts.push_back(SEL);
  // Check if the select can be simplified, and if so replace it.
  for (SelectInst *SEL : BoolSelectInsts) {
    Value *Cond = SEL->getCondition();
    Value *TrueVal = SEL->getTrueValue();
    Value *FalseVal = SEL->getFalseValue();
    // Match for AND pattern
    //  Select %A, %B, FALSE -> And %A, %B
    if (auto *Const = dyn_cast<ConstantInt>(FalseVal)) {
      if (Const->isZero()) {
        Builder->SetInsertPoint(SEL);
        Value *AND = Builder->CreateAnd(Cond, TrueVal, SEL->getName());
        SEL->replaceAllUsesWith(AND);
      }
    }
    // Match for OR pattern
    //  Select %A, TRUE, %B -> OR %A, %B
    if (auto *Const = dyn_cast<ConstantInt>(TrueVal)) {
      if (Const->isOne()) {
        Builder->SetInsertPoint(SEL);
        Value *OR = Builder->CreateOr(Cond, FalseVal, SEL->getName());
        SEL->replaceAllUsesWith(OR);
      }
    }
  }
  FAM->invalidate(*F, PreservedAnalyses::none());
}

/// Sycl handles half types as i16's to help targets that doesn't have half
/// type support, however we would like to keep the type as half whenever
/// possible during conversion.
void Preprocessor::removeHalfToI16() {
  LLVM_DEBUG(
      dbgs() << "[Preprocessing] Removing redundant half to i16 casts\n");
  SmallVector<BitCastInst *> HalfToI16PtrBitcasts;
  for (BasicBlock &BB : *F) {
    for (Instruction &I : BB) {
      if (auto *BC = dyn_cast<BitCastInst>(&I)) {
        Type *SrcTy = BC->getSrcTy();
        if (!SrcTy->isPointerTy())
          continue;
        Type *SrcEltTy = SrcTy->getPointerElementType();
        if (SrcEltTy->isStructTy())
          SrcEltTy = SrcEltTy->getStructElementType(0);
        Type *DestEltTy = BC->getDestTy()->getPointerElementType();
        if (SrcEltTy->isHalfTy() && DestEltTy->isIntegerTy(16))
          HalfToI16PtrBitcasts.push_back(BC);
      }
    }
  }
  for (BitCastInst *BC : HalfToI16PtrBitcasts) {
    // Create a new BitCast to cast into a Half *. This is because when
    // building ops like Store, they require the Ptr type to be exiplicit.
    Builder->SetInsertPoint(BC);
    unsigned DestAS = BC->getOperand(0)->getType()->getPointerAddressSpace();
    Type *DestTy = PointerType::get(Builder->getHalfTy(), DestAS);
    auto NewBCName = BC->getName();
    auto *NewBC = Builder->CreateBitCast(BC->getOperand(0), DestTy, NewBCName);
    // Go through each of the users of the original BitCast and replace them
    // with Half types
    for (auto *User : BC->users()) {
      if (auto *SI = dyn_cast<StoreInst>(User)) {
        if (auto *Const = dyn_cast<ConstantInt>(SI->getValueOperand())) {
          // If the user is a store instruction, storing a constant I16 into
          // the pointer, then replace it such that it is storing a half
          // instead.
          Builder->SetInsertPoint(SI);
          auto NewVal = APFloat(llvm::APFloat::IEEEhalf(), Const->getValue());
          auto *NewConst = ConstantFP::get(Builder->getHalfTy(), NewVal);
          auto NewAlign = SI->getAlign();
          auto *NewStore =
              Builder->CreateAlignedStore(NewConst, NewBC, NewAlign);
          NewStore->copyMetadata(*SI);
          SI->eraseFromParent();
        }
      }
    }
  }
  FAM->invalidate(*F, PreservedAnalyses::none());
}

/// Although MLIR can technically support select between Memrefs, the analysis
/// is very complicated and is better suited as an optimization down the
/// pipeline. This preprocessing pass will look for select instructions that
/// operate on pointers and rotate it so the use comes before the select.
/// For example:
///   %sel = select i1 %cond, float* %A, float* %B
///   %ld = load float, float* %sel, align 4, !tbaa !7
///            |
///            V
///   %ld_true = load float, float* %A, align 4, !tbaa !7
///   %ld_false = load float, float* %B, align 4, !tbaa !7
///   %sel = select i1 %cond, float %ld_true, float %ld_false
void Preprocessor::removePtrSelects() {
  LLVM_DEBUG(
      dbgs() << "[Preprocessing] Removing pointer select instructions\n");
  // Collect all select instructions that return pointers.
  SmallPtrSet<SelectInst *, 4> PointerSelects;
  for (BasicBlock &BB : *F)
    for (Instruction &I : BB)
      if (auto *SEL = dyn_cast<SelectInst>(&I))
        if (SEL->getType()->isPointerTy())
          PointerSelects.insert(SEL);
  // Iterate over the pointer selects.
  while (!PointerSelects.empty()) {
    SelectInst *SEL = *PointerSelects.begin();
    // Create a new select for each use of the select.
    for (auto &Use : SEL->uses()) {
      auto *User = Use.getUser();
      // TODO: Handle the case when the select result is being used as an
      // operator.
      auto *UserInst = dyn_cast<Instruction>(User);
      if (!UserInst)
        continue;
      // Safety: if the user of this select is a pointer select, it is likely
      // going to be in PointerSelects; and it is a terrible idea to transform
      // it!
      // TODO: Handle this; need to handle the second ptr select before the
      // first.
      if (isa<SelectInst>(User) && User->getType()->isPointerTy())
        continue;
      // Create the true Inst
      auto *TrueInst = UserInst->clone();
      TrueInst->setName(UserInst->getName() + "_true");
      TrueInst->setOperand(Use.getOperandNo(), SEL->getTrueValue());
      TrueInst->insertBefore(SEL);
      // Create the false Inst
      auto *FalseInst = UserInst->clone();
      FalseInst->setName(UserInst->getName() + "_false");
      FalseInst->setOperand(Use.getOperandNo(), SEL->getFalseValue());
      FalseInst->insertBefore(SEL);
      // Create the select that will be selecting these two values
      Builder->SetInsertPoint(SEL);
      Value *NewSEL = Builder->CreateSelect(SEL->getCondition(), TrueInst,
                                            FalseInst, SEL->getName(), SEL);
      // Replace the old instruction with the select
      UserInst->replaceAllUsesWith(NewSEL);
      // If the new select instruction is of pointer type, then we need to add
      // it to the list of pointer selects.
      if (NewSEL->getType()->isPointerTy())
        PointerSelects.insert(cast<SelectInst>(NewSEL));
    }
    PointerSelects.erase(SEL);
  }
  FAM->invalidate(*F, PreservedAnalyses::none());
}

/// LLVM performs the following optimization:
/// bool = !((LB <= x) && (x <= UB)) with 2 cmp and 1 or into:
///    %add = add nsw i64 %x, -(UB+1)
///    %bool = icmp ult i64 %add, -(UB+1-LB)
/// This function tries to match for this and convert it into:
///    %cmp1 = icmp ult i64 %x, LB
///    %cmp2 = icmp uge i64 %x, UB+1
///    %bool = or i1 cmp1, cmp2
void Preprocessor::undoCombinedCmp() {
  LLVM_DEBUG(dbgs() << "[Preprocessing] Undoing combined compares\n");
  // Populate worklist
  SmallVector<CmpInst *> Worklist;
  for (auto &BB : *F) {
    for (auto &I : BB) {
      auto *CI = dyn_cast<CmpInst>(&I);
      if (!CI || !CI->isUnsigned())
        continue;
      // The compares should have canonicalized to have constants on RHS:
      auto *ConstRHS = dyn_cast<ConstantInt>(CI->getOperand(1));
      if (!ConstRHS)
        continue;
      int64_t RHSVal = ConstRHS->getSExtValue();
      if (RHSVal >= 0)
        continue;
      // Currently only saw this one variant. TODO: there might be other
      // variations of this family of transformations?
      if (CI->getPredicate() != CmpInst::ICMP_ULT)
        continue;
      Worklist.push_back(CI);
    }
  }

  for (auto *CI : Worklist) {
    LLVM_DEBUG(dbgs() << "[Preprocessing] Transforming unsigned cmp with "
                         "negative constant: ";
               CI->dump());
    Value *AddVal = CI->getOperand(0);
    // The less-than bound is the difference between the add and cmp operand.
    int64_t NegCmpOp =
        cast<ConstantInt>(CI->getOperand(1))->getSExtValue() * -1;
    Value *AddRHS, *InboundVal;
    if (!match(AddVal, m_Add(m_Value(InboundVal), m_Value(AddRHS)))) {
      LLVM_DEBUG(
          dbgs() << "\t[WARNING] Failed to find add inst from first operand\n");
      continue;
    }
    auto *AddRHSVal = dyn_cast<ConstantInt>(AddRHS);
    if (!AddRHSVal) {
      LLVM_DEBUG(dbgs() << "\t[WARNING] Expecting Inbound Check to be addition "
                           "with constant\n");
      continue;
    }
    // The greater-or-equal value is the negated addition operand
    int64_t GE = AddRHSVal->getSExtValue() * -1;
    if (GE < 0) {
      LLVM_DEBUG(
          dbgs() << "\t[WARNING] Expecting Inbound add to be negative\n");
      continue;
    }
    int64_t LT = GE - NegCmpOp;
    Type *ITy = InboundVal->getType();
    Builder->SetInsertPoint(CI);
    auto *GEInst = Builder->CreateICmpUGE(InboundVal, ConstantInt::get(ITy, GE),
                                          "transform.ub");
    auto *LTInst = Builder->CreateICmpULT(InboundVal, ConstantInt::get(ITy, LT),
                                          "transform.lb");
    auto *OrInst = Builder->CreateOr(GEInst, LTInst, "transform.combine");
    CI->replaceAllUsesWith(OrInst);
  }
  FAM->invalidate(*F, PreservedAnalyses::none());
}

/// Recursive function to hoist the multiplication of the range value out of
/// add instructions such that the GEP index using the linearized offset will
/// always be an addition of two indices.
static void liftOperand(Value *Mul, User *AddOrSub, IRBuilder<> &Builder) {
  auto IsAddOrSub = [&](Value *V) {
    auto *BO = dyn_cast<BinaryOperator>(V);
    if (!BO)
      return false;
    auto Opcode = BO->getOpcode();
    if (Opcode != Instruction::Add && Opcode != Instruction::Sub)
      return false;
    return true;
  };

  if (!IsAddOrSub(AddOrSub))
    return;

  // Find the other operand that is not Mul
  Value *LHS = AddOrSub->getOperand(0);
  if (LHS == Mul)
    LHS = AddOrSub->getOperand(1);

  for (User *Usr : AddOrSub->users()) {
    if (!IsAddOrSub(Usr))
      continue;
    // Find the operand that is not the AddOrSub
    Value *RHS = Usr->getOperand(0);
    if (RHS == AddOrSub)
      RHS = Usr->getOperand(1);
    // Create the new adds
    Builder.SetInsertPoint(cast<Instruction>(Usr));
    Value *NewRHS = Builder.CreateAdd(LHS, RHS, "add.move");
    auto *ReplaceWith =
        cast<User>(Builder.CreateNUWAdd(Mul, NewRHS, "add.lift"));
    Usr->replaceAllUsesWith(ReplaceWith);
    // Recursive call
    liftOperand(Mul, ReplaceWith, Builder);
  }
}

/// Try to transform:
/// ((i1 * Range[0]) + Offset[0]) + i0 =>
/// (i1 * Range[0]) + (Offset[0] + i0)
///                 ^ Make sure this is the add instruction a GEP would see.
/// So that delinearization can be performed more easily
static void moveOffsetAddition(Argument *Range, IRBuilder<> &Builder) {
  SmallVector<Value *> LoadedOffsets;
  DenseSet<User *> ArgUsers;
  ArgUsers.insert(Range->user_begin(), Range->user_end());

  // Search for all the loaded values of the offset past the use-def chain
  while (!ArgUsers.empty()) {
    User *Usr = *ArgUsers.begin();
    assert(!isa<PHINode>(Usr) &&
           "sycl offset should never be used in a PHINode");
    if (isa<LoadInst>(Usr))
      LoadedOffsets.push_back(Usr);
    else
      ArgUsers.insert(Usr->user_begin(), Usr->user_end());

    ArgUsers.erase(Usr);
  }

  SmallVector<std::pair<BinaryOperator *, Value *>> CandidateAdds;
  // For all loaded values, check for their uses and search for the nested
  // Mul->Add pattern
  for (auto *LoadVal : LoadedOffsets) {
    for (User *Usr : LoadVal->users()) {
      // check if the user is an add with the previous multiplied index:
      if (auto *MaybeMul = dyn_cast<BinaryOperator>(Usr)) {
        if (MaybeMul->getOpcode() != BinaryOperator::Mul)
          continue;
        for (User *MulUsr : MaybeMul->users()) {
          if (auto *NestedAdd = dyn_cast<BinaryOperator>(MulUsr)) {
            if (NestedAdd->getOpcode() != BinaryOperator::Add)
              continue;
            CandidateAdds.push_back({NestedAdd, MaybeMul});
          }
        }
      }
    }
  } // End user search loop

  // Look for all add and sub instructions involving the Mul Value, and
  // gradually lift the use of Mul to the root of the binary use tree
  for (auto Pair : CandidateAdds) {
    Value *Mul = Pair.second;
    BinaryOperator *Add = Pair.first;
    liftOperand(Mul, Add, Builder);
  }
}

// Since we are converting rotated for loops back into for loops, we no longer
// require loop guards, and they only interfere with the conversion process.
void Preprocessor::removeLoopGuard() {
  LLVM_DEBUG(dbgs() << "[Preprocessing] Converting loop guard branches to "
                       "unconditional branches.\n");
  // Loop guard recognition requires loop simplify:
  FunctionPassManager FPM;
  FPM.addPass(SimplifyCFGPass());
  FPM.addPass(LoopSimplifyPass());
  FPM.run(*F, *FAM);

  // NOTE: There are loop guards that this method cannot recognize, but they
  // don't impact correctness so we're ignoring them for now.
  bool Changed = false;
  // Populate worklist of loop guard insts.
  LoopInfo *LI = &FAM->getResult<LoopAnalysis>(*F);
  for (auto *L : LI->getLoopsInPreorder()) {
    auto *Guard = L->getLoopGuardBranch();
    if (!Guard)
      continue;
    // Verify that this is actually indeed a loop guard, not just an if
    // condition wrapping a loop.
    BasicBlock *Preheader;
    matchLoopGuard(L, Guard, Preheader);
    if (!Preheader)
      continue;
    LLVM_DEBUG(dbgs() << "\tRecognized and removing loop guard: ";
               Guard->dump());
    // If there are any phi nodes in the other child of Guard, remove its
    // value from this BB.
    BasicBlock *ParentBlk = Guard->getParent();
    BasicBlock *NonLoopChild = Guard->getSuccessor(0);
    if (NonLoopChild == Preheader)
      NonLoopChild = Guard->getSuccessor(1);
    for (auto &PHI : NonLoopChild->phis()) {
      PHI.removeIncomingValue(ParentBlk);
      if (PHI.getNumIncomingValues() == 1)
        PHI.replaceAllUsesWith(PHI.getIncomingValue(0));
    }
    // Now that the guard is verified to be a guard generated from a rotation,
    // set the guard branch to be an unconditional and point to Preheader
    Builder->SetInsertPoint(Guard);
    Builder->CreateBr(Preheader);
    Guard->eraseFromParent();

    Changed = true;
  }

  if (Changed) {
    // Invalidate the analyis passes since we've made change to the IR.
    FAM->invalidate(*F, PreservedAnalyses::none());
  }
}

/// Simplifies "if"s the same way that loop-simplify simplifies loops - adds
/// dedicated exits to conditional branches such that there are never more than
/// 3 incoming edges to any BB. The dedicated exit would be have the if header
/// as the immediate dominator.
void Preprocessor::ifSimplify() {
  LLVM_DEBUG(dbgs() << "[Preprocessing] Simplifying if/else control flow\n");
  IfElseSimplifier IES(F, FAM, Builder);
  IES.run();
  FAM->invalidate(*F, PreservedAnalyses::none());
}

/// Tries to pattern match for device accessors as function arguments:
///
/// float addrspace(1)* %_arg_,
/// %"class._ZTSN2cl4sycl5rangeILi3EEE.cl::sycl::range"* ... %_arg_1,
/// %"class._ZTSN2cl4sycl5rangeILi3EEE.cl::sycl::range"* ... %_arg_2,
/// %"class._ZTSN2cl4sycl5rangeILi3EEE.cl::sycl::range"* ... %_arg_3
///
/// Where _arg_ is the base pointer, _arg_2 is the accessor range, and _arg_3
/// is the offset. _arg_1 is the memory range, which seems to be a oneAPI
/// extension.
/// Returns the number of shapes inserted into ShapeMap
unsigned
Preprocessor::parseAccessorArguments(ValueMap<Value *, Shape> &ShapeMap) {
  LLVM_DEBUG(dbgs() << "[Preprocessing] Parsing accessors in arguments\n");
  unsigned RetVal = 0;
  unsigned GEPIdx;
  Type *ArrElTy = nullptr;
  Argument *Arg = F->arg_begin();
  auto *ArgEnd = F->arg_end();

  SmallVector<Argument *> Ranges;

  // Lambda to match arg 1 to 3
  auto GetRangeDims = [&](Argument *Arg) -> int {
    if (Arg == ArgEnd)
      return -1;
    auto *RangePtrTy = dyn_cast<PointerType>(Arg->getType());
    // device accessor range objects should be in addrspace 0
    if (!RangePtrTy)
      return -1;
    const Type *ElTy = RangePtrTy->getElementType();
    unsigned TypeLevel = unwrapStructs(ElTy);
    auto *RangeArrTy = dyn_cast<ArrayType>(ElTy);
    // the range array need to be of integers
    if (!RangeArrTy)
      return -1;
    const Type *OldType = ArrElTy;
    ArrElTy = RangeArrTy->getElementType();
    // The array need to contain integers, and all the ranges should contain
    // the same type.
    if (!ArrElTy->isIntegerTy() || (OldType && OldType != ArrElTy))
      return -1;
    // GEP index which would index into the final array
    GEPIdx = TypeLevel + 1;
    return RangeArrTy->getNumElements();
  };

  while (Arg != ArgEnd) {
    // In case pattern matching failed, go to PrevArgIdx+1.
    auto *PrevArg = Arg;
    Argument *PtrVal = Arg++;
    auto *PtrTy = dyn_cast<PointerType>(PtrVal->getType());
    // Right now we only support buffers of vectors or scalar types
    if (!PtrTy)
      continue;
    Type *BaseElTy = PtrTy->getElementType();
    if (!BaseElTy->isSingleValueType())
      continue;

    // Now parse _arg_1, which needs to be an array type underneath
    Argument *MemRange = Arg++;
    int MemRangeDim = GetRangeDims(MemRange);
    if (MemRangeDim < 0) {
      Arg = PrevArg + 1;
      continue;
    }

    // _arg_2 and _arg_3 needs to be of the same dimension as _arg_1 to be a
    // valid accessor
    Argument *AccRange = Arg++;
    int AccRangeDim = GetRangeDims(AccRange);
    if (AccRangeDim != MemRangeDim) {
      Arg = PrevArg + 1;
      continue;
    }

    // The offset in reality is a sycl::id, not a sycl::range, however since
    // in LLVM IR, they are the same "struct wrapping an array", it might get
    // optimized to be the same type.
    Argument *OffsetRange = Arg++;
    int OffRangeDim = GetRangeDims(OffsetRange);
    if (OffRangeDim != MemRangeDim) {
      Arg = PrevArg + 1;
      continue;
    }

    Ranges.push_back(AccRange);

    // If all 3 matches succeeded, then we assume this is an accessor.
    // Generate the Shape's for each argument.
    auto MapIt = ShapeMap.insert({PtrVal, Shape()}).first;
    Shape *BaseShape = &MapIt->second;
    MapIt = ShapeMap.insert({AccRange, Shape()}).first;
    Shape *RangeShape = &MapIt->second;
    MapIt = ShapeMap.insert({OffsetRange, Shape()}).first;
    Shape *OffsetShape = &MapIt->second;
    // Setting all the parameters for each shape
    // Starting with the base shape
    BaseShape->setName("v" + std::to_string(RetVal++));
    BaseShape->setElementType(BaseElTy);
    BaseShape->setDynamic(true);
    BaseShape->setShapeType(Shape::Input);
    BaseShape->setRoot(PtrVal);
    BaseShape->setAddrSpace(PtrTy->getAddressSpace());
    // Since this is a bare pointer, the gep index should just be 0
    BaseShape->addDim(1, 0);
    BaseShape->setNumDims(AccRangeDim);
    BaseShape->setOffset(OffsetShape);
    BaseShape->setRange(RangeShape);

    // Then RangeShape
    RangeShape->setName("v" + std::to_string(RetVal++));
    RangeShape->setElementType(ArrElTy);
    RangeShape->setShapeType(Shape::Input | Shape::Range);
    RangeShape->setRoot(AccRange);
    RangeShape->setAddrSpace(0);
    RangeShape->addDim(AccRangeDim, GEPIdx);
    RangeShape->setBase(BaseShape);

    // Finally OffsetShape
    OffsetShape->setName("v" + std::to_string(RetVal++));
    OffsetShape->setElementType(ArrElTy);
    OffsetShape->setShapeType(Shape::Input | Shape::Offset);
    OffsetShape->setRoot(OffsetRange);
    OffsetShape->setAddrSpace(0);
    OffsetShape->addDim(OffRangeDim, GEPIdx);
    OffsetShape->setBase(BaseShape);

    LLVM_DEBUG(dbgs() << "\tFound " << OffRangeDim << "D accessor for "
                      << PtrVal->getName() << "\n");
  }

  // Additional transformation to make delinearization easier
  for (auto *Range : Ranges)
    moveOffsetAddition(Range, *Builder);
  FAM->invalidate(*F, PreservedAnalyses::none());

  return RetVal;
}

Preprocessor::Preprocessor(Function *F, IRBuilder<> *Builder,
                           FunctionAnalysisManager *FAM)
    : F(F), Builder(Builder), FAM(FAM) {}

void Preprocessor::run() {
  // Some preprocessing passes may depend on ones before it.
  mergeLocalContext();
  eliminateRedundantIntToPtr();
  undoURemStrReduction();
  undoDivRemStrReduction();
  removeRedundantInstrs();
  removeFreezeInsts();
  eliminateAllocas();
  mergeGEPs();
  undoMulDivStrReduction();
  undoTruncAdd();
  removeRedundantExts();
  simplifySelectLogic();
  removeHalfToI16();
  removePtrSelects();
  undoCombinedCmp();
  removeLoopGuard();
  ifSimplify();

  // Run other builtin LLVM passes.
  FunctionPassManager FPM;
  FPM.addPass(DCEPass());
  FPM.addPass(LoopSimplifyPass());
  FPM.addPass(LCSSAPass());
  LLVM_DEBUG(dbgs() << "[Preprocessing] Running registered passes\n");
  FPM.run(*F, *FAM);
}

} // namespace converter
} // namespace llvm
