//===-- TrampolineBuilder.h - Trampoline Func Call Builder ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The SYCLops Converter builds kernels from LLVMIR functions and the arguments
// passed into these kernels likely will not match the original function. The
// purpose of the trampoline function is to provide an interface/translation
// between the old function and the new kernel, such that the old converter is
// converted into a single call instruction, calling what the kernel should look
// like after lowering back to LLVMIR. Some other instructions may be required
// to translate the sycl constructs to bare pointers and scalars.
//
//===----------------------------------------------------------------------===//

#include "TrampolineBuilder/TrampolineBuilder.h"
#include "Util/ConverterUtil.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Value.h"
#include "llvm/Support/Debug.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"

using namespace llvm;
using namespace llvm::converter;
using std::string;

#define DEBUG_TYPE "trampoline-builder"

/// Constructor for the TrampolineBuilder.
///
/// Sets up the IRBuilder used throughout the trampoline generation using the
/// given context.
TrampolineBuilder::TrampolineBuilder(LLVMContext &Ctx) : Builder(Ctx) {}

/// Initialize the trampoline builder.
///
/// This creates a new block at the end of the given function that will house
/// the trampoline function. This method also resets all internal variables used
/// by this builder.
void TrampolineBuilder::initialize(Function *F) {
  // Create the Trampoline Block and set the insertion point to the end of it.
  this->TrampBlock = BasicBlock::Create(Builder.getContext(), "entry", F);
  Builder.SetInsertPoint(this->TrampBlock);
  // Reset internal variables
  this->F = F;
  this->TrampolineFunc = nullptr;
  this->KernelName.clear();
  this->TrampolineArgs.clear();
  this->RangeMap.clear();
  this->CastMap.clear();
}

/// Internal method for creating the GEP and Load necessary to get a SYCL Array
/// from a wrapper class.
///
/// This method keeps an internal map to prevent duplicate GEPs and Loads for
/// the same Root.
Value *TrampolineBuilder::getSyclArrayFromWrapper(Value *Root) {
  // clang-format off
    // %0 = getelementptr inbounds %struct._ZTS15__wrapper_class.__wrapper_class,
    //         %struct._ZTS15__wrapper_class.__wrapper_class* %_arg_, i64 0, i32 0
    // %1 = load %"struct._ZTSN2cl4sycl5ArrayIfLm4EJLm16ELm16ELm5EEEE.cl::sycl::Array" addrspace(1)*,
    //         %"struct._ZTSN2cl4sycl5ArrayIfLm4EJLm16ELm16ELm5EEEE.cl::sycl::Array" addrspace(1)** %0, align 8,
    //         !tbaa !7
  // clang-format on

  assert(Root->getType()->isPointerTy() &&
         "SyclWrapper root expected to be a pointer.");

  // Check if this value has already been created, if so return it. Set the
  // second arg of the pair to zero just so it has a value.
  auto RangeMapKey = std::make_pair(Root, 0);
  auto It = this->RangeMap.find(RangeMapKey);
  if (It != this->RangeMap.end())
    return It->second;

  // Set the insertion point to the end of the Trampoline Block.
  Builder.SetInsertPoint(this->TrampBlock);

  // Create the GEP instruction
  Type *GEPType = getPointerElementType(Root);
  ConstantInt *ConstZero = Builder.getInt32(0);
  Value *GEPInst = Builder.CreateGEP(GEPType, Root, {ConstZero, ConstZero});
  // Create the load instruction
  Type *LoadType = cast<GetElementPtrInst>(GEPInst)->getResultElementType();
  Value *SyclArray = Builder.CreateLoad(LoadType, GEPInst);

  // Set the Range name
  string Name = Root->getName().str() + ".array";
  SyclArray->setName(Name);

  // Insert the value into the map
  this->RangeMap[RangeMapKey] = SyclArray;
  return SyclArray;
}

/// Internal method for creating the GEP and Load necessary to get a single dim
/// from a SYCL Range.
///
/// This method keeps an internal map to prevent duplicate GEPs and Loads for
/// the same Root, Dim pair.
Value *TrampolineBuilder::getSyclRangeDim(Value *Root, uint64_t Dim) {
  // clang-format off
    // %0 = getelementptr inbounds %"class._ZTSN2cl4sycl5rangeILi1EEE.cl::sycl::range",
    //         %"class._ZTSN2cl4sycl5rangeILi1EEE.cl::sycl::range"* %_arg_1, i64 0, i32 0, i32 0, i64 Dim
    // %1 = load i64, i64* %0, align 8
  // clang-format on

  assert(Root->getType()->isPointerTy() &&
         "SyclRange root expected to be a pointer.");

  // Check if this value has already been created, if so return it.
  auto RangeMapKey = std::make_pair(Root, Dim);
  auto It = this->RangeMap.find(RangeMapKey);
  if (It != this->RangeMap.end())
    return It->second;

  // Set the insertion point to the end of the Trampoline Block.
  Builder.SetInsertPoint(this->TrampBlock);

  // Create the GEP instruction
  Type *GEPType = getPointerElementType(Root);
  ConstantInt *ConstZero = Builder.getInt32(0);
  Value *GEPInst = Builder.CreateGEP(
      GEPType, Root, {ConstZero, ConstZero, ConstZero, Builder.getInt64(Dim)});
  // Create the load instruction
  Type *LoadType = cast<GetElementPtrInst>(GEPInst)->getResultElementType();
  Value *RangeDim = Builder.CreateLoad(LoadType, GEPInst);

  // Set the Range name
  string Name = Root->getName().str() + ".range." + std::to_string(Dim);
  RangeDim->setName(Name);

  // Insert the value into the map
  this->RangeMap[RangeMapKey] = RangeDim;
  return RangeDim;
}

/// Internal method for creating the GEP and Load necessary to get a single
/// offset from a SYCL ID.
///
/// Since the GEP and Load pattern for SYCL ID is identical to SYCL Ranges, this
/// method simply calls getSyclRangeDim, however this may not always be true.
Value *TrampolineBuilder::getSyclID(Value *Root, uint64_t Dim) {
  // clang-format off
    // %0 = getelementptr inbounds %"class._ZTSN2cl4sycl2idILi2EEE.cl::sycl::id",
    //         %"class._ZTSN2cl4sycl2idILi2EEE.cl::sycl::id"* %_arg_13, i64 0, i32 0, i32 0, i64 1
    // %1 = load i64, i64* %0, align 8
  // clang-format on

  // Since the GEP and Load necessary for getting the offset from a SYCL ID is
  // the same as getting a dim from a Sycl Range, reuse the method.
  Value *SyclID = getSyclRangeDim(Root, Dim);

  // Set the ID name
  string Name = Root->getName().str() + ".id." + std::to_string(Dim);
  SyclID->setName(Name);

  return SyclID;
}

/// Internal method for creating the GEP and Load necessary to get the value of
/// a SYCL Dim.
///
/// This method keeps an internal map to prevent duplicate GEPs and Loads for
/// the same Root.
Value *TrampolineBuilder::getSyclDim(Value *Root) {
  // clang-format off
    // %dim.value.ptr = getelementptr %"struct._ZTSN2cl4sycl3DimE.cl::sycl::Dim",
    //                     %"struct._ZTSN2cl4sycl3DimE.cl::sycl::Dim"* %size1, i32 0, i32 0
    // %dim.value = load i64, i64* %dim.value.ptr, align 8
  // clang-format on

  assert(Root->getType()->isPointerTy() &&
         "SyclDim root expected to be a pointer.");

  // Check if this value has already been created, if so return it. Set the
  // second arg of the pair to zero just so it has a value.
  auto RangeMapKey = std::make_pair(Root, 0);
  auto It = this->RangeMap.find(RangeMapKey);
  if (It != this->RangeMap.end())
    return It->second;

  // Set the insertion point to the end of the Trampoline Block.
  Builder.SetInsertPoint(this->TrampBlock);

  // Create the GEP
  // Dim should be in the form on {i64}*, hence `GEP Val, 0, 0`
  Type *GEPType = getPointerElementType(Root);
  ConstantInt *ConstZero = Builder.getInt32(0);
  Value *GEPInst = Builder.CreateGEP(GEPType, Root, {ConstZero, ConstZero});

  // Create the Load
  Type *LoadType = cast<GetElementPtrInst>(GEPInst)->getResultElementType();
  Value *Dim = Builder.CreateLoad(LoadType, GEPInst);

  // Set the Dim name
  string Name = Root->getName().str() + ".dim.value";
  Dim->setName(Name);

  // Insert the value into the map
  this->RangeMap[RangeMapKey] = Dim;
  return Dim;
}

/// Internal method for casting arg values to a target type.
///
/// This method keeps an internal map to prevent duplicate casts to the same
/// type.
Value *TrampolineBuilder::castArg(Value *Arg, Type *TargetTy) {
  // clang-format off
    // for pointers:
    // %hybrid_arg = bitcast %"struct._ZTSN2cl4sycl5ArrayIfLm16EJLm32EEEE.cl::sycl::Array"
    //                  addrspace(1)* %_arg_ to float addrspace(1)*
    // for integers:
    // %_arg_12.range.0.cast = trunc i64 %_arg_12.range.0 to i32
  // clang-format on

  // If the Arg has the same type as TargetTy no casting is needed.
  if (Arg->getType() == TargetTy)
    return Arg;

  // Check if this value has already been cast to the target type, if so return
  // the casted value.
  auto CastMapKey = std::make_pair(Arg, TargetTy);
  auto It = this->CastMap.find(CastMapKey);
  if (It != this->CastMap.end())
    return It->second;

  // Set the insertion point to the end of the Trampoline Block.
  Builder.SetInsertPoint(this->TrampBlock);

  // Set the name
  string Name = Arg->getName().str() + ".cast";

  // Create the cast instruction
  Value *CastArg = nullptr;
  if (Arg->getType()->isPointerTy()) {
    assert(TargetTy->isPointerTy() && "TargetTy expected to be a pointer.");
    CastArg = Builder.CreatePointerBitCastOrAddrSpaceCast(Arg, TargetTy, Name);
  } else if (Arg->getType()->isIntegerTy()) {
    assert(TargetTy->isIntegerTy() && "TargetTy expected to be integer.");
    CastArg = Builder.CreateZExtOrTrunc(Arg, TargetTy, Name);
  } else if (Arg->getType()->isFloatingPointTy()) {
    assert(TargetTy->isFloatingPointTy() && "TargetTy expected to be FPTy.");
    CastArg = Builder.CreateFPCast(Arg, TargetTy, Name);
  } else {
    llvm_unreachable("Unhandled scalar trampoline arg type.");
  }

  // Add the cast value to the map and the trampoline args
  this->CastMap[CastMapKey] = CastArg;
  return CastArg;
}

/// Adds a scalar arg (integer or floating point type) to the trampoline call.
///
/// This method will cast the provided Arg to the given TargetTy if needed.
void TrampolineBuilder::addScalarArg(Value *Arg, Type *TargetTy) {
  Value *CastArg = castArg(Arg, TargetTy);
  this->TrampolineArgs.push_back(CastArg);
}

/// Adds a pointer arg to the trampoline call.
///
/// This method will cast the provided Arg to the given TargetTy if needed.
void TrampolineBuilder::addPointerArg(Value *PtrArg, Type *TargetTy) {
  Value *CastArg = castArg(PtrArg, TargetTy);
  this->TrampolineArgs.push_back(CastArg);
}

/// Adds a SYCL wrapper class arg to the trampoline call.
///
/// A SYCL wrapper class is a struct that contains a SYCL Array, which is just a
/// pointer arg.
///
/// This method will cast the provided Arg to the given TargetTy if needed.
void TrampolineBuilder::addSyclWrapperArg(Value *Root, Type *TargetTy) {
  Value *SyclArray = getSyclArrayFromWrapper(Root);
  addPointerArg(SyclArray, TargetTy);
}

/// Adds a SYCL Range arg to the trampoline call.
///
/// A SYCL Range contains a number of elements defined by the shape of the sycl
/// construct. A dim must be provided to specify which dimension you want the
/// range for.
///
/// This method will cast the SYCL Range dim value into the TargetTy if  needed.
void TrampolineBuilder::addSyclRangeArg(Value *Root, Type *TargetTy,
                                        uint64_t Dim) {
  Value *RangeDim = getSyclRangeDim(Root, Dim);
  addScalarArg(RangeDim, TargetTy);
}

/// Add the stride argument to the memref.
///
/// The stride of a memref with shape <A, B, C> should be: [B*C, C, 1], here we
/// create the mul instructions necessary to construct the strides.
///
/// Casts the arguments into the required types if needed.
void TrampolineBuilder::addStrideArgs(Value *Root, unsigned Dims,
                                      Type *TargetTy) {
  Builder.SetInsertPoint(this->TrampBlock);
  SmallVector<Value *, 3> Strides;
  // Initialize Strides to the shape, offset by one
  for (unsigned Dim = 1; Dim < Dims; Dim++) {
    Value *TypeCorrected = castArg(getSyclRangeDim(Root, Dim), TargetTy);
    Strides.push_back(TypeCorrected);
  }
  assert(Strides.size() > 0 && "Expecting input to result in valid dims");
  // Create the multiplications required for higher dimension strides. The
  // stride of the last dimension does not need changing here.
  using Iterator = SmallVector<Value *, 3>::iterator;
  for (Iterator ToUpdate = Strides.begin(), End = Strides.end();
       ToUpdate != End - 1; ++ToUpdate) {
    Value *NewStride = *ToUpdate;
    for (Iterator ToMult = ToUpdate + 1; ToMult != End; ++ToMult) {
      LLVM_DEBUG(dbgs() << "Creating Mul of: "; NewStride->dump();
                 dbgs() << "\tand "; (*ToMult)->dump());
      NewStride = Builder.CreateNUWMul(NewStride, *ToMult, "stride");
    }
    *ToUpdate = NewStride;
  }
  // Last dimension should always have a stride of 1.
  assert(isa<IntegerType>(TargetTy) &&
         "Expecting stride arguments to be integer types");
  Strides.push_back(ConstantInt::get(TargetTy, 1));
  // Finally put the arguments in.
  for (Value *Stride : Strides)
    addScalarArg(Stride, TargetTy);
}

/// Adds a SYCL ID arg to the trampoline call.
///
/// A SYCL ID contains a number of elements defined by the offsets of the sycl
/// construct. A dim must be provided to specify which dimension you want the
/// offset for.
///
/// This method will cast the SYCL ID offset value into the TargetTy if needed.
void TrampolineBuilder::addSyclIDArg(Value *Root, Type *TargetTy,
                                     uint64_t Dim) {
  Value *SyclID = getSyclID(Root, Dim);
  addScalarArg(SyclID, TargetTy);
}

/// Add a single sycl offset arg to the trampoline call.
///
/// SYCL provides an offset for each dimension, however MLIR currently only
/// allows for 1 offset. Will need to compute a single offset from the multiple
/// offsets and the range. Ex)
///   Ranges: A, B, C
///   IDs:    X, Y, Z
///          single offset: B*C*X + C*Y + Z
/// Essentially we will be striding the offset.
///
/// The Rank must be provided so we know how many elements are in the ID and
/// Range pointers.
///
/// This method will cast the resulting offset value into the TargetTy if
/// needed.
void TrampolineBuilder::addSingleSyclIDArg(Value *IDRoot, Value *RangeRoot,
                                           Type *TargetTy, int64_t Rank) {
  assert(Rank > 0 && "Rank of offsets and ranges should be 1 or more.");

  // Set the insertion point to the end of the Trampoline Block.
  Builder.SetInsertPoint(this->TrampBlock);

  // Create the single offset by multiplying and adding Sycl Ranges and IDs
  Value *Offset = getSyclID(IDRoot, Rank - 1);
  for (int64_t i = 0; i < Rank - 1; i++) {
    Value *IDDim = getSyclID(IDRoot, i);
    for (int64_t j = i + 1; j < Rank; j++) {
      Value *RangeDim = getSyclRangeDim(RangeRoot, j);
      IDDim = Builder.CreateMul(IDDim, RangeDim);
    }
    Offset = Builder.CreateAdd(Offset, IDDim);
  }

  // Set the name to something identifiable.
  string Name = IDRoot->getName().str() + ".single.id";
  Offset->setName(Name);

  // Add the offset to the args as a scalar, this will cast the Value if needed.
  addScalarArg(Offset, TargetTy);
}

/// Adds a SYCL Dim arg to the trampoline call.
///
/// This method will cast the Dim value into the TargetTy if needed.
void TrampolineBuilder::addSyclDimArg(Value *Root, Type *TargetTy) {
  Value *Dim = getSyclDim(Root);
  addScalarArg(Dim, TargetTy);
}

/// Set the name of the kernel that will be called by the function.
void TrampolineBuilder::setKernelName(std::string KernelName) {
  this->KernelName = KernelName;
}

/// Get the name of the kernel that will be called by the function.
std::string TrampolineBuilder::getKernelName() const {
  return this->KernelName;
}

/// Finalize the creation of the trampoline function.
///
/// This method will erase all of the blocks inside the function provided at
/// initialization except for the trampoline block created by this builder which
/// houses all of the argument casting and loading required for the trampoline.
/// It will then create the trampoline call based on the provided arguments and
/// the kernel name.
///
/// A kernel name must have been set before calling this method.
void TrampolineBuilder::finalize() {
  assert(!this->KernelName.empty() && "Kernel must have a name.");
  assert(this->F && "Target function must be defined.");
  assert(this->TrampBlock && "Trampoline block must exist.");

  // Set the insertion point to the end of the Trampoline Block.
  Builder.SetInsertPoint(this->TrampBlock);

  // Remove all BB from function, except for the trampoline block.
  std::vector<BasicBlock *> DelBBList;
  for (BasicBlock &BB : *this->F) {
    if (&BB == this->TrampBlock)
      continue;
    auto I = BB.getTerminator();
    I->replaceAllUsesWith(UndefValue::get(I->getType()));
    DelBBList.push_back(&BB);
  }
  DeleteDeadBlocks(DelBBList);

  // Get the trampoline arg types
  SmallVector<Type *> ArgTys = {};
  for (Value *Arg : this->TrampolineArgs)
    ArgTys.push_back(Arg->getType());
  // Create the trampoline function
  FunctionType *FnTy = FunctionType::get(Builder.getVoidTy(), ArgTys, false);
  GlobalValue::LinkageTypes Linkage = Function::ExternalLinkage;
  Module *M = this->F->getParent();
  this->TrampolineFunc = Function::Create(FnTy, Linkage, this->KernelName, *M);
  // Add a call to the trampoline function
  Builder.CreateCall(FunctionCallee(this->TrampolineFunc),
                     this->TrampolineArgs);
  // Add a return void
  Builder.CreateRetVoid();
}
