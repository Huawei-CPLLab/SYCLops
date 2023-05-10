//===-- ConverterUtil.cpp - Utility Functions and Classes -------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Util/ConverterUtil.h"
#include "Util/Matcher.h"
#include "ConverterCodeGen.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Operator.h"

#define DEBUG_TYPE "converter"

namespace llvm {
namespace converter {

ExitOnError ExitOnErr("Error: ");
void terminate(const std::string &Reason) { ExitOnErr(createError(Reason)); }
void checkError(Error &&E) { ExitOnErr(std::move(E)); }

/// Helper methods for simplify creating Errors used throughout the codegen.
Error createError(std::string EM, std::error_code EC) {
  return make_error<StringError>(EM, EC);
}

Error createError(Error &OtherError, std::string EM, std::error_code EC) {
  return joinErrors(std::move(OtherError), make_error<StringError>(EM, EC));
}

/// Checks the function for sycl builtin function calls that we cannot handle,
/// and emit the corresponding error message.
Error rejectUnsupportedBuiltin(const Function *F) {
  StringSet<> SupportedSyclFuncs =
#include "Util/SupportedSyclFuncs.inc"
      ;

  for (auto &BB : *F) {
    for (const Instruction &I : BB) {
      auto *CI = dyn_cast<CallInst>(&I);
      if (!CI)
        continue;
      const Function *CalledFunc = CI->getCalledFunction();
      if (!CalledFunc)
        return createError("Cannot support indirect function calls");
      // If called function is defined in the module, let it go through
      if (!CalledFunc->empty())
        continue;
      // We also don't care about intrinsic functions
      if (CalledFunc->isIntrinsic())
        continue;

      StringRef FuncName = CalledFunc->getName();
      // Demangle function name to get the builtin function, if it is one.
      // Currently we only support functions with mangled names
      if (!FuncName.consume_front("_Z"))
        return createError("Unsupported function: " + FuncName.str());
      // Demangle function name:
      int NameLen;
      if (FuncName.consumeInteger(10, NameLen))
        return createError("Unsupported function: " + FuncName.str());
      FuncName = FuncName.slice(0, NameLen);
      if (!FuncName.consume_front("__spirv_ocl_"))
        return createError("Unsupported function: " + FuncName.str());
      // By here we would have dropped the prefix of the function "spirv_ocl_"
      if (!SupportedSyclFuncs.count(FuncName))
        return createError("Unsupported function: " + FuncName.str());
    } // End iterating through I in BB
  }   // End iterating through BB in F
  return Error::success();
}

/// Recursive helper of getGlobalUsersInFunction.
static void getInstFromUsr(const Value *V, const Function *F,
                           SmallVector<const Instruction *> &GIDUsrs) {
  // Since users might be Operators instead of Instructions, this method will be
  // recursive and get the users of the operators as well.
  if (const auto *Inst = dyn_cast<Instruction>(V)) {
    if (Inst->getParent()->getParent() == F)
      GIDUsrs.push_back(Inst);
    return;
  }

  for (const User *Usr : V->users()) {
    getInstFromUsr(Usr, F, GIDUsrs);
  }
}

/// Finds a user of a given global value in a function. Will only return the
/// first user in the function that it finds, but they should only have one use
/// per function anyway.
void getGlobalUsersInFunction(const GlobalValue *GV, const Function *F,
                              SmallVector<const Instruction *> &GIDUsrs) {
  getInstFromUsr(GV, F, GIDUsrs);
}

/// Some cast instructions might get get inlined as operators, hence we have
/// this method to check for those as well.
Value *tracePastCastInsts(Value *V) {
  while (1) {
    if (auto *CI = dyn_cast<CastInst>(V))
      V = CI->getOperand(0);
    else if (auto *ASCastOp = dyn_cast<AddrSpaceCastOperator>(V))
      V = ASCastOp->getOperand(0);
    else if (auto *BCOp = dyn_cast<BitCastOperator>(V))
      V = BCOp->getOperand(0);
    else if (auto *P2IOp = dyn_cast<PtrToIntOperator>(V))
      V = P2IOp->getOperand(0);
    else if (auto *ZExtOp = dyn_cast<ZExtOperator>(V))
      V = ZExtOp->getOperand(0);
    else {
      // if V is no longer a casting operation/instruction, return.
      return V;
    }
  }
}

/// Same as above, but also trace past GEP instructions and operators.
Value *tracePastCastAndGEP(Value *V) {
  while (1) {
    V = tracePastCastInsts(V);
    if (auto *GEPOp = dyn_cast<GEPOperator>(V))
      V = GEPOp->getPointerOperand();
    else if (auto *GEP = dyn_cast<GetElementPtrInst>(V))
      V = GEP->getPointerOperand();
    else
      return V;
  }
}

/// Tries to unwrap a struct type if it only contains a single element type.
/// Returns the levels of nesting traversed, and updates Ty to be the unwrapped
/// type.
/// e.g. Ty = { { { {float*, [2xi64]} } } } will return 3 and update Ty =
/// {float*, [2xi64]}.
unsigned unwrapStructs(const Type *&Ty) {
  unsigned Levels = 0;
  while (Ty->isStructTy() && Ty->getStructNumElements() == 1) {
    Levels++;
    Ty = Ty->getStructElementType(0);
  }
  return Levels;
}

static Value *traceUseForPointer(Value *V) {
  User *Usr = *V->user_begin();
  // return on GEP/load/store
  if (isa<GetElementPtrInst>(Usr) || isa<LoadInst>(Usr) || isa<StoreInst>(Usr))
    return Usr;
  if (isa<CastInst>(Usr))
    return traceUseForPointer(Usr);
  llvm_unreachable("Unexpected use of pointer");
}

/// Tries to infer the pointer element type from the opaque pointer.
Type *getPointerElementType(Value *V) {
  assert(isa<PointerType>(V->getType()) &&
         "Expecting input to be a pointer type");
  // Tracing the def chain from v:
  Value *Def = tracePastCastInsts(V);
  if (auto *Arg = dyn_cast<Argument>(Def))
    if (auto *ElType = Arg->getPointeeInMemoryValueType())
      return ElType;
  if (auto *AI = dyn_cast<AllocaInst>(Def))
    return AI->getAllocatedType();
  if (auto *GV = dyn_cast<GlobalValue>(Def))
    return GV->getValueType();

  // Tracing the use chain from V
  V = traceUseForPointer(V);
  if (auto *GEP = dyn_cast<GetElementPtrInst>(V))
    return GEP->getSourceElementType();
  if (auto *LI = dyn_cast<LoadInst>(V))
    return LI->getType();
  if (auto *SI = dyn_cast<StoreInst>(V))
    return SI->getValueOperand()->getType();
  llvm_unreachable("Unhandled instruction for getPointerElementType()");
}

// Shape Methods ===============================================================

Shape::Shape()
    : ElementType(nullptr), ShapeClass(Unset), NumDims(0), IsDynamic(false),
      IsSimpleWrapper(false), OffsetShape(nullptr), RangeShape(nullptr),
      BaseShape(nullptr) {}

void Shape::setName(StringRef N) { this->Name = N.str(); }

std::string Shape::getName() const { return Name; }

void Shape::setElementType(Type *Ty) { ElementType = Ty; }

Type *Shape::getElementType() const { return ElementType; }

void Shape::setShapeType(int32_t ST) { ShapeClass = ST; }

int32_t Shape::getShapeType() const { return ShapeClass; }

void Shape::setRoot(Value *Val) { Root = Val; }

Value *Shape::getRoot() const { return Root; }

void Shape::setAddrSpace(unsigned AS) { AddrSpace = AS; }

unsigned Shape::getAddrSpace() const { return AddrSpace; }

void Shape::addDim(int D, unsigned Index) {
  Dims.push_back(D);
  if (Index > 1)
    IsDynamic = false;
  GEPIndices.push_back(Index);
  NumDims++;
}

void Shape::setNumDims(unsigned ND) { NumDims = ND; }

unsigned Shape::getNumDims() const { return NumDims; }

unsigned Shape::getDim(unsigned D) const { return Dims[D]; }

unsigned Shape::getGEPIdx(unsigned D) const { return GEPIndices[D]; }

bool Shape::isDynamic() const { return IsDynamic; }

void Shape::setDynamic(bool IsDynamic) { this->IsDynamic = IsDynamic; }

bool Shape::isSimpleWrapper() const { return IsSimpleWrapper; }

void Shape::setSimpleWrapper(bool IsSimpleWrapper) {
  this->IsSimpleWrapper = IsSimpleWrapper;
}

Shape *Shape::getOffset() const { return OffsetShape; }

Shape *Shape::getRange() const { return RangeShape; }

Shape *Shape::getBase() const { return BaseShape; }

void Shape::setOffset(Shape *OS) { OffsetShape = OS; }

void Shape::setRange(Shape *RS) { RangeShape = RS; }

void Shape::setBase(Shape *BS) { BaseShape = BS; }

} // namespace converter
} // namespace llvm
