//===-- ConverterUtil.h - Utility Functions and Classes -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef SYCLOPS_INCLUDE_UTIL_CONVERTERUTIL_H
#define SYCLOPS_INCLUDE_UTIL_CONVERTERUTIL_H

#include "llvm/IR/PassManager.h"
#include <cstdint>
#include <string>

namespace llvm {
class BinaryOperator;
class BranchInst;
class ICmpInst;
namespace converter {

// =============================================================================
// Utility Functions
// =============================================================================
extern ExitOnError ExitOnErr;

// Error handling methods
void terminate(const std::string &Reason);

template <class T>
T checkError(Expected<T> &&E) {
  return ExitOnErr(std::move(E));
}
void checkError(Error &&E);
Error createError(std::string EM, std::error_code EC = std::error_code());
Error createError(Error &OtherError, std::string EM,
                  std::error_code EC = std::error_code());

Error rejectUnsupportedBuiltin(const Function *F);

void getGlobalUsersInFunction(const GlobalValue *GV, const Function *F,
                              SmallVector<const Instruction *> &GIDUsrs);

Value *tracePastCastInsts(Value *V);

Value *tracePastCastAndGEP(Value *V);

unsigned unwrapStructs(const Type *&Ty);

Type *getPointerElementType(Value *V);

// =============================================================================
// Utility classes
// =============================================================================

/// Base class for data types/shapes, delinearizing etc. All objects of Shape
/// and subclasses of Shape should be created with std::shared_ptr.
class Shape {
public:
  typedef enum {
    Unset = 0,
    Input = 1 << 0,
    Output = 1 << 1,
    Local = 1 << 2,
    Global = 1 << 3,
    Constant = 1 << 4,
    Index = 1 << 5,
    Offset = 1 << 6,
    Range = 1 << 7
  } ShapeType;

private:
  std::string Name;
  // base element type of the shape
  Type *ElementType;
  // class of shape
  int32_t ShapeClass;
  // the value from which the shape is derived from
  Value *Root;
  // size of each dimension of the shape
  SmallVector<int> Dims;
  // the corresponding index of GEP for each dimension
  SmallVector<unsigned> GEPIndices;
  // the dimension of this shape
  unsigned NumDims;
  // the address space in which the shape is stored
  unsigned AddrSpace;
  // whether or not this shape is dynamically sized.
  bool IsDynamic;
  // If the root value is a simple wrapper
  bool IsSimpleWrapper;
  Shape *OffsetShape;
  Shape *RangeShape;
  Shape *BaseShape;

public:
  Shape();

  void setName(StringRef N);
  std::string getName() const;
  void setElementType(Type *Ty);
  Type *getElementType() const;
  void setShapeType(int32_t ST);
  int32_t getShapeType() const;
  void setRoot(Value *Val);
  Value *getRoot() const;
  void setAddrSpace(unsigned AS);
  unsigned getAddrSpace() const;
  void addDim(int D, unsigned Index);
  void setNumDims(unsigned ND);
  unsigned getNumDims() const;
  unsigned getDim(unsigned D) const;
  unsigned getGEPIdx(unsigned D) const;
  bool isDynamic() const;
  void setDynamic(bool IsDynamic);
  bool isSimpleWrapper() const;
  void setSimpleWrapper(bool IsSimpleWrapper);
  // In case there is an accessor for this shape
  // Accessor arguments are ordered: <Ptr> <MemRange> <AccessRange> <Offset>
  Shape *getOffset() const;
  Shape *getRange() const;
  void setOffset(Shape *OS);
  void setRange(Shape *RS);
  // Or if this is an accessor component, link back to original shape
  Shape *getBase() const;
  void setBase(Shape *BS);
};

// LoopComponents stores the necessary information regarding loops from LLVM IR.
struct LoopComponents {
  PHINode *IV;
  Value *Start;
  Value *Bound;
  Value *Step;
  const ICmpInst *Condition;
  bool IsDynamic;
};

} // namespace converter
} // namespace llvm

#endif // SYCLOPS_INCLUDE_UTIL_CONVERTERUTIL_H
