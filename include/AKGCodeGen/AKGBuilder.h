//===-- AKGBuilder.h - AKG Builder Declarations ---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef SYCLOPS_INCLUDE_AKGCODEGEN_AKGBUILDER_H
#define SYCLOPS_INCLUDE_AKGCODEGEN_AKGBUILDER_H

#include "llvm/IR/ValueMap.h"
#include "llvm/Support/StringSaver.h"
#include <string>

namespace llvm {
class Value;
class Loop;
namespace converter {

namespace akg_impl {
struct Object {
  // For LLVM RTTI setup (dyn_cast<> etc.)
  enum ObjectKind { O_Statement, O_Block, O_For, O_IfElse };

private:
  // For LLVM RTTI setup (dyn_cast<> etc.)
  ObjectKind Kind;

public:
  // For LLVM RTTI setup (dyn_cast<> etc.)
  explicit Object(ObjectKind K);
  virtual ~Object();
  ObjectKind getKind() const;

  Object *Parent;

  // Used for debugging correctness of control flow
  virtual void printStructure(std::string &Output,
                              unsigned Indent = 0) const = 0;
  // Used to generate the code output once the tree is constructed
  virtual void genCode(std::string &CodeOutput, unsigned Indent = 0) const = 0;
};

struct Statement : public Object {
  Statement *LHS;
  Statement *RHS;
  SmallVector<Statement *> Subscripts;
  bool Bracketed;

  Statement();
  ~Statement() override;
  // Note: the input Str should be a stringref that is returned by StringSaver.
  explicit Statement(StringRef Str);
  static bool classof(const Object *O);
  void setContent(StringRef Str);

  std::string toString() const;
  // these are equivalent in the case of statement
  void printStructure(std::string &Output, unsigned Indent = 0) const override;
  void genCode(std::string &CodeOutput, unsigned Indent = 0) const override;

private:
  StringRef Content;
  void appendToString(std::string &Output) const;
};

struct Block : public Object {
  SmallVector<Object *> Contents;

  Block();
  ~Block() override;
  static bool classof(const Object *O);

  void printStructure(std::string &Output, unsigned Indent = 0) const override;
  void genCode(std::string &CodeOutput, unsigned Indent = 0) const override;
};

struct ForLoop : public Object {

  ForLoop();
  ~ForLoop() override;
  static bool classof(const Object *O);

  void setBody(Object *O);
  // The Condition should be in the form of `i0, 0, 16`
  // Note: the input Str should be a stringref that is returned by StringSaver.
  void setCondition(StringRef Cond);
  void printStructure(std::string &Output, unsigned Indent = 0) const override;
  void genCode(std::string &CodeOutput, unsigned Indent = 0) const override;

private:
  Object *Body;
  StringRef Condition;
};

struct IfElse : public Object {

  IfElse();
  ~IfElse() override;
  static bool classof(const Object *O);

  void setCondition(Statement *Cond);
  void setIfBody(Object *Obj);
  void setElseBody(Object *Obj);
  void printStructure(std::string &Output, unsigned Indent = 0) const override;
  void genCode(std::string &CodeOutput, unsigned Indent = 0) const override;

private:
  Statement *Condition;
  Object *IfBody;
  Object *ElseBody;
};
} // namespace akg_impl

using namespace akg_impl;

class AKGBuilder {
private:
  // String saver for storage of variables and operands etc.
  BumpPtrAllocator SaverAllocator;
  UniqueStringSaver StrSaver;

  // Statements that doesn't necessarily have a value associated with them
  std::vector<Statement *> Statements;
  ValueMap<const Value *, Block> BBMap;
  // For loops can have more than one loops associated with a single variable...
  ValueMap<const Value *, SmallVector<ForLoop *>> ForMap;
  ValueMap<const Value *, IfElse> IfElseMap;

  Block *BlockToInsert;

  Block RootBlock;

public:
  AKGBuilder();
  ~AKGBuilder();

  // AKG objects creation methods
  Statement *getStmt(StringRef Str);
  Statement *concatStmt(Statement *LHS, Statement *RHS,
                        const std::string &Op = "");
  Block *getRootBlock();
  // Blocks are typically keyed by the BasicBlock* they are generated from. Can
  // also be Function* or IntrInsts*
  Block *getBlock(const Value *V);
  // ForLoops are typically keyed by the loop headers BasicBlock*s
  ForLoop *getForLoop(const Value *V, unsigned Variant = 0);
  // IfElse's are keyed by the branch instruction that caused them.
  IfElse *getIfElse(const Value *V);

  // Instruction creation methods
  void setInsertBlock(Block *B);
  void setInsertBlock(const Value *BB);
  void appendStmt(const std::string &Str);
  void prependStmt(const std::string &Str);
  void append(Object *O);
  void prepend(Object *O);
  ForLoop *createForLoop(const Value *Key, StringRef Cond, Object *Body);

  void clear();
  StringRef getStrRef(std::string Str);
  void dumpToString(std::string &CodeOutput);
};

} // namespace converter
} // namespace llvm

#endif // SYCLOPS_INCLUDE_AKGCODEGEN_AKGBUILDER_H
