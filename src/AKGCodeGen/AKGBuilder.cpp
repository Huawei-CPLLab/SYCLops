//===-- AKGBuilder.cpp - AKG Builder Definitions ----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "AKGCodeGen/AKGBuilder.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include <string>

#define DEBUG_TYPE "akg-converter"

using std::string;
namespace llvm {
namespace converter {

namespace akg_impl {

Object::Object(ObjectKind K) : Kind(K), Parent(nullptr) {}

Object::~Object() {}

Object::ObjectKind Object::getKind() const { return this->Kind; }

Statement::Statement()
    : Object(O_Statement), LHS(nullptr), RHS(nullptr), Bracketed(false),
      Content() {}

Statement::~Statement() {}

Statement::Statement(StringRef Str)
    : Object(O_Statement), LHS(nullptr), RHS(nullptr), Bracketed(false),
      Content(Str) {}

bool Statement::classof(const Object *O) { return O->getKind() == O_Statement; }

/// Note: the input Str should be a stringref that is returned by StringSaver.
void Statement::setContent(StringRef Str) { Content = Str; }

string Statement::toString() const {
  string RetVal;
  genCode(RetVal);
  return RetVal;
}

void Statement::printStructure(string &Output, unsigned Indent) const {
  genCode(Output, Indent);
}

void Statement::genCode(string &CodeOutput, unsigned Indent) const {
  CodeOutput.append(Indent, '\t');
  this->appendToString(CodeOutput);
}

void Statement::appendToString(string &Output) const {
  if (Bracketed)
    Output += "(";
  if (LHS)
    LHS->appendToString(Output);
  if (!Content.empty())
    Output += Content.str();
  if (RHS)
    RHS->appendToString(Output);
  for (Statement *Subscript : Subscripts) {
    Output += "[";
    Subscript->appendToString(Output);
    Output += "]";
  }
  if (Bracketed)
    Output += ")";
}

Block::Block() : Object(O_Block) {}

Block::~Block() {}

bool Block::classof(const Object *O) { return O->getKind() == O_Block; }

void Block::printStructure(string &Output, unsigned Indent) const {
  Output.append(Indent, '\t');
  Output += "Block{\n";
  for (Object *Obj : Contents)
    Obj->printStructure(Output, Indent + 1);
  Output.append(Indent, '\t');
  Output += "}\n";
}

void Block::genCode(string &CodeOutput, unsigned Indent) const {
  for (Object *Obj : Contents) {
    Obj->genCode(CodeOutput, Indent);
    if (isa<Statement>(Obj))
      CodeOutput += "\n";
  }
}

ForLoop::ForLoop() : Object(O_For), Body(nullptr), Condition() {}

ForLoop::~ForLoop() {}

bool ForLoop::classof(const Object *O) { return O->getKind() == O_For; }

void ForLoop::setBody(Object *O) {
  this->Body = O;
  O->Parent = this;
}

void ForLoop::setCondition(StringRef Cond) { Condition = Cond; }

void ForLoop::printStructure(string &Output, unsigned Indent) const {
  Output.append(Indent, '\t');
  Output += "for (";
  Output += Condition;
  Output += ") {\n";
  Body->printStructure(Output, Indent + 1);
  Output.append(Indent, '\t');
  Output += "}\n";
}

void ForLoop::genCode(string &CodeOutput, unsigned Indent) const {
  CodeOutput.append(Indent, '\t');
  CodeOutput += "for (";
  CodeOutput += Condition;
  CodeOutput += ") {\n";
  Body->genCode(CodeOutput, Indent + 1);
  CodeOutput.append(Indent, '\t');
  CodeOutput += "}\n";
}

IfElse::IfElse()
    : Object(O_IfElse), Condition(nullptr), IfBody(nullptr), ElseBody(nullptr) {
}

IfElse::~IfElse() {}

bool IfElse::classof(const Object *O) { return O->getKind() == O_IfElse; }

void IfElse::setCondition(Statement *Cond) {
  Cond->Parent = this;
  this->Condition = Cond;
}

void IfElse::setIfBody(Object *Obj) {
  this->IfBody = Obj;
  Obj->Parent = this;
}

void IfElse::setElseBody(Object *Obj) {
  this->ElseBody = Obj;
  Obj->Parent = this;
}

void IfElse::printStructure(string &Output, unsigned Indent) const {
  assert(IfBody && "IfBody not assigned!");
  Output.append(Indent, '\t');
  Output += "if (";
  Condition->genCode(Output);
  Output += ") {\n";
  IfBody->printStructure(Output, Indent + 1);
  Output.append(Indent, '\t');
  Output += "}\n";

  if (ElseBody) {
    Output.append(Indent, '\t');
    Output += "else {\n";
    ElseBody->printStructure(Output, Indent + 1);
    Output.append(Indent, '\t');
    Output += "}\n";
  }
}

void IfElse::genCode(string &CodeOutput, unsigned Indent) const {
  assert(IfBody && "IfBody not assigned!");
  CodeOutput.append(Indent, '\t');
  CodeOutput += "if(";
  Condition->genCode(CodeOutput);
  CodeOutput += "){\n";
  IfBody->genCode(CodeOutput, Indent + 1);
  CodeOutput.append(Indent, '\t');
  CodeOutput += "}\n";

  if (ElseBody) {
    CodeOutput.append(Indent, '\t');
    CodeOutput += "else {\n";
    ElseBody->genCode(CodeOutput, Indent + 1);
    CodeOutput.append(Indent, '\t');
    CodeOutput += "}\n";
  }
}

} // namespace akg_impl

using namespace akg_impl;
// =============================================================================
// AKGBuilder Methods ==========================================================
// =============================================================================

AKGBuilder::AKGBuilder() : SaverAllocator(), StrSaver(SaverAllocator) {}
AKGBuilder::~AKGBuilder() {
  for (auto *Ptr : Statements)
    delete Ptr;

  for (auto Pair : ForMap) {
    for (auto *Ptr : Pair.second)
      delete Ptr;
  }
}

// -------------------------------------------------AKG objects creation methods

Statement *AKGBuilder::getStmt(StringRef S) {
  StringRef Str = StrSaver.save(S);
  auto *RetVal = new Statement(Str);
  Statements.push_back(RetVal);
  return RetVal;
}

/// Concatenates two statements together and stores the resulting statement
Statement *AKGBuilder::concatStmt(Statement *LHS, Statement *RHS,
                                  const string &Op) {
  Statement *RetVal = getStmt("");
  RetVal->LHS = LHS;
  RetVal->RHS = RHS;
  if (!Op.empty()) {
    StringRef SavedOp = StrSaver.save(Op);
    RetVal->setContent(SavedOp);
  }
  return RetVal;
}

Block *AKGBuilder::getRootBlock() { return &RootBlock; }

/// Queries BBMap from a Value to see if exists, if not then create and insert
/// into BBMap, then return the pointer to the Block object
Block *AKGBuilder::getBlock(const Value *V) {
  auto It = BBMap.find(V);
  if (It != BBMap.end())
    return &It->second;

  auto InsResult = BBMap.insert(std::make_pair(V, Block()));
  assert(InsResult.second && "BBMap insertion failed");
  return &InsResult.first->second;
}

/// Queries ForMap from a Value to see if exists, if not then create and insert
/// into ForMap, then return the pointer to the Block object
ForLoop *AKGBuilder::getForLoop(const Value *V, unsigned Variant) {
  auto It = ForMap.find(V);
  if (It == ForMap.end()) {
    auto InsResult = ForMap.insert(std::make_pair(V, SmallVector<ForLoop *>()));
    assert(InsResult.second && "ForMap insertion failed");
    It = InsResult.first;
  }

  auto &Variants = It->second;
  assert(Variant <= Variants.size());
  if (Variant == Variants.size()) {
    auto *RetVal = new ForLoop();
    Variants.push_back(RetVal);
    return RetVal;
  }
  return Variants[Variant];
}

/// Queries IfElseMap from a Value to see if exists, if not then create and
/// insert into IfElseMap, then return the pointer to the Block object
IfElse *AKGBuilder::getIfElse(const Value *V) {
  auto It = IfElseMap.find(V);
  if (It != IfElseMap.end())
    return &It->second;

  auto InsResult = IfElseMap.insert(std::make_pair(V, IfElse()));
  assert(InsResult.second && "IfElseMap insertion failed");
  return &InsResult.first->second;
}

// -------------------------------------------------Instruction Creation Methods

/// Analogous to llvm::IRBuilder<>::setInsertPoint(BB), except it only points to
/// the Block itself, using appendStmt/prependStmt to insert
void AKGBuilder::setInsertBlock(Block *B) { BlockToInsert = B; }

void AKGBuilder::setInsertBlock(const Value *BB) {
  BlockToInsert = getBlock(BB);
}

void AKGBuilder::appendStmt(const string &Str) {
  auto *S = getStmt(Str);
  BlockToInsert->Contents.push_back(S);
}
void AKGBuilder::prependStmt(const string &Str) {
  auto *S = getStmt(Str);
  auto &Contents = BlockToInsert->Contents;
  Contents.insert(Contents.begin(), S);
}

void AKGBuilder::append(Object *O) {
  BlockToInsert->Contents.push_back(O);
  O->Parent = BlockToInsert;
}

void AKGBuilder::prepend(Object *O) {
  auto &Contents = BlockToInsert->Contents;
  Contents.insert(Contents.begin(), O);
  O->Parent = BlockToInsert;
}

ForLoop *AKGBuilder::createForLoop(const Value *Key, StringRef Cond,
                                   Object *Body) {
  auto CondStr = StrSaver.save(Cond);
  auto *FL = getForLoop(Key);
  FL->setBody(Body);
  FL->setCondition(CondStr);
  return FL;
}

// -------------------------------------------------Instruction Creation Methods

void AKGBuilder::clear() {
  for (auto *Ptr : Statements)
    delete Ptr;

  for (auto Pair : ForMap) {
    for (auto *Ptr : Pair.second)
      delete Ptr;
  }
  Statements.clear();
  BBMap.clear();
  ForMap.clear();
  IfElseMap.clear();
  RootBlock.Contents.clear();
}

StringRef AKGBuilder::getStrRef(std::string Str) { return StrSaver.save(Str); }

void AKGBuilder::dumpToString(string &CodeOutput) {
  RootBlock.genCode(CodeOutput);
}

} // namespace converter
} // namespace llvm
