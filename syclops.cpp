//===-- syclops.cpp - The SYCLops Converter ---------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// SYCLops is a converter capable of taking LLVM IR, generated from oneAPI's
// SYCL Device Front-end Compiler, and raising it into a target backend.
//
// A notable supported backend is the MLIR (affine dialect) backend.
//
// For more information, SYCLops was described in a paper presented at the
// IWOCL'22 conference: https://doi.org/10.1145/3529538.3529992
//
//===----------------------------------------------------------------------===//

#include "AKGCodeGen/AKGCodeGen.h"
#include "ConverterCodeGen.h"
#include "MLIRCodeGen/MLIRCodeGen.h"
#include "Util/ConverterUtil.h"
#include "llvm/Bitcode/BitcodeWriter.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/SourceMgr.h"

using namespace llvm;
using namespace llvm::converter;
using std::error_code;
using std::string;

#define DEBUG_TYPE "syclops"

static cl::OptionCategory SYCLopsCat{"SYCLops options"};

// InputFilename - The filename to read from.
static cl::opt<string> InputFilename{cl::Positional,
                                     cl::desc("<input bitcode file>"),
                                     cl::init(""), cl::value_desc("filename")};
// The converter output file.
static cl::opt<string> OutputFilename{"o", cl::desc("Output filename"),
                                      cl::value_desc("filename"), cl::init(""),
                                      cl::cat(SYCLopsCat)};
// File that will house the trampoline function bitcode (.bc).
static cl::opt<string> TrampolineFilename{
    "otf", cl::desc{"Output trampoline filename"}, cl::value_desc("filename"),
    cl::init(""), cl::cat(SYCLopsCat)};
// Flag to emit AKG IR
cl::opt<bool> emit_akg{"emit-akg", cl::desc("Convert to AKG IR."),
                       cl::init(false), cl::cat(SYCLopsCat)};
// Flag to emit MLIR
cl::opt<bool> emit_mlir{"emit-mlir", cl::desc("Convert to MLIR."),
                        cl::init(false), cl::cat(SYCLopsCat)};
// Flag to emit trampoline functions
cl::opt<bool> emit_trampoline{"emit-trampoline",
                              cl::desc("Create trampoline function calls."),
                              cl::init(false), cl::cat(SYCLopsCat)};
// Flag to run MLIR in testing mode
cl::opt<bool> MLIRConverterRunTest{"mlir-converter-test",
                                   cl::desc("MLIR converter's testing flag."),
                                   cl::init(false), cl::cat(SYCLopsCat)};
// Flag to declare the bitwidth of the MLIR IndexType
cl::opt<unsigned> mlirIndexBitwdith{
    "mlir-index-bitwidth",
    cl::desc(
        "Bitwidth of the index type in MLIR, 0 to use size of machine word."),
    cl::init(0), cl::cat(SYCLopsCat)};

/// Method that checks the command line options to make sure that the converter
/// has all the information it needs.
static Error checkCommandLineOptions() {
  // If neither conversion target is selected, set to AKG.
  if (!emit_akg && !emit_mlir)
    emit_akg.setValue<bool>(true);
  // If both conversion targets are selected, set to only AKG.
  if (emit_akg && emit_mlir)
    emit_mlir.setValue<bool>(false);
  // Handle if the input is empty.
  if (InputFilename.empty())
    return createError("Input file not specified.");
  // Make sure the output filename is specified.
  if (OutputFilename.empty())
    return createError("Output file not specified.");
  // If we are emitting trampoline functions, make sure the filename is
  // specified
  if (emit_trampoline && TrampolineFilename.empty())
    return createError("Trampoline file not specified.");

  return Error::success();
}

/// Method that collects all of the functions that will need to be converted.
static Error populateFuncList(Module *M, SmallVector<Function *> &FuncList) {
  SmallVector<Function *> SpirFuncs;
  for (auto &F : *M) {
    if (F.isDeclaration())
      continue;
    if (F.getName().contains("16AssertInfoCopier"))
      continue;
    if (F.getCallingConv() == CallingConv::SPIR_KERNEL)
      FuncList.push_back(&F);
    else
      SpirFuncs.push_back(&F);
  }
  // append SpirFuncs to back of FuncList, so that spir_kernels are converter
  // first
  FuncList.append(SpirFuncs);

  return Error::success();
}

/// Method for converting the functions in FuncList into AKG IR.
static Error convertAKG(LLVMContext &Context,
                        SmallVector<Function *> &FuncList) {
  AKGCodeGen ACG(Context);
  // Clear the output file before converting functions. The ACG.writeToFile
  // method will always append.
  sys::fs::remove(OutputFilename);
  // Run the converter on each function in the FuncList.
  for (auto *&F : FuncList) {
    LLVM_DEBUG(
        dbgs()
        << "Converting Function=========================================="
           "===========================================================\n");

    ACG.reset();
    if (Error E = ACG.convert(F))
      return createError(E, "AKG converter failed.");
    if (Error E = ACG.writeToFile(OutputFilename.getValue()))
      return createError(E, "AKG converter failed to write to file.");
    LLVM_DEBUG(dbgs() << "AKG Converter Succeeded\n");
  }

  return Error::success();
}

/// Method for converting the functions in FuncList into MLIR.
static Error convertMLIR(Module *M, SmallVector<Function *> &FuncList) {
  MLIRCodeGen MCG(*M, mlirIndexBitwdith);
  // Run the converter on each function in the FuncList.
  for (auto *&F : FuncList) {
    LLVM_DEBUG(
        dbgs()
        << "Converting Function=========================================="
           "===========================================================\n");
    if (Error E = MCG.convert(F))
      return createError(E, "MLIR converter failed.");
  }
  // If running converter test, generate a main function
  if (MLIRConverterRunTest)
    if (Error E = MCG.generateMainForTesting())
      return createError(E,
                         "MLIR converter failed to generate main for testing.");
  // Verify that the converter is emitting legal MLIR code
  if (Error E = MCG.verify())
    return createError(E, "MLIR converter failed to verify.");
  // Write the module to the output file.
  if (Error E = MCG.writeToFile(OutputFilename))
    return createError(E, "MLIR converter failed to write to file.");
  LLVM_DEBUG(dbgs() << "MLIR_Converter Succeeded\n");

  return Error::success();
}

/// Main convert method for converting an input LLVMIR file to a target.
static Error convert() {
  // Check the command line options
  if (Error E = checkCommandLineOptions())
    return createError(E, "Command line options check failed.");

  // Parse the input file into an LLVM Module.
  LLVMContext Context;
  SMDiagnostic Err;
  std::unique_ptr<Module> M = parseIRFile(InputFilename, Err, Context);
  if (!M)
    return createError("Unable to parse input file: " + InputFilename);

  // Cache all existing functions in order to avoid processing functions added
  // by pass.
  SmallVector<Function *> FuncList;
  if (Error E = populateFuncList(M.get(), FuncList))
    return createError(E, "Failed to populate function list.");

  LLVM_DEBUG(dbgs() << "*** IR Dump before Conversion ******\n"; M->dump());

  // Run the akg side of the converter
  if (emit_akg) {
    if (Error E = convertAKG(Context, FuncList))
      return createError(E, "Error in AKG side of the converter.");
  }

  // Run the MLIR side of the converter
  if (emit_mlir) {
    if (Error E = convertMLIR(M.get(), FuncList))
      return createError(E, "Error in the MLIR side of the converter.");
  }

  // Generate the trampoline file. The trampoline functions are generated by
  // deleting the contents of the module functions and replacing them with a
  // trampoline call. Thus, the trampoline file is just the current module.
  if (emit_trampoline) {
    error_code EC;
    raw_fd_ostream OS(TrampolineFilename, EC, sys::fs::OF_Text);
    if (EC)
      return createError("Unable to write to trampoline output file: " +
                             TrampolineFilename,
                         EC);
    WriteBitcodeToFile(*M.get(), OS);
    OS.flush();
    LLVM_DEBUG(dbgs() << "Trampoline functions have been written to: "
                      << TrampolineFilename << "\n");
  }

  return Error::success();
}

int main(int argc, char **argv) {
  InitLLVM X{argc, argv};
  cl::HideUnrelatedOptions(SYCLopsCat);
  cl::ParseCommandLineOptions(argc, argv,
                              "Convert SYCL LLVMIR into a target backend.");

  auto error = convert();
  int exitCode = EXIT_SUCCESS;
  handleAllErrors(std::move(error), [&exitCode](const ErrorInfoBase &info) {
    errs() << "error: ";
    info.log(errs());
    errs() << '\n';
    exitCode = EXIT_FAILURE;
  });

  return exitCode;
}
