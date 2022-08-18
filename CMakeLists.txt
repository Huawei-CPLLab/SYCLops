set(LLVM_LINK_COMPONENTS
  BitWriter
  IRReader
  Scalar
  Support
  TransformUtils
)

add_llvm_tool(syclops
  syclops.cpp
  src/ConverterCodeGen.cpp
  src/AKGCodeGen/AKGBuilder.cpp
  src/AKGCodeGen/AKGCodeGen.cpp
  src/MLIRCodeGen/MLIRCodeGen.cpp
  src/MLIRCodeGen/LLVMToMLIR.cpp
  src/TrampolineBuilder/TrampolineBuilder.cpp
  src/Util/ConverterUtil.cpp
  src/Util/Matcher.cpp
  src/Util/Preprocessing.cpp
)

# Link to the MLIR libs and headers
set(MLIR_LIBS
  MLIRAffineDialect
  MLIRArithmeticDialect
  MLIRFuncDialect
  MLIRIR
  MLIRLLVMDialect
  MLIRMathDialect
  MLIRMemRefDialect
  MLIRSCFDialect
  MLIRVectorDialect
)
target_link_libraries(syclops PRIVATE ${MLIR_LIBS})
target_include_directories(syclops PRIVATE
  "include"
  ${LLVM_INCLUDE_DIRS}
  ${LLVM_BINARY_DIR}/tools/mlir/include
  ${LLVM_MAIN_SRC_DIR}/../mlir/include
)

set(SYCLOPS_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(SYCLOPS_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR})

add_subdirectory(test)