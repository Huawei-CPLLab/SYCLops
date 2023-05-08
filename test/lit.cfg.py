import os

import lit.formats

from lit.llvm import llvm_config

# name: The name of this test suite.
config.name = "SYCLops"

# testFormat: The test format to use to interpret tests.
config.test_format = lit.formats.ShTest()

# suffixes: A list of file extensions to treat as test files.
config.suffixes = ['.cpp']

# excludes: A list of directories to exclude from the testsuite.
config.excludes = ['CMakeLists.txt']

# test_source_root: The root path where tests are located.
config.test_source_root = os.path.dirname(__file__)

# test_exec_root: The root path where tests should be run.
config.test_exec_root = os.path.join(config.syclops_obj_root, 'test')

# Add substitutions.
config.substitutions.append( ('%syclops-clang-device-only-flags', '-fsycl-device-only -fno-sycl-instrument-device-code -fno-sycl-dead-args-optimization -O2 -mllvm -disable-loop-idiom-memset -mllvm -sycl-opt=false -fno-unroll-loops -fno-vectorize -ffp-contract=off -D__SYCL_DISABLE_PARALLEL_FOR_RANGE_ROUNDING__ -S -emit-llvm') )

llvm_config.add_tool_substitutions(['syclops', 'clang', 'FileCheck'], [config.llvm_tools_dir])