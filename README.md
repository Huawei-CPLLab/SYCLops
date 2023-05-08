# SYCLops: A SYCL Specific LLVM to MLIR Converter

There is a growing need for higher level abstractions for device kernels in heterogeneous environments, and the multi-level nature of the MLIR infrastructure perfectly addresses this requirement. As SYCL begins to gain industry adoption for heterogeneous applications and MLIR continues to develop, we present SYCLops: a converter capable of translating SYCL specific LLVM IR to MLIR. This will allow for both target and application specific optimizations within the same framework to exploit opportunities for improvement present at different levels.

For more information, see our paper published and presented at the IWOCL'22: International Workshop on OpenCL: https://dl.acm.org/doi/10.1145/3529538.3529992

Presentation video: https://www.youtube.com/watch?v=yhArFYsiLcg

## Build Instructions

### 1. Clone SYCLops

```sh
git clone --recursive https://github.com/Huawei-PTLab/SYCLops.git
cd SYCLops
```

### 2. Install Clang, SYCL, MLIR, and SYCLops

```sh
mkdir build
cd build
cmake -G Ninja ../llvm/llvm \
  -DLLVM_ENABLE_PROJECTS="mlir;clang;sycl;opencl" \
  -DLLVM_EXTERNAL_PROJECTS="syclops;sycl;opencl" \
  -DLLVM_EXTERNAL_SYCLOPS_SOURCE_DIR=.. \
  -DLLVM_EXTERNAL_SYCL_SOURCE_DIR=../llvm/sycl \
  -DSYCL_ENABLE_PLUGINS="opencl;level_zero" \
  -DLLVM_TARGETS_TO_BUILD="host" \
  -DLLVM_ENABLE_ASSERTIONS=ON \
  -DSYCL_ENABLE_KERNEL_FUSION=OFF \
  -DCMAKE_BUILD_TYPE=Release
ninja check-syclops
```

`check-syclops` will compile the required executables (clang++, opt, syclops) and run the tests located in the `/test` directory which will ensure that SYCLops built correctly.

### Requirements

SYCLops follows the same requirements as LLVM, specified here: https://llvm.org/docs/GettingStarted.html#software

Namely:
- CMake 3.13.4 and above
  - For the build command above, we use `ninja` as our generator. This is also needed.
- Python3 3.6 and above
- GCC 7.1.0 and above
- g++

Older versions "may" work.

## Example

Once you have run the above installation, all of the executables you will need to use SYCLops would have been built for you in the `build/bin` directory. For a simple example, we will be converting the Kmeans kernel found in `test/static_usmbuffer_tests/parallel_usmbuffer_kmeans.cpp` into MLIR using SYCLops.

```sh
# Set your path variable to include the binaries built above.
PATH=<SYCLops_path>/SYCLops/build/bin:$PATH
# Run oneAPI's SYCL Device Compiler to compile the device kernel to LLVMIR.
# This will also run some preconditioning passes to clean up the IR.
clang++ \
  parallel_usmbuffer_kmeans.cpp \
  -fsycl-device-only \
  -fno-sycl-instrument-device-code \
  -fno-sycl-dead-args-optimization \
  -S -emit-llvm \
  -O2 \
  -mllvm -sycl-opt=false \
  -mllvm -disable-loop-idiom-memset \
  -fno-unroll-loops \
  -fno-vectorize \
  -ffp-contract=off \
  -D__SYCL_DISABLE_PARALLEL_FOR_RANGE_ROUNDING__ \
  -o parallel_usmbuffer_kmeans.ll
# Run SYCLops, setting the emit-mlir flag.
syclops \
  parallel_usmbuffer_kmeans.ll \
  -emit-mlir \
  -o parallel_usmbuffer_kmeans.mlir
```

## FAQ

### Why LLVMIR to MLIR? Why not convert directly from the Clang AST?
This is explained in much more detail in our recorded presentation at the IWOCL '22 conference at this time stamp: https://www.youtube.com/watch?v=yhArFYsiLcg&t=302s

In short, there are four main reasons we chose to enter into LLVM first as opposed to converting directly from the Clang AST:
1. MLIR is a very young framework, while LLVM has had many years to mature. This means that if the user wrote code that cannot be expressed in MLIR, we can perform transformations within LLVM to turn them into a form that MLIR can work with. This gives us more coverage (with more simplicity) than if we had gone directly from the AST.
2. It is easier to enter into LLVM first. oneAPI's DPC++ SYCL Compiler is already set up to lower to LLVMIR, and the optimizations it performs to clean up the IR is already implemented for us. This makes translation into MLIR's Affine dialect incedibly simple and allows us to take advantage of the work already done by the community without rewritting huge swaths of code.
3. Many other projects, such as Polygeist for example, are already trying to generate MLIR code from C++ source files, only a few are trying to go from SYCL C++ source file to MLIR; however all of these solutions are far from production ready and require more work.
4. As was explained in the introduction of the paper, SYCLops was designed with hardware accelerators in mind. The main motivation of this work was to emit MLIR for optimizing towards hardware accelerators as opposed to CPU, and we felt that MLIR was a better framework for this goal compared to LLVM. Thus we wanted SYCLops to convert only the device side of the compiler, allowing the host side of the compiler to go down LLVM (as we felt LLVM was sufficient for targetting CPU).

## Supported SYCL Features

### Memory Buffers
- USM Pointers
  - Static shape: for static shape these pointers must be cast to a nested array structure such that the shape information can be passed into the device kernel.
- SYCL Buffers
  - 1D buffers: Supported
  - Multi-D buffers: WIP: A change with upstream LICM has broken the delinearization of the multi-D SYCL buffers. We are working on fixing this.

### SYCL Command Groups
- `single_task`
- `parallel_for`

## Citing
```bibtex
@inproceedings{10.1145/3529538.3529992,
author = {Singer, Alexandre and Gao, Frank and Wang, Kai-Ting Amy},
title = {SYCLops: A SYCL Specific LLVM to MLIR Converter},
year = {2022},
isbn = {9781450396585},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3529538.3529992},
doi = {10.1145/3529538.3529992},
abstract = {There is a growing need for higher level abstractions for device kernels in heterogeneous environments, and the multi-level nature of the MLIR infrastructure perfectly addresses this requirement. As SYCL begins to gain industry adoption for heterogeneous applications and MLIR continues to develop, we present SYCLops: a converter capable of translating SYCL specific LLVM IR to MLIR. This will allow for both target and application specific optimizations within the same framework to exploit opportunities for improvement present at different levels.},
booktitle = {International Workshop on OpenCL},
articleno = {13},
numpages = {8},
keywords = {IR Converter, MLIR, Heterogeneous Computing, LLVM, SYCL},
location = {Bristol, United Kingdom, United Kingdom},
series = {IWOCL'22}
}
```

