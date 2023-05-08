//===- single_usmbuffer_vec_add_16_32.cpp ---------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: clang++ %syclops-clang-device-only-flags %s -o %t.ll
// RUN: syclops %t.ll -emit-mlir -o %t.mlir
// RUN: FileCheck %s --input-file=%t.mlir --check-prefix=MLIR
// RUN: syclops %t.ll -emit-akg -o %t.txt

#include "../test_common.hpp"
#include <CL/sycl.hpp>

using namespace ::sycl;

#define N 16
#define M 32

using _Array = Array<float, N, M>;

int main() {
  queue deviceQueue(default_selector_v);

  const device dev = deviceQueue.get_device();
  const context ctx = deviceQueue.get_context();

  auto A_acc = (_Array *)malloc_shared(sizeof(_Array), dev, ctx);
  auto B_acc = (_Array *)malloc_shared(sizeof(_Array), dev, ctx);
  auto OUT_acc = (_Array *)malloc_shared(sizeof(_Array), dev, ctx);

  deviceQueue.submit([&](handler &cgh) {
    auto kern = [=]() {
      for (int i = 0; i < N; i++) {
        for (int j = 0; j < M; j++) {
          (*OUT_acc)[i][j] = (*A_acc)[i][j] + (*B_acc)[i][j];
        }
      }
    };
    cgh.single_task<class vec_add>(kern);
  });

  deviceQueue.wait();

  sycl::free(OUT_acc, deviceQueue);
  sycl::free(B_acc, deviceQueue);
  sycl::free(A_acc, deviceQueue);

  return 0;
}

// clang-format off
// MLIR:      func.func @{{[a-zA-Z0-9\$_]+}}(%arg0: memref<16x32xf32, 1>, %arg1: memref<16x32xf32, 1>, %arg2: memref<16x32xf32, 1>) {
// MLIR-NEXT:   affine.for %arg3 = 0 to 16 {
// MLIR-NEXT:     affine.for %arg4 = 0 to 32 {
// MLIR-NEXT:       %0 = affine.load %arg1[%arg3, %arg4] : memref<16x32xf32, 1>
// MLIR-NEXT:       %1 = affine.load %arg2[%arg3, %arg4] : memref<16x32xf32, 1>
// MLIR-NEXT:       %2 = arith.addf %0, %1 : f32
// MLIR-NEXT:       affine.store %2, %arg0[%arg3, %arg4] : memref<16x32xf32, 1>
// MLIR-NEXT:     }
// MLIR-NEXT:   }
// MLIR-NEXT:   return
// MLIR-NEXT: }
// clang-format on