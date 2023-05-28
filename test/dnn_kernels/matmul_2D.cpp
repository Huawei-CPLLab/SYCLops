//===- matmul_2D.cpp ------------------------------------------------------===//
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

// First FC layer of ResNet-50 with 1024 neurons.
//    Input size: 1x1000
//    Output size: 1x1024
//      => Weight matrix size: 1000 x 1024
#define M 1
#define K 1000
#define N 1024

using namespace ::sycl;

using _Array_a = Array<float, M * K>;
using _Array_b = Array<float, K * N>;
using _Array_c = Array<float, M * N>;

int main() {
  queue deviceQueue(default_selector_v);
  const device dev = deviceQueue.get_device();
  const context ctx = deviceQueue.get_context();

  auto A_acc = (_Array_a *)malloc_shared(sizeof(_Array_a), dev, ctx);
  auto B_acc = (_Array_b *)malloc_shared(sizeof(_Array_b), dev, ctx);
  auto C_acc = (_Array_c *)malloc_shared(sizeof(_Array_c), dev, ctx);

  deviceQueue.submit([&](handler &cgh) {
    auto kern = [=]() {
      float *a = A_acc->data();
      float *b = B_acc->data();
      float *c = C_acc->data();

      for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
          for (int k = 0; k < K; k++) {
            c[N * i + j] += a[K * i + k] + b[N * k + j];
          }
        }
      }
    };
    cgh.single_task<class matmul_2D>(kern);
  });

  deviceQueue.wait();

  sycl::free(C_acc, deviceQueue);
  sycl::free(B_acc, deviceQueue);
  sycl::free(A_acc, deviceQueue);

  return 0;
}

// clang-format off
// MLIR:      func.func {{@[a-zA-Z0-9\$_]+}}(%arg0: memref<1000xf32, 1>, %arg1: memref<1024000xf32, 1>, %arg2: memref<1024xf32, 1>) {
// MLIR-NEXT:   affine.for %arg3 = 0 to 1024 {
// MLIR-NEXT:     %0 = affine.load %arg2[%arg3] : memref<1024xf32, 1>
// MLIR-NEXT:     %1 = affine.for %arg4 = 0 to 1000 iter_args(%arg5 = %0) -> (f32) {
// MLIR-NEXT:       %2 = affine.load %arg0[%arg4] : memref<1000xf32, 1>
// MLIR-NEXT:       %3 = affine.load %arg1[%arg4 * 1024 + %arg3] : memref<1024000xf32, 1>
// MLIR-NEXT:       %4 = arith.addf %2, %3 : f32
// MLIR-NEXT:       %5 = arith.addf %arg5, %4 : f32
// MLIR-NEXT:       affine.store %5, %arg2[%arg3] : memref<1024xf32, 1>
// MLIR-NEXT:       affine.yield %5 : f32
// MLIR-NEXT:     }
// MLIR-NEXT:   }
// MLIR-NEXT:   return
// MLIR-NEXT: }
// clang-format on
