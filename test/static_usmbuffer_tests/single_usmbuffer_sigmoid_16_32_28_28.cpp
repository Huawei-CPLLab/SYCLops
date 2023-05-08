//===- single_usmbuffer_sigmoid_16_32_28_28.cpp ---------------------------===//
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

#define N 8
#define C 8
#define H 16
#define W 32

using _Array = Array<float, N, C, H, W>;

int main() {
  queue deviceQueue(default_selector_v);

  const device dev = deviceQueue.get_device();
  const context ctx = deviceQueue.get_context();

  auto X_acc = (_Array *)malloc_shared(sizeof(_Array), dev, ctx);
  auto OUT_acc = (_Array *)malloc_shared(sizeof(_Array), dev, ctx);

  deviceQueue.submit([&](sycl::handler &cgh) {
    auto kern = [=]() {
      float exp_x[N][C][H][W];
      for (int i0 = 0; i0 < N; i0++) {
        for (int i1 = 0; i1 < C; i1++) {
          for (int i2 = 0; i2 < H; i2++) {
            for (int i3 = 0; i3 < W; i3++) {
              exp_x[i0][i1][i2][i3] = sycl::exp(0 - (*X_acc)[i0][i1][i2][i3]);
            }
          }
        }
      }
      for (int i0 = 0; i0 < N; i0++) {
        for (int i1 = 0; i1 < C; i1++) {
          for (int i2 = 0; i2 < H; i2++) {
            for (int i3 = 0; i3 < W; i3++) {
              (*OUT_acc)[i0][i1][i2][i3] =
                  1.0f / (1.0f + exp_x[i0][i1][i2][i3]);
            }
          }
        }
      }
    };
    cgh.single_task<class sigmoid>(kern);
  });

  deviceQueue.wait();

  sycl::free(X_acc, deviceQueue);
  sycl::free(OUT_acc, deviceQueue);
  return 0;
}

// clang-format off
// MLIR:      func.func @{{[a-zA-Z0-9\$_]+}}(%arg0: memref<8x8x16x32xf32, 1>, %arg1: memref<8x8x16x32xf32, 1>) {
// MLIR-NEXT:   %cst = arith.constant 1.000000e+00 : f32
// MLIR-NEXT:   %cst_0 = arith.constant 0.000000e+00 : f32
// MLIR-NEXT:   %alloca = memref.alloca() : memref<8x8x16x32xf32>
// MLIR-NEXT:   affine.for %arg2 = 0 to 8 {
// MLIR-NEXT:     affine.for %arg3 = 0 to 8 {
// MLIR-NEXT:       affine.for %arg4 = 0 to 16 {
// MLIR-NEXT:         affine.for %arg5 = 0 to 32 {
// MLIR-NEXT:           %0 = affine.load %arg0[%arg2, %arg3, %arg4, %arg5] : memref<8x8x16x32xf32, 1>
// MLIR-NEXT:           %1 = arith.subf %cst_0, %0 : f32
// MLIR-NEXT:           %2 = math.exp %1 : f32
// MLIR-NEXT:           affine.store %2, %alloca[%arg2, %arg3, %arg4, %arg5] : memref<8x8x16x32xf32>
// MLIR-NEXT:         }
// MLIR-NEXT:       }
// MLIR-NEXT:     }
// MLIR-NEXT:   }
// MLIR-NEXT:   affine.for %arg2 = 0 to 8 {
// MLIR-NEXT:     affine.for %arg3 = 0 to 8 {
// MLIR-NEXT:       affine.for %arg4 = 0 to 16 {
// MLIR-NEXT:         affine.for %arg5 = 0 to 32 {
// MLIR-NEXT:           %0 = affine.load %alloca[%arg2, %arg3, %arg4, %arg5] : memref<8x8x16x32xf32>
// MLIR-NEXT:           %1 = arith.addf %0, %cst : f32
// MLIR-NEXT:           %2 = arith.divf %cst, %1 : f32
// MLIR-NEXT:           affine.store %2, %arg1[%arg2, %arg3, %arg4, %arg5] : memref<8x8x16x32xf32, 1>
// MLIR-NEXT:         }
// MLIR-NEXT:       }
// MLIR-NEXT:     }
// MLIR-NEXT:   }
// MLIR-NEXT:   return
// MLIR-NEXT: }
// clang-format on