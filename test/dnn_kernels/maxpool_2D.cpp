//===- maxpool_2D.cpp -----------------------------------------------------===//
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

// Last pooling layer of ResNet-50 before FC layer.
//    Input size: 7x7
//    Pool size: 7x7
//    Stride: N/A (1)
#define H 7
#define W 7
#define POOL_H 7
#define POOL_W 7
#define STRIDE_H 1
#define STRIDE_W 1

#define OUT_H (((H - POOL_H) / STRIDE_H) + 1)
#define OUT_W (((W - POOL_W) / STRIDE_W) + 1)

using namespace ::sycl;

using _Array_in = Array<float, H * W>;
using _Array_out = Array<float, OUT_H * OUT_W>;

int main() {
  queue deviceQueue(default_selector_v);
  const device dev = deviceQueue.get_device();
  const context ctx = deviceQueue.get_context();

  auto IN_acc = (_Array_in *)malloc_shared(sizeof(_Array_in), dev, ctx);
  auto OUT_acc = (_Array_out *)malloc_shared(sizeof(_Array_out), dev, ctx);

  deviceQueue.submit([&](handler &cgh) {
    auto kern = [=]() {
      float *in = IN_acc->data();
      float *out = OUT_acc->data();

      for (int i = 0; i < OUT_H; i++) {
        for (int j = 0; j < OUT_W; j++) {
          for (int k = 0; k < POOL_H; k++) {
            for (int l = 0; l < POOL_W; l++) {
              int idx_H = i * STRIDE_H + k;
              int idx_W = j * STRIDE_W + l;
              out[OUT_W * i + j] =
                  sycl::max(out[OUT_W * i + j], in[W * idx_H + idx_W]);
            }
          }
        }
      }
    };
    cgh.single_task<class maxpool_2D>(kern);
  });

  deviceQueue.wait();

  sycl::free(IN_acc, deviceQueue);
  sycl::free(OUT_acc, deviceQueue);

  return 0;
}

// clang-format off
// MLIR:      func.func {{@[a-zA-Z0-9\$_]+}}(%arg0: memref<49xf32, 1>, %arg1: memref<1xf32, 1>) {
// MLIR-NEXT:   %0 = affine.load %arg1[0] : memref<1xf32, 1>
// MLIR-NEXT:   %1 = affine.for %arg2 = 0 to 7 iter_args(%arg3 = %0) -> (f32) {
// MLIR-NEXT:     %2 = affine.for %arg4 = 0 to 7 iter_args(%arg5 = %arg3) -> (f32) {
// MLIR-NEXT:       %3 = affine.load %arg0[%arg4 + %arg2 * 7] : memref<49xf32, 1>
// MLIR-NEXT:       %4 = arith.maxf %arg5, %3 : f32
// MLIR-NEXT:       affine.store %4, %arg1[0] : memref<1xf32, 1>
// MLIR-NEXT:       affine.yield %4 : f32
// MLIR-NEXT:     }
// MLIR-NEXT:     affine.yield %2 : f32
// MLIR-NEXT:   }
// MLIR-NEXT:   return
// MLIR-NEXT: }
// clang-format on
