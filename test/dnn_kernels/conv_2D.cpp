//===- conv_2D.cpp --------------------------------------------------------===//
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

// First convolutional layer of ResNet-50.
//    Input size: 224x224
//    Kernel size: 7x7
//    Stride: 2
//    Output size: 112x112 (pad = 3)
#define H 224
#define W 224
#define K_H 7
#define K_W 7
#define STRIDE_H 2
#define STRIDE_W 2
#define PAD_H 3
#define PAD_W 3

#define OUT_H ((H - K_H + 2 * PAD_H + STRIDE_H) / STRIDE_H)
#define OUT_W ((W - K_W + 2 * PAD_W + STRIDE_W) / STRIDE_W)

using namespace ::sycl;

using _Array_in = Array<float, H * W>;
using _Array_kern = Array<float, K_H * K_W>;
using _Array_out = Array<float, OUT_H * OUT_W>;

int main() {
  queue deviceQueue(default_selector_v);
  const device dev = deviceQueue.get_device();
  const context ctx = deviceQueue.get_context();

  auto IN_acc = (_Array_in *)malloc_shared(sizeof(_Array_in), dev, ctx);
  auto KERNEL_acc = (_Array_kern *)malloc_shared(sizeof(_Array_kern), dev, ctx);
  auto OUT_acc = (_Array_out *)malloc_shared(sizeof(_Array_out), dev, ctx);

  deviceQueue.submit([&](handler &cgh) {
    auto kern = [=]() {
      float *in = IN_acc->data();
      float *kernel = KERNEL_acc->data();
      float *out = OUT_acc->data();

      for (int i = 0; i < OUT_H; i++) {
        for (int j = 0; j < OUT_W; j++) {
          for (int k = 0; k < K_H; k++) {
            for (int l = 0; l < K_W; l++) {
              int idx_H = i * STRIDE_H + k - PAD_H;
              int idx_W = j * STRIDE_W + l - PAD_W;
              float t = idx_H < H && idx_H >= 0 && idx_W < W && idx_W >= 0
                            ? in[W * idx_H + idx_W]
                            : 0;
              out[OUT_W * i + j] += kernel[K_W * k + l] * t;
            }
          }
        }
      }
    };
    cgh.single_task<class conv_2D>(kern);
  });

  deviceQueue.wait();

  sycl::free(IN_acc, deviceQueue);
  sycl::free(KERNEL_acc, deviceQueue);
  sycl::free(OUT_acc, deviceQueue);

  return 0;
}

// clang-format off
// MLIR:      func.func {{@[a-zA-Z0-9\$_]+}}(%arg0: memref<50176xf32, 1>, %arg1: memref<49xf32, 1>, %arg2: memref<12544xf32, 1>) {
// MLIR-NEXT:   %cst = arith.constant 0.000000e+00 : f32
// MLIR-NEXT:   affine.for %arg3 = 0 to 112 {
// MLIR-NEXT:     affine.for %arg4 = 0 to 112 {
// MLIR-NEXT:       %0 = affine.load %arg2[%arg4 + %arg3 * 112] : memref<12544xf32, 1>
// MLIR-NEXT:       %1 = affine.for %arg5 = 0 to 7 iter_args(%arg6 = %0) -> (f32) {
// MLIR-NEXT:         %2 = affine.for %arg7 = 0 to 7 iter_args(%arg8 = %arg6) -> (f32) {
// MLIR-NEXT:           %3 = affine.if #set(%arg5, %arg3, %arg7, %arg4) -> f32 {
// MLIR-NEXT:             %7 = affine.load %arg0[(%arg5 + %arg3 * 2) * 224 + %arg7 + %arg4 * 2 - 675] : memref<50176xf32, 1>
// MLIR-NEXT:             affine.yield %7 : f32
// MLIR-NEXT:           } else {
// MLIR-NEXT:             affine.yield %cst : f32
// MLIR-NEXT:           }
// MLIR-NEXT:           %4 = affine.load %arg1[%arg7 + %arg5 * 7] : memref<49xf32, 1>
// MLIR-NEXT:           %5 = arith.mulf %3, %4 : f32
// MLIR-NEXT:           %6 = arith.addf %arg8, %5 : f32
// MLIR-NEXT:           affine.store %6, %arg2[%arg4 + %arg3 * 112] : memref<12544xf32, 1>
// MLIR-NEXT:           affine.yield %6 : f32
// MLIR-NEXT:         }
// MLIR-NEXT:         affine.yield %2 : f32
// MLIR-NEXT:       }
// MLIR-NEXT:     }
// MLIR-NEXT:   }
// MLIR-NEXT:   return
// MLIR-NEXT: }
// clang-format on
