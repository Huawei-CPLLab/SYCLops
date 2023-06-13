//===- simple_dnn_inference.cpp -------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is a simple convolutional neural network inference function that was
// used to demonstrate Tiling for DMA-Based Hardware Accelerators (WIP),
// presented at LCTES'23.
//    doi: https://doi.org/10.1145/3589610.3596283
//    presentation:
//    https://pldi23.sigplan.org/details/LCTES-2023/10/-WIP-Tiling-for-DMA-Based-Hardware-Accelerators
//
//===----------------------------------------------------------------------===//

// RUN: clang++ %syclops-clang-device-only-flags %s -o %t.ll
// RUN: syclops %t.ll -emit-mlir -o %t.mlir
// RUN: FileCheck %s --input-file=%t.mlir --check-prefix=MLIR
// RUN: syclops %t.ll -emit-akg -o %t.txt

#include "../test_common.hpp"
#include <CL/sycl.hpp>

#define IN_H 64
#define IN_W 64
#define CONV_K_H 5
#define CONV_K_W 5
#define CONV_STRIDE_H 1
#define CONV_STRIDE_W 1
#define NUM_KERN 32
#define POOL_H ((IN_H - CONV_K_H + CONV_STRIDE_H) / CONV_STRIDE_H)
#define POOL_W ((IN_W - CONV_K_W + CONV_STRIDE_W) / CONV_STRIDE_W)
#define NUM_CLASSES 64

using namespace ::sycl;

using _Array_in = Array<float, IN_H, IN_W>;
using _Array_kern = Array<float, NUM_KERN, CONV_K_H, CONV_K_W>;
using _Array_cout = Array<float, NUM_KERN, POOL_H, POOL_W>;
using _Array_pout = Array<float, NUM_KERN>;
using _Array_mat = Array<float, NUM_KERN, NUM_CLASSES>;
using _Array_out = Array<float, NUM_CLASSES>;

int main() {
  queue deviceQueue(default_selector_v);
  const device dev = deviceQueue.get_device();
  const context ctx = deviceQueue.get_context();

  auto IN_acc = (_Array_in *)malloc_shared(sizeof(_Array_in), dev, ctx);
  auto KERNEL_acc = (_Array_kern *)malloc_shared(sizeof(_Array_kern), dev, ctx);
  auto COUT_acc = (_Array_cout *)malloc_shared(sizeof(_Array_cout), dev, ctx);
  auto POUT_acc = (_Array_pout *)malloc_shared(sizeof(_Array_pout), dev, ctx);
  auto FCMAT_acc = (_Array_mat *)malloc_shared(sizeof(_Array_mat), dev, ctx);
  auto OUT_acc = (_Array_out *)malloc_shared(sizeof(_Array_out), dev, ctx);

  deviceQueue.submit([&](handler &cgh) {
    auto kern = [=]() {
      auto *in = IN_acc->data();
      auto *kernel = KERNEL_acc->data();
      auto *fcmat = FCMAT_acc->data();
      // output variables made volatile to prevent loads from being hoisted out
      // of loops by LICM in SYCL's early optimizations.
      volatile float *pout = POUT_acc->data();
      volatile float *out = OUT_acc->data();

      // Convolution Layer
      for (int n = 0; n < NUM_KERN; n++) {
        for (int i = 0; i < POOL_H; i++) {
          for (int j = 0; j < POOL_W; j++) {
            for (int k = 0; k < CONV_K_H; k++) {
              for (int l = 0; l < CONV_K_W; l++) {
                int idx_H = i * CONV_STRIDE_H + k;
                int idx_W = j * CONV_STRIDE_W + l;
                float t =
                    idx_H < IN_H && idx_H >= 0 && idx_W < IN_W && idx_W >= 0
                        ? in[idx_H][idx_W]
                        : 0;
                volatile float *cout = (*COUT_acc)[n][i].data();
                cout[j] += kernel[n][k][l] * t;
              }
            }
          }
        }
      }

      // Pooling Layer + Flattening
      for (int n = 0; n < NUM_KERN; n++) {
        for (int k = 0; k < POOL_H; k++) {
          for (int l = 0; l < POOL_W; l++) {
            pout[n] = sycl::max(pout[n], (*COUT_acc)[n][k][l]);
          }
        }
      }

      // Fully Connected Layer (MatMul)
      for (int i = 0; i < 1; i++) {
        for (int j = 0; j < NUM_CLASSES; j++) {
          for (int k = 0; k < NUM_KERN; k++) {
            out[NUM_CLASSES * i + j] += pout[NUM_KERN * i + k] + fcmat[k][j];
          }
        }
      }

      // ReLU activation
      for (int i = 0; i < NUM_CLASSES; i++) {
        float t = out[i];
        out[i] = t < 0.f ? 0.f : t;
      }
    };
    cgh.single_task<class simple_dnn_inference>(kern);
  });

  deviceQueue.wait();

  sycl::free(IN_acc, deviceQueue);
  sycl::free(KERNEL_acc, deviceQueue);
  sycl::free(COUT_acc, deviceQueue);
  sycl::free(POUT_acc, deviceQueue);
  sycl::free(FCMAT_acc, deviceQueue);
  sycl::free(OUT_acc, deviceQueue);

  return 0;
}

// clang-format off
// MLIR:      func.func {{@[a-zA-Z0-9\$_]+}}(%arg0: memref<64x64xf32, 1>, %arg1: memref<32x5x5xf32, 1>, %arg2: memref<32x64xf32, 1>, %arg3: memref<32xf32, 1>, %arg4: memref<64xf32, 1>, %arg5: memref<32x60x60xf32, 1>) {
// MLIR-NEXT:   %cst = arith.constant 0.000000e+00 : f32
// MLIR-NEXT:   affine.for %arg6 = 0 to 32 {
// MLIR-NEXT:     affine.for %arg7 = 0 to 60 {
// MLIR-NEXT:       affine.for %arg8 = 0 to 60 {
// MLIR-NEXT:         affine.for %arg9 = 0 to 5 {
// MLIR-NEXT:           affine.for %arg10 = 0 to 5 {
// MLIR-NEXT:             %0 = affine.load %arg5[%arg6, %arg7, %arg8] : memref<32x60x60xf32, 1>
// MLIR-NEXT:             %1 = affine.load %arg0[%arg9 + %arg7, %arg10 + %arg8] : memref<64x64xf32, 1>
// MLIR-NEXT:             %2 = affine.load %arg1[%arg6, %arg9, %arg10] : memref<32x5x5xf32, 1>
// MLIR-NEXT:             %3 = arith.mulf %1, %2 : f32
// MLIR-NEXT:             %4 = arith.addf %0, %3 : f32
// MLIR-NEXT:             affine.store %4, %arg5[%arg6, %arg7, %arg8] : memref<32x60x60xf32, 1>
// MLIR-NEXT:           }
// MLIR-NEXT:         }
// MLIR-NEXT:       }
// MLIR-NEXT:     }
// MLIR-NEXT:   }
// MLIR-NEXT:   affine.for %arg6 = 0 to 32 {
// MLIR-NEXT:      affine.for %arg7 = 0 to 60 {
// MLIR-NEXT:        affine.for %arg8 = 0 to 60 {
// MLIR-NEXT:          %0 = affine.load %arg3[%arg6] : memref<32xf32, 1>
// MLIR-NEXT:          %1 = affine.load %arg5[%arg6, %arg7, %arg8] : memref<32x60x60xf32, 1>
// MLIR-NEXT:          %2 = arith.maxf %0, %1 : f32
// MLIR-NEXT:          affine.store %2, %arg3[%arg6] : memref<32xf32, 1>
// MLIR-NEXT:        }
// MLIR-NEXT:      }
// MLIR-NEXT:    }
// MLIR-NEXT:    affine.for %arg6 = 0 to 64 {
// MLIR-NEXT:      affine.for %arg7 = 0 to 32 {
// MLIR-NEXT:        %0 = affine.load %arg4[%arg6] : memref<64xf32, 1>
// MLIR-NEXT:        %1 = affine.load %arg3[%arg7] : memref<32xf32, 1>
// MLIR-NEXT:        %2 = affine.load %arg2[%arg7, %arg6] : memref<32x64xf32, 1>
// MLIR-NEXT:        %3 = arith.addf %1, %2 : f32
// MLIR-NEXT:        %4 = arith.addf %0, %3 : f32
// MLIR-NEXT:        affine.store %4, %arg4[%arg6] : memref<64xf32, 1>
// MLIR-NEXT:      }
// MLIR-NEXT:    }
// MLIR-NEXT:    affine.for %arg6 = 0 to 64 {
// MLIR-NEXT:      %0 = affine.load %arg4[%arg6] : memref<64xf32, 1>
// MLIR-NEXT:      %1 = arith.cmpf olt, %0, %cst : f32
// MLIR-NEXT:      %2 = arith.select %1, %cst, %0 : f32
// MLIR-NEXT:      affine.store %2, %arg4[%arg6] : memref<64xf32, 1>
// MLIR-NEXT:    }
// MLIR-NEXT:    return
// MLIR-NEXT:  }
// MLIR-NEXT:}
// clang-format on
