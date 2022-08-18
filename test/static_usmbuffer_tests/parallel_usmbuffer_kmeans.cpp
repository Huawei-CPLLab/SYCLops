//===- single_usmbuffer_kmeans.cpp ----------------------------------------===//
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

#define FLT_MAX 500000.0
#define NFEATURES 2
#define NCLUSTERS 3
#define PROBLEM_SIZE 3072
#define FEATURE_SIZE (2 * 3072)
#define CLUSTER_SIZE (3 * 3072)

using _Array_features = Array<float, FEATURE_SIZE>;
using _Array_clusters = Array<float, CLUSTER_SIZE>;
using _Array_membership = Array<int, PROBLEM_SIZE>;

int main() {
  queue deviceQueue(default_selector{});

  const device dev = deviceQueue.get_device();
  const context ctx = deviceQueue.get_context();

  auto features_acc =
      (_Array_features *)malloc_shared(sizeof(_Array_features), dev, ctx);
  auto clusters_acc =
      (_Array_clusters *)malloc_shared(sizeof(_Array_clusters), dev, ctx);
  auto membership_acc =
      (_Array_membership *)malloc_shared(sizeof(_Array_membership), dev, ctx);

  deviceQueue.submit([&](handler &cgh) {
    auto kern = [=](id<1> idx) {
      size_t gid = idx[0];

      if (gid < PROBLEM_SIZE) {
        int index = 0;
        float min_dist = FLT_MAX;
        for (size_t i = 0; i < NCLUSTERS; i++) {
          float dist = 0;
          for (size_t l = 0; l < NFEATURES; l++) {
            dist += ((*features_acc)[l * PROBLEM_SIZE + gid] -
                     (*clusters_acc)[i * NFEATURES + l]) *
                    ((*features_acc)[l * PROBLEM_SIZE + gid] -
                     (*clusters_acc)[i * NFEATURES + l]);
          }
          if (dist < min_dist) {
            min_dist = dist;
            index = gid;
          }
        }
        (*membership_acc)[gid] = index;
      }
    };

    cgh.parallel_for<class kmeans>(range(PROBLEM_SIZE), kern);
  });

  deviceQueue.wait();

  sycl::free(features_acc, deviceQueue);
  sycl::free(clusters_acc, deviceQueue);
  sycl::free(membership_acc, deviceQueue);

  return 0;
}

// clang-format off
// MLIR:      #set = affine_set<()[s0] : (-s0 + 3071 >= 0)>
// MLIR:      func.func @{{[a-zA-Z0-9\$_]+}}(%arg0: memref<3xi64, 1>, %arg1: memref<6144xf32, 1>, %arg2: memref<9216xf32, 1>, %arg3: memref<3072xi32, 1>) {
// MLIR-NEXT:   %c0_i32 = arith.constant 0 : i32
// MLIR-NEXT:   %cst = arith.constant 0.000000e+00 : f32
// MLIR-NEXT:   %cst_0 = arith.constant 5.000000e+05 : f32
// MLIR-NEXT:   %0 = affine.load %arg0[0] : memref<3xi64, 1>
// MLIR-NEXT:   %1 = arith.index_cast %0 : i64 to index
// MLIR-NEXT:   affine.if #set()[%1] {
// MLIR-NEXT:     %2 = arith.trunci %0 : i64 to i32
// MLIR-NEXT:     %3:2 = affine.for %arg4 = 0 to 3 iter_args(%arg5 = %cst_0, %arg6 = %c0_i32) -> (f32, i32) {
// MLIR-NEXT:       %4 = affine.for %arg7 = 0 to 2 iter_args(%arg8 = %cst) -> (f32) {
// MLIR-NEXT:         %8 = affine.load %arg1[%arg7 * 3072 + symbol(%1)] : memref<6144xf32, 1>
// MLIR-NEXT:         %9 = affine.load %arg2[%arg4 * 2 + %arg7] : memref<9216xf32, 1>
// MLIR-NEXT:         %10 = arith.subf %8, %9 : f32
// MLIR-NEXT:         %11 = arith.mulf %10, %10 : f32
// MLIR-NEXT:         %12 = arith.addf %arg8, %11 : f32
// MLIR-NEXT:         affine.yield %12 : f32
// MLIR-NEXT:       }
// MLIR-NEXT:       %5 = arith.cmpf olt, %4, %arg5 : f32
// MLIR-NEXT:       %6 = arith.select %5, %4, %arg5 : f32
// MLIR-NEXT:       %7 = arith.select %5, %2, %arg6 : i32
// MLIR-NEXT:       affine.yield %6, %7 : f32, i32
// MLIR-NEXT:     }
// MLIR-NEXT:     affine.store %3#1, %arg3[symbol(%1)] : memref<3072xi32, 1>
// MLIR-NEXT:   }
// MLIR-NEXT:   return
// MLIR-NEXT: }
// clang-format on