//===- single_usmbuffer_offset_loads.cpp ----------------------------------===//
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

using _Array_a = Array<float, 256>;
using _Array_c = Array<float, 232>;

int main() {
  queue deviceQueue(default_selector_v);
  const device dev = deviceQueue.get_device();
  const context ctx = deviceQueue.get_context();

  auto A_acc = (_Array_a *)malloc_shared(sizeof(_Array_a), dev, ctx);
  auto C_acc = (_Array_c *)malloc_shared(sizeof(_Array_c), dev, ctx);

  deviceQueue.submit([&](handler &cgh) {
    auto kern = [=]() {
      float *c = C_acc->data();
      float *a = A_acc->data();

      for (int i = 8; i < 240; i++) {
        c[i] = a[i - 8] + a[i + 2] + a[i + 16];
      }
    };
    cgh.single_task<class simple_offset_loads>(kern);
  });

  deviceQueue.wait();

  sycl::free(C_acc, deviceQueue);
  sycl::free(A_acc, deviceQueue);

  return 0;
}

// clang-format off
// MLIR:      func.func @{{[a-zA-Z0-9\$_]+}}(%arg0: memref<232xf32, 1>, %arg1: memref<256xf32, 1>) {
// MLIR-NEXT:   affine.for %arg2 = 8 to 240 {
// MLIR-NEXT:     %0 = affine.load %arg1[%arg2 - 8] : memref<256xf32, 1>
// MLIR-NEXT:     %1 = affine.load %arg1[%arg2 + 2] : memref<256xf32, 1>
// MLIR-NEXT:     %2 = arith.addf %0, %1 : f32
// MLIR-NEXT:     %3 = affine.load %arg1[%arg2 + 16] : memref<256xf32, 1>
// MLIR-NEXT:     %4 = arith.addf %2, %3 : f32
// MLIR-NEXT:     affine.store %4, %arg0[%arg2] : memref<232xf32, 1>
// MLIR-NEXT:   }
// MLIR-NEXT:   return
// MLIR-NEXT: }
// clang-format on
