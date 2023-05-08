//===- single_1dbuffer_vec_add_16_32.cpp ----------------------------------===//
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

constexpr sycl::access::mode sycl_read = sycl::access::mode::read;
constexpr sycl::access::mode sycl_write = sycl::access::mode::write;
constexpr sycl::access::mode sycl_read_write = sycl::access::mode::read_write;

#define N 256

int main() {
  const size_t tensorSize = N;
  std::array<float, tensorSize> A_tensor;
  std::array<float, tensorSize> B_tensor;
  std::array<float, tensorSize> OUT_tensor;

  for (size_t i = 0; i < tensorSize; i++) {
    A_tensor[i] = 1;
    B_tensor[i] = 2;
    OUT_tensor[i] = 0;
  }
  queue deviceQueue(default_selector_v);
  buffer<float, 1> A_buf(A_tensor.data(), range<1>(N));
  buffer<float, 1> B_buf(B_tensor.data(), range<1>(N));
  buffer<float, 1> OUT_buf(OUT_tensor.data(), range<1>(N));

  deviceQueue.submit([&](sycl::handler &cgh) {
    auto A_acc = A_buf.get_access<sycl_read>(cgh);
    auto B_acc = B_buf.get_access<sycl_read>(cgh);
    auto OUT_acc = OUT_buf.get_access<sycl_read_write>(cgh);

    auto kern = [=]() {
      for (int i = 0; i < N; i++) {
        OUT_acc[i] = A_acc[i] + B_acc[i];
      }
    };
    cgh.single_task<class vec_add>(kern);
  });
  deviceQueue.wait();

  return 0;
}

// clang-format off
// MLIR:      #map = affine_map<(d0)[s0] -> (d0 + s0)>
// MLIR:      func.func @{{[a-zA-Z0-9\$_]+}}(%arg0: memref<?xf32, #map, 1>, %arg1: memref<?xf32, #map, 1>, %arg2: memref<?xf32, #map, 1>) {
// MLIR-NEXT:   affine.for %arg3 = 0 to 256 {
// MLIR-NEXT:     %0 = affine.load %arg1[%arg3] : memref<?xf32, #map, 1>
// MLIR-NEXT:     %1 = affine.load %arg2[%arg3] : memref<?xf32, #map, 1>
// MLIR-NEXT:     %2 = arith.addf %0, %1 : f32
// MLIR-NEXT:     affine.store %2, %arg0[%arg3] : memref<?xf32, #map, 1>
// MLIR-NEXT:   }
// MLIR-NEXT:   return
// MLIR-NEXT: }
// clang-format on