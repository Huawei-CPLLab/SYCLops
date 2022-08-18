//===- test_common.hpp ----------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <array>
#include <cstddef>

template <typename T, size_t N, size_t... Rest>
struct Array : std::array<Array<T, Rest...>, N> {
  using std::array<Array<T, Rest...>, N>::operator[];
  Array() = default;
  Array(T *data) { memcpy(this, data, sizeof(*this)); }
  Array(Array<T, N, Rest...> *data) { memcpy(this, data, sizeof(*this)); }
};

template <typename T, size_t N> struct Array<T, N> : std::array<T, N> {
  using std::array<T, N>::operator[];
  Array() = default;
  Array(T *data) { memcpy(this, data, sizeof(*this)); }
  Array(Array<T, N> *data) { memcpy(this, data, sizeof(*this)); }
};