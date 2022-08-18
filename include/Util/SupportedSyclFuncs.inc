//===-- SupportedSyclFuncs.inc - List of supported sycl builtins-*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
{
    "exp",        // sycl::exp -> VEXP
    "s_min",      // sycl::min (signed ints) -> VMIN
    "s_max",      // sycl::max (signed ints) -> VMAX
    "s_min",      // sycl::min (unsigned ints) -> VMIN
    "s_max",      // sycl::max (unsigned ints) -> VMAX
    "fmax",       // sycl::fmax
    "fmin",       // sycl::fmin
    "sqrt",       // sycl::sqrt
    "fmin_common" // another version of sycl::fmin generated based on type
};
// End supported builtin list ==================================================

// Could add support:
/*
"mad", // VMLA/VMADD
"degrees", // multiply by constant
"radians", // same as above
"log", // VLN
"rsqrt", // VRSQRT
"maxmag", // = max(|x|, |y|)
// probably more can be added
*/
