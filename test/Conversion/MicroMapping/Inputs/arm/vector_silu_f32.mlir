//===- vector_silu_f32.mlir - ARM f32 SiLU coverage fixture --------------===//
//
// Part of the LLK Compiler Project, under the Apache License v2.0 with
// LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Framework E6 (#181): SiLU is advertised for f32 only. It is an activation,
// not an approximate-conversion flag: an explicit `math_mode` would be part of
// the operation's semantics, and this fixture leaves it at the exact default.

module {
  micro.kernel @arm_vector_silu_f32(%a: tensor<8x8xf32>) -> tensor<8x8xf32> {
    %ta = micro.tile_view %a {shape = array<i64: 8, 8>} : tensor<8x8xf32> -> !micro.tile<8x8xf32, memory = #micro.memory<sram>>
    %r = micro.vector "silu" %ta : !micro.tile<8x8xf32, memory = #micro.memory<sram>> -> !micro.tile<8x8xf32, memory = #micro.memory<sram>>
    micro.yield %r : !micro.tile<8x8xf32, memory = #micro.memory<sram>>
  }
}
