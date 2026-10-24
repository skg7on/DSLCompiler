//===- vector_add_bf16.mlir - ARM bf16 vector add coverage fixture --------===//
//
// Part of the LLK Compiler Project, under the Apache License v2.0 with
// LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Framework E6 (#181): the bf16 add. A 512-bit register holds 32 bf16 lanes,
// so the selected blocked layout must solve VW=32 for this port -- the same
// rule solves 16 for f32 and 64 for i8 from the same finite domain.

module {
  micro.kernel @arm_vector_add_bf16(%a: tensor<8x8xbf16>,
                                    %b: tensor<8x8xbf16>) -> tensor<8x8xbf16> {
    %ta = micro.tile_view %a {shape = array<i64: 8, 8>} : tensor<8x8xbf16> -> !micro.tile<8x8xbf16, memory = #micro.memory<sram>>
    %tb = micro.tile_view %b {shape = array<i64: 8, 8>} : tensor<8x8xbf16> -> !micro.tile<8x8xbf16, memory = #micro.memory<sram>>
    %r = micro.vector "add" %ta, %tb : !micro.tile<8x8xbf16, memory = #micro.memory<sram>>, !micro.tile<8x8xbf16, memory = #micro.memory<sram>> -> !micro.tile<8x8xbf16, memory = #micro.memory<sram>>
    micro.yield %r : !micro.tile<8x8xbf16, memory = #micro.memory<sram>>
  }
}
