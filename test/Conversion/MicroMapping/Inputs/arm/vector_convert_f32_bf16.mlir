//===- vector_convert_f32_bf16.mlir - ARM f32-to-bf16 convert fixture ----===//
//
// Part of the LLK Compiler Project, under the Apache License v2.0 with
// LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Framework E6 (#181): the dtype-narrowing conversion. The operand is f32, so
// its blocked layout solves VW=16 even though the result is bf16.

module {
  micro.kernel @arm_vector_convert_f32_bf16(%a: tensor<8x8xf32>)
      -> tensor<8x8xbf16> {
    %ta = micro.tile_view %a {shape = array<i64: 8, 8>} : tensor<8x8xf32> -> !micro.tile<8x8xf32, memory = #micro.memory<sram>>
    %r = micro.vector "convert" %ta : !micro.tile<8x8xf32, memory = #micro.memory<sram>> -> !micro.tile<8x8xbf16, memory = #micro.memory<sram>>
    micro.yield %r : !micro.tile<8x8xbf16, memory = #micro.memory<sram>>
  }
}
