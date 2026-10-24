//===- vector_convert_bf16_f32.mlir - ARM bf16-to-f32 convert fixture ----===//
//
// Part of the LLK Compiler Project, under the Apache License v2.0 with
// LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Framework E6 (#181): the dtype-widening conversion. Conversion is spelled
// `micro.vector "convert"` -- `micro.transform` re-represents a value and its
// verifier refuses an element-type change. The layout width is solved at the
// operand's own type (bf16 -> 32), not at the result's f32 width.

module {
  micro.kernel @arm_vector_convert_bf16_f32(%a: tensor<8x8xbf16>)
      -> tensor<8x8xf32> {
    %ta = micro.tile_view %a {shape = array<i64: 8, 8>} : tensor<8x8xbf16> -> !micro.tile<8x8xbf16, memory = #micro.memory<sram>>
    %r = micro.vector "convert" %ta : !micro.tile<8x8xbf16, memory = #micro.memory<sram>> -> !micro.tile<8x8xf32, memory = #micro.memory<sram>>
    micro.yield %r : !micro.tile<8x8xf32, memory = #micro.memory<sram>>
  }
}
