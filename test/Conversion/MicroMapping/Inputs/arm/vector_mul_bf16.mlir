//===- vector_mul_bf16.mlir - ARM bf16 vector multiply coverage fixture --===//
//
// Part of the LLK Compiler Project, under the Apache License v2.0 with
// LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Framework E6 (#181): the bf16 multiply, expected to solve VW=32.

module {
  micro.kernel @arm_vector_mul_bf16(%a: tensor<8x8xbf16>,
                                    %b: tensor<8x8xbf16>) -> tensor<8x8xbf16> {
    %ta = micro.tile_view %a {shape = array<i64: 8, 8>} : tensor<8x8xbf16> -> !micro.tile<8x8xbf16, memory = #micro.memory<sram>>
    %tb = micro.tile_view %b {shape = array<i64: 8, 8>} : tensor<8x8xbf16> -> !micro.tile<8x8xbf16, memory = #micro.memory<sram>>
    %r = micro.vector "mul" %ta, %tb : !micro.tile<8x8xbf16, memory = #micro.memory<sram>>, !micro.tile<8x8xbf16, memory = #micro.memory<sram>> -> !micro.tile<8x8xbf16, memory = #micro.memory<sram>>
    micro.yield %r : !micro.tile<8x8xbf16, memory = #micro.memory<sram>>
  }
}
