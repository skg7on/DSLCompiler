//===- fused_convert_silu_mul.mlir - ARM fused epilogue fixture ----------===//
//
// Part of the LLK Compiler Project, under the Apache License v2.0 with
// LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Framework E6 (#181): bf16 input converted to f32, SiLU, then multiplied by
// an f32 gate. The operand order is part of the fused rule's edge pattern:
// `mul` reads the activation first and the gate second. Each boundary port
// solves its own width (bf16 -> 32, f32 -> 16); a graph-wide dtype would pin
// the f32 gate to the bf16 assignment.

module {
  micro.kernel @arm_fused_convert_silu_mul(%input: tensor<8x8xbf16>,
                                           %gate: tensor<8x8xf32>)
      -> tensor<8x8xf32> {
    %ti = micro.tile_view %input {shape = array<i64: 8, 8>} : tensor<8x8xbf16> -> !micro.tile<8x8xbf16, memory = #micro.memory<sram>>
    %tg = micro.tile_view %gate {shape = array<i64: 8, 8>} : tensor<8x8xf32> -> !micro.tile<8x8xf32, memory = #micro.memory<sram>>
    %c = micro.vector "convert" %ti : !micro.tile<8x8xbf16, memory = #micro.memory<sram>> -> !micro.tile<8x8xf32, memory = #micro.memory<sram>>
    %s = micro.vector "silu" %c : !micro.tile<8x8xf32, memory = #micro.memory<sram>> -> !micro.tile<8x8xf32, memory = #micro.memory<sram>>
    %r = micro.vector "mul" %s, %tg : !micro.tile<8x8xf32, memory = #micro.memory<sram>>, !micro.tile<8x8xf32, memory = #micro.memory<sram>> -> !micro.tile<8x8xf32, memory = #micro.memory<sram>>
    micro.yield %r : !micro.tile<8x8xf32, memory = #micro.memory<sram>>
  }
}
