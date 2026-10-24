//===- vector_silu_bf16.mlir - ARM unsupported bf16 SiLU fixture ----------===//
//
// Part of the LLK Compiler Project, under the Apache License v2.0 with
// LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Framework E6 (#181): SiLU is advertised for f32 only. This bf16 activation
// is legal MicroIR and must retain no complete plan: the vector compute table
// has no entry for it, so no rule may silently pick the f32 implementation.

module {
  micro.kernel @arm_vector_silu_bf16(%a: tensor<8x8xbf16>)
      -> tensor<8x8xbf16> {
    %ta = micro.tile_view %a {shape = array<i64: 8, 8>} : tensor<8x8xbf16> -> !micro.tile<8x8xbf16, memory = #micro.memory<sram>>
    %r = micro.vector "silu" %ta : !micro.tile<8x8xbf16, memory = #micro.memory<sram>> -> !micro.tile<8x8xbf16, memory = #micro.memory<sram>>
    micro.yield %r : !micro.tile<8x8xbf16, memory = #micro.memory<sram>>
  }
}
