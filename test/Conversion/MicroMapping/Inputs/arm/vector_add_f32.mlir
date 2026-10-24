//===- vector_add_f32.mlir - ARM f32 vector add coverage fixture ----------===//
//
// Part of the LLK Compiler Project, under the Apache License v2.0 with
// LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Framework E6 (#181): one typed rank-two f32 add. The operand layout width is
// not written here: the ARM policy solves the blocked layout's VW from this
// port's element type (16 for f32 on the 512-bit profiles). The fixture exists
// to prove the dtype table is exercised by the public mapping pipeline, not by
// a hand-built workload node.

module {
  micro.kernel @arm_vector_add_f32(%a: tensor<8x8xf32>, %b: tensor<8x8xf32>)
      -> tensor<8x8xf32> {
    %ta = micro.tile_view %a {shape = array<i64: 8, 8>} : tensor<8x8xf32> -> !micro.tile<8x8xf32, memory = #micro.memory<sram>>
    %tb = micro.tile_view %b {shape = array<i64: 8, 8>} : tensor<8x8xf32> -> !micro.tile<8x8xf32, memory = #micro.memory<sram>>
    %r = micro.vector "add" %ta, %tb : !micro.tile<8x8xf32, memory = #micro.memory<sram>>, !micro.tile<8x8xf32, memory = #micro.memory<sram>> -> !micro.tile<8x8xf32, memory = #micro.memory<sram>>
    micro.yield %r : !micro.tile<8x8xf32, memory = #micro.memory<sram>>
  }
}
