//===- mma_i8.mlir - ARM unsupported i8 contraction fixture --------------===//
//
// Part of the LLK Compiler Project, under the Apache License v2.0 with
// LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Framework E6 (#181): an integer contraction the ARM target pair does not
// claim. The IR is well formed -- `micro.mma` permits i8 inputs with an f32
// accumulator -- so the rejection must come from the typed rule table, not from
// the dialect. No complete plan may be retained for it.

module {
  micro.kernel @arm_mma_i8(%a: tensor<16x16xi8>, %b: tensor<16x16xi8>,
                           %init: tensor<16x16xf32>) -> tensor<16x16xf32> {
    %ta = micro.tile_view %a {shape = array<i64: 16, 16>} : tensor<16x16xi8> -> !micro.tile<16x16xi8, memory = #micro.memory<sram>>
    %tb = micro.tile_view %b {shape = array<i64: 16, 16>} : tensor<16x16xi8> -> !micro.tile<16x16xi8, memory = #micro.memory<sram>>
    %tc = micro.tile_view %init {shape = array<i64: 16, 16>} : tensor<16x16xf32> -> !micro.tile<16x16xf32, memory = #micro.memory<acc>>
    %r = micro.mma %ta, %tb, %tc {shape = array<i64: 16, 16, 16>, input = #micro.dtype<i8>, accumulator = #micro.dtype<f32>} : !micro.tile<16x16xi8, memory = #micro.memory<sram>>, !micro.tile<16x16xi8, memory = #micro.memory<sram>>, !micro.tile<16x16xf32, memory = #micro.memory<acc>> -> !micro.tile<16x16xf32, memory = #micro.memory<acc>>
    micro.yield %r : !micro.tile<16x16xf32, memory = #micro.memory<acc>>
  }
}
