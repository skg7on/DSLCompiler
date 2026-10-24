//===- reduce_sum_f32.mlir - ARM f32 reduce-sum coverage fixture ----------===//
//
// Part of the LLK Compiler Project, under the Apache License v2.0 with
// LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Framework E6 (#181): f32 sum along axis 1. The reduce rule declares no
// layout requirement, so this signature must map without a solved VW while
// still completing the shared analysis. The result tile narrows to rank one,
// which is why it cannot share the blocked rank-two requirement.

module {
  micro.kernel @arm_reduce_sum_f32(%input: tensor<8x8xf32>) -> tensor<8xf32> {
    %tile = micro.tile_view %input {shape = array<i64: 8, 8>} : tensor<8x8xf32> -> !micro.tile<8x8xf32, memory = #micro.memory<sram>>
    %sum = micro.reduce "sum" %tile {axis = 1 : i64} : !micro.tile<8x8xf32, memory = #micro.memory<sram>> -> !micro.tile<8xf32, memory = #micro.memory<acc>>
    micro.yield %sum : !micro.tile<8xf32, memory = #micro.memory<acc>>
  }
}
