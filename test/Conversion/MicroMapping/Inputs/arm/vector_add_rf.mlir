//===- vector_add_rf.mlir - ARM absent-memory-kind fixture ---------------===//
//
// Part of the LLK Compiler Project, under the Apache License v2.0 with
// LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Framework E6 (#181): a supported f32 add whose tiles live in `rf`. The
// dialect's memory enum names `rf`, but neither ARM profile models a register
// file: the accumulator budget is `acc`, and there is deliberately no second
// disjoint register pool. The plan must be rejected when storage resolves this
// occurrence, not silently remapped to SRAM.

module {
  micro.kernel @arm_vector_add_rf(%a: tensor<8x8xf32>, %b: tensor<8x8xf32>)
      -> tensor<8x8xf32> {
    %ta = micro.tile_view %a {shape = array<i64: 8, 8>} : tensor<8x8xf32> -> !micro.tile<8x8xf32, memory = #micro.memory<rf>>
    %tb = micro.tile_view %b {shape = array<i64: 8, 8>} : tensor<8x8xf32> -> !micro.tile<8x8xf32, memory = #micro.memory<rf>>
    %r = micro.vector "add" %ta, %tb : !micro.tile<8x8xf32, memory = #micro.memory<rf>>, !micro.tile<8x8xf32, memory = #micro.memory<rf>> -> !micro.tile<8x8xf32, memory = #micro.memory<rf>>
    micro.yield %r : !micro.tile<8x8xf32, memory = #micro.memory<rf>>
  }
}
