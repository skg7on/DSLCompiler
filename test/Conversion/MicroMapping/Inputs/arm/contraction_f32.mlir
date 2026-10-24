//===- contraction_f32.mlir - ARM f32 contraction acceptance fixture ------===//
//
// Part of the LLK Compiler Project, under the Apache License v2.0 with
// LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Framework E5 (#180): the one directly authored M=N=K=16 contraction both ARM
// profiles map. It is the *same* source for `arm-sve` and `arm-sme`, so the
// only difference in the selected plan is the target's policy and machine.
//
// The dataflow is fixed by the refined design (§9.1):
//
//   av/bv/iv  logical DRAM views of the kernel arguments
//   as/bs/is  staged DRAM -> SRAM copies, each awaited before use
//   c         the initializer SRAM -> acc copy (initialized accumulation)
//   r         one MMA over as, bs and c
//   out       the result acc -> SRAM copy
//   micro.yield out
//
// Deliberately absent: a concrete engine id, a target-specific owner, spatial
// replication, a pipeline, an uninitialized accumulator, or an implicit output.
// The fixture is a bounded conformance program, not a tuned GEMM.

module {
  micro.kernel @arm_contraction(%a: tensor<16x16xf32>, %b: tensor<16x16xf32>,
                                %init: tensor<16x16xf32>) -> tensor<16x16xf32> {
    %av = micro.tile_view %a {shape = array<i64: 16, 16>} : tensor<16x16xf32> -> !micro.tile<16x16xf32, layout = #micro.layout<row_major>, memory = #micro.memory<dram>, owner = #micro.owner<worker>>
    %bv = micro.tile_view %b {shape = array<i64: 16, 16>} : tensor<16x16xf32> -> !micro.tile<16x16xf32, layout = #micro.layout<row_major>, memory = #micro.memory<dram>, owner = #micro.owner<worker>>
    %iv = micro.tile_view %init {shape = array<i64: 16, 16>} : tensor<16x16xf32> -> !micro.tile<16x16xf32, layout = #micro.layout<row_major>, memory = #micro.memory<dram>, owner = #micro.owner<worker>>

    %as, %at = micro.tile_async_copy %av {dst_memory = #micro.memory<sram>, owner = #micro.owner<worker>} : !micro.tile<16x16xf32, layout = #micro.layout<row_major>, memory = #micro.memory<dram>, owner = #micro.owner<worker>> -> !micro.tile<16x16xf32, layout = #micro.layout<row_major>, memory = #micro.memory<sram>, owner = #micro.owner<worker>>, !micro.async_token
    micro.wait %at
    %bs, %bt = micro.tile_async_copy %bv {dst_memory = #micro.memory<sram>, owner = #micro.owner<worker>} : !micro.tile<16x16xf32, layout = #micro.layout<row_major>, memory = #micro.memory<dram>, owner = #micro.owner<worker>> -> !micro.tile<16x16xf32, layout = #micro.layout<row_major>, memory = #micro.memory<sram>, owner = #micro.owner<worker>>, !micro.async_token
    micro.wait %bt
    %is, %it = micro.tile_async_copy %iv {dst_memory = #micro.memory<sram>, owner = #micro.owner<worker>} : !micro.tile<16x16xf32, layout = #micro.layout<row_major>, memory = #micro.memory<dram>, owner = #micro.owner<worker>> -> !micro.tile<16x16xf32, layout = #micro.layout<row_major>, memory = #micro.memory<sram>, owner = #micro.owner<worker>>, !micro.async_token
    micro.wait %it

    %c, %ct = micro.tile_async_copy %is {dst_memory = #micro.memory<acc>, owner = #micro.owner<worker>} : !micro.tile<16x16xf32, layout = #micro.layout<row_major>, memory = #micro.memory<sram>, owner = #micro.owner<worker>> -> !micro.tile<16x16xf32, layout = #micro.layout<row_major>, memory = #micro.memory<acc>, owner = #micro.owner<worker>>, !micro.async_token
    micro.wait %ct

    %r = micro.mma %as, %bs, %c {shape = array<i64: 16, 16, 16>, input = #micro.dtype<f32>, accumulator = #micro.dtype<f32>} : !micro.tile<16x16xf32, layout = #micro.layout<row_major>, memory = #micro.memory<sram>, owner = #micro.owner<worker>>, !micro.tile<16x16xf32, layout = #micro.layout<row_major>, memory = #micro.memory<sram>, owner = #micro.owner<worker>>, !micro.tile<16x16xf32, layout = #micro.layout<row_major>, memory = #micro.memory<acc>, owner = #micro.owner<worker>> -> !micro.tile<16x16xf32, layout = #micro.layout<row_major>, memory = #micro.memory<acc>, owner = #micro.owner<worker>>

    %out, %ot = micro.tile_async_copy %r {dst_memory = #micro.memory<sram>, owner = #micro.owner<worker>} : !micro.tile<16x16xf32, layout = #micro.layout<row_major>, memory = #micro.memory<acc>, owner = #micro.owner<worker>> -> !micro.tile<16x16xf32, layout = #micro.layout<row_major>, memory = #micro.memory<sram>, owner = #micro.owner<worker>>, !micro.async_token
    micro.wait %ot

    micro.yield %out : !micro.tile<16x16xf32, layout = #micro.layout<row_major>, memory = #micro.memory<sram>, owner = #micro.owner<worker>>
  }
}
