//===- contraction_bf16.mlir - ARM bf16 contraction acceptance fixture ----===//
//
// Part of the LLK Compiler Project, under the Apache License v2.0 with
// LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Framework E5 (#180): the bf16-input variant of contraction_f32.mlir. Only
// the a/b operands and the tiles that stage them are bf16; the initializer,
// accumulator, result and output stay f32, because the profiles accumulate in
// f32. The dataflow, shapes and ownership are otherwise identical, and the
// input/source graph hash is the identity both ARM target comparisons retain
// for their own dtype.

module {
  micro.kernel @arm_contraction(%a: tensor<16x16xbf16>, %b: tensor<16x16xbf16>,
                                %init: tensor<16x16xf32>) -> tensor<16x16xf32> {
    %av = micro.tile_view %a {shape = array<i64: 16, 16>} : tensor<16x16xbf16> -> !micro.tile<16x16xbf16, layout = #micro.layout<row_major>, memory = #micro.memory<dram>, owner = #micro.owner<worker>>
    %bv = micro.tile_view %b {shape = array<i64: 16, 16>} : tensor<16x16xbf16> -> !micro.tile<16x16xbf16, layout = #micro.layout<row_major>, memory = #micro.memory<dram>, owner = #micro.owner<worker>>
    %iv = micro.tile_view %init {shape = array<i64: 16, 16>} : tensor<16x16xf32> -> !micro.tile<16x16xf32, layout = #micro.layout<row_major>, memory = #micro.memory<dram>, owner = #micro.owner<worker>>

    %as, %at = micro.tile_async_copy %av {dst_memory = #micro.memory<sram>, owner = #micro.owner<worker>} : !micro.tile<16x16xbf16, layout = #micro.layout<row_major>, memory = #micro.memory<dram>, owner = #micro.owner<worker>> -> !micro.tile<16x16xbf16, layout = #micro.layout<row_major>, memory = #micro.memory<sram>, owner = #micro.owner<worker>>, !micro.async_token
    micro.wait %at
    %bs, %bt = micro.tile_async_copy %bv {dst_memory = #micro.memory<sram>, owner = #micro.owner<worker>} : !micro.tile<16x16xbf16, layout = #micro.layout<row_major>, memory = #micro.memory<dram>, owner = #micro.owner<worker>> -> !micro.tile<16x16xbf16, layout = #micro.layout<row_major>, memory = #micro.memory<sram>, owner = #micro.owner<worker>>, !micro.async_token
    micro.wait %bt
    %is, %it = micro.tile_async_copy %iv {dst_memory = #micro.memory<sram>, owner = #micro.owner<worker>} : !micro.tile<16x16xf32, layout = #micro.layout<row_major>, memory = #micro.memory<dram>, owner = #micro.owner<worker>> -> !micro.tile<16x16xf32, layout = #micro.layout<row_major>, memory = #micro.memory<sram>, owner = #micro.owner<worker>>, !micro.async_token
    micro.wait %it

    %c, %ct = micro.tile_async_copy %is {dst_memory = #micro.memory<acc>, owner = #micro.owner<worker>} : !micro.tile<16x16xf32, layout = #micro.layout<row_major>, memory = #micro.memory<sram>, owner = #micro.owner<worker>> -> !micro.tile<16x16xf32, layout = #micro.layout<row_major>, memory = #micro.memory<acc>, owner = #micro.owner<worker>>, !micro.async_token
    micro.wait %ct

    %r = micro.mma %as, %bs, %c {shape = array<i64: 16, 16, 16>, input = #micro.dtype<bf16>, accumulator = #micro.dtype<f32>} : !micro.tile<16x16xbf16, layout = #micro.layout<row_major>, memory = #micro.memory<sram>, owner = #micro.owner<worker>>, !micro.tile<16x16xbf16, layout = #micro.layout<row_major>, memory = #micro.memory<sram>, owner = #micro.owner<worker>>, !micro.tile<16x16xf32, layout = #micro.layout<row_major>, memory = #micro.memory<acc>, owner = #micro.owner<worker>> -> !micro.tile<16x16xf32, layout = #micro.layout<row_major>, memory = #micro.memory<acc>, owner = #micro.owner<worker>>

    %out, %ot = micro.tile_async_copy %r {dst_memory = #micro.memory<sram>, owner = #micro.owner<worker>} : !micro.tile<16x16xf32, layout = #micro.layout<row_major>, memory = #micro.memory<acc>, owner = #micro.owner<worker>> -> !micro.tile<16x16xf32, layout = #micro.layout<row_major>, memory = #micro.memory<sram>, owner = #micro.owner<worker>>, !micro.async_token
    micro.wait %ot

    micro.yield %out : !micro.tile<16x16xf32, layout = #micro.layout<row_major>, memory = #micro.memory<sram>, owner = #micro.owner<worker>>
  }
}
