//===- vector_mul_f32_full.mlir - ARM f32 full-width vector mul fixture --===//
//
// Part of the LLK Compiler Project, under the Apache License v2.0 with
// LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Framework E10 (#184): the full-width companion to vector_mul_f32.mlir. One
// row of a 16x16 f32 tile is exactly the modeled 512-bit SVE vector (16 f32
// lanes at vscale = 4), so this shape has no masked remainder lane.
// Multiplication is a separate typed rule from addition, so it needs its own
// full-width source rather than reusing the add fixture's rule match. This is a
// bounded conformance program, not a tuned kernel.

module {
  micro.kernel @arm_vector_mul_f32_full(%a: tensor<16x16xf32>, %b: tensor<16x16xf32>)
      -> tensor<16x16xf32> {
    %ta = micro.tile_view %a {shape = array<i64: 16, 16>} : tensor<16x16xf32> -> !micro.tile<16x16xf32, memory = #micro.memory<sram>>
    %tb = micro.tile_view %b {shape = array<i64: 16, 16>} : tensor<16x16xf32> -> !micro.tile<16x16xf32, memory = #micro.memory<sram>>
    %r = micro.vector "mul" %ta, %tb : !micro.tile<16x16xf32, memory = #micro.memory<sram>>, !micro.tile<16x16xf32, memory = #micro.memory<sram>> -> !micro.tile<16x16xf32, memory = #micro.memory<sram>>
    micro.yield %r : !micro.tile<16x16xf32, memory = #micro.memory<sram>>
  }
}
