//===- vector_add_i8.mlir - ARM i8 vector add coverage fixture ------------===//
//
// Part of the LLK Compiler Project, under the Apache License v2.0 with
// LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Framework E6 (#181): the i8 add. A 512-bit register holds 64 i8 lanes, so
// the selected blocked layout must solve VW=64. The fixture covers the integer
// elementwise row of the advertised table; it is not an integer contraction.

module {
  micro.kernel @arm_vector_add_i8(%a: tensor<8x8xi8>, %b: tensor<8x8xi8>)
      -> tensor<8x8xi8> {
    %ta = micro.tile_view %a {shape = array<i64: 8, 8>} : tensor<8x8xi8> -> !micro.tile<8x8xi8, memory = #micro.memory<sram>>
    %tb = micro.tile_view %b {shape = array<i64: 8, 8>} : tensor<8x8xi8> -> !micro.tile<8x8xi8, memory = #micro.memory<sram>>
    %r = micro.vector "add" %ta, %tb : !micro.tile<8x8xi8, memory = #micro.memory<sram>>, !micro.tile<8x8xi8, memory = #micro.memory<sram>> -> !micro.tile<8x8xi8, memory = #micro.memory<sram>>
    micro.yield %r : !micro.tile<8x8xi8, memory = #micro.memory<sram>>
  }
}
