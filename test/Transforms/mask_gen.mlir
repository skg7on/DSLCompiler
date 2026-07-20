// RUN: llk-opt --llk-to-linalg --tile-and-vectorize %s | FileCheck %s
//
// Test mask generation for N-tails not multiple of VN=8.
// N=127 is not a multiple of VN=8, so the last vector iteration
// must use vector.create_mask / vector.mask for the tail.
//
// Pipeline: LLK dialect -> Linalg -> tiled (scf.for) -> Vector SIMD (masked)
// Tile sizes: BM=32, BN=64, BK=64, VM=4, VN=8

func.func @swiglu_odd_n(%x: tensor<32x64xbf16>, %wg: tensor<64x127xbf16>, %wu: tensor<64x127xbf16>, %init: tensor<32x127xbf16>) -> tensor<32x127xbf16> {
  %y = llk.fused_swiglu ins(%x, %wg, %wu : tensor<32x64xbf16>, tensor<64x127xbf16>, tensor<64x127xbf16>)
      outs(%init : tensor<32x127xbf16>)
      {accumulator_type = f32, activation = #llk.activation<silu>, math_mode = #llk.math_mode<bounded_fast>}
      -> tensor<32x127xbf16>
  return %y : tensor<32x127xbf16>
}
// CHECK-LABEL: func.func @swiglu_odd_n
// CHECK: vector.create_mask
// CHECK: vector.mask
// N=127, not multiple of VN=8 → masked tail required
