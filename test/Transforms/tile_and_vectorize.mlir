// RUN: llk-opt --llk-to-linalg --tile-and-vectorize %s | FileCheck %s
//
// Test that the TileAndVectorize pass produces vectorized IR from LLK ops.
// Pipeline: LLK dialect -> Linalg -> tiled (scf.for + scf.forall) -> Vector SIMD
//
// Input: llk.fused_swiglu with concrete shapes
// Parameters: BM=32, BN=64, BK=64, VM=4, VN=8
//
// Expected output: scf.forall, scf.for, vector.contract, vector.transfer_read/write

func.func @swiglu_vectorize(%x: tensor<64x128xbf16>, %wg: tensor<128x256xbf16>, %wu: tensor<128x256xbf16>, %init: tensor<64x256xbf16>) -> tensor<64x256xbf16> {
  %y = llk.fused_swiglu ins(%x, %wg, %wu : tensor<64x128xbf16>, tensor<128x256xbf16>, tensor<128x256xbf16>)
      outs(%init : tensor<64x256xbf16>)
      {accumulator_type = f32, activation = #llk.activation<silu>, math_mode = #llk.math_mode<bounded_fast>}
      -> tensor<64x256xbf16>
  return %y : tensor<64x256xbf16>
}

// CHECK-LABEL: func.func @swiglu_vectorize
// CHECK: scf.forall
// CHECK: scf.for
// CHECK: vector.contract
// CHECK: vector.transfer_read
// CHECK: vector.transfer_write
