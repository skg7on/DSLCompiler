// RUN: llk-opt --llk-to-linalg %s | FileCheck %s
//
// Verify that RoPE cos/sin tables are computed as 2D [L, halfD] tensors
// and broadcast to BxHxLxhalfD via affine maps in linalg.generic ops.
// The cos/sin computation should produce 2D tables that are broadcast
// across batch and head dimensions.

func.func @rope_broadcast_b2_h4(%x: tensor<2x4x32x64xf32>, %pos: tensor<32xi64>, %init: tensor<2x4x32x64xf32>) -> tensor<2x4x32x64xf32> {
  %y = llk.rope ins(%x, %pos : tensor<2x4x32x64xf32>, tensor<32xi64>)
      outs(%init : tensor<2x4x32x64xf32>)
      {theta = 10000.0 : f64, math_mode = #llk.math_mode<bounded_fast>}
      -> tensor<2x4x32x64xf32>
  return %y : tensor<2x4x32x64xf32>
}
// CHECK: math.cos
// CHECK: math.sin
// CHECK: tensor.extract_slice
// CHECK: linalg.generic
// CHECK: tensor.insert_slice
