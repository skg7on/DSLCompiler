// RUN: llk-opt --llk-to-linalg %s | FileCheck %s

// Verify causal mask generates -inf for upper triangle positions.
func.func @causal_mask_test(%q: tensor<1x1x32x64xf32>, %k: tensor<1x1x32x64xf32>, %v: tensor<1x1x32x64xf32>, %init: tensor<1x1x32x64xf32>) -> tensor<1x1x32x64xf32> {
  %o = llk.attention ins(%q, %k, %v : tensor<1x1x32x64xf32>, tensor<1x1x32x64xf32>, tensor<1x1x32x64xf32>)
      outs(%init : tensor<1x1x32x64xf32>)
      {scale = 0.125 : f32, causal_mask = true, softmax_mode = #llk.softmax_mode<online>, math_mode = #llk.math_mode<bounded_fast>}
      -> tensor<1x1x32x64xf32>
  return %o : tensor<1x1x32x64xf32>
}
// CHECK: arith.cmpi
// CHECK: arith.select
