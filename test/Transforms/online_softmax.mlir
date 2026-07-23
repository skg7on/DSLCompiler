// RUN: llk-opt --llk-to-linalg %s | FileCheck %s

func.func @online_softmax_structure(%q: tensor<1x1x128x128xf32>, %k: tensor<1x1x128x128xf32>, %v: tensor<1x1x128x128xf32>, %init: tensor<1x1x128x128xf32>) -> tensor<1x1x128x128xf32> {
  %o = llk.attention ins(%q, %k, %v : tensor<1x1x128x128xf32>, tensor<1x1x128x128xf32>, tensor<1x1x128x128xf32>)
      outs(%init : tensor<1x1x128x128xf32>)
      {scale = 0.08838834764 : f32, causal_mask = true, softmax_mode = #llk.softmax_mode<online>, math_mode = #llk.math_mode<bounded_fast>}
      -> tensor<1x1x128x128xf32>
  return %o : tensor<1x1x128x128xf32>
}
// CHECK: arith.maximumf
// CHECK: math.exp
// CHECK: arith.subf
// CHECK: arith.divf
