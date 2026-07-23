// RUN: llk-opt --llk-to-linalg %s | FileCheck %s

func.func @rope_even_odd_interleave(%x: tensor<1x1x32x64xf32>, %pos: tensor<32xi64>, %init: tensor<1x1x32x64xf32>) -> tensor<1x1x32x64xf32> {
  %y = llk.rope ins(%x, %pos : tensor<1x1x32x64xf32>, tensor<32xi64>)
      outs(%init : tensor<1x1x32x64xf32>)
      {theta = 10000.0 : f64, math_mode = #llk.math_mode<bounded_fast>}
      -> tensor<1x1x32x64xf32>
  return %y : tensor<1x1x32x64xf32>
}
// CHECK: tensor.extract_slice
// CHECK: tensor.extract_slice
// CHECK-SAME: [1, 1, 1, 2]
// CHECK: tensor.insert_slice
// CHECK: tensor.insert_slice
