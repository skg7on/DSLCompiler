func.func @matmul(%a: tensor<16x64xbf16>, %b: tensor<64x64xbf16>,
                  %init: tensor<16x64xbf16>) -> tensor<16x64xbf16> {
  %y = llk.matmul ins(%a, %b : tensor<16x64xbf16>, tensor<64x64xbf16>)
      outs(%init : tensor<16x64xbf16>)
      {accumulator_type = f32, math_mode = #llk.math_mode<bounded_fast>}
      -> tensor<16x64xbf16>
  return %y : tensor<16x64xbf16>
}

// The source function is left alone.
// CHECK-LABEL: func.func @matmul
// CHECK-NOT: llk.matmul
// CHECK: linalg.matmul ins(%arg0, %arg1 : tensor<16x64xbf16>, tensor<64x64xbf16>) outs(%arg2 : tensor<16x64xbf16>) -> tensor<16x64xbf16>
// CHECK: return

func.func @nonzero_init(%a: tensor<2x4xf32>, %b: tensor<4x8xf32>) -> tensor<2x8xf32> {
  %init = arith.constant dense<2.0> : tensor<2x8xf32>
  %y = llk.matmul ins(%a, %b : tensor<2x4xf32>, tensor<4x8xf32>)
      outs(%init : tensor<2x8xf32>)
      {accumulator_type = f32, math_mode = #llk.math_mode<strict>}
      -> tensor<2x8xf32>
  return %y : tensor<2x8xf32>
}
// CHECK-LABEL: func.func @nonzero_init
// CHECK: %[[INIT:.*]] = arith.constant dense<2.000000e+00> : tensor<2x8xf32>
// CHECK-NOT: llk.matmul
// CHECK: linalg.matmul ins(%arg0, %arg1 : tensor<2x4xf32>, tensor<4x8xf32>) outs(%[[INIT]] : tensor<2x8xf32>) -> tensor<2x8xf32>
// CHECK: return
