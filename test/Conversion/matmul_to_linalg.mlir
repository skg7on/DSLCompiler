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
// CHECK: %[[PROMOTED:.*]] = linalg.generic
// CHECK: arith.extf {{.*}} : bf16 to f32
// CHECK: linalg.matmul ins(%arg0, %arg1 : tensor<16x64xbf16>, tensor<64x64xbf16>) outs(%[[PROMOTED]] : tensor<16x64xf32>) -> tensor<16x64xf32>
// CHECK: linalg.generic
// CHECK: arith.truncf {{.*}} : f32 to bf16
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

func.func @dynamic_init(%a: tensor<?x2xf16>, %b: tensor<2x1xf16>, %init: tensor<?x1xf16>) -> tensor<?x1xf16> {
  %y = llk.matmul ins(%a, %b : tensor<?x2xf16>, tensor<2x1xf16>)
      outs(%init : tensor<?x1xf16>)
      {accumulator_type = f32, math_mode = #llk.math_mode<strict>} -> tensor<?x1xf16>
  return %y : tensor<?x1xf16>
}
// CHECK-LABEL: func.func @dynamic_init
// CHECK: tensor.dim %arg2
// CHECK: arith.extf {{.*}} : f16 to f32
// CHECK: linalg.matmul
// CHECK-SAME: tensor<?x1xf32>
// CHECK: arith.truncf {{.*}} : f32 to f16
// CHECK: return

func.func @equal_width_formats(%a: tensor<1x2xf16>, %b: tensor<2x1xf16>, %init: tensor<1x1xf16>) -> tensor<1x1xf16> {
  %y = llk.matmul ins(%a, %b : tensor<1x2xf16>, tensor<2x1xf16>)
      outs(%init : tensor<1x1xf16>)
      {accumulator_type = bf16, math_mode = #llk.math_mode<strict>} -> tensor<1x1xf16>
  return %y : tensor<1x1xf16>
}
// CHECK-LABEL: func.func @equal_width_formats
// CHECK: arith.extf {{.*}} : f16 to f32
// CHECK: arith.truncf {{.*}} : f32 to bf16
// CHECK: linalg.matmul
// CHECK-SAME: tensor<1x2xbf16>, tensor<2x1xbf16>
// CHECK: arith.extf {{.*}} : bf16 to f32
// CHECK: arith.truncf {{.*}} : f32 to f16
// CHECK: return
