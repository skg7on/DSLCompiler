func.func @matmul(%a: tensor<16x64xbf16>, %b: tensor<64x64xbf16>,
                  %init: tensor<16x64xbf16>) -> tensor<16x64xbf16> {
  %y = llk.matmul ins(%a, %b : tensor<16x64xbf16>, tensor<64x64xbf16>)
      outs(%init : tensor<16x64xbf16>)
      {accumulator_type = f32, math_mode = #llk.math_mode<bounded_fast>}
      -> tensor<16x64xbf16>
  return %y : tensor<16x64xbf16>
}

// The source function is left alone.
func.func @rope_test(%x: tensor<1x1x128x64xf32>, %pos: tensor<128xi64>, %init: tensor<1x1x128x64xf32>) -> tensor<1x1x128x64xf32> {
  %y = llk.rope ins(%x, %pos : tensor<1x1x128x64xf32>, tensor<128xi64>)
      outs(%init : tensor<1x1x128x64xf32>)
      {theta = 10000.0 : f64, math_mode = #llk.math_mode<bounded_fast>}
      -> tensor<1x1x128x64xf32>
  return %y : tensor<1x1x128x64xf32>
}
// CHECK: llk.matmul
// CHECK: llk.rope
// CHECK-NOT: micro.search_space @rope
// CHECK: micro.search_space @matmul_M16_N64_K64
// CHECK-NOT: micro.search_space
