func.func @matmul(%a: tensor<16x64xbf16>, %b: tensor<64x64xbf16>,
                  %init: tensor<16x64xbf16>) -> tensor<16x64xbf16> {
  // expected-remark @below {{M=16 -> bucket 2}}
  %y = llk.matmul ins(%a, %b : tensor<16x64xbf16>, tensor<64x64xbf16>)
      outs(%init : tensor<16x64xbf16>)
      {accumulator_type = f32, math_mode = #llk.math_mode<bounded_fast>}
      -> tensor<16x64xbf16>
  return %y : tensor<16x64xbf16>
}

// The source function is left alone.
func.func @attention_test(%q: tensor<1x1x128x128xf32>, %k: tensor<1x1x128x128xf32>, %v: tensor<1x1x128x128xf32>, %init: tensor<1x1x128x128xf32>) -> tensor<1x1x128x128xf32> {
  // expected-remark @below {{M=1 -> bucket 0}}
  %o = llk.attention ins(%q, %k, %v : tensor<1x1x128x128xf32>, tensor<1x1x128x128xf32>, tensor<1x1x128x128xf32>)
      outs(%init : tensor<1x1x128x128xf32>)
      {scale = 0.08838834764 : f32, causal_mask = true, softmax_mode = #llk.softmax_mode<online>, math_mode = #llk.math_mode<bounded_fast>}
      -> tensor<1x1x128x128xf32>
  return %o : tensor<1x1x128x128xf32>
}
func.func @dynamic_shape(%x: tensor<?x64xbf16>, %wg: tensor<64x64xbf16>,
                         %wu: tensor<64x64xbf16>, %init: tensor<?x64xbf16>)
    -> tensor<?x64xbf16> {
  // expected-remark @below {{dynamic M → bucket -1 (generic)}}
  %y = llk.fused_swiglu ins(%x, %wg, %wu : tensor<?x64xbf16>,
                            tensor<64x64xbf16>, tensor<64x64xbf16>)
      outs(%init : tensor<?x64xbf16>)
      {accumulator_type = f32, activation = #llk.activation<silu>,
       math_mode = #llk.math_mode<bounded_fast>}
      -> tensor<?x64xbf16>
  return %y : tensor<?x64xbf16>
}

