// Tutorial source: Y = SiLU(X * Wg) * (X * Wu).
func.func @swiglu(%x: tensor<16x64xbf16>, %wg: tensor<64x64xbf16>,
                  %wu: tensor<64x64xbf16>, %init: tensor<16x64xbf16>)
    -> tensor<16x64xbf16> {
  %y = llk.fused_swiglu ins(%x, %wg, %wu : tensor<16x64xbf16>,
                            tensor<64x64xbf16>, tensor<64x64xbf16>)
      outs(%init : tensor<16x64xbf16>)
      {accumulator_type = f32, activation = #llk.activation<silu>,
       math_mode = #llk.math_mode<bounded_fast>}
      -> tensor<16x64xbf16>
  return %y : tensor<16x64xbf16>
}
