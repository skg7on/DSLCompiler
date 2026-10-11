module {
  func.func @swiglu(%x: tensor<8x32xbf16>, %wg: tensor<32x16xbf16>,
                    %wu: tensor<32x16xbf16>, %init: tensor<8x16xbf16>)
      -> tensor<8x16xbf16> {
    %result = llk.fused_swiglu ins(%x, %wg, %wu : tensor<8x32xbf16>,
                                    tensor<32x16xbf16>, tensor<32x16xbf16>)
        outs(%init : tensor<8x16xbf16>)
        {accumulator_type = f32, activation = #llk.activation<silu>,
         math_mode = #llk.math_mode<bounded_fast>}
        -> tensor<8x16xbf16>
    return %result : tensor<8x16xbf16>
  }
}
