module {
  func.func @two_roots(%a0: tensor<8x32xbf16>, %b0: tensor<32x16xbf16>,
                       %init0: tensor<8x16xbf16>, %a1: tensor<8x32xbf16>,
                       %b1: tensor<32x16xbf16>, %init1: tensor<8x16xbf16>)
      -> (tensor<8x16xbf16>, tensor<8x16xbf16>) {
    %first = llk.matmul ins(%a0, %b0 : tensor<8x32xbf16>, tensor<32x16xbf16>)
        outs(%init0 : tensor<8x16xbf16>)
        {accumulator_type = f32, math_mode = #llk.math_mode<bounded_fast>}
        -> tensor<8x16xbf16>
    %second = llk.matmul ins(%a1, %b1 : tensor<8x32xbf16>, tensor<32x16xbf16>)
        outs(%init1 : tensor<8x16xbf16>)
        {accumulator_type = f32, math_mode = #llk.math_mode<bounded_fast>}
        -> tensor<8x16xbf16>
    return %first, %second : tensor<8x16xbf16>, tensor<8x16xbf16>
  }
}
