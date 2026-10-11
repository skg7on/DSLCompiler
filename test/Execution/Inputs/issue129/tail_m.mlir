module {
  func.func @matmul(%a: tensor<5x8xbf16>, %b: tensor<8x8xbf16>,
                    %init: tensor<5x8xbf16>) -> tensor<5x8xbf16> {
    %y = llk.matmul ins(%a, %b : tensor<5x8xbf16>, tensor<8x8xbf16>)
        outs(%init : tensor<5x8xbf16>)
        {accumulator_type = f32, math_mode = #llk.math_mode<strict>}
        -> tensor<5x8xbf16>
    return %y : tensor<5x8xbf16>
  }
}
