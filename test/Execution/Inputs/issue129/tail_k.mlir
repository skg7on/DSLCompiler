module {
  func.func @matmul(%a: tensor<4x7xbf16>, %b: tensor<7x8xbf16>,
                    %init: tensor<4x8xbf16>) -> tensor<4x8xbf16> {
    %y = llk.matmul ins(%a, %b : tensor<4x7xbf16>, tensor<7x8xbf16>)
        outs(%init : tensor<4x8xbf16>)
        {accumulator_type = f32, math_mode = #llk.math_mode<strict>}
        -> tensor<4x8xbf16>
    return %y : tensor<4x8xbf16>
  }
}
