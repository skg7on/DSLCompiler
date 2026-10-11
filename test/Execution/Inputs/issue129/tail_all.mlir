module {
  func.func @matmul(%a: tensor<5x7xbf16>, %b: tensor<7x9xbf16>,
                    %init: tensor<5x9xf32>) -> tensor<5x9xf32> {
    %y = llk.matmul ins(%a, %b : tensor<5x7xbf16>, tensor<7x9xbf16>)
        outs(%init : tensor<5x9xf32>)
        {accumulator_type = f32, math_mode = #llk.math_mode<bounded_fast>}
        -> tensor<5x9xf32>
    return %y : tensor<5x9xf32>
  }
}
