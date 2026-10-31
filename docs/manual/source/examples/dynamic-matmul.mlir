func.func @dynamic_matmul(%a: tensor<?x?xbf16>, %b: tensor<?x?xbf16>,
                          %init: tensor<?x?xbf16>) -> tensor<?x?xbf16> {
  %y = llk.matmul ins(%a, %b : tensor<?x?xbf16>, tensor<?x?xbf16>)
      outs(%init : tensor<?x?xbf16>)
      {accumulator_type = f32, math_mode = #llk.math_mode<bounded_fast>}
      -> tensor<?x?xbf16>
  return %y : tensor<?x?xbf16>
}
