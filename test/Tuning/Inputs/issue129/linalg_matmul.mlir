module {
  func.func @linalg_source(%a: tensor<8x32xbf16>, %b: tensor<32x16xbf16>,
                           %init: tensor<8x16xf32>) -> tensor<8x16xf32> {
    %result = linalg.matmul ins(%a, %b : tensor<8x32xbf16>, tensor<32x16xbf16>)
        outs(%init : tensor<8x16xf32>) -> tensor<8x16xf32>
    return %result : tensor<8x16xf32>
  }
}
