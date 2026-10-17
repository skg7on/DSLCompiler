#id = affine_map<(d0, d1) -> (d0, d1)>
module {
  func.func @elementwise(%a: tensor<8x16xf32>, %b: tensor<8x16xf32>)
      -> tensor<8x16xf32> {
    %init = tensor.empty() : tensor<8x16xf32>
    %out = linalg.generic {
      indexing_maps = [#id, #id, #id],
      iterator_types = ["parallel", "parallel"]
    } ins(%a, %b : tensor<8x16xf32>, tensor<8x16xf32>)
      outs(%init : tensor<8x16xf32>) {
    ^bb0(%x: f32, %y: f32, %unused: f32):
      %sum = arith.addf %x, %y : f32
      linalg.yield %sum : f32
    } -> tensor<8x16xf32>
    return %out : tensor<8x16xf32>
  }
}
