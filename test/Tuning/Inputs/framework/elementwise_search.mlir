// The bounded structured elementwise source with two tile axes. The four
// bindings T0 x T1 are four genuinely different source schedules: the full
// tile (8x16) and three tiled variants the adapter emits as nested spatial
// loops.
module {
  func.func @elementwise(%a: tensor<8x16xf32>, %b: tensor<8x16xf32>)
      -> tensor<8x16xf32> {
    %init = tensor.empty() : tensor<8x16xf32>
    %out = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                       affine_map<(d0, d1) -> (d0, d1)>,
                       affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]
    } ins(%a, %b : tensor<8x16xf32>, tensor<8x16xf32>)
      outs(%init : tensor<8x16xf32>) {
    ^bb0(%x: f32, %y: f32, %unused: f32):
      %sum = arith.addf %x, %y : f32
      linalg.yield %sum : f32
    } -> tensor<8x16xf32>
    return %out : tensor<8x16xf32>
  }

  micro.search_space @elementwise_tiles attributes {workload = "elementwise"} {
    micro.param "T0" {kind = "integer", role = "axis:d0",
                      choices = [4 : i64, 8 : i64]}
    micro.param "T1" {kind = "integer", role = "axis:d1",
                      choices = [8 : i64, 16 : i64]}
    micro.objective {direction = "minimize", metric = "latency_cycles"}
  }
}
