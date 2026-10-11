module {
  func.func @matmul(%a: tensor<8x32xbf16>, %b: tensor<32x16xbf16>,
                    %init: tensor<8x16xbf16>) -> tensor<8x16xbf16> {
    %result = llk.matmul ins(%a, %b : tensor<8x32xbf16>, tensor<32x16xbf16>)
        outs(%init : tensor<8x16xbf16>)
        {accumulator_type = f32, math_mode = #llk.math_mode<bounded_fast>}
        -> tensor<8x16xbf16>
    return %result : tensor<8x16xbf16>
  }

  micro.search_space @mapped_matmul attributes {workload = "matmul"} {
    micro.param "BM" {kind = "integer", choices = [8 : i64]}
    micro.param "BN" {kind = "integer", choices = [16 : i64]}
    micro.param "BK" {kind = "integer", choices = [32 : i64]}
    micro.param "fragment" {kind = "fragment_shape", choices = ["8x16x32"]}
    micro.param "tail" {kind = "tail_policy", choices = ["none"]}
    micro.param "pipeline_stages" {kind = "integer", choices = [1 : i64]}
    micro.param "vector_width" {kind = "integer", choices = [8 : i64]}
    micro.param "num_threads" {kind = "integer", choices = [1 : i64]}
    micro.param "grain_size" {kind = "integer", choices = [1 : i64]}
    micro.constraint "mma_compatible" {params = ["BM", "BN", "BK"]}
    micro.constraint "acc_capacity" {params = ["BM", "BN"]}
    micro.objective {direction = "minimize", metric = "latency_cycles"}
  }
}
