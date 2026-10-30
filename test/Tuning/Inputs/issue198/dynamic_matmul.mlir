module {
  func.func @matmul(%a: tensor<?x?xbf16>, %b: tensor<?x?xbf16>,
                    %init: tensor<?x?xbf16>) -> tensor<?x?xbf16> {
    %result = llk.matmul ins(%a, %b : tensor<?x?xbf16>, tensor<?x?xbf16>)
        outs(%init : tensor<?x?xbf16>)
        {accumulator_type = f32, math_mode = #llk.math_mode<bounded_fast>}
        -> tensor<?x?xbf16>
    return %result : tensor<?x?xbf16>
  }

  micro.search_space @mapped_matmul attributes {workload = "matmul"} {
    micro.param "BM" {kind = "integer", choices = [8 : i64]}
    micro.param "BN" {kind = "integer", choices = [16 : i64]}
    micro.param "BK" {kind = "integer", choices = [32 : i64]}
    micro.param "fragment" {kind = "fragment_shape", choices = ["8x16x32"]}
    micro.param "tail" {kind = "tail_policy", choices = ["bounded"]}
    micro.param "pipeline_stages" {kind = "integer", choices = [1 : i64]}
    micro.param "vector_width" {kind = "integer", choices = [8 : i64]}
    micro.param "num_threads" {kind = "integer", choices = [1 : i64]}
    micro.param "grain_size" {kind = "integer", choices = [1 : i64]}
    micro.constraint "mma_compatible" {params = ["BM", "BN", "BK"]}
    micro.constraint "acc_capacity" {params = ["BM", "BN"]}
    micro.objective {direction = "minimize", metric = "latency_cycles"}
  }
}
