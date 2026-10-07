// One legal schedule for a quick, reproducible CLI introduction.
micro.search_space @fused_swiglu_M8_N64_K64 attributes {workload = "fused_swiglu"} {
  micro.param "BM" {kind = "integer", choices = [8 : i64]}
  micro.param "BN" {kind = "integer", choices = [64 : i64]}
  micro.param "BK" {kind = "integer", choices = [64 : i64]}
  micro.param "VM" {kind = "integer", choices = [1 : i64]}
  micro.param "VN" {kind = "integer", choices = [4 : i64]}
  micro.param "vector_width" {kind = "integer", choices = [8 : i64]}
  micro.param "num_threads" {kind = "integer", choices = [8 : i64]}
  micro.param "grain_size" {kind = "integer", choices = [1 : i64]}
  micro.param "pipeline_stages" {kind = "integer", choices = [1 : i64]}
  micro.param "tile_layout" {kind = "layout", choices = ["row_major"]}
  micro.param "memory_path" {kind = "memory_path", choices = ["dram:sram:acc"]}
  micro.param "owner_mapping" {kind = "owner_mapping", choices = ["worker/vector_engine"]}
  micro.param "fragment_shape" {kind = "fragment_shape", choices = ["16x16x32"]}
  micro.param "tail_policy" {kind = "tail_policy", choices = ["mask"]}
  micro.constraint "sram_capacity" {params = ["BM", "BN", "BK"]}
  micro.constraint "acc_capacity" {params = ["BM", "BN"]}
  micro.constraint "mapping_extent" {params = ["num_threads", "BM", "BN"]}
  micro.objective {direction = "minimize", metric = "latency_cycles", secondary = ["matrix_utilization", "dram_bytes"]}
}
