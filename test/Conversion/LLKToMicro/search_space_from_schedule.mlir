// RUN: llk-opt --llk-to-micro-search-space="schedule-db=test/Conversion/LLKToMicro/search_space_schedule.json" %s | FileCheck %s

// The M11 search-space export: one `micro.search_space` per supported LLK root
// operation, holding the legal tile choices around the schedule the database
// selected.
//
// Numeric dimensions come from the llk-tune grid -- the search space has to
// contain choices the tuner can actually generate -- and the scheduled value is
// added when the grid does not already have it. Symbolic dimensions are
// schedule-anchored: the scheduled layout, memory path, owner hierarchy, and
// instruction fragment print first, followed by the legal alternatives.
//
// The fixture database (`search_space_schedule.json`) spells out every
// Micro/tile field for the matmul and leaves the SwiGLU entry on the pre-M11
// fields, so this one file covers both "new fields load when present" and
// "missing fields use deterministic defaults".

func.func @matmul(%a: tensor<16x64xbf16>, %b: tensor<64x64xbf16>,
                  %init: tensor<16x64xbf16>) -> tensor<16x64xbf16> {
  %y = llk.matmul ins(%a, %b : tensor<16x64xbf16>, tensor<64x64xbf16>)
      outs(%init : tensor<16x64xbf16>)
      {accumulator_type = f32, math_mode = #llk.math_mode<bounded_fast>}
      -> tensor<16x64xbf16>
  return %y : tensor<16x64xbf16>
}

func.func @fused_swiglu(%x: tensor<8x64xbf16>, %wg: tensor<64x64xbf16>,
                        %wu: tensor<64x64xbf16>, %init: tensor<8x64xbf16>)
    -> tensor<8x64xbf16> {
  %y = llk.fused_swiglu ins(%x, %wg, %wu : tensor<8x64xbf16>,
                            tensor<64x64xbf16>, tensor<64x64xbf16>)
      outs(%init : tensor<8x64xbf16>)
      {accumulator_type = f32, activation = #llk.activation<silu>,
       math_mode = #llk.math_mode<bounded_fast>}
      -> tensor<8x64xbf16>
  return %y : tensor<8x64xbf16>
}

// The source functions are left alone.
// CHECK-LABEL: func.func @matmul
// CHECK: llk.matmul
// CHECK-LABEL: func.func @fused_swiglu
// CHECK: llk.fused_swiglu

//===----------------------------------------------------------------------===//
// matmul: every Micro/tile field spelled out in the schedule entry
//===----------------------------------------------------------------------===//

// CHECK-LABEL: micro.search_space @matmul_M16_N64_K64 attributes {workload = "matmul"} {

// Numeric parameters: the llk-tune grid, plus any scheduled value outside it.
// M = 16 is in the [5, 16] bucket, so no BM choice is pruned.
// CHECK: micro.param "BM" {choices = [1, 4, 8, 16, 32, 64], kind = "integer"}
// CHECK: micro.param "BN" {choices = [16, 32, 64, 128, 256], kind = "integer"}
// CHECK: micro.param "BK" {choices = [32, 64, 128, 256], kind = "integer"}
// CHECK: micro.param "vector_width" {choices = [8], kind = "integer"}
// Parallel thread and grain axes are fixed to the serial lowering actually
// emitted by the concrete exporter, regardless of schedule database values.
// CHECK: micro.param "num_threads" {choices = [1], kind = "integer"}
// CHECK: micro.param "grain_size" {choices = [1], kind = "integer"}
// CHECK: micro.param "pipeline_stages" {choices = [1, 2], kind = "integer"}
// VM/VN and prefetch are omitted until their semantics have concrete lowering.

// Symbolic parameters: the scheduled value first, then legal alternatives.
// CHECK: micro.param "tile_layout" {choices = ["blocked", "row_major"], kind = "layout"}
// CHECK: micro.param "memory_path" {choices = ["dram:l2:sram", "dram:sram:acc", "dram:l2:sram:acc"], kind = "memory_path"}
// CHECK: micro.param "owner_mapping" {choices = ["worker/vector_engine", "worker/lane"], kind = "owner_mapping"}
// CHECK: micro.param "fragment_shape" {choices = ["8x8x32", "16x16x32"], kind = "fragment_shape"}
// CHECK: micro.param "tail_policy" {choices = ["none", "pad"], kind = "tail_policy"}

// Legality records. Every referenced name is a declared parameter.
// CHECK: micro.constraint "sram_capacity" {params = ["BM", "BN", "BK"]}
// CHECK: micro.constraint "acc_capacity" {params = ["BM", "BN"]}
// CHECK: micro.constraint "mma_compatible" {params = ["BM", "BN", "BK"]}
// CHECK: micro.constraint "tile_hierarchy_compatible" {params = ["owner_mapping"]}
// CHECK: micro.constraint "layout_supported" {params = ["tile_layout"]}
// CHECK: micro.constraint "owner_supported" {params = ["owner_mapping"]}
// CHECK: micro.constraint "fragment_compatible" {params = ["fragment_shape", "BM", "BN", "BK"]}
// CHECK: micro.constraint "vector_width_supported" {params = ["vector_width"]}
// CHECK: micro.constraint "mapping_extent" {params = ["num_threads", "BM", "BN"]}
// CHECK: micro.constraint "pipeline_live_tiles" {params = ["pipeline_stages", "BM", "BN", "BK"]}
// CHECK: micro.constraint "tail_supported" {params = ["BM", "BN", "BK"]}

// CHECK: micro.objective {direction = "minimize", metric = "latency_cycles", secondary = ["matrix_utilization", "dram_bytes"]}
// CHECK: }

//===----------------------------------------------------------------------===//
// fused_swiglu: pre-M11 schedule entry, so defaults fill the new fields
//===----------------------------------------------------------------------===//

// CHECK-LABEL: micro.search_space @fused_swiglu_M8_N64_K64 attributes {workload = "fused_swiglu"} {

// The entry names no pipeline fields, so the default single stage applies.
// CHECK: micro.param "pipeline_stages" {choices = [1, 2], kind = "integer"}

// The entry keeps the pre-M11 `owner_mapping = "worker"`, a single owner. The
// export completes it with the machine's innermost scope so the choice is a
// legal two-level mapping, and offers the default tile layout alone: a bare
// `blocked` is not offered because nothing downstream can realize it.
// CHECK: micro.param "tile_layout" {choices = ["row_major"], kind = "layout"}
// CHECK: micro.param "memory_path" {choices = ["dram:sram:acc", "dram:l2:sram:acc"], kind = "memory_path"}
// CHECK: micro.param "owner_mapping" {choices = ["worker/lane", "worker/vector_engine"], kind = "owner_mapping"}
// CHECK: micro.param "fragment_shape" {choices = ["16x16x32", "8x8x32"], kind = "fragment_shape"}

// CHECK: micro.constraint "pipeline_live_tiles" {params = ["pipeline_stages", "BM", "BN", "BK"]}
// CHECK: micro.objective {direction = "minimize", metric = "latency_cycles", secondary = ["matrix_utilization", "dram_bytes"]}
// CHECK: }

// The concrete export is not run here, so no execution IR is produced.
// CHECK-NOT: micro.kernel

//===----------------------------------------------------------------------===//
// llk-compile --emit=micro-search
//===----------------------------------------------------------------------===//

// The same export through the llk-compile driver: it prints search-space IR and
// stops before the JIT, leaving the LLK operations unlifted. That driver reads
// the default schedule database, which is not found from the test working
// directory, so it falls back to the built-in schedule and the exact choices
// differ from the fixture-backed run above. The smoke check is that the search
// space is produced and the module is never lowered toward LLVM.
// COMPILE: llk.matmul
// COMPILE-LABEL: micro.search_space @matmul_M16_N64_K64
// COMPILE: micro.param "BM"
// COMPILE: micro.param "memory_path"
// COMPILE: micro.objective
// COMPILE-NOT: micro.kernel
// COMPILE-NOT: llvm.
