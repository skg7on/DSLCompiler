// The `llk-tune` driver in Micro mode (issue #50): a `micro.search_space` and a
// MachineModel YAML go in, a ranked schedule YAML comes out. The checks below
// describe that output file, not this input. The driver is invoked by the
// `LLKTuneSearchSpace` CTest entry, which passes this file as --input and
// FileChecks the --output YAML with --check-prefix=YAML.
//
// The legacy grid search is preserved when no search space is given: the old
// flags still write the JSON schedule_db entry, covered by `LLKTuneLegacyJson`
// with --check-prefix=LEGACY.

// A small fused_swiglu space with one legal candidate, so the driver's output
// is a single deterministic schedule record.
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

// The YAML output: one record carrying the workload identity, machine
// identity, tile decisions, and predicted metrics.
// YAML: schema_version: 1
// YAML: workload: fused_swiglu
// YAML: target: x86-avx2-cpu
// YAML: machine: {{.*}}machines/x86-avx2-cpu.yaml
// YAML: shape:
// YAML:   M_bucket: 2
// YAML:   M: 8
// YAML: dtype:
// YAML:   accumulator: f32
// YAML: candidate:
// YAML:     hierarchy:
// YAML:       worker: [8, 64, 64]
// YAML:       fragment: [16, 16, 32]
// YAML:     layout:
// YAML:       input: row_major
// YAML:     memory_path: [dram, sram, acc]
// YAML:     owner_mapping:
// YAML:       outer: worker
// YAML:       fragment: vector_engine
// YAML:     tail_policy: mask
// YAML: metrics:
// YAML:   perf_level: 1
// YAML:   predicted_cycles: {{[0-9]+}}
// YAML: measurement:
// YAML:   measured: false

// The legacy JSON path still writes a schedule_db entry with the grid search's
// tile parameters.
// LEGACY: "entries"
// LEGACY: "schedule"
// LEGACY: "num_threads"
// LEGACY: "version"
