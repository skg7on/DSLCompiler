// RUN: llk-opt --verify-diagnostics --split-input-file %s

// Invalid search-space constructs must produce actionable diagnostics.

//===----------------------------------------------------------------------===//
// micro.search_space
//===----------------------------------------------------------------------===//

// The workload is required and must be non-empty.

func.func @empty_workload() {
  // expected-error @+1 {{workload must be non-empty}}
  micro.search_space @bad attributes {workload = ""} {
    micro.param "BM" {kind = "integer", choices = [8 : i64]}
  }
  return
}

// A search space must declare at least one tunable parameter.

func.func @no_params() {
  // expected-error @+1 {{search space requires at least one micro.param}}
  micro.search_space @bad attributes {workload = "fused_swiglu"} {
    micro.constraint "sram_capacity" {params = ["BM"]}
  }
  return
}

// Parameter names must be unique within their search space.

func.func @duplicate_param() {
  micro.search_space @bad attributes {workload = "fused_swiglu"} {
    micro.param "BM" {kind = "integer", choices = [8 : i64]}
    // expected-error @+1 {{duplicate parameter name 'BM'}}
    micro.param "BM" {kind = "integer", choices = [16 : i64]}
  }
  return
}

//===----------------------------------------------------------------------===//
// micro.param — name, kind, and domain shape
//===----------------------------------------------------------------------===//

func.func @empty_param_name() {
  micro.search_space @bad attributes {workload = "fused_swiglu"} {
    // expected-error @+1 {{parameter name must be non-empty}}
    micro.param "" {kind = "integer", choices = [8 : i64]}
  }
  return
}

func.func @unknown_param_kind() {
  micro.search_space @bad attributes {workload = "fused_swiglu"} {
    // expected-error @+1 {{unsupported parameter kind 'vector_width'}}
    micro.param "BM" {kind = "vector_width", choices = [8 : i64]}
  }
  return
}

func.func @empty_choices() {
  micro.search_space @bad attributes {workload = "fused_swiglu"} {
    // expected-error @+1 {{parameter must declare at least one choice}}
    micro.param "BM" {kind = "integer", choices = []}
  }
  return
}

// Integer domains must be positive integers in strictly increasing order, so
// that the printed IR has one canonical spelling.

func.func @integer_choices_not_integers() {
  micro.search_space @bad attributes {workload = "fused_swiglu"} {
    // expected-error @+1 {{integer parameter choices must be integers}}
    micro.param "BM" {kind = "integer", choices = ["8"]}
  }
  return
}

func.func @integer_choices_not_positive() {
  micro.search_space @bad attributes {workload = "fused_swiglu"} {
    // expected-error @+1 {{integer parameter choices must be positive}}
    micro.param "BM" {kind = "integer", choices = [8 : i64, 0 : i64]}
  }
  return
}

func.func @integer_choices_not_increasing() {
  micro.search_space @bad attributes {workload = "fused_swiglu"} {
    // expected-error @+1 {{integer parameter choices must be unique and strictly increasing}}
    micro.param "BM" {kind = "integer", choices = [16 : i64, 8 : i64]}
  }
  return
}

func.func @integer_choices_duplicate() {
  micro.search_space @bad attributes {workload = "fused_swiglu"} {
    // expected-error @+1 {{integer parameter choices must be unique and strictly increasing}}
    micro.param "BM" {kind = "integer", choices = [8 : i64, 8 : i64]}
  }
  return
}

//===----------------------------------------------------------------------===//
// micro.param — symbolic domains
//===----------------------------------------------------------------------===//

func.func @symbolic_choices_not_strings() {
  micro.search_space @bad attributes {workload = "fused_swiglu"} {
    // expected-error @+1 {{symbolic parameter choices must be strings}}
    micro.param "tile_layout" {kind = "layout", choices = [8 : i64]}
  }
  return
}

func.func @unknown_layout_choice() {
  micro.search_space @bad attributes {workload = "fused_swiglu"} {
    // expected-error @+1 {{layout choice 'hilbert' is not a known layout kind}}
    micro.param "tile_layout" {kind = "layout", choices = ["row_major", "hilbert"]}
  }
  return
}

func.func @unknown_memory_path_space() {
  micro.search_space @bad attributes {workload = "fused_swiglu"} {
    // expected-error @+1 {{memory_path choice 'dram:bogus' has unknown memory space 'bogus'}}
    micro.param "path" {kind = "memory_path", choices = ["dram:sram", "dram:bogus"]}
  }
  return
}

func.func @memory_path_single_space() {
  micro.search_space @bad attributes {workload = "fused_swiglu"} {
    // expected-error @+1 {{memory_path choice 'sram' must join at least two memory spaces with ':'}}
    micro.param "path" {kind = "memory_path", choices = ["sram"]}
  }
  return
}

func.func @unknown_owner_mapping() {
  micro.search_space @bad attributes {workload = "fused_swiglu"} {
    // expected-error @+1 {{owner_mapping choice 'worker/bogus' has unknown owner 'bogus'}}
    micro.param "owner" {kind = "owner_mapping", choices = ["worker/bogus"]}
  }
  return
}

func.func @owner_mapping_single_owner() {
  micro.search_space @bad attributes {workload = "fused_swiglu"} {
    // expected-error @+1 {{owner_mapping choice 'worker' must join at least two owners with '/'}}
    micro.param "owner" {kind = "owner_mapping", choices = ["worker"]}
  }
  return
}

func.func @fragment_shape_not_triple() {
  micro.search_space @bad attributes {workload = "fused_swiglu"} {
    // expected-error @+1 {{fragment_shape choice '16x16' must be a positive MxNxK triple}}
    micro.param "frag" {kind = "fragment_shape", choices = ["16x16"]}
  }
  return
}

func.func @fragment_shape_not_positive() {
  micro.search_space @bad attributes {workload = "fused_swiglu"} {
    // expected-error @+1 {{fragment_shape choice '16x0x32' must be a positive MxNxK triple}}
    micro.param "frag" {kind = "fragment_shape", choices = ["16x0x32"]}
  }
  return
}

func.func @unsupported_tail_policy() {
  micro.search_space @bad attributes {workload = "fused_swiglu"} {
    // expected-error @+1 {{tail_policy choice 'none' is not supported (only 'mask')}}
    micro.param "tail" {kind = "tail_policy", choices = ["mask", "none"]}
  }
  return
}

//===----------------------------------------------------------------------===//
// micro.constraint
//===----------------------------------------------------------------------===//

func.func @unknown_constraint_kind() {
  micro.search_space @bad attributes {workload = "fused_swiglu"} {
    micro.param "BM" {kind = "integer", choices = [8 : i64]}
    // expected-error @+1 {{unsupported constraint kind 'fits_in_registers'}}
    micro.constraint "fits_in_registers" {params = ["BM"]}
  }
  return
}

func.func @constraint_unknown_param() {
  micro.search_space @bad attributes {workload = "fused_swiglu"} {
    micro.param "BM" {kind = "integer", choices = [8 : i64]}
    // expected-error @+1 {{constraint references unknown parameter 'BK'}}
    micro.constraint "sram_capacity" {params = ["BM", "BK"]}
  }
  return
}

func.func @constraint_no_params() {
  micro.search_space @bad attributes {workload = "fused_swiglu"} {
    micro.param "BM" {kind = "integer", choices = [8 : i64]}
    // expected-error @+1 {{constraint must reference at least one parameter}}
    micro.constraint "sram_capacity" {params = []}
  }
  return
}

//===----------------------------------------------------------------------===//
// micro.objective
//===----------------------------------------------------------------------===//

func.func @bad_objective_direction() {
  micro.search_space @bad attributes {workload = "fused_swiglu"} {
    micro.param "BM" {kind = "integer", choices = [8 : i64]}
    // expected-error @+1 {{objective direction must be 'minimize' or 'maximize'}}
    micro.objective {direction = "min", metric = "latency_cycles"}
  }
  return
}

func.func @unknown_primary_metric() {
  micro.search_space @bad attributes {workload = "fused_swiglu"} {
    micro.param "BM" {kind = "integer", choices = [8 : i64]}
    // expected-error @+1 {{unsupported objective metric 'energy'}}
    micro.objective {direction = "minimize", metric = "energy"}
  }
  return
}

func.func @unknown_secondary_metric() {
  micro.search_space @bad attributes {workload = "fused_swiglu"} {
    micro.param "BM" {kind = "integer", choices = [8 : i64]}
    // expected-error @+1 {{unsupported secondary metric 'throughput'}}
    micro.objective {direction = "maximize", metric = "matrix_utilization", secondary = ["throughput"]}
  }
  return
}

func.func @duplicate_secondary_metric() {
  micro.search_space @bad attributes {workload = "fused_swiglu"} {
    micro.param "BM" {kind = "integer", choices = [8 : i64]}
    // expected-error @+1 {{duplicate secondary metric 'dram_bytes'}}
    micro.objective {direction = "minimize", metric = "latency_cycles", secondary = ["dram_bytes", "dram_bytes"]}
  }
  return
}

func.func @multiple_objectives() {
  micro.search_space @bad attributes {workload = "fused_swiglu"} {
    micro.param "BM" {kind = "integer", choices = [8 : i64]}
    micro.objective {direction = "minimize", metric = "latency_cycles"}
    // expected-error @+1 {{at most one micro.objective is allowed}}
    micro.objective {direction = "minimize", metric = "dram_bytes"}
  }
  return
}

//===----------------------------------------------------------------------===//
// micro.candidate
//===----------------------------------------------------------------------===//

func.func @candidate_unknown_param() {
  micro.search_space @bad attributes {workload = "fused_swiglu"} {
    micro.param "BM" {kind = "integer", choices = [8 : i64]}
    // expected-error @+1 {{candidate binds unknown parameter 'BK'}}
    micro.candidate @c0 {bindings = {BM = 8 : i64, BK = 8 : i64}}
  }
  return
}

func.func @candidate_missing_param() {
  micro.search_space @bad attributes {workload = "fused_swiglu"} {
    micro.param "BM" {kind = "integer", choices = [8 : i64]}
    micro.param "BK" {kind = "integer", choices = [16 : i64]}
    // expected-error @+1 {{candidate does not bind parameter 'BK'}}
    micro.candidate @c0 {bindings = {BM = 8 : i64}}
  }
  return
}

func.func @candidate_value_not_in_domain() {
  micro.search_space @bad attributes {workload = "fused_swiglu"} {
    micro.param "BM" {kind = "integer", choices = [8 : i64]}
    // expected-error @+1 {{candidate value for 'BM' is not one of the declared choices}}
    micro.candidate @c0 {bindings = {BM = 64 : i64}}
  }
  return
}

func.func @candidate_wrong_value_type() {
  micro.search_space @bad attributes {workload = "fused_swiglu"} {
    micro.param "BM" {kind = "integer", choices = [8 : i64]}
    // expected-error @+1 {{candidate binding for integer parameter 'BM' must be an integer}}
    micro.candidate @c0 {bindings = {BM = "8"}}
  }
  return
}

func.func @candidate_empty_bindings() {
  micro.search_space @bad attributes {workload = "fused_swiglu"} {
    micro.param "BM" {kind = "integer", choices = [8 : i64]}
    // expected-error @+1 {{candidate must bind at least one parameter}}
    micro.candidate @c0 {bindings = {}}
  }
  return
}

func.func @duplicate_candidate_name() {
  micro.search_space @bad attributes {workload = "fused_swiglu"} {
    micro.param "BM" {kind = "integer", choices = [8 : i64]}
    micro.candidate @c0 {bindings = {BM = 8 : i64}}
    // expected-error @+1 {{duplicate candidate name 'c0'}}
    micro.candidate @c0 {bindings = {BM = 8 : i64}}
  }
  return
}

//===----------------------------------------------------------------------===//
// Nesting
//===----------------------------------------------------------------------===//

// Concrete execution ops belong in micro.kernel, not in a search space.
func.func @concrete_op_in_search_space() {
  micro.search_space @bad attributes {workload = "fused_swiglu"} {
    micro.param "BM" {kind = "integer", choices = [8 : i64]}
    // expected-error @+1 {{only search ops are allowed in a micro.search_space}}
    %c0 = arith.constant 0 : index
  }
  return
}

// Search ops are metadata for the scheduler and never run on hardware.
func.func @search_op_in_kernel() {
  micro.kernel @bad attributes {workload = "fused_swiglu"} {
    // expected-error @+1 {{search ops are not allowed inside micro.kernel}}
    micro.param "BM" {kind = "integer", choices = [8 : i64]}
  }
  return
}
