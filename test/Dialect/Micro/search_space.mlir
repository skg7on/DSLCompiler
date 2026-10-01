// RUN: llk-opt %s | llk-opt | FileCheck %s

// Tile-aware search-space Micro-IR (M9). Verifies that micro.search_space,
// micro.param, micro.constraint, micro.objective, and micro.candidate parse,
// print, and round-trip. Printed order is deterministic: parameter and
// candidate order is preserved, while attribute dictionaries and candidate
// bindings print in sorted key order.

//===----------------------------------------------------------------------===//
// micro.search_space + micro.param
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @test_search_space_params
func.func @test_search_space_params() {
  // CHECK: micro.search_space @basic attributes {workload = "fused_swiglu"} {
  micro.search_space @basic attributes {workload = "fused_swiglu"} {
    // CHECK: micro.param "BM" {choices = [8, 16, 32], kind = "integer"}
    micro.param "BM" {kind = "integer", choices = [8 : i64, 16 : i64, 32 : i64]}
    // CHECK: micro.param "tile_layout" {choices = ["blocked", "row_major"], kind = "layout"}
    micro.param "tile_layout" {kind = "layout", choices = ["blocked", "row_major"]}
  }
  return
}

//===----------------------------------------------------------------------===//
// micro.constraint, micro.objective, micro.candidate
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @test_search_space_full
func.func @test_search_space_full() {
  // CHECK: micro.search_space @full attributes {workload = "tile_matmul"} {
  micro.search_space @full attributes {workload = "tile_matmul"} {
    // CHECK: micro.param "BM" {choices = [32, 64, 128], kind = "integer"}
    micro.param "BM" {kind = "integer", choices = [32 : i64, 64 : i64, 128 : i64]}
    // CHECK: micro.param "BK" {choices = [16, 32], kind = "integer"}
    micro.param "BK" {kind = "integer", choices = [16 : i64, 32 : i64]}
    // CHECK: micro.param "tile_layout" {choices = ["row_major", "blocked"], kind = "layout"}
    micro.param "tile_layout" {kind = "layout", choices = ["row_major", "blocked"]}
    // CHECK: micro.constraint "sram_capacity" {params = ["BM", "BK"]}
    micro.constraint "sram_capacity" {params = ["BM", "BK"]}
    // CHECK: micro.objective {direction = "minimize", metric = "latency_cycles", secondary = ["matrix_utilization", "dram_bytes"]}
    micro.objective {direction = "minimize", metric = "latency_cycles", secondary = ["matrix_utilization", "dram_bytes"]}
    // CHECK: micro.candidate @candidate_17 {bindings = {BK = 32 : i64, BM = 64 : i64, tile_layout = "blocked"}}
    micro.candidate @candidate_17 {bindings = {BM = 64 : i64, BK = 32 : i64, tile_layout = "blocked"}}
    // CHECK: micro.candidate @candidate_18 {bindings = {BK = 16 : i64, BM = 128 : i64, tile_layout = "row_major"}}
    micro.candidate @candidate_18 {bindings = {BM = 128 : i64, BK = 16 : i64, tile_layout = "row_major"}}
  }
  return
}

//===----------------------------------------------------------------------===//
// Symbolic parameter domains
//===----------------------------------------------------------------------===//

// Choice order is declaration order, not sorted, so that a parameter's
// preferred value can be listed first.

// CHECK-LABEL: func.func @test_search_space_symbolic_domains
func.func @test_search_space_symbolic_domains() {
  // CHECK: micro.search_space @domains attributes {workload = "tile_matmul"} {
  micro.search_space @domains attributes {workload = "tile_matmul"} {
    // CHECK: micro.param "memory_path" {choices = ["dram:sram", "dram:l2:sram"], kind = "memory_path"}
    micro.param "memory_path" {kind = "memory_path", choices = ["dram:sram", "dram:l2:sram"]}
    // CHECK: micro.param "owner_mapping" {choices = ["worker/vector_engine", "worker/matrix_engine"], kind = "owner_mapping"}
    micro.param "owner_mapping" {kind = "owner_mapping", choices = ["worker/vector_engine", "worker/matrix_engine"]}
    // CHECK: micro.param "fragment_shape" {choices = ["16x16x32", "16x8x64"], kind = "fragment_shape"}
    micro.param "fragment_shape" {kind = "fragment_shape", choices = ["16x16x32", "16x8x64"]}
    // CHECK: micro.param "tail_policy" {choices = ["mask"], kind = "tail_policy"}
    micro.param "tail_policy" {kind = "tail_policy", choices = ["mask"]}
  }
  return
}
