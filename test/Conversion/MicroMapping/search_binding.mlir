// RUN: llk-opt %s | llk-opt | FileCheck %s

// Fixture for the SearchBindingLoaderTest GTest (phase-4 task 1). It is the
// smallest space the loader must accept: two parameters of different kinds and
// two pre-baked candidates. The square brackets around the candidate names in
// the CHECK lines are literal, because @ is not a FileCheck metacharacter.

// CHECK-LABEL: func.func @test_search_binding
func.func @test_search_binding() {
  // CHECK: micro.search_space @spatial_conv2d attributes {workload = "spatial_conv2d"} {
  micro.search_space @spatial_conv2d attributes {workload = "spatial_conv2d"} {
    // CHECK: micro.param "BM" {choices = [32, 64, 128], kind = "integer"}
    micro.param "BM" {kind = "integer", choices = [32 : i64, 64 : i64, 128 : i64]}
    // CHECK: micro.param "tile_layout" {choices = ["row_major", "blocked"], kind = "layout"}
    micro.param "tile_layout" {kind = "layout", choices = ["row_major", "blocked"]}
    // CHECK: micro.candidate @candidate_17 {bindings = {BM = 64 : i64, tile_layout = "blocked"}}
    micro.candidate @candidate_17 {bindings = {BM = 64 : i64, tile_layout = "blocked"}}
    // CHECK: micro.candidate @candidate_18 {bindings = {BM = 128 : i64, tile_layout = "row_major"}}
    micro.candidate @candidate_18 {bindings = {BM = 128 : i64, tile_layout = "row_major"}}
  }
  return
}
