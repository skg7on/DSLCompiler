// RUN: llk-opt %s | llk-opt | FileCheck %s

// Test that the micro dialect is registered and that micro.kernel
// operations with attributes can be parsed and printed (round-trip).

// CHECK-LABEL: func.func @test_kernel_roundtrip
func.func @test_kernel_roundtrip() {
  // CHECK: micro.kernel
  // CHECK: candidate = "candidate_17"
  micro.kernel @my_kernel attributes {workload = "fused_swiglu", target = "x86-avx2-cpu", candidate = "candidate_17"} {
    micro.yield
  }
  return
}

// Check that the kernel with optional attributes round-trips.
// CHECK-LABEL: func.func @test_kernel_minimal
func.func @test_kernel_minimal() {
  // CHECK: micro.kernel @minimal
  micro.kernel @minimal {
    micro.yield
  }
  return
}

// Test that all enum attribute spellings parse and round-trip.
// Attributes in dicts are printed in alphabetical order:
// dtype_all, then map_all, then mem_all.
// CHECK-LABEL: func.func @test_all_attrs
// CHECK: dtype_all = [#micro.dtype<f32>, #micro.dtype<f16>, #micro.dtype<bf16>, #micro.dtype<i32>, #micro.dtype<i8>]
// CHECK: map_all = [#micro.map<cluster_x>, #micro.map<cluster_y>, #micro.map<core_x>, #micro.map<core_y>, #micro.map<pe_x>, #micro.map<pe_y>, #micro.map<lane>, #micro.map<worker>, #micro.map<matrix_engine>, #micro.map<vector_engine>, #micro.map<dma>]
// CHECK: mem_all = [#micro.memory<dram>, #micro.memory<l2>, #micro.memory<sram>, #micro.memory<rf>, #micro.memory<acc>, #micro.memory<scratch>]
func.func @test_all_attrs() attributes {
  mem_all = [
    #micro.memory<dram>,
    #micro.memory<l2>,
    #micro.memory<sram>,
    #micro.memory<rf>,
    #micro.memory<acc>,
    #micro.memory<scratch>
  ],
  map_all = [
    #micro.map<cluster_x>,
    #micro.map<cluster_y>,
    #micro.map<core_x>,
    #micro.map<core_y>,
    #micro.map<pe_x>,
    #micro.map<pe_y>,
    #micro.map<lane>,
    #micro.map<worker>,
    #micro.map<matrix_engine>,
    #micro.map<vector_engine>,
    #micro.map<dma>
  ],
  dtype_all = [
    #micro.dtype<f32>,
    #micro.dtype<f16>,
    #micro.dtype<bf16>,
    #micro.dtype<i32>,
    #micro.dtype<i8>
  ]
} {
  return
}

// Test async_token type in a function signature.
// CHECK-LABEL: func.func @test_async_token
func.func @test_async_token(%tok : !micro.async_token) -> !micro.async_token {
  // CHECK: return
  return %tok : !micro.async_token
}
