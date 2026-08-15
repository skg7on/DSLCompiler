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

//===----------------------------------------------------------------------===//
// Concrete execution ops
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @test_loops
func.func @test_loops(%M : index, %K : index) {
  %c0 = arith.constant 0 : index
  %c64 = arith.constant 64 : index
  // CHECK: micro.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} {
  micro.for %i = %c0 to %M step %c64 {
    // CHECK: micro.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} {
    micro.for %j = %c0 to %K step %c64 {
    }
  }
  // CHECK: micro.spatial_for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} map = #micro.map<worker> {
  micro.spatial_for %bm = %c0 to %M step %c64 map = #micro.map<worker> {
  }
  return
}

// CHECK-LABEL: func.func @test_pipeline
func.func @test_pipeline() {
  // CHECK: micro.pipeline stages = 2 {
  micro.pipeline stages = 2 {
  }
  return
}

// CHECK-LABEL: func.func @test_movement
func.func @test_movement(%A : tensor<32x64xbf16>) {
  // CHECK: micro.alloc {memory = #micro.memory<acc>} : tensor<32x64xf32>
  %acc = micro.alloc {memory = #micro.memory<acc>} : tensor<32x64xf32>
  // CHECK: %[[D:.*]], %[[T:.*]] = micro.async_copy %{{.*}} {dst_memory = #micro.memory<sram>, src_memory = #micro.memory<dram>} : tensor<32x64xbf16> -> tensor<32x64xbf16>, !micro.async_token
  %dst, %tok = micro.async_copy %A {src_memory = #micro.memory<dram>, dst_memory = #micro.memory<sram>} : tensor<32x64xbf16> -> tensor<32x64xbf16>, !micro.async_token
  // CHECK: micro.wait %[[T]]
  micro.wait %tok
  // CHECK: micro.store %[[D]] {dst_memory = #micro.memory<dram>, src_memory = #micro.memory<sram>} : tensor<32x64xbf16>
  micro.store %dst {src_memory = #micro.memory<sram>, dst_memory = #micro.memory<dram>} : tensor<32x64xbf16>
  return
}

// A concrete tile-centric kernel nesting spatial_for -> pipeline -> for.
// CHECK-LABEL: func.func @test_nested_kernel
func.func @test_nested_kernel() {
  micro.kernel @nested attributes {target = "x86-avx2-cpu"} {
    %c0 = arith.constant 0 : index
    %c64 = arith.constant 64 : index
    %M = arith.constant 1024 : index
    %K = arith.constant 2048 : index
    // CHECK: micro.spatial_for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} map = #micro.map<worker> {
    micro.spatial_for %bm = %c0 to %M step %c64 map = #micro.map<worker> {
      // CHECK: micro.pipeline stages = 2 {
      micro.pipeline stages = 2 {
        // CHECK: micro.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} {
        micro.for %bk = %c0 to %K step %c64 {
        }
      }
    }
    micro.yield
  }
  return
}
