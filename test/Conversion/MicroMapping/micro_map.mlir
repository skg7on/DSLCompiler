// RUN: llk-opt %s --micro-map="target=x86-avx2 machine=%p/../../../machines/x86-avx2-v2.yaml layouts=%p/../../../mapping/x86-avx2/layouts.llkmap rules=%p/../../../mapping/x86-avx2/rules.llkmap emitters=avx2_vector_add,avx2_mma,avx2_reduce,avx2_copy mode=deterministic" | FileCheck %s
//
// The RUN line above is documentation only: this project has no lit runner, and
// `//` lines are comments. The real invocation is the MicroMappingMap CTest
// entry in CMakeLists.txt, which supplies the same option string with absolute
// paths.
//
// The kernel is the elementwise fixture from test/Mapping/e2e_workflow.cpp: a
// staged copy plus a vector add -- two pieces of work for the mapper to place.

module {
  micro.kernel @mapped {
    %ext = tensor.empty() : tensor<8x8xf32>
    %t, %tok = micro.async_copy %ext {src_memory = #micro.memory<dram>, dst_memory = #micro.memory<sram>} : tensor<8x8xf32> -> tensor<8x8xf32>, !micro.async_token
    %v = micro.tile_view %t {shape = array<i64: 8, 8>} : tensor<8x8xf32> -> !micro.tile<8x8xf32, memory = #micro.memory<sram>>
    %r = micro.vector "add" %v, %v : !micro.tile<8x8xf32, memory = #micro.memory<sram>>, !micro.tile<8x8xf32, memory = #micro.memory<sram>> -> !micro.tile<8x8xf32, memory = #micro.memory<sram>>
    micro.yield
  }
}

// After --micro-map the kernel carries the selected plan's metadata, and each
// covered operation carries its placement.
// CHECK: micro.kernel
// CHECK-SAME: micro.plan
// CHECK: micro.async_copy
// CHECK-SAME: micro.mapping
// CHECK: micro.vector
// CHECK-SAME: micro.mapping
