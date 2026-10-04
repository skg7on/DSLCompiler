// RUN: llk-opt %s --micro-map="target=x86-avx2 machine=%p/../../../machines/x86-avx2-v2.yaml layouts=%p/../../../mapping/x86-avx2/layouts.llkmap rules=%p/../../../mapping/x86-avx2/rules.llkmap emitters=avx2_vector_add,avx2_vector_convert,avx2_vector_silu,avx2_vector_mul,avx2_mma,avx2_reduce,avx2_copy,avx2_tile_copy,avx2_tile_store mode=deterministic" --micro-verify-mapping="target=x86-avx2 machine=%p/../../../machines/x86-avx2-v2.yaml layouts=%p/../../../mapping/x86-avx2/layouts.llkmap rules=%p/../../../mapping/x86-avx2/rules.llkmap emitters=avx2_vector_add,avx2_vector_convert,avx2_vector_silu,avx2_vector_mul,avx2_mma,avx2_reduce,avx2_copy,avx2_tile_copy,avx2_tile_store" | FileCheck %s
//
// The RUN line above is documentation only: this project has no lit runner, and
// `//` lines are comments. The real invocation is the MicroMappingVerify CTest
// entry in CMakeLists.txt, which supplies the same option string with absolute
// paths.
//
// Phase 2 of the layered verification (design §18.3): `--micro-map` binds a plan
// (phase 1's structural metadata), then `--micro-verify-mapping` loads the same
// target and resolves every id the binder recorded -- rule, executor, memories,
// layouts, emitter, and route memories. The module below the pass must be
// exactly the module `--micro-map` produced: the verifier reports and changes
// nothing (design §21, "verifying already-mapped Micro-IR").

module {
  micro.kernel @mapped {
    %ext = tensor.empty() : tensor<8x8xf32>
    %t, %tok = micro.async_copy %ext {src_memory = #micro.memory<dram>, dst_memory = #micro.memory<sram>} : tensor<8x8xf32> -> tensor<8x8xf32>, !micro.async_token
    %v = micro.tile_view %t {shape = array<i64: 8, 8>} : tensor<8x8xf32> -> !micro.tile<8x8xf32, memory = #micro.memory<sram>>
    %r = micro.vector "add" %v, %v : !micro.tile<8x8xf32, memory = #micro.memory<sram>>, !micro.tile<8x8xf32, memory = #micro.memory<sram>> -> !micro.tile<8x8xf32, memory = #micro.memory<sram>>
    micro.yield
  }
}

// The plan and the per-operation placement survive the verifier untouched: a
// successful `--micro-verify-mapping` is a report, not a rewrite.
// CHECK: micro.kernel
// CHECK-SAME: micro.plan
// CHECK: micro.async_copy
// CHECK-SAME: micro.mapping
// CHECK-SAME: rule = "avx2.async_copy"
// CHECK: micro.vector
// CHECK-SAME: micro.mapping
// CHECK-SAME: rule = "avx2.vector_add"
