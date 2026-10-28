// RUN: llk-opt %s --micro-map="target=x86-avx2 machine=%p/../../../machines/x86-avx2-v2.yaml layouts=%p/../../../mapping/x86-avx2/layouts.llkmap rules=%p/../../../mapping/x86-avx2/rules.llkmap emitters=avx2_vector_add,avx2_vector_convert,avx2_vector_silu,avx2_vector_mul,avx2_mma,avx2_reduce,avx2_copy,avx2_tile_copy,avx2_tile_store,avx2_fused_convert_silu_mul mode=deterministic require-executable=1" 2>&1 | FileCheck --check-prefix=MAP %s
//
// The RUN line is documentation only; the real invocation is the
// MicroMappingDynamicLoopUnranked CTest entry in CMakeLists.txt, which supplies
// the same option string with absolute paths and merges stderr.
//
// Dynamic Shape Task 3 (issue #194). A loop whose bound is a runtime value no
// longer makes the plan *illegal*: its tiles keep a static capacity and a
// temporal loop reuses one buffer, so storage certification succeeds for the
// whole declared domain. What the unknown work count forbids is a *rankable*
// latency -- the plan's cost cannot be proven without concrete shape samples --
// so an executable run still refuses to claim a complete, scored plan. This is
// the storage-verdict/unranked-latency split the task requires, and it is why
// the refusal is now an analysis-incomplete rejection rather than a storage
// one: the earlier message conflated the two facts.

module {
  micro.kernel @strict_incomplete_loop {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %n = arith.addi %c4, %c1 : index
    %a = tensor.empty() : tensor<8x8xf32>
    %ta = micro.tile_view %a {shape = array<i64: 8, 8>} : tensor<8x8xf32> -> !micro.tile<8x8xf32, memory = #micro.memory<sram>>
    micro.for %i = %c0 to %n step %c1 {
      %r = micro.vector "add" %ta, %ta : !micro.tile<8x8xf32, memory = #micro.memory<sram>>, !micro.tile<8x8xf32, memory = #micro.memory<sram>> -> !micro.tile<8x8xf32, memory = #micro.memory<sram>>
    }
    micro.yield
  }
}

// Storage certified the plan; the selected-kernel analysis refuses to rank it
// because a loop's trip count is not a compile-time fact, and no shape profile
// supplied one. No plan is claimed and no partial artifact is materialized.
// MAP: error: analysis_incomplete: selected analysis refused the plan:
// MAP-SAME: the event stream is incomplete
// MAP: a loop has non-static bounds
// MAP: micro-map: the search produced no complete plan
// MAP-NOT: materialized = false
