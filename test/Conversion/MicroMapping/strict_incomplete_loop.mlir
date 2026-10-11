// RUN: llk-opt %s --micro-map="target=x86-avx2 machine=%p/../../../machines/x86-avx2-v2.yaml layouts=%p/../../../mapping/x86-avx2/layouts.llkmap rules=%p/../../../mapping/x86-avx2/rules.llkmap emitters=avx2_vector_add,avx2_vector_convert,avx2_vector_silu,avx2_vector_mul,avx2_mma,avx2_reduce,avx2_copy,avx2_tile_copy,avx2_tile_store,avx2_fused_convert_silu_mul mode=deterministic require-executable=1" 2>&1 | FileCheck --check-prefix=MAP %s
//
// The RUN line is documentation only; the real invocation is the
// MicroMappingStrictIncompleteLoop CTest entry in CMakeLists.txt, which supplies
// the same option string with absolute paths and merges stderr.
//
// Issue #129 review finding 2: a loop whose bound is not a literal constant
// cannot have its execution multiplicity recovered, so a strict (executable)
// run must refuse it rather than fall back to a partial analysis artifact that
// charges a single iteration and offers the plan as executable. Before the fix
// this mapped successfully with `require-executable=1` and `micro.plan
// materialized=false`, scored one vector event instead of the loop's work.

module {
  micro.kernel @strict_incomplete_loop {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %n = arith.addi %c4, %c1 : index
    %a = tensor.empty() : tensor<8x8xf32>
    %ta = micro.tile_view %a {shape = array<i64: 8, 8>} : tensor<8x8xf32> -> !micro.tile<8x8xf32>
    micro.for %i = %c0 to %n step %c1 {
      %r = micro.vector "add" %ta, %ta : !micro.tile<8x8xf32>, !micro.tile<8x8xf32> -> !micro.tile<8x8xf32>
    }
    micro.yield
  }
}

// The strict run is refused, naming the fact that could not be recovered; no
// plan is claimed and no partial artifact is materialized.
// MAP: error: storage plan: node 0 ('micro.vector') has unknown execution multiplicity
// MAP-SAME: strict executable planning cannot assume a single iteration
// MAP: micro-map: the search produced no complete plan
// MAP-NOT: materialized = false
