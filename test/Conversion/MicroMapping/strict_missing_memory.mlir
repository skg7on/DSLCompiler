// RUN: llk-opt %s --micro-map="target=x86-avx2 machine=%p/../../../machines/x86-avx2-v2.yaml layouts=%p/../../../mapping/x86-avx2/layouts.llkmap rules=%p/../../../mapping/x86-avx2/rules.llkmap emitters=avx2_vector_add,avx2_vector_convert,avx2_vector_silu,avx2_vector_mul,avx2_mma,avx2_reduce,avx2_copy,avx2_tile_copy,avx2_tile_store,avx2_fused_convert_silu_mul mode=deterministic require-executable=1" 2>&1 | FileCheck --check-prefix=MAP %s
//
// The RUN line is documentation only; the real invocation is the
// MicroMappingStrictMissingMemory CTest entry in CMakeLists.txt. It merges
// stderr, because the pass's diagnostic goes there while FileCheck reads
// stdout.
//
// Issue #129, task R3. `mapping/x86-avx2/rules.llkmap`'s `avx2.vector_add` is a
// rule that declares no memory requirement: the tile its operands and results
// carry is what states where the value lives. Here those tiles ask for `rf`
// (register file), a Micro memory space the x86 profile models no node for, so
// the occurrence has no reachable physical memory.
//
// Before the fix the search skipped storage finalization entirely whenever a
// placement bound no memory -- which was every placement of these rules -- so
// this run succeeded under `require-executable=1` and its report carried a
// "storage planning was skipped" note. Now the endpoint fact must resolve: the
// run fails with the memory diagnostic that names the kind and the executor,
// and never binds a plan whose values have no physical home. This is the
// negative counterpart of `micro_map.mlir`, whose SRAM tiles do resolve.

module {
  micro.kernel @strict_missing_memory {
    %a = micro.tile_alloc : !micro.tile<8x8xf32, memory = #micro.memory<rf>>
    %r = micro.vector "add" %a, %a : !micro.tile<8x8xf32, memory = #micro.memory<rf>>, !micro.tile<8x8xf32, memory = #micro.memory<rf>> -> !micro.tile<8x8xf32, memory = #micro.memory<rf>>
    micro.yield
  }
}

// The strict run is refused, and the reason names the memory fact that is
// missing rather than a bare "no plan".
// MAP: error: bindPlan: the plan is not physically complete
// MAP: value 0 written by node 0 output 0
// MAP-SAME: needs a 'rf' memory the executor can address
// MAP-SAME: worker.0
//
// The search never claims a plan whose storage planning was skipped.
// MAP-NOT: storage planning was skipped
