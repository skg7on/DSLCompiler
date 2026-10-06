// RUN: llk-opt %s --micro-map="target=probe machine=%p/../../../test/Conversion/MicroMapping/probe_machine.yaml layouts=%p/../../../mapping/x86-avx2/layouts.llkmap rules=%p/../../../test/Conversion/MicroMapping/unmaterialized_rules.llkmap emitters=e1 mode=deterministic require-executable=1" 2>&1 | FileCheck --check-prefix=MAP %s
//
// The RUN line is documentation only; the real invocation is the
// MicroMappingUnmaterialized CTest entry. It merges stderr, because the pass's
// diagnostic goes to stderr while FileCheck reads stdout.
//
// The fixture pairs a vector with a tile store that the rules place in a
// different memory, so the plan's edge between them is a real movement whose
// value is a `!micro.tile`. Its consumer sits in cluster.a (SRAM) and the store
// in cluster.b (DRAM), so under design §10.2 the store cannot read the vector's
// SRAM tile directly: the plan's edge is a `Transfer`.
//
// Canonical materialization (design §18.2) turns that edge into a typed
// `micro.tile_async_copy` of the moved tile into the store's memory, followed
// by its `micro.wait`, and rewires the store's operand to the destination-memory
// tile. The store's own `dst_memory` is L2 -- a different space from the
// destination -- so the rewired `tile_store` still satisfies the dialect
// verifier ("source and destination memory must differ"). `require-executable=1`
// asserts the plan is now *fully* executable: no connection is left
// unmaterialized and no warning is emitted.

module {
  micro.kernel @mapped {
    %ext = tensor.empty() : tensor<8x8xf32>
    %t, %tok = micro.async_copy %ext {src_memory = #micro.memory<dram>, dst_memory = #micro.memory<sram>} : tensor<8x8xf32> -> tensor<8x8xf32>, !micro.async_token
    %v = micro.tile_view %t {shape = array<i64: 8, 8>} : tensor<8x8xf32> -> !micro.tile<8x8xf32, memory = #micro.memory<sram>>
    %r = micro.vector "add" %v, %v : !micro.tile<8x8xf32, memory = #micro.memory<sram>>, !micro.tile<8x8xf32, memory = #micro.memory<sram>> -> !micro.tile<8x8xf32, memory = #micro.memory<sram>>
    micro.tile_store %r {dst_memory = #micro.memory<l2>} : !micro.tile<8x8xf32, memory = #micro.memory<sram>>
    micro.yield
  }
}

// The movement is a typed tile copy into the store's memory (DRAM), carrying
// the destination memory on both the op and its result tile.
// MAP: micro.tile_async_copy
// MAP-SAME: dst_memory = #micro.memory<dram>
// MAP-SAME: !micro.tile<8x8xf32, memory = #micro.memory<sram>> -> !micro.tile<8x8xf32, memory = #micro.memory<dram>>
// MAP: micro.wait
//
// The recorded consumer reads the movement: the store's operand is the
// destination-memory tile the copy produced, not the SRAM tile it read before.
// MAP: micro.tile_store
// MAP-SAME: : !micro.tile<8x8xf32, memory = #micro.memory<dram>>
//
// A fully materialized plan is not warned about.
// MAP-NOT: warning
