// RUN: llk-opt %s --micro-map="target=probe machine=%p/../../../machines/x86-avx2-v2.yaml layouts=%p/../../../mapping/x86-avx2/layouts.llkmap rules=%p/../../../test/Conversion/MicroMapping/unmaterialized_rules.llkmap emitters=e1 mode=deterministic" 2>&1 | FileCheck --check-prefix=WARN %s
//
// The RUN line is documentation only; the real invocation is the
// MicroMappingUnmaterialized CTest entry. It merges stderr, because the pass's
// diagnostic goes to stderr while FileCheck reads stdout.
//
// The fixture pairs a vector with a tile store that the rules place in a
// different memory, so the plan's edge between them is a movement whose value
// is a `!micro.tile`. The binder cannot materialize that as a copy, so it
// reports the connection (design §18.2: reported, not silently dropped). The
// pass must surface that report; this test pins the warning.

module {
  micro.kernel @mapped {
    %ext = tensor.empty() : tensor<8x8xf32>
    %t, %tok = micro.async_copy %ext {src_memory = #micro.memory<dram>, dst_memory = #micro.memory<sram>} : tensor<8x8xf32> -> tensor<8x8xf32>, !micro.async_token
    %v = micro.tile_view %t {shape = array<i64: 8, 8>} : tensor<8x8xf32> -> !micro.tile<8x8xf32, memory = #micro.memory<sram>>
    %r = micro.vector "add" %v, %v : !micro.tile<8x8xf32, memory = #micro.memory<sram>>, !micro.tile<8x8xf32, memory = #micro.memory<sram>> -> !micro.tile<8x8xf32, memory = #micro.memory<sram>>
    micro.tile_store %r {dst_memory = #micro.memory<dram>} : !micro.tile<8x8xf32, memory = #micro.memory<sram>>
    micro.yield
  }
}

// The binder's report must reach the user as a warning, not be discarded with
// the BoundPlan. The value id is content-derived; the reason is the binder's.
// WARN: warning: mapping: connection not materialized: value {{[0-9]+}}: 'micro.tile_async_copy' needs a destination-memory tile type
//
// The mapped IR is still emitted -- the warning does not fail the pass. The
// diagnostic's own "see current operation" dump prints the *unbound* kernel,
// which has no micro.plan, so this line only matches the final output.
// WARN: micro.plan
