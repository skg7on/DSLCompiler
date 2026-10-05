// RUN: llk-opt %s --micro-map="target=barrier machine=%p/barrier_machine.yaml layouts=%p/../../../mapping/x86-avx2/layouts.llkmap rules=%p/unmaterialized_rules.llkmap emitters=e1 mode=deterministic require-executable=1" 2>&1 | FileCheck --check-prefix=MAP %s
//
// The RUN line is documentation only; the real invocation is the
// MicroMappingBarrier CTest entry in CMakeLists.txt, which supplies the same
// option string with absolute paths and merges stderr.
//
// A movement that needs more than one transfer engine is barrier-synchronized:
// `finalizeStoragePlan` sets `requiresBarrier` on the synchronization step and
// the canonical materializer emits a `micro.barrier` over the movement's
// tokens. This fixture forces that path -- the vector's SRAM tile is stored
// into DRAM, and this machine can only reach DRAM from SRAM through L2, so the
// connection routes across two links and records two engines.
//
// It also pins the storage plan's been *run* on the selected plan: before the
// fix the plan carried no allocations or synchronization, no barrier was ever
// emitted, and this CHECK could not fire.

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

// The routed movement lands in the store's memory (DRAM), one awaited copy per
// hop: SRAM -> L2, then L2 -> DRAM.
// MAP: micro.tile_async_copy
// MAP-SAME: dst_memory = #micro.memory<l2>
// MAP: micro.tile_async_copy
// MAP-SAME: dst_memory = #micro.memory<dram>
//
// The multi-engine movement is barrier-synchronized, so a `micro.barrier` over
// its tokens is emitted and stamped with the connection it represents, ahead of
// the movement's wait.
// MAP: micro.barrier
// MAP-SAME: micro.connection
// MAP: micro.wait
//
// The plan is fully executable, so no connection is left unmaterialized.
// MAP-NOT: warning
