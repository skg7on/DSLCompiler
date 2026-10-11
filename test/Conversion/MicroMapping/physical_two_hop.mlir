// RUN: llk-opt %s --micro-map="target=issue129-two-hop machine=%p/../../Mapping/Inputs/issue129/two_hop_machine.yaml layouts=%p/../../../mapping/x86-avx2/layouts.llkmap rules=%p/../../Mapping/Inputs/issue129/two_hop_rules.llkmap emitters=e1 mode=deterministic require-executable=1" 2>&1 | FileCheck --check-prefix=MAP %s
//
// The RUN line above is documentation only: this project has no lit runner, and
// `//` lines are comments. The real invocation is the PhysicalTwoHop CTest
// entry in CMakeLists.txt, which supplies the same option string with absolute
// paths and merges stderr.
//
// Issue #129 finding 6, second half: a movement whose route crosses an
// intermediate memory is materialized as one awaited copy per hop, and *every*
// hop's destination is a storage decision the plan made. Before task R4 the
// selected plan reserved only the route's final memory, so the SRAM -> L2 ->
// DRAM movement this fixture forces had an L2 copy in the IR and no L2
// allocation to run it in.
//
// The machine's L2 holds exactly this value's 256 bytes, so the intermediate's
// reservation is load-bearing: a plan that failed to reserve it could not run,
// and a plan that double-counted it (one buffer per hop rather than one shared
// intermediate) would not fit either.

module {
  micro.kernel @physical_two_hop {
    %a = micro.tile_alloc : !micro.tile<8x8xf32, memory = #micro.memory<sram>>
    %r = micro.vector "add" %a, %a : !micro.tile<8x8xf32, memory = #micro.memory<sram>>, !micro.tile<8x8xf32, memory = #micro.memory<sram>> -> !micro.tile<8x8xf32, memory = #micro.memory<sram>>
    micro.tile_store %r {dst_memory = #micro.memory<l2>} : !micro.tile<8x8xf32, memory = #micro.memory<sram>>
    micro.yield
  }
}

// The kernel carries the selected plan, the routes, and -- the check that would
// fail before task R4 -- a reservation in the *intermediate* memory. Reserving
// only the route's destination left `l2.0` with no allocation at all.
// MAP: micro.kernel
// MAP-SAME: micro.plan
// MAP-SAME: memory = "l2.0"
// MAP-SAME: hops = [
//
// One awaited copy per hop, in route order: the value stages into L2 and then
// lands in DRAM. Each hop names the storage decision it materializes, so the
// emitted IR states which planned buffer the copy reads and writes instead of
// leaving a reader to infer one from the memory kind.
// MAP: micro.tile_async_copy
// MAP-SAME: dst_memory = #micro.memory<l2>
// MAP-SAME: micro.dst_storage
// MAP-SAME: micro.hop = 1
// MAP-SAME: micro.src_storage
// MAP: micro.tile_async_copy
// MAP-SAME: dst_memory = #micro.memory<dram>
// MAP-SAME: micro.dst_storage
// MAP-SAME: micro.hop = 2
// MAP-SAME: micro.src_storage
//
// The executable contract ran strictly, so nothing was left unmaterialized.
// MAP-NOT: warning
