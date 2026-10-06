// RUN: llk-opt --llk-to-micro="schedule-db=missing.json" %s | FileCheck %s --check-prefix=CHECK
// RUN: llk-opt --llk-to-micro="schedule-db=missing.json" %s \
// RUN:   | micro-perf --machine=%S/../../../machines/x86-avx2-v2.yaml --level=1 - \
// RUN:   | FileCheck %s --check-prefix=PERF

// Fused SwiGLU exported to concrete tile-centric Micro-IR.
//
// The schedule database is pointed at a path that does not exist on purpose:
// the pass then uses its built-in conservative schedule (BM=8, BN=32, BK=32,
// one pipeline stage, row-major, worker-owned), which makes the tile sizes
// below independent of the working directory and of the shipped database.
//
// The kernel is a two-level spatial nest over the output tile, a K loop whose
// body is a pipeline holding the staged copies and the MMAs, and the
// elementwise epilogue that writes the result back.
//
// Note the MMA shape: 8x32x32 is the whole materialized tile, because MmaOp's
// verifier requires its operands to be exactly [M,K], [K,N], [M,N] and the
// cost model charges `shape` as the work performed. The instruction fragment
// is spelled separately, as a micro.tile_partition annotation and as the
// kernel's fragment_shape attribute.

func.func @swiglu(%x: tensor<16x64xbf16>, %wg: tensor<64x64xbf16>,
                  %wu: tensor<64x64xbf16>, %init: tensor<16x64xbf16>)
    -> tensor<16x64xbf16> {
  %y = llk.fused_swiglu ins(%x, %wg, %wu : tensor<16x64xbf16>,
                            tensor<64x64xbf16>, tensor<64x64xbf16>)
      outs(%init : tensor<16x64xbf16>)
      {accumulator_type = f32, activation = #llk.activation<silu>,
       math_mode = #llk.math_mode<bounded_fast>}
      -> tensor<16x64xbf16>
  return %y : tensor<16x64xbf16>
}

// The export is non-destructive: the function and its LLK operation are still
// there, and the kernel is added beside them.
// CHECK-LABEL: func.func @swiglu
// CHECK: llk.fused_swiglu

// CHECK: micro.kernel @fused_swiglu_M16_N64_K64(%[[X:.*]]: tensor<16x64xbf16>, %[[WG:.*]]: tensor<64x64xbf16>, %[[WU:.*]]: tensor<64x64xbf16>) attributes {
// CHECK-SAME: fragment_shape = array<i64: 8, 16, 32>
// CHECK-SAME: memory_path = "dram:sram:acc"
// CHECK-SAME: mma_shape = array<i64: 16, 16, 32>
// CHECK-SAME: owner_mapping = "worker/lane"
// CHECK-SAME: schedule_id = 2 : i64
// CHECK-SAME: tail_policy = "none"
// CHECK-SAME: target = "x86-avx2-cpu"
// CHECK-SAME: tile_layout = "row_major"
// CHECK-SAME: workload = "fused_swiglu"

// The kernel is isolated from above, so it reads the three operands its own
// signature declared rather than materializing entry tensors.
// CHECK-NOT: tensor.empty

// One spatial loop per tiled output axis, mapped onto machine resources.
// CHECK: micro.spatial_for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} map = #micro.map<worker> {
// CHECK: micro.spatial_for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} map = #micro.map<lane> {

// Two accumulators, one per projection.
// CHECK: %[[GATE_ACC:.*]] = micro.tile_alloc : !micro.tile<8x32xf32, memory = #micro.memory<acc>, owner = #micro.owner<worker>>
// CHECK: %[[UP_ACC:.*]] = micro.tile_alloc : !micro.tile<8x32xf32, memory = #micro.memory<acc>, owner = #micro.owner<worker>>

// The K loop carries the pipeline, which is the position the performance model
// reads as software pipelining.
// CHECK: micro.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} {
// CHECK: micro.pipeline stages = 1 {

// Each input is viewed logically, then materialized into the staging level.
// CHECK: %[[XV:.*]] = micro.tile_view %[[X]]{{.*}} {layout = #micro.layout<row_major, vector = 8>, shape = array<i64: 8, 32>} : tensor<16x64xbf16> -> !micro.tile<8x32xbf16, layout = #micro.layout<row_major, vector = 8>, memory = #micro.memory<dram>>
// CHECK: %[[XT:.*]], %[[XTOK:.*]] = micro.tile_async_copy %[[XV]] {dst_memory = #micro.memory<sram>, owner = #micro.owner<worker>} : {{.*}} -> {{.*}}, !micro.async_token
// CHECK: %[[WGV:.*]] = micro.tile_view %[[WG]]{{.*}} {layout = #micro.layout<row_major, vector = 8>, shape = array<i64: 32, 32>} : tensor<64x64xbf16> -> !micro.tile<32x32xbf16, layout = #micro.layout<row_major, vector = 8>, memory = #micro.memory<dram>>
// CHECK: %[[WGT:.*]], %[[WGTOK:.*]] = micro.tile_async_copy %[[WGV]] {dst_memory = #micro.memory<sram>, owner = #micro.owner<worker>} : {{.*}} -> {{.*}}, !micro.async_token
// CHECK: %[[WUV:.*]] = micro.tile_view %[[WU]]{{.*}} {layout = #micro.layout<row_major, vector = 8>, shape = array<i64: 32, 32>} : tensor<64x64xbf16> -> !micro.tile<32x32xbf16, layout = #micro.layout<row_major, vector = 8>, memory = #micro.memory<dram>>
// CHECK: %[[WUT:.*]], %[[WUTOK:.*]] = micro.tile_async_copy %[[WUV]] {dst_memory = #micro.memory<sram>, owner = #micro.owner<worker>} : {{.*}} -> {{.*}}, !micro.async_token

// ...and synchronized before compute.
// CHECK: micro.wait %[[XTOK]], %[[WGTOK]], %[[WUTOK]]

// Instruction fragments. The left-hand fragment is clamped to the tile itself
// (8x32), so no partition is emitted for it; the weights are cut from 32x32
// into 32x16 fragments, which is the 16-wide N fragment from the schedule.
// CHECK: micro.tile_partition %[[WGT]] {owner = #micro.owner<vector_engine>, shape = array<i64: 32, 16>} : {{.*}} -> !micro.tile<32x16xbf16, {{.*}}owner = #micro.owner<vector_engine>>
// CHECK: micro.tile_partition %[[WUT]] {owner = #micro.owner<vector_engine>, shape = array<i64: 32, 16>} : {{.*}} -> !micro.tile<32x16xbf16, {{.*}}owner = #micro.owner<vector_engine>>

// One MMA per projection, both feeding the accumulators in acc memory.
// CHECK: micro.mma %[[XT]], %[[WGT]], %[[GATE_ACC]] {accumulator = #micro.dtype<f32>, input = #micro.dtype<bf16>, shape = array<i64: 8, 32, 32>}
// CHECK: micro.mma %[[XT]], %[[WUT]], %[[UP_ACC]] {accumulator = #micro.dtype<f32>, input = #micro.dtype<bf16>, shape = array<i64: 8, 32, 32>}

// The epilogue runs once per output tile, after the K loop: SiLU on the gate
// accumulator, the gating multiply, then the narrowing to the output dtype and
// the write back to external memory.
// CHECK: %[[GATE:.*]] = micro.vector "silu" %[[GATE_ACC]] {math_mode = "bounded_fast"}
// CHECK: %[[MUL:.*]] = micro.vector "mul" %[[GATE]], %[[UP_ACC]]
// CHECK: %[[OUT:.*]] = micro.vector "convert" %[[MUL]] : {{.*}} -> !micro.tile<8x32xbf16, memory = #micro.memory<acc>, owner = #micro.owner<worker>>
// CHECK: micro.tile_store %[[OUT]] {dst_memory = #micro.memory<dram>}

// The kernel contains no high-level LLK semantics.
// CHECK-NOT: llk.

//===----------------------------------------------------------------------===//
// The same IR, evaluated by the performance model
//===----------------------------------------------------------------------===//

// This is the acceptance check that the exported metadata is sufficient: the
// simulator reads the kernel with no schedule knowledge of its own and still
// recovers the work, the movement, the layout, and the owner of every tile.
//
// 2 x 2 output tiles, 2 K iterations each: 16 MMAs of 8x32x32 (2 flops per
// MAC), 8 waits, 24 staged copies, 4 stores and 12 vector ops.
//
// The flop count is exact: 2 projections x M x N x K x 2 = 262144, which is
// what makes the full-tile MMA choice (rather than a sub-tile one) the right
// call -- the model charges the MMA's `shape`, so a fragment-sized MMA here
// would have under-reported the kernel by the fragment ratio.
//
// The report also lists capacity violations for `acc` and `sram`. Those are a
// property of machine model + mapping rather than of this lowering: MicroDAG
// multiplies live tile bytes by the owner count of each spatial loop, so
// mapping the M axis to `worker` (8) and the N axis to `lane` (8) claims 64
// concurrent copies of every tile, and 64 x 4 KiB of accumulator cannot fit in
// a 4 KiB register file. They are deliberately not asserted here.
// PERF: machine: x86-avx2
// PERF: kernel: fused_swiglu_M16_N64_K64
// PERF: level: 1
// PERF: flops: 262144
// PERF: operations:
// PERF: async_copy: 24
// PERF: mma: 16
// PERF: store: 4
// PERF: vector: 12
// PERF: wait: 8
// PERF: row_major: 40
// PERF: lane: 12
// PERF: worker: 52
// PERF: bottleneck: dma
