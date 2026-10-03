// RUN: llk-opt --llk-to-micro="schedule-db=missing.json" %s | FileCheck %s

// The M11 export on a plain matmul: the same tile-GEMM core as SwiGLU with one
// weight operand, one accumulator, and no gating epilogue.
//
// As in the SwiGLU test the schedule database is pointed at a path that does
// not exist, so the built-in conservative schedule (BM=8, BN=32, BK=32) makes
// the tile sizes below independent of the working directory.

func.func @matmul(%a: tensor<16x64xbf16>, %b: tensor<64x64xbf16>,
                  %init: tensor<16x64xbf16>) -> tensor<16x64xbf16> {
  %y = llk.matmul ins(%a, %b : tensor<16x64xbf16>, tensor<64x64xbf16>)
      outs(%init : tensor<16x64xbf16>)
      {accumulator_type = f32, math_mode = #llk.math_mode<bounded_fast>}
      -> tensor<16x64xbf16>
  return %y : tensor<16x64xbf16>
}

// The source function is left alone.
// CHECK-LABEL: func.func @matmul
// CHECK: llk.matmul

// CHECK-LABEL: micro.kernel @matmul_M16_N64_K64 attributes {
// CHECK-SAME: fragment_shape = array<i64: 8, 16, 32>
// CHECK-SAME: memory_path = "dram:sram:acc"
// CHECK-SAME: mma_shape = array<i64: 16, 16, 32>
// CHECK-SAME: owner_mapping = "worker/lane"
// CHECK-SAME: schedule_id = 2 : i64
// CHECK-SAME: tail_policy = "none"
// CHECK-SAME: target = "x86-avx2-cpu"
// CHECK-SAME: tile_layout = "row_major"
// CHECK-SAME: workload = "matmul"

// Two entry tensors, not three: A and B.
// CHECK: %[[A:.*]] = tensor.empty() : tensor<16x64xbf16>
// CHECK: %[[B:.*]] = tensor.empty() : tensor<64x64xbf16>
// CHECK-NOT: tensor.empty

// CHECK: micro.spatial_for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} map = #micro.map<worker> {
// CHECK: micro.spatial_for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} map = #micro.map<lane> {

// One accumulator, in acc memory.
// CHECK: %[[ACC:.*]] = micro.tile_alloc : !micro.tile<8x32xf32, memory = #micro.memory<acc>, owner = #micro.owner<worker>>

// CHECK: micro.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} {
// CHECK: micro.pipeline stages = 1 {

// Both operands are viewed logically and staged through SRAM.
// CHECK: %[[AV:.*]] = micro.tile_view %[[A]]{{.*}} {layout = #micro.layout<row_major, vector = 8>, shape = array<i64: 8, 32>} : tensor<16x64xbf16> -> !micro.tile<8x32xbf16, layout = #micro.layout<row_major, vector = 8>, memory = #micro.memory<dram>>
// CHECK: %[[AT:.*]], %[[ATOK:.*]] = micro.tile_async_copy %[[AV]] {dst_memory = #micro.memory<sram>, owner = #micro.owner<worker>} : {{.*}} -> {{.*}}, !micro.async_token
// CHECK: %[[BV:.*]] = micro.tile_view %[[B]]{{.*}} {layout = #micro.layout<row_major, vector = 8>, shape = array<i64: 32, 32>} : tensor<64x64xbf16> -> !micro.tile<32x32xbf16, layout = #micro.layout<row_major, vector = 8>, memory = #micro.memory<dram>>
// CHECK: %[[BT:.*]], %[[BTOK:.*]] = micro.tile_async_copy %[[BV]] {dst_memory = #micro.memory<sram>, owner = #micro.owner<worker>} : {{.*}} -> {{.*}}, !micro.async_token
// CHECK: micro.wait %[[ATOK]], %[[BTOK]]

// The left-hand tile is already fragment-sized (8x32), so only B is partitioned.
// CHECK: micro.tile_partition %[[BT]] {owner = #micro.owner<vector_engine>, shape = array<i64: 32, 16>} : {{.*}} -> !micro.tile<32x16xbf16, {{.*}}owner = #micro.owner<vector_engine>>

// Exactly one MMA inside the K loop.
// CHECK: micro.mma %[[AT]], %[[BT]], %[[ACC]] {accumulator = #micro.dtype<f32>, input = #micro.dtype<bf16>, shape = array<i64: 8, 32, 32>}
// CHECK-NOT: micro.mma
// CHECK-NOT: micro.vector "silu"
// CHECK-NOT: micro.vector "mul"

// The epilogue is the narrowing conversion and the write back.
// CHECK: %[[OUT:.*]] = micro.vector "convert" %[[ACC]] : {{.*}} -> !micro.tile<8x32xbf16, memory = #micro.memory<acc>, owner = #micro.owner<worker>>
// CHECK: micro.tile_store %[[OUT]] {dst_memory = #micro.memory<dram>}

// CHECK-NOT: llk.
