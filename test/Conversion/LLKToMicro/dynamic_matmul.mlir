// RUN: llk-opt --llk-to-micro="schedule-db=missing.json" %s | FileCheck %s

// Fully dynamic M/N/K matmul exported to a compile-once tile program.
//
// The schedule database is missing on purpose, so the export uses its explicit
// bounded baseline (BM = 8, BN = 32, BK = 32, tail_policy = "bounded"). A
// dynamic extent is never resolved into a local tile shape: local tiles keep a
// static, compile-time bounded capacity, while loop bounds, the copied
// rectangle and the reduction length are runtime values reified from the
// caller's tensors.
//
// The output is an explicit caller-owned destination argument. There is no
// whole-output SRAM allocation and no whole-input padding: every boundary
// access is a clipped `micro.tile_load` / `micro.tile_store`, and the
// reduction carries an exact `valid_k`.
//
// The initializer is read: it is a separate input port that is copied into the
// accumulators before the reduction, so a nonzero initializer is preserved.

func.func @dynamic_matmul_root(%a: tensor<?x?xbf16>, %b: tensor<?x?xbf16>,
                               %init: tensor<?x?xbf16>) -> tensor<?x?xbf16> {
  %y = llk.matmul ins(%a, %b : tensor<?x?xbf16>, tensor<?x?xbf16>)
      outs(%init : tensor<?x?xbf16>)
      {accumulator_type = f32, math_mode = #llk.math_mode<bounded_fast>}
      -> tensor<?x?xbf16>
  return %y : tensor<?x?xbf16>
}

// A mixed-static signature: K is pinned at 64 by both operands, M and N are
// dynamic. The static axis needs no guard; the dynamic relationships do.
func.func @mixed_static_matmul(%a: tensor<?x64xbf16>, %b: tensor<64x?xbf16>,
                               %init: tensor<?x?xbf16>) -> tensor<?x?xbf16> {
  %y = llk.matmul ins(%a, %b : tensor<?x64xbf16>, tensor<64x?xbf16>)
      outs(%init : tensor<?x?xbf16>)
      {accumulator_type = f32, math_mode = #llk.math_mode<bounded_fast>}
      -> tensor<?x?xbf16>
  return %y : tensor<?x?xbf16>
}

// One independently dynamic axis at a time. The other two extents are known,
// so only the unknown relationship is guarded.
func.func @m_only_dynamic(%a: tensor<?x64xbf16>, %b: tensor<64x32xbf16>,
                          %init: tensor<?x32xbf16>) -> tensor<?x32xbf16> {
  %y = llk.matmul ins(%a, %b : tensor<?x64xbf16>, tensor<64x32xbf16>)
      outs(%init : tensor<?x32xbf16>)
      {accumulator_type = f32, math_mode = #llk.math_mode<bounded_fast>}
      -> tensor<?x32xbf16>
  return %y : tensor<?x32xbf16>
}

func.func @n_only_dynamic(%a: tensor<16x64xbf16>, %b: tensor<64x?xbf16>,
                          %init: tensor<16x?xbf16>) -> tensor<16x?xbf16> {
  %y = llk.matmul ins(%a, %b : tensor<16x64xbf16>, tensor<64x?xbf16>)
      outs(%init : tensor<16x?xbf16>)
      {accumulator_type = f32, math_mode = #llk.math_mode<bounded_fast>}
      -> tensor<16x?xbf16>
  return %y : tensor<16x?xbf16>
}

func.func @k_only_dynamic(%a: tensor<16x?xbf16>, %b: tensor<?x32xbf16>,
                          %init: tensor<16x32xbf16>) -> tensor<16x32xbf16> {
  %y = llk.matmul ins(%a, %b : tensor<16x?xbf16>, tensor<?x32xbf16>)
      outs(%init : tensor<16x32xbf16>)
      {accumulator_type = f32, math_mode = #llk.math_mode<bounded_fast>}
      -> tensor<16x32xbf16>
  return %y : tensor<16x32xbf16>
}

// The export is non-destructive: every source function and its LLK operation
// is still there, and the kernels are added beside them.
// CHECK-LABEL: func.func @dynamic_matmul_root
// CHECK: llk.matmul
// CHECK-LABEL: func.func @mixed_static_matmul
// CHECK: llk.matmul

//===----------------------------------------------------------------------===//
// Fully dynamic M/N/K
//===----------------------------------------------------------------------===//

// The signature names both the read initializer and the caller-owned output
// destination. `argument_bindings` says which argument is which port; the
// output destination is bound to the output port and yielded as the result.
// CHECK-LABEL: micro.kernel @matmul_dynamic(%{{.*}}: tensor<?x?xbf16>, %{{.*}}: tensor<?x?xbf16>, %{{.*}}: tensor<?x?xbf16>, %{{.*}}: tensor<?x?xbf16>) -> tensor<?x?xbf16> attributes {
// CHECK-SAME: argument_bindings = [{port = 0 : i64, role = "input"}, {port = 1 : i64, role = "input"}, {port = 2 : i64, role = "input"}, {port = 3 : i64, role = "output"}]
// CHECK-SAME: fragment_shape = array<i64: 8, 16, 32>
// CHECK-SAME: result_bindings = [3]
// CHECK-SAME: tail_policy = "bounded"

// The dynamic guard prologue reifies every extent and checks the relationships
// the compiler cannot prove. All three axes are dynamic, so all three are
// asserted -- K first, then N, then the destination's M.
// CHECK: %{{.*}} = tensor.dim
// CHECK: %{{.*}} = arith.cmpi eq
// CHECK: cf.assert %{{.*}}, "contraction dimension mismatch: K"
// CHECK: cf.assert %{{.*}}, "contraction dimension mismatch: N"
// CHECK: cf.assert %{{.*}}, "contraction dimension mismatch: M"

// The destination is a logical view of the caller's tensor, not an
// allocation: its shape stays dynamic and its memory is the caller's.
// CHECK: %[[DST:.*]] = micro.tile_view %{{.*}}[%{{.*}}, %{{.*}}] {shape = array<i64: -9223372036854775808, -9223372036854775808>} : tensor<?x?xbf16> -> !micro.tile<?x?xbf16, memory = #micro.memory<dram>>

// The trip count is computed from the runtime extent as quotient plus a
// nonzero-remainder bit -- never `extent + B - 1`, which could overflow. The
// loop iterates over tile numbers and the offset is `tile * B`.
// CHECK: %[[QUOT:.*]] = arith.divui %{{.*}}, %{{.*}} {micro.shape_only} : index
// CHECK: %[[REM:.*]] = arith.remui %{{.*}}, %{{.*}} {micro.shape_only} : index
// CHECK: %[[NZ:.*]] = arith.cmpi ne, %[[REM]], %{{.*}} {micro.shape_only} : index
// CHECK: %[[EXTRA:.*]] = arith.select %[[NZ]], %{{.*}}, %{{.*}} {micro.shape_only} : index
// CHECK: %[[TILES:.*]] = arith.addi %[[QUOT]], %[[EXTRA]] {micro.shape_only} : index

// One spatial loop per output axis, with the runtime tile count as its upper
// bound. The destination is threaded through both loops, so the clipped write
// of one tile is carried to the next. Each iteration's offset is `tile * B`.
// CHECK: micro.spatial_for %{{.*}} = %{{.*}} to %[[TILES]] step %{{.*}} map = #micro.map<worker> iter_args(%{{.*}} = %[[DST]]) -> (!micro.tile<?x?xbf16, memory = #micro.memory<dram>>) {
// CHECK: %[[BMOFFS:.*]] = arith.muli %{{.*}}, %{{.*}} {micro.shape_only} : index
// CHECK: micro.spatial_for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} map = #micro.map<lane> iter_args(%{{.*}} = %{{.*}}) -> (!micro.tile<?x?xbf16, memory = #micro.memory<dram>>) {
// CHECK: %[[BNOFFS:.*]] = arith.muli %{{.*}}, %{{.*}} {micro.shape_only} : index

// The read initializer is copied into the accumulator through a bounded load:
// only the valid rectangle is read, the rest of the fixed 8x32 tile is the
// declared zero fill.
// CHECK: %[[INIT_TILE:.*]], %{{.*}} = micro.tile_load %{{.*}}[%{{.*}}, %{{.*}}] valid[%{{.*}}, %{{.*}}] {dst_memory = #micro.memory<acc>, owner = #micro.owner<worker>, padding = 0.000000e+00 : bf16} : tensor<?x?xbf16> -> !micro.tile<8x32xbf16, memory = #micro.memory<acc>, owner = #micro.owner<worker>>, !micro.async_token

// The reduction loop has a runtime trip count and a runtime reduction length.
// CHECK: micro.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%{{.*}} = %{{.*}}) -> (!micro.tile<8x32xf32, memory = #micro.memory<acc>, owner = #micro.owner<worker>>) {
// CHECK: micro.pipeline stages = 1
// CHECK: %[[VK:.*]] = arith.minsi %{{.*}}, %{{.*}} : index
// CHECK: %[[LT:.*]], %{{.*}} = micro.tile_load
// CHECK: %[[RT:.*]], %{{.*}} = micro.tile_load
// CHECK: micro.wait
// CHECK: micro.mma %[[LT]], %[[RT]], %{{.*}} valid_k %[[VK]] {accumulator = #micro.dtype<f32>, input = #micro.dtype<bf16>, shape = array<i64: 8, 32, 32>}

// The write-back clips to the valid rectangle of the output tile, and the
// updated destination is what the kernel yields.
// CHECK: %[[WRITTEN:.*]] = micro.tile_store %{{.*}} into %{{.*}}[%{{.*}}, %{{.*}}] valid[%{{.*}}, %{{.*}}] {dst_memory = #micro.memory<dram>} : !micro.tile<8x32xbf16, memory = #micro.memory<acc>, owner = #micro.owner<worker>>, !micro.tile<?x?xbf16, memory = #micro.memory<dram>> -> !micro.tile<?x?xbf16, memory = #micro.memory<dram>>
// CHECK: micro.yield %[[WRITTEN]]

// No whole-output SRAM storage and no whole-input padding.
// CHECK-NOT: micro.tile_alloc : !micro.tile<?x?
// CHECK-NOT: tensor.pad
// CHECK-NOT: llk.

//===----------------------------------------------------------------------===//
// Mixed-static signature: K known, M and N dynamic
//===----------------------------------------------------------------------===//

// The signature keeps the static K on both operands. K needs no guard; M and
// N are still dynamic and still guarded.
// CHECK-LABEL: micro.kernel @matmul_dynamic{{.*}}(%{{.*}}: tensor<?x64xbf16>, %{{.*}}: tensor<64x?xbf16>, %{{.*}}: tensor<?x?xbf16>, %{{.*}}: tensor<?x?xbf16>) -> tensor<?x?xbf16>
// CHECK-SAME: tail_policy = "bounded"
// CHECK: cf.assert %{{.*}}, "contraction dimension mismatch: N"
// CHECK: cf.assert %{{.*}}, "contraction dimension mismatch: M"
// CHECK-NOT: cf.assert %{{.*}}, "contraction dimension mismatch: K"
// CHECK: micro.tile_store {{.*}} valid[%{{.*}}, %{{.*}}]

//===----------------------------------------------------------------------===//
// Independently dynamic axes
//===----------------------------------------------------------------------===//

// Only M is unknown: exactly one relationship needs a runtime guard.
// CHECK-LABEL: micro.kernel @matmul_dynamic{{.*}}(%{{.*}}: tensor<?x64xbf16>, %{{.*}}: tensor<64x32xbf16>, %{{.*}}: tensor<?x32xbf16>, %{{.*}}: tensor<?x32xbf16>) -> tensor<?x32xbf16>
// CHECK: cf.assert %{{.*}}, "contraction dimension mismatch: M"
// CHECK-NOT: cf.assert %{{.*}}, "contraction dimension mismatch: N"
// CHECK-NOT: cf.assert %{{.*}}, "contraction dimension mismatch: K"

// Only N is unknown.
// CHECK-LABEL: micro.kernel @matmul_dynamic{{.*}}(%{{.*}}: tensor<16x64xbf16>, %{{.*}}: tensor<64x?xbf16>, %{{.*}}: tensor<16x?xbf16>, %{{.*}}: tensor<16x?xbf16>) -> tensor<16x?xbf16>
// CHECK: cf.assert %{{.*}}, "contraction dimension mismatch: N"
// CHECK-NOT: cf.assert %{{.*}}, "contraction dimension mismatch: M"
// CHECK-NOT: cf.assert %{{.*}}, "contraction dimension mismatch: K"

// Only K is unknown. The reduction length is the runtime value, while the
// output extent stays a compile-time constant.
// CHECK-LABEL: micro.kernel @matmul_dynamic{{.*}}(%{{.*}}: tensor<16x?xbf16>, %{{.*}}: tensor<?x32xbf16>, %{{.*}}: tensor<16x32xbf16>, %{{.*}}: tensor<16x32xbf16>) -> tensor<16x32xbf16>
// CHECK: cf.assert %{{.*}}, "contraction dimension mismatch: K"
// CHECK-NOT: cf.assert %{{.*}}, "contraction dimension mismatch: N"
// CHECK-NOT: cf.assert %{{.*}}, "contraction dimension mismatch: M"
// CHECK: micro.mma {{.*}} valid_k
