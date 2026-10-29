// RUN: llk-opt --llk-to-micro="schedule-db=missing.json" %s | FileCheck %s

// Fully dynamic two-arm SwiGLU export.
//
// Both projections are tiled contractions over the same dynamic extents. Each
// weight arm is a separate input port, the output destination is an explicit
// caller-owned argument, and every boundary access is clipped: no whole-output
// allocation and no whole-input padding. The epilogue keeps the source's SiLU
// math mode.

func.func @swiglu_dynamic(%x: tensor<?x?xbf16>, %wg: tensor<?x?xbf16>,
                          %wu: tensor<?x?xbf16>, %init: tensor<?x?xbf16>)
    -> tensor<?x?xbf16> {
  %y = llk.fused_swiglu ins(%x, %wg, %wu : tensor<?x?xbf16>,
                            tensor<?x?xbf16>, tensor<?x?xbf16>)
      outs(%init : tensor<?x?xbf16>)
      {accumulator_type = f32, activation = #llk.activation<silu>,
       math_mode = #llk.math_mode<bounded_fast>}
      -> tensor<?x?xbf16>
  return %y : tensor<?x?xbf16>
}

// CHECK-LABEL: func.func @swiglu_dynamic
// CHECK: llk.fused_swiglu

// Both weights and the destination are signature arguments. The SwiGLU
// destination has no read effect, so it is bound to the output port rather
// than to an input port.
// CHECK: micro.kernel @fused_swiglu_dynamic(%{{.*}}: tensor<?x?xbf16>, %{{.*}}: tensor<?x?xbf16>, %{{.*}}: tensor<?x?xbf16>, %{{.*}}: tensor<?x?xbf16>) -> tensor<?x?xbf16> attributes {
// CHECK-SAME: argument_bindings = [{port = 0 : i64, role = "input"}, {port = 1 : i64, role = "input"}, {port = 2 : i64, role = "input"}, {port = 3 : i64, role = "output"}]
// CHECK-SAME: result_bindings = [3]
// CHECK-SAME: tail_policy = "bounded"
// CHECK-SAME: workload = "fused_swiglu"

// The guard prologue: K is shared by X, Wg and Wu; N is shared by Wg, Wu and
// the destination; M is shared by X and the destination.
// CHECK: cf.assert %{{.*}}, "contraction dimension mismatch: K"
// CHECK: cf.assert %{{.*}}, "contraction dimension mismatch: N"
// CHECK: cf.assert %{{.*}}, "contraction dimension mismatch: M"

// The output destination is a view of the caller's tensor with dynamic shape.
// CHECK: %[[DST:.*]] = micro.tile_view %{{.*}}[%{{.*}}, %{{.*}}] {shape = array<i64: -9223372036854775808, -9223372036854775808>} : tensor<?x?xbf16> -> !micro.tile<?x?xbf16, memory = #micro.memory<dram>>
// CHECK: micro.spatial_for {{.*}} map = #micro.map<worker> iter_args(%{{.*}} = %[[DST]])
// CHECK: micro.spatial_for {{.*}} map = #micro.map<lane>

// The accumulators start from the declared zero fill, so a zero-length K
// reduction still evaluates the epilogue from zeros. The fill copies no source
// lane: its valid rectangle is empty.
// CHECK: micro.tile_load {{.*}} valid[%{{.*}}, %{{.*}}] {dst_memory = #micro.memory<acc>,

// Each arm materializes its own staged weight tile and reduces the same
// dynamic X tile over an exact runtime K length.
// CHECK: micro.for {{.*}} iter_args
// CHECK: micro.pipeline stages = 1
// CHECK: %[[XT:.*]], %{{.*}} = micro.tile_load
// CHECK: %[[WGT:.*]], %{{.*}} = micro.tile_load
// CHECK: %[[WUT:.*]], %{{.*}} = micro.tile_load
// CHECK: micro.wait
// CHECK: %[[GATE:.*]] = micro.mma %[[XT]], %[[WGT]], {{.*}} valid_k %{{.*}} {accumulator = #micro.dtype<f32>, input = #micro.dtype<bf16>, shape = array<i64: 8, 32, 32>}
// CHECK: %[[UP:.*]] = micro.mma %[[XT]], %[[WUT]], {{.*}} valid_k %{{.*}} {accumulator = #micro.dtype<f32>, input = #micro.dtype<bf16>, shape = array<i64: 8, 32, 32>}

// The epilogue preserves the source math mode and narrows to the output dtype.
// CHECK: %[[SILU:.*]] = micro.vector "silu" {{.*}} {math_mode = "bounded_fast"}
// CHECK: %[[MUL:.*]] = micro.vector "mul" %[[SILU]]
// CHECK: micro.vector "convert"

// The clipped write-back updates the carried destination and yields it.
// CHECK: %[[WRITTEN:.*]] = micro.tile_store {{.*}} into {{.*}}[%{{.*}}, %{{.*}}] valid[%{{.*}}, %{{.*}}] {dst_memory = #micro.memory<dram>} : !micro.tile<8x32xbf16, memory = #micro.memory<acc>, owner = #micro.owner<worker>>, !micro.tile<?x?xbf16, memory = #micro.memory<dram>> -> !micro.tile<?x?xbf16, memory = #micro.memory<dram>>
// CHECK: micro.yield %[[WRITTEN]]

// No whole-output storage, no whole-input padding, and no high-level semantics.
// CHECK-NOT: micro.tile_alloc : !micro.tile<?x?
// CHECK-NOT: tensor.pad
// CHECK-NOT: llk.
