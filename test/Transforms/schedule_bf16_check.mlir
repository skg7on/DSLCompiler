// RUN: llk-opt %s | FileCheck %s
//
// Parse-only test for the AVX2 BF16 transform schedule.
// The --tile-and-vectorize pass (Task 2.4) applies this schedule via
//   llk-opt --tile-and-vectorize="schedule-file=schedules/x86_avx2/schedule_bf16.mlir"
// That end-to-end test is added once the pass is implemented.

// CHECK-LABEL: module attributes {transform.with_named_sequence}
// CHECK: transform.named_sequence @schedule_avx2_bf16
// CHECK: transform.structured.match ops{["llk.fused_swiglu"]}
// CHECK: transform.structured.tile_using_forall
// CHECK: transform.structured.fuse_into_containing_op
// CHECK: transform.structured.tile_reduction_using_for
// CHECK: transform.structured.vectorize
// CHECK: transform.yield

module attributes {transform.with_named_sequence} {
  transform.named_sequence @schedule_avx2_bf16(
      %root: !transform.any_op {transform.readonly}) {

    %swiglu = transform.structured.match ops{["llk.fused_swiglu"]} in %root
      : (!transform.any_op) -> !transform.any_op

    %tiled, %loops = transform.structured.tile_using_forall %swiglu
        tile_sizes [32, 64]
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

    %fused, %new_containing = transform.structured.fuse_into_containing_op
        %tiled into %loops
      : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)

    %fill, %k_tiled, %combine, %k_loops = transform.structured.tile_reduction_using_for
        %fused by tile_sizes = [64]
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)

    transform.structured.vectorize %k_tiled
        vector_sizes [4, 8]
      : !transform.any_op

    transform.yield
  }
}
