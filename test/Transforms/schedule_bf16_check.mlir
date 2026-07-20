// RUN: llk-opt %s | FileCheck %s
//
// Parse-only test for the AVX2 BF16 transform schedule.
// The --tile-and-vectorize pass applies this schedule via
//   llk-opt --tile-and-vectorize="schedule-file=schedules/x86_avx2/schedule_bf16.mlir"

// CHECK-LABEL: module attributes {transform.with_named_sequence}
// CHECK: transform.named_sequence @schedule_avx2_bf16
// CHECK: transform.structured.match ops
// CHECK-SAME: "linalg.matmul"
// CHECK: transform.structured.tile_reduction_using_for
// CHECK: transform.structured.tile_using_forall
// CHECK: transform.structured.vectorize
// CHECK: transform.structured.match ops
// CHECK-SAME: "linalg.generic"
// CHECK: transform.structured.tile_using_forall
// CHECK: transform.structured.vectorize
// CHECK: transform.yield

module attributes {transform.with_named_sequence} {
  transform.named_sequence @schedule_avx2_bf16(
      %root: !transform.any_op {transform.readonly}) {

    %matmuls = transform.structured.match ops{["linalg.matmul"]}
      in %root
      : (!transform.any_op) -> !transform.any_op

    %fill, %k_tiled, %combine, %k_loops = transform.structured.tile_reduction_using_for
        %matmuls by tile_sizes = [64]
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)

    %tiled_matmul, %forall_matmul = transform.structured.tile_using_forall %k_tiled
        tile_sizes [32, 64]
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

    transform.structured.vectorize %tiled_matmul
        vector_sizes [4, 8]
      : !transform.any_op

    %generics = transform.structured.match ops{["linalg.generic"]}
      in %root
      : (!transform.any_op) -> !transform.any_op

    %tiled_generic, %forall_generic = transform.structured.tile_using_forall %generics
        tile_sizes [32, 64]
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

    transform.structured.vectorize %tiled_generic
        vector_sizes [4, 8]
      : !transform.any_op

    transform.yield
  }
}
