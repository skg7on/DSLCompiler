// RUN: llk-opt --llk-to-linalg --fuse-double-contraction %s | FileCheck %s
//
// Test: FuseDoubleContraction fuses 2 linalg.matmul ops sharing X + SiLU
// consumer into a single linalg.generic with 2 reduction outputs.
//
// Input: llk.fused_swiglu with BF16 tensors
// Expected: linalg.generic with parallel + reduction iterator types, arith ops
//
// CHECK: linalg.generic
// CHECK-SAME: iterator_types = ["parallel", "parallel", "reduction"]
// CHECK-DAG: arith.addf
// CHECK-DAG: arith.mulf
// CHECK-DAG: arith.extf

func.func @fusible_swiglu(%x: tensor<32x64xbf16>, %wg: tensor<64x128xbf16>,
    %wu: tensor<64x128xbf16>, %init: tensor<32x128xbf16>) -> tensor<32x128xbf16> {
  %y = llk.fused_swiglu ins(%x, %wg, %wu : tensor<32x64xbf16>, tensor<64x128xbf16>, tensor<64x128xbf16>)
      outs(%init : tensor<32x128xbf16>)
      {accumulator_type = f32, activation = #llk.activation<silu>, math_mode = #llk.math_mode<bounded_fast>}
      -> tensor<32x128xbf16>
  return %y : tensor<32x128xbf16>
}
