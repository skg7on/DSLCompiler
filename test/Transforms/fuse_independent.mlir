// RUN: llk-opt --llk-to-linalg --fuse-double-contraction %s | FileCheck %s
//
// Test: Two independent swiglus with different X are each independently fused.
// Each swiglu's internal matmuls share the same X (%x1 for swiglu1, %x2 for
// swiglu2), so each produces its own fused generic with 2 reduction outputs.
//
// CHECK-COUNT-2: iterator_types = ["parallel", "parallel", "reduction"]

func.func @different_x(%x1: tensor<32x64xbf16>, %x2: tensor<32x64xbf16>,
    %wg: tensor<64x128xbf16>, %wu: tensor<64x128xbf16>,
    %init1: tensor<32x128xbf16>, %init2: tensor<32x128xbf16>) -> (tensor<32x128xbf16>, tensor<32x128xbf16>) {
  %y1 = llk.fused_swiglu ins(%x1, %wg, %wu : tensor<32x64xbf16>, tensor<64x128xbf16>, tensor<64x128xbf16>)
      outs(%init1 : tensor<32x128xbf16>)
      {accumulator_type = f32, activation = #llk.activation<silu>, math_mode = #llk.math_mode<bounded_fast>}
      -> tensor<32x128xbf16>
  %y2 = llk.fused_swiglu ins(%x2, %wg, %wu : tensor<32x64xbf16>, tensor<64x128xbf16>, tensor<64x128xbf16>)
      outs(%init2 : tensor<32x128xbf16>)
      {accumulator_type = f32, activation = #llk.activation<silu>, math_mode = #llk.math_mode<bounded_fast>}
      -> tensor<32x128xbf16>
  return %y1, %y2 : tensor<32x128xbf16>, tensor<32x128xbf16>
}
