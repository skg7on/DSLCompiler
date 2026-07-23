// RUN: llk-opt --select-schedule %s 2>&1 | FileCheck %s

func.func @swiglu_M2_N8192_K8192(%x: tensor<2x8192xbf16>, %wg: tensor<8192x8192xbf16>, %wu: tensor<8192x8192xbf16>, %init: tensor<2x8192xbf16>) -> tensor<2x8192xbf16> {
  %y = llk.fused_swiglu ins(%x, %wg, %wu : tensor<2x8192xbf16>, tensor<8192x8192xbf16>, tensor<8192x8192xbf16>)
      outs(%init : tensor<2x8192xbf16>)
      {accumulator_type = f32, activation = #llk.activation<silu>, math_mode = #llk.math_mode<bounded_fast>}
      -> tensor<2x8192xbf16>
  return %y : tensor<2x8192xbf16>
}
// CHECK: selected schedule for fused_swiglu: BM={{.*}} BN={{.*}} BK={{.*}}
