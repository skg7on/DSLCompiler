// RUN: llk-opt --select-schedule %s 2>&1 | FileCheck %s

func.func @swiglu_M1_N4096_K4096(%x: tensor<1x4096xbf16>, %wg: tensor<4096x4096xbf16>, %wu: tensor<4096x4096xbf16>, %init: tensor<1x4096xbf16>) -> tensor<1x4096xbf16> {
  %y = llk.fused_swiglu ins(%x, %wg, %wu : tensor<1x4096xbf16>, tensor<4096x4096xbf16>, tensor<4096x4096xbf16>)
      outs(%init : tensor<1x4096xbf16>)
      {accumulator_type = f32, activation = #llk.activation<silu>, math_mode = #llk.math_mode<bounded_fast>}
      -> tensor<1x4096xbf16>
  return %y : tensor<1x4096xbf16>
}
// CHECK: selected schedule for fused_swiglu: BM=

func.func @swiglu_M128_N4096_K4096(%x: tensor<128x4096xbf16>, %wg: tensor<4096x4096xbf16>, %wu: tensor<4096x4096xbf16>, %init: tensor<128x4096xbf16>) -> tensor<128x4096xbf16> {
  %y = llk.fused_swiglu ins(%x, %wg, %wu : tensor<128x4096xbf16>, tensor<4096x4096xbf16>, tensor<4096x4096xbf16>)
      outs(%init : tensor<128x4096xbf16>)
      {accumulator_type = f32, activation = #llk.activation<silu>, math_mode = #llk.math_mode<bounded_fast>}
      -> tensor<128x4096xbf16>
  return %y : tensor<128x4096xbf16>
}
// CHECK: selected schedule for fused_swiglu: BM=
