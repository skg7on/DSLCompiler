// RUN: llk-opt --llk-to-linalg %s | FileCheck %s

// math_mode = triton_fast must select the bounded-fast exp path in
// createApproxExp, the way createApproxSin/createApproxCos already do for it.
// The approximation is written with math.exp2 (2^n * P(r)); falling through to
// the strict path instead emits math.exp, so math.exp2 is the observable
// difference between the two.

// CHECK: math.exp2
// CHECK: arith.truncf

module {
  %x = tensor.empty() : tensor<2x4xbf16>
  %wg = tensor.empty() : tensor<4x8xbf16>
  %wu = tensor.empty() : tensor<4x8xbf16>
  %init = tensor.empty() : tensor<2x8xbf16>
  %y = "llk.fused_swiglu"(%x, %wg, %wu, %init) {accumulator_type = f32, activation = #llk.activation<silu>, math_mode = #llk.math_mode<triton_fast>} : (tensor<2x4xbf16>, tensor<4x8xbf16>, tensor<4x8xbf16>, tensor<2x8xbf16>) -> tensor<2x8xbf16>
}
