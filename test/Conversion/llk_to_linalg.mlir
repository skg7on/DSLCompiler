// RUN: llk-opt --llk-to-linalg %s | FileCheck %s

// Test lowering of llk.fused_swiglu to linalg named ops + generic.
// M1 produces:
//   - linalg.fill   (accumulator init)
//   - linalg.matmul (gate projection)
//   - linalg.matmul (up projection)
//   - linalg.generic (SiLU, computed in f32, trunc to bf16)

// CHECK-LABEL: module
// CHECK: linalg.fill
// CHECK: linalg.matmul
// CHECK: linalg.matmul
// CHECK: linalg.generic
// CHECK: arith.extf
// CHECK: math.exp
// CHECK: arith.truncf

module {
  %x = tensor.empty() : tensor<2x4xbf16>
  %wg = tensor.empty() : tensor<4x8xbf16>
  %wu = tensor.empty() : tensor<4x8xbf16>
  %init = tensor.empty() : tensor<2x8xbf16>
  %y = "llk.fused_swiglu"(%x, %wg, %wu, %init) {accumulator_type = f32, activation = #llk.activation<silu>, math_mode = #llk.math_mode<bounded_fast>} : (tensor<2x4xbf16>, tensor<4x8xbf16>, tensor<4x8xbf16>, tensor<2x8xbf16>) -> tensor<2x8xbf16>
}
