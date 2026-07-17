// RUN: llk-opt --llk-to-linalg %s | FileCheck %s

// Test lowering of llk.fused_swiglu to scalar loop nests.
// M1 produces scf.for + arith + math + tensor ops.
// CHECK-DAG is used because ops appear in multiple regions
// (matmul bodies vs elementwise body).

// CHECK-LABEL: module
// CHECK-DAG: scf.for
// CHECK-DAG: arith.mulf
// CHECK-DAG: arith.addf
// CHECK-DAG: arith.extf
// CHECK-DAG: math.exp
// CHECK-DAG: arith.truncf
// CHECK-DAG: arith.divf
// CHECK-DAG: arith.negf

module {
  %x = tensor.empty() : tensor<2x4xbf16>
  %wg = tensor.empty() : tensor<4x8xbf16>
  %wu = tensor.empty() : tensor<4x8xbf16>
  %init = tensor.empty() : tensor<2x8xbf16>
  %y = "llk.fused_swiglu"(%x, %wg, %wu, %init) {accumulator_type = f32, activation = #llk.activation<silu>, math_mode = #llk.math_mode<bounded_fast>} : (tensor<2x4xbf16>, tensor<4x8xbf16>, tensor<4x8xbf16>, tensor<2x8xbf16>) -> tensor<2x8xbf16>
}
