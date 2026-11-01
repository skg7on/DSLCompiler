// RUN: llk-opt --llk-to-linalg %s | llk-opt | FileCheck %s
//
// PF11. `llk.fused_swiglu` covers two uniform storage contracts: the original
// bf16 operands/result with an f32 accumulator, and the uniform-f32 form a
// frontend source declares. Mixed storage is rejected by the verifier (see
// test/Dialect/ops_invalid.mlir). The Linalg lowering keeps the f32 form f32
// end to end -- no extension or truncation is invented -- while the bf16 form
// keeps its f32 internal arithmetic and rounds only at the result.

func.func @swiglu_f32(%x: tensor<2x4xf32>, %wg: tensor<4x8xf32>,
                      %wu: tensor<4x8xf32>, %init: tensor<2x8xf32>)
    -> tensor<2x8xf32> {
  %y = llk.fused_swiglu ins(%x, %wg, %wu : tensor<2x4xf32>,
                            tensor<4x8xf32>, tensor<4x8xf32>)
      outs(%init : tensor<2x8xf32>)
      {accumulator_type = f32, activation = #llk.activation<silu>,
       math_mode = #llk.math_mode<strict>}
      -> tensor<2x8xf32>
  return %y : tensor<2x8xf32>
}

// The two projections become ordinary f32 matmuls over the declared operands.
// CHECK-LABEL: func.func @swiglu_f32
// CHECK: linalg.matmul
// A uniform f32 source needs no width conversion at all.
// CHECK-NOT: arith.extf
// CHECK-NOT: arith.truncf
// The exact activation policy is the IEEE exponential, not an approximation.
// CHECK: math.exp
// CHECK-NOT: arith.extf
// CHECK-NOT: arith.truncf
// CHECK: return

func.func @swiglu_bf16(%x: tensor<2x4xbf16>, %wg: tensor<4x8xbf16>,
                       %wu: tensor<4x8xbf16>, %init: tensor<2x8xbf16>)
    -> tensor<2x8xbf16> {
  %y = llk.fused_swiglu ins(%x, %wg, %wu : tensor<2x4xbf16>,
                            tensor<4x8xbf16>, tensor<4x8xbf16>)
      outs(%init : tensor<2x8xbf16>)
      {accumulator_type = f32, activation = #llk.activation<silu>,
       math_mode = #llk.math_mode<strict>}
      -> tensor<2x8xbf16>
  return %y : tensor<2x8xbf16>
}

// The bf16 contract is unchanged: widen to f32 for the activation, round once
// at the result.
// CHECK-LABEL: func.func @swiglu_bf16
// CHECK: arith.extf
// CHECK: math.exp
// CHECK: arith.truncf
// CHECK: return
