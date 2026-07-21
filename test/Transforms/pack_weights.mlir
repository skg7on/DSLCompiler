// RUN: llk-opt --llk-to-linalg --pack-weights %s | FileCheck %s
//
// Test that the PackWeights pass annotates weight operands with packed layout.
// Pipeline: LLK dialect -> Linalg -> annotate weight operands
//
// Input: llk.fused_swiglu with concrete shapes (function block arguments)
// Parameters: M=32, N=128, K=64
//
// Expected output: weight function arguments have llk.packed_layout = 64

func.func @swiglu_packed(%x: tensor<32x64xbf16>, %wg: tensor<64x128xbf16>,
    %wu: tensor<64x128xbf16>, %init: tensor<32x128xbf16>) -> tensor<32x128xbf16> {
  %y = llk.fused_swiglu ins(%x, %wg, %wu : tensor<32x64xbf16>, tensor<64x128xbf16>, tensor<64x128xbf16>)
      outs(%init : tensor<32x128xbf16>)
      {accumulator_type = f32, activation = #llk.activation<silu>, math_mode = #llk.math_mode<bounded_fast>}
      -> tensor<32x128xbf16>
  return %y : tensor<32x128xbf16>
}

// CHECK-LABEL: func.func @swiglu_packed
// The pass annotates weight block arguments with packed_layout
// CHECK: tensor<64x128xbf16> {llk.packed_layout = 64 : i64}
