// RUN: llk-opt --verify-diagnostics --split-input-file %s

// Test that the verifier catches shape inconsistencies in llk.fused_swiglu.

// -----
// Test: mismatched M dimension. X has M=2, init has M=3.

module {
  %x = tensor.empty() : tensor<2x4xbf16>
  %wg = tensor.empty() : tensor<4x8xbf16>
  %wu = tensor.empty() : tensor<4x8xbf16>
  %init = tensor.empty() : tensor<3x8xbf16>
  // expected-error@+1 {{M dimension mismatch}}
  %y = llk.fused_swiglu ins(%x, %wg, %wu : tensor<2x4xbf16>, tensor<4x8xbf16>, tensor<4x8xbf16>)
      outs(%init : tensor<3x8xbf16>)
      {accumulator_type = f32, activation = #llk.activation<silu>, math_mode = #llk.math_mode<bounded_fast>}
      -> tensor<?x?xbf16>
}

// -----
// Test: mismatched K dimension. X has K=4, Wg has K=5.

module {
  %x = tensor.empty() : tensor<2x4xbf16>
  %wg = tensor.empty() : tensor<5x8xbf16>
  %wu = tensor.empty() : tensor<4x8xbf16>
  %init = tensor.empty() : tensor<2x8xbf16>
  // expected-error@+1 {{K dimension mismatch}}
  %y = llk.fused_swiglu ins(%x, %wg, %wu : tensor<2x4xbf16>, tensor<5x8xbf16>, tensor<4x8xbf16>)
      outs(%init : tensor<2x8xbf16>)
      {accumulator_type = f32, activation = #llk.activation<silu>, math_mode = #llk.math_mode<bounded_fast>}
      -> tensor<?x?xbf16>
}
