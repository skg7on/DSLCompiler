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

// -----
// PF11: the f32 contract is uniform. A bf16 gate weight beside an f32 input
// would either round the source or widen the weight silently, so it is
// rejected instead of being given an implicit conversion.

module {
  %x = tensor.empty() : tensor<2x4xf32>
  %wg = tensor.empty() : tensor<4x8xbf16>
  %wu = tensor.empty() : tensor<4x8xf32>
  %init = tensor.empty() : tensor<2x8xf32>
  // expected-error@+1 {{must share one storage element type}}
  %y = llk.fused_swiglu ins(%x, %wg, %wu : tensor<2x4xf32>, tensor<4x8xbf16>, tensor<4x8xf32>)
      outs(%init : tensor<2x8xf32>)
      {accumulator_type = f32, activation = #llk.activation<silu>, math_mode = #llk.math_mode<strict>}
      -> tensor<2x8xf32>
}

// -----
// The result is written in the same storage the operands declare. A bf16
// result of f32 operands would hide a narrowing conversion the source never
// asked for.

module {
  %x = tensor.empty() : tensor<2x4xf32>
  %wg = tensor.empty() : tensor<4x8xf32>
  %wu = tensor.empty() : tensor<4x8xf32>
  %init = tensor.empty() : tensor<2x8xf32>
  // expected-error@+1 {{result must use the operands' storage element type}}
  %y = llk.fused_swiglu ins(%x, %wg, %wu : tensor<2x4xf32>, tensor<4x8xf32>, tensor<4x8xf32>)
      outs(%init : tensor<2x8xf32>)
      {accumulator_type = f32, activation = #llk.activation<silu>, math_mode = #llk.math_mode<strict>}
      -> tensor<2x8xbf16>
}

// -----
// Both storage contracts accumulate in f32. A bf16 accumulator would be a
// different numerical contract that neither lowering implements.

module {
  %x = tensor.empty() : tensor<2x4xbf16>
  %wg = tensor.empty() : tensor<4x8xbf16>
  %wu = tensor.empty() : tensor<4x8xbf16>
  %init = tensor.empty() : tensor<2x8xbf16>
  // expected-error@+1 {{accumulator_type must be f32}}
  %y = llk.fused_swiglu ins(%x, %wg, %wu : tensor<2x4xbf16>, tensor<4x8xbf16>, tensor<4x8xbf16>)
      outs(%init : tensor<2x8xbf16>)
      {accumulator_type = bf16, activation = #llk.activation<silu>, math_mode = #llk.math_mode<strict>}
      -> tensor<2x8xbf16>
}
