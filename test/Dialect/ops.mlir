// RUN: llk-opt %s | llk-opt | FileCheck %s

// Test that the LLK dialect is registered and that llk.fused_swiglu
// operations can be parsed and printed (round-trip).
//
// Note: Uses the generic MLIR form because func.func custom assembly format
// is not available in this environment (MLIR 20.1.8 build configuration).
// The LLK op itself is tested in custom form below.

// CHECK-LABEL: func.func @test_swiglu_roundtrip
func.func @test_swiglu_roundtrip(%x: tensor<?x?xbf16>, %wg: tensor<?x?xbf16>, %wu: tensor<?x?xbf16>, %init: tensor<?x?xbf16>) -> tensor<?x?xbf16> {
  // CHECK: llk.fused_swiglu
  %y = llk.fused_swiglu ins(%x, %wg, %wu : tensor<?x?xbf16>, tensor<?x?xbf16>, tensor<?x?xbf16>)
      outs(%init : tensor<?x?xbf16>)
      {accumulator_type = f32, activation = #llk.activation<silu>, math_mode = #llk.math_mode<bounded_fast>}
      -> tensor<?x?xbf16>
  // CHECK: return
  return %y : tensor<?x?xbf16>
}
