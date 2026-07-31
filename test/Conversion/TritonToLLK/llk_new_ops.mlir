// RUN: llk-opt --verify-diagnostics --split-input-file %s | FileCheck %s

// CHECK-LABEL: func @test_matmul_op
func.func @test_matmul_op(%a: tensor<32x64xbf16>, %b: tensor<64x32xbf16>,
                          %init: tensor<32x32xbf16>) -> tensor<32x32xbf16> {
  // CHECK: llk.matmul
  %0 = llk.matmul ins(%a, %b : tensor<32x64xbf16>, tensor<64x32xbf16>)
      outs(%init : tensor<32x32xbf16>)
      {accumulator_type = f32, math_mode = #llk.math_mode<triton_fast>}
      -> tensor<32x32xbf16>
  func.return %0 : tensor<32x32xbf16>
}

// -----

// CHECK-LABEL: func @test_make_tensor_op
func.func @test_make_tensor_op(%ptr : i64) -> tensor<16x32xbf16> {
  // CHECK: llk.make_tensor
  %0 = llk.make_tensor from(%ptr : i64) shape(16, 32) dtype(bf16) -> tensor<16x32xbf16>
  func.return %0 : tensor<16x32xbf16>
}

// -----

// CHECK-LABEL: func @test_matmul_dynamic_k
func.func @test_matmul_dynamic_k(%a: tensor<32x?xbf16>, %b: tensor<?x32xbf16>,
                                 %init: tensor<32x32xbf16>) -> tensor<32x32xbf16> {
  // CHECK: llk.matmul
  %0 = llk.matmul ins(%a, %b : tensor<32x?xbf16>, tensor<?x32xbf16>)
      outs(%init : tensor<32x32xbf16>)
      {accumulator_type = f32, math_mode = #llk.math_mode<triton_fast>}
      -> tensor<32x32xbf16>
  func.return %0 : tensor<32x32xbf16>
}
