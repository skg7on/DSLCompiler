// RUN: llk-opt --triton-to-structured --allow-unregistered-dialect %s | FileCheck %s

// CHECK-LABEL: func @generic_dot
func.func @generic_dot(%a: tensor<32x64xbf16>, %b: tensor<64x32xbf16>, %init: tensor<32x32xf32>) -> tensor<32x32xf32> {
  // Single tt.dot should become linalg.matmul
  %result = "tt.dot"(%a, %b, %init) : (tensor<32x64xbf16>, tensor<64x32xbf16>, tensor<32x32xf32>) -> tensor<32x32xf32>
  // CHECK: linalg.matmul
  // CHECK-NOT: tt.dot
  func.return %result : tensor<32x32xf32>
}
