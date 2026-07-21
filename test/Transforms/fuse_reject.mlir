// RUN: llk-opt --fuse-double-contraction %s | FileCheck %s
//
// Test: Two matmuls sharing X, but consumer is NOT SiLU (just add).
// The pass should NOT fuse because the body pattern doesn't match SiLU.
//
// CHECK: linalg.matmul
// CHECK: linalg.matmul
// CHECK-NOT: iterator_types = ["parallel", "parallel", "reduction"]
// Two matmuls remain separate because consumer is not SiLU

func.func @shared_x_non_silu_consumer(%x: tensor<32x64xf32>, %wg: tensor<64x128xf32>,
    %wu: tensor<64x128xf32>, %init: tensor<32x128xf32>) -> tensor<32x128xf32> {
  %c0 = arith.constant 0.0 : f32
  %gate_init = tensor.empty() : tensor<32x128xf32>
  %gate_fill = linalg.fill ins(%c0 : f32) outs(%gate_init : tensor<32x128xf32>) -> tensor<32x128xf32>
  %up_init = tensor.empty() : tensor<32x128xf32>
  %up_fill = linalg.fill ins(%c0 : f32) outs(%up_init : tensor<32x128xf32>) -> tensor<32x128xf32>
  %gate = linalg.matmul ins(%x, %wg : tensor<32x64xf32>, tensor<64x128xf32>)
      outs(%gate_fill : tensor<32x128xf32>) -> tensor<32x128xf32>
  %up = linalg.matmul ins(%x, %wu : tensor<32x64xf32>, tensor<64x128xf32>)
      outs(%up_fill : tensor<32x128xf32>) -> tensor<32x128xf32>
  %result = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                       affine_map<(d0, d1) -> (d0, d1)>,
                       affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]}
      ins(%gate, %up : tensor<32x128xf32>, tensor<32x128xf32>)
      outs(%init : tensor<32x128xf32>) {
  ^bb0(%g: f32, %u: f32, %o: f32):
    %sum = arith.addf %g, %u : f32
    linalg.yield %sum : f32
  } -> tensor<32x128xf32>
  return %result : tensor<32x128xf32>
}
