// RUN: llk-opt %s | FileCheck %s

func.func @assume_test(%x: tensor<128x64xbf16>, %init: tensor<128x64xbf16>) -> tensor<128x64xbf16> {
  %0 = llk.assume ins(%x : tensor<128x64xbf16>)
      outs(%init : tensor<128x64xbf16>)
      {divisible_dims = array<i64: 0, 1>, divisible_factors = array<i64: 8, 64>, alignment_bytes = 64}
      -> tensor<128x64xbf16>
  return %0 : tensor<128x64xbf16>
}
// CHECK: llk.assume
// CHECK: alignment_bytes = 64 : i64
// CHECK: divisible_dims
