// RUN: llk-opt --llk-to-linalg %s | FileCheck %s --implicit-check-not=math.cos --implicit-check-not=math.sin

func.func @rope_test(%x: tensor<1x1x128x64xf32>, %pos: tensor<128xi64>, %init: tensor<1x1x128x64xf32>) -> tensor<1x1x128x64xf32> {
  %y = llk.rope ins(%x, %pos : tensor<1x1x128x64xf32>, tensor<128xi64>)
      outs(%init : tensor<1x1x128x64xf32>)
      {theta = 10000.0 : f64, math_mode = #llk.math_mode<bounded_fast>}
      -> tensor<1x1x128x64xf32>
  return %y : tensor<1x1x128x64xf32>
}
// CHECK: math.powf
// CHECK: math.round
// CHECK: arith.remsi
// CHECK: math.round
// CHECK: arith.remsi
// CHECK: linalg.generic
