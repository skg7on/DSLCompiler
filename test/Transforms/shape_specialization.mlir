// RUN: llk-opt --shape-specialize %s 2>&1 | FileCheck %s

func.func @bucket_M1(%x: tensor<1x256xbf16>, %wg: tensor<256x512xbf16>, %wu: tensor<256x512xbf16>, %init: tensor<1x512xbf16>) -> tensor<1x512xbf16> {
  %y = llk.fused_swiglu ins(%x, %wg, %wu : tensor<1x256xbf16>, tensor<256x512xbf16>, tensor<256x512xbf16>)
      outs(%init : tensor<1x512xbf16>)
      {accumulator_type = f32, activation = #llk.activation<silu>, math_mode = #llk.math_mode<bounded_fast>}
      -> tensor<1x512xbf16>
  return %y : tensor<1x512xbf16>
}
// CHECK: M=1 -> bucket 0

func.func @bucket_M4(%x: tensor<4x256xbf16>, %wg: tensor<256x512xbf16>, %wu: tensor<256x512xbf16>, %init: tensor<4x512xbf16>) -> tensor<4x512xbf16> {
  %y = llk.fused_swiglu ins(%x, %wg, %wu : tensor<4x256xbf16>, tensor<256x512xbf16>, tensor<256x512xbf16>)
      outs(%init : tensor<4x512xbf16>)
      {accumulator_type = f32, activation = #llk.activation<silu>, math_mode = #llk.math_mode<bounded_fast>}
      -> tensor<4x512xbf16>
  return %y : tensor<4x512xbf16>
}
// CHECK: M=4 -> bucket 1

func.func @bucket_M16(%x: tensor<16x256xbf16>, %wg: tensor<256x512xbf16>, %wu: tensor<256x512xbf16>, %init: tensor<16x512xbf16>) -> tensor<16x512xbf16> {
  %y = llk.fused_swiglu ins(%x, %wg, %wu : tensor<16x256xbf16>, tensor<256x512xbf16>, tensor<256x512xbf16>)
      outs(%init : tensor<16x512xbf16>)
      {accumulator_type = f32, activation = #llk.activation<silu>, math_mode = #llk.math_mode<bounded_fast>}
      -> tensor<16x512xbf16>
  return %y : tensor<16x512xbf16>
}
// CHECK: M=16 -> bucket 2

func.func @bucket_M64(%x: tensor<64x256xbf16>, %wg: tensor<256x512xbf16>, %wu: tensor<256x512xbf16>, %init: tensor<64x512xbf16>) -> tensor<64x512xbf16> {
  %y = llk.fused_swiglu ins(%x, %wg, %wu : tensor<64x256xbf16>, tensor<256x512xbf16>, tensor<256x512xbf16>)
      outs(%init : tensor<64x512xbf16>)
      {accumulator_type = f32, activation = #llk.activation<silu>, math_mode = #llk.math_mode<bounded_fast>}
      -> tensor<64x512xbf16>
  return %y : tensor<64x512xbf16>
}
// CHECK: M=64 -> bucket 3

func.func @bucket_M128(%x: tensor<128x256xbf16>, %wg: tensor<256x512xbf16>, %wu: tensor<256x512xbf16>, %init: tensor<128x512xbf16>) -> tensor<128x512xbf16> {
  %y = llk.fused_swiglu ins(%x, %wg, %wu : tensor<128x256xbf16>, tensor<256x512xbf16>, tensor<256x512xbf16>)
      outs(%init : tensor<128x512xbf16>)
      {accumulator_type = f32, activation = #llk.activation<silu>, math_mode = #llk.math_mode<bounded_fast>}
      -> tensor<128x512xbf16>
  return %y : tensor<128x512xbf16>
}
// CHECK: M=128 -> bucket 4
