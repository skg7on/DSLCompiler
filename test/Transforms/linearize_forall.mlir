// RUN: llk-opt --linearize-forall %s | FileCheck %s
//
// Test that multi-dimensional scf.forall is linearized to 1D.

// CHECK-LABEL: func.func @linearize_2d
// CHECK:       scf.forall
// CHECK-COUNT-1:   arith.divsi
// CHECK-COUNT-1:   arith.remsi
func.func @linearize_2d(%arg0: memref<4x8xf32>) {
  scf.forall (%m, %n) in (4, 8) {
    %val = arith.constant 1.0 : f32
    memref.store %val, %arg0[%m, %n] : memref<4x8xf32>
  }
  return
}

// CHECK-LABEL: func.func @passthrough_1d
// CHECK:       scf.forall
// CHECK-NOT:   arith.divsi
// CHECK-NOT:   arith.remsi
func.func @passthrough_1d(%arg0: memref<32xf32>) {
  scf.forall (%i) in (32) {
    %val = arith.constant 2.0 : f32
    memref.store %val, %arg0[%i] : memref<32xf32>
  }
  return
}
