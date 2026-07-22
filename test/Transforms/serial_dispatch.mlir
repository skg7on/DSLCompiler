// RUN: llk-opt --serial-parallel-dispatch %s | FileCheck %s

// Small tile count (< num_threads) → serial scf.for
// CHECK-LABEL: func.func @small_tiles
// CHECK:       scf.for
// CHECK-NOT:   scf.forall

func.func @small_tiles(%arg0: memref<3xf32>) {
  scf.forall (%i) in (3) {
    %val = arith.constant 1.0 : f32
    memref.store %val, %arg0[%i] : memref<3xf32>
  }
  return
}

// Large tile count (>= num_threads) → scf.forall kept
// CHECK-LABEL: func.func @large_tiles
// CHECK:       scf.forall
// CHECK-NOT:   scf.for

func.func @large_tiles(%arg0: memref<100xf32>) {
  scf.forall (%i) in (100) {
    %val = arith.constant 2.0 : f32
    memref.store %val, %arg0[%i] : memref<100xf32>
  }
  return
}
