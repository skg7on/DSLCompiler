// RUN: llk-opt --forall-to-openmp %s | FileCheck %s

// CHECK:       omp.parallel
// CHECK:       omp.wsloop
// CHECK:       memref.store
func.func @parallel_add(%arg0: memref<128x64xf32>) {
  scf.forall (%i, %j) in (8, 4) {
    %val = arith.constant 1.0 : f32
    memref.store %val, %arg0[%i, %j] : memref<128x64xf32>
  }
  return
}
