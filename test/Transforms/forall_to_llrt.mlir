// RUN: llk-opt --forall-to-llrt %s | FileCheck %s

// CHECK:       func.func private @llk_worker
// CHECK-LABEL: func.func @forall_parallel
// CHECK:       call @llrtParallelFor
// CHECK-NOT:   scf.forall
func.func @forall_parallel(%arg0: memref<100xf32>) {
  scf.forall (%i) in (100) {
    %cst = arith.constant 1.0 : f32
    memref.store %cst, %arg0[%i] : memref<100xf32>
  }
  return
}
