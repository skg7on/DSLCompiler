// RUN: llk-opt --allow-unregistered-dialect --triton-grid-to-forall %s | FileCheck %s

// CHECK-LABEL: func @one_d_grid
func.func @one_d_grid(%M: index, %BM: index) {
  // CHECK: arith.ceildivsi
  // CHECK: scf.forall
  %pid = "tt.get_program_id"() {axis = 0 : i32} : () -> index
  %c2 = arith.constant 2 : index
  %result = arith.muli %pid, %c2 : index
  // CHECK: arith.muli
  // CHECK-NOT: tt.get_program_id
  func.return
}

// -----

// CHECK-LABEL: func @two_d_grid
func.func @two_d_grid(%M: index, %N: index, %BM: index, %BN: index) {
  // CHECK: scf.forall
  // CHECK: arith.divsi
  // CHECK: arith.remsi
  %pid_m = "tt.get_program_id"() {axis = 0 : i32} : () -> index
  %pid_n = "tt.get_program_id"() {axis = 1 : i32} : () -> index
  %result = arith.muli %pid_m, %pid_n : index
  // CHECK: arith.muli
  // CHECK-NOT: tt.get_program_id
  func.return
}

// -----

// Regression test for Task 6 Finding B: grid dims are index-typed function
// args.  A real Triton signature may lead with pointer/block args, so the
// pass must scan for index-typed args instead of assuming args[0..] encode
// the grid.  Before the fix this fed the pointer arg into arith.ceildivsi
// (type mismatch) and the module failed verification.

// CHECK-LABEL: func @leading_ptr_arg
func.func @leading_ptr_arg(%ptr: !tt.ptr<f32>, %M: index, %BM: index) {
  // CHECK: arith.ceildivsi %{{.*}}, %{{.*}} : index
  // CHECK: scf.forall
  %pid = "tt.get_program_id"() {axis = 0 : i32} : () -> index
  %c2 = arith.constant 2 : index
  %result = arith.muli %pid, %c2 : index
  // CHECK: arith.muli
  // CHECK-NOT: tt.get_program_id
  func.return
}
