// RUN: llk-opt --allow-unregistered-dialect --triton-atomic-to-llrt %s | FileCheck %s

// CHECK-LABEL: func @atomic_add
func.func @atomic_add(%ptr: !llvm.ptr, %val: f32) {
  // CHECK: call @llrt.atomic_add_f32
  %old = "tt.atomic_add"(%ptr, %val) : (!llvm.ptr, f32) -> f32
  // CHECK-NOT: tt.atomic_add
  func.return
}

// -----

// CHECK-LABEL: func @atomic_max
func.func @atomic_max(%ptr: !llvm.ptr, %val: f32) {
  // CHECK: call @llrt.atomic_max_f32
  %old = "tt.atomic_max"(%ptr, %val) : (!llvm.ptr, f32) -> f32
  func.return
}

// -----

// CHECK-LABEL: func @atomic_cas
func.func @atomic_cas(%ptr: !llvm.ptr, %cmp: i32, %val: i32) {
  // CHECK: call @llrt.atomic_cas_i32
  %old = "tt.atomic_cas"(%ptr, %cmp, %val) : (!llvm.ptr, i32, i32) -> i32
  func.return
}
