// RUN: llk-opt --allow-unregistered-dialect --verify-diagnostics --split-input-file --triton-atomic-to-llrt %s

func.func @atomic_add(%ptr: !llvm.ptr, %val: f32) {
  // expected-error @+1 {{triton_unsupported_op}}
  %old = "tt.atomic_add"(%ptr, %val) : (!llvm.ptr, f32) -> f32
  func.return
}

// -----

func.func @atomic_max(%ptr: !llvm.ptr, %val: f32) {
  // expected-error @+1 {{triton_unsupported_op}}
  %old = "tt.atomic_max"(%ptr, %val) : (!llvm.ptr, f32) -> f32
  func.return
}

// -----

func.func @atomic_cas(%ptr: !llvm.ptr, %cmp: i32, %val: i32) {
  // expected-error @+1 {{triton_unsupported_op}}
  %old = "tt.atomic_cas"(%ptr, %cmp, %val) : (!llvm.ptr, i32, i32) -> i32
  func.return
}
