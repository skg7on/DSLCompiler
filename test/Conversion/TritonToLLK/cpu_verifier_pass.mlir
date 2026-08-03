// RUN: llk-opt --allow-unregistered-dialect --triton-cpu-verify %s | FileCheck %s

// Whitelisted Triton ops must pass through the CPU verifier unchanged.
// The pass emits no diagnostics and the module round-trips unmodified.

// CHECK-LABEL: func @supported_ops
func.func @supported_ops(%ptr: !tt.ptr, %val: f32) {
  // CHECK: tt.load
  %x = "tt.load"(%ptr) : (!tt.ptr) -> f32
  // CHECK: tt.store
  "tt.store"(%ptr, %val) : (!tt.ptr, f32) -> ()
  // CHECK: return
  func.return
}
