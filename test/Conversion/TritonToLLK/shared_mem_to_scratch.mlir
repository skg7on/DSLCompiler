// RUN: llk-opt --allow-unregistered-dialect --triton-shared-mem-to-scratch %s | FileCheck %s

// CHECK-LABEL: func @shared_mem_alloc
func.func @shared_mem_alloc() {
  // CHECK: memref.alloc
  %smem = "tt.alloc"() : () -> memref<32x64xbf16, 3>
  // Keep the buffer alive so DCE cannot fold the newly created alloc away.
  memref.dealloc %smem : memref<32x64xbf16, 3>
  // CHECK: memref.dealloc
  // CHECK-NOT: tt.alloc
  func.return
}

// -----

// CHECK-LABEL: func @async_copy_to_scratch
func.func @async_copy_to_scratch(%global: memref<32x64xbf16>) {
  %smem = "tt.alloc"() : () -> memref<32x64xbf16, 3>
  "tt.async_copy"(%global, %smem) : (memref<32x64xbf16>, memref<32x64xbf16, 3>) -> ()
  // CHECK: memref.copy
  // CHECK-NOT: tt.async_copy
  func.return
}
