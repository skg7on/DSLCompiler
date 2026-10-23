// RUN: llk-opt --allow-unregistered-dialect --verify-diagnostics --split-input-file --triton-shared-mem-to-scratch %s

func.func @shared_mem_alloc() {
  // expected-error @+1 {{triton_unsupported_op}}
  %smem = "tt.alloc"() : () -> memref<32x64xbf16, 3>
  // Keep the buffer alive so DCE cannot fold the newly created alloc away.
  memref.dealloc %smem : memref<32x64xbf16, 3>
  func.return
}

// -----

func.func @async_copy_to_scratch(%global: memref<32x64xbf16>) {
  // expected-error @+1 {{triton_unsupported_op}}
  %smem = "tt.alloc"() : () -> memref<32x64xbf16, 3>
  "tt.async_copy"(%global, %smem) : (memref<32x64xbf16>, memref<32x64xbf16, 3>) -> ()
  func.return
}
