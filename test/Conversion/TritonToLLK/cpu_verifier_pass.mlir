// RUN: llk-opt --allow-unregistered-dialect --triton-cpu-verify %s | FileCheck %s
// Experimental input validation does not imply executable upstream TTIR.
// CHECK-LABEL: func @supported_ops
func.func @supported_ops(%base: memref<128x64xf32>) {
  %zero = arith.constant 0 : index
  %ptr = "tt.make_block_ptr"(%base, %zero, %zero) {
    shape = array<i64: 128, 64>, strides = array<i64: 64, 1>,
    block_shape = array<i64: 32, 64>, order = array<i64: 0, 1>
  } : (memref<128x64xf32>, index, index) -> !tt.ptr
  // CHECK: tt.load
  %x = "tt.load"(%ptr) : (!tt.ptr) -> vector<32x64xf32>
  // CHECK: tt.store
  "tt.store"(%ptr, %x) : (!tt.ptr, vector<32x64xf32>) -> ()
  // CHECK: return
  return
}
