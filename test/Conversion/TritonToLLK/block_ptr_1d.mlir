// RUN: llk-opt --allow-unregistered-dialect --triton-block-ptr-to-vector %s | FileCheck %s

// CHECK-LABEL: func @one_d_block_ptr_load
func.func @one_d_block_ptr_load(%base: memref<128xbf16>, %off: index)
    -> vector<32xbf16> {
  // CHECK: memref.subview
  %ptr = "tt.make_block_ptr"(%base, %off) {
    shape = array<i64: 128>,
    strides = array<i64: 1>,
    block_shape = array<i64: 32>,
    order = array<i64: 0>
  } : (memref<128xbf16>, index) -> !tt.ptr
  %loaded = "tt.load"(%ptr) : (!tt.ptr) -> vector<32xbf16>
  // CHECK: vector.transfer_read
  // CHECK-NOT: tt.make_block_ptr
  // CHECK-NOT: tt.load
  func.return %loaded : vector<32xbf16>
}
