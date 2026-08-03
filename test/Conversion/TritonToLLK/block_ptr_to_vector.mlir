// RUN: llk-opt --allow-unregistered-dialect --triton-block-ptr-to-vector %s | FileCheck %s

// CHECK-LABEL: func @two_d_block_ptr_load
func.func @two_d_block_ptr_load(%base: memref<128x64xbf16>,
                                %off_m: index, %off_k: index)
    -> vector<32x64xbf16> {
  // CHECK: memref.subview
  %ptr = "tt.make_block_ptr"(%base, %off_m, %off_k) {
    shape = array<i64: 128, 64>,
    strides = array<i64: 64, 1>,
    block_shape = array<i64: 32, 64>,
    order = array<i64: 0, 1>
  } : (memref<128x64xbf16>, index, index) -> !tt.ptr
  %loaded = "tt.load"(%ptr) : (!tt.ptr) -> vector<32x64xbf16>
  // CHECK: vector.transfer_read
  // CHECK-NOT: tt.make_block_ptr
  // CHECK-NOT: tt.load
  func.return %loaded : vector<32x64xbf16>
}

// -----

// CHECK-LABEL: func @masked_store
func.func @masked_store(%base: memref<128x64xbf16>, %off_m: index, %off_n: index,
                        %val: vector<32x8xbf16>, %mask: vector<32x8xi1>) {
  // CHECK: vector.transfer_write
  %ptr = "tt.make_block_ptr"(%base, %off_m, %off_n) {
    shape = array<i64: 128, 64>,
    strides = array<i64: 64, 1>,
    block_shape = array<i64: 32, 8>,
    order = array<i64: 0, 1>
  } : (memref<128x64xbf16>, index, index) -> !tt.ptr
  "tt.store"(%ptr, %val, %mask) : (!tt.ptr, vector<32x8xbf16>, vector<32x8xi1>) -> ()
  // CHECK-NOT: tt.store
  func.return
}
