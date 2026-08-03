// RUN: llk-opt --allow-unregistered-dialect --triton-block-ptr-to-vector %s | FileCheck %s

// CHECK-LABEL: func @partial_tile_mask
func.func @partial_tile_mask(%base: memref<128x127xbf16>,
                             %off_m: index, %off_n: index)
    -> vector<32x8xbf16> {
  // N=127 is not a multiple of VN=8 — the last tile needs masking.
  // The mask covers the remaining elements: (128 - off_m, 127 - off_n).
  %ptr = "tt.make_block_ptr"(%base, %off_m, %off_n) {
    shape = array<i64: 128, 127>,
    strides = array<i64: 127, 1>,
    block_shape = array<i64: 32, 8>,
    order = array<i64: 0, 1>
  } : (memref<128x127xbf16>, index, index) -> !tt.ptr
  %loaded = "tt.load"(%ptr) : (!tt.ptr) -> vector<32x8xbf16>
  // CHECK: %[[MASK:.*]] = vector.create_mask
  // CHECK: vector.transfer_read {{.*}} %[[MASK]]
  // CHECK-NOT: tt.make_block_ptr
  // CHECK-NOT: tt.load
  func.return %loaded : vector<32x8xbf16>
}
