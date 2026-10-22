// RUN: llk-opt --allow-unregistered-dialect \
// RUN:   --triton-to-structured --canonicalize --triton-cpu-verify \
// RUN:   --triton-block-ptr-to-vector --triton-shared-mem-to-scratch \
// RUN:   --triton-atomic-to-llrt --triton-grid-to-forall \
// RUN:   --triton-cpu-verify=reject-residual=true %s \
// RUN:   | FileCheck %s

// Experimental generic-op pipeline: compute, block pointers and void grids.
// Unsupported scratch/atomic programs live in expected-diagnostic fixtures.
// The final verifier rejects every residual tt.* operation. This fixture
// demonstrates native pass integration, not Python or upstream Triton execution.

// ----- Stage 1: tt.dot -> linalg.matmul
// CHECK-LABEL: func @stage1_dot
func.func @stage1_dot(%a: tensor<32x64xbf16>, %b: tensor<64x32xbf16>,
                      %c: tensor<32x32xf32>) -> tensor<32x32xf32> {
  %dot = "tt.dot"(%a, %b, %c)
    : (tensor<32x64xbf16>, tensor<64x32xbf16>, tensor<32x32xf32>) -> tensor<32x32xf32>
  // CHECK: linalg.matmul
  // CHECK-NOT: tt.dot
  func.return %dot : tensor<32x32xf32>
}

// ----- Stage 2: block pointers -> vector.transfer_read/write
// CHECK-LABEL: func @stage2_block_ptr
func.func @stage2_block_ptr(%base: memref<128x64xbf16>, %off_m: index, %off_k: index) {
  %ptr = "tt.make_block_ptr"(%base, %off_m, %off_k) {
    shape = array<i64: 128, 64>,
    strides = array<i64: 64, 1>,
    block_shape = array<i64: 32, 64>,
    order = array<i64: 0, 1>
  } : (memref<128x64xbf16>, index, index) -> !tt.ptr
  %v = "tt.load"(%ptr) : (!tt.ptr) -> vector<32x64xbf16>
  "tt.store"(%ptr, %v) : (!tt.ptr, vector<32x64xbf16>) -> ()
  // CHECK: memref.subview
  // CHECK: vector.transfer_read
  // CHECK: vector.transfer_write
  // CHECK-NOT: tt.make_block_ptr
  // CHECK-NOT: tt.load
  // CHECK-NOT: tt.store
  func.return
}

// Unsupported scratch/atomic programs have separate diagnostic fixtures.

// ----- Stage 5: grid dispatch -> scf.forall
// CHECK-LABEL: func @stage5_grid
func.func @stage5_grid(%M: index, %BM: index) {
  %pid = "tt.get_program_id"() {axis = 0 : i32} : () -> index
  %c2 = arith.constant 2 : index
  %result = arith.muli %pid, %c2 : index
  // CHECK: scf.forall
  // CHECK-NOT: tt.get_program_id
  func.return
}
