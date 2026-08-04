//===- triton_swiglu_staged.cpp - Triton IR → LLK staged pipeline ---------===//
//
// Integration test that builds Triton IR for SwiGLU programmatically, runs it
// through the staged lowering pipeline (TritonToStructured → GridToForall →
// existing LLK pipeline), JIT-executes, and compares against a reference.
//
// SKELETON: For M8b, this starts as a manual IR construction test and evolves
// into a full pipeline test once all stages are integrated.
//
//===----------------------------------------------------------------------===//

#include <gtest/gtest.h>

TEST(TritonStagedExecution, DISABLED_SwiGLUCorrectness) {
  // SKELETON: Full staged pipeline correctness test for Triton IR -> LLK.
  //
  // Prerequisites for implementation:
  //   1. Triton dialect ops registered in MLIRContext (tt.load, tt.store,
  //      tt.make_block_ptr, tt.dot, tt.advance, tt.arange).
  //   2. TritonToStructured pass lowering tt.dot -> linalg.matmul,
  //      tt.load/store -> vector.transfer_read/write.
  //   3. GridToForall pass expanding tt.grid -> scf.forall with
  //      program_id-based offsets.
  //   4. BlockPointerToVector pass (exists) lowering block_ptr ->
  //      memref.subview + transfer ops.
  //   5. Integration with existing LLK pipeline (LLKToLinalg ->
  //      TileAndVectorize -> OneShotBufferize -> JIT).
  //
  // Reference shapes: M=32, N=64, K=128 (medium batch, AVX2-friendly).
  //
  // Once all prerequisites are met, the test should:
  //   a. Build Triton IR programmatically for SwiGLU.
  //   b. Run the staged pipeline end-to-end.
  //   c. JIT execute and compare against FP64 reference (same approach as
  //      swiglu_vector_avx2.cpp).
  GTEST_SKIP() << "Triton staged pipeline awaiting dialect ops (M8b "
                  "prerequisites)";
}
