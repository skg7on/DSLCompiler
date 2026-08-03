//===- triton_swiglu_staged.cpp - Triton IR → LLK staged pipeline ---------===//
//
// Integration test that builds Triton IR for SwiGLU programmatically, runs it
// through the staged lowering pipeline (TritonToStructured → GridToForall →
// existing LLK pipeline), JIT-executes, and compares against a reference.
//
// For M8b, this starts as a manual IR construction test and evolves into a
// full pipeline test once all stages are integrated.
//
//===----------------------------------------------------------------------===//

#include <gtest/gtest.h>

#include <cmath>
#include <random>
#include <vector>

TEST(TritonStagedExecution, DISABLED_SwiGLUCorrectness) {
  // Placeholder — will be implemented once the full staged pipeline works
  GTEST_SKIP() << "M8b staged pipeline integration in progress";
}
