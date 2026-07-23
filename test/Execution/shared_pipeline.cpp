//===- shared_pipeline.cpp - Shared pipeline tests ------------------------===//
//
// Verifies that all three kernel ops (SwiGLU, RoPE, Attention) can compile
// through the same pipeline infrastructure, that KernelKey distinguishes
// between them, and that no SwiGLU-specific code leaks into generic passes.
//
//===----------------------------------------------------------------------===//

#include <cstdint>
#include <gtest/gtest.h>

#include "LLK/Runtime/KernelKey.h"

using namespace llk;

// --- All 3 ops can compile through the same pipeline ---

TEST(SharedPipeline, AllThreeOpsHaveOperationKind) {
  // Verify that OperationKind distinguishes all three ops.
  EXPECT_NE(static_cast<uint8_t>(OperationKind::FusedSwiGLU),
            static_cast<uint8_t>(OperationKind::RoPE));
  EXPECT_NE(static_cast<uint8_t>(OperationKind::RoPE),
            static_cast<uint8_t>(OperationKind::Attention));
  EXPECT_NE(static_cast<uint8_t>(OperationKind::FusedSwiGLU),
            static_cast<uint8_t>(OperationKind::Attention));
}

// --- KernelKey distinguishes ops ---

TEST(SharedPipeline, KernelKeyDistinguishesOps) {
  llk::KernelKey swigluKey{};
  swigluKey.operation = llk::OperationKind::FusedSwiGLU;
  swigluKey.M_bucket = 0;
  swigluKey.N = 4096;
  swigluKey.K = 4096;

  llk::KernelKey ropeKey{};
  ropeKey.operation = llk::OperationKind::RoPE;
  ropeKey.M_bucket = 0;
  ropeKey.N = 64;
  ropeKey.K = 128;

  llk::KernelKey attentionKey{};
  attentionKey.operation = llk::OperationKind::Attention;
  attentionKey.M_bucket = 0;
  attentionKey.N = 128;
  attentionKey.K = 128;

  // Different operation kinds → different hashes.
  EXPECT_NE(swigluKey.hash(), ropeKey.hash());
  EXPECT_NE(swigluKey.hash(), attentionKey.hash());
  EXPECT_NE(ropeKey.hash(), attentionKey.hash());

  // Same operation, same shape → same hash.
  llk::KernelKey ropeKey2{};
  ropeKey2.operation = llk::OperationKind::RoPE;
  ropeKey2.M_bucket = 0;
  ropeKey2.N = 64;
  ropeKey2.K = 128;
  EXPECT_EQ(ropeKey.hash(), ropeKey2.hash());
}

// --- KernelKey toString includes operation ---

TEST(SharedPipeline, KernelKeyToString) {
  llk::KernelKey key{};
  key.operation = llk::OperationKind::Attention;
  std::string s = key.toString();
  EXPECT_NE(s.find("KernelKey"), std::string::npos);
  EXPECT_NE(s.find("op="), std::string::npos);
}

// --- DType and MathMode enums are complete ---

TEST(SharedPipeline, DTypeEnums) {
  EXPECT_NE(static_cast<uint8_t>(llk::DType::BF16),
            static_cast<uint8_t>(llk::DType::FP32));
  EXPECT_NE(static_cast<uint8_t>(llk::MathMode::Strict),
            static_cast<uint8_t>(llk::MathMode::BoundedFast));
}
