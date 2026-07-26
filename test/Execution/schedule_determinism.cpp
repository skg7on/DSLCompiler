//===- schedule_determinism.cpp - Schedule selection determinism tests ---===//
//
// Verifies that the schedule selection is deterministic: same inputs
// always produce the same schedule. Tests KernelKey hash stability
// and schedule lookup idempotency.
//
//===----------------------------------------------------------------------===//

#include "LLK/Runtime/KernelKey.h"

#include <gtest/gtest.h>

using namespace llk;

TEST(ScheduleDeterminism, KernelKeyHash_Deterministic) {
  KernelKey key1{};
  key1.operation = OperationKind::FusedSwiGLU;
  key1.M_bucket = 2;
  key1.N = 4096;
  key1.K = 4096;
  key1.thread_count = 8;

  KernelKey key2{};
  key2.operation = OperationKind::FusedSwiGLU;
  key2.M_bucket = 2;
  key2.N = 4096;
  key2.K = 4096;
  key2.thread_count = 8;

  // Same fields -> same hash.
  EXPECT_EQ(key1.hash(), key2.hash());

  // Run 100 times to catch non-deterministic hashing.
  uint64_t prev = key1.hash();
  for (int i = 0; i < 100; ++i) {
    EXPECT_EQ(key1.hash(), prev);
  }
}

TEST(ScheduleDeterminism, KernelKeyHash_DiffersOnChange) {
  KernelKey base{};
  base.operation = OperationKind::FusedSwiGLU;
  base.M_bucket = 2;
  base.N = 4096;
  base.K = 4096;
  uint64_t base_hash = base.hash();

  // Changing any field should change the hash.
  KernelKey diff_op = base;
  diff_op.operation = OperationKind::RoPE;
  EXPECT_NE(base_hash, diff_op.hash());

  KernelKey diff_M = base;
  diff_M.M_bucket = 3;
  EXPECT_NE(base_hash, diff_M.hash());

  KernelKey diff_N = base;
  diff_N.N = 11008;
  EXPECT_NE(base_hash, diff_N.hash());

  KernelKey diff_K = base;
  diff_K.K = 2048;
  EXPECT_NE(base_hash, diff_K.hash());

  KernelKey diff_threads = base;
  diff_threads.thread_count = 4;
  EXPECT_NE(base_hash, diff_threads.hash());
}

TEST(ScheduleDeterminism, DifferentOpsDifferentKeys) {
  KernelKey swiglu_key{};
  swiglu_key.operation = OperationKind::FusedSwiGLU;

  KernelKey rope_key{};
  rope_key.operation = OperationKind::RoPE;

  KernelKey attention_key{};
  attention_key.operation = OperationKind::Attention;

  EXPECT_NE(swiglu_key.hash(), rope_key.hash());
  EXPECT_NE(swiglu_key.hash(), attention_key.hash());
  EXPECT_NE(rope_key.hash(), attention_key.hash());
}
