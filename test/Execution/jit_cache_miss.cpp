//===- jit_cache_miss.cpp - JIT cache miss behavior tests -----------------===//
//
// Verifies that different shapes produce different cache entries
// (cache miss), and same shapes reuse cached entries (cache hit).
//
//===----------------------------------------------------------------------===//

#include "LLK/Runtime/JitCache.h"
#include "LLK/Runtime/KernelKey.h"

#include <gtest/gtest.h>

using namespace llk;

TEST(JitCacheMiss, DifferentShapes_DifferentKeys) {
  // Two KernelKeys with different (M,N,K) must have different hashes.
  KernelKey key1{};
  key1.operation = OperationKind::FusedSwiGLU;
  key1.M_bucket = 0; // M=1
  key1.N = 4096;
  key1.K = 4096;

  KernelKey key2{};
  key2.operation = OperationKind::FusedSwiGLU;
  key2.M_bucket = 4; // M>=65
  key2.N = 4096;
  key2.K = 4096;

  EXPECT_NE(key1.hash(), key2.hash())
      << "Different M-buckets must produce different cache keys";
}

TEST(JitCacheMiss, DifferentOperations_DifferentKeys) {
  // SwiGLU, RoPE, and Attention must never collide.
  KernelKey swiglu{};
  swiglu.operation = OperationKind::FusedSwiGLU;
  swiglu.N = 4096;
  swiglu.K = 4096;

  KernelKey rope{};
  rope.operation = OperationKind::RoPE;
  rope.N = 64;
  rope.K = 0;

  KernelKey attention{};
  attention.operation = OperationKind::Attention;
  attention.N = 64;
  attention.K = 64;

  EXPECT_NE(swiglu.hash(), rope.hash());
  EXPECT_NE(swiglu.hash(), attention.hash());
  EXPECT_NE(rope.hash(), attention.hash());
}

TEST(JitCacheMiss, L1_CacheMiss_OnNewKey) {
  JitCache cache;

  // First lookup: miss.
  EXPECT_FALSE(cache.lookupIR("new_module_v1").has_value());

  // Insert and verify hit.
  cache.insertIR("new_module_v1", "module_content_v1");
  EXPECT_TRUE(cache.lookupIR("new_module_v1").has_value());

  // Different key: miss.
  EXPECT_FALSE(cache.lookupIR("new_module_v2").has_value());
}
