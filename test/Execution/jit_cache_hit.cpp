//===- jit_cache_hit.cpp - JIT cache hit/miss and eviction tests ---------===//
//
// Tests for the 4-level JIT cache: KernelKey hashing, cache hit/miss/eviction.
//
//===----------------------------------------------------------------------===//

#include "LLK/Runtime/JitCache.h"
#include "LLK/Runtime/KernelKey.h"

#include <gtest/gtest.h>

#include <cstring>

using namespace llk;

// ---------------------------------------------------------------------------
// KernelKey tests
// ---------------------------------------------------------------------------

TEST(KernelKeyTest, SameKeyHashesEqual) {
  KernelKey a;
  a.operation = OperationKind::FusedSwiGLU;
  a.input_type = DType::BF16;
  a.output_type = DType::BF16;
  a.M_bucket = 0;
  a.N = 4096;
  a.K = 4096;
  a.weight_layout = Layout::PackedKN;
  a.isa = CpuIsa::AVX2;
  a.math_mode = MathMode::BoundedFast;
  a.thread_count = 8;
  a.compiler_version = 1;
  a.schedule_version = 1;

  KernelKey b = a;

  EXPECT_EQ(a.hash(), b.hash());
  EXPECT_EQ(a, b);
}

TEST(KernelKeyTest, DifferentBucketHashesDiffer) {
  KernelKey a, b;
  a.M_bucket = 0;
  b.M_bucket = 4;
  EXPECT_NE(a.hash(), b.hash());
  EXPECT_NE(a, b);
}

TEST(KernelKeyTest, DifferentNHashesDiffer) {
  KernelKey a, b;
  a.N = 4096;
  b.N = 8192;
  EXPECT_NE(a.hash(), b.hash());
}

TEST(KernelKeyTest, DifferentScheduleVersionHashesDiffer) {
  KernelKey a, b;
  a.schedule_version = 1;
  b.schedule_version = 2;
  EXPECT_NE(a.hash(), b.hash());
}

TEST(KernelKeyTest, ToStringIsNonEmpty) {
  KernelKey k;
  k.M_bucket = 2;
  k.N = 4096;
  k.K = 4096;
  std::string s = k.toString();
  EXPECT_FALSE(s.empty());
  EXPECT_NE(s.find("M_bucket=2"), std::string::npos);
}

// ---------------------------------------------------------------------------
// Cache level (L3 object-code) hit/miss tests
// ---------------------------------------------------------------------------

TEST(JitCacheTest, ObjectCodeCacheMissReturnsNullopt) {
  JitCache cache;
  KernelKey key;
  key.M_bucket = 0;
  key.N = 4096;
  key.K = 4096;

  auto result = cache.lookupObjectCode(key);
  EXPECT_FALSE(result.has_value());
}

TEST(JitCacheTest, ObjectCodeCacheHitReturnsInsertedPointer) {
  JitCache cache;

  // Use a known sentinel pointer as our "function."
  JitCache::KernelFn sentinel =
      reinterpret_cast<JitCache::KernelFn>(static_cast<uintptr_t>(0xDEADBEEF));

  KernelKey key;
  key.M_bucket = 0;
  key.N = 4096;
  key.K = 4096;

  cache.insertObjectCode(key, sentinel);

  auto result = cache.lookupObjectCode(key);
  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(*result, sentinel);

  // Second lookup should also hit.
  auto result2 = cache.lookupObjectCode(key);
  ASSERT_TRUE(result2.has_value());
  EXPECT_EQ(*result2, sentinel);
}

TEST(JitCacheTest, DifferentKernelKeysAreSeparate) {
  JitCache cache;

  JitCache::KernelFn fnA =
      reinterpret_cast<JitCache::KernelFn>(static_cast<uintptr_t>(0xAAA));
  JitCache::KernelFn fnB =
      reinterpret_cast<JitCache::KernelFn>(static_cast<uintptr_t>(0xBBB));

  KernelKey keyA;
  keyA.M_bucket = 0;
  keyA.N = 4096;
  keyA.K = 4096;

  KernelKey keyB;
  keyB.M_bucket = 4;
  keyB.N = 4096;
  keyB.K = 4096;

  cache.insertObjectCode(keyA, fnA);
  cache.insertObjectCode(keyB, fnB);

  auto resultA = cache.lookupObjectCode(keyA);
  auto resultB = cache.lookupObjectCode(keyB);

  ASSERT_TRUE(resultA.has_value());
  ASSERT_TRUE(resultB.has_value());
  EXPECT_EQ(*resultA, fnA);
  EXPECT_EQ(*resultB, fnB);
}

// ---------------------------------------------------------------------------
// LRU eviction tests
// ---------------------------------------------------------------------------

TEST(JitCacheTest, LRUEvictionRemovesLeastRecentlyUsed) {
  JitCache cache;

  // Set max entries for L3 to 3.
  cache.setMaxEntries(3, 3);

  KernelKey k1, k2, k3, k4;
  k1.M_bucket = 0;
  k2.M_bucket = 1;
  k3.M_bucket = 2;
  k4.M_bucket = 3;

  auto makeFn = [](int v) {
    return reinterpret_cast<JitCache::KernelFn>(static_cast<uintptr_t>(v));
  };

  cache.insertObjectCode(k1, makeFn(1));
  cache.insertObjectCode(k2, makeFn(2));
  cache.insertObjectCode(k3, makeFn(3));

  // Access k1 — moves it to MRU front.
  EXPECT_TRUE(cache.lookupObjectCode(k1).has_value());

  // Insert k4 — should evict k2 (the LRU, since k1 was just accessed).
  cache.insertObjectCode(k4, makeFn(4));

  EXPECT_TRUE(cache.lookupObjectCode(k1).has_value());  // still MRU
  EXPECT_FALSE(cache.lookupObjectCode(k2).has_value()); // evicted
  EXPECT_TRUE(cache.lookupObjectCode(k3).has_value());  // still present
  EXPECT_TRUE(cache.lookupObjectCode(k4).has_value());  // just inserted
}

// ---------------------------------------------------------------------------
// Cache statistics tests
// ---------------------------------------------------------------------------

TEST(JitCacheTest, StatsCountsHitsAndMisses) {
  JitCache cache;

  KernelKey key;
  key.M_bucket = 0;
  key.N = 4096;
  key.K = 4096;

  // Initial state: 0 hits, 0 misses for all levels.
  CacheStats s0 = cache.getStats();
  EXPECT_EQ(s0.hits[2], 0u);
  EXPECT_EQ(s0.misses[2], 0u);

  // Miss.
  cache.lookupObjectCode(key);
  CacheStats s1 = cache.getStats();
  EXPECT_EQ(s1.misses[2], 1u);

  // Insert + hit.
  JitCache::KernelFn fn =
      reinterpret_cast<JitCache::KernelFn>(static_cast<uintptr_t>(0x1));
  cache.insertObjectCode(key, fn);
  cache.lookupObjectCode(key);
  CacheStats s2 = cache.getStats();
  EXPECT_EQ(s2.hits[2], 1u);
  EXPECT_EQ(s2.misses[2], 1u); // still 1 (the first miss)
}

// ---------------------------------------------------------------------------
// Legacy API compatibility
// ---------------------------------------------------------------------------

TEST(JitCacheTest, ClearRemovesAllEntriesFromAllLevels) {
  JitCache cache;

  KernelKey key;
  key.M_bucket = 0;
  key.N = 4096;
  key.K = 4096;
  JitCache::KernelFn fn =
      reinterpret_cast<JitCache::KernelFn>(static_cast<uintptr_t>(0x1));
  cache.insertObjectCode(key, fn);
  cache.insertOptimizedMLIR("test", "some module string");
  cache.insertIR("ir_key", "parsed IR");

  EXPECT_GT(cache.size(), 0u);

  cache.clear();
  EXPECT_EQ(cache.size(), 0u);

  EXPECT_FALSE(cache.lookupObjectCode(key).has_value());
  EXPECT_FALSE(cache.lookupOptimizedMLIR("test").has_value());
  EXPECT_FALSE(cache.lookupIR("ir_key").has_value());
}

// ---------------------------------------------------------------------------
// WeightKey tests
// ---------------------------------------------------------------------------

TEST(WeightKeyTest, SameKeyHashesEqual) {
  int dummy;
  WeightKey a;
  a.weight_ptr = &dummy;
  a.shape_hash = 0xABCD;
  a.BK = 64;
  a.BN = 64;

  WeightKey b = a;

  std::hash<WeightKey> hasher;
  EXPECT_EQ(hasher(a), hasher(b));
  EXPECT_EQ(a, b);
}

TEST(WeightKeyTest, DifferentPointersHashDiffer) {
  int x, y;
  WeightKey a, b;
  a.weight_ptr = &x;
  b.weight_ptr = &y;
  std::hash<WeightKey> hasher;
  EXPECT_NE(hasher(a), hasher(b));
}
