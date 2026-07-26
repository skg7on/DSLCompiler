//===- cache_eviction.cpp - JIT cache LRU eviction tests -----------------===//
//
// Verifies multi-level cache behavior: hits, misses, evictions,
// and LRU ordering.
//
//===----------------------------------------------------------------------===//

#include "LLK/Runtime/JitCache.h"
#include "LLK/Runtime/KernelKey.h"

#include <gtest/gtest.h>
#include <string>

using namespace llk;

TEST(JitCache, L1_IR_Cache_HitAndMiss) {
  JitCache cache;

  // L1: Insert and lookup IR.
  cache.insertIR("key1", "ir_module_string_1");
  cache.insertIR("key2", "ir_module_string_2");

  auto val1 = cache.lookupIR("key1");
  auto val2 = cache.lookupIR("key2");
  auto val3 = cache.lookupIR("key3"); // miss

  EXPECT_TRUE(val1.has_value());
  EXPECT_EQ(*val1, "ir_module_string_1");
  EXPECT_TRUE(val2.has_value());
  EXPECT_EQ(*val2, "ir_module_string_2");
  EXPECT_FALSE(val3.has_value());
}

TEST(JitCache, LRU_Eviction_Beyond_Capacity) {
  JitCache cache;

  // Set max entries to 2 for L1.
  cache.setMaxEntries(1, 2);

  // Insert 3 entries — the first should be evicted.
  cache.insertIR("a", "value_a");
  cache.insertIR("b", "value_b");
  cache.insertIR("c", "value_c"); // triggers LRU eviction of "a"

  EXPECT_FALSE(cache.lookupIR("a").has_value()) << "key 'a' should be evicted";
  EXPECT_TRUE(cache.lookupIR("b").has_value());
  EXPECT_TRUE(cache.lookupIR("c").has_value());
}

TEST(JitCache, LRU_Access_Updates_Recency) {
  JitCache cache;
  cache.setMaxEntries(1, 2);

  cache.insertIR("a", "value_a");
  cache.insertIR("b", "value_b");

  // Access "a" — moves to MRU front.
  cache.lookupIR("a");

  // Insert "c" — should evict "b" (LRU), not "a".
  cache.insertIR("c", "value_c");

  EXPECT_TRUE(cache.lookupIR("a").has_value()) << "key 'a' should survive";
  EXPECT_FALSE(cache.lookupIR("b").has_value()) << "key 'b' should be evicted";
  EXPECT_TRUE(cache.lookupIR("c").has_value());
}

TEST(JitCache, L2_OptimizedMLIR_Cache) {
  JitCache cache;

  cache.insertOptimizedMLIR("opt_key", "optimized_mlir_module");
  auto val = cache.lookupOptimizedMLIR("opt_key");

  EXPECT_TRUE(val.has_value());
  EXPECT_EQ(*val, "optimized_mlir_module");
}

TEST(JitCache, L4_PackedWeight_Cache) {
  JitCache cache;

  WeightKey key{};
  key.weight_ptr = reinterpret_cast<const void *>(0xDEADBEEF);
  key.shape_hash = 12345;
  key.BK = 64;
  key.BN = 64;

  cache.insertPackedWeights(key, "packed_blob_data");
  auto val = cache.lookupPackedWeights(key);

  EXPECT_TRUE(val.has_value());
  EXPECT_EQ(*val, "packed_blob_data");
}

TEST(JitCache, CacheStats_Accumulate) {
  JitCache cache;

  // L1: miss then hit.
  cache.lookupIR("nonexistent");       // miss
  cache.insertIR("stats_key", "data"); // (no stat change)
  cache.lookupIR("stats_key");         // hit

  auto stats = cache.getStats();
  EXPECT_EQ(stats.hits[0], 1);   // L1 hit
  EXPECT_EQ(stats.misses[0], 1); // L1 miss
}

TEST(JitCache, Clear_Removes_All_Entries) {
  JitCache cache;

  cache.insertIR("k1", "v1");
  cache.insertOptimizedMLIR("k2", "v2");

  cache.clear();

  EXPECT_FALSE(cache.lookupIR("k1").has_value());
  EXPECT_FALSE(cache.lookupOptimizedMLIR("k2").has_value());
  EXPECT_EQ(cache.size(), 0);
}
