//===- JitCache.h - JIT cache for compiled LLK kernels -------------------===//
//
// 4-level JIT cache with LRU eviction per level:
//   L1: IR Cache        — verified MLIR module (pre-lowering)
//   L2: Optimized Cache  — post-optimization MLIR (pre-bufferization)
//   L3: Object Cache     — compiled function pointer (KernelKey → fn ptr)
//   L4: Weight Cache     — packed weight buffer (WeightKey → PackedWeights)
//
//===----------------------------------------------------------------------===//

#ifndef LLK_RUNTIME_JITCACHE_H
#define LLK_RUNTIME_JITCACHE_H

#include "LLK/Runtime/KernelKey.h"

#include "mlir/IR/BuiltinOps.h"
#include "llvm/ExecutionEngine/Orc/LLJIT.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <list>
#include <memory>
#include <optional>
#include <shared_mutex>
#include <string>
#include <unordered_map>

/// Stable C ABI: 2D tensor descriptor used to construct kernel inputs/outputs.
/// Matches the memory layout expected by the ORC JIT entry point.
struct Tensor2D {
  void *data;
  int64_t dim0;
  int64_t dim1;
  int64_t stride0;
  int64_t stride1;
};

/// MLIR-compatible 2D memref descriptor used as the JIT kernel calling ABI.
///
/// Layout must match the LLVM struct produced by MLIR's LLVM lowering of
/// memref<MxNxT>: {T*, T*, i64, [2 x i64], [2 x i64]}.
/// sizeof(MemRef2D) == 56 bytes on 64-bit systems.
struct MemRef2D {
  void *allocated;
  void *aligned;
  int64_t offset;
  int64_t size0;
  int64_t size1;
  int64_t stride0;
  int64_t stride1;
};

/// Helper: populate a MemRef2D from a Tensor2D that holds row-major data.
inline MemRef2D makeMemRef2D(const Tensor2D &t) {
  return MemRef2D{t.data, t.data, 0, t.dim0, t.dim1, t.stride0, t.stride1};
}

/// Stable C ABI: scratch and context pointer for compiled kernels.
struct KernelContext {
  void *scratch;
  size_t scratch_size;
};

namespace llk {

/// Aggregate cache statistics across all 4 levels.
struct CacheStats {
  size_t hits[4]{};
  size_t misses[4]{};
  size_t evictions[4]{};
};

/// Thread-safe 4-level JIT cache backed by ORC LLJIT.
///
/// Each level is independently locked with a shared_mutex and uses
/// LRU eviction when the entry count exceeds the configured max.
class JitCache {
public:
  /// Five MemRef2D arguments: x, wg, wu, init, result.
  using KernelFn = void (*)(MemRef2D *, MemRef2D *, MemRef2D *, MemRef2D *,
                            MemRef2D *);

  /// Constructs the JIT cache, creating an ORC LLJIT instance.
  explicit JitCache();
  ~JitCache();

  JitCache(const JitCache &) = delete;
  JitCache &operator=(const JitCache &) = delete;
  JitCache(JitCache &&) = delete;
  JitCache &operator=(JitCache &&) = delete;

  // -----------------------------------------------------------------------
  // Legacy API (M1-M4): string-keyed single-level cache for lookupOrCompile
  // -----------------------------------------------------------------------

  llvm::Expected<KernelFn> lookupOrCompile(const std::string &cache_key,
                                           mlir::ModuleOp module);

  // -----------------------------------------------------------------------
  // L3: Object-code cache (KernelKey → function pointer)
  // -----------------------------------------------------------------------

  /// Look up a previously compiled kernel by its KernelKey.
  std::optional<KernelFn> lookupObjectCode(const KernelKey &key);

  /// Insert a compiled kernel into the object-code cache.
  void insertObjectCode(const KernelKey &key, KernelFn fn);

  // -----------------------------------------------------------------------
  // L2: Optimized-MLIR cache (string key → serialized MLIR module)
  // -----------------------------------------------------------------------

  /// Look up a previously optimized MLIR module (serialized form).
  std::optional<std::string> lookupOptimizedMLIR(const std::string &key);

  /// Store an optimized MLIR module in the cache.
  void insertOptimizedMLIR(const std::string &key, std::string moduleStr);

  // -----------------------------------------------------------------------
  // L1: IR cache (string key → serialized MLIR module)
  // -----------------------------------------------------------------------

  std::optional<std::string> lookupIR(const std::string &key);
  void insertIR(const std::string &key, std::string moduleStr);

  // -----------------------------------------------------------------------
  // L4: Packed-weight cache (WeightKey → serialized packed weights blob)
  // -----------------------------------------------------------------------

  /// Look up packed weights. Returns serialized blob on hit.
  std::optional<std::string> lookupPackedWeights(const WeightKey &key);

  /// Store packed weights. The blob is the serialized PackedWeights content.
  void insertPackedWeights(const WeightKey &key, std::string packedBlob);

  // -----------------------------------------------------------------------
  // Cache management
  // -----------------------------------------------------------------------

  /// Evict LRU entries from a level until ≤ max_entries.
  /// Level is 1-indexed (1=L1, 2=L2, 3=L3, 4=L4).
  void evictLRU(size_t level, size_t max_entries);

  /// Configure max entries for a level.
  void setMaxEntries(size_t level, size_t max_entries);

  /// Return aggregate cache statistics.
  CacheStats getStats() const;

  void clear();
  size_t size() const;

private:
  /// Per-level LRU cache.
  ///
  /// Uses a doubly-linked list for LRU tracking (front = MRU, back = LRU)
  /// and an unordered_map for O(1) lookup. Read-write lock per level.
  struct CacheLevel {
    struct Entry {
      std::string key;
      std::string value;
    };
    using ListIter = std::list<Entry>::iterator;

    std::unordered_map<std::string, ListIter> index;
    std::list<Entry> lru_list;
    size_t max_entries{1024};
    mutable std::shared_mutex mutex;
    size_t hit_count{0};
    size_t miss_count{0};
    size_t eviction_count{0};

    /// Look up key; returns pointer to value on hit, nullptr on miss.
    /// Moves entry to MRU front on hit.
    std::string *lookup(const std::string &key);

    /// Insert a key-value pair. Evicts LRU if over capacity.
    void insert(const std::string &key, std::string value);

    /// Remove a specific key if present.
    void remove(const std::string &key);

    void clear();
    size_t size() const { return index.size(); }
  };

  /// Convert a KernelKey to a string key for L3 cache storage.
  static std::string kernelKeyToString(const KernelKey &key);

  /// Convert a WeightKey to a string key for L4 cache storage.
  static std::string weightKeyToString(const WeightKey &key);

  std::unique_ptr<llvm::orc::LLJIT> jit_;

  // Legacy string-keyed cache (used by lookupOrCompile for backward compat).
  std::unordered_map<std::string, KernelFn> cache_;
  mutable std::shared_mutex mutex_;

  // 4 cache levels (all store serialized strings)
  CacheLevel ir_cache_;        // L1
  CacheLevel optimized_cache_; // L2
  CacheLevel object_cache_;    // L3
  CacheLevel weight_cache_;    // L4
};

} // namespace llk

#endif // LLK_RUNTIME_JITCACHE_H
