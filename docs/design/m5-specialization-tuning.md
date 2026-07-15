# Milestone 5: Specialization & Tuning

**Parent:** [ARCHITECTURE.md](../../ARCHITECTURE.md) — See §4 Implementation Sequence
**Previous:** [Milestone 4: Parallel Execution](m4-parallel-execution.md)
**Next:** [Milestone 6: Multi-Kernel](m6-multi-kernel.md)
**Status:** Design Approved | **Level:** Algorithm & Data Structure

---

## 1. Objectives & Exit Criterion

Move from a single hardcoded schedule to a shape-bucketed specialization system with a multi-level JIT cache and deterministic schedule selection. Add an autotuning harness.

**Exit criterion:** Given any supported shape (M,N,K), the compiler deterministically selects a schedule from `schedule_db.json`. Changing the schedule version produces reproducible, measurable performance differences. The JIT cache correctly avoids recompilation for repeated (shape, schedule) pairs. The autotuning harness runs a grid search and produces persistent schedule records.

---

## 2. Shape Bucketing

```
M buckets:
  Bucket 0: M = 1           → GEMV-like, parallelize across N blocks
  Bucket 1: M ∈ [2, 4]      → Micro-GEMM, limited threads (2-4)
  Bucket 2: M ∈ [5, 16]     → Small tile, modest parallelism
  Bucket 3: M ∈ [17, 64]    → Medium tile, full parallelism
  Bucket 4: M ≥ 65          → Large tile, conventional GEMM parallelism

N, K: exact specialization (determined by model architecture)
  Common values: 1024, 2048, 4096, 8192, 11008, 14336
```

**Note on M4 integration:** When the schedule database specifies `num_threads` and `grain_size` for a bucket, those values override the static thresholds in the SerialParallelDispatch pass (M4). E.g., M ∈ [2,4] with `num_threads: 4` in the schedule will use the thread pool despite M4's hardcoded `SERIAL_MAX_M = 3`.

---

## 3. Components to Build

```
New/modified files:
  include/LLK/Transforms/ShapeSpecialization.h       # Bucket inference, dispatch gen
  lib/Transforms/ShapeSpecialization.cpp              # M bucket classifier, guard emission
  include/LLK/Runtime/JitCache.h                      # UPGRADE: single-level → 4-level
  runtime/JitCache.cpp                                # LRU eviction per level
  include/LLK/Runtime/KernelKey.h                     # KernelKey struct + SHA-256 hash
  include/LLK/Transforms/ScheduleSelection.h          # Deterministic schedule lookup
  lib/Transforms/ScheduleSelection.cpp                # Schedule DB query + fallback
  tools/llk-bench/llk-bench.cpp                       # Benchmarking harness
  tools/llk-tune/llk-tune.cpp                         # Autotuning grid search driver
  schedules/schedule_db.json                          # Schedule database
  test/Transforms/shape_specialization.mlir            # FileCheck: dispatch tree
  test/Transforms/schedule_selection.mlir              # FileCheck: correct schedule chosen
  test/Transforms/schedule_fallback.mlir               # FileCheck: fallback behavior
  test/Execution/jit_cache_hit.cpp                     # Cache hit verification
  test/Execution/jit_cache_miss.cpp                    # Cache miss → recompile
  test/Execution/cache_eviction.cpp                    # LRU eviction
  test/Execution/specialization_dispatch.cpp           # Correctness across all 5 buckets
  test/Execution/schedule_determinism.cpp              # Same inputs → same binary
  benchmark/schedule_regression.cpp                    # Performance regression database
```

---

## 4. ShapeSpecialization Pass

```
shapeSpecialize(func, M, N, K):
    bucket = classifyM(M)

    generate dispatch tree with guards:
        @variant_M1 when (M==1 and N==N and K==K)
        @variant_M2_4 when (M>=2 and M<=4 and N==N and K==K)
        @variant_M5_16 when (M>=5 and M<=16 ...)
        @variant_M17_64 when (M>=17 and M<=64 ...)
        @variant_M65p when (M>=65 ...)
        @generic otherwise
```

**IR representation:**

```mlir
%y = llk.dispatch %x, %wg, %wu, %init
    variants {
        @swiglu_M1_N4096_K4096 when
            (%M == 1 and %N == 4096 and %K == 4096),
        @swiglu_M2_4_N4096_K4096 when
            (%M in [2,4] and %N == 4096 and %K == 4096),
        @swiglu_M5_16_N4096_K4096 when
            (%M in [5,16] and %N == 4096 and %K == 4096),
        @swiglu_M17_64_N4096_K4096 when
            (%M in [17,64] and %N == 4096 and %K == 4096),
        @swiglu_M65p_N4096_K4096 when
            (%M >= 65 and %N == 4096 and %K == 4096),
        @swiglu_generic otherwise
    } → tensor<?x?xbf16>
```

---

## 5. Multi-Level JIT Cache

### 5.1 Cache Levels

```
Level 1: IR Cache
  Key:   (kernel_source_hash, operation_kind)
  Value: Verified MLIR module (pre-lowering)
  Hit:   Skip parse + verify

Level 2: Optimized MLIR Cache
  Key:   (kernel_source_hash, M_bucket, N, K, dtypes, schedule_version)
  Value: Post-optimization MLIR module (pre-bufferization)
  Hit:   Skip LLKToLinalg through vectorize

Level 3: Object-Code Cache
  Key:   KernelKey (13-field struct)
  Value: Compiled function pointer
  Hit:   Skip entire compilation pipeline

Level 4: Packed-Weight Cache
  Key:   (weight_data_ptr, weight_shape_hash, BK, BN)
  Value: PackedWeights struct
  Hit:   Skip repacking
```

### 5.2 Cache Implementation

```cpp
class JitCache {
public:
    std::optional<KernelFn> lookupObjectCode(const KernelKey& key);
    void insertObjectCode(const KernelKey& key, KernelFn fn);

    std::optional<std::unique_ptr<mlir::ModuleOp>> lookupOptimizedMLIR(const std::string& key);
    void insertOptimizedMLIR(const std::string& key, std::unique_ptr<mlir::ModuleOp> mod);

    std::optional<PackedWeights> lookupPackedWeights(const WeightKey& key);
    void insertPackedWeights(const WeightKey& key, PackedWeights pw);

    void evictLRU(size_t level, size_t max_entries);
    struct Stats { size_t hits[4]; size_t misses[4]; size_t evictions[4]; };
    Stats getStats() const;

private:
    template<typename V>
    struct CacheLevel {
        std::unordered_map<std::string, V> entries;
        std::list<std::string> lru_list;            // Front = most-recently-used
        size_t max_entries{1024};
        mutable std::shared_mutex mutex;
    };
    CacheLevel<VerifiedModule> ir_cache_;            // L1
    CacheLevel<OwningModule>    optimized_cache_;     // L2
    CacheLevel<KernelFn>        object_cache_;        // L3
    CacheLevel<PackedWeights>   weight_cache_;        // L4
};
```

### 5.3 KernelKey

```cpp
struct KernelKey {
    OperationKind operation;     // FusedSwiGLU, RoPE, Attention
    DType input_type;            // BF16, FP16, FP32
    DType output_type;
    int64_t M_bucket;            // 0-4
    int64_t N;                   // exact
    int64_t K;                   // exact
    Layout weight_layout;        // RowMajor, PackedKN
    CpuIsa isa;                  // AVX2, AVX512_BF16, AVX512_VNNI, AMX_BF16, NEON, SVE
    MathMode math_mode;          // Strict, BoundedFast, UnsafeFast
    uint16_t thread_count;
    uint64_t compiler_version;   // git commit hash
    uint64_t schedule_version;   // schedule_db.json content hash

    uint64_t hash() const;       // SHA-256 truncated to 64 bits
    bool operator==(const KernelKey&) const = default;
};
```

---

## 6. Schedule Database

### 6.1 Schema

```json
{
  "version": 1,
  "entries": [
    {
      "operation": "fused_swiglu",
      "target": "x86-avx2",
      "shape": { "M_bucket": 0, "N": 4096, "K": 4096 },
      "dtype": "bf16",
      "math_mode": "bounded_fast",
      "schedule": {
        "BM": 1, "BN": 64, "BK": 64,
        "VM": 1, "VN": 8,
        "vector_width": 8,
        "num_threads": 8,
        "parallel_axis": "n",
        "grain_size": 1
      }
    },
    {
      "operation": "fused_swiglu",
      "target": "x86-avx2",
      "shape": { "M_bucket": 4, "N": 4096, "K": 4096 },
      "dtype": "bf16",
      "math_mode": "bounded_fast",
      "schedule": {
        "BM": 32, "BN": 64, "BK": 64,
        "VM": 4, "VN": 8,
        "vector_width": 8,
        "num_threads": 8,
        "parallel_axis": "m",
        "grain_size": 4
      }
    }
  ]
}
```

### 6.2 Schedule Selection with Fallback

```
select_schedule(op, M, N, K, dtype, ISA, math_mode):
    bucket = classifyM(M)

    // Try in order: exact match → relax K → relax N → bucket only → default
    entry = db.lookup(bucket, N, K)          // exact
    entry ??= db.lookup(bucket, N, *)        // closest K
    entry ??= db.lookup(bucket, *, K)        // closest N
    entry ??= db.lookup(bucket, *, *)        // bucket default
    entry ??= default_schedule(op, dtype, ISA)  // hardcoded conservative

    return entry
```

---

## 7. Autotuning Harness

### 7.1 Config Space

```cpp
struct TuningConfig {
    int64_t BM, BN, BK;
    int64_t VM, VN;
    int64_t vector_width;
    int64_t num_threads;
    int64_t grain_size;
    std::string parallel_axis;
};

std::vector<TuningConfig> generateConfigs(CpuIsa isa, int64_t M_bucket,
                                           int64_t N, int64_t K) {
    // Feasibility constraints
    size_t l1_size = 32 * 1024;   // 32KB L1
    size_t elem_size = 2;          // BF16

    for (BM : {1, 4, 8, 16, 32, 64}) {
        if (M_bucket == 0 && BM != 1) continue;
        for (BN : {16, 32, 64, 128, 256}) {
            for (BK : {32, 64, 128, 256}) {
                // L1 capacity check: X_tile + 2× weight tiles
                size_t tile_mem = (BM * BK + 2 * BK * BN) * elem_size;
                if (tile_mem > l1_size * 0.8) continue;

                for (num_threads : {1, 2, 4, 8, 16}) {
                    for (grain_size : {1, 2, 4, 8}) {
                        if (grain_size > total_tiles / num_threads) continue;
                        configs.push_back({...});
                    }
                }
            }
        }
    }
    return configs;
}
```

### 7.2 Measurement & Persistence

```
measure each config: warmup 500ms → measure 2000ms → collect gflops, cycles, cache misses
sort by gflops descending
save top-3 to schedule_db.json
```

**Python autotuning API (future, documented for M7+):**
```python
@llk.autotune(
    configs=[llk.Config(BLOCK_M=8, BLOCK_N=64, BLOCK_K=64, ...), ...],
    key=["M_BUCKET", "N", "K", "dtype", "isa"],
)
@llk.jit
def fused_swiglu_kernel(...): ...
```

---

## 8. Test Specifications

| Test | What it verifies | Tool |
|------|-----------------|------|
| `test/Transforms/shape_specialization.mlir` | M=1 → M1 variant; M=128 → M65p variant; dispatch guards correct | `llk-opt` + FileCheck |
| `test/Transforms/schedule_selection.mlir` | (M=1,N=4096,K=4096) → entry with BM=1,BN=64 selected | `llk-opt` + FileCheck |
| `test/Transforms/schedule_fallback.mlir` | Unknown (N,K) → falls back with warning; default schedule emitted | `llk-opt` + FileCheck |
| `test/Execution/jit_cache_hit.cpp` | Compile shape A twice → second call returns cached pointer (verify no recompilation via compile counter) | GTest |
| `test/Execution/jit_cache_miss.cpp` | Compile shape A then shape B (different M_bucket) → two cache entries | GTest |
| `test/Execution/cache_eviction.cpp` | Fill object cache beyond max → LRU eviction removes least-recently-used | GTest |
| `test/Execution/specialization_dispatch.cpp` | All 5 M-buckets × 3 (N,K) combos → correct results vs FP64 | GTest |
| `test/Execution/schedule_determinism.cpp` | Same (shape, schedule_version) → byte-level identical .text section | GTest + memcmp |
| `benchmark/schedule_regression.cpp` | Baseline gflops per M-bucket at N=K=4096; CI compares ±5% tolerance | Benchmark harness |

---

## 9. Dependencies

- [Milestone 4](m4-parallel-execution.md) — thread pool, parallel decomposition
- [Milestone 3](m3-fused-memory.md) — fused double-contraction, weight packing
- [Milestone 2](m2-explicit-vector.md) — tiling + vectorization
- [Milestone 1](m1-scalar-pipeline.md) — JIT + test harness
- nlohmann/json or similar JSON library

---

## 10. Related

- [ARCHITECTURE.md](../../ARCHITECTURE.md) — §2.3 Scheduling, §2.8 JIT & Caching, §2.9 Shape Specialization
- [m4-parallel-execution.md](m4-parallel-execution.md) — schedule-driven dispatch refines static thresholds
- [m6-multi-kernel.md](m6-multi-kernel.md) — specialization extends to RoPE + Attention
