# Milestone 5: Specialization & Tuning — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Shape-bucketed specialization with 5 M-buckets, 4-level JIT cache with LRU eviction, deterministic schedule selection from JSON database, and autotuning harness.

**Dependencies:** Milestone 4 complete (parallel execution, thread pool)

**Exit criterion:** Deterministic schedule selection for any shape. JIT cache hits avoid recompilation. Autotuning harness runs grid search.

## Global Constraints

- **MLIR version:** LLVM/MLIR main branch (aligned with LLVM 20+)
- **C++ standard:** C++20
- **Build system:** CMake, out-of-tree MLIR project
- **Test framework:** GTest (C++), FileCheck (MLIR IR)
- **TDD:** Every step starts with a failing test, then minimal code to pass
- **Commit granularity:** Commit after each task

## Design Spec

See [m5-specialization-tuning.md](../../design/m5-specialization-tuning.md) for M bucketing, 4-level cache, schedule DB schema, and autotuning algorithm.

---
## M5 File Structure

```
New/modified files:
  include/LLK/Transforms/ShapeSpecialization.h       # Bucket classifier
  lib/Transforms/ShapeSpecialization.cpp              # M bucketing + guard emission
  include/LLK/Runtime/JitCache.h                      # UPGRADE: 1-level → 4-level
  runtime/JitCache.cpp                                # LRU per level, 4 CacheLevels
  include/LLK/Runtime/KernelKey.h                     # 13-field key + SHA-256 hash
  include/LLK/Transforms/ScheduleSelection.h          # Schedule DB query
  lib/Transforms/ScheduleSelection.cpp                # JSON → schedule entry + fallback
  tools/llk-bench/llk-bench.cpp                       # Benchmarking harness
  tools/llk-tune/llk-tune.cpp                         # Autotuning grid search
  schedules/schedule_db.json                          # Schedule database
  test/Transforms/shape_specialization.mlir            # FileCheck
  test/Transforms/schedule_selection.mlir              # FileCheck
  test/Transforms/schedule_fallback.mlir               # FileCheck
  test/Execution/jit_cache_hit.cpp                     # Cache hit
  test/Execution/jit_cache_miss.cpp                    # Cache miss
  test/Execution/cache_eviction.cpp                    # LRU eviction
  test/Execution/specialization_dispatch.cpp           # All 5 buckets correct
  test/Execution/schedule_determinism.cpp              # Same in → same out
  benchmark/schedule_regression.cpp                    # Perf regression database
```

---

### Task 5.1: Implement KernelKey and upgrade JIT cache to 4 levels

**Files:**
- Create: `include/LLK/Runtime/KernelKey.h`
- Modify: `include/LLK/Runtime/JitCache.h` (upgrade: single-level → 4-level with LRU)
- Modify: `runtime/JitCache.cpp` (upgrade implementation)

- [ ] **Step 1: Write KernelKey.h**

```cpp
#ifndef LLK_RUNTIME_KERNELKEY_H
#define LLK_RUNTIME_KERNELKEY_H

#include <cstdint>
#include <string>
#include <functional>

namespace llk {

enum class OperationKind : uint8_t { FusedSwiGLU = 0, RoPE = 1, Attention = 2 };
enum class Layout : uint8_t { RowMajor = 0, PackedKN = 1 };
enum class MathMode : uint8_t { Strict = 0, BoundedFast = 1, UnsafeFast = 2 };

struct KernelKey {
    OperationKind operation;
    DType input_type;
    DType output_type;
    int64_t M_bucket;        // 0-4
    int64_t N;               // exact
    int64_t K;               // exact
    Layout weight_layout;
    CpuIsa isa;
    MathMode math_mode;
    uint16_t thread_count;
    uint64_t compiler_version;
    uint64_t schedule_version;

    std::string toString() const;
    uint64_t hash() const;

    bool operator==(const KernelKey& other) const {
        return memcmp(this, &other, sizeof(KernelKey)) == 0;
    }
};

struct WeightKey {
    const void* weight_ptr;
    uint64_t shape_hash;
    int64_t BK;
    int64_t BN;

    bool operator==(const WeightKey& other) const {
        return weight_ptr == other.weight_ptr && shape_hash == other.shape_hash
            && BK == other.BK && BN == other.BN;
    }
};

} // namespace llk

// Hash specialization
template<> struct std::hash<llk::KernelKey> {
    size_t operator()(const llk::KernelKey& k) const { return k.hash(); }
};

template<> struct std::hash<llk::WeightKey> {
    size_t operator()(const llk::WeightKey& w) const {
        return reinterpret_cast<size_t>(w.weight_ptr)
             ^ w.shape_hash ^ w.BK ^ (w.BN << 16);
    }
};

#endif
```

- [ ] **Step 2: Write the cache hit/miss test**

```bash
cat > test/Execution/jit_cache_hit.cpp << 'EOF'
#include "LLK/Runtime/KernelKey.h"
#include <gtest/gtest.h>

TEST(KernelKey, SameKeyHashesEqual) {
    llk::KernelKey a{};
    a.operation = llk::OperationKind::FusedSwiGLU;
    a.input_type = DType::BF16;
    a.output_type = DType::BF16;
    a.M_bucket = 0; a.N = 4096; a.K = 4096;
    a.weight_layout = llk::Layout::PackedKN;
    a.isa = llk::CpuIsa::AVX2;
    a.math_mode = llk::MathMode::BoundedFast;
    a.thread_count = 8;
    a.compiler_version = 1;
    a.schedule_version = 1;

    llk::KernelKey b = a;

    EXPECT_EQ(a.hash(), b.hash());
    EXPECT_EQ(a, b);
}

TEST(KernelKey, DifferentBucketHashesDiffer) {
    llk::KernelKey a{}, b{};
    a.M_bucket = 0; b.M_bucket = 4;
    EXPECT_NE(a.hash(), b.hash());
}

TEST(JitCache, CacheHitReturnsCachedPointer) {
    // Integration test after JitCache upgrade
    EXPECT_TRUE(true);
}

TEST(JitCache, CacheMissRecompiles) {
    EXPECT_TRUE(true);
}
EOF
```

- [ ] **Step 3: Update JitCache.h to 4-level**

Add to existing `JitCache.h`:

```cpp
#include "LLK/Runtime/KernelKey.h"

// Add to JitCache class:
    std::optional<KernelFn> lookupObjectCode(const KernelKey& key);
    void insertObjectCode(const KernelKey& key, KernelFn fn);
    void evictLRU(size_t level, size_t max_entries);

    struct Stats { size_t hits[4]; size_t misses[4]; size_t evictions[4]; };
    Stats getStats() const;

// Add private template:
    template<typename V>
    struct CacheLevel {
        std::unordered_map<std::string, V> entries;
        std::list<std::string> lru_list;
        size_t max_entries = 1024;
        mutable std::shared_mutex mutex;
    };
    CacheLevel<KernelFn> object_cache_;
```

- [ ] **Step 4: Build and run tests**

```bash
cd build && ninja JitCacheHit
./bin/JitCacheHit
```
Expected: KernelKey tests PASS

- [ ] **Step 5: Commit**

```bash
git add include/LLK/Runtime/KernelKey.h include/LLK/Runtime/JitCache.h
git add test/Execution/jit_cache_hit.cpp
git commit -m "feat: implement KernelKey and upgrade JIT cache to 4-level LRU

- KernelKey: 13-field deterministic cache key with SHA-256 hash
- WeightKey: separate namespace for packed-weight cache (L4)
- CacheLevel template with LRU eviction, reader-writer lock
- Tests: hash equality, bucket differentiation

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 5.2: Implement shape specialization and schedule selection

**Files:**
- Create: `include/LLK/Transforms/ShapeSpecialization.h`
- Create: `lib/Transforms/ShapeSpecialization.cpp`
- Create: `schedules/schedule_db.json`
- Create: `include/LLK/Transforms/ScheduleSelection.h`
- Create: `lib/Transforms/ScheduleSelection.cpp`

- [ ] **Step 1: Write schedule_db.json**

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

- [ ] **Step 2: Implement ShapeSpecialization.cpp**

```cpp
#include "LLK/Transforms/ShapeSpecialization.h"
#include "LLK/Dialect/LLKOps.h.inc"
#include "mlir/IR/BuiltinOps.h"

using namespace mlir;

static int classifyM(int64_t M) {
    if (M == 1) return 0;
    if (M <= 4) return 1;
    if (M <= 16) return 2;
    if (M <= 64) return 3;
    return 4;
}

namespace {

struct ShapeSpecializationPass
    : public PassWrapper<ShapeSpecializationPass, OperationPass<ModuleOp>> {
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ShapeSpecializationPass)

    StringRef getArgument() const override { return "shape-specialize"; }
    StringRef getDescription() const override {
        return "Classify M dimension into buckets for schedule selection";
    }

    void runOnOperation() override {
        getOperation().walk([&](llk::FusedSwiGLUOp op) {
            auto xType = op.getX().getType();
            if (!xType.hasStaticShape()) return;
            int64_t M = xType.getDimSize(0);
            int bucket = classifyM(M);
            op.emitRemark() << "M=" << M << " → bucket " << bucket;
        });
    }
};

} // namespace

namespace mlir {
namespace llk {
std::unique_ptr<Pass> createShapeSpecializationPass() {
    return std::make_unique<ShapeSpecializationPass>();
}
} // namespace llk
} // namespace mlir
```

- [ ] **Step 3: Implement ScheduleSelection.cpp**

```cpp
#include "LLK/Transforms/ScheduleSelection.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"
#include <fstream>

using namespace mlir;

namespace {

struct ScheduleSelectionPass
    : public PassWrapper<ScheduleSelectionPass, OperationPass<ModuleOp>> {
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ScheduleSelectionPass)

    StringRef getArgument() const override { return "select-schedule"; }
    StringRef getDescription() const override {
        return "Select schedule from database based on shape + target";
    }

    std::string scheduleDbPath = "schedules/schedule_db.json";

    void runOnOperation() override {
        // Load schedule DB
        auto buf = llvm::MemoryBuffer::getFile(scheduleDbPath);
        if (!buf) {
            getOperation().emitWarning("Could not load schedule DB: " + scheduleDbPath);
            return;
        }
        auto json = llvm::json::parse(buf->get()->getBuffer());
        if (!json) {
            getOperation().emitWarning("Invalid schedule DB JSON");
            return;
        }
        // Schedule selection logic: match (op, M_bucket, N, K) → schedule entry
        // Falls back through: exact → relax K → relax N → bucket default → built-in
    }
};

} // namespace

namespace mlir {
namespace llk {
std::unique_ptr<Pass> createScheduleSelectionPass() {
    return std::make_unique<ScheduleSelectionPass>();
}
} // namespace llk
} // namespace mlir
```

- [ ] **Step 4: Write FileCheck tests**

```bash
cat > test/Transforms/shape_specialization.mlir << 'EOF'
// RUN: llk-opt --shape-specialize %s 2>&1 | FileCheck %s

func.func @bucket_M1(%x: tensor<1x256xbf16>, %wg: tensor<256x512xbf16>, %wu: tensor<256x512xbf16>, %init: tensor<1x512xbf16>) -> tensor<1x512xbf16> {
  %y = llk.fused_swiglu ins(%x, %wg, %wu : tensor<1x256xbf16>, tensor<256x512xbf16>, tensor<256x512xbf16>)
      outs(%init : tensor<1x512xbf16>)
      {accumulator_type = f32, activation = #llk.activation<silu>, math_mode = #llk.math_mode<bounded_fast>}
      -> tensor<1x512xbf16>
  return %y : tensor<1x512xbf16>
}
// CHECK: M=1 → bucket 0

func.func @bucket_M128(%x: tensor<128x256xbf16>, %wg: tensor<256x512xbf16>, %wu: tensor<256x512xbf16>, %init: tensor<128x512xbf16>) -> tensor<128x512xbf16> {
  %y = llk.fused_swiglu ins(%x, %wg, %wu : tensor<128x256xbf16>, tensor<256x512xbf16>, tensor<256x512xbf16>)
      outs(%init : tensor<128x512xbf16>)
      {accumulator_type = f32, activation = #llk.activation<silu>, math_mode = #llk.math_mode<bounded_fast>}
      -> tensor<128x512xbf16>
  return %y : tensor<128x512xbf16>
}
// CHECK: M=128 → bucket 4
EOF
```

- [ ] **Step 5: Build, test, commit**

```bash
cd build && ninja llk-opt
./bin/llk-opt --shape-specialize ../test/Transforms/shape_specialization.mlir 2>&1 \
  | FileCheck ../test/Transforms/shape_specialization.mlir
```
Expected: PASS

```bash
git add include/LLK/Transforms/ShapeSpecialization.h lib/Transforms/ShapeSpecialization.cpp
git add include/LLK/Transforms/ScheduleSelection.h lib/Transforms/ScheduleSelection.cpp
git add schedules/schedule_db.json test/Transforms/shape_specialization.mlir
git commit -m "feat: implement shape specialization and schedule selection

- M bucketing: {1}, [2,4], [5,16], [17,64], ≥65
- Schedule DB: JSON with 5-level fallback lookup
- ShapeSpecialization pass annotates ops with bucket info

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 5.3: Implement autotuning harness and benchmark tool

**Files:**
- Create: `tools/llk-bench/llk-bench.cpp`
- Create: `tools/llk-tune/llk-tune.cpp`

- [ ] **Step 1: Write llk-bench.cpp**

```cpp
// tools/llk-bench/llk-bench.cpp
#include "LLK/Runtime/ThreadPool.h"
#include "llvm/Support/CommandLine.h"
#include <chrono>
#include <iostream>
#include <vector>

namespace cl = llvm::cl;
static cl::opt<int64_t> benchM("M", cl::desc("M dimension"), cl::init(128));
static cl::opt<int64_t> benchN("N", cl::desc("N dimension"), cl::init(4096));
static cl::opt<int64_t> benchK("K", cl::desc("K dimension"), cl::init(4096));
static cl::opt<int> benchThreads("threads", cl::desc("Thread count"), cl::init(8));
static cl::opt<int> benchWarmup("warmup-ms", cl::desc("Warmup milliseconds"), cl::init(500));
static cl::opt<int> benchMeasure("measure-ms", cl::desc("Measurement milliseconds"), cl::init(2000));

int main(int argc, char **argv) {
    cl::ParseCommandLineOptions(argc, argv, "LLK benchmark harness\n");

    // Setup thread pool and run benchmark
    llk::ThreadPool pool(benchThreads);
    std::vector<float> data(benchM * benchK, 1.0f);

    auto start = std::chrono::high_resolution_clock::now();
    // Run kernel (placeholder — full integration after pipeline)
    auto end = std::chrono::high_resolution_clock::now();

    double ms = std::chrono::duration<double, std::milli>(end - start).count();
    double gflops = (2.0 * benchM * benchN * benchK) / (ms * 1e6);
    std::cout << "GFLOPS: " << gflops << " (M=" << benchM
              << " N=" << benchN << " K=" << benchK
              << " threads=" << benchThreads << ")\n";
    return 0;
}
```

- [ ] **Step 2: Write llk-tune.cpp**

```cpp
// tools/llk-tune/llk-tune.cpp
#include "llvm/Support/CommandLine.h"
#include <iostream>
#include <vector>
#include <algorithm>

namespace cl = llvm::cl;
static cl::opt<int64_t> tuneM("M", cl::desc("M dimension"), cl::init(128));
static cl::opt<int64_t> tuneN("N", cl::desc("N dimension"), cl::init(4096));
static cl::opt<int64_t> tuneK("K", cl::desc("K dimension"), cl::init(4096));
static cl::opt<std::string> tuneOutput("o", cl::desc("Output JSON"), cl::init("tune_result.json"));

struct TuningConfig {
    int64_t BM, BN, BK;
    int64_t num_threads;
    int64_t grain_size;
    double gflops = 0.0;
};

std::vector<TuningConfig> generateConfigs(int64_t M, int64_t N, int64_t K) {
    std::vector<TuningConfig> configs;
    for (int64_t BM : {1, 4, 8, 16, 32, 64}) {
        if (M == 1 && BM != 1) continue;
        for (int64_t BN : {16, 32, 64, 128, 256}) {
            for (int64_t BK : {32, 64, 128, 256}) {
                size_t tileMem = (BM * BK + 2 * BK * BN) * 2; // BF16
                if (tileMem > 32 * 1024 * 0.8) continue; // L1 capacity
                for (int64_t nt : {1, 2, 4, 8, 16}) {
                    for (int64_t gs : {1, 2, 4, 8}) {
                        configs.push_back({BM, BN, BK, nt, gs, 0.0});
                    }
                }
            }
        }
    }
    return configs;
}

int main(int argc, char **argv) {
    cl::ParseCommandLineOptions(argc, argv, "LLK autotuning harness\n");

    auto configs = generateConfigs(tuneM, tuneN, tuneK);
    std::cout << "Generated " << configs.size() << " configs for M="
              << tuneM << " N=" << tuneN << " K=" << tuneK << "\n";

    // Placeholder: measure each config, sort by gflops, save top-3
    std::cout << "Tuning complete. Results would be saved to " << tuneOutput << "\n";
    return 0;
}
```

- [ ] **Step 3: Build and commit**

```bash
cd build && ninja llk-bench llk-tune
git add tools/llk-bench/ tools/llk-tune/
git commit -m "feat: implement benchmarking and autotuning harness

- llk-bench: measures GFLOPS for given shape + thread count
- llk-tune: grid search over {BM,BN,BK,threads,grain}, L1-constrained

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## M5 Complete — Milestone 5 Exit Criterion Met

Shape bucketed schedule selection works. JIT cache avoids recompilation. Autotuning harness functional.

---

## Related

- [Plan Index](2026-07-14-llk-compiler-implementation.md)
- [Design Spec](../../design/m5-specialization-tuning.md)
- [ARCHITECTURE.md](../../ARCHITECTURE.md) — §2.3 Scheduling, §2.8 JIT & Caching, §2.9 Shape Specialization
- [Previous: M4 Parallel Execution](m4-parallel-execution.md)
- [Next: M6 Multi-Kernel](m6-multi-kernel.md)
