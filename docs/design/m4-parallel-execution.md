# Milestone 4: Parallel Execution

**Parent:** [ARCHITECTURE.md](../../ARCHITECTURE.md) — See §4 Implementation Sequence
**Previous:** [Milestone 3: Fused Memory Lowering](m3-fused-memory.md)
**Next:** [Milestone 5: Specialization & Tuning](m5-specialization-tuning.md)
**Status:** Design Approved | **Level:** Algorithm & Data Structure

---

## 1. Objectives & Exit Criterion

Add multi-threaded parallel decomposition of the outer M×N tile space. Prototype with OpenMP for fast functional validation, then replace with a persistent C++ thread pool for production inference. Support dynamic dispatch: parallel for large shapes, serial for small shapes to avoid overhead.

**Exit criterion:** Wall-clock time decreases monotonically with thread count (up to physical cores) for large shapes (M×N ≥ 1000 tiles). Small shapes (M < 4) correctly dispatch serial without thread-pool invocation. All M1-M3 correctness tests pass with parallelism enabled.

---

## 2. Components to Build

```
New files:
  include/LLK/Runtime/ThreadPool.h              # Persistent thread pool interface
  runtime/ThreadPool.cpp                         # Thread pool implementation
  include/LLK/Transforms/ParallelDecompose.h     # scf.forall decomposition pass
  lib/Transforms/ParallelDecompose.cpp           # M×N → linearized scf.forall
  include/LLK/Transforms/SerialParallelDispatch.h # Dispatch threshold pass
  lib/Transforms/SerialParallelDispatch.cpp      # Shape-based serial/parallel selection
  lib/Conversion/LinalgToCPU/ForallToOpenMP.cpp  # scf.forall → OpenMP (prototype)
  lib/Conversion/LinalgToCPU/ForallToRuntime.cpp # scf.forall → llrt.parallel_for (production)
  test/Transforms/parallel_decompose.mlir         # FileCheck
  test/Transforms/forall_to_openmp.mlir           # FileCheck: OpenMP lowering
  test/Transforms/forall_to_runtime.mlir          # FileCheck: runtime pool lowering
  test/Transforms/serial_dispatch.mlir            # FileCheck: threshold-based dispatch
  test/Execution/swiglu_parallel_scaling.cpp      # Scaling + correctness
  test/Execution/swiglu_small_shape.cpp           # Serial dispatch verification
  test/Execution/thread_pool_stress.cpp           # No leaks, deadlocks
  test/Execution/affinity_test.cpp                # Core pinning (Linux)
```

---

## 3. Parallel Decomposition Strategy

### 3.1 Work Linearization

```
num_M_tiles = ceil(M / BM)
num_N_tiles = ceil(N / BN)
total_tiles = num_M_tiles × num_N_tiles

Linearize 2D tile space:
  global_id = mTile * num_N_tiles + nTile
```

### 3.2 Serial/Parallel Decision

```
if M <= SERIAL_MAX_M:       // ≤ 3 (autoregressive decode)
    → fully serial
elif total_tiles < num_threads:
    → serial (not enough work to justify dispatch overhead)
else:
    → parallel: distribute tiles across workers
```

Thresholds are target-specific, stored with the schedule config. In Milestone 5, the schedule database's `num_threads` field overrides this static heuristic.

### 3.3 Tile→Worker Mapping (Static Chunking)

```
tiles_per_worker = ceil(total_tiles / num_threads)

Worker w handles tiles:
    [w * tiles_per_worker, min((w+1) * tiles_per_worker, total_tiles))
```

Static chunking provides predictable cache behavior. Work-stealing is a future optimization for highly imbalanced workloads.

---

## 4. Persistent Thread Pool

### 4.1 Design

```cpp
class ThreadPool {
public:
    explicit ThreadPool(int num_threads = std::thread::hardware_concurrency());
    ~ThreadPool();  // Signals stop, joins all workers

    // Primary interface: parallel-for with grain_size
    void parallelFor(int64_t total_tiles, int64_t grain_size,
                     std::function<void(int64_t tile_id, int worker_id)> callback);

    // Barrier across all workers
    void barrier(int worker_id);

    // Per-worker scratch (avoids false sharing on temp buffers)
    void* getScratch(int worker_id);
    void  setScratch(int worker_id, void* ptr, size_t size);

    int numWorkers() const;
    int workerId() const;  // -1 if called from non-pool thread

    // NUMA control
    void setAffinity(int worker_id, int cpu_core);

private:
    struct Worker {
        int id;
        std::thread thread;
        std::atomic<bool> running{true};
    };
    std::vector<Worker> workers_;

    // Work distribution state (reset per parallelFor)
    std::atomic<int64_t> next_tile_{0};
    int64_t total_tiles_{0};
    int64_t grain_size_{0};
    std::function<void(int64_t, int)>* current_callback_{nullptr};

    // Synchronization
    std::mutex mutex_;
    std::condition_variable cv_work_;
    std::condition_variable cv_done_;
    std::atomic<int> workers_done_{0};

    // Barrier: sense-reversing centralized barrier
    std::atomic<int> barrier_count_{0};
    std::atomic<int> barrier_epoch_{0};
};
```

### 4.2 Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| Workers created once at pool construction | No repeated `pthread_create` overhead per kernel invocation |
| Static chunking (default) | Predictable cache behavior; good for regular tile workloads |
| `grain_size` parameter | Minimum tiles per worker assignment; small grain = better balance, large grain = better locality |
| Per-worker scratch | Avoids false sharing on per-thread temporary buffers |
| NUMA affinity via `setAffinity()` | Pin workers to specific cores; avoid migration |
| Sense-reversing barrier | Low-contention barrier for multi-phase kernels |

---

## 5. IR Transformations

### 5.1 ParallelDecompose Pass

```
Input:  nested scf.for over M tiles, N tiles
Output: scf.forall over linearized tile space

algorithm:
    match:
        %m_loop = scf.for %m = 0 to num_M_tiles step 1
            %n_loop = scf.for %n = 0 to num_N_tiles step 1
                // tile body using %m, %n

    replace with:
        %total = arith.muli %num_M_tiles, %num_N_tiles
        %tiles = scf.forall (%tid) in (%total) shared_outs(...) {
            %m = arith.divsi %tid, %num_N_tiles
            %n = arith.remsi %tid, %num_N_tiles
            // inline tile body with %m, %n
            scf.forall.in_parallel { ... }
        }
```

### 5.2 OpenMP Lowering (Prototype)

Uses MLIR's existing conversion chain:

```
scf.forall
  → convert-scf-forall-to-scf-parallel    (MLIR built-in)
  → convert-scf-parallel-to-openmp         (MLIR built-in)
  → omp.parallel + omp.wsloop
  → LLVM OpenMP runtime calls
```

### 5.3 Runtime Pool Lowering (Production)

```
scf.forall {total_tiles, grain_size}
  → llrt.parallel_for(%total_tiles, %grain_size, @worker_fn, %ctx)
```

Where `@worker_fn` is an outlined function taking `(tile_id, worker_id)` and `%ctx` is the `KernelContext*`.

```mlir
llrt.parallel_for %total_tiles grain_size(%grain_size)
    worker(@worker_fn) context(%ctx)
    : (index, i32, !llvm.ptr) → ()
```

---

## 6. Updated KernelContext

```cpp
struct KernelContext {
    void* scratch;             // Base scratch buffer
    size_t scratch_size;
    ThreadPool* thread_pool;   // Added in this milestone
    int64_t M, N, K;          // Runtime dimensions (for dispatch decisions)
    int64_t total_tiles;
    int64_t grain_size;
};
```

---

## 7. Updated Pass Pipeline

Pipeline additions from M3 shown in **bold**:

```
llk.fused_swiglu → LLKToLinalg → canonicalize → shape-specialize
  → fuse-double-contraction → pack-weights
  → tile-and-vectorize
  → parallel-decompose               (NEW: outer forall on linearized M×N)
  → serial-parallel-dispatch         (NEW: threshold-based serial/parallel)
  → canonicalize
  → One-Shot Bufferize
  → scratch-analysis → allocation-hoisting → buffer-deallocation
  → forall-to-openmp                 (NEW: prototype path)
    OR
    forall-to-llrt-runtime           (NEW: production path)
  → convert-vector-to-llvm
  → scf/cf/arith/math/func/memref-to-llvm
  → LLVM IR
```

---

## 8. Test Specifications

| Test | What it verifies | Tool |
|------|-----------------|------|
| `test/Transforms/parallel_decompose.mlir` | Nested scf.for → single scf.forall with linearized tid + divsi/remsi 2D mapping | `llk-opt` + FileCheck |
| `test/Transforms/forall_to_openmp.mlir` | scf.forall → scf.parallel → omp.parallel conversion chain | `llk-opt` + FileCheck |
| `test/Transforms/forall_to_runtime.mlir` | scf.forall → llrt.parallel_for call with outlined worker | `llk-opt` + FileCheck |
| `test/Transforms/serial_dispatch.mlir` | M=1,N=64 → zero parallel ops; M=128,N=4096 → parallel ops present | `llk-opt` + FileCheck |
| `test/Execution/swiglu_parallel_scaling.cpp` | M=128,N=4096,K=4096 at {1,2,4,8} threads: monotonic wall-time decrease. 8-thread == 1-thread result | GTest + timer |
| `test/Execution/swiglu_small_shape.cpp` | M=1,N=64,K=4096: serial dispatch, ThreadPool::parallelFor NEVER called | GTest + mock |
| `test/Execution/thread_pool_stress.cpp` | 10,000 rapid parallelFor calls: no thread leaks, no deadlocks | GTest |
| `test/Execution/affinity_test.cpp` | `setAffinity(w, core)` → `sched_getcpu()` within 1-core tolerance | GTest (Linux) |

---

## 9. Dependencies

- [Milestone 3](m3-fused-memory.md) — fused double-contraction, scratch analysis
- [Milestone 2](m2-explicit-vector.md) — tiling + vectorization
- [Milestone 1](m1-scalar-pipeline.md) — JIT + test harness
- OpenMP development libraries (prototype path only)

---

## 10. Related

- [ARCHITECTURE.md](../../ARCHITECTURE.md) — §2.7 CPU Parallel Runtime
- [m3-fused-memory.md](m3-fused-memory.md) — fused tiles this parallelizes
- [m5-specialization-tuning.md](m5-specialization-tuning.md) — schedule-driven dispatch overrides static thresholds
