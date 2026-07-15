# Domain-Specific LLM Kernel Compiler — Architecture

## 1. Core Concept

**What it is:** An out-of-tree MLIR compiler that takes a single, computationally dense LLM operation (SwiGLU, RoPE, Attention) and aggressively optimizes its memory and compute lowering to produce high-performance CPU kernels — essentially a miniature, CPU-focused version of Triton.

**What it is NOT:** A general model compiler, an ONNX importer, a full Triton GPU backend, or a collection of handwritten intrinsic kernels.

### The Central Principle

```
LLM-domain semantics        ←  "what" — the mathematical operation
        ↓
structured tensor computation  ←  linalg: tiling, fusion, producer-consumer
        ↓
parameterized schedule       ←  tile sizes, vector width, thread count (data, not code)
        ↓
explicit vector microkernel  ←  vector.contract, transfer_read/write — no auto-vec
        ↓
physical memory & thread runtime  ←  memref, persistent thread pool
        ↓
LLVM target code             ←  ORC JIT → machine code
```

Each arrow is a lowering pass. Each stage can be dumped, inspected, and FileCheck-tested independently.

### First Target

`Y = SiLU(X·Wg) ⊙ (X·Wu)` — fused SwiGLU with BF16/FP16 inputs, FP32 accumulation, x86-64 AVX2. This operation exercises tiling, register-resident intermediates, vector contraction, weight packing, parallel decomposition, math approximation, and tail handling — all the hard compiler problems in one kernel.

### Progressive Lowering Pipeline

```
┌──────────────────────────────────────┐
│ 1. LLK Dialect                       │  Domain semantics: llk.fused_swiglu,
│    Kernel semantics, assumptions,    │  llk.rope, llk.attention
│    specialization contracts          │
└────────────────┬─────────────────────┘
                 │ LLKToLinalg
                 ▼
┌──────────────────────────────────────┐
│ 2. Tensor-Level Payload IR           │  func + tensor + linalg + arith + math
│    Structured computation, explicit  │  Producer-consumer relationships visible
│    producer-consumer edges           │
└────────────────┬─────────────────────┘
                 │ Transform dialect schedule
                 ▼
┌──────────────────────────────────────┐
│ 3. Tiled & Fused Structured IR       │  linalg + tensor + scf.forall
│    Tiling, fusion, interchange       │  Schedule from schedule_db.json
└────────────────┬─────────────────────┘
                 │ Structured vectorization
                 ▼
┌──────────────────────────────────────┐
│ 4. Explicit SIMD IR                  │  vector.transfer_read/write, vector.contract,
│    vector + scf + tensor             │  vector.mask, vector.shuffle, vector.fma
└────────────────┬─────────────────────┘
                 │ One-Shot Bufferize
                 ▼
┌──────────────────────────────────────┐
│ 5. Physical Memory IR                │  memref + vector + scf
│    Allocations, scratch planning     │  Scratch analysis pass audits allocations
└────────────────┬─────────────────────┘
                 │ CPU lowering
                 ▼
┌──────────────────────────────────────┐
│ 6. Control Flow + Thread Runtime     │  cf + llvm + omp (prototype) or
│    Parallel dispatch, barriers       │  llrt.parallel_for (production)
└────────────────┬─────────────────────┘
                 │
                 ▼
┌──────────────────────────────────────┐
│ 7. LLVM IR → ORC JIT / Object File   │  Machine code, cached by KernelKey
└──────────────────────────────────────┘
```

**Key rule:** Keep tensor-level transformations (tiling, fusion, vectorization) before bufferization. Tensor semantics make producer-consumer relationships explicit; One-Shot Bufferize converts to memrefs only after all optimizations.

---

## 2. Major Components

### 2.1 Custom Dialect (llk)

A **small, focused** dialect that only retains domain information unavailable in generic MLIR. Does NOT recreate tensor, linalg, vector, or memref.

| Op | Purpose | Detailed in |
|----|---------|-------------|
| `llk.fused_swiglu` | SiLU-gated double projection: `SiLU(X·Wg) ⊙ (X·Wu)` | [m1](docs/design/m1-scalar-pipeline.md), [m3](docs/design/m3-fused-memory.md) |
| `llk.rope` | Rotary Position Embedding with pairwise permutation | [m6](docs/design/m6-multi-kernel.md) |
| `llk.attention` | Simplified attention with online softmax | [m6](docs/design/m6-multi-kernel.md) |
| `llk.assume` | Divisibility and alignment hints | [m1](docs/design/m1-scalar-pipeline.md) |
| `llk.dispatch` | Multi-variant dispatch guard for shape specialization | [m5](docs/design/m5-specialization-tuning.md) |

**Attributes:** `#llk.activation<silu>`, `#llk.math_mode<strict|bounded_fast|unsafe_fast>`, `#llk.softmax_mode<online>`, `#llk.layout<row_major|packed_kn>`

### 2.2 Python Frontend (Milestone 7+)

Designed but deferred to keep the initial scope narrow. Two complementary entry paths converge at the LLK dialect:

- **Explicit DSL:** Triton-inspired `@llk.jit` decorator with block-programming model (`tl.load`, `tl.store`, `tl.dot`, `tl.arange`). Python AST captured via `inspect.getsource`, lowered to LLK IR via MLIR Python bindings.
- **PyTorch Capture:** `torch.compile(backend="llk")` or `torch.export` → pattern recognition converts recognized ATen ops to `llk.*` ops.

All expensive optimization, code generation, caching, and runtime execution remains in C++.

### 2.3 Scheduling System

**Separation of concerns:** Semantics (what) ≠ Schedule (how) ≠ Target (where).

Schedules are stored as data — Transform dialect IR or JSON configuration — never embedded in lowering passes.

```
semantic IR + target profile + shape profile → selected schedule (from schedule_db.json)
```

The schedule database maps `(operation, M_bucket, N, K, dtype, ISA, math_mode)` → `{BM, BN, BK, VM, VN, vector_width, num_threads, grain_size, ...}`.

Detailed in: [m2](docs/design/m2-explicit-vector.md), [m5](docs/design/m5-specialization-tuning.md)

### 2.4 Explicit SIMD (Vector Dialect)

The compiler emits explicit `vector` dialect operations; LLVM auto-vectorization is only used as cleanup for residual loops.

```
vector.transfer_read   → contiguous/masked load
vector.contract        → outer-product accumulation with FMA
vector.fma             → fused multiply-add
vector.shuffle         → permute/swizzle (critical for RoPE interleaving)
vector.mask            → predication for tail handling
vector.transfer_write  → contiguous/masked store
```

**Target profiles:** x86-64 + AVX2 + FMA (milestones 1-6), extensible to AVX-512/BF16, AMX, AArch64+NEON, AArch64+SVE.

Detailed in: [m2](docs/design/m2-explicit-vector.md)

### 2.5 Math Approximation System

Nonlinear math (exp, sigmoid, cos, sin) is a first-class compiler concern with explicit IR contracts:

| Mode | Behavior |
|------|----------|
| `strict` | IEEE-sensitive, accurate exp, preserve NaN/Inf |
| `bounded_fast` | Documented ULP bound, vector polynomial approximation, finite input range |
| `unsafe_fast` | Reassociation allowed, reciprocal approximation, reduced exception guarantees |

Example: SiLU uses `exp2(x * log2(e))` polynomial approximation, evaluated with FMA, validated for downstream model error.

Detailed in: [m3](docs/design/m3-fused-memory.md) (SiLU), [m6](docs/design/m6-multi-kernel.md) (cos/sin for RoPE, exp for softmax)

### 2.6 Memory & Bufferization

Transform in tensor form, bufferize once:

```
tile/fuse/vectorize (tensor) → One-Shot Bufferize → allocation hoisting → scratch placement → buffer deallocation
```

**Goal for optimized paths:** Zero full-size intermediate buffers. Only: weight packing, per-thread tile scratch, alignment padding, fallback paths.

A scratch analysis pass audits all `memref.alloc` calls post-bufferization to verify this invariant.

Detailed in: [m3](docs/design/m3-fused-memory.md)

### 2.7 CPU Parallel Runtime

**Prototype path:** `scf.forall` → `scf.parallel` → OpenMP dialect → LLVM/OpenMP runtime (fast to functional).

**Production path:** Persistent C++ thread pool with `llrt.parallel_for`, `llrt.barrier`, `llrt.get_worker_id`:

- Workers created once, reused across kernel invocations
- Static chunking work distribution (predictable cache behavior)
- Per-worker scratch allocation (no false sharing)
- NUMA-aware core affinity
- Dynamic serial dispatch for small shapes (M ≤ 3, or when total_tiles < num_threads)

Detailed in: [m4](docs/design/m4-parallel-execution.md)

### 2.8 JIT & Caching

**Stable public ABI:** C structs (`Tensor2D`, `KernelContext`), not MLIR memref descriptors. A thin wrapper converts C structs ↔ memref descriptors.

**Four-level cache:**

| Level | Key | What's cached | Detailed in |
|-------|-----|---------------|-------------|
| L1: IR | (source_hash, op_kind) | Verified MLIR module | [m5](docs/design/m5-specialization-tuning.md) |
| L2: Optimized MLIR | (source_hash, M_bucket, N, K, dtypes, schedule_version) | Post-optimization MLIR (pre-bufferize) | [m5](docs/design/m5-specialization-tuning.md) |
| L3: Object Code | KernelKey (13 fields) | Compiled function pointer | [m5](docs/design/m5-specialization-tuning.md) |
| L4: Packed Weights | (weight_ptr, shape, BK, BN) | Packed weight buffer | [m3](docs/design/m3-fused-memory.md), [m5](docs/design/m5-specialization-tuning.md) |

### 2.9 Shape Specialization

Bounded specialization rather than one fully dynamic kernel:

```
M buckets:  {1}, [2,4], [5,16], [17,64], [65,∞)
N, K:       exact match (model architecture determines)
ISA, dtype: exact match
```

| Regime | M | Strategy |
|--------|---|----------|
| Decode | 1 | Vectorized GEMV, parallelize across N blocks |
| Small batch | 2–4 | Micro-GEMM, limited threads |
| Medium batch | 5–64 | Tiled GEMM, full parallelism |
| Prefill | ≥65 | Conventional M×N tile parallelism |

Detailed in: [m5](docs/design/m5-specialization-tuning.md)

---

## 3. Project Structure

```
llk-compiler/
├── CMakeLists.txt
├── include/LLK/
│   ├── Dialect/         # LLKDialect.td, LLKOps.td, LLKAttributes.td
│   ├── Conversion/      # LLKToLinalg.h, LinalgToCPU.h
│   ├── Transforms/      # TileAndVectorize.h, FuseDoubleContraction.h, PackWeights.h,
│   │                      ShapeSpecialization.h, ScheduleSelection.h, MathApproximation.h,
│   │                      ParallelDecompose.h, ScratchAnalysis.h
│   ├── Target/          # X86/TargetAVX2.h, AArch64/TargetSVE.h
│   └── Runtime/         # ThreadPool.h, JitCache.h, PackedWeights.h, CpuFeatures.h
├── lib/
│   ├── Dialect/LLK/
│   ├── Conversion/LLKToLinalg/
│   ├── Transforms/      # Pass implementations + Common/ utilities
│   ├── Target/X86/, AArch64/
│   └── Runtime/
├── tools/
│   ├── llk-opt/         # IR dump + pass pipeline tool
│   ├── llk-compile/     # End-to-end compilation driver
│   ├── llk-bench/       # Benchmarking harness
│   └── llk-tune/        # Autotuning grid search
├── python/llk/          # Python frontend (Milestone 7+)
│   ├── language/        # DSL: tl.load, tl.store, tl.dot, etc.
│   ├── runtime/         # JIT invocation from Python
│   └── torch/           # torch.compile backend
├── runtime/             # C++ runtime library (ThreadPool, JitCache, PackedWeights)
├── schedules/           # Target-specific schedule definitions (.mlir + schedule_db.json)
│   ├── x86_avx2/
│   ├── x86_avx512/
│   └── aarch64_sve/
├── test/
│   ├── Dialect/         # FileCheck: op round-trip + verifier
│   ├── Conversion/      # FileCheck: lowering correctness
│   ├── Transforms/      # FileCheck: pass-specific tests
│   ├── Execution/       # JIT end-to-end + numerical
│   ├── Numerical/       # Precision + stability
│   └── benchmark/       # Performance regression suite
└── docs/
    ├── design/          # Per-milestone detailed design docs
    └── superpowers/     # Brainstorming + planning artifacts
```

---

## 4. Implementation Sequence

| Milestone | What | Exit Criterion | Design Doc |
|-----------|------|----------------|------------|
| M1 | Scalar end-to-end pipeline | Arbitrary shapes execute correctly through JIT | [m1-scalar-pipeline.md](docs/design/m1-scalar-pipeline.md) |
| M2 | Explicit vector path (AVX2) | No scalar inner-K loop; `vfmadd231ps` in assembly | [m2-explicit-vector.md](docs/design/m2-explicit-vector.md) |
| M3 | Fused memory lowering | Zero full-size gate/up intermediates | [m3-fused-memory.md](docs/design/m3-fused-memory.md) |
| M4 | Parallel execution | Monotonic scaling with thread count | [m4-parallel-execution.md](docs/design/m4-parallel-execution.md) |
| M5 | Specialization & tuning | Deterministic schedule selection; JIT cache hits work | [m5-specialization-tuning.md](docs/design/m5-specialization-tuning.md) |
| M6 | Second/third kernels (RoPE, Attention) | Both kernels compile + execute via same pipeline | [m6-multi-kernel.md](docs/design/m6-multi-kernel.md) |
| M7+ | Python frontend | `@llk.jit` + `torch.compile(backend="llk")` functional | (future) |

---

## 5. Engineering Rules

1. **Keep semantics, scheduling, and target lowering separate.** Never embed `BM=32` in LLKToLinalg.
2. **Lower to Linalg before writing custom loop generation.** Linalg already provides tiling, fusion, and vectorization infrastructure.
3. **Do not lower to memrefs too early.** Tile and fuse while tensor producer-consumer relationships remain explicit.
4. **Do not rely on LLVM auto-vectorization for the central microkernel.** Emit explicit Vector dialect operations.
5. **Treat numerical approximation as IR semantics,** not merely compiler command-line flags.
6. **Specialize on layout and ISA before specializing on every exact shape.**
7. **Do not introduce autotuning until deterministic schedules work.** Otherwise tuning hides compiler defects.
8. **Measure packing and parallel dispatch overhead.** A faster inner loop can still produce a slower inference kernel.
9. **Use external optimized GEMM as a baseline, not necessarily as the final implementation.** The compiler must demonstrate value through fusion, specialization, reduced materialization, or better small-shape behavior.
10. **Keep the first target narrow:** one operation, one dtype pair, one layout, one ISA, a small number of shape regimes.

---

## 6. Design Decisions

| Decision | Rationale |
|----------|-----------|
| Out-of-tree MLIR project | Leverages MLIR's multi-level IR infrastructure without forking |
| First target: fused SwiGLU | Exercises tiling, fusion, register intermediates, vector contract, packing, parallelism, math approximation — all the hard problems |
| Small custom dialect (llk) | Only retains domain info; reuses linalg/vector/memref for everything else |
| Transform dialect for schedules | Separates policy from mechanism; enables target-specific schedules as data |
| Explicit Vector dialect (not LLVM autovec) | Deterministic SIMD; stable performance across compiler versions |
| Python frontend deferred to M7+ | Stabilize IR and execution model first; C++ builder for testing during M1-M6 |
| Persistent thread pool (not OpenMP for production) | Low-latency inference; NUMA control; avoids nested parallelism conflicts |
| Bounded specialization (5 M-buckets) | Handles decode (M=1) vs. prefill (M≥65) divergence without combinatorial explosion |
| C struct ABI (not memref descriptors) | Stable public interface; framework-agnostic |
| Four-level JIT cache | Avoids re-parsing, re-optimizing, re-compiling, and re-packing independently |
