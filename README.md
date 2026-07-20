# DSLCompiler — Domain-Specific LLM Kernel Compiler

An out-of-tree [MLIR](https://mlir.llvm.org/) compiler that takes computationally dense LLM operations (SwiGLU, RoPE, Attention) and aggressively optimizes their memory and compute lowering to produce high-performance CPU kernels — essentially a miniature, CPU-focused version of [Triton](https://triton-lang.org/).

## Core Concept

```
LLM-domain semantics          ←  "what" — the mathematical operation
        ↓
structured tensor computation ←  linalg: tiling, fusion, producer-consumer
        ↓
parameterized schedule        ←  tile sizes, vector width, thread count
        ↓
explicit vector microkernel   ←  vector.contract — no auto-vectorization
        ↓
physical memory & threads     ←  persistent thread pool, scratch planning
        ↓
LLVM target code              ←  ORC JIT → machine code
```

## First Target

`Y = SiLU(X·Wg) ⊙ (X·Wu)` — fused SwiGLU with BF16 inputs, FP32 accumulation, x86-64 AVX2.

## Project Status

**Phase:** M1 (scalar pipeline) and M2 (AVX2 vector) in progress.

| Milestone | Status |
|-----------|--------|
| M1: Scalar end-to-end pipeline | In progress |
| M2: Explicit vector path (AVX2) | In progress |
| M3: Fused memory lowering | Design approved |
| M4: Parallel execution | Design approved |
| M5: Specialization & tuning | Design approved |
| M6: Multi-kernel (RoPE + Attention) | Design approved |
| M7+: Python frontend (Triton-like DSL) | Future |

## Documentation

| Document | Description |
|----------|-------------|
| [ARCHITECTURE.md](ARCHITECTURE.md) | High-level architecture, 7-stage lowering pipeline, component map, engineering rules |
| [docs/design/m1-scalar-pipeline.md](docs/design/m1-scalar-pipeline.md) | M1 design: LLK dialect, verifier, LLKToLinalg, JIT cache, ABI |
| [docs/design/m2-explicit-vector.md](docs/design/m2-explicit-vector.md) | M2 design: tiling schedule, vector dialect, AVX2 target, masked tails |
| [docs/design/m3-fused-memory.md](docs/design/m3-fused-memory.md) | M3 design: double-contraction fusion, weight packing, scratch analysis |
| [docs/design/m4-parallel-execution.md](docs/design/m4-parallel-execution.md) | M4 design: thread pool, parallel decomposition, dispatch thresholds |
| [docs/design/m5-specialization-tuning.md](docs/design/m5-specialization-tuning.md) | M5 design: M-bucketing, 4-level JIT cache, schedule DB, autotuning |
| [docs/design/m6-multi-kernel.md](docs/design/m6-multi-kernel.md) | M6 design: RoPE (shuffles), Attention (online softmax), shared infra |
| [docs/superpowers/plans/](docs/superpowers/plans/) | Implementation plans: ~29 TDD tasks with complete code |

## Architecture

### 7-Stage Progressive Lowering

```
LLK Dialect  →  Linalg Tensor IR  →  Tiled & Fused  →  Explicit SIMD (Vector)
      ↓              ↓                    ↓                    ↓
Domain ops    Structured compute    scf.forall tiles    vector.contract
                                  Transform schedule    vector.transfer_*

      ↓              ↓                    ↓
Physical Memory  →  CF + Runtime  →  LLVM IR  →  ORC JIT
   memref           llrt.parallel_for      Machine code
  scratch audit     persistent pool
```

### Key Design Decisions

- **Explicit SIMD, not LLVM autovec** — emit `vector.contract`, `vector.transfer_read/write` directly
- **Schedules as data** — Transform dialect IR or JSON, never embedded in passes
- **Small custom dialect** — only domain semantics (llk.fused_swiglu, llk.rope, llk.attention); reuse linalg/vector/memref
- **Persistent thread pool** — workers created once, reused across kernel invocations (not OpenMP for production)
- **Bounded specialization** — 5 M-buckets {1}, [2,4], [5,16], [17,64], ≥65; exact N, K
- **C struct ABI** — `Tensor2D`, `KernelContext`; not MLIR memref descriptors
- **Python frontend deferred** — stabilize IR model first; C++ builder for M1-M6 testing

## Tech Stack

- **C++20**, **CMake** ≥ 3.20
- **MLIR/LLVM** main branch (aligned with LLVM 20+)
- **Google Test** for C++ tests
- **FileCheck** for MLIR IR tests
- **PyTorch** for numerical reference implementations (test dependency only)

## Getting Started

```bash
# Clone
git clone https://github.com/skg7on/DSLCompiler.git
cd DSLCompiler

# Build (requires LLVM/MLIR built from source)
mkdir build && cd build
cmake .. -G Ninja -DLLVM_PROJECT_BUILD_DIR=/path/to/llvm-project/build
ninja

# Run tests
ninja check-llk
```

When `LLVM_PROJECT_BUILD_DIR` is set, `MLIR_DIR` and `LLVM_DIR` are inferred and
system-installed LLVM/MLIR are ignored. If omitted, `find_package` searches
the standard CMake prefixes.

## Engineering Rules

1. Keep semantics, scheduling, and target lowering separate
2. Lower to Linalg before writing custom loop generation
3. Do not lower to memrefs too early — tile and fuse in tensor form
4. Do not rely on LLVM auto-vectorization for the central microkernel
5. Treat numerical approximation as IR semantics, not command-line flags
6. Specialize on layout and ISA before specializing on every exact shape
7. No autotuning until deterministic schedules work
8. Measure packing and parallel dispatch overhead
9. Use external GEMM as baseline, not final implementation
10. Keep the first target narrow: one op, one dtype, one layout, one ISA

## License

MIT
