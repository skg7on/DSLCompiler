# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

An out-of-tree MLIR compiler for tiled CPU LLM kernels (SwiGLU, RoPE, Attention). Progressive lowering through 7 MLIR stages: LLK dialect → Linalg → tiled/fused → Vector SIMD → memref → CF+runtime → ORC JIT. First target: fused SwiGLU with BF16, FP32 accumulation, x86-64 AVX2.

## Build & Test (once implementation begins)

```bash
mkdir -p build && cd build
cmake .. -G Ninja -DMLIR_DIR=$HOME/llvm-project/build/lib/cmake/mlir -DLLVM_DIR=$HOME/llvm-project/build/lib/cmake/llvm
ninja                            # Build all targets
ninja llk-opt                    # Build IR tool only
ninja check-llk                  # Run all tests
./bin/llk-opt input.mlir         # Parse + print IR
./bin/llk-opt --llk-to-linalg input.mlir  # Run a specific pass
./bin/<TestName>                 # Run a single GTest
```

## Architecture

**7-stage progressive lowering:**

```
LLK dialect (domain ops) → Linalg (structured compute) → Tiled+Fused (scf.forall) → Vector SIMD (vector.contract) → memref (bufferized) → CF+LLVM+runtime → LLVM IR → ORC JIT
```

**Key design rules (from ARCHITECTURE.md):**
1. Semantics, scheduling, and target lowering are separate — never embed tile sizes in lowering passes
2. Lower to Linalg first; do not write custom loop generation before using Linalg's tiling/fusion/vectorization
3. Keep tensor form through tiling/fusion; One-Shot Bufferize only after all optimizations
4. Emit explicit Vector dialect ops; do not rely on LLVM auto-vectorization for the microkernel
5. Numerical approximation (exp, sigmoid, cos, sin) is IR semantics, not compiler flags — use `#llk.math_mode`
6. Specialize on layout and ISA before exact shapes
7. No autotuning until deterministic schedules work
8. Measure packing and parallel dispatch overhead — a faster inner loop can still slow inference
9. Use external GEMM as baseline, not necessarily final implementation
10. Keep first target narrow: one op, one dtype, one layout, one ISA

**Custom dialect (llk):** Small — only retains domain info unavailable in generic MLIR. Never recreate tensor/linalg/vector/memref. Ops: `llk.fused_swiglu`, `llk.rope`, `llk.attention`.

**Schedules are data:** Transform dialect `.mlir` files or `schedule_db.json`. Keyed by `(operation, M_bucket, N, K, dtype, ISA, math_mode)`. 5 M-buckets: {1}, [2,4], [5,16], [17,64], ≥65.

## Conventions

- C++20, CMake ≥ 3.20, LLVM/MLIR 20+, GTest, FileCheck
- Naming: `llk` dialect prefix, `llk-*` tools, CamelCase passes, snake_case files
- Error handling: MLIR `emitError()` for verifier failures, `llvm::Expected<T>` for JIT ops
- ABI: C structs (`Tensor2D`, `KernelContext`) — not MLIR memref descriptors
- TDD: every task starts with a failing test, then minimal code; commit per task

## Milestone Sequence

M1 (scalar pipeline) → M2 (AVX2 vector) → M3 (fused memory) → M4 (parallel) → M5 (specialization + cache + tuning) → M6 (RoPE + Attention + shared infra). Each builds on the previous. Python frontend deferred to M7+.

## Key Files

| File | Purpose |
|------|---------|
| `ARCHITECTURE.md` | Full component map, pipeline diagram, design decisions |
| `docs/design/m[1-6]-*.md` | Per-milestone design specs (MLIR defs, algorithms, tests) |
| `docs/superpowers/plans/m[1-6]-*.md` | Implementation plans with TDD tasks + complete code |
| `docs/superpowers/plans/2026-07-14-llk-compiler-implementation.md` | Plan index + file map |
