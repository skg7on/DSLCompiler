# LLK Compiler — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement these plans task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an out-of-tree MLIR compiler for tiled CPU LLM kernels across 6 sequential milestones.

**Architecture:** Progressive lowering through 7 MLIR stages: LLK dialect → Linalg tensor IR → tiled+fused IR → explicit Vector SIMD → memref physical memory → CF+LLVM+runtime → ORC JIT machine code.

**Tech Stack:** C++20, MLIR/LLVM (main branch, aligned with LLVM 20+), CMake ≥ 3.20, GTest, FileCheck, PyTorch (test reference only), nlohmann/json

## Global Constraints

- **MLIR version:** LLVM/MLIR main branch (aligned with LLVM 20+)
- **C++ standard:** C++20
- **Build system:** CMake, out-of-tree MLIR project
- **Test framework:** GTest (C++), FileCheck (MLIR IR), PyTorch (numerical references only)
- **Naming:** `llk` prefix for dialect; `llk-*` for tools; CamelCase for passes; snake_case for files
- **Error handling:** MLIR `emitError()` for verifier failures; `llvm::Expected<T>` for JIT ops; `abort()` on unrecoverable runtime errors
- **Target:** x86-64 Linux or macOS with AVX2-capable CPU
- **TDD:** Every implementation step starts with a failing test, then minimal code to pass
- **Commit granularity:** Commit after each task (test + implementation together)

---

## Milestone Plans

| Milestone | What | Tasks | Plan File | Design Doc |
|-----------|------|-------|-----------|------------|
| M1 | Scalar end-to-end pipeline | 1.1–1.8 | [m1-scalar-pipeline.md](m1-scalar-pipeline.md) | [m1-scalar-pipeline.md](../../design/m1-scalar-pipeline.md) |
| M2 | Explicit vector path (AVX2) | 2.1–2.6 | [m2-explicit-vector.md](m2-explicit-vector.md) | [m2-explicit-vector.md](../../design/m2-explicit-vector.md) |
| M3 | Fused memory lowering | 3.1–3.4 | [m3-fused-memory.md](m3-fused-memory.md) | [m3-fused-memory.md](../../design/m3-fused-memory.md) |
| M4 | Parallel execution | 4.1–4.3 | [m4-parallel-execution.md](m4-parallel-execution.md) | [m4-parallel-execution.md](../../design/m4-parallel-execution.md) |
| M5 | Specialization & tuning | 5.1–5.3 | [m5-specialization-tuning.md](m5-specialization-tuning.md) | [m5-specialization-tuning.md](../../design/m5-specialization-tuning.md) |
| M6 | Multi-kernel (RoPE + Attention) | 6.1–6.5 | [m6-multi-kernel.md](m6-multi-kernel.md) | [m6-multi-kernel.md](../../design/m6-multi-kernel.md) |

## Execution Order

Milestones are strictly sequential — each builds on the previous. Implement M1 through M6 in order.

```
M1 (scalar) → M2 (vector) → M3 (fusion) → M4 (parallel) → M5 (specialize) → M6 (multi-kernel)
```

## Project File Map

```
llk-compiler/
├── CMakeLists.txt                    # M1: bootstrap
├── include/LLK/
│   ├── Dialect/                      # M1: llk dialect + ops; M6: rope, attention
│   ├── Conversion/                   # M1: LLKToLinalg; M4: ForallToOpenMP/Runtime
│   ├── Transforms/                   # M2: TileAndVectorize; M3: FuseDoubleContraction,
│   │                                  #   PackWeights, ScratchAnalysis
│   │                                  # M4: ParallelDecompose, SerialParallelDispatch
│   │                                  # M5: ShapeSpecialization, ScheduleSelection
│   │                                  # M6: OnlineSoftmax
│   ├── Target/X86/                   # M2: TargetAVX2
│   └── Runtime/                      # M1: JitCache; M3: PackedWeights
│                                      # M4: ThreadPool; M5: KernelKey
├── lib/                              # (mirrors include/ layout)
├── tools/
│   ├── llk-opt/                      # M1: IR dump tool
│   ├── llk-compile/                  # M1: compilation driver
│   ├── llk-bench/                    # M5: benchmark harness
│   └── llk-tune/                     # M5: autotuning harness
├── runtime/                          # M1: JitCache; M3: PackedWeights; M4: ThreadPool
├── schedules/                        # M2: schedule .mlir files; M5: schedule_db.json
└── test/
    ├── Dialect/                      # M1: op round-trip + verifier
    ├── Conversion/                   # M1: LLK→Linalg; M6: rope, attention lowering
    ├── Transforms/                   # M2-M6: per-pass FileCheck tests
    ├── Execution/                    # M1-M6: end-to-end JIT + numerical
    ├── Numerical/                    # M2: vector vs scalar; M5: precision; M6: stability
    └── benchmark/                    # M5: regression database
```

---

## Architecture Reference

See [ARCHITECTURE.md](../../ARCHITECTURE.md) for the full component map, 7-stage lowering pipeline diagram, and engineering rules.
