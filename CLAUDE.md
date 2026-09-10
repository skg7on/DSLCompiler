# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Worktree Rules (HARD GATE)

See [.claude/rules/worktree-isolation.md](.claude/rules/worktree-isolation.md) for the mandatory worktree isolation policy. All writes require an isolated git worktree.

## Project Overview

An out-of-tree MLIR compiler centered on a canonical DNN **Micro-IR** (`micro` dialect) for AI-chipset performance evaluation and auto-scheduling. Semantic/structured IR lowers into `micro`, a hardware-near execution IR modeling compute engines, memory hierarchy, data movement, spatial mapping, pipeline overlap, synchronization, and tunable schedule choices. Intel AVX2 CPU is the first validation backend and machine profile — not the final scope.

## Build & Test

```bash
mkdir -p build && cd build
cmake .. -G Ninja -DLLVM_PROJECT_BUILD_DIR=/Users/skg7on/Workspace/Projects/llvm-project/build
ninja                            # Build all targets
ninja llk-opt                    # Build IR tool only
ctest --output-on-failure        # Run all tests (CTest; there is no check-llk target)
ctest -R MicroDialectTileOps     # Run a single FileCheck test
./bin/llk-opt input.mlir         # Parse + print IR
./bin/llk-opt --llk-to-linalg input.mlir  # Run a specific pass
./bin/<TestName>                 # Run a single GTest
```

**CI runs almost none of this suite.** It configures with `-DLLK_BUILD_TOOLS=OFF -DLLK_BUILD_E2E_TESTS=OFF`, so the JIT execution tests (`test/Execution/*`, `test/Numerical/*`) are never built. It also builds LLVM with `-DLLVM_INCLUDE_TESTS=OFF`, so no `FileCheck` binary exists, `find_program(FILECHECK_BIN)` fails, and `if(FILECHECK_BIN)` silently registers none of the `.mlir` tests. Green CI therefore says nothing about either path — run `ctest` locally before trusting a change.

## Architecture

**Two lowering paths from LLK/Linalg:**

1. **CPU validation path** — Linalg → tiled/fused (scf.forall) → Vector SIMD → memref → CF+runtime → LLVM → ORC JIT → AVX2
2. **Micro-IR path (M9+)** — `LLKToMicro` → `micro.search_space` (parametric) or concrete `micro.kernel` → performance model / future target lowering

**Micro-IR conceptual layering** (one dialect, one `micro` namespace):
`uTile` (tile dataflow) → `uMap` (memory placement, owner mapping, pipeline, sync) → `uHW` (instruction fragments, machine-visible events)

**Primary `micro` value is a tile:** `Tile = (shape, dtype, layout, memory_space, owner)`. Four cost classes: *logical* (zero-cost views/slices/partitions), *memory* (materialized), *execution* (owner-mapped), *instruction fragment* (MMA/vector/DMA granularity). Concrete `micro.kernel` must contain no DNN semantic ops.

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

**Custom dialects:** `llk` is small — only retains domain info unavailable in generic MLIR (`llk.fused_swiglu`, `llk.rope`, `llk.attention`); never recreate tensor/linalg/vector/memref. `micro` (`include/LLK/Dialect/Micro/`) is the tile-centric execution IR. Implemented: `!micro.tile`, `#micro.layout`, `#micro.owner`, `#micro.memory`; ops `micro.kernel`, `micro.{tile_view,tile_alloc,tile_partition,tile_async_copy,tile_store}`, `micro.{mma,vector,reduce}`, `micro.{for,spatial_for,pipeline,alloc,async_copy,wait,store}`. Search-space ops (`micro.search_space`, `micro.param`, `micro.constraint`, `micro.objective`, `micro.candidate`) are M9 remaining work (issue #44).

**Gotcha — `MicroEnums.h` is hand-written:** the CMake tablegen config does not run `-gen-enum-decls/-gen-enum-defs`. Adding an enum in `MicroDialect.td` requires a matching hand-written `stringify*`/`symbolize*` entry in `include/LLK/Dialect/Micro/MicroEnums.h`.

**Schedules are data:** Transform dialect `.mlir` files or `schedule_db.json`. Keyed by `(operation, M_bucket, N, K, dtype, ISA, math_mode)`. 5 M-buckets: {1}, [2,4], [5,16], [17,64], ≥65.

## Conventions

- C++20, CMake ≥ 3.20, LLVM/MLIR 20+, GTest, FileCheck
- Naming: `llk` dialect prefix, `llk-*` tools, CamelCase passes, snake_case files
- Error handling: MLIR `emitError()` for verifier failures, `llvm::Expected<T>` for JIT ops
- ABI: C structs (`Tensor2D`, `KernelContext`) — not MLIR memref descriptors
- TDD: every task starts with a failing test, then minimal code; commit per task
- FileCheck tests are plain `add_test` entries in the root `CMakeLists.txt` — there is no lit runner, and `// RUN:` lines are comments only. Register new `.mlir` tests by hand: `add_llk_filecheck_test(Name test/Dialect/Micro/foo.mlir)` (~line 407), or a raw `add_test` using `--verify-diagnostics --split-input-file` for invalid-IR tests (~line 600)

## Milestone Sequence

M1–M6 complete (CPU pipeline, AVX2 vector, fused memory, parallel dispatch, specialization/tuning, RoPE + Attention). **Current work is the M9+ Micro-IR roadmap** (tracker: issue #41): M9 dialect infra + concrete execution ops ✅, search-space ops (open) → M10 MachineModel YAML + AVX2 perf simulator → M11 LLKToMicro lowering + search-space export → M12 tuning core + candidate ranking → M13 measurement loop + calibration. Python frontend (M7) deferred.

## Key Files

| File | Purpose |
|------|---------|
| `ARCHITECTURE.md` | Full component map, pipeline diagram, design decisions |
| `docs/design/m[1-6]-*.md` | Per-milestone design specs (MLIR defs, algorithms, tests) |
| `docs/design/m9-canonical-micro-ir-architecture.md` | Micro-IR project-level redesign, layer responsibilities |
| `docs/design/m9-micro-ir-core-concepts.md` | `micro` dialect semantics: tile model, op families, attrs, verifier rules |
| `docs/superpowers/specs/2026-08-13-micro-ir-tile-programming-model-spec.md` | Detailed tile programming model |
| `include/LLK/Dialect/Micro/` + `lib/Dialect/Micro/` | `micro` dialect implementation |
| `docs/superpowers/plans/m[1-6]-*.md` | Implementation plans with TDD tasks + complete code |
| `docs/superpowers/plans/2026-07-14-llk-compiler-implementation.md` | Plan index + file map |
