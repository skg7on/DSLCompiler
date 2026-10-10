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
ninja check-llk                  # Build the test binaries + run the whole suite
ctest --output-on-failure        # Same suite, without building first
ctest -R MicroDialectTileOps     # Run a single FileCheck test
./bin/llk-opt input.mlir             # Parse + print IR
./bin/llk-opt --llk-to-linalg input.mlir  # Run a specific pass
./bin/llk-opt --micro-map="target=x86-avx2 machine=machines/x86-avx2-v2.yaml layouts=mapping/x86-avx2/layouts.llkmap rules=mapping/x86-avx2/rules.llkmap emitters=avx2_vector_add,avx2_vector_convert,avx2_vector_silu,avx2_vector_mul,avx2_mma,avx2_reduce,avx2_copy,avx2_tile_copy,avx2_tile_store" input.mlir  # Search, bind best plan
./bin/LLKMappingTests --gtest_filter=MappingPlan.*  # Run selected GTest cases
```

Root `CMakeLists.txt` builds libraries/tools; `test/CMakeLists.txt` and category subdirectories build one `LLK<Category>Tests` binary per category, with individual CTest cases. FileCheck/verifier registrations live in `test/FileCheck.cmake`; libraries go to `build/lib`, all executables to `build/bin`. `BUILD_TESTING=OFF` disables tests and GoogleTest; `LLK_BUILD_E2E_TESTS=OFF` omits JIT execution tests.

Configure writes a compilation database to `build/compile_commands.json` for clangd and other editors; pass `-DCMAKE_EXPORT_COMPILE_COMMANDS=OFF` to suppress it.

The compiler is the `clang` on `PATH` — on macOS that is Homebrew's clang, not the Apple toolchain at `/usr/bin/cc`. `CMakeLists.txt` drops an **absolute** `CC`/`CXX` that names a missing file (typically a `brew --prefix llvm@NN` path left behind after the keg was uninstalled) and prints an `Ignoring ...` status line rather than aborting. Bare names such as `CC=gcc` are left alone. Both the fallback and the compiler search apply only to a fresh build tree: a `build/` that already cached a compiler keeps it, so `rm -rf build` before reconfiguring whenever the toolchain changes.

CI builds the tools, E2E tests, and the `FileCheck` utility from the same pinned LLVM source, so the full registered suite runs. Local LLVM builds must use `-DLLVM_BUILD_UTILS=ON`; configuration fails when `LLK_BUILD_TOOLS=ON` and a matching `FileCheck` is unavailable. `check-llk` is a thin wrapper over CTest, so it runs exactly the registered suite — run it locally before trusting a change.

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

**Custom dialects:** `llk` is small — only retains domain info unavailable in generic MLIR (`llk.fused_swiglu`, `llk.rope`, `llk.attention`); never recreate tensor/linalg/vector/memref. `micro` (`include/LLK/Dialect/Micro/`) is the tile-centric execution IR. Implemented: `!micro.tile`, `#micro.layout`, `#micro.owner`, `#micro.memory`; ops `micro.kernel`, `micro.{tile_view,tile_alloc,tile_partition,tile_async_copy,tile_store}`, `micro.{mma,vector,reduce}`, `micro.{for,spatial_for,pipeline,alloc,async_copy,wait,store}`, and `micro.transform` (a layout conversion naming each side by its affine map, not a target layout id). Search-space ops (`micro.search_space`, `micro.param`, `micro.constraint`, `micro.objective`, `micro.candidate`) are implemented (#44). A bound plan adds target-neutral metadata attributes to the kernel: `micro.plan`, `micro.mapping`, `micro.routes`, plus `micro.value`/`micro.dst_node` on each emitted copy.

**Mapping subsystem (epic #67):** target policy is declarative and consumed by generic code. Libraries: `LLKMachine` (`include/LLK/Machine/` — v2 topology: executor/memory/compute/transfer nodes with `contains`/`dominates`/`attached_to`/`link` edges) and `LLKMapping` (`include/LLK/Mapping/` — workload graph, candidate/instance/connection/covering, routing, LLKMap, placement, covering search, plan binder, cost/latency/diagnostics/report). Target packages live in `lib/Target/<Target>/Mapping/` (`LLKTargetMapping`) with policy in `mapping/<target>/{layouts,rules}.llkmap`; machines are `machines/*-v2.yaml`. Generic `lib/Mapping`/`lib/Machine` code must never branch on a target name — policy belongs in the `.llkmap`/`.yaml` files.

**Gotcha — `MicroEnums.h` is hand-written:** the CMake tablegen config does not run `-gen-enum-decls/-gen-enum-defs`. Adding an enum in `MicroDialect.td` requires a matching hand-written `stringify*`/`symbolize*` entry in `include/LLK/Dialect/Micro/MicroEnums.h`. The mapping diagnostics enum (`include/LLK/Mapping/Diagnostics.h`) follows the same hand-written pattern, as do the MachineModel v2 kind/direction stringifiers.

**Gotcha — adding a MachineModel field:** give it a default member initializer (e.g. `std::vector<std::string> equivalentTo{};`) or every aggregate test literal emits `-Wmissing-field-initializers`. The loader preserves the minor version (`MachineModel::schemaMinor`) and tolerates *unknown keys only when the file declares a minor greater than `kSupportedSchemaMinor`*; missing required keys, wrong node types, malformed values, and unknown kinds are always rejected. `schemaMinor` is excluded from the content hash.

**Gotcha — compiler version is generated:** `CMakeLists.txt` does `configure_file(include/LLK/Version.h.in → ${CMAKE_BINARY_DIR}/include/LLK/Version.h)` producing `LLK_COMPILER_VERSION` from the project version plus `git describe`. The plan report prints it; there is no hand-written version constant.

**Schedules are data:** Transform dialect `.mlir` files or `schedule_db.json`. Keyed by `(operation, M_bucket, N, K, dtype, ISA, math_mode)`. 5 M-buckets: {1}, [2,4], [5,16], [17,64], ≥65.

## Conventions

- C++20, CMake ≥ 3.20, LLVM/MLIR 20+, GTest, FileCheck
- Naming: `llk` dialect prefix, `llk-*` tools, CamelCase passes, snake_case files
- Error handling: MLIR `emitError()` for verifier failures, `llvm::Expected<T>` for JIT ops
- ABI: C structs (`Tensor2D`, `KernelContext`) — not MLIR memref descriptors
- TDD: every task starts with a failing test, then minimal code; commit per task
- FileCheck tests are plain `add_test` entries in `test/FileCheck.cmake` — there is no lit runner, and `// RUN:` lines are comments only. Register new `.mlir` tests by hand with `add_llk_filecheck_test(Name test/Dialect/Micro/foo.mlir)` (an optional trailing argument is passed through to the tool, e.g. pass options), or use a raw `add_test` with `--verify-diagnostics --split-input-file` for invalid-IR tests
- Mapping/target boundary: generic `lib/Mapping` + `lib/Machine` stay target-neutral; target policy is data (`mapping/*.llkmap`, `machines/*.yaml`). `LLKMapping` takes `-fno-rtti -fno-exceptions` and links only `LLVMSupport`/`LLKMachine` (+ `MLIRIR`/`MLIRAsmParser` via `mlir_target_link_libraries`)

## Milestone Sequence

M1–M6 complete (CPU pipeline, AVX2 vector, fused memory, parallel dispatch, specialization/tuning, RoPE + Attention). M7 (Python frontend) deferred.

**M9–M12 and the epic #67 mapping subsystem complete.** Search-space IR (#44), LLKToMicro + search export (#47/#48), MachineModel + `micro-perf` (#45/#46), tuning core + candidate binding/ranking (#49/#50), and the topology-aware mapping subsystem: MachineModel v2 topology, deterministic routing, declarative LLKMap layouts and rules, placement and connection synthesis, deterministic/beam/exact covering search, AVX2 + generic-accelerator target packages, plan binding and emission, §22.3 diagnostics, §22.2 versioned plan report, and an optional `LatencyProvider`. The 27-task improvement plan's A/B/C stages are merged: endpoint identity and layered verification (A), complete plan materialization, storage, synchronization and joint exact connection search (B), and explicit kernel ABI, target-owned emitter lowering, a descriptor-pointer `MappedExecutable`, shared mapped compilation, tuner integration, fused rules and acceptance chains (C1–C8); C9 removed backend owner vocabulary from generic Micro ODS. Revision-pinned evidence for every normative §29 criterion is published in `docs/reviews/issue67-final-acceptance.md`.

**Remaining: M13 — measurement loop and calibration (#51/#52).** The mapped compilation path (C4/C5) executes selected kernels and supplies the measurement callback, event keys and ABI identity #51/#52 need; delivering calibrated predictions remains separate work. A directly-interpreting Micro emulator (#54) stays deferred and is not a prerequisite. Tracker: issue #41; epic #67.

## Key Files

| File | Purpose |
|------|---------|
| `ARCHITECTURE.md` | Full component map, pipeline diagram, design decisions |
| `docs/design/m[1-6]-*.md` | Per-milestone design specs (MLIR defs, algorithms, tests) |
| `docs/design/m9-canonical-micro-ir-architecture.md` | Micro-IR project-level redesign, layer responsibilities |
| `docs/design/m9-micro-ir-core-concepts.md` | `micro` dialect semantics: tile model, op families, attrs, verifier rules |
| `docs/superpowers/specs/2026-08-13-micro-ir-tile-programming-model-spec.md` | Detailed tile programming model |
| `docs/superpowers/specs/2026-09-18-microir-inspired-dslcompiler-enhancement-design.md` | Epic #67 design: mapping engine, MachineModel v2, routing, LLKMap, covering search, plan binding (§-numbered throughout the code) |
| `docs/design/micro-ir-mapping-workflow.md` | Contributor workflow: extract → match → place → connect → cover → bind → verify |
| `docs/reviews/issue67-final-acceptance.md` | Revision-pinned §29 acceptance evidence: criterion → tests → result → limitation, and the #51/#52 handoff |
| `docs/design/llkmap-layout-grammar.md`, `docs/design/llkmap-rule-grammar.md` | LLKMap declarative grammar: layouts, rules, target bundles |
| `include/LLK/Machine/` + `lib/Machine/` | `MachineModel` v2 topology, loader, content hash |
| `include/LLK/Mapping/` + `lib/Mapping/` | Mapping core: workload graph, routing, LLKMap, rules, placement, covering search, plan binder, cost/latency/diagnostics/report |
| `include/LLK/Target/*/Mapping/` + `lib/Target/*/Mapping/` | AVX2 and generic-accelerator mapping targets (`LLKTargetMapping`) |
| `mapping/<target>/{layouts,rules}.llkmap` | Declarative target policy (layouts + mapping rules + bundles) |
| `machines/*-v2.yaml` | MachineModel v2 topology profiles |
| `docs/superpowers/plans/2026-10-03-micro-ir-gap-closure.md` | Gap-closure plan (31 TDD tasks) |
| `include/LLK/Dialect/Micro/` + `lib/Dialect/Micro/` | `micro` dialect implementation |
| `docs/superpowers/plans/m[1-6]-*.md` | Implementation plans with TDD tasks + complete code |
| `docs/superpowers/plans/2026-07-14-llk-compiler-implementation.md` | Plan index + file map |
