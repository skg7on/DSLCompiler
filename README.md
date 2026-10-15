# DSLCompiler

[![CI](https://github.com/skg7on/DSLCompiler/actions/workflows/ci.yml/badge.svg)](https://github.com/skg7on/DSLCompiler/actions/workflows/ci.yml)
[![Coverage](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/skg7on/DSLCompiler/badges/badges/coverage.json)](https://github.com/skg7on/DSLCompiler/actions/workflows/coverage.yml)
[![Developer manual](https://img.shields.io/badge/docs-developer_manual-235c86)](https://skg7on.github.io/DSLCompiler/)
[![License](https://img.shields.io/badge/license-Apache--2.0%20WITH%20LLVM--exception-blue)](LICENSE)

DSLCompiler is an out-of-tree [MLIR](https://mlir.llvm.org/) project whose goal is to **design a generic MicroIR-based, mapping- and search-based DNN compiler framework**. It represents tile execution, maps computations onto machine resources, searches legal compilation choices, and lowers or analyzes the selected plan. **Matmul and fused SwiGLU are example operator kernels for testing the framework.** Intel AVX2 is the first executable validation target; additional targets extend machine descriptions, mapping policy and backend emitters.

Read the [framework goal and design specification](docs/superpowers/specs/2026-10-12-generic-microir-dnn-compiler-framework-design.md) for reusable contracts, capability-based acceptance and current genericity limits.

## Core concept

**Semantic IR describes the computation; Micro-IR describes its execution.** The canonical `micro` dialect makes tile dataflow, placement, movement, scheduling, and synchronization explicit. MachineModel YAML describes hardware resources and costs; LLKMap files describe target layouts and mapping rules.

```text
Tile = (shape, dtype, layout, memory_space, owner)
Micro-IR = tile dataflow + tile mapping + tile schedule
Predicted performance = F(Micro-IR, MachineModel, calibration)
```

One dialect has three conceptual layers: **uTile** for tile dataflow, **uMap** for placement and scheduling, and **uHW** for hardware-sized compute and transfer fragments. A concrete `micro.kernel` contains execution primitives; `micro.search_space` represents tunable choices.

## Architecture

```mermaid
flowchart TD
    LLK[Semantic adapters: current LLK examples] --> CPU[CPU validation: Linalg / Tensor scheduling]
    CPU --> Vector[Vector / bufferization / LLVM]
    Vector --> JIT[CPU ORC JIT]
    LLK --> Export[Supported LLK-to-Micro export]
    Export --> Micro[Micro-IR kernels and search spaces]
    Direct[Direct MicroIR input] --> Micro
    Micro --> Candidate[Original-source candidate instantiation]
    Candidate --> Map[Mapping and covering search]
    Policy[LLKMap layouts and rules] --> Map
    Machine[MachineModel YAML] --> Map
    Map --> Feasible[Complete-plan materialization / feasibility / scoring]
    Feasible --> Plan[Selected and bound execution plan]
    Plan --> Perf[Performance analysis]
    Machine --> Perf
    Plan --> Backend[Mapped target lowering / JIT]
```

The established CPU path uses MLIR tiling, fusion, explicit vectorization, and LLVM execution. The Micro-IR path currently exports LLK matmul and fused SwiGLU; a general Linalg-to-Micro importer is future work.

Mapping follows **extract → match → place → connect → cover → bind → verify**. Generic mapping and machine libraries stay target-neutral; target policy lives in `mapping/`, machine topology in `machines/`, and backend emitters in `lib/Target/`. See the [architecture guide](docs/manual/source/architecture.rst) and [mapping workflow](docs/design/micro-ir-mapping-workflow.md).

## Main features

- **Execution IR:** tile types and verifiers, temporal/spatial loops, pipelines, MMA/vector/reduction operations, layout transforms, explicit movement and synchronization.
- **Declarative mapping:** topology-aware placement and routing, single-operation and fused rules, deterministic/beam/bounded exact covering search, plan binding, reports, and replay.
- **Performance exploration:** static work/bandwidth bounds and dependency/resource scheduling; predicted cycles, traffic, utilization, bottlenecks, and capacity diagnostics.
- **Schedule exploration:** finite parameter spaces, original-source candidate instantiation, target-aware mapping, static ranking, frozen plan replay, and optional typed verified measurement.
- **CPU validation:** scheduling, fusion, packing, specialization, parallel runtime, and JIT infrastructure for supported kernels, including SwiGLU, RoPE, and attention. Selected AVX2 mapped GEMM/SwiGLU, padded-tail, transform and multi-output numerical fixtures are verified.
- **Target examples:** AVX2 and generic accelerator machine profiles and mapping packages.

[PR #133](https://github.com/skg7on/DSLCompiler/pull/133) delivers selected resource identity, capacity-feasible search, per-hop storage/lifetimes, shared static scoring, typed invocation with scratch release, selected vector/fused lowering, and original-source mapped tuning. Its pinned LLVM 22.1.8 Linux CI passes selected AVX2 execution without required skips. Three legacy JIT cases remain skipped; the [acceptance matrix](docs/reviews/issue67-final-acceptance.md) records exact evidence and limits. Performance predictions are uncalibrated, and the generic accelerator package has no hardware execution backend. The [feature matrix](docs/manual/source/features.rst) records current support and limits.

## Start understanding the project

1. Read the [concepts](https://skg7on.github.io/DSLCompiler/concepts.html), then the [architecture](https://skg7on.github.io/DSLCompiler/architecture.html) and [feature matrix](https://skg7on.github.io/DSLCompiler/features.html).
2. Follow the tutorials: [first kernel](https://skg7on.github.io/DSLCompiler/tutorials/01-first-kernel.html), [mapping](https://skg7on.github.io/DSLCompiler/tutorials/02-mapping.html), [performance](https://skg7on.github.io/DSLCompiler/tutorials/03-performance.html), and [tuning](https://skg7on.github.io/DSLCompiler/tutorials/04-tuning.html).
3. Explore `include/LLK/` and `lib/`: `Dialect/` defines the IR, `Conversion/` and `Transforms/` implement compilation, `Machine/` and `Mapping/` implement target exploration, and `Target/` supplies backend implementations. `tools/` provides the command-line entry points; `test/` shows executable examples.

The [developer manual](https://skg7on.github.io/DSLCompiler/) is the main documentation entry point, rebuilt from `main` on every merge. Its [sources](docs/manual/source/index.rst) and [build instructions](docs/manual/README.md) live in the repository. [ARCHITECTURE.md](ARCHITECTURE.md) defines the current framework architecture; the [historical CPU design](docs/design/legacy-cpu-architecture.md) covers the established validation pipeline in more depth.

## Build

You need Git, a C++20 compiler, CMake 3.20+, Ninja, Python 3, and a matching LLVM/MLIR development build with headers, CMake packages, TableGen, and FileCheck. CI pins LLVM **22.1.8**. See [setup instructions](docs/manual/source/getting-started.rst) for building LLVM with shared LLVM/MLIR libraries and `LLVM_BUILD_UTILS=ON`. Configure LLVM with `LLVM_BUILD_LLVM_DYLIB=ON`, `LLVM_LINK_LLVM_DYLIB=ON`, `LLVM_DYLIB_COMPONENTS=all`, and `MLIR_LINK_MLIR_DYLIB=ON`.

Use an isolated worktree, as required by the [repository policy](.claude/rules/worktree-isolation.md):

```bash
git clone https://github.com/skg7on/DSLCompiler.git
cd DSLCompiler
git worktree add .claude/worktrees/explore -b exp/explore
cd .claude/worktrees/explore

export LLVM_BUILD=/absolute/path/to/llvm-project/build
cmake -S . -B build -G Ninja \
  -DLLVM_PROJECT_BUILD_DIR="$LLVM_BUILD" \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLK_BUILD_TOOLS=ON -DLLK_BUILD_E2E_TESTS=ON
cmake --build build --parallel 4
cmake --build build --target check-llk --parallel 4
```

`LLVM_PROJECT_BUILD_DIR` is required and supplies all LLVM/MLIR package, header, tool, and library paths. The project links the monolithic shared `LLVM` and `MLIR` targets. Each worktree needs its own project build directory; the LLVM build can be shared. `check-llk` builds the registered tests and runs CTest.

Libraries are written to `build/lib`; tools and category test executables are written to `build/bin`. Test configuration lives under [`test/`](test/CMakeLists.txt), with GTest sources combined by category and individual cases labeled in CTest. For example, run mapping tests with `ctest --test-dir build -L Mapping --output-on-failure` or `build/bin/LLKMappingTests --gtest_filter='MappingPlan.*'`.

Use `-DBUILD_TESTING=OFF` to omit tests and GoogleTest, or `-DLLK_BUILD_E2E_TESTS=OFF` to exclude JIT execution tests. Configure emits `build/compile_commands.json` for editors unless `-DCMAKE_EXPORT_COMPILE_COMMANDS=OFF` is set.

## Use

Run these commands from the worktree root after building:

```bash
mkdir -p build/examples

# Parse and inspect a semantic kernel.
build/bin/llk-opt docs/manual/source/examples/matmul.mlir

# Export a concrete execution kernel and a tunable search space.
build/bin/llk-compile --emit=micro docs/manual/source/examples/matmul.mlir \
  > build/examples/matmul.micro.mlir
build/bin/llk-compile --emit=micro-search docs/manual/source/examples/swiglu.mlir \
  > build/examples/swiglu.search.mlir

# Estimate a concrete tile kernel against a machine profile.
build/bin/micro-perf --machine machines/x86-avx2-v2.yaml --level 1 \
  docs/manual/source/examples/gemm-tile.micro.mlir

# Tune the original semantic source, retaining replayable candidate plans.
build/bin/llk-tune --input=test/Tuning/Inputs/issue129/mapped_tune.mlir \
  --machine=machines/x86-avx2-v2.yaml -M=8 -N=16 -K=32 \
  --mapping-target=x86-avx2 --mapping-root=. --mapping-mode=exact \
  --mapping-backend=reference --candidate-source=semantic --source-symbol=matmul \
  --top-k=1 --mapping-report=build/examples/tune.json \
  --candidate-artifacts=build/examples/candidates --output=build/examples/schedules.yaml
```

Exporting IR does not execute a kernel. Target-aware tuning instantiates the selected original source and evaluates its mapped plan. The example uses the portable reference backend; use `--mapping-backend=selected-target` for compatible x86_64 AVX2 execution. Static ranking is uncalibrated; `--measure-top` with a typed `--measurement-inputs` fixture enables verified timing. Follow the [mapping tutorial](docs/manual/source/tutorials/02-mapping.rst) to select, bind, inspect, and replay a plan.

| Tool | Purpose and reference |
| --- | --- |
| `llk-opt` | [Parse, transform, map, and verify IR](docs/manual/source/tools/optimizer.rst) |
| `llk-compile` | [Export Micro-IR and use compilation pipelines](docs/manual/source/tools/compiler.rst) |
| `micro-perf` | [Analyze modeled performance](docs/manual/source/tools/performance.rst) |
| `llk-tune` | [Search and persist schedule candidates](docs/manual/source/tools/tuning.rst) |
| `llk-bench` | [Exercise a synthetic host timing harness](docs/manual/source/tools/benchmark.rst) |

## Contribute

Read [CLAUDE.md](CLAUDE.md) and the [contributor guide](docs/manual/source/contributing.rst). Create a worktree under `.claude/worktrees/` with a branch such as **feat/tile-operation**, **fix/mapping-route**, or **docs/tutorial-update**.

- Keep computation semantics, schedules, machine data, and target lowering separate. Add target policy through YAML/LLKMap and target emitters.
- For behavior changes, start with a reproducible failing test. Register GTest sources in the category `test/*/CMakeLists.txt` files and FileCheck tests in `test/FileCheck.cmake`; `// RUN:` comments are not executed by a lit runner.
- Run affected checks and `check-llk`, update relevant documentation, and report toolchain details and skips. Commit focused changes and open a PR explaining the problem, result, and validation.

## Future plan

- **Generalize workload and search contracts:** extend structured import and source-candidate schemas beyond the current M/N/K contraction template; use non-contraction fixtures to prove the extension.
- **Extend framework conformance:** broaden primitive, dtype, layout, movement/topology and target-emitter coverage while preserving the [PR #133 acceptance contract](docs/reviews/issue67-final-acceptance.md). Select operators to test reusable capabilities.
- **Add measured feedback and calibration:** persist measurements, fit cost models, and validate prediction error ([#51](https://github.com/skg7on/DSLCompiler/issues/51), [#52](https://github.com/skg7on/DSLCompiler/issues/52)).
- **Broaden frontend and target coverage:** expand semantic import and GPU/NPU/systolic/custom accelerator backends. A general Python frontend and direct Micro-IR emulator remain deferred.

Follow the [project tracker](https://github.com/skg7on/DSLCompiler/issues/41) for sequencing and status.

## License

The LLK Compiler Project is licensed under the **Apache License 2.0 with LLVM
Exceptions**. The full license text is in [LICENSE](LICENSE) and the LLVM
exceptions are in [EXCEPTIONS](EXCEPTIONS). Every source file carries the SPDX
identifier `Apache-2.0 WITH LLVM-exception`, checked in CI by
`scripts/check-license-headers.py`.
