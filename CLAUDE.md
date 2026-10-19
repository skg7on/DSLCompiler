# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Worktree Rules (HARD GATE)

See [.claude/rules/worktree-isolation.md](.claude/rules/worktree-isolation.md) for the mandatory worktree isolation policy. All writes require an isolated git worktree.

## Project Overview

**Project goal: design a generic MicroIR-based, mapping- and search-based DNN compiler framework.** Matmul and fused SwiGLU are example operator kernels used to test the IR, mapping, search and lowering contracts; their operator-specific assumptions must not define shared framework interfaces.

An out-of-tree MLIR compiler centered on a canonical DNN **Micro-IR** (`micro` dialect) for AI-chipset performance evaluation and auto-scheduling. Semantic/structured IR lowers into `micro`, a hardware-near execution IR modeling compute engines, memory hierarchy, data movement, spatial mapping, pipeline overlap, synchronization, and tunable schedule choices. Intel AVX2 CPU is the first executable validation backend and machine profile. The generic accelerator package supplies static mapping conformance. The [framework goal and design spec](docs/superpowers/specs/2026-10-12-generic-microir-dnn-compiler-framework-design.md) governs project scope; the feature matrix and revision-pinned reviews govern delivered support.

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
./bin/llk-opt --micro-map="target=x86-avx2 machine=machines/x86-avx2-v2.yaml layouts=mapping/x86-avx2/layouts.llkmap rules=mapping/x86-avx2/rules.llkmap emitters=avx2_vector_add,avx2_vector_convert,avx2_vector_silu,avx2_vector_mul,avx2_mma,avx2_reduce,avx2_copy,avx2_tile_copy,avx2_tile_store,avx2_fused_convert_silu_mul" input.mlir  # Search, bind best plan
./bin/LLKMappingTests --gtest_filter=MappingPlan.*  # Run selected GTest cases
```

Root `CMakeLists.txt` orchestrates `cmake/LLK*.cmake` (compiler, dependencies, build options, version, TableGen), `lib/`, `runtime/`, and `tools/`; each library directory declares its own targets; `test/CMakeLists.txt` and category subdirectories build one `LLK<Category>Tests` binary per category, with individual CTest cases. FileCheck/verifier registrations live in `test/FileCheck.cmake`; libraries go to `build/lib`, all executables to `build/bin`. `BUILD_TESTING=OFF` disables tests and GoogleTest; `LLK_BUILD_E2E_TESTS=OFF` omits JIT execution tests.

Configure writes a compilation database to `build/compile_commands.json` for clangd and other editors; pass `-DCMAKE_EXPORT_COMPILE_COMMANDS=OFF` to suppress it.

The compiler is the `clang` on `PATH` — on macOS that is Homebrew's clang, not the Apple toolchain at `/usr/bin/cc`. `CMakeLists.txt` drops an **absolute** `CC`/`CXX` that names a missing file (typically a `brew --prefix llvm@NN` path left behind after the keg was uninstalled) and prints an `Ignoring ...` status line rather than aborting. Bare names such as `CC=gcc` are left alone. Both the fallback and the compiler search apply only to a fresh build tree: a `build/` that already cached a compiler keeps it, so `rm -rf build` before reconfiguring whenever the toolchain changes.

`LLVM_PROJECT_BUILD_DIR` is mandatory. CMake loads LLVM/MLIR packages only from this build, replacing stale `LLVM_DIR`/`MLIR_DIR` cache entries. Header paths (including source headers outside the build tree) come from those packages; TableGen and FileCheck come from that build's exported tool directory (including configuration-specific paths). Link LLVM/MLIR through the monolithic shared `LLVM` and `MLIR` targets, never LLVM component archives. Configure LLVM with `-DLLVM_BUILD_LLVM_DYLIB=ON -DLLVM_LINK_LLVM_DYLIB=ON -DLLVM_DYLIB_COMPONENTS=all -DMLIR_LINK_MLIR_DYLIB=ON`.

CI builds the tools, E2E tests, and the `FileCheck` utility from the same pinned LLVM source, so the full registered suite runs. Local LLVM builds must use `-DLLVM_BUILD_UTILS=ON`; configuration fails when `LLK_BUILD_TOOLS=ON` and a matching `FileCheck` is unavailable. `check-llk` is a thin wrapper over CTest, so it runs exactly the registered suite — run it locally before trusting a change.

## Python frontend environment and tests

Use the project-level `.venv` for Python frontend work, rather than an ambient
Python or a pipx application environment. When working in an isolated worktree,
point `LLK_PYTHON` at the existing project `.venv/bin/python` with an absolute
path; keep source edits and generated artifacts in that worktree. An environment
at the root checkout can be shared for these Python checks; each compiler build
still belongs to its own worktree.

```bash
# Run from the isolated worktree root; adjust this interpreter path.
export LLK_PYTHON=/absolute/path/to/DSLCompiler/.venv/bin/python
"$LLK_PYTHON" -m pip install -e './python[test]'
"$LLK_PYTHON" -m pytest
"$LLK_PYTHON" -m pytest test/python/test_torch_registration.py test/python/test_package_install.py -q
```

The current local snapshot (2026-10-19) is Python 3.14.8, Torch 2.14.1,
NumPy 2.5.4, pytest 9.1.1, build 1.6.1, setuptools 84.0.0 and wheel 0.48.0.
Torchvision 0.29.1 and torchaudio 2.11.0 also import successfully but are optional
for PF01. The full PF01 suite passes 21/21 in this project `.venv`. Preserve the
separate Python 3.11/3.12 + Torch 2.9.0 reference matrix; use `[test]` with the
current environment and `[torch-test]` only when selecting that reference Torch
version. The checked-in compatibility manifest records the reference matrix,
not every installed environment.

PF01 delivers packaging, optional imports, stable registration, ownership checks,
structured errors and truthful capability reporting. `llk.torch.get_backend()`
is the registered singleton; importing/reloading must not reset global Dynamo
state or replace another owner's backend. Native execution raises
`UnsupportedFeatureError(code="native_backend_unavailable")`; actual
`torch.compile` wraps it in `BackendCompilerFailed.inner_exception`. Do not count
registration or eager execution as LLK compilation. Typed capture/native JIT,
Inductor, training and real upstream Triton remain later tasks under #203.

`pytest.ini` restricts default discovery to `test/python` and adds the local
`python/` package path. Do not run vendored GoogleTest Python tests as project
frontend acceptance. The package tests build/install a real wheel in a clean
Torch-absent environment; registration tests require real Torch and must not
skip. Python frontend tests are separate from `check-llk`; required native/CI
integration remains PF20. See the [frontend guide](docs/manual/source/python-frontend.rst)
and [PF01 environment evidence](docs/reviews/2026-10-19-pf01-python-environment.md).

## Architecture

**Primary framework path:** semantic/structured input or directly authored MicroIR → source/schedule candidates → workload graph → mapping/covering search → complete-plan materialization and shared feasibility/analysis → selected bound MicroIR → target lowering/execution or performance reports. Current LLK semantic export covers matmul and fused SwiGLU; general structured/Linalg import remains future work.

**Two implemented integration paths:**

1. **CPU validation path** — Linalg → tiled/fused (scf.forall) → Vector SIMD → memref → CF+runtime → LLVM → ORC JIT → AVX2
2. **Micro-IR path (M9+)** — `LLKToMicro` → `micro.search_space` (parametric) or concrete `micro.kernel` → target-aware mapping / shared static analysis / selected or reference lowering / ORC JIT

**Micro-IR conceptual layering** (one dialect, one `micro` namespace):
`uTile` (tile dataflow) → `uMap` (memory placement, owner mapping, pipeline, sync) → `uHW` (instruction fragments, machine-visible events)

**Primary `micro` value is a tile:** `Tile = (shape, dtype, layout, memory_space, owner)`. Four cost classes: *logical* (zero-cost views/slices/partitions), *memory* (materialized), *execution* (owner-mapped), *instruction fragment* (MMA/vector/DMA granularity). Concrete `micro.kernel` must contain no DNN semantic ops.

**Key framework rules (from ARCHITECTURE.md):**
1. Separate semantic computation, schedule/search choices, machine facts and target lowering.
2. Keep canonical MicroIR tile/dataflow semantics independent of workload and target names.
3. Use semantic adapters and primitive coverage to extend workloads; shared mapping/analysis must not branch on `swiglu` or `matmul` labels.
4. Keep target policy in machine profiles, LLKMap and target emitters; generic machine/mapping code never branches on target names.
5. Materialize complete plans and check storage/capacity with shared static analysis before top-K retention; executable compilation also verifies backend realizability.
6. Preserve original-source semantics, concrete resource identity, numerical contracts and frozen replay identity across search and lowering. Numerical approximation is explicit IR semantics (e.g. `#llk.math_mode`), not a compiler-flag side effect.
7. Record deterministic search bounds and truncation. Keep predicted metrics separate from verified measurement; missing measured metrics never rank as zero.
8. Validate framework capabilities with contraction, non-contraction, fusion, movement, topology and extension fixtures; matmul/SwiGLU alone do not establish general DNN support.
9. Treat M/N/K, `BM/BN/BK`, contraction arm counts, five M-buckets and rank-two invocation as current adapter/runtime limits rather than generic framework requirements.
10. For the existing CPU validation path, reuse Linalg tensor tiling/fusion before bufferization, emit explicit Vector operations, and measure packing/dispatch overhead. Those CPU rules do not mandate a Linalg-first route for every future MicroIR input.

**Custom dialects:** `llk` is small — only retains domain info unavailable in generic MLIR (`llk.fused_swiglu`, `llk.rope`, `llk.attention`); never recreate tensor/linalg/vector/memref. `micro` (`include/LLK/Dialect/Micro/`) is the tile-centric execution IR. Implemented: `!micro.tile`, `#micro.layout`, `#micro.owner`, `#micro.memory`; ops `micro.kernel`, `micro.{tile_view,tile_alloc,tile_partition,tile_async_copy,tile_store}`, `micro.{mma,vector,reduce}`, `micro.{for,spatial_for,pipeline,alloc,async_copy,wait,store}`, and `micro.transform` (a layout conversion naming each side by its affine map, not a target layout id). Search-space ops (`micro.search_space`, `micro.param`, `micro.constraint`, `micro.objective`, `micro.candidate`) are implemented (#44). A bound plan adds target-neutral metadata attributes to the kernel: `micro.plan`, `micro.mapping`, `micro.routes`, plus `micro.value`/`micro.dst_node` on each emitted copy.

**Mapping subsystem (epic #67):** target policy is declarative and consumed by generic code. Libraries: `LLKMachine` (`include/LLK/Machine/` — v2 topology: executor/memory/compute/transfer nodes with `contains`/`dominates`/`attached_to`/`link` edges) and `LLKMapping` (`include/LLK/Mapping/` — workload graph, candidate/instance/connection/covering, routing, LLKMap, placement, covering search, plan binder, cost/latency/diagnostics/report). Target packages live in `lib/Target/<Target>/Mapping/` (`LLKTargetMapping`) with policy in `mapping/<target>/{layouts,rules}.llkmap`; machines are `machines/*-v2.yaml`. Generic `lib/Mapping`/`lib/Machine` code must never branch on a target name — policy belongs in the `.llkmap`/`.yaml` files.

**Gotcha — `MicroEnums.h` is hand-written:** the CMake tablegen config does not run `-gen-enum-decls/-gen-enum-defs`. Adding an enum in `MicroDialect.td` requires a matching hand-written `stringify*`/`symbolize*` entry in `include/LLK/Dialect/Micro/MicroEnums.h`. The mapping diagnostics enum (`include/LLK/Mapping/Diagnostics.h`) follows the same hand-written pattern, as do the MachineModel v2 kind/direction stringifiers.

**Gotcha — adding a MachineModel field:** give it a default member initializer (e.g. `std::vector<std::string> equivalentTo{};`) or every aggregate test literal emits `-Wmissing-field-initializers`. The loader preserves the minor version (`MachineModel::schemaMinor`) and tolerates *unknown keys only when the file declares a minor greater than `kSupportedSchemaMinor`*; missing required keys, wrong node types, malformed values, and unknown kinds are always rejected. `schemaMinor` is excluded from the content hash.

**Gotcha — compiler version is generated:** `cmake/LLKVersion.cmake` does `configure_file(include/LLK/Version.h.in → ${CMAKE_BINARY_DIR}/include/LLK/Version.h)` producing `LLK_COMPILER_VERSION` from the project version plus `git describe`. The plan report prints it; there is no hand-written version constant.

**Schedules and search choices are data:** persistent MicroIR search spaces/bindings and selected plans are the framework contract. The existing CPU contraction adapter also uses Transform dialect `.mlir` files or `schedule_db.json`, keyed by `(operation, M_bucket, N, K, dtype, ISA, math_mode)` with five M-buckets: {1}, [2,4], [5,16], [17,64], ≥65. These fields are example-specific; future workloads need capability-appropriate parameter schemas.

## Conventions

- C++20, CMake ≥ 3.20, LLVM/MLIR 20+, GTest, FileCheck
- Naming: `llk` dialect prefix, `llk-*` tools, CamelCase passes, snake_case files
- Error handling: MLIR `emitError()` for verifier failures, `llvm::Expected<T>` for JIT ops
- ABI: the established CPU wrapper uses `Tensor2D`/`KernelContext`; mapped execution uses typed `InvocationBuffer2D` carrying a `MemRef2D` descriptor, dtype and backing allocation size. Prefer the typed invocation API; the deprecated untagged descriptor overload cannot check dtype or allocation bounds. Caller buffers remain caller-owned; compiler scratch is released per invocation.
- TDD: every task starts with a failing test, then minimal code; commit per task
- FileCheck tests are plain `add_test` entries in `test/FileCheck.cmake` — there is no lit runner, and `// RUN:` lines are comments only. Register new `.mlir` tests by hand with `add_llk_filecheck_test(Name test/Dialect/Micro/foo.mlir)` (an optional trailing argument is passed through to the tool, e.g. pass options), or use a raw `add_test` with `--verify-diagnostics --split-input-file` for invalid-IR tests
- Mapping/target boundary: generic `lib/Mapping` + `lib/Machine` stay target-neutral; target policy is data (`mapping/*.llkmap`, `machines/*.yaml`). `LLKMapping` links `LLKMachine` plus shared `LLVM`/`MLIR`; `LLKBuildOptions` matches the LLVM build's RTTI/exception settings for all project targets

## Milestone Sequence

Historical milestone numbers record delivery order. Future goals are framework capabilities: broader workload/import and search contracts, explicit modeling coverage, primitive/target extension, analysis parity, measured feedback/calibration and wider runtime integration as required. Choose example operators to test those capabilities.

M1–M6 complete (CPU pipeline, AVX2 vector, fused memory, parallel dispatch, specialization/tuning, RoPE + Attention). Python frontend PF01 packaging/registration is implemented in PR #233; executable frontend milestones remain tracked by #203.

**M9–M12 and mapped acceptance:** PR #133 merged on 2026-10-11 at `5ee61b193d85b873497e9a7aecbd07f1fea73c55`. It implements the #129 resource, execution and tuning repair series: concrete selected resource identity, feasibility before top-K, per-hop storage/lifetimes, per-engine DMA scheduling, selected-plan/materialized static parity, typed invocation and scratch ownership, explicit selected-width/fused lowering, and original-source mapped tuning with frozen replay.

**Verification:** final PR head `d038fed` passed pinned Linux static LLVM/MLIR 22.1.8 CI run `38126440190`. Implementation-head artifacts record selected AVX2 13/13 without skips and full CTest 1,215 registered / 1,211 passed / zero failed / three legacy JIT skips / one disabled test. Seven selected-plan manifests capture exact LLVM IR, target/ABI/plan/machine identity and zero reference groups. Counts depend on registration/toolchain; retain the exact revision and skip reasons. See `docs/reviews/issue67-final-acceptance.md` for the twelve-criterion matrix and historical snapshots.

**Supported boundary:** selected-target execution requires compatible x86_64 AVX2; reference execution is a separate portable correctness mode. Generic-accelerator coverage is static conformance, not hardware execution. Legacy JIT skips remain documented; production persistence/calibration (#51/#52), general frontend coverage and the directly interpreting Micro emulator (#54) remain separate.

**Mapped implementation guidance:** `LLKMapping` remains independent of perf/runtime/JIT libraries. Complete-plan integration finalizes materialization and shared analysis before retention. `compileMappedKernel` is shared by mapped compiler/tuner workflows. Runtime LLVM lowering registers dialect conversion interfaces and uses unified ConvertToLLVM after structural/vector lowering; keep the unit-row transfer rewrite guarded by its shape, index, mask and in-bounds checks. Source-candidate and measurement identities must preserve full content, selected plan, machine, ABI and execution mode; missing measured metrics cannot rank as zero.

## Key Files

| File | Purpose |
|------|---------|
| `ARCHITECTURE.md` | Current generic framework architecture, components, mapping/search and extension rules |
| `docs/superpowers/specs/2026-10-12-generic-microir-dnn-compiler-framework-design.md` | Authoritative project goal, subsystem contracts, current genericity gaps and framework acceptance |
| `docs/superpowers/plans/2026-10-12-generic-microir-framework-implementation.md` | Current-code audit and detailed capability increments for analysis coverage, workload/search, conformance and measurement validation |
| `docs/design/legacy-cpu-architecture.md` | Historical CPU validation design and M1–M7 rationale |
| `docs/design/m[1-6]-*.md` | Per-milestone design specs (MLIR defs, algorithms, tests) |
| `docs/design/m9-canonical-micro-ir-architecture.md` | Micro-IR project-level redesign, layer responsibilities |
| `docs/design/m9-micro-ir-core-concepts.md` | `micro` dialect semantics: tile model, op families, attrs, verifier rules |
| `docs/superpowers/specs/2026-08-13-micro-ir-tile-programming-model-spec.md` | Detailed tile programming model |
| `docs/superpowers/specs/2026-09-18-microir-inspired-dslcompiler-enhancement-design.md` | Epic #67 design: mapping engine, MachineModel v2, routing, LLKMap, covering search, plan binding (§-numbered throughout the code) |
| `docs/design/micro-ir-mapping-workflow.md` | Contributor workflow: extract → match → place → connect → cover → bind → verify |
| `docs/reviews/issue67-final-acceptance.md` | Revision-pinned PR #133 §29 evidence, historical snapshots, capability limits and the #51/#52 handoff |
| `docs/reviews/2026-10-07-issue67-current-gap-assessment.md` | Historical pre-#133 assessment of the ten #129 gaps; use the final acceptance matrix for current evidence |
| `docs/design/llkmap-layout-grammar.md`, `docs/design/llkmap-rule-grammar.md` | LLKMap declarative grammar: layouts, rules, target bundles |
| `include/LLK/Machine/` + `lib/Machine/` | `MachineModel` v2 topology, loader, content hash |
| `include/LLK/Mapping/` + `lib/Mapping/` | Mapping core: workload graph, routing, LLKMap, rules, placement, covering search, plan binder, cost/latency/diagnostics/report |
| `include/LLK/Target/*/Mapping/` + `lib/Target/*/Mapping/` | AVX2 and generic-accelerator mapping targets (`LLKTargetMapping`) |
| `mapping/<target>/{layouts,rules}.llkmap` | Declarative target policy (layouts + mapping rules + bundles) |
| `machines/*-v2.yaml` | MachineModel v2 topology profiles |
| `docs/superpowers/plans/2026-10-03-micro-ir-gap-closure.md` | Gap-closure plan (31 TDD tasks) |
| `docs/superpowers/plans/2026-10-07-issue129-gap-closure.md` | Completed #133 repair plan and R/E/T verification ledger |
| `include/LLK/Dialect/Micro/` + `lib/Dialect/Micro/` | `micro` dialect implementation |
| `docs/superpowers/plans/m[1-6]-*.md` | Implementation plans with TDD tasks + complete code |
| `docs/superpowers/plans/2026-07-14-llk-compiler-implementation.md` | Plan index + file map |
