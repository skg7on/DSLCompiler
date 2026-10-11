# Issue #138: Operator-Kernel Lowering Framework Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans to implement this plan task by task. Steps use checkbox syntax for tracking.

**Goal:** Move operator knowledge into explicitly registered kernel implementations while preserving existing conversion IR and mapped execution.

**Architecture:** LLKKernel owns an ordered string-keyed registry, contracts, and one translation unit per LLK operator. Conversion and transform passes consume registry capabilities. LLKToMicro owns target-neutral schedule resolution and kernel assembly; LLKKernel never depends on that driver or runtime/JIT libraries.

**Tech stack:** C++20, MLIR static libraries, CMake, GTest, FileCheck, CTest.

**Spec:** `docs/superpowers/specs/2026-10-11-conversion-kernel-lowering-framework-design.md` (merged through #139).

**Reviewed baseline:** `fb99595`. Issue #138 has no comments at review time.

## Global constraints

- All writes in `.claude/worktrees/issue138-kernel-framework`, branch `refactor/kernel-lowering-framework`.
- Explicit populate anchor; no static registration, RTTI, exceptions, or stored OperationName keys.
- Preserve collect-before-mutate, registration order, sink gating, and original semantic functions during Micro export.
- Preserve source-root ordinals and the public exportMicroKernelFromSchedule API used by mapped compilation/tuning.
- Keep existing tests unchanged; capture actual textual IR before migration in addition to FileCheck assertions.
- Use modern static Op::create APIs and mlir_target_link_libraries for MLIR link dependencies.
- T1–T9 each begin with a failing test and end with a separate commit plus check-llk.
- D1: implement Matmul Linalg lowering. D2: pass contract.K to schedule selection; do not add N/K filtering.

## Review findings and necessary spec amendments

1. **The supported root set has grown:** LLKToMicro.cpp supports linalg.matmul in both passes and the public schedule export. Preserve this with a LinalgMatmul adapter registered explicitly beside Matmul; it exposes Micro capability only. Transforms must retain their LLK-only scope by filtering the `llk` namespace before lookup. The spec's four-kernel list is outdated.
2. **Matmul init is observable:** current kernels have `(lhs, rhs, init)` arguments and stage init into the accumulator. Add `bool hasOutputInit = false` to MicroKernelFragment. Matmul adapters set true; SwiGLU sets false. Do not infer this from armCount or workload strings.
3. **Callback lifetime:** function_ref cannot own a lambda captured locally by describeMicroKernel. Use `std::function<Value(OpBuilder &, Location, ArrayRef<Value>)>` so captured math mode survives the hook's return. No RTTI or exceptions are needed for normal callable use; verify compilation under both disabled flags.
4. **Dynamic transform behavior:** shape specialization currently emits a generic bucket remark; schedule selection emits a warning and skips. Micro rejects dynamic shapes. Separate descriptive inference from Micro eligibility validation: contracts may describe dynamic dimensions with mBucket=-1, and describeMicroKernel performs strict export checks before any IR write. This keeps transform diagnostics independent of Micro-only restrictions.
5. **Ordering:** Linalg currently lowers all SwiGLU roots, then RoPE, then Attention. A single walk followed by immediate dispatch changes insertion/order behavior in mixed modules. Collect by registry entry order while preserving walk order within each entry.
6. **IR equality:** FileCheck only asserts selected lines, not byte equality. Store canonical printed output before refactoring and compare complete output after T3–T7 with the same executable/toolchain and input paths.

These amendments require review before implementation because they change the written interface rather than merely selecting implementation details.

## Review focus

- Nonzero Matmul init must survive both exports and execution.
- Mixed LLK/Linalg roots must keep exact source ordinal selection and symbol suffixes.
- Dynamic-shape transforms must avoid newly emitted Micro errors.
- Captured epilogue math mode must remain valid after description returns.
- Static archive linking must retain every populate anchor and all dialect dependencies.

## Setup and regression capture

- [ ] Configure a new worktree-local build; do not reconfigure the main checkout build:

```sh
cmake -S . -B build -G Ninja -DLLVM_PROJECT_BUILD_DIR=/Users/skg7on/Workspace/Projects/llvm-project/build -DCMAKE_BUILD_TYPE=Debug
cmake --build build --target check-llk
```

- [ ] Record toolchain/revision, test totals and skip reasons. Save complete llk-opt output for each conversion regression fixture in ignored worktree-local verification storage. Record exact commands and compare stdout and diagnostic stderr separately.

## T1 — Ordered registry and capability contract

**Files:** create `include/LLK/Conversion/LLKKernel/LLKKernelLowering.h`, `lib/Conversion/LLKKernel/Registry.cpp`, `test/Conversion/LLKKernel/registry.cpp`; modify root and test/Conversion CMakeLists.

**Interfaces:** getOpName, inferContract, hasLinalgLowering/lowerToLinalg, hasMicroLowering/describeMicroKernel; registry add<T>, lookup(StringRef), entries().

- [ ] Add GTests with dummy lowerings: unknown lookup returns nullptr, default sink flags are false, insertion order is stable, pointers survive subsequent additions, duplicate registration has a documented rejection policy.
- [ ] Build LLKConversionTests and observe missing-interface failure.
- [ ] Implement owning vector of unique_ptr plus StringMap lookup and ordered pointer view. Declare Contract/Fragment interfaces from §4.2 with the amendments above. Start with empty populate.
- [ ] Run registry tests and check-llk; commit T1.

## T2 — Shared contraction facts and strict export validation

**Files:** create `lib/Conversion/LLKKernel/Contract.cpp`, `test/Conversion/LLKKernel/contract.cpp`; update header and library sources.

**Interfaces:** inferContractionContract(Operation*, Value lhs, Value rhs, Value out, Attribute accumulatorType, KernelContract&), plus explicit Micro validation invoked by description hooks.

- [ ] Parse synthetic operations in GTests; test M/N/K, input/output/accumulator types and each bucket boundary (1,2,4,5,16,17,64,65).
- [ ] Test rank mismatch, contraction/output mismatch, dtype mismatch, invalid accumulator type, and dynamic shape description versus export rejection. Capture diagnostic text.
- [ ] Observe missing-helper failures; extract current validation without changing Micro diagnostic wording. Keep strict validation separate from transform description.
- [ ] Run contract tests, LLKToMicroLoweringInvalid and check-llk; commit T2.

## T3 — SwiGLU Linalg registration

**Files:** create `lib/Conversion/LLKKernel/SwiGLU.cpp`; modify Registry.cpp and LLKToLinalg.cpp.

- [ ] Add a populate test asserting SwiGLU lookup and Linalg capability; observe failure against empty populate.
- [ ] Move lowerSwiGLU verbatim, replacing pass failure signaling with LogicalResult propagation and getContext with builder.getContext.
- [ ] Register SwiGLU explicitly. Preserve remaining legacy RoPE/Attention dispatch until T4; collect all roots before lowering.
- [ ] Compare full baseline output; run LLKToLinalgConversion, SwiGLU execution cases and check-llk; commit T3.

## T4 — RoPE and Attention registration

**Files:** create `LLKKernel/RoPE.cpp`, `Attention.cpp`; modify Registry.cpp and LLKToLinalg.cpp.

- [ ] Add capability/contract tests: both have Linalg capability, neither has Micro capability; scheduling M is dim0 and N is dim3 to preserve current semantics. Pin contract K explicitly without changing schedule matching.
- [ ] Observe unregistered lookup failures. Move existing lowering bodies unchanged and provide per-op contracts.
- [ ] Finish generic collect-and-dispatch Linalg pass using ordered registry entries; propagate lowering failures.
- [ ] Compare mixed-operator textual output, RoPE/Attention and Triton regressions; run check-llk; commit T4.

## T5 — Micro driver and operator fragments

**Files:** create `LLKToMicro/MicroKernelDriver.h`, `.cpp`, `LLKKernel/Matmul.cpp`, `LinalgMatmul.cpp`; modify SwiGLU.cpp, Registry.cpp, LLKToMicro.cpp and CMake.

- [ ] Add tests for two-arm SwiGLU, one-arm Matmul, owned callback lifetime/math mode and hasOutputInit. Observe absent Micro capability failures.
- [ ] Add exact textual export tests for nonzero init and linalg.matmul compatibility, plus mixed-root source ordinal and duplicate-name suffix tests. Verify invalid ordinal/schedule leaves the module byte-identical.
- [ ] Move schedule parsing, TilePlan, resolvePlan and kernel assembly unchanged into driver. Replace every fused-dependent branch with contract.workload, fragment.armCount, rhsOperandTypes, hasOutputInit and buildEpilogue.
- [ ] Keep structural conversion, padding, staged init copies and accumulator-to-output conversion unchanged. Validate fragments before emission.
- [ ] Register both Matmul source adapters explicitly; registry only supplies facts/fragments, never calls driver functions.
- [ ] Convert concrete pass and public schedule export to shared capability-gated lookup and identical traversal order.
- [ ] Compare complete output; run SwiGLU/Matmul/perf roundtrip/invalid export regressions and mapped acceptance, then check-llk; commit T5. Review this diff separately before T6.

## T6 — Registry-driven search spaces

**Files:** LLKToMicro.cpp and search-space regression tests.

- [ ] Add mixed supported/unsupported-root test showing capability gates and matching concrete/search-space workload identities; observe unsupported registration behavior before conversion.
- [ ] Replace search-space root literals and inference with registry lookup/contracts, invoking the same strict export validation before building search spaces.
- [ ] Preserve schedule loader, candidate/domain ordering, M bucket semantics and source functions.
- [ ] Compare full output and run SearchSpace, SearchSpaceMBucket and micro-search compile tests; run check-llk; commit T6.

## T7 — Transform consumers

**Files:** ShapeSpecialization.cpp, ScheduleSelection.cpp, Common/ScheduleLoader.cpp, transform fixtures and FileCheck registrations.

- [ ] Add remarks tests for Matmul, RoPE and Attention; add dynamic SwiGLU tests preserving existing generic remark/warning behavior. Observe missing generalized remarks before migration.
- [ ] Replace LLK op casts with namespace-filtered registry lookup and descriptive contracts. Delete private classifyM; use contract.mBucket.
- [ ] Pass contract.N and contract.K to both loader and selector; document that neither is currently used as a filtering key.
- [ ] Confirm preexisting warnings/remarks and static shape fixtures unchanged; run shape_specialization, schedule_selection, schedule_fallback and check-llk; commit T7.

## T8 — Matmul Linalg lowering

**Files:** Matmul.cpp, LLKOps.td description, `test/Conversion/LLKToLinalg/matmul_to_linalg.mlir`, test/FileCheck.cmake.

- [ ] Register FileCheck fixture matching linalg.matmul with the original A/B/init operands, original result type and no remaining llk.matmul. Include nonzero init and supported mixed element/accumulator types.
- [ ] Observe CHECK failure against pass leaving llk.matmul intact.
- [ ] Enable Matmul Linalg hook using linalg::MatmulOp::create, replace result uses, erase root. Preserve destination initialization and numerical semantics; do not introduce zero fill.
- [ ] Correct ODS description to describe actual behavior. Run fixture and numerical Matmul execution check plus check-llk; commit T8.

## T9 — Dependency and acceptance audit

**Files:** root CMakeLists, acceptance ledger beside this plan, PR description.

- [ ] Build static archives with LLKKernel compiled under -fno-rtti -fno-exceptions and -Werror=deprecated-declarations; verify operator registration through a linked test binary.
- [ ] Inspect dependencies: no LLKKernel→LLKToMicro/runtime/perf/JIT edge; four consumer libraries explicitly link LLKKernel using existing conventions.
- [ ] Audit operator-type literals in pass bodies, obsolete RootInfo/inferRootInfo duplication, deprecated APIs and registration ownership/order. Operator implementations are allowed to name their ops.
- [ ] Run final check-llk and report all skips/failures with exact revision/toolchain, then git diff --check.
- [ ] Review full diff against every acceptance criterion; record amendments and remaining limitations.
- [ ] Commit T9; push branch; open PR referencing #138 and attach it to this chat. Wait for pinned Linux CI/static-link acceptance before claiming completion.

## Execution ledger

Plan written after reviewing the merged spec and current root export/transform implementations. No production edits or test runs have occurred yet. Implementation awaits review of the interface amendments above.
