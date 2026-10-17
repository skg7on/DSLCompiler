# Dynamic Shape LLK/MicroIR Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Compile fully dynamic M/N/K contractions into reusable, checked MicroIR kernels and optimize them without losing static resource or numerical guarantees.

**Architecture:** Runtime shape symbols govern loop bounds and boundary lengths. Fixed-capacity tiles and complete selected plans govern resources; versioned contracts connect semantic adapters, invocation, analysis and replay.

**Tech Stack:** C++20, MLIR/LLVM, CMake, GTest, FileCheck; existing reference and AVX2 mapping backends.

**Spec:** [Dynamic shapes design](../specs/2026-10-17-dynamic-shape-llk-microir-design.md).

**Status:** Draft implementation sequence for design review. No product code is authorized or changed by this document. Baseline `45560cbdebe452dbec2ae1d5859f43bb868cf4aa`.

## Global Constraints

- Fully dynamic contraction dimensions, including M, N and K, are the first functional milestone. Fixed weights or dynamic M alone are insufficient.
- Local materialized tiles, instruction shapes and maximum concurrent tile count are compile-time bounded. Unbounded dynamic local allocation is rejected.
- Preserve source accumulator types, output conversion points, math mode, initialization and reduction semantics.
- No workload-name branches in shared mapping/analysis; no target-name branches in generic machine/mapping code. `LLKMapping` remains independent of runtime/JIT/perf libraries.
- Static version-1 artifacts remain readable and retain their identity. Symbolic contracts and ABI v2 have explicit versions and distinct hashes.
- Mapped execution initially stays rank two, compact row-major, offset zero, at most 12 descriptors, with read-only inputs and disjoint caller-owned outputs.
- Use C++20, modern MLIR APIs, local LLVM 24+ and pinned CI LLVM 22.1.8. No dependency on a new external shape solver.
- Work only in an isolated worktree. Register all CTest/FileCheck fixtures explicitly; `// RUN:` comments do not execute tests.

## Review Focus

- Mixed static/dynamic dimensions must be related by symbols and checked before access, including on empty invocations (Tasks 1, 4, 6).
- Small tensors and independent M/N/K tails must never cause a full-width external overread/write (Tasks 2, 5, 6).
- K=0, nonzero initialization, signed zero and nonfinite inputs must preserve numerical semantics (Tasks 2, 5, 6).
- Large external tensors must not create whole-output SRAM storage or multiply residency by all logical tiles (Tasks 3, 4, 5).
- Reusing code across shapes must not reuse incompatible measurements, packed weights or replay guards (Tasks 7, 8).

## File and dependency map

New low-level `LLKShape` (`include/LLK/Shape/`, `lib/Shape/`) owns plain shape data, deterministic encoding and checked resolution. It depends on LLVM Support, not MLIR, Perf or Runtime. Existing `LLKWorkloadContracts` owns MLIR attribute bridging. Runtime and generic mapping may depend on `LLKShape`; no dependency points back to adapters or JIT.

Existing files below are modified in place unless marked Create. CMake/source registration belongs to each task. New tests are grouped into `DynamicShapeContractTest`, `DynamicShapeMappingTest`, `DynamicShapeAnalysisTest`, `DynamicMappedAcceptance` and `DynamicMappedAVX2Acceptance` where standalone binaries simplify CTest registration. Use existing category binaries for adjacent verifier/conversion fixtures. All paths are repository-relative.

Public interface names below are proposed and must be introduced by the producing task before consumers migrate. `shape::` abbreviates the plain-data namespace `::llk::shape`. `Dimension` is a tagged constant/symbol value; construction rejects invalid values through contract validation. `ShapeValues` is `std::map<std::string,int64_t>`. No consumer passes MLIR's dynamic sentinel into numeric shape arithmetic.

### Task 1: Symbolic contracts and semantic guards

**Files:** Create `include/LLK/Shape/ShapeContract.h`, `lib/Shape/{CMakeLists.txt,ShapeContract.cpp}`, `test/Tuning/dynamic_shape_contract.cpp`; modify `lib/CMakeLists.txt`, `test/Tuning/CMakeLists.txt`, `include/LLK/Conversion/Workload/WorkloadContract.h`, `lib/Conversion/Workload/{WorkloadContract.cpp,ContractionSourceAdapter.cpp}`, `lib/Conversion/Workload/CMakeLists.txt`, `include/LLK/Conversion/LLKKernel/LLKKernelLowering.h`, `lib/Conversion/LLKKernel/{CMakeLists.txt,Contract.cpp,SwiGLU.cpp,Matmul.cpp,LinalgMatmul.cpp}`, `lib/Dialect/LLK/LLKOps.cpp`; migrate static construction/serialization users and their tests to the tagged `Dimension` representation in the same commit.

**Interfaces:** Produce `shape::Dimension`, `shape::ShapeContract`, `shape::ShapeValues`; `llvm::Error validateShapeContract(const ShapeContract &)`, `llvm::Expected<ShapeValues> bindShapeValues(const ShapeContract &, llvm::ArrayRef<PortShapeValues>)`, `std::string canonicalShapeContract(const ShapeContract &)`. `PortShapeValues` identifies port direction/index and a concrete extent vector. `ShapeContract` contains port dimensions and symbol constraints; `Dimension` holds either a nonnegative `int64_t` or a nonempty symbol name. Produce `llk::ContractionDimensions` and `FailureOr<ContractionDimensions> inferContractionDimensions(Operation *)` through the existing kernel registry; M/N/K are adapter facts. Workload v2 uses `Dimension` port/axis values, while its v1 serializer accepts only positive constants and preserves old bytes.

- [ ] Add tests `ShapeContract.{AllDynamicMNK,MixedStaticEquality,RejectConflicts,EmptyStillChecksK,RejectNegative,Version1IdentityStable,Version2RoundTrip,ElementwiseAxes}`. Assert X[7,5], Wg[5,13], Wu[5,13], out[7,13] binds `{M:7,N:13,K:5}`; Wu[6,13] fails; M=0 does not suppress the failure; v1 golden renderings/hashes remain equal to baseline.
- [ ] Register `DynamicShapeContractTest`; run `ctest --test-dir build -R '^DynamicShapeContractTest$' --output-on-failure` and confirm new cases fail before implementation.
- [ ] Implement shape data/validation and workload v2 encoding, then add adapter dimension inference. Static-known contradictions remain compile-time errors; unresolved relationships become contract entries. Retain legacy integer `KernelContract::M/N/K` for the still-static export until Task 4; only fill/use them after static resolution, and retain the old dynamic-export rejection during this intermediate task. This keeps the commit buildable without sending unknown extents into the old driver. Migrate all static workload-contract construction/serialization consumers to tagged constant dimensions.
- [ ] Build `DynamicShapeContractTest FrameworkSourceTest LLKConversionTests`; run the new test plus `ctest --test-dir build -R 'DynamicShapeContractTest|FrameworkSourceTest|LLKKernel' --output-on-failure`. All must pass, including legacy v1 tests.
- [ ] Commit as `feat: add versioned symbolic shape contracts`.

### Task 2: Bounded movement and exact reduction primitives

**Files:** Modify `include/LLK/Dialect/Micro/MicroOps.td`, `lib/Dialect/Micro/MicroOps.cpp`, `lib/Conversion/MicroToLinalg/MicroToLinalg.cpp`, `test/FileCheck.cmake`; create `test/Dialect/Micro/dynamic_boundary{,_invalid}.mlir`, `test/Conversion/MicroToLinalg/dynamic_boundary.mlir`. Migrate positional generated-builder call sites in the exporter, materializer and tests in this same commit, passing absent optional values for legacy behavior.

**Interfaces:** Produce `micro.tile_load(source, offsets, valid_sizes, padding, dst_memory, owner?) -> (static_tile, async_token)`; add optional `valid_sizes` to destination `micro.tile_store`, and optional index `valid_k` to `micro.mma`. ODS-generated builders are consumed by Task 4. Existing forms retain parsing and full-tile behavior. Load padding is typed and fills all non-copied lanes. A bounded load's source is ranked tensor/memref, not a tile operand.

- [ ] Register `MicroDynamicBoundary`, `MicroDynamicBoundaryInvalid`, `MicroToLinalgDynamicBoundary`. Check `(validM,validK)=(3,5)` into an 8x32 tile yields a clipped extraction plus tile-local padding, store writes only 3x7 from an 8x32 tile, and MMA reduction bound is valid_k. Negative constant sizes, wrong rank/padding dtype and valid_k>BK must fail. Legacy store/MMA spelling must round-trip unchanged.
- [ ] Build `llk-opt`, run `ctest --test-dir build -R '^(MicroDynamicBoundary|MicroDynamicBoundaryInvalid|MicroToLinalgDynamicBoundary)$' --output-on-failure`; confirm failures for the missing ops/operands.
- [ ] Implement verifier/parser/printer and reference lowering using tensor slices/pad and clipped inserts. Lower exact K via runtime reduction bounds, never a full BK reduction with an assumed-equivalent padded tail. Keep guarded bounds/masks through later Vector lowering.
- [ ] Rebuild and rerun the focused tests and existing `MicroToLinalg`/tile-op verifier suite. Inspect IR for fixed local tile types and no whole-input padding.
- [ ] Commit as `feat: define bounded micro tile movement and exact k tails`.

### Task 3: Dynamic plan legality and resource certification

**Files:** Modify `include/LLK/Mapping/{TileFacts.h,WorkloadGraph.h,MappingTarget.h}`, `lib/Mapping/{CMakeLists.txt,TileFacts.cpp,WorkloadGraph.cpp,StoragePlan.cpp,MappingRules.cpp,PlanBinder.cpp,CostModel.cpp}`, `lib/Conversion/MicroMapping/{PlanMaterialization.cpp,CompletePlanEvaluation.cpp}`, `lib/Target/X86/Mapping/AVX2BackendLowering.cpp`, target mapping rules/registrations in `lib/Target/{X86,GenericAccelerator}/Mapping/` and `mapping/{x86-avx2,generic-ai-accel}/rules.llkmap`; create `test/Mapping/dynamic_shape_mapping.cpp`; modify `test/Mapping/CMakeLists.txt`.

**Interfaces:** Produce `shape::PredicateVerdict {Proven, Disproven, RuntimeGuard}` in `LLKShape`; mapping facts separate static tile footprint from optional workload multiplicity. Add typed `BoundaryRequirements {bool boundedLoad; bool boundedStore; bool partialK;}` and a pure `TargetEmitter::verifyBoundary(const TargetBundle &, const BoundaryRequirements &) -> llvm::Error` hook alongside existing `verify`/`hasLowering`. The default rejects nonempty requirements; implemented emitters opt in. Generic code checks typed requirements without interpreting opaque bundle parameters. Rules select the required boundary implementation; complete-plan evaluation calls the hook. The materialized plan retains boundary operands, selected concrete route resources and runtime guards.

- [ ] Add `DynamicShapeMapping.{FixedCapacityIndependentOfMNK,BoundedResidency,MissingBoundaryEmitter,PartialKCapability,RouteAndPaddingStorage,RejectUnknownLocalAllocation,ElementwiseBoundary}`. Hold BM/BN/BK/stages fixed; changing external domain from small to large must not grow local footprint. Constrain SRAM to reject the same oversized local tiles before top-K. A full-fragment-only backend must reject a partial-K domain without a remainder emitter.
- [ ] Register/build `DynamicShapeMappingTest`; run `ctest --test-dir build -R '^DynamicShapeMappingTest$' --output-on-failure` and observe failure before implementation.
- [ ] Integrate new primitives into extraction, placement, routing, covering, binding and materialization. Certify maximum spatial residency from selected executor pools, actual lifetime overlap and pipeline stages; do not use logical grid size as concurrent allocation count. Carry unknown work counts explicitly. Implement selected AVX2 boundary realization with existing safe copy/fill/vector mechanisms and a runtime K loop; declare generic-accelerator capability limits explicitly.
- [ ] Run focused tests and `ctest --test-dir build -R 'DynamicShapeMappingTest|Mapping|CompletePlan' --output-on-failure`; inspect that a no-profile dynamic plan has a storage verdict but no rankable latency. Existing static scoring remains unchanged.
- [ ] Commit as `feat: certify dynamic micro plans with bounded local storage`.

### Task 4: Fully dynamic LLK export and explicit destinations

**Files:** Modify `lib/Conversion/LLKToMicro/{MicroKernelDriver.h,MicroKernelDriver.cpp,LLKToMicro.cpp}`, `lib/Conversion/LLKKernel/{SwiGLU.cpp,Matmul.cpp,LinalgMatmul.cpp}`, `lib/Conversion/Workload/{ContractionSourceAdapter.cpp,SourceAdapter.cpp}`, `lib/Conversion/MicroMapping/CandidateInstantiation.cpp`, `lib/Dialect/Micro/MicroOps.cpp`, `lib/Transforms/{ShapeSpecialization.cpp,ScheduleSelection.cpp,TileAndVectorize.cpp}`, `include/LLK/Transforms/Common/ScheduleLoader.h`, `lib/Transforms/Common/ScheduleLoader.cpp`, `test/FileCheck.cmake`; create `test/Conversion/LLKToMicro/dynamic_{matmul,swiglu,search}.mlir` and `test/Conversion/LLKToLinalg/dynamic_swiglu.mlir`; update the old dynamic rejection in `test/Conversion/LLKToMicro/lowering_invalid.mlir`.

**Interfaces:** Consume Tasks 1–3. Produce kernel `argument_bindings` entries `{role: input|output, port: index}` and `result_bindings` output indices. All signature arguments are bound exactly once; yielded results must match their destinations. Reify `ContractionDimensions` as SSA values in the isolated kernel, and remove integer-only facts from the dynamic path. Dynamic schedules use an explicit baseline choice, not `classifyM(-1)`; static lookup is unchanged. Add tail policy `bounded` to schedule validation/serialization and binding receipts; `none` requires divisibility proof/guards, `pad` retains static legacy semantics.

- [ ] Add CTest cases `LLKToMicroDynamicMatmul`, `LLKToMicroDynamicSwiGLU`, `LLKToMicroDynamicSearch`, `LLKToLinalgDynamicSwiGLU`. Assert `tensor<?x?x...>` ports, runtime dim-based tile counts, static tile types, bounded loads/stores and valid_k; output is an explicit external destination, with no whole-output `micro.tile_alloc` or whole-input pad. Search choices remain finite constants. Tensor-level SwiGLU empties receive dynamic sizes.
- [ ] Run the new cases with the existing tool and confirm failure. Include a mixed-static signature and a nonzero initialized matmul fixture.
- [ ] Implement runtime export using tile-number loops and quotient/remainder ceildiv from the spec. Thread external output SSA through loops; mark input/output effects separately. Supply the dynamic guard prologue and annotate its shape-only operations for analysis coverage. Update static-only legacy optimizations to dispatch to supported dynamic transformations or preserve the legal structured operation, with an explicit capability diagnostic when necessary; never select a fixed-shape packed schedule using unknown extents.
- [ ] Build tools and run `ctest --test-dir build -R 'LLKToMicro|LLKToLinalgDynamic|FrameworkSourceTest|CandidateInstantiationTest' --output-on-failure`. Inspect IR for each independently dynamic axis and all-dynamic M/N/K. This task proves export/structural behavior; execution gate is Task 6.
- [ ] Commit as `feat: export fully dynamic contractions to fixed tile microir`.

### Task 5: ABI v2 and caller-owned output lowering

**Files:** Modify `include/LLK/Runtime/{KernelAbi.h,MappedInvocation.h}`, `runtime/MappedInvocation.cpp`, `include/LLK/Conversion/MappedKernelAbi.h`, `lib/Conversion/MicroMapping/{MappedKernelAbi.cpp,MappedCompilation.cpp}`, `runtime/CMakeLists.txt`, `test/Execution/{mapped_invocation.cpp,mapped_allocation_lifetime.cpp,CMakeLists.txt}`; create `test/Execution/dynamic_mapped_invocation.cpp`.

**Interfaces:** Consume canonical shape/binding contracts. Extend `KernelAbi` with v2 symbolic metadata and argument/result bindings, keeping v1 validation/hash. Extend `validateMappedInvocation(...)` to bind symbol values and return errors before entry. Add `llvm::Expected<shape::ShapeValues> resolveMappedShapeValues(const KernelAbi &, ArrayRef<InvocationBuffer2D> inputs, ArrayRef<InvocationBuffer2D> outputs)` in `MappedInvocation.h`; successful resolution includes all descriptor/layout/allocation/alias checks. Compile reusable dynamic wrapper signatures through the existing `compileMappedKernel` API.

- [ ] Add `DynamicMappedInvocation.{BindMNK,StaticMismatch,SharedSymbolMismatch,EmptyValid,EmptyStillRejectsMismatch,NegativeSize,SizeOverflow,InsufficientAllocation,DtypeStrideAlias,Version1HashStable}` to a new non-JIT target `DynamicMappedInvocationTest`. Empty buffers use null pointers and zero bytes; negative values fail before unsigned conversion. Check the 12-descriptor and rank-two restrictions remain enforced.
- [ ] Run the new test before implementation and confirm failures. Add output-binding/lifetime tests asserting no output-sized compiler allocation and that caller storage is not released.
- [ ] Implement ABI v2 validation and hashing. Lower explicit output destination arguments before ordinary bufferization and verify result/destination equivalence. Keep matmul initialization read-only and output disjoint. Handle zero touched spans consistently; guard shape errors before scratch allocation.
- [ ] Build/run `DynamicMappedInvocationTest MappedInvocationTest MappedAllocationLifetimeTest`; execute their CTest entries and inspect lowered function signatures. Dynamic dimensions must survive in descriptors; large external outputs must not become SRAM scratch.
- [ ] Commit as `feat: add checked dynamic mapped abi and output destinations`.

### Task 6: Compile-once numerical and memory acceptance

**Files:** Create `test/Execution/dynamic_mapped_acceptance.cpp`, `test/Conversion/MicroMapping/dynamic_elementwise.mlir`; modify `test/Execution/CMakeLists.txt`, `test/CMakeLists.txt`, `test/Execution/mapped_codegen_requirements.cpp` and `.github/workflows/ci.yml` selected-acceptance commands.

**Interfaces:** Consume Tasks 1–5 through existing selected/reference compilation APIs and typed invocation. Register two acceptance binaries, `DynamicMappedAcceptance` and `DynamicMappedAVX2Acceptance`; the latter requires selected execution and fails on unsupported hosts in the designated CI job. No new runtime interface is introduced.

- [ ] Write one fully dynamic module per operator, compile once per execution mode and call it for `(1,1,1),(7,13,5),(8,32,32),(9,33,35),(17,65,67)`, plus each zero dimension and mixed-static forms. Assert compile count stays one, all valid elements meet the existing source math contract, canaries remain unchanged, and repeated invocation frees scratch. Matmul uses nonzero init. Exercise strict signed-zero/nonfinite K tails separately against an exact-K reference, and bounded-fast SwiGLU against its existing tolerance/rounding contract.
- [ ] Add negative tests for K/N disagreement in Wu and malformed output descriptors, a large-output/small-SRAM mapping case, and dynamic direct elementwise execution. Run the new CTests before final integration and identify expected failures.
- [ ] Integrate backend/legacy pipeline gaps exposed by these tests without relaxing numerical checks. Assert selected emission realizes bounded copies, clipped writes and runtime K. Any full-output temporary or reference fallback fails selected acceptance. Confirm empty M/N executes no compute/movement and K=0 initializes and stores the specified result.
- [ ] Run portable reference tests locally; run required selected tests on pinned x86_64 AVX2 CI with LLVM 22.1.8. Run `cmake --build build --target check-llk`; record revision, test counts, actual execution modes and any pre-existing skips. Milestone B is complete only after both acceptance modes pass.
- [ ] Commit as `test: establish fully dynamic contraction execution acceptance`.

### Task 7: Concrete-sample dynamic analysis and ranking

**Files:** Create `include/LLK/Perf/ShapeProfile.h`, `lib/Perf/ShapeProfile.cpp`, `include/LLK/Mapping/ShapeEvaluation.h`, `lib/Mapping/ShapeEvaluation.cpp`, `test/Perf/dynamic_shape_analysis.cpp`; modify `include/LLK/Perf/{MicroDAG.h,TuningSession.h}`, `lib/Perf/{CMakeLists.txt,MicroDAG.cpp,TuningSession.cpp}`, `lib/Mapping/{CMakeLists.txt,WorkloadGraph.cpp,CostModel.cpp}`, `lib/Conversion/MicroMapping/{CMakeLists.txt,CompletePlanEvaluation.cpp,MappedTuningSession.cpp}`, `tools/{micro-perf/micro-perf.cpp,llk-tune/llk-tune.cpp}`, `test/Perf/CMakeLists.txt`, `test/Tuning/CMakeLists.txt`.

**Interfaces:** Produce `perf::ShapeSample {shape::ShapeValues values; double weight;}`, `perf::ShapeProfile {uint32_t version=1; vector<ShapeSample> samples; std::string aggregation="weighted_mean"; uint32_t maxVariants=1;}` and `llvm::Error validateShapeProfile(const shape::ShapeContract &, const ShapeProfile &)`. Add an MLIR integration helper `llvm::Expected<OwningOpRef<ModuleOp>> materializeShapeSample(ModuleOp source, const shape::ShapeContract &, const shape::ShapeValues &)` in `include/LLK/Conversion/MicroMapping/ShapeSpecialization.h` / `lib/Conversion/MicroMapping/ShapeSpecialization.cpp`; it clones and resolves only shape computations/guards. Shared `mapping::ShapeEvaluationContext` stores resolved index SSA values; `llvm::Expected<int64_t> evaluateShapeIndex(Value, const ShapeEvaluationContext &)` evaluates supported checked shape arithmetic. Per-iteration boundary facts drive both plan cost aggregation and MicroDAG handlers. Perf plain data does not depend on adapters. Add `micro-perf --shape-values` and `llk-tune --shape-profile` using the spec's exact JSON schema.

- [ ] Add `DynamicShapeAnalysis.{NoValuesIncomplete,PartialValuesRejected,SampleMatchesStaticClone,EdgeTrafficAndValidK,EmptyHasZeroEvents,PlanMaterializedParity,ResidencyIndependentOfGrid,EventLimitExplicit,WeightedRanking,MalformedProfile}`. For every acceptance triple, compare work/bytes/capacity and scores for the sampled dynamic plan with an equivalent exact-shape analysis; distinguish external bytes from padded local footprint/fill. Unknown loops never produce a rankable one-iteration score.
- [ ] Register/build `DynamicShapeAnalysisTest`; run it before implementation and observe failure. Assert missing/nonpositive/NaN weights, duplicate/unknown symbols and invalid bindings fail deterministically.
- [ ] Implement sample materialization and shared analysis integration; preserve original source and symbolic executable domain. Bind each loop IV to `lower + iteration*step` in the evaluation context before evaluating clipped sizes/valid_k; constants substituted for M/N/K alone are insufficient. Preserve pipeline/dependency structure, and reject unsupported index expressions. Map/retain an executable candidate only after full-domain capacity/backend checks, then score every sample. Handle simulator event limits explicitly. Keep the no-profile compile-with-supplied-schedule path deterministic and unscored. Dynamic workload extraction must distinguish a known zero trip count from an unknown one.
- [ ] Run focused tests plus existing Perf, tuning and complete-plan tests. Record profile contents, bounds and truncation in reports; verify static behavior and missing-measurement handling remain unchanged.
- [ ] Commit as `feat: analyze and rank dynamic plans on explicit shape samples`.

### Task 8: Replay identity and bounded specialization

**Files:** Modify `lib/Conversion/MicroMapping/MappedTuningSession.cpp`, `include/LLK/Runtime/JitCache.h`, `runtime/JitCache.cpp`, `lib/Transforms/PackWeights.cpp`, `test/Tuning/{mapped_tuning_session.cpp,candidate_instantiation.cpp}`, `test/Execution/{jit_cache_hit.cpp,cache_eviction.cpp}`, `tools/llk-compile/llk-compile.cpp`, `docs/manual/source/{features.rst,architecture.rst}`; create `test/Tuning/dynamic_shape_replay.cpp` and register it in `test/Tuning/CMakeLists.txt`.

**Interfaces:** Compiled variant identity includes symbolic contract, guard and ABI versions; measurement identity additionally includes concrete shape/profile. Optional `DynamicVariantSet` consists of ordered guarded compiled variants and one full-domain baseline, with an explicit maximum variant count. Pack-cache identity includes actual weight K/N plus existing content/lifetime identity. Exact shapes are not mandatory executable cache keys for the baseline.

- [ ] Add `DynamicShapeReplay.{DifferentShapesReuseBaseline,DifferentShapesDoNotShareMeasurements,GuardChangesInvalidate,ProfileChangesInvalidateSelection,AbiVersionsDiffer,PackedShapeMismatch,FrozenOrderStable,VariantBudgetBounded}`. Assert repeated equivalent baseline requests reuse code; distinct shape samples cannot reuse latency; overlaps select the first matching guard deterministically; baseline handles uncovered valid shapes.
- [ ] Run the new replay tests before implementation. Keep optional specialization disabled until baseline reuse and measurement separation pass.
- [ ] Implement two-level identity and frozen replay first. Then optionally introduce divisible/interior fast variants bounded by the declared budget. Include dispatch and packing in measurements. Do not require persistence or calibration infrastructure to finish baseline support.
- [ ] Run replay/cache/tuning tests and full `check-llk`, then repeat dynamic selected acceptance only if codegen changed. Update feature documentation solely for capabilities backed by revision-pinned evidence, distinguishing generic accelerator modeling from execution.
- [ ] Commit as `feat: preserve dynamic shape identity and guarded replay`.

## Execution order and handoff

Tasks 1→2→3→4→5→6 establish the first functional milestone with fully dynamic M/N/K. Tasks 7→8 add trustworthy optimization and replay; optional specialization is last. No milestone is complete because parsing/export alone accepts `?`.

Before implementation, review the design calls in the spec, especially bounded-load semantics, exact K tails, explicit output bindings and ABI v2. Use native execution for the tightly coupled interface migration unless a different execution method is requested. Each task begins with failing behavioral tests and ends with focused passing checks and a commit. This document is a plan, not evidence that those checks have run.
