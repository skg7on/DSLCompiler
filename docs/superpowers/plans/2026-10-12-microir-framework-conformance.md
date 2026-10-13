# MicroIR Primitive and Target Conformance Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Establish framework acceptance across non-contraction computation, movement/layout and distinct topologies, with honest analysis/reference/selected/runtime capability reports.

**Architecture:** Reuse existing primitive rules, injected analysis builders, finalized storage plans and target emitters. Add integration-level capability receipts and conformance fixtures, not target-name policy in generic mapping. Reports describe a concrete source/plan and validated envelope, rather than a blanket promise that every instance of an opcode is supported.

**Tech Stack:** C++20, MLIR/Micro/Linalg/Vector, GTest, LLKMap/YAML, Python evidence checks and selected AVX2 CI.

**Spec:** [Framework design §§4 and 7](../specs/2026-10-12-generic-microir-dnn-compiler-framework-design.md); [index](2026-10-12-generic-microir-framework-implementation.md).

## Global Constraints

- “Generic machine and mapping libraries must not branch on target names or interpret opaque emitter/bundle names as policy.”
- “Adding a target requires machine/rule conformance and explicit lowering capabilities.”
- “A profile with estimated costs establishes modeled mapping support; executing hardware requires a backend and execution evidence.”
- “Selected-target lowering must realize the chosen implementation and reject unsupported decisions.”
- “Portable reference execution is separately identified.”
- A and B's completeness/source contracts are prerequisites. Existing AVX2/generic accelerator packages remain the targets; no physical accelerator execution is claimed.

## Review Focus

1. A known emitter key may only verify a bundle: C1 must distinguish `isKnownEmitter` from `hasLowering` and actual realization.
2. Same-kind engines, multi-hop transfers and constrained memories must retain concrete identity and capacity: C2.
3. Reduction removes an axis and may exceed ABI v1's rank envelope; analysis/lowering must not be advertised as invocation: C3.
4. A layout transform must have physical traffic/storage/resource cost while a true view remains logical: C2/C3.
5. Renaming a target/profile must not change generic algorithm dispatch; changed topology must change legal choices through data: C4.

---

## Proposed integration report

Create `include/LLK/Conversion/FrameworkCapabilities.h` and `lib/Conversion/MicroMapping/FrameworkCapabilities.cpp` in `LLKMappedCompilation`, above mapping/perf/runtime. Do not move these fields into MachineModel or encode one global support bit in MicroIR.

```cpp
// namespace mlir::llk::framework
enum class CapabilityStatus { Supported, Unsupported, NotChecked };
struct CapabilityReceipt {
  uint32_t version = 1;
  CapabilityStatus structural = CapabilityStatus::NotChecked;
  CapabilityStatus interpretation = CapabilityStatus::NotChecked;
  CapabilityStatus mapping = CapabilityStatus::NotChecked;
  CapabilityStatus physicalPlan = CapabilityStatus::NotChecked;
  CapabilityStatus referenceLowering = CapabilityStatus::NotChecked;
  CapabilityStatus selectedLowering = CapabilityStatus::NotChecked;
  CapabilityStatus invocation = CapabilityStatus::NotChecked;
  std::vector<std::string> reasons;
  std::string sourceIdentity;
  std::string contractIdentity;
  std::string targetIdentity;
  uint64_t planId = 0;
};
CapabilityReceipt describeCapabilities(
    Operation *kernel, const workload::WorkloadContract &,
    const mapping::MappingTarget &,
    const perf::SelectedKernelAnalysis *analysis = nullptr,
    const mapping::CoveringPlan *plan = nullptr,
    const ::llk::MappedCompilation *compilation = nullptr);
llvm::json::Object capabilityReceiptToJson(const CapabilityReceipt &);
```

`::llk::MappedCompilation` is the existing compilation result, not a new type. A status is Supported only after the applicable stage succeeds for this artifact. Known emitter keys/declared backend requirements establish availability, not actual lowering success. Runtime status distinguishes ABI-valid preparation from host ISA availability and successful invocation; receipt reasons include those separate facts. Source/target identities reuse the existing complete identity rendering, including B's contract.

## C1: Report stage-specific validated capabilities

**Files:** Create the capability header/implementation above, `test/Conversion/framework_capabilities.cpp`; Modify root CMake, Conversion CMake, `tools/llk-compile/llk-compile.cpp`, `tools/llk-tune/llk-tune.cpp`.

**Interfaces:** Produces `describeCapabilities`/JSON; consumes existing `MappingTarget::{isKnownEmitter,createEmitter,codegenRequirements}`, `TargetEmitter::hasLowering`, A's analysis and B's contract. Add a `capabilities` object to plan/tuning sidecar reports; absent stages use `not_checked`.

- [ ] **Write failing test** using the B2 direct elementwise source and `target::generic_accel::createMappingTarget(LLK_SOURCE_DIR)`. Verify a target lacking a codegen contract does not claim selected execution even after complete static mapping:

```cpp
auto report = framework::describeCapabilities(kernel, contract, **target,
                                               &analysis, &plan);
EXPECT_EQ(report.interpretation, framework::CapabilityStatus::Supported);
EXPECT_EQ(report.mapping, framework::CapabilityStatus::Supported);
EXPECT_EQ(report.selectedLowering, framework::CapabilityStatus::Unsupported);
EXPECT_EQ(report.invocation, framework::CapabilityStatus::NotChecked);
```

Here `kernel`, `contract`, `analysis`, `plan` are obtained from the actual mapped source fixture and strict analysis in the test, never handcrafted success flags. Add AVX2 `MappedStop::MappedMicro` vs `Lowered` cases: selected lowering is not checked before lowering; after selected lowering it is Supported only with realized groups and zero reference fallback. Add malformed/unsupported bundle tests using current emitter validation.
- [ ] **Run red:** `cmake --build build --target LLKConversionTests -j 6`; run the new `FrameworkCapabilities.*` suite. Expected: missing APIs/status fields.
- [ ] **Implement the receipt**. Call structural verification; read analysis completeness/reasons; read finalized plan physical readiness and selected rule/emitter availability; read actual compilation mode/counters. Preserve `NotChecked` for any stage not attempted. Validate ABI against B's declared ordered ports before invocation eligibility. Selected mode with any reference fallback is Unsupported, even if portable execution would succeed. Serialize stable lowercase status strings, version and reasons; never use an absent bool as true.

```cpp
receipt.interpretation = !analysis ? CapabilityStatus::NotChecked
    : (analysis->complete ? CapabilityStatus::Supported
                          : CapabilityStatus::Unsupported);
```

Do not run backend lowering or JIT merely to print a static report. Separate target backend unavailable from unattempted host execution. Add schema tests that every receipt field is present and measured/predicted origin remains outside capability status.
- [ ] **Run green:** capability tests plus existing `MappedCodegenRequirementsTest`, selected-group lowering and analysis-only target tests. Expected: honest stage status and no behavioral change to successful mapped compilation.
- [ ] **Commit:** stage named files; `git commit -m 'feat: report validated framework capabilities by stage'`.

## C2: Add movement/resource conformance with literal expectations

**Files:** Create `test/Conversion/framework_conformance.cpp`; Modify `test/Mapping/resource_regression_fixture.cpp` only for additional fixture inputs, `test/Conversion/CMakeLists.txt`; optionally add target rules only when an existing primitive rule is missing.

**Interfaces:** Consumes `issue129::{ResourceCase,resourceCase,searchCase}`, `compileMappedKernel`, `analyzeSelectedKernel`; produces framework conformance tests, not a second evaluator.

- [ ] **Write failing parity/coverage assertions** for existing `two-compute`, `named-ports`, `two-hop`, `padded-layout`, `capacity-topk` fixtures. The central test chain is:

```cpp
auto fixture = issue129::resourceCase("two-hop");
ASSERT_TRUE(bool(fixture)) << llvm::toString(fixture.takeError());
mapping::MappingSearchOptions options;
options.mode = mapping::SearchMode::Exact;
options.topK = 1;
auto search = issue129::searchCase(*fixture, options);
ASSERT_TRUE(bool(search)) << llvm::toString(search.takeError());
ASSERT_EQ(search->plans.size(), 1u);
::llk::MappedCompileOptions compile;
compile.requireExecutable = false;
compile.stop = ::llk::MappedStop::MappedMicro;
auto bound = ::llk::compileMappedKernel(fixture->source.get(),
                                        *fixture->target,
                                        search->plans.front(), compile);
ASSERT_TRUE(bool(bound)) << llvm::toString(bound.takeError());
auto kernel = perf::findMicroKernel(bound->module.get(), "");
ASSERT_TRUE(bool(kernel)) << llvm::toString(kernel.takeError());
auto analysis = perf::analyzeSelectedKernel(*kernel,
    fixture->target->machine(), true, search->plans.front(), fixture->graph);
ASSERT_TRUE(bool(analysis)) << llvm::toString(analysis.takeError());
EXPECT_TRUE(analysis->complete);
EXPECT_DOUBLE_EQ(analysis->cost.latencyCycles,
                 search->plans.front().totalCost.latencyCycles);
```

Add literal fixture-specific assertions for hop identities, transfer-engine names, per-memory bytes/live peaks and selected same-kind compute engines using the already documented expected values in existing resource tests. Do not derive expected peaks/cycles by calling the planner a second time. Add at least one new mixed vector→physical-transform→store fixture whose original source is preserved and whose traffic/work counts are hand-calculated from its static tile shapes. A logical view control case must add zero movement traffic; the physical transform case must add transform work and allocated bytes.
- [ ] **Run red:** build/run `FrameworkConformance.*`. Newly strict coverage/capability assertions may expose missing scaffolding classification or primitive rule support; record which exact boundary fails, rather than relaxing completeness.
- [ ] **Implement only the missing conformance seam**. Existing resource/cost behavior is already delivered. If a fixture exposes a real unsupported primitive, add its verifier/handler/rule/emitter coverage in the owning package with focused tests; reject genuinely unsupported semantics. Extend synthetic fixture input tables with explicit nodes/links/capacities, keeping test expected values independent of production algorithms.

```cpp
// Conformance assertion for the unchanged logical-view control fixture:
EXPECT_EQ(logicalAnalysis.trafficBytes, baselineAnalysis.trafficBytes);
// For the physical transform fixture, assert literal bytes for its shape/dtype,
// and a concrete selected transform resource, in addition to shared parity.
```

`logicalAnalysis` and `baselineAnalysis` are strict results for the two corresponding actual fixtures; their inputs differ only by a layout-preserving view. Parity alone is insufficient, so each fixture also carries literal counts/bytes.
- [ ] **Run green:** conformance suite plus existing route identity, storage lifetime, exact-resource oracle and capacity-topK tests. Expected: strict complete parity and concrete resource identity, with no target-name conditions introduced.
- [ ] **Commit:** stage named fixture/test/owning-package fixes; `git commit -m 'test: establish MicroIR movement and resource conformance'`.

## C3: Prove direct reduction lowering and composed vector execution

**Files:** Create `test/Execution/Inputs/framework/reduce.mlir`, `test/Execution/Inputs/framework/add_mul.mlir`, `test/Execution/framework_primitives.cpp`, `test/Conversion/framework_selected_evidence.py`, `lib/Conversion/MicroMapping/TargetCoverageValidation.h`, `lib/Conversion/MicroMapping/TargetCoverageValidation.cpp`; Modify root `CMakeLists.txt`, `test/Execution/CMakeLists.txt`, `test/Conversion/CMakeLists.txt`, `include/LLK/Mapping/MappingTarget.h`, `lib/Conversion/MicroMapping/CompletePlanEvaluation.cpp`, `mapping/generic-ai-accel/rules.llkmap`, `lib/Target/GenericAccelerator/Mapping/GenericAcceleratorMappingTarget.cpp`.

**Interfaces:** Reuses `micro.reduce`/`micro.vector` semantics, target emitters and ABI v1. Current `ReduceOp::verify` requires the reduced axis removed; there is no init operand. Thus rank-two reduction yields rank one and is tested through static/selected-lowering stages, with invocation explicitly unsupported. The add→mul chain supplies rank-two numeric execution. This adds neither a Linalg reduction importer nor a new canonical op.

- [ ] **Write the reduction fixture and failing conformance tests**:

```mlir
module {
  micro.kernel @row_sum(%a: tensor<8x8xf32>) -> tensor<8xf32> {
    %tile = micro.tile_view %a {shape = array<i64: 8, 8>}
      : tensor<8x8xf32> -> !micro.tile<8x8xf32, memory = #micro.memory<sram>>
    %sum = micro.reduce "sum" %tile {axis = 1 : i64}
      : !micro.tile<8x8xf32, memory = #micro.memory<sram>>
        -> !micro.tile<8xf32, memory = #micro.memory<acc>>
    micro.yield %sum : !micro.tile<8xf32, memory = #micro.memory<acc>>
  }
}
```

Assert verifier success, exactly one reduce with 64 work items, output shape `[8]`, strict static completeness and chosen concrete vector resource. Compile selected AVX2 through `MappedStop::Lowered`, assert realized vector reduction evidence, then request executable preparation and assert `runtime_capability: mapped ABI v1 requires rank-two ports`. Invalid axis=2, result shape `[8,1]` and mismatched dtype must fail the existing verifier. A supported max/prod dialect variant lacking a selected rule must report unavailable mapping/lowering rather than be treated as sum.

Write `add_mul.mlir` from B2's direct `[8,8]` add fixture by feeding the add result into a `micro.vector "mul"` with the first input tile and yielding its `[8,8]` result. The numeric test uses literal arrays and this reference:

```cpp
for (size_t i = 0; i < 64; ++i)
  EXPECT_NEAR(outputs[i], (left[i] + right[i]) * left[i], 1.0e-5f);
EXPECT_GT(compilation.backendGroupsRealized, 0u);
EXPECT_EQ(compilation.referenceGroupsLowered, 0u);
```

Define `left`, `right`, `outputs` as owning 64-element f32 arrays, with `left[i]=i/8.0f`, `right[i]=(63-i)/16.0f`; wrap them as typed `[8,8]` ABI-v1 buffers. `compilation` is the actual selected MappedCompilation result. Add the add→mul and add→mul→reduce sources to strict static conformance on both profiles, while executing only the rank-two add→mul variant.
- [ ] **Run red:** build/run the new static and execution tests and `FrameworkSelectedEvidence`. Expected: absent fixture/evidence or any uncovered selected reduction case is visible, never skipped as success.
- [ ] **Implement the fixtures and missing accelerator policy coverage**. The audited generic accelerator only declares vector-add/MMA/copy rules, so add explicit `accel.vector_mul` and `accel.reduce_sum_f32` rules and corresponding validating emitter keys. These emitters remain analysis-only (`hasLowering=false`); do not claim a hardware backend. Use the existing vector-engine machine facts and LLKMap predicates. AVX2 already declares `avx2.reduce_sum_f32`/`avx2_reduce`; reuse that path. Selected evidence must contain the realized reduction/vector operations and reject reference fallthrough. Persist analyzed traffic/storage/events, stage capability and numeric results in a versioned conformance manifest keyed by source/plan/target/compiler/mode. A rank-one reduction's invocation row is unsupported with the ABI reason, while its static/lowering rows can pass.

The f32 reduce rule must validate the operand/result dtype in its target bundle verifier, since `micro.reduce` has no dtype attribute in the match expression. Add these declarative rules; emitter validators require f32 for the reduction fixture and validate the existing operand/result port names:

```text
rule accel.vector_mul {
  match micro.vector(op = "mul");
  require executor kind pe;
  require compute kind vector_engine;
  require layout operand0 satisfies accel.row_major;
  input "operand0";
  input "operand1";
  output "result";
  bundle "accel.vector.mul";
  emit "accel_vector_mul";
  cost 2;
}
rule accel.reduce_sum_f32 {
  match micro.reduce(op = "sum");
  require executor kind pe;
  require compute kind vector_engine;
  input "operand0";
  output "result";
  bundle "accel.reduce.sum.f32";
  emit "accel_reduce_sum";
  cost 3;
}
```

Bundle declaration validates structural policy; a validator receiving only bundle parameters cannot check operation types. Add this target-owned, read-only hook and integration helper:

```cpp
// Add to mapping::TargetEmitter (default preserves existing plugins):
virtual llvm::Error verifyOperations(
    llvm::ArrayRef<mlir::Operation *> coveredOps,
    const TargetBundle &bundle) const {
  return verify(bundle);
}
// TargetCoverageValidation.h, namespace mlir::llk::mapping:
llvm::Error verifyTargetOperationCoverage(
    mlir::ModuleOp bound, const CoveringPlan &, const MappingTarget &);
```

The helper groups bound operations by the existing selected instance metadata, resolves each plan bundle's emitter, and calls verifyOperations without rewriting IR. Call it from complete evaluation before analysis/scoring; a coverage failure becomes UnsupportedMaterialization candidate rejection. Accelerator sum/mul emitters override it to require the stated variant and f32 operand/result TileTypes. They use MicroDialect type accessors, not LLKPerf, so target package dependencies stay unchanged. Test bf16/f64 coverage refusal and unchanged IR on rejection. Keep actual realization/host execution separate from this support check.
- [ ] **Run green:** static conformance on both profiles; portable and required-selected AVX2 add→mul runtime tests on supported x86; reduction selected-lowering and invocation-rejection tests. Compare explicit IR evidence, numeric outputs and shared static parity. Selected execution CI must run without skips before acceptance.
- [ ] **Commit:** stage named files and focused primitive fixes; `git commit -m 'test: validate composed vector and reduction framework paths'`.

## C4: Validate target extension boundaries and publish acceptance

**Files:** Create `test/Conversion/framework_target_portability.cpp`, `docs/reviews/2026-10-12-framework-conformance.md`; Modify Conversion CMake, `docs/manual/source/{features.rst,architecture.rst}`, `docs/design/micro-ir-mapping-workflow.md` and relevant target/tool pages.

**Interfaces:** Consumes all prior capability/conformance receipts. New test-only target wrapper forwards existing `MappingTarget` methods to an owned/borrowed target while returning a different `name()`. Test lifetime must outlive the wrapper; never register it globally.

- [ ] **Write failing portability tests**. Map identical primitive sources with the wrapper and base target; compare legal selected decisions after excluding provenance-only target-name fields. Change machine resource IDs/capacities/links in a fixture and assert the legal covering set changes from data, not from the wrapper name. Renamed targets still lack selected support when their base target lacks a backend. Use two-compute/capacity-topK literal fixtures to assert actual selected IDs and rejection reasons.

```cpp
EXPECT_EQ(baseReport.mapping, renamedReport.mapping);
EXPECT_EQ(baseReport.interpretation, renamedReport.interpretation);
EXPECT_NE(baseReport.targetIdentity, renamedReport.targetIdentity);
EXPECT_EQ(renamedReport.selectedLowering,
          framework::CapabilityStatus::Unsupported); // analysis-only base
```

Add a source/target package boundary scan for new target/operator-name switches in generic `lib/Mapping`/`lib/Machine`, and a link-dependency check that neither library links LLKPerf/LLKRuntime/JIT. Such scans enforce boundaries; functional topology tests establish behavior.
- [ ] **Run red:** build/run target-portability tests and the manifest validator. Missing capability receipts/manifest rows must fail; do not invent failure when existing algorithms already satisfy a portability assertion.
- [ ] **Implement boundary fixes only when demonstrated**. Put profile-specific constraints in YAML/LLKMap and target emitters; remove accidental new generic dispatch introduced during B/C. Forward wrapper emitter/backend methods without interpreting their keys. Write the acceptance review with revision, exact commands/counts, per-row source/plan identities, selected/reference evidence, ABI/input bounds and unsupported rows. Add a manifest validator requiring every framework row to be either passed with evidence or unsupported with reason.
- [ ] **Run green:** full `check-llk`, supported x86 selected CI, boundary/manifest checks, documentation-reference checks and strict Sphinx. Expected: conformance reports can establish modeled portability independently of hardware execution.
- [ ] **Commit:** stage named files; `git commit -m 'docs: publish capability-based framework acceptance'`.

## Increment acceptance

- [ ] Non-contraction/primitive composition, movement/layout, resource feasibility and target-extension rows have revision-pinned evidence.
- [ ] Every new receipt distinguishes not checked, unsupported and validated support.
- [ ] AVX2 selected execution realizes the chosen backend; generic accelerator remains explicitly analysis-only.
- [ ] No generic machine/mapping target-name policy or reversed library dependency was introduced.
- [ ] Later primitive/target contributors can follow the same verifier→extraction→analysis→rule→emitter→runtime conformance checklist.
