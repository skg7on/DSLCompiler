# ARM SVE and SME Mapping Targets Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deliver two static-conformance ARM targets with typed, capacity-feasible vector/contraction mappings and explicit rejection of executable compilation.

**Architecture:** Repair shared contraction analysis to honor selected vector or matrix compute resources. Add declarative profiles/policy plus file-backed target packages. Prove extraction, complete-plan evaluation, materialization, analysis parity and replay through typed public pipelines; no ARM code generator is implemented.

**Tech Stack:** C++20, CMake/Ninja, MLIR/LLVM (local configured build; CI 22.1.8), GTest, FileCheck, Python CLI integration, LLKMap and MachineModel v2.

**Spec:** [ARM SVE/SME mapping targets](../specs/2026-10-17-arm-sve-sme-mapping-targets-design.md). This revision replaces the original seven-task, configuration-only plan.

## Global Constraints

- Work in an isolated worktree under `.claude/worktrees/`; the main checkout is read-only. Do not edit another active session's locked worktree.
- Run every command below from the worktree root. Configure `build/` using `-DLLVM_PROJECT_BUILD_DIR=/Users/skg7on/Workspace/Projects/llvm-project/build`; use `ninja -C build` and `./build/bin/...` consistently.
- C++20; existing shared LLVM/MLIR linkage; preserve current RTTI/exception settings. No `LLKMapping` dependency on `LLKPerf`, runtime or JIT.
- Target-neutral shared analysis only: no branching on `arm-sve`, `arm-sme`, target bundle names or emitter prefixes in generic code.
- No Micro dialect or machine-schema change. No SVE/SME executable emitters, `codegenRequirements`, default tuner-machine selection or legacy SVE stub repair.
- VL=512 and SVL=512 are independent modeled choices. SVE accumulator capacity 2048 bytes; SME ZA capacity 4096 bytes. No additional disjoint RF pool; never inflate capacities to accept a fixture.
- Preserve source/numerical semantics, selected resource identity, per-hop storage and complete-plan feasibility before top-K retention. Static completeness does not claim hardware realizability.
- Prefix/profile/policy/package names and APIs are fixed by spec §§6–8. f32/bf16/i8 vector widths are 16/32/64; use the three-value layout domain, not `[4..16]`.
- Follow existing license banners and deprecated-API errors. Global test code fully qualifies `mlir::llk::machine` or uses `auto`. Check `llvm::Expected` before dereferencing; propagate errors from non-void helpers.
- Commit each task with a message describing its deliverable. Do not add fabricated authorship trailers.
- Register every new C++ source in its category CMake list and every CLI/FileCheck test explicitly. `// RUN:` is documentation only.
- Historical specs/reviews remain revision-pinned. Update current support enumerations only after implementation and evidence exist.

## Review Focus

- A typed contraction mapped to a vector engine must survive shared analysis, retain `svp`, and contend with vector epilogues on that same pool (Task 1).
- Initialized accumulators and result extraction must retain their storage/routes; a 512-byte override must reject the 1024-byte live fixture before retention (Tasks 2–4).
- Mixed-dtype conversion/fusion must solve each operand's layout at its own width, including BF16=32 and i8=64 (Tasks 2 and 4).
- Both registry names, machine overrides and explicit pass-spec entry points must resolve actual packages, not merely direct factory calls (Tasks 2 and 4).
- A physically complete static plan must still refuse selected-target compilation; replay/parity must not introduce reference-mode or fabricated measurement claims (Tasks 3–5).

## File map

| Area | Files and responsibility |
|---|---|
| Capability predicate | New `include/LLK/Perf/ContractionSupport.h`, `lib/Perf/ContractionSupport.cpp`; shared vector/matrix dtype-pair predicate |
| Interpretation | `include/LLK/Perf/OpDAGBuilder.h`, `lib/Perf/{MicroDAG,MicroOpDAGBuilders,Legality}.cpp`, `lib/Perf/CMakeLists.txt` |
| Analysis tests | Existing `test/Perf/{op_dag_builders,l0_static_bound,l1_resource_dag,legality}.cpp`; extend these fixtures without global target registration |
| Profiles/policy | New `machines/arm-{sve,sme}-v2.yaml`, `mapping/arm-{sve,sme}/{layouts,rules}.llkmap` |
| Packages | New `include/LLK/Target/AArch64/Mapping/{SVEMappingTarget,SMEMappingTarget}.h`, `lib/Target/AArch64/Mapping/{SVEMappingTarget,SMEMappingTarget}.cpp`; modify `lib/Target/{CMakeLists.txt,MappingTargets.cpp}` |
| Loader/package tests | `test/Machine/machine_model_loader.cpp`; new `test/Mapping/arm_targets.cpp`; modify `test/Mapping/CMakeLists.txt` |
| Public fixture support | New `test/Mapping/arm_target_fixture.{h,cpp}`, `test/Mapping/arm_placement.cpp`; existing canonical materializer and complete-plan evaluator |
| CLI inputs and tests | New `test/Conversion/MicroMapping/Inputs/arm/{contraction_f32,contraction_bf16}.mlir`, `arm_target_pipeline.py`; explicit CTest registration in `test/FileCheck.cmake` |
| Current documentation | `docs/manual/source/features.rst`, current target-list paragraphs in `ARCHITECTURE.md`, `CLAUDE.md` and developer manual as applicable |

## Preparation and baseline

- [ ] Read `CLAUDE.md`, its worktree policy, the revised spec and actual target/analysis interfaces before changing code.
- [ ] Configure and build the baseline:

```sh
cmake -S . -B build -G Ninja \
  -DLLVM_PROJECT_BUILD_DIR=/Users/skg7on/Workspace/Projects/llvm-project/build \
  -DBUILD_TESTING=ON -DLLK_BUILD_E2E_TESTS=ON
ninja -C build LLKMachineTests LLKMappingTests LLKPerfTests llk-opt micro-perf
./build/bin/LLKMachineTests
./build/bin/LLKMappingTests
./build/bin/LLKPerfTests
```

Record source/LLVM/compiler and baseline failures/skips. A known required AVX2 host-capability failure on arm64 is not ARM execution evidence. New static ARM tests must run on every host. Do not claim the full suite is green without a successful full run.

---

### Task 1: Selected vector contraction analysis and legality

**Files:** Shared interpretation/capability files and existing Perf tests in the file map.

**Interfaces:**
- Consumes: `ComputeNode::{kind,elementTypes,accumulatorDTypes,lanes,throughputPerCycle,shapes}`, mapped compute/executor metadata and existing `EventKind`/`ResourceKind`.
- Produces in `mlir::llk::perf`:

```cpp
bool supportsContractionTypes(const machine::ComputeNode &engine,
                               llvm::StringRef inputDtype,
                               llvm::StringRef accumulatorDtype);
// Add non-pure virtual to OpDAGBuildContext; default calls pickMatrixEngine.
virtual const machine::ComputeNode *
pickContractionEngine(llvm::StringRef requested, std::string &reason) const;
// Non-pure compatibility default returns existing mmaCycles(...).
virtual llvm::Expected<uint64_t>
contractionCycles(const machine::ComputeNode &engine,
                  llvm::ArrayRef<int64_t> shape, uint64_t flops,
                  llvm::StringRef accumulatorDtype) const;
```

`DAGBuilder::Context` overrides both methods. Keep the existing `pickMatrixEngine` API for injected/reference contexts. No target names enter these signatures.

- [ ] **Step 1: Write failing tests.** In the existing Perf fixtures add `SelectedContraction` tests for a synthetic vector-only profile and fully typed MMA. Build mapped attributes with `compute = {vector_engine = "v0"}` and executor `w0` using the existing metadata shape. Use a second node `v1` to ensure selected identity cannot fall back to declaration order. Assert the following facts on the emitted `MicroEvent` (the variables below are fixture-local, not new production APIs):

```cpp
EXPECT_EQ(event.kind, EventKind::Mma);
EXPECT_EQ(event.resource, ResourceKind::VectorEngine);
EXPECT_EQ(event.resourceName, "v1");
EXPECT_EQ(event.ownerPool, "w0");
EXPECT_EQ(event.workItems, 4096u); // M=N=K=16 MACs.
EXPECT_EQ(event.minCycles, 256u); // 8192 FLOPs / throughput 32.
```

Add predicate tests accepting f32/f32 and bf16/f32 on compatible vector and matrix nodes, rejecting absent accumulator dtype, unsupported kind and unsupported input. Test missing, dual vector+matrix, foreign-executor and unknown selected ids, and explicit-engine disagreement. All must return an analysis error with no fallback. Add a matrix-only control and unmapped legacy matrix-default control.

In `legality.cpp`, add `MmaCompatible` cases: vector-only N=16/f32 accepts; N=15 rejects without masking and accepts with mask; missing f32 lanes rejects; matrix fragment behavior remains; a mixed machine accepts a later compatible engine even if an earlier one fails. Missing referenced bindings must still fail.

- [ ] **Step 2: Run the new tests and confirm failures originate in the current matrix-only selection/legality.**

```sh
ninja -C build LLKPerfTests
./build/bin/LLKPerfTests --gtest_filter='SelectedContraction.*:ContractionSupport.*:*MmaCompatible*'
```

- [ ] **Step 3: Implement the predicate, context override, event resource selection and legality contract from spec §5.** `supportsContractionTypes` lives in the new Perf library files. `buildMma` calls the new picker and checked cycle method, propagates `llvm::Error` before event emission, and fails selected dtype incompatibility. Validate positive shape and checked MAC/FLOP products before calling either cycle method; the new Expected return carries missing-lane, cost-range and overflow errors. Keep existing matrix costing through the checked core adapter and old injected-context defaults. Preserve MAC dependencies, owner and measured-cycle overrides. Matrix costing retains its existing path. Vector fallback without throughput is M*K*ceil(N/accumulator-lanes)*issue; add a no-throughput test expecting 256 cycles at lanes=16/issue=1, and zero/missing lanes plus overflow error tests. Do not hide overflow or unavailable cost facts behind a one-cycle event.

- [ ] **Step 4: Add resource/parity regression controls.** A vector contraction and independent vector op both bound to `v0`, with concurrency=1, must serialize; bind the vector op to `v1` for an overlap control. Compare normalized owner/resource/cycles and scheduled costs with the existing selected analysis path. Confirm ResourceKind and bottleneck/utilization reports classify vector contraction as vector work.

```sh
ninja -C build LLKPerfTests LLKMappingTests
./build/bin/LLKPerfTests
./build/bin/LLKMappingTests
```

Expected: new synthetic cases and existing matrix/AVX2/accelerator analysis controls pass. If a new assumption is found in a downstream consumer, repair that target-neutral assumption in this task and add its specific regression.

- [ ] **Step 5: Commit** `feat(perf): analyze contractions on selected vector resources`.

### Task 2: Profiles, finite policy, packages and registry

**Files:** Profiles/policy/packages/loader/package tests from the file map.

**Interfaces:**
- Consumes: Task 1's selected vector contraction support; spec §§6–8's normative topology, budgets, ids, widths and rule table.
- Produces the following APIs in both `mlir::llk::target::sve` and `::sme`:

```cpp
llvm::ArrayRef<llvm::StringRef> emitterKeys();
llvm::Expected<std::unique_ptr<mapping::MappingTarget>>
createMappingTarget(llvm::StringRef configurationRoot,
                    llvm::StringRef machinePath = {});
```

Targets register as `arm-sve` and `arm-sme`. No new target subclass or native backend API is added.

- [ ] **Step 1: Write failing loader and registry tests.** `MachineModelLoader.LoadsArmSve` asserts f32/bf16/i8 lanes 16/32/64, accumulator `acc.0` kind acc/capacity2048, no `rf` memory and no matrix compute. `LoadsArmSme` asserts `za.0` capacity4096, `za` matrix compute and both vector/matrix owner refinements. Assert all link endpoints and compute attachments resolve. Use `auto` or fully qualified machine types.

Add `ArmTargets.RegistryLoadsBoth` in `arm_targets.cpp`:

```cpp
mlir::llk::target::registerAllMappingTargets();
for (llvm::StringRef name : {"arm-sve", "arm-sme"}) {
  auto target = mlir::llk::mapping::createRegisteredMappingTarget(
      name, LLK_SOURCE_DIR);
  ASSERT_TRUE(bool(target)) << llvm::toString(target.takeError());
  EXPECT_EQ((*target)->name(), name);
  EXPECT_FALSE((*target)->codegenRequirements().has_value());
}
```

Include `LLK/Target/MappingTargets.h`, `LLK/Mapping/MappingTarget.h` and LLVM Error. Add source to the category before running the red test. Also test a same-target machine override with modified capacity: loaded machine hash/capacity must reflect the file. A missing override file must return an error. `--help` is not part of this test.

- [ ] **Step 2: Verify missing profiles/packages cause the expected failures.**

```sh
ninja -C build LLKMachineTests LLKMappingTests
./build/bin/LLKMachineTests --gtest_filter='MachineModelLoader.LoadsArm*'
./build/bin/LLKMappingTests --gtest_filter='ArmTargets.*'
```

- [ ] **Step 3: Implement profiles and packages.** Build each YAML from the current AVX2 v2 schema, replacing its executor/memory/compute/link sections with the exact spec §6 tables; do not carry lane executors, `mxu`, 8192-byte accumulators, RF links or target-name comments across accidentally. Copy only the existing loader's shape, not executable AVX2 overrides. Preserve common seed comments and label all ARM values modeled.

Declare the ten emitter keys per target from spec §7, with `sve_mma` / `sme_mma_za`. Factory roots resolve the three exact target files. Add package sources to `LLKTargetMapping`, includes and registrations to its composition root. Keep `codegenRequirements()==nullopt` and base backend rejection.

- [ ] **Step 4: Implement bounded typed LLKMap policy.** Use parameterless row-major and the exact three-assignment blocked layout in spec §7; no `za_blocked_2d`. Generate the separately typed rules listed there with stable ids, exact port order and costs. For contraction the complete f32 predicate is:

```
match micro.mma(input[0].element_type = f32,
                input[1].element_type = f32,
                input[2].element_type = f32,
                output[0].element_type = f32);
```

The bf16 variant changes only the first two types. Both require input `acc` and output `result` in memory kind acc, row-major layouts on lhs/rhs/acc/result, worker executor, and the target's vector/matrix compute kind. No duplicate bundle width parameter is needed. Use typed graph-node predicates for fusion and per-port layout requirements; never use a graph-global dtype to decide a mixed-dtype operand's width. Do not claim unsupported i8 contractions or non-f32 SiLU.

- [ ] **Step 5: Verify configuration and finite width domains.** Extend `ArmTargets` to load every rule emitter, instantiate every key and verify well-formed/parameterless bundles; reject unknown keys, cross-key bundles and malformed parameters. Solve blocked layouts for each dtype, asserting exactly one VW solution of 16/32/64 and no undecided result. f64 must have no legal mapping. These tests are typed; do not copy the original untyped vector graph.

```sh
ninja -C build LLKMachineTests LLKMappingTests
./build/bin/LLKMachineTests
./build/bin/LLKMappingTests --gtest_filter='ArmTargets.*'
```

- [ ] **Step 6: Commit** `feat(target): register modeled SVE and SME target packages`.

### Task 3: Real contraction placement, storage and static-only boundary

**Files:** New public fixture support, `arm_placement.cpp` and both contraction inputs from the file map; `test/Mapping/CMakeLists.txt`.

**Interfaces:**
- Consumes: registered packages and Task 1 analysis; existing `CoveringSearch`, canonical complete-plan evaluator/materializer, `bindPlan`, `verifyMappedMicroIR` and `analyzeSelectedKernel`.
- Produces test-only namespace `arm_test` in `arm_target_fixture.h/.cpp`:

```cpp
struct MappedFixture {
  std::unique_ptr<mlir::MLIRContext> context;
  mlir::OwningOpRef<mlir::ModuleOp> source;
  std::unique_ptr<mlir::llk::mapping::MappingTarget> target;
  mlir::llk::mapping::WorkloadGraph graph;
  mlir::llk::mapping::CoveringPlan plan;
  mlir::llk::mapping::BoundPlan bound;
  mlir::llk::perf::SelectedKernelAnalysis analysis;
};
llvm::Expected<MappedFixture>
mapFixture(llvm::StringRef targetName, llvm::StringRef inputPath,
           llvm::StringRef machineOverride = {});
const mlir::llk::mapping::PlanPlacement *
contractionPlacement(const MappedFixture &fixture);
```

`inputPath` is a source-root-relative fixture path, resolved using `LLK_SOURCE_DIR`. Own context before modules/attrs so it is destroyed last. `mapFixture` propagates errors, including zero complete plans; it never dereferences a failed `Expected`. Preserve source and graph while plan attrs/source associations are used. `contractionPlacement` returns null unless exactly one graph node is `micro.mma`, locating its actual selected placement by node id.

- [ ] **Step 1: Add the fixed typed source fixtures.** Use this dataflow for `contraction_f32.mlir`; spell all tiles rank-two 16×16, `layout = #micro.layout<row_major>`, `owner = #micro.owner<worker>`. Every copy returns a token awaited before use:

```text
micro.kernel @arm_contraction(a: tensor<16x16xf32>,
                             b: tensor<16x16xf32>,
                             init: tensor<16x16xf32>) -> tensor<16x16xf32>
  av/bv/iv = tile_view arguments in dram
  as/bs/is = tile_async_copy av/bv/iv to sram; wait
  c = tile_async_copy is to acc; wait
  r = micro.mma as, bs, c {shape=[16,16,16], input=f32, accumulator=f32}
  out = tile_async_copy r to sram; wait
  micro.yield out
```

`contraction_bf16.mlir` changes only a/b tensor and viewed/staged tile element types to bf16 and MMA `input=bf16`; init/acc/output remain f32. Use actual MicroIR assembly syntax from `tile_ops.mlir`/existing compiler exports. No engine attr, target-specific owner, tensor.empty initializer, uninitialized accumulator, spatial replication or implicit output is allowed. Parse/verify both fixtures before mapping. The dataflow and shapes are normative; the implementer supplies assembly spelling.

The review's revised configuration probe already maps both variants on SME, with a 1024-byte ZA peak. An explicit-engine SVE probe establishes the same storage peak but is not acceptance evidence: it bypasses the missing picker and mislabels the resource under old code. The final tests use the canonical source above and require Task 1's correct vector resource classification.

- [ ] **Step 2: Write the failing public-pipeline test and helper.** `mapFixture` must register/load through the registry, parse/extract the real kernel, use deterministic search with the existing canonical complete-plan evaluator, bind with the canonical materializer under the static/partial contract, verify mapped IR, and analyze with `(kernel,machine,requireComplete=true,finalizedPlan,graph)`. Inject `mapping::evaluateCompletePlan(source,graph,target,proposal,BindContract::Partial,memoryBudgetBytes,AnalysisRequirement::Complete)` via the existing evaluator callback; include `CompletePlanEvaluation.h`/`PlanMaterialization.h` using the category's existing integration include directory. Link `LLKPerf` explicitly in the Mapping category for its direct analysis calls. A raw `CoveringSearch` without complete-plan evaluation is insufficient; follow the integration used by `MicroMapPass`, not only `test/Mapping/e2e_workflow.cpp`'s bare-search helper.

`ArmPlacement.SameTypedContractionSelectsDifferentEngines` maps the exact same input path on both targets, for both dtypes. Assert:

```cpp
const auto *placement = arm_test::contractionPlacement(*mapped);
ASSERT_NE(placement, nullptr);
EXPECT_EQ(placement->executor, "worker.0");
// SVE case; SME case uses matrix_engine/za.
EXPECT_EQ(placement->computeBindings.lookup("vector_engine"), "svp");
EXPECT_TRUE(mapped->analysis.complete);
EXPECT_TRUE(mapped->analysis.incompleteReasons.empty());
EXPECT_TRUE(mapped->analysis.storageLivenessFromPlan);
EXPECT_EQ(mapped->analysis.peakBytes.at("acc.0"), 1024u);
EXPECT_TRUE(mapped->bound.unmaterialized.empty());
```

SME uses `peakBytes.at("za.0")==1024`. Also inspect the selected memory bindings and copy storage ids: initializer reaches accumulator, MMA consumes it, result reaches SRAM and is yielded. Compare unchanged source hashes across targets. Hashing target-specific plan ids does not prove source equivalence.

- [ ] **Step 3: Run and inspect the new tests.**

```sh
ninja -C build LLKMappingTests
./build/bin/LLKMappingTests --gtest_filter='ArmPlacement.*'
```

Expected: both dtypes select the concrete engine and exact 1024-byte accumulator peak stated above. Diagnose failures in the shared path or fixture lifetimes; do not increase capacities, add a fake matrix node or replace the initializer. A changed fixture requires revising the spec and both targets' source together.

- [ ] **Step 4: Pin the static capability boundary.** Add `ArmTargets.NoExecutableBackend` for both targets: `codegenRequirements` absent; each emitter `hasLowering()==false`; `verify(validBundle)` succeeds; `lower(validBundle)` returns `target_lowering_unsupported`. Then call `compileMappedKernel` with `MappedCompileOptions.backend=MappedBackend::SelectedTarget` on an otherwise valid mapped fixture and assert the existing missing-ISA-requirements rejection. Do not use `require-executable=1` to demand ARM instruction lowering from this increment.

- [ ] **Step 5: Commit** `test(mapping): verify typed ARM contraction placement and storage`.

### Task 4: Typed coverage, negative controls, CLI parity and replay

**Files:** `arm_targets.cpp`, `arm_placement.cpp`, fixture helper, `arm_target_pipeline.py`, new typed vector inputs under `Inputs/arm/`, and `test/FileCheck.cmake`.

**Interfaces:**
- Consumes: Tasks 1–3, the ten emitter keys, fixed fixture paths and concrete plan metadata.
- Produces CTest `ArmTargetPipeline`, invoked as:

```text
python3 arm_target_pipeline.py <llk-opt> <micro-perf> <source-dir> <work-dir>
```

Use `sys.executable`/`Python3_EXECUTABLE` from CMake, subprocess argument arrays, a per-test work directory and parsed JSON assertions. No network or native ARM host is required.

- [ ] **Step 1: Add typed positive and negative policy tests.** Add/mul each of f32/bf16/i8, both conversion directions, f32 silu/reduce, and bf16→f32→silu→f32-mul fusion must produce complete static plans. In mixed-dtype tests inspect per-port solved VW: 32 for BF16 and 16 for f32. Include unsupported f64 vector input, i8 contraction and non-f32 silu negatives. Example assertion on a layout solution's parameter:

```cpp
ASSERT_TRUE(std::holds_alternative<int64_t>(solution.parameters.lookup("VW")));
EXPECT_EQ(std::get<int64_t>(solution.parameters.lookup("VW")), 32);
```

Use distinct tests/fixture operands for each signature; one generic untyped node cannot establish the table. Keep integer/float semantics as verified by MicroIR and do not add conversion/numerical approximations through flags.

- [ ] **Step 2: Add materialization and feasibility negative controls.** Create temporary same-target profile copies from shipped data, recompute identities through the loader, and mutate one fact per test: remove accumulator memory (and its links to keep YAML structurally valid), reduce accumulator capacity to512, or remove `sram_to_acc.0`/`sram_to_za.0`. Add a typed fixture requesting an absent memory kind. The corresponding search must retain no complete plan and report a memory/capacity/route reason. Do not assert a plan is valid merely because a rule matched. Keep topK=1 to pin rejection before retention. Add selected metadata tampering tests for unknown/foreign compute ids, agreeing with Task 1's direct-analysis controls.

- [ ] **Step 3: Implement both explicit pass-spec CLI paths.** For each target and contraction dtype run `llk-opt --micro-map` with absolute machine/layout/rule paths, all declared emitter keys, `mode=deterministic` and `report=<work-dir>/...json`. Check JSON selected compute ids, memory ids, physicalComplete, empty physicalReasons, source hash and bounded-search metadata. Run `--micro-verify-mapping` using its existing target options and preserve its diagnostic behavior. Obtain the reported selectedPlanId, then replay via `--micro-bind-plan` with identical target/options/top-k in a fresh subprocess. Compare parsed selected state and normalized emitted IR, not regex presence of `micro.plan` alone.

- [ ] **Step 4: Compare analysis/report parity.** Run `micro-perf --machine=<absolute-profile> --level=1 --fail-on-capacity-violation --format=yaml <bound.mlir>`, as in `acceptance_pipeline.py`; parse the named YAML fields with that test's standard-library approach (no new package dependency). Compare planner/materialized latency, per-level traffic and live storage under the documented conventions; preserve concrete compute identity and complete-analysis status. The fixture inputs and profiles must be identical across processes. A bounded exact control must agree on the best explored cost or explicitly report truncation; no claim of global optimality is made. Add fresh-process repeatability controls. Never assert that different emitter strings guarantee different costs or that modeled costs are measurements.

- [ ] **Step 5: Register and run the integration test.** Follow the existing Python pipeline CTest registrations in `test/FileCheck.cmake` and pass the built `llk-opt`/`micro-perf` paths, source root and a unique binary-dir output path. Ensure test discovery includes it:

```sh
ninja -C build LLKMappingTests llk-opt micro-perf
./build/bin/LLKMappingTests --gtest_filter='ArmTargets.*:ArmPlacement.*'
ctest --test-dir build -R '^ArmTargetPipeline$' --output-on-failure
```

Expected: typed positives pass, each negative rejects for its changed fact, registry tests remain active, and both pass-spec paths replay/analyze with parity.

- [ ] **Step 6: Commit** `test(mapping): cover ARM dtype domains, rejection and replay`.

### Task 5: Regression evidence and current support documentation

**Files:** Current support documents from the file map; a revision-pinned evidence ledger at `docs/reviews/arm-mapping-targets-acceptance.md` created only when implementation evidence exists.

**Interfaces:**
- Consumes: all implementation tasks, baseline records and actual CI/local artifacts.
- Produces: current support wording and evidence identifying the exact implementation head; no executable ARM claim.

- [ ] **Step 1: Run the full registered suite once changes are final.**

```sh
ninja -C build check-llk
```

Record registered/passed/failed/skipped/disabled counts, toolchain, source head and reasons. Compare any expected local host capability limitation to the baseline; do not suppress or relabel failures. CI LLVM22.1.8 on its supported x86 host must pass required selected AVX2 and full suite. The ARM tests themselves must have no host skips. Repeat only for new changes/failures or unresolved concerns.

- [ ] **Step 2: Write the acceptance ledger from actual artifacts.** Record both contraction dtypes' selected engines, exact1024-byte accumulator peak, initializer/output movement, per-dtype layout widths, storage/route negatives, registry and pass-spec results, static-only rejection, parity and replay identity. Record any remaining limits. Preserve the original CPU-schedule rejection audit as history; it is not delivery evidence for retargeting those schedules.

- [ ] **Step 3: Update current documentation.** In `features.rst` and current target lists describe SVE/SME as static mapping/analysis targets at modeled VL=SVL512, with shared selected-vector-contraction support and the bounded direct-MicroIR fixture domain. Explicitly exclude executable SVE/SME, physical ZA realization, calibrated timing and unchanged CPU adapter schedules. Preserve historical review claims and source revisions. Correct blanket NEON-only or fixed maximum512 hardware claims wherever they describe these new targets.

- [ ] **Step 4: Verify documentation and commit.** Run the developer-manual build using the repository's documented command, `git diff --check`, and review the evidence ledger against artifacts. Commit as `docs: record bounded ARM mapping support and acceptance`.

## Self-review and handoff

- Spec §§1–5 → Task1; §§6–8 → Task2; §9 positive contraction/storage/static boundary → Task3; typed signatures/negatives/registry/pass-spec/parity/replay → Tasks2/4; regression/claims → Task5.
- Five Review Focus items each have explicit owning tests. Required success is actual complete-plan evaluation and bound analysis, not a declared rule or plan marker.
- Signatures use existing package/plan/analysis types; proposed production additions are declared in Task1 and test-only helpers in Task3. `PlanPlacement::computeBindings`, `BoundPlan::unmaterialized` and `SelectedKernelAnalysis::{complete,peakBytes,storageLivenessFromPlan}` are real fields.
- The plan gives decisions, test assertions and fixture contracts rather than duplicating hundreds of lines of package boilerplate. Shared profile differences are specified by normative tables; no unchecked substitution table remains.
- Spec §11 constrains a future executable-backend design; it is deliberately not implemented here. No canonical scalable or ZA dialect extension is prescribed without a separate design comparison, and no remote issue edit is included.
- This document is a plan. None of its unchecked tasks, expected outcomes or future evidence ledger establishes delivered support. The next action after document review is implementation through the chosen execution workflow.
