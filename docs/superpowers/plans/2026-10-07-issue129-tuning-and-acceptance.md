# Issue #129 Mapped Tuning and Acceptance Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close G9/G10 and verify all ten repairs through original-source tuning, selected mapped compilation, frozen replay, substantive public chains and revision-pinned acceptance evidence.

**Architecture:** Put a mapped-tuning orchestrator above LLKPerf/Mapping/Runtime. It applies complete bindings to the original source, searches and finalizes plans, compiles the chosen plan through the shared facade, and optionally measures a verified executable with owned typed inputs. Public acceptance asserts actual rules/resources/storage/events and numerical results; documentation reflects exact tested revisions and execution modes.

**Tech Stack:** C++20, MLIR conversion and mapped compilation APIs, LLVM JSON/YAML adapters, GTest, FileCheck, Python subprocess runners, CMake/CTest and GitHub Actions.

**Spec:** [Master plan](2026-10-07-issue129-gap-closure.md), normative design §§8, 17, 21–22, 25–29; October 4 C5/C6/C8/C10 contracts; [issue #129](https://github.com/skg7on/DSLCompiler/issues/129).

## Global Constraints

- “`micro` remains the only canonical execution IR.”
- “No two independently evolving candidate, legality, or MachineModel implementations should remain after M12.” (Spec §26.3.)
- “Machine, layout, and rule files are versioned and content-hashed. Reports record those hashes.” (Spec §26.4.)
- All master-plan constraints apply. LLKPerf retains no runtime/JIT dependency; synthetic compatibility mode remains explicit and available.
- Measurement persistence/calibration are #51/#52. This plan needs correct in-memory measurement identity and optional fixture inputs, not a database or performance-accuracy claim.
- No task may establish acceptance from compile/lookup success, metadata substrings, arbitrary test counts or a skipped selected-x86 invocation.

## Review Focus

1. Original-source math mode, accumulator, operand order and output semantics must survive every candidate binding: T1/T2/T7.
2. A syntactically legal candidate with no resource-feasible/executable plan must be reported and excluded, not measured as a synthetic stand-in: T2/T3.
3. A missing measurement is different from compile/numeric failure; absent metrics are never zero-valued winning candidates: T3/T4.
4. Frozen replay must reproduce the actual selected plan in fresh processes and reject changed content under identical filenames: T5/T6.
5. x86 CI must fail if required selected-AVX2 tests skip or fail; documentation must identify the exact tested tree and capabilities: T8/T9.

---

## File responsibilities and mapped-session contract

`CandidateInstantiation.h/.cpp` belongs to LLKMicroMapping and reuses LLKToMicro/binding/schedule vocabulary for original-source export. New `LLKMappedTuning` in root CMake contains MappedTuningSession.cpp and depends on LLKMicroMapping, LLKMappedCompilation and LLKPerf; the dependency cannot point back from LLKPerf. `llk-tune.cpp` loads target/source/options and owns output adapters. Test/Tuning integration tests link this library explicitly. The existing runTuningSession remains the synthetic/legacy library entry point until a separate migration replaces it.

```cpp
// T1: CandidateInstantiation.h, namespace llk.
enum class CandidateSourceMode { SemanticSource, ConcreteMicro, Synthetic };
struct CandidateInstantiationOptions {
  CandidateSourceMode mode = CandidateSourceMode::SemanticSource;
  std::string sourceSymbol;
  uint64_t sourceRootOrdinal = 0;
};
struct InstantiatedCandidate {
  mlir::OwningOpRef<mlir::ModuleOp> module;
  std::string kernelSymbol;
  mlir::llk::perf::BoundTileDecisions decisions;
  uint64_t originalSourceHash = 0;
  uint64_t sourceGraphHash = 0;
  uint64_t bindingHash = 0;
};
llvm::Expected<InstantiatedCandidate>
instantiateCandidate(mlir::ModuleOp source,
                      const mlir::llk::perf::SearchSpace &space,
                      const mlir::llk::perf::Candidate &candidate,
                      const mlir::llk::perf::WorkloadShape &shape,
                      const CandidateInstantiationOptions &options);

// T2/T3: MappedTuningSession.h, namespace llk.
struct MappedTuningCandidate {
  mlir::llk::perf::RankedCandidate ranking;
  mlir::llk::mapping::CoveringPlan plan;
  std::string planReport;
  uint64_t originalSourceHash = 0;
  uint64_t instantiatedSourceHash = 0;
};
struct MappedMeasurementIdentity {
  uint64_t originalSourceHash = 0, sourceGraphHash = 0, bindingHash = 0, planId = 0;
  uint64_t targetHash = 0, machineHash = 0, layoutHash = 0, ruleHash = 0;
  uint64_t abiHash = 0, costModelVersion = 3;
  std::string compilerVersion, backend, cpu;
  std::vector<std::string> requiredFeatures, operationKeys, connectionKeys;
};
struct OwnedInvocationBuffers {
  std::vector<std::vector<std::byte>> inputStorage, outputStorage;
  std::vector<InvocationBuffer2D> inputs, outputs;
};
using MappedInputProvider = std::function<llvm::Expected<OwnedInvocationBuffers>(
    const KernelAbi &, const MappedTuningCandidate &)>;
struct MappedMeasurementRequest {
  MappedExecutable &executable;
  llvm::ArrayRef<InvocationBuffer2D> inputs, outputs;
  const MappedTuningCandidate &candidate;
  const MappedMeasurementIdentity &identity;
};
using MappedMeasurementProvider = std::function<
    llvm::Expected<std::optional<mlir::llk::perf::CandidateMetrics>>(
      const MappedMeasurementRequest &)>;
using MappedOutputVerifier = std::function<llvm::Error(
    const MappedTuningCandidate &, llvm::ArrayRef<InvocationBuffer2D>)>;
struct MappedTuningOptions {
  mlir::llk::perf::TuningSessionOptions tuning;
  mlir::llk::mapping::MappingSearchOptions mapping;
  CandidateInstantiationOptions instantiation;
  MappedCompileOptions compilation;
  MappedInputProvider inputs;
  MappedMeasurementProvider measure;
  MappedOutputVerifier verifyOutputs;
};
struct MappedTuningReport {
  uint64_t generated = 0;
  std::vector<MappedTuningCandidate> ranked;
  std::vector<mlir::llk::perf::TuningResult> rejected;
  std::vector<mlir::llk::perf::TuningResult> unrankable;
  mlir::llk::perf::SearchObjective objective;
  CandidateSourceMode sourceMode = CandidateSourceMode::SemanticSource;
  bool mappingTruncated = false;
  bool measuredCohortOnly = false;
};
llvm::Expected<MappedTuningReport>
runMappedTuningSession(mlir::ModuleOp source,
                       const mlir::llk::perf::SearchSpace &space,
                       const mlir::llk::perf::WorkloadShape &shape,
                       const mlir::llk::mapping::MappingTarget &target,
                       const MappedTuningOptions &options);
```

Candidate/plan AffineMaps are owned by the caller's MLIRContext; the report borrows no Operation pointers. `originalSourceHash` covers the selected semantic root/region, port types/order, attributes and math mode before schedule application; `sourceGraphHash`/`instantiatedSourceHash` cover the concrete exported workload graph. Destroy temporary modules after static analysis; instantiate deterministically again for measurement, verify both source hashes plus binding/plan hashes and reuse the exact selected decisions. Output buffers remain alive for the callback via OwnedInvocationBuffers. Measurement requests cannot outlive the synchronous callback.

### Task T0: Correct interim completion claims before feature work

**Files:** Modify `docs/reviews/issue67-final-acceptance.md`, `CLAUDE.md`, `docs/design/micro-ir-mapping-workflow.md` on the actual #128/current-main tree; retain the PR128 tuner crash fix. Test `test/Docs/check_doc_references.py`.

**Interfaces:** Consumes the pinned assessment and issue #129. Produces an honest interim matrix: landed infrastructure, verified portable reference invocation, and remaining mandatory selected/resource/tuner/acceptance work. No implementation API changes.

- [ ] **Step 1: Add an interim table for G1–G10 with issue #129 and this plan linked.** State that stage delivery is not the same as passing all acceptance gates. Credit resolved #106 defects and previously delivered matching/materialization APIs.
- [ ] **Step 2: Correct exact baseline/test/CI claims.** #127 baseline has 138 tests; #128 head's run has 139 with two legacy SwiGLU skips. Distinguish Linux static build/coverage success from those skipped runtime tests. Link exact head/job and identify selected-target execution as still unproved.

```markdown
Status: mandatory issue #67 acceptance remains open under #129.
Portable reference numerical invocation is verified on the audited tree.
Selected AVX2 execution, complete resource feasibility and mapping-driven
tuning require the release gates in the October 7 plan.
```

- [ ] **Step 3: Clarify documentation lint versus workflow execution.** DocReferences checks paths/help names; it does not run code-fence commands or prove their argument values. T9 adds a controlled workflow smoke runner.
- [ ] **Step 4: Run `ctest --test-dir build --output-on-failure -R DocReferences` and `git diff --check`.** Commit with `docs: qualify issue67 acceptance pending verified gap closure`. Gate 0: no #67/#109 closure claim while mandatory rows are open.

### Task T1: Instantiate complete candidates from their original source

**Files:** Create CandidateInstantiation header/source; modify `lib/Conversion/LLKToMicro/LLKToMicro.cpp`, its public header, shared tiling/binding helpers and SearchBinding loader; test `test/Tuning/candidate_instantiation.cpp` (`CandidateInstantiationTest`), source fixtures under `test/Tuning/Inputs/issue129/`; root CMake.

**Interfaces:** Produces instantiateCandidate. Factor the concrete export entry below in LLKToMicro's public header/source; the old pass delegates to it. The returned operation is owned by the supplied cloned module, and its exact root selection is validated before any write. Reuse existing Candidate/SearchSpace/BindingFacts/checkLegality and E9's valid-extent helpers. Never introduce another machine/legality implementation.

```cpp
// namespace mlir::llk; uses existing ::llk::ScheduleEntry vocabulary.
llvm::Expected<mlir::Operation *>
exportMicroKernelFromSchedule(mlir::ModuleOp module,
                              llvm::StringRef sourceSymbol,
                              uint64_t sourceRootOrdinal,
                              const ::llk::ScheduleEntry &schedule);
```

- [ ] **Step 1: Add source-preservation tests.** Use a supported source GEMM with nonzero init, swapped/distinct operands, declared math mode, and a SwiGLU whose epilogue/order is observable. Bind two schedules and assert the source module remains byte-identical, both candidates retain original input/result semantics, and each full binding becomes deterministic Micro-IR. A function with two eligible roots requires source symbol/root selection rather than taking the first root silently.

```cpp
auto one = instantiateCandidate(*source, space, candidateA, shape, options);
auto two = instantiateCandidate(*source, space, candidateB, shape, options);
ASSERT_TRUE(one);
ASSERT_TRUE(two);
EXPECT_NE(one->bindingHash, two->bindingHash);
EXPECT_NE(one->sourceGraphHash, 0u);
EXPECT_EQ(printModule(*source), originalText);
// Inspect the exported kernel's init operand/math-mode/output order and
// numerically compare in T7; do not infer semantics from a workload string.
```

Here `printModule` is a test-local helper implemented in this task using raw_string_ostream and ModuleOp::print.

- [ ] **Step 2: Run CandidateInstantiationTest.** Baseline's synthetic binder does not consume the original source module or exact root.
- [ ] **Step 3: Resolve source mode explicitly.** SemanticSource clones the module, selects the exact supported LLK/Linalg root, adapts the complete candidate to ScheduleEntry, and invokes the shared concrete exporter. ConcreteMicro accepts only bindings whose structural effects are representable from the retained source/template; reject an unsupported retile axis with its name. Synthetic calls the existing bindCandidateToMicroKernel and reports that mode. Source-backed requests never silently fall back to Synthetic.
- [ ] **Step 4: Share complete-binding validation.** Preserve non-MMA axes (layout, memory_path, owner_mapping, pipeline_stages, VW, tail_policy); reject unresolved domains, unsupported semantic effects or mismatched original/bound contraction facts. An offered axis that cannot be realized is a stable `axis_not_realized` rejection, not an accepted schedule-only annotation. In the initial selected CPU repair, serial thread/grain defaults are supported; non-default thread/grain choices must use a verified existing parallel lowering or be rejected and excluded from the advertised supported search domain. Binding legality uses original workload extents and every contraction's facts, not a post-clamping surrogate. Implement E9 pad semantics in the shared exporter; tail-none rejects remainder shapes.

```cpp
// Adapter order:
// validate full binding -> resolve original BindingFacts -> checkLegality
// -> convert binding to existing schedule vocabulary -> concrete export
// -> attach durable candidate/hash -> extract actual exported workload graph.
```

- [ ] **Step 5: Add controls for unknown root, duplicate roots, unsupported source, changed math mode and all binding axes.** A changed semantic attribute changes source identity; choosing another binding changes binding identity without mutating source. Run export/search-binding tests and CandidateInstantiationTest. Commit with `feat: instantiate tuning candidates from original source`.

### Task T2: Route each candidate through mapping, binding and shared compilation

**Files:** Create MappedTuningSession header/source and LLKMappedTuning target; test `test/Tuning/mapped_tuning_session.cpp` (`MappedTuningSessionTest`); root CMake. Reuse R7 evaluator and R6 static analysis; do not change LLKPerf into a mapper.

**Interfaces:** Produces runMappedTuningSession and static MappedTuningReport/MappedTuningCandidate. Consumes T1 and Gate R. For selected executable tuning, Gate E supplies lowering; static analysis-only second-target tuning reports materialization readiness separately from executable support.

- [ ] **Step 1: Add three candidates:** one legal/feasible, one cheap but capacity-illegal, one source-legality violation. Require generated=3, one ranked selected plan, stable reasons for the other two and a self-contained selected report. Create a source that differs numerically from the synthetic default; test the bound artifact's source identity/ports.

```cpp
auto report = runMappedTuningSession(*source, space, shape, target, options);
ASSERT_TRUE(report);
EXPECT_EQ(report->generated, 3u);
ASSERT_EQ(report->ranked.size(), 1u);
EXPECT_EQ(report->rejected.size(), 2u);
EXPECT_NE(report->ranked.front().plan.sourceBindingHash, 0u);
EXPECT_FALSE(report->ranked.front().planReport.empty());
EXPECT_EQ(report->ranked.front().ranking.result.metrics.predictedCycles,
          report->ranked.front().plan.totalCost.latencyCycles);
```

- [ ] **Step 2: Run MappedTuningSessionTest.** Baseline has no target-aware candidate orchestration and cannot supply the selected plan/report.
- [ ] **Step 3: Implement the static loop.** Generate finite complete candidates using existing CandidateGenerator; apply T1/checkLegality; construct the actual graph, bound layout/owner/memory axes and CoveringSearch callback; select the best feasible covering under the same objective; bind/reanalyze with the shared path; record plan/report and mapping truncation. Rejected source binding, no route, capacity failure and missing backend remain distinct reasons.
- [ ] **Step 4: Validate executable candidates via compileMappedKernel at `MappedStop::Lowered`.** Use the declared backend and strict physical contract; this proves selected lowering can produce backend IR without creating a JIT during static ranking. Analysis-only targets stop at MappedMicro and are labeled non-executable. Keep source-backed and synthetic provenance explicit. Only the selected plan supplies final metrics; the synthetic pre-map estimate cannot win final ranking.
- [ ] **Step 5: Make candidate and plan limits honest.** Preserve generator truncation, mapping candidate/route/event caps and every rejection count. Tie-break by candidate ID then plan ID. Keep enough static candidates for measurement/fallback before final output topK; do not trim away all candidates needed to recover from a compile failure.
- [ ] **Step 6: Run candidate/session/mapping-parity tests and inspect CMake dependency direction.** Commit with `feat: tune through selected mapped compilation`. Review gate: each reported candidate has a reproducible selected mapping, not just a synthetic bound kernel.

### Task T3: Measure the exact selected executable with full content identity

**Files:** Extend MappedTuningSession; tests MappedTuningSessionTest, allocation/ABI fixtures; modify existing measurement tests to call compileMappedKernel in the new session rather than compileConcreteMicroKernel as target-execution evidence.

**Interfaces:** Produces MappedInputProvider, MappedMeasurementRequest/Provider/Verifier, OwnedInvocationBuffers and MappedMeasurementIdentity above. MeasurementOptions' existing module callback remains the legacy synthetic contract. New mapped session uses the executable callback and exact frozen plan.

- [ ] **Step 1: Add a provider test that observes the selected executable and typed buffers.** The provider invokes it, validates outputs through a separate scalar reference, returns a finite positive measured duration and receives operation/connection keys plus ABI/target/backend identity. Compile failure is rejection; provider nullopt is an unavailable measurement that preserves the static candidate.

```cpp
options.measure = [&](const MappedMeasurementRequest &request)
    -> llvm::Expected<std::optional<perf::CandidateMetrics>> {
  EXPECT_EQ(request.identity.planId, request.candidate.plan.id);
  EXPECT_NE(request.identity.abiHash, 0u);
  EXPECT_FALSE(request.identity.operationKeys.empty());
  if (auto error = request.executable.invoke(request.inputs, request.outputs))
    return std::move(error);
  perf::CandidateMetrics measured;
  measured.measuredNs = 100.0; // deterministic callback test, not benchmark data
  return std::optional<perf::CandidateMetrics>(measured);
};
```

- [ ] **Step 2: Run mapped measurement tests.** Baseline callback receives a module/synthetic candidate and cannot establish selected mapped compilation identity.
- [ ] **Step 3: Reinstantiate and compile the exact selected plan.** Validate source/binding hashes against the static candidate, restore maps in the live context, re-finalize/re-score through R7, require the same durable ID, then compileMappedKernel to Executable. Input provider allocates typed owned buffers; checked invoke refuses mismatched descriptors. Run output verification before accepting a measured metric. Verification failure rejects the candidate, even if timing completed.
- [ ] **Step 4: Build length-delimited content identity.** Include full existing B8 operation/connection signatures, concrete compute/memory/layout/route/hop choices, original/instantiated graph and binding hashes, all target/profile/library hashes, compiler/cost-model version, ABI hash, backend CPU/features and math mode. Reuse existing key builders, extending fields only where missing. Filenames and target-name strings alone are insufficient.
- [ ] **Step 5: Add key-isolation controls.** Different VW, compute, intermediate memory, affine maps, math mode, dtype, ABI shape, source operand order or profile bytes changes identity. Identical file content at a different path retains content identity. A provider miss changes no legality; stale/wrong identity cannot substitute a measurement. No persistent record store is added.
- [ ] **Step 6: Add allocation balance during measured invocation.** E3's JIT scratch tracker confirms each warmup/timed call releases scratch. Reject nonfinite/negative/zero-invalid metric values and wrong output data. Run mapped measurement/lifetime tests; commit with `feat: measure selected mapped executables with complete identity`.

### Task T4: Define measured ranking and unavailable-metric behavior

**Files:** Modify MappedTuningSession, `lib/Perf/TuningSession.cpp` shared metric access/validation where appropriate, report schema fields; tests `test/Tuning/mapped_tuning_session.cpp`, `test/Perf/tuning_session.cpp`.

**Interfaces:** Introduce `MetricValue { std::optional<double> value; MetricOrigin origin; }`, with `MetricOrigin { Static, Measured, Unavailable }`, and one shared metric resolver `resolveMetric(const CandidateMetrics &, llvm::StringRef)` in TuningSession.h/.cpp. Static objectives use static values. Measured objectives require actual measurements for their comparison cohort; unavailable results remain legal static candidates in the report's unrankable list, never a zero score.

- [ ] **Step 1: Add reversed measurement ranking and miss controls.** Two statically ranked candidates are measured in reversed order; final measured ranking reflects measured values. A compile error rejects; a miss remains static and cannot win measured_gflops with an invented zero. Absent machine clock prevents predicted_ns from masquerading as measured_ns.

```cpp
auto missing = resolveMetric(metricsWithoutMeasurement, "measured_gflops");
EXPECT_FALSE(missing.value.has_value());
EXPECT_EQ(missing.origin, MetricOrigin::Unavailable);
auto observed = resolveMetric(metricsWithMeasurement, "measured_ns");
ASSERT_TRUE(observed.value);
EXPECT_EQ(*observed.value, 12.0);
EXPECT_EQ(observed.origin, MetricOrigin::Measured);
```

- [ ] **Step 2: Run synthetic and mapped objective tests.** Baseline measured_gflops miss currently resolves to zero, and top-K measurement need not re-establish the requested ordering.
- [ ] **Step 3: Select and account for the static finalist cohort before measurement.** Measure exactly configured `measureTop` eligible candidates, excluding warmup, replacing failed compilations with the next static candidate while respecting an explicit attempt cap. Re-sort measured finalists under measured objectives; report `measuredCohortOnly=true` and the cohort/attempt sizes, not global measured optimality. Final output topK is applied after rejection/reranking.
- [ ] **Step 4: Handle misses without changing legality.** Static objectives retain static order and attach optional measurements. Measured objectives put unavailable candidates in unrankable (with static metrics intact), and return a clear no-measured-result status if the cohort has no values. Reject a measured objective requested with no provider before starting work. Update legacy synthetic metric behavior/documentation consistently; do not retain two different zero-fallback implementations.
- [ ] **Step 5: Pin secondary metrics and ties.** Validate every objective metric, origin, direction and finite value; compare candidate ID then plan ID for ties. Test partial measurement success, equal measured values, unsupported metrics, missing static clock and measured throughput only when FLOP accounting is actually defined. Never derive measured GFLOPS from an undefined op count.
- [ ] **Step 6: Run tuning/measurement tests.** Commit with `fix: rank measured candidates without fabricated metrics`. Review gate: the report identifies what was measured and why any legal candidate was not comparable under the chosen objective.

### Task T5: Expose target-aware tuning and frozen selected-plan replay

**Files:** Modify `tools/llk-tune/llk-tune.cpp`, schedule-record adapter/report writer, `tools/llk-compile/llk-compile.cpp`; create `test/Tuning/mapped_tune_cli.py` and `test/Tuning/Inputs/issue129/measurement_inputs.json`; root CMake, workflow docs.

**Interfaces:** Add explicit target-aware flags to llk-tune: `--mapping-target`, `--mapping-root`, `--mapping-mode`, `--mapping-backend`, `--mapping-report`, `--candidate-artifacts`, `--candidate-source=semantic|concrete-micro|synthetic`, `--source-symbol`, `--source-root`, `--measure-top`, `--measurement-inputs`. Retain existing `--input`, `--machine`, `--output`, `-M/-N/-K` and no-mapping CLI behavior. JSON tuning report records the full binding, selected plan/report, source mode, metrics/origins and truncation. `--candidate-artifacts=<dir>` saves exact instantiated source and a frozen selected report per candidate ID, so a caller can reproduce the selection without reconstructing a synthetic module. Schedule YAML remains an adapter with a versioned mapping provenance reference.

- [ ] **Step 1: Add a public CLI test that supplies original source and a search space in the same exported module.** Require one selected mapping report, schedule output referencing its plan/binding, and a compiler replay of the frozen report through the existing `--plan-report` path. Run in fresh subprocesses and compare normalized selected IR/decisions.

```python
subprocess.run([llk_tune, "--input=" + str(source),
                "--machine=" + str(repo / "machines/x86-avx2-v2.yaml"),
                "--output=" + str(schedule), "-M", "16", "-N", "64", "-K", "64",
                "--mapping-target=x86-avx2", "--mapping-root=" + str(repo),
                "--mapping-mode=exact", "--mapping-backend=reference",
                "--candidate-source=semantic", "--mapping-report=" + str(report),
                "--candidate-artifacts=" + str(artifact_dir)],
               check=True, capture_output=True, text=True)
subprocess.run([llk_compile, "--mapping-target=x86-avx2",
                "--mapping-root=" + str(repo), "--plan-report=" + str(frozen),
                "--mapping-stop=lowered", str(instantiated_source)],
               check=True, capture_output=True, text=True)
```

These existing flags are confirmed against the audited tool's help; new mapping/artifact flags are introduced by this task. Freeze the selected plan subreport (retaining the existing `selectedState` schema key) and its exact instantiated source artifact; do not replay it against unrelated original pre-export IR.

- [ ] **Step 2: Run MappedTuneCLI.** Baseline discards the original module after loading the search space and exposes neither a target adapter nor a selected report.
- [ ] **Step 3: Wire loading without duplicating target policy.** Use registered MappingTarget loading shared with llk-compile. Preserve the original parsed module; load source/root and existing SearchSpace/BindingFacts. Extend the shared registered-target loader with an optional machine-path override so existing --machine selects the same MachineModel used by mapping/perf/measurement; never load one profile for legality and another for target compilation. Honor selected mapping objective and explicit backend. A target needing hardware cannot be measured on an unsupported host; static analysis mode remains available and labeled.
- [ ] **Step 4: Implement optional measurement inputs with no new numeric-file dependency.** A versioned JSON fixture contains ordered input ports (`dtype`, `shape`, `data`) and expected output ports/tolerances. Validate against KernelAbi, populate OwnedInvocationBuffers, warm up a declared count, measure a declared repeat count, verify outputs before accepting timing and record setup/warmup policy. Missing inputs with --measure-top>0 is a usage error; static default needs no provider. No random unverifiable performance measurement is invented for an arbitrary user program.
- [ ] **Step 5: Write deterministic static output and separate observed timing.** Static report, bindings, source and frozen plan bytes repeat exactly. Observed timings are optional measurement records outside the byte-identical static report; their identity is deterministic but values are observations. Use output paths supplied by the caller; no timestamps/temp filenames in plan identity. Validate old schedule YAML continues to parse and existing legacy CLI tests pass.
- [ ] **Step 6: Add usage/replay negatives.** Missing target files, unknown flags/values, no matching root, ambiguous roots, stale source, changed profile contents and unsupported selected host return errors. Copied profile paths with identical content replay. Run CLI/legacy tuning/compiler/report tests; commit with `feat: expose mapped tuning and frozen plan replay`.

### Task T6: Complete all five public acceptance chains with substantive assertions

**Files:** Modify `test/Conversion/MicroMapping/acceptance_pipeline.py`; add fixtures and target data under `test/Conversion/MicroMapping/Inputs/issue129/`; modify perf report output adapter if a stable machine-readable event summary is needed; root CMake.

**Interfaces:** Consumes Gate R, selected compiler API and T5 frozen report output. The runner accepts existing tool paths and produces normalized source/search/binding/plan/mapped IR/static-event/perf artifacts per chain. Add a versioned `--format=json` / `--events` output on micro-perf only if existing output cannot expose these fields reliably; reuse the same SelectedKernelAnalysis data, not another analysis path.

- [ ] **Step 1: Add a checked chain manifest with explicit expectations.** Each chain declares exact selected rule set/group, named compute/memory/route expectations, required inserted operations, traffic/peak conditions and execution capability.

| Chain | Required evidence |
|---|---|
| Elementwise vector add | Selected vector engine/width; typed numeric invocation in T7 |
| Staged GEMM | Explicit staging/copy/wait, feasible physical allocations, contraction engine and initialized accumulator |
| Fused SwiGLU/comparable graph | Actual shipped fused rule selection, one complete group, resolved VW, numerical fused execution |
| Required transform | Incompatible source/destination maps force an emitted transform; removing it changes observed values |
| Generic-accelerator two-hop | Canonical tile operations, concrete SRAM→L2→DRAM route/engines, L2 allocation and capacity proof; hardware invocation not required |

- [ ] **Step 2: Make current runner fail missing contracts.** Assert it has five entries, calls llk-compile, round-trips search spaces and binds frozen reports. The old four chains/substrings/nonempty stdout must not satisfy the new tests.

```python
assert len(chains) == 5
assert set(chain_names) == {"vector-add", "staged-gemm", "fused-swiglu",
                            "required-transform", "second-target-two-hop"}
assert plan_analysis["events"] == perf_analysis["events"]
assert plan_analysis["predicted_cycles"] == perf_analysis["predicted_cycles"]
assert plan_analysis["traffic_bytes"] == perf_analysis["traffic_bytes"]
assert plan_analysis["peak_bytes"] == perf_analysis["peak_bytes"]
assert not perf_analysis["capacity_violations"]
```

- [ ] **Step 3: Run the full public chain for each manifest entry.** Source/search-space export (where appropriate), parse/print twice, complete candidate binding, search/report twice in fresh processes, frozen read/rebind, micro-verify-mapping, llk-compile inspection/lowering stops, and micro-perf static analysis. Compare normalized reports and IR bytes under identical tool/config revisions.
- [ ] **Step 4: Inspect selection and semantics, not key substrings.** Parse report/metadata objects for exact rules/compute/ports/routes/hops; check required copies/waits/transforms/storage IDs and group boundaries. Validate no physical-skip note and no unmaterialized strict decision. Compare all normalized event fields including owners/occurrences/accesses, not only totals. Capacities must be independently within the profile's concrete nodes.
- [ ] **Step 5: Add negative controls per chain.** Undersized intermediate memory, missing link/compute node, transform removed, changed binding, fused unresolved width and tampered storage provenance each fail at the expected stage. Second target retains generic ODS neutrality. Preserve separate ID-rebind tests while proving real frozen report replay as well.
- [ ] **Step 6: Run `ctest --test-dir build --output-on-failure -R 'MicroMappingAcceptancePipeline|MicroMappingBindPlanRoundtrip|MappingE2EWorkflowTest|MappedTuneCLI'`.** Commit with `test: verify all five mapped acceptance chains`. Review gate: the public runner invokes the compiler and verifies static selected-state equivalence for every required fixture.

### Task T7: Add a deterministic numerical acceptance matrix

**Files:** Extend `test/Execution/mapped_acceptance.cpp`; create shared numeric-reference helpers and `test/Execution/mapped_avx2_acceptance.cpp`; root CMake. Use E9 fixture/source adapter and E1 checked invocation.

**Interfaces:** Produces portable reference acceptance plus a separate mandatory selected-AVX2 executable. A scalar oracle uses actual stored bf16-rounded inputs and the declared math contract; it does not call the production lowering/vector math helper to generate expected output.

- [ ] **Step 1: Add fixed-seed signed random data and nonzero init.** Use an explicit xorshift32 sequence in the test rather than implementation-defined standard-library distributions; seed `0x12967`, map values to [-0.5,0.5], then round to stored bf16 for bf16 inputs. Test aligned `(16,64,64)`, M/N/K/all tails from E9, width4/8 add, fused convert/SILU/mul, required transform and two different outputs.

```cpp
uint32_t state = 0x12967u;
auto nextValue = [&]() {
  state ^= state << 13; state ^= state >> 17; state ^= state << 5;
  return static_cast<float>(state & 0xffffu) / 65535.0f - 0.5f;
};
// Fill inputs with nextValue(); compute expected from the stored values.
// Strict f32 GEMM: abs<=2e-4 + rel<=2e-4*|expected|, for these bounded inputs.
// SILU reference: x/(1+exp(-x)); bounded_fast uses its separately declared bound.
// Apply the declared output conversion before comparison. BF16 output checks
// compare stored BF16 values/bits under its contract, not the f32 tolerance.
```

- [ ] **Step 2: Run reference numeric tests before implementation completes.** Tails/fused group/transform/multiple outputs must expose the missing cases; successful all-ones tests cannot substitute.
- [ ] **Step 3: Compile the exact selected plan and invoke it.** Assert actual fused selection/backend manifest where requested. Inputs/outputs use typed descriptors; output is initialized to a sentinel that differs from valid results, with canaries around buffer spans. Verify every output element, input immutability and guard bytes, including aligned subviews inside larger allocations. Nonzero descriptor offsets and padded boundary rows are negative controls for the first checked ABI.
- [ ] **Step 4: Add negative/ownership controls.** Wrong rank/dtype/shape/short backing allocation never calls machine code. Returned input and shared internal result write separate outputs correctly. Repeat invocation with changing data and E3 allocation balance. For transform-required fixtures, identity/no-transform lowering must produce a mismatch, proving the fixture exercises the transformation.
- [ ] **Step 5: Separate supported and unsupported tail/math contracts.** tail-none remainder shapes reject; pad passes. Unsupported policy/op/dtype/feature diagnostics are asserted. Strict versus bounded_fast tolerances are explicit and justified by the tested value range; never increase tolerance simply to hide a discrepancy.
- [ ] **Step 6: Run portable numeric, selected-x86 numeric, ABI and lifetime suites.** Commit with `test: exercise mapped random tails transforms and multiple outputs`. Review gate: actual calls cover each positive supported contract and each rejected ABI/backend case.

### Task T8: Require non-skipped selected AVX2 evidence in x86 CI

**Files:** Modify `.github/workflows/ci.yml`, root CMake, selected acceptance executable and a small `test/Execution/check_required_execution.py` parser if required; retain coverage/CodeQL workflows' independent purposes.

**Interfaces:** Register `MappedAVX2Acceptance` with label `selected-avx2`, portable tests with label `mapped-reference`. Add an executable option `--require-selected-target` that treats missing features/JIT or any skipped required case as failure; local portable runs may still diagnose unsupported hosts honestly.

- [x] **Step 1: Add a parser/control test using saved CTest outcomes.** All required selected cases pass -> success; one skipped/failed/not-run/missing case -> failure. Empty output cannot pass. Include the audited legacy-skip records as a negative documentation control, not a mandatory dependency for new mapped execution. `RequiredExecutionEvidenceParser` covers pass, skip, failure, missing, and empty reports.

```python
required = {"MappedAVX2Acceptance", "MappedAllocationLifetimeTest"}
for name in required:
    assert name in outcomes, f"missing required execution: {name}"
    assert outcomes[name] == "passed", f"required execution did not pass: {name}"
```

- [x] **Step 2: Run parser tests and invoke the executable on an injected unsupported-feature case with --require-selected-target.** It must fail; ordinary local diagnostics may report unsupported. `SelectedTargetFlagContract` sets `LLK_TEST_DISABLE_AVX2=1` and confirms that required selected execution fails with the injected-feature diagnostic.
- [ ] **Step 3: Add an explicit selected-x86 step to pinned LLVM22 static CI.** Record host architecture/features and exact toolchain; run the selected acceptance binary with --require-selected-target and then the full CTest suite. Save CTest result artifacts and selected Vector/LLVM/manifest/numeric/lifetime evidence. If the runner lacks AVX2, use a known compatible runner or report CI blocked; do not silently skip the gate. The workflow now captures the post-lowering LLVM dialect, translated LLVM IR, and a selected-plan manifest through an opt-in JIT evidence sink, validates those artifacts, and uploads them with the JUnit/log and host record. Local sink/file-writer/error-propagation tests and five validator controls pass; the full local run still fails the required selected binary on arm64, so this step remains open until pinned LLVM22 x86 CI passes.
- [ ] **Step 4: Validate new static library dependencies on that job.** Ownership pipelines, target vectorization and runtime math/copy/alloc symbols must resolve without a monolithic MLIR dylib masking missing links. Local LLVM24/arm64 runs the reference and policy tests and cannot replace this x86 gate.
- [x] **Step 5: Retain legacy tests with truthful status.** On the local Darwin arm64 LLVM24 build, `SwigluScalar` passes four tests and skips `JitCompilationSmoke` and `E2EWithAbiWrapper` because ORC LLJIT is not configured; `SwiGLUVector` passes `PipelineSmoke` and skips `Correctness` because JIT compilation is unavailable. These are local runtime-capability skips, not selected mapped-target evidence. The pinned Linux LLVM22 job must still report their actual outcomes and is covered by Step 6.
- [ ] **Step 6: Run final candidate CI and inspect artifacts/skips.** Commit with `ci: require selected mapped AVX2 execution evidence`. Review gate: actual selected-target numerical and ownership tests pass on the exact repaired head, with no required skips.

### Task T9: Publish a reproducible twelve-criterion closure matrix

**Files:** Update `CLAUDE.md`, `docs/reviews/issue67-final-acceptance.md`, `docs/design/micro-ir-mapping-workflow.md`, report/schema documentation; create `test/Docs/workflow_smoke.py` and an explicit command manifest; root CMake. Reconcile GitHub #67/#106/#109/#129 when the implementation is complete and publication is within the authorized workflow.

**Interfaces:** Consumes every release gate. Produces exact revision/toolchain/host/capability/test-artifact evidence for each normative criterion and each issue finding. DocReferences remains link/help lint; WorkflowSmoke executes a small allowlisted set of actual documented commands with controlled output directories.

- [x] **Step 1: Add a workflow smoke test with real arguments.** Cover parse/export, mapped search/report, frozen replay/verify, compiler lowered stop, micro-perf static output and mapped-tune static output. Use checked-in fixtures and actual option values. It must fail for a nonexistent fixture or invalid pass option; do not execute arbitrary shell fences from Markdown. `WorkflowSmokeNegativeControls` proves that a missing fixture root and invalid `llk-opt` option fail through the manifest runner.

```python
for command in manifest["commands"]:
    result = subprocess.run(command, cwd=repo, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    for relative_path in manifest_outputs(command):
        assert (artifact_dir / relative_path).is_file()
```

`manifest_outputs` is a local manifest lookup function implemented here; commands are JSON argv arrays, never eval/shell text. The docs reference the same fixtures/options, and a drift check compares those exact examples to the manifest.

- [x] **Step 2: Run DocReferences and WorkflowSmoke.** Correct the claimed scope of each check and replace stale examples/options. Preserve runnable relative CLI commands in docs; the evidence table identifies the actual isolated checkout and revision separately. At the current candidate tree, `DocReferences`, `WorkflowSmoke`, and `WorkflowSmokeNegativeControls` pass.
- [x] **Step 3: Write the criterion matrix using exact artifacts:** Candidate evidence and local status for all twelve criteria are recorded in `docs/reviews/issue67-final-acceptance.md` at `1c678de`. The matrix explicitly leaves selected-x86 invocation open; final release artifacts still depend on T8.

| Design §29 | Required repaired evidence |
|---|---|
| 1 Search round trip | Five-chain search-space parse/print and complete binding with all axes, T1/T6 |
| 2 Machine topology | Concrete compute/memory/DMA identity and occupancy tests, R1–R5 |
| 3 Routing | Deterministic direct/two-hop selection and per-hop physical realization, R4/R8/T6 |
| 4 Declarative layout | Target-owned solved layouts/maps and width/transform controls, E5–E7/T6 |
| 5 Rules | Single-op and real fused selection/group execution, E4–E7 |
| 6 Complete coverings | Exact coverage, concrete bindings and capacity-feasible top-K, R7/R8 |
| 7 Search modes | Shared callback, independent exact oracle, honest caps/beam limits, R7/R8 |
| 8 Materialization | Frozen replay, named allocations/copies/waits/transforms and backend realization, R4/E4–E9/T6 |
| 9 Perf agreement | Full static normalized event/resource/traffic/peak/cycle parity, R6/R8/T6 |
| 10 AVX2 and legacy | Non-skipped selected-x86 invocation plus legacy regression evidence, E8/T7/T8 |
| 11 Second target | Canonical tile two-hop materialization/perf and generic ODS neutrality, T6 |
| 12 Determinism | Fresh-process byte comparisons of static reports/IR and replay, R8/T5/T6 |

- [ ] **Step 4: Record proof with honest capability boundaries.** Pin the implementation head/CI run rather than quoting the #127/#128 baseline. List registered/passed/failed/skipped test counts from that head, every required skip reason, compiler/LLVM revisions and execution mode. Portable reference correctness, actual selected AVX2 execution, static model parity and future calibrated prediction are separate rows.
- [ ] **Step 5: Reconcile historical issue checklists.** Credit #106's resolved defects; link still-relevant #109 items to exact new tests; connect #67 mandatory criteria to the matrix; check #129 G1–G10 only with corresponding evidence. Keep production persistence/calibration in #51/#52 and accelerator hardware execution outside this closure. Do not close any mandatory row on the basis that a stage PR merged. The local G1–G10 crosswalk is in `docs/reviews/issue67-final-acceptance.md`; tracker checklist reconciliation remains for the final authorized publication after the x86 gate.
- [ ] **Step 6: Run documentation checks, final full CTest and final x86 selected gate on the release head.** Complete whole-branch review and resolve findings. Commit with `docs: record verified issue129 and issue67 acceptance evidence`. Gate close requires every mandatory matrix row passed; unresolved gaps remain open with concrete diagnostics and artifacts.
