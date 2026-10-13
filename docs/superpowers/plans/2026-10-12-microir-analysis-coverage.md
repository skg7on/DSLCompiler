# Explicit MicroIR Analysis Coverage Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make interpretation coverage explicit and prevent unknown computation from receiving a complete zero-cost analysis or a retained ranked plan.

**Architecture:** Extend the merged injectable leaf-builder registry and the existing `MicroDAG::incompleteReasons`. Preserve permissive legacy entry points; shared analysis uses audited coverage and strict scoring rejects incomplete results. Keep traversal and structural operations core-owned.

**Tech Stack:** C++20, MLIR operation interfaces, LLVM Error, GTest, Python CLI regressions and the existing shared event scheduler.

**Spec:** [Framework design §§4.2, 4.5, 5 and 7](../specs/2026-10-12-generic-microir-dnn-compiler-framework-design.md); [plan index and baseline](2026-10-12-generic-microir-framework-implementation.md).

## Global Constraints

- “A new primitive requires a documented coverage path through extraction, legality, mapping, performance interpretation and lowering.”
- “Unsupported behavior must be identified at the applicable boundary.”
- “The core owns traversal and graph invariants; handlers contribute events, storage and aliases.”
- “`LLKMapping` remains independent of performance, runtime and JIT libraries.”
- “Shared complete-plan materialization, live-storage/capacity checks and static analysis run before top-K retention.”
- Do not repeat PR #141's registry refactor. All new source registrations use root `CMakeLists.txt` at the audited base.

## Review Focus

1. Unknown effect-free arithmetic is computational, not automatically free: A1/A3.
2. Index/control constants and genuine logical views must remain explicitly zero-cost without dropping data dependencies: A2.
3. A custom callback may produce aliases/storage without an event; coverage must follow its declared interpretation, not its event count: A1/A2.
4. Missing coverage inside repeated loops must produce stable bounded diagnostics, without weakening structural traversal or event limits: A3.
5. Analysis-only ranking must require complete modeling even when no executable backend exists: A4.

---

## File ownership and proposed interfaces

Modify `include/LLK/Perf/{OpDAGBuilder,MicroDAG}.h`, `lib/Perf/{OpDAGBuilder,MicroDAG,MicroOpDAGBuilders,SelectedKernelAnalysis,MicroPerfReport}.cpp`. Shared scoring integration stays in `lib/Conversion/MicroMapping/`; no `LLKMapping` dependency reversal.

The following APIs are **new** and shared by A1–A3:

```cpp
// namespace mlir::llk::perf, in MicroDAG.h
enum class CoverageMode { LegacyPermissive, Audit, Strict };
struct DAGBuildOptions {
  CoverageMode coverage = CoverageMode::Audit;
};
llvm::Expected<MicroDAG> buildMicroDAG(
    mlir::Operation *kernel, const machine::MachineModel &machine,
    const OpDAGBuilderRegistry &builders, const DAGBuildOptions &options,
    const MeasuredOverrides *measured = nullptr);

// OpDAGBuilder.h; existing registerBuilder/replaceBuilder/lookup remain.
enum class DAGLeafKind { Modeled, ZeroCost };
// Add to OpDAGBuilderRegistry:
llvm::Error registerZeroCost(llvm::StringRef name, Callback callback);
std::optional<DAGLeafKind> classification(llvm::StringRef name) const;
// Add to OpDAGBuildContext:
virtual void noteIncomplete(std::string reason) = 0;
```

`Callback` is the existing `std::function<llvm::Error(mlir::Operation &, OpDAGBuildContext &)>`. A registry entry owns `{Callback callback; DAGLeafKind kind;}`. The old `lookup` still returns a callback pointer; `replaceBuilder` preserves classification. An explicit zero-cost callback may alias values or validate a compile-time/control operation. A supported handler's intentional zero events does not make it unsupported.

The old two `buildMicroDAG` overloads remain source-compatible and select `LegacyPermissive`. `analyzeSelectedKernel` and report construction explicitly select `Audit`. Strict callers use `requireComplete` as before; `CoverageMode::Strict` is also available to direct DAG consumers.

## A1: Add declared leaf coverage and audited extraction

**Files:** Modify `include/LLK/Perf/{OpDAGBuilder,MicroDAG}.h`, `lib/Perf/{OpDAGBuilder,MicroDAG}.cpp`; Test `test/Perf/op_dag_builders.cpp`.

**Interfaces:** Consumes the existing registry callback/traversal; produces the interfaces above. No new global registry.

- [ ] **Write failing tests** in the existing test namespace, reusing its defined `parseKernel` helper:

```cpp
TEST(OpDAGBuilders, AuditNamesUnknownPureComputation) {
  auto parsed = parseKernel(R"mlir(
module { micro.kernel @unknown {
  %a = arith.constant 1.0 : f32
  %b = arith.addf %a, %a : f32
  micro.yield
} })mlir");
  ASSERT_NE(parsed, nullptr);
  OpDAGBuilderRegistry builders;
  ASSERT_FALSE(builders.registerZeroCost("arith.constant", noEvent));
  DAGBuildOptions options;
  auto dag = buildMicroDAG(parsed->kernel, machine::MachineModel{},
                           builders, options);
  ASSERT_TRUE(bool(dag)) << llvm::toString(dag.takeError());
  ASSERT_EQ(dag->incompleteReasons.size(), 1u);
  EXPECT_NE(dag->incompleteReasons[0].find("unmodeled_leaf: arith.addf"),
            std::string::npos);
}

TEST(OpDAGBuilders, ExplicitZeroCostRegistrationIsNotInferredFromEvents) {
  OpDAGBuilderRegistry builders;
  ASSERT_FALSE(builders.registerZeroCost("test.alias", noEvent));
  EXPECT_EQ(builders.classification("test.alias"), DAGLeafKind::ZeroCost);
  ASSERT_FALSE(builders.registerBuilder("test.storage", noEvent));
  EXPECT_EQ(builders.classification("test.storage"), DAGLeafKind::Modeled);
  EXPECT_FALSE(builders.classification("test.unknown"));
  expectError(builders.registerZeroCost("test.alias", noEvent), "test.alias");
}
```

- [ ] **Run red:** `cmake --build build --target LLKPerfTests -j 6`. Expected: compilation fails on the new API. After introducing declarations, the first test must fail until the unknown path records a reason. Do not accept a zero-test invocation.
- [ ] **Implement the registry records and extraction path.** Keep existing duplicate/empty callback/structural-op rejection. Add a core-owned `noteIncomplete` implementation which appends first-seen reasons deterministically. In `DAGBuilder::buildOp`, dispatch supported handlers exactly once; for an unregistered leaf, use this decision:

```cpp
if (!handler) {
  if (options.coverage != CoverageMode::LegacyPermissive)
    recordIncomplete("unmodeled_leaf: " +
                     op.getName().getStringRef().str());
  return llvm::Error::success();
}
```

`recordIncomplete(const std::string &)` is a new private DAGBuilder method backed by an insertion-order vector plus a membership set. Store the `DAGBuildOptions` in the builder. At the new overload's return boundary, Strict returns an `llvm::StringError` containing the ordered reasons when they are nonempty; Audit returns the annotated DAG. Do not add a blanket `isMemoryEffectFree` exemption.
- [ ] **Run green:** `build/bin/LLKPerfTests '--gtest_filter=OpDAGBuilders.*'`. Expected: all matched tests pass, including existing custom callback and deterministic graph tests.
- [ ] **Commit:** `git add include/LLK/Perf lib/Perf/OpDAGBuilder.cpp lib/Perf/MicroDAG.cpp test/Perf/op_dag_builders.cpp` then `git commit -m 'feat: declare MicroDAG leaf coverage'`.

## A2: Explicitly cover constant/control and logical scaffolding

**Files:** Modify `lib/Perf/MicroOpDAGBuilders.cpp`, `include/LLK/Perf/OpDAGBuilder.h`; Test `test/Perf/op_dag_builders.cpp`, `test/Perf/candidate_binding.cpp`.

**Interfaces:** Consumes A1; produces a default registry covering the actual exporter scaffolding. Existing storage/logical/movement handlers remain authoritative.

- [ ] **Write failing tests**:

```cpp
TEST(OpDAGBuilders, AuditAcceptsIndexScaffoldingWithoutEvents) {
  auto parsed = parseKernel(R"mlir(
module { micro.kernel @indices {
  %a = arith.constant 1 : index
  %b = arith.addi %a, %a : index
  %c = arith.muli %b, %a : index
  micro.yield
} })mlir");
  ASSERT_NE(parsed, nullptr);
  auto dag = buildMicroDAG(parsed->kernel, machine::MachineModel{},
                           makeDefaultOpDAGBuilders(), DAGBuildOptions{});
  ASSERT_TRUE(bool(dag)) << llvm::toString(dag.takeError());
  EXPECT_TRUE(dag->events.empty());
  EXPECT_TRUE(dag->incompleteReasons.empty());
}

TEST(OpDAGBuilders, IntegerDataArithmeticIsNotControlScaffolding) {
  auto parsed = parseKernel(R"mlir(
module { micro.kernel @data {
  %a = arith.constant 1 : i32
  %b = arith.addi %a, %a : i32
  micro.yield
} })mlir");
  ASSERT_NE(parsed, nullptr);
  auto dag = buildMicroDAG(parsed->kernel, machine::MachineModel{},
                           makeDefaultOpDAGBuilders(), DAGBuildOptions{});
  ASSERT_TRUE(bool(dag)) << llvm::toString(dag.takeError());
  ASSERT_EQ(dag->incompleteReasons.size(), 1u);
  EXPECT_NE(dag->incompleteReasons[0].find("arith.addi"), std::string::npos);
}
```

- [ ] **Run red:** rebuild `LLKPerfTests`, run `--gtest_filter=OpDAGBuilders.AuditAcceptsIndexScaffoldingWithoutEvents`. Expected: constants/index operations lack declared coverage.
- [ ] **Implement explicit validators**, using the following callback bodies:

```cpp
auto constant = [](Operation &, OpDAGBuildContext &) {
  return llvm::Error::success(); // immutable compile-time arith.constant
};
auto indexOnly = [](Operation &op, OpDAGBuildContext &context) {
  if (op.getNumResults() != 1 || !op.getResult(0).getType().isIndex())
    context.noteIncomplete("unmodeled_leaf: " +
                           op.getName().getStringRef().str());
  return llvm::Error::success();
};
llvm::cantFail(builders.registerZeroCost("arith.constant", constant));
llvm::cantFail(builders.registerZeroCost("arith.addi", indexOnly));
llvm::cantFail(builders.registerZeroCost("arith.muli", indexOnly));
```

Audit actual matmul/SwiGLU and direct tile exporter bodies for additional index-only operations. Add only operations observed in those fixtures, with type checks and tests; do not predeclare all Arith/Tensor operations free. Pure metadata `tensor.empty` requires a documented distinction from materialized allocation; tile allocation is already handled by its modeled builder. Existing logical tile handlers must retain alias/dependency/layout-transform behavior. An arbitrary `tensor.extract`/`tensor.insert` cannot be treated as free metadata.
- [ ] **Run green:** build/run `LLKPerfTests` filters `OpDAGBuilders.*:CandidateBinding.*`, and the registered LLK-to-Micro round-trip tests. Expected: audited exported fixtures have no unmodeled scaffolding, and logical views still preserve dependencies and nonzero physical-transform costs.
- [ ] **Commit:** `git add lib/Perf/MicroOpDAGBuilders.cpp include/LLK/Perf/OpDAGBuilder.h test/Perf` then `git commit -m 'feat: classify MicroIR control and logical coverage'`.

## A3: Propagate completeness through shared analysis and reports

**Files:** Modify `lib/Perf/{SelectedKernelAnalysis,MicroPerfReport}.cpp`, `include/LLK/Perf/MicroDAG.h`; Test `test/Perf/op_dag_builders.cpp`; Create `test/Perf/analysis_coverage_cli.py`; Modify `test/Perf/CMakeLists.txt`.

**Interfaces:** Consumes A1/A2; existing `analyzeSelectedKernel(Operation *, const MachineModel &, bool)` and `analyzeSelectedDag(const MicroDAG &, const MachineModel &, bool)` retain signatures. They now report unknown coverage through the existing completeness fields.

- [ ] **Write failing test** using the A1 unknown-computation IR:

```cpp
TEST(OpDAGBuilders, UnknownCoverageRefusesStrictAnalysis) {
  auto parsed = parseKernel(R"mlir(
module { micro.kernel @strict {
  %a = arith.constant 1.0 : f32
  %b = arith.addf %a, %a : f32
  micro.yield
} })mlir");
  ASSERT_NE(parsed, nullptr);
  auto partial = analyzeSelectedKernel(parsed->kernel,
                                       machine::MachineModel{}, false);
  ASSERT_TRUE(bool(partial)) << llvm::toString(partial.takeError());
  EXPECT_FALSE(partial->complete);
  auto strict = analyzeSelectedKernel(parsed->kernel,
                                      machine::MachineModel{}, true);
  ASSERT_FALSE(bool(strict));
  EXPECT_NE(llvm::toString(strict.takeError()).find("arith.addf"),
            std::string::npos);
}
```

Include `LLK/Perf/SelectedKernelAnalysis.h`. Add a loop variant containing the unknown op in two iterations; assert one ordered reason naming the operation, with no event-limit or dependency behavior change. Add a custom registered alias-only callback case proving supported zero-event handlers remain complete.
- [ ] **Run red:** rebuild/run `OpDAGBuilders.UnknownCoverageRefusesStrictAnalysis`. Expected: current shared entry point calls the permissive builder and incorrectly reports complete.
- [ ] **Implement audited call sites**:

```cpp
auto dag = buildMicroDAG(kernel, machine, makeDefaultOpDAGBuilders(),
                         DAGBuildOptions{CoverageMode::Audit});
```

Apply this at shared analysis and at the report's single DAG extraction; pass the same extracted DAG into `analyzeSelectedDag`. Keep strictness at the analysis boundary, so permissive reports carry `complete=false` plus reasons. The CLI regression invokes `micro-perf` on the unknown fixture and asserts its JSON completeness and `unmodeled_leaf` reason; reuse the actual JSON key spelling found in `MicroPerfReport.cpp`, without adding a parallel schema. Register the Python test with `$<TARGET_FILE:micro-perf>` and the source directory.
- [ ] **Run green:** `cmake --build build --target LLKPerfTests micro-perf -j 6`; run the coverage GTests and the new CTest `AnalysisCoverageCLI`. Expected: explicit partial report, strict rejection, and unchanged existing complete reports.
- [ ] **Commit:** stage the named files and `git commit -m 'fix: report unmodeled MicroIR analysis as incomplete'`.

## A4: Require complete analysis for ranked plans

**Files:** Modify `lib/Conversion/MicroMapping/{CompletePlanEvaluation.h,CompletePlanEvaluation.cpp,MappedTuningSession.cpp,MicroMappingCommon.h}`; Test `test/Conversion/MicroMapping/complete_plan_evaluation.cpp`, `test/Perf/op_dag_builders.cpp`; Modify `docs/manual/source/features.rst` and `docs/design/micro-ir-mapping-workflow.md`.

**Interfaces:** Extend the integration helper's existing parameters by appending `AnalysisRequirement requirement = AnalysisRequirement::Complete`, where the new enum in `CompletePlanEvaluation.h` is `{Complete, PartialReport}`. The existing return remains `llvm::Expected<mapping::CompletePlanEvaluation>`. Executable binding remains strict regardless of this argument. PartialReport is for explicitly labeled report artifacts, never top-K scoring.

- [ ] **Write failing integration regression** in the existing complete-plan tests. Start with `issue129::resourceCase("two-compute")`, insert `arith.constant` and `arith.addf` before the source kernel's terminator using `OpBuilder`, and search with `issue129::searchCase`. This fixture API is defined in `test/Mapping/resource_regression_fixture.h` and already supplies complete evaluation. The central assertion is:

```cpp
auto fixture = issue129::resourceCase("two-compute");
ASSERT_TRUE(bool(fixture)) << llvm::toString(fixture.takeError());
auto kernel = perf::findMicroKernel(fixture->source.get(), "");
ASSERT_TRUE(bool(kernel)) << llvm::toString(kernel.takeError());
OpBuilder builder((*kernel)->getContext());
builder.setInsertionPoint((*kernel)->getRegion(0).front().getTerminator());
Value one = arith::ConstantOp::create(builder, (*kernel)->getLoc(),
    builder.getF32FloatAttr(1.0));
arith::AddFOp::create(builder, (*kernel)->getLoc(), one, one);
mapping::MappingSearchOptions options;
options.topK = 1;
auto result = issue129::searchCase(*fixture, options);
ASSERT_TRUE(bool(result)) << llvm::toString(result.takeError());
EXPECT_TRUE(result->plans.empty());
```

Also assert the frontier diagnostics contain `unmodeled_leaf: arith.addf`. The extraction graph already ignores this scalar leaf, making the test specifically exercise materialized analysis coverage. Re-extract the graph if inserting new tile nodes; this test adds none.
- [ ] **Run red:** `cmake --build build --target LLKConversionTests -j 6`; `build/bin/LLKConversionTests '--gtest_filter=CompletePlanEvaluationTest.*'`. Expected: the otherwise feasible proposal is retained before the new audit requirement is applied. Register the new test under `CompletePlanEvaluationTest` and include `mlir/Dialect/Arith/IR/Arith.h` and `mlir/IR/Builders.h`.
- [ ] **Implement complete analysis as an independent scoring requirement**. In `evaluateCompletePlan`, derive `requireComplete = executableContract || requirement == Complete`; use it for `analyzeSelectedKernel`, and reject `!physicalComplete` when complete scoring is requested. Preserve `BindContract::Partial` for analysis-only targets without requiring a backend. Change ranking callers to Complete; partial-report callers explicitly select PartialReport. Keep evaluator invocation before retention in `CoveringSearch` unchanged.

The existing analysis-error branch tests `strict` before falling back to synthesized events. Change that condition to `requireComplete` too; complete analysis-only scoring must reject an extraction error and never take the partial fallback. Structural binding strictness and backend availability remain separate.

```cpp
if (requireComplete && !analysis->complete)
  return rejected(DiagnosticCode::UnsupportedMaterialization,
                  "analysis_incomplete: " +
                  llvm::join(analysis->incompleteReasons, "; "));
```

Use the helper's existing rejection constructor/diagnostic shape; do not add a new exception/error channel. Test an explicitly requested PartialReport result separately and assert it is labeled incomplete and excluded from ranked results.
- [ ] **Run green:** existing capacity-topK/resource parity tests plus the new coverage regression; then the fresh full `check-llk`. Update feature/workflow docs to explain audited, strict and legacy-custom behavior. Expected: infeasible/incomplete plans never mask feasible complete plans, while intentional partial analysis artifacts remain inspectable.
- [ ] **Commit:** stage the named files and `git commit -m 'fix: gate ranked MicroIR plans on complete analysis'`.

## Increment acceptance

- [ ] Full fresh suite passes; existing #141 custom-builder tests run with a nonzero count.
- [ ] Complete defaults cover delivered exporter scaffolding and canonical primitives; unknown computation/effects are diagnosed.
- [ ] Selected-plan and materialized analysis still share one extraction and scheduler.
- [ ] Analysis-only targets may produce complete modeled rankings without an executable backend; partial artifacts remain explicitly partial.
- [ ] PR evidence records the tested revision, host, exact counts and unchanged support limits.
