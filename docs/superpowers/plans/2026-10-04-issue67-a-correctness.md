# Issue #67 Stage A — Correctness and Legality Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Eliminate the confirmed PR #111 correctness defects while preserving working mapping behavior.

**Architecture:** Make operand/result occurrences explicit, reuse rule/layout evaluators for verification, and evaluate constraints from required facts. Keep restricted exact search honest while stage B adds joint branching.

**Tech Stack:** C++20, LLVM/MLIR, LLKMap, GoogleTest, FileCheck, CMake/Ninja/CTest.

**Spec:** [Normative design](../specs/2026-09-18-microir-inspired-dslcompiler-enhancement-design.md), [master contracts/dependencies](2026-10-04-issue67-improvement-plan.md), [review evidence](../../reviews/2026-10-04-issue67-pr111-status.md).

## Global Constraints

Apply every global constraint and interface decision in the master plan, including isolated worktrees, canonical Micro-IR, target-owned policy, LLVM 22/24 portability, bounded-solver soundness and legacy compatibility. Proposed APIs below are implemented by their owning task before dependent tasks use them.

## Review Focus

Apply the master's five review-focus cases. Each owning task specifies the negative input, positive control and verification command. Source snippets are implementation/test contracts; adapt includes and registration to existing file conventions without weakening the assertions.

## Delivery and file responsibilities

A1/A2 change endpoint identity and connection synthesis. A3–A5 own semantic/structural verification. A6 owns restriction disclosure. A7 owns persistent binding facts/roles. A8 owns safe view lowering. A9 owns transform events/costs. Commit/PR A1+A2 together if an intermediate API cannot preserve a working build; the remaining fixes are reviewable independently.

### A1: Introduce operand/result endpoint identity

**Dependencies:** Baseline

**Files:**
- Modify: include/LLK/Mapping/WorkloadGraph.h
- Modify: lib/Mapping/WorkloadGraph.cpp
- Modify: include/LLK/Mapping/MappingPlan.h
- Modify: lib/Mapping/MappingPlan.cpp
- Test: test/Mapping/workload_graph.cpp

**Interfaces:** Produces PortRef, lookupPort and canonicalPortRefString from the master. LayoutRequirement/SolvedLayout/PortSpec gain optional endpoint fields during migration; canonicalization happens after finalize().

- [ ] **Step 1 — add the regression and legal controls.**

Add this identity assertion and extraction tests for two operands carrying one value:
```cpp
TEST(WorkloadGraph, EndpointIdentityDistinguishesRepeatedUses) {
  PortRef lhs{7, PortDirection::Input, 0};
  PortRef rhs{7, PortDirection::Input, 1};
  EXPECT_NE(canonicalPortRefString(lhs), canonicalPortRefString(rhs));
  EXPECT_EQ(canonicalPortRefString(lhs), "node=7,input=0");
}
```
Extract equivalent graphs built in reversed insertion order; after finalization verify identical endpoint/value relationships and serialized output. Add invalid node, invalid direction/index, and external-source controls.

- [ ] **Step 2 — run the focused test before changing behavior.**

```sh
cmake --build build --target MappingWorkloadGraphTest -j 6
build/MappingWorkloadGraphTest
```

Expected before the change: the new behavioral assertion fails (or the new interface test cannot compile). Keep existing legal controls green; record the observed failure.

- [ ] **Step 3 — implement this contract.**

Implement the renderer and lookup without using SSA pointers as identity:
```cpp
std::string canonicalPortRefString(const PortRef &p) {
  return "node=" + std::to_string(p.node) +
      (p.direction == PortDirection::Input ? ",input=" : ",output=") +
      std::to_string(p.index);
}
```
lookupPort locates the finalized node and selects inputs/outputs by index; invalid references return nullptr. Include port direction/index in candidate/connection canonical identity. Preserve value IDs for data provenance, not endpoint disambiguation. B1 owns persisted schema migration.

- [ ] **Step 4 — rerun the focused command and review the gate.**

Every repeated use has a stable, distinct endpoint; reorder tests remain deterministic. Run relevant neighboring tests when the interface changes. No broad success claim follows merely from a new schema parsing.

- [ ] **Step 5 — commit the independently testable change.**

```sh
git add include/LLK/Mapping/WorkloadGraph.h lib/Mapping/WorkloadGraph.cpp include/LLK/Mapping/MappingPlan.h lib/Mapping/MappingPlan.cpp test/Mapping/workload_graph.cpp
git commit -m "feat(mapping): model explicit workload port references"
```

### A2: Connect each operand use under its own solved layout

**Dependencies:** A1

**Files:**
- Modify: lib/Mapping/MappingRules.cpp
- Modify: lib/Mapping/Placement.cpp
- Modify: include/LLK/Mapping/Placement.h
- Modify: lib/Mapping/CoveringSearch.cpp
- Test: test/Mapping/covering_search.cpp
- Test: test/Mapping/connections.cpp

**Interfaces:** Consumes PortRef. ConnectionRequest gains producerPort/consumerPort; ConnectionPlan gains producerPort/consumerPorts. Solutions resolve against an endpoint, replacing boundSolvedLayoutForValue.

- [ ] **Step 1 — add the regression and legal controls.**

Move the review appendix's repeated-value probe into the registered suite with real ranked tensor types and change its assertions to desired semantics:
```cpp
// Producer's result is plain; the same value feeds both consumer ports.
// operand0 requires plain; operand1 requires blocked.
EXPECT_EQ(result->plans.front().connectionPlans.size(), 2u);
```
Collect connection.consumerPorts: input0 must have Direct, input1 must have LayoutTransform (or TransferAndTransform if visibility requires movement). Add same-family/different-VW and same-family/equal-VW controls, plus two different values needing one family.

- [ ] **Step 2 — run the focused test before changing behavior.**

```sh
cmake --build build --target MappingCoveringSearchTest MappingConnectionsTest -j 6
build/MappingCoveringSearchTest
build/MappingConnectionsTest
```

Expected before the change: the new behavioral assertion fails (or the new interface test cannot compile). Keep existing legal controls green; record the observed failure.

- [ ] **Step 3 — implement this contract.**

Resolve the requirement's port name to its declared input/output index while constructing the candidate. Copy that PortRef into the solved layout. Build one request per actual boundary use, even when SSA values are equal:
```cpp
// Algorithm contract: grouping occurs only AFTER endpoint layout resolution.
// for each consumer input occurrence:
//   resolve producer endpoint and this input's solved layout;
//   synthesize legal alternatives for those two endpoints;
//   group only equal memory + concrete layout + access/type representations.
```
Do not turn ambiguity into "no layout". Duplicate requirements for one endpoint must agree or reject the candidate. Include endpoints, solved parameters and map identity in representation keys. Retain node-level lists only as derived compatibility projections.

- [ ] **Step 4 — rerun the focused command and review the gate.**

The previous one-Direct result is impossible for incompatible operand uses; equal-layout sharing remains legal. Run relevant neighboring tests when the interface changes. No broad success claim follows merely from a new schema parsing.

- [ ] **Step 5 — commit the independently testable change.**

```sh
git add lib/Mapping/MappingRules.cpp lib/Mapping/Placement.cpp include/LLK/Mapping/Placement.h lib/Mapping/CoveringSearch.cpp test/Mapping/covering_search.cpp test/Mapping/connections.cpp
git commit -m "fix(mapping): preserve per-use layout obligations in connections"
```

### A3: Verify the full selected rule and resource capabilities

**Dependencies:** A1/A2

**Files:**
- Modify: include/LLK/Mapping/MappingRules.h
- Modify: lib/Mapping/MappingRules.cpp
- Modify: lib/Mapping/PlanBinder.cpp
- Test: test/Mapping/plan_binder.cpp
- Test: test/Mapping/mapping_rules.cpp

**Interfaces:** Produces verifyRuleSelection from the master; consumes existing evaluatePredicate. Record resolved rule parameters and compute selections in transient placement state; B1 persists them.

- [ ] **Step 1 — add the regression and legal controls.**

Using existing makeFixture/selectPlan helpers, add this mapped mutation:
```cpp
TEST(PlanBinder, RejectsASelectedRuleWhosePredicateNoLongerMatches) {
  auto f = makeFixture();
  auto p = selectPlan(*f.context, *f.module, *f.target);
  ASSERT_TRUE(bool(p));
  auto b = bindPlan(*f.module, *p, *f.target);
  ASSERT_TRUE(bool(b));
  b->module->walk([&](Operation *op) {
    if (op->getName().getStringRef() == "micro.vector")
      op->setAttr("op", StringAttr::get(f.context.get(), "mul"));
  });
  llvm::Error e = verifyMappedMicroIR(*b->module, *f.target);
  ASSERT_TRUE(bool(e));
  EXPECT_NE(llvm::toString(std::move(e)).find("no_matching_rule"), std::string::npos);
}
```
Also mutate input dtype, a port access map, executor kind and an attached compute requirement. The unmodified mapped fixture must verify.

- [ ] **Step 2 — run the focused test before changing behavior.**

```sh
cmake --build build --target MappingRulesTest MappingPlanBinderTest -j 6
build/MappingRulesTest
build/MappingPlanBinderTest
```

Expected before the change: the new behavioral assertion fails (or the new interface test cannot compile). Keep existing legal controls green; record the observed failure.

- [ ] **Step 3 — implement this contract.**

Use the same predicate semantics as generation:
```cpp
if (rule.matchOp != node.opName)
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                "no_matching_rule: operation name differs");
for (const RulePredicate &p : rule.predicates)
  if (!evaluatePredicate(p, node))
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                  "no_matching_rule: predicate differs");
```
Evaluate recorded rule parameters against declared domains/constraints; unknown or missing required names reject. Machine checks verify executor kind, compute attachment/capability and memory kind/visibility. Do not merely confirm that IDs exist. Verification uses the original workload endpoints and strips bookkeeping attributes before interpreting rule predicates.

- [ ] **Step 4 — rerun the focused command and review the gate.**

Wrong mnemonic/dtype/map/capability reject consistently with generation; valid selected assignments still verify. Run relevant neighboring tests when the interface changes. No broad success claim follows merely from a new schema parsing.

- [ ] **Step 5 — commit the independently testable change.**

```sh
git add include/LLK/Mapping/MappingRules.h lib/Mapping/MappingRules.cpp lib/Mapping/PlanBinder.cpp test/Mapping/plan_binder.cpp test/Mapping/mapping_rules.cpp
git commit -m "fix(mapping): share complete rule legality with mapped verification"
```

### A4: Verify recorded layout assignments without choosing replacements

**Dependencies:** A2/A3

**Files:**
- Modify: include/LLK/Mapping/LayoutConstraints.h
- Modify: lib/Mapping/LayoutConstraints.cpp
- Modify: lib/Mapping/PlanBinder.cpp
- Test: test/Mapping/layout_constraints.cpp
- Test: test/Mapping/plan_binder.cpp

**Interfaces:** Produces verifySolvedLayout from the master. Exact endpoint type/rank supplies LayoutContext; params/maps are recorded state, not newly enumerated guesses.

- [ ] **Step 1 — add the regression and legal controls.**

Add a verifier test mutating only VW in the mapped vector layout:
```cpp
TEST(PlanBinder, RejectsTamperedSolvedVectorWidth) {
  auto f = makeFixture();
  ASSERT_TRUE(f.module);
  ASSERT_TRUE(f.target);
  auto p = selectPlan(*f.context, *f.module, *f.target);
  ASSERT_TRUE(bool(p)) << llvm::toString(p.takeError());
  auto b = bindPlan(*f.module, *p, *f.target);
  ASSERT_TRUE(bool(b)) << llvm::toString(b.takeError());
  auto legal = verifyMappedMicroIR(*b->module, *f.target);
  ASSERT_FALSE(bool(legal)) << llvm::toString(std::move(legal));
  unsigned mutations = 0;
  b->module->walk([&](Operation *op) {
    if (op->getName().getStringRef() != "micro.vector") return;
    auto mapping = op->getAttrOfType<DictionaryAttr>("micro.mapping");
    ASSERT_TRUE(mapping);
    auto layouts = mapping.getAs<DictionaryAttr>("layout_parameters");
    ASSERT_TRUE(layouts);
    NamedAttrList newLayouts(layouts);
    for (NamedAttribute entry : layouts) {
      auto parameters = dyn_cast<DictionaryAttr>(entry.getValue());
      if (!parameters || !parameters.getAs<IntegerAttr>("VW")) continue;
      NamedAttrList newParameters(parameters);
      newParameters.set("VW", IntegerAttr::get(
          IntegerType::get(f.context.get(), 64), 4));
      newLayouts.set(entry.getName(), newParameters.getDictionary(f.context.get()));
      ++mutations;
    }
    NamedAttrList newMapping(mapping);
    newMapping.set("layout_parameters", newLayouts.getDictionary(f.context.get()));
    op->setAttr("micro.mapping", newMapping.getDictionary(f.context.get()));
  });
  ASSERT_GT(mutations, 0u);
  auto error = verifyMappedMicroIR(*b->module, *f.target);
  ASSERT_TRUE(bool(error));
  EXPECT_NE(llvm::toString(std::move(error)).find("no_legal_layout"),
            std::string::npos);
}
```
In layout_constraints.cpp call verifySolvedLayout on avx2.blocked_2d with {M=4,N=8,VW=4}; assert an Error. Include out-of-domain, missing VW, unknown parameter, wrong rebuilt affine map and exhausted-quantifier cases.

- [ ] **Step 2 — run the focused test before changing behavior.**

```sh
cmake --build build --target MappingLayoutTest MappingPlanBinderTest -j 6
build/MappingLayoutTest
build/MappingPlanBinderTest
```

Expected before the change: the new behavioral assertion fails (or the new interface test cannot compile). Keep existing legal controls green; record the observed failure.

- [ ] **Step 3 — implement this contract.**

First validate names, types and declared-domain membership. Copy LayoutDef, pin each declared domain to the recorded singleton, and use solveLayout to evaluate that exact assignment:
```cpp
// Reject unknown/missing/domain-invalid parameters before pinning.
// solveLayout(pinnedDef, machine, context, portContext, limits)
// Reject errors, empty solutions, or undecided results.
// Rebuild expected map from the recorded parameters and compare to recorded map.
```
Never allow the solver to substitute a different legal VW. Check endpoint layout-family selection against its rule role. During v1 migration, rebuild a missing map only from a complete legal parameter assignment; ambiguous or missing assignments cannot pass executable verification.

- [ ] **Step 4 — rerun the focused command and review the gate.**

VW4 and map/role tampering reject; all legal non-first-domain solutions can verify, not just the enumerator’s preferred solution. Run relevant neighboring tests when the interface changes. No broad success claim follows merely from a new schema parsing.

- [ ] **Step 5 — commit the independently testable change.**

```sh
git add include/LLK/Mapping/LayoutConstraints.h lib/Mapping/LayoutConstraints.cpp lib/Mapping/PlanBinder.cpp test/Mapping/layout_constraints.cpp test/Mapping/plan_binder.cpp
git commit -m "fix(mapping): validate persisted solved layout assignments"
```

### A5: Validate materialized movement identity before exempting coverage

**Dependencies:** A2/A3

**Files:**
- Modify: lib/Mapping/PlanBinder.cpp
- Test: test/Mapping/plan_binder.cpp
- Test: test/Conversion/MicroMapping/verify_mapping_invalid.mlir

**Interfaces:** Consumes endpoint-aware ConnectionPlan. A movement exemption requires a recognized canonical movement op and a matching selected connection/hop; micro.value alone grants nothing.

- [ ] **Step 1 — add the regression and legal controls.**

Mutation on a valid mapped vector:
```cpp
TEST(PlanBinder, ValueStampCannotExemptAnUnmappedComputeOperation) {
  auto f = makeFixture();
  ASSERT_TRUE(f.module);
  ASSERT_TRUE(f.target);
  auto p = selectPlan(*f.context, *f.module, *f.target);
  ASSERT_TRUE(bool(p)) << llvm::toString(p.takeError());
  auto b = bindPlan(*f.module, *p, *f.target);
  ASSERT_TRUE(bool(b)) << llvm::toString(b.takeError());
  auto legal = verifyMappedMicroIR(*b->module, *f.target);
  ASSERT_FALSE(bool(legal)) << llvm::toString(std::move(legal));
  unsigned mutations = 0;
  b->module->walk([&](Operation *op) {
    if (op->getName().getStringRef() != "micro.vector") return;
    op->removeAttr("micro.mapping");
    op->setAttr("micro.value", IntegerAttr::get(
        IntegerType::get(f.context.get(), 64), 0));
    ++mutations;
  });
  ASSERT_EQ(mutations, 1u);
  auto error = verifyMappedMicroIR(*b->module, *f.target);
  ASSERT_TRUE(bool(error));
  EXPECT_NE(llvm::toString(std::move(error)).find("no_matching_rule"),
            std::string::npos);
}
```
Assert verification fails with missing mapping. Add ordinary user copy with a forged value stamp, nonexistent connection/hop ID, wrong source/destination, unsupported engine and a correctly materialized copy/wait control.

- [ ] **Step 2 — run the focused test before changing behavior.**

```sh
cmake --build build --target MappingPlanBinderTest llk-opt -j 6
build/MappingPlanBinderTest
ctest --test-dir build -R MicroMappingVerify --output-on-failure
```

Expected before the change: the new behavioral assertion fails (or the new interface test cannot compile). Keep existing legal controls green; record the observed failure.

- [ ] **Step 3 — implement this contract.**

Replace the unconditional stamp exemption with typed operation + connection validation:
```cpp
// if no micro.mapping:
//   reject unless this is a canonical materialized connection operation;
//   resolve micro.connection, micro.hop, source/destination endpoint/node;
//   verify type, engine/link, selected consumers and actual SSA operand uses;
//   otherwise report invalid_mapping_metadata / no_matching_rule.
```
Use checked attribute reads for every field. Add explicit connection/hop stamps to existing tensor copies. B1 upgrades persisted route metadata; do not relax verification to accommodate old ambiguous data.

- [ ] **Step 4 — rerun the focused command and review the gate.**

Compute spoofing is rejected; genuine connection operations retain their completeness exemption with validated provenance. Run relevant neighboring tests when the interface changes. No broad success claim follows merely from a new schema parsing.

- [ ] **Step 5 — commit the independently testable change.**

```sh
git add lib/Mapping/PlanBinder.cpp test/Mapping/plan_binder.cpp test/Conversion/MicroMapping/verify_mapping_invalid.mlir
git commit -m "fix(mapping): require verified connection stamps for materialized ops"
```

### A6: Disclose all restricted exact connection choices

**Dependencies:** Baseline

**Files:**
- Modify: lib/Mapping/CoveringSearch.cpp
- Test: test/Mapping/covering_search.cpp
- Test: test/Mapping/plan_report.cpp

**Interfaces:** No new search semantics yet. Set connectionChoicesUnexplored and ConnectionChoiceUnexplored for every locally collapsed alternative list, including gather feeds.

- [ ] **Step 1 — add the regression and legal controls.**

Use the review's fanInGraph/fanMachine/kFanRules helpers in covering_search.cpp:
```cpp
// Add a second legal parallel route, then run exact with caps lifted.
EXPECT_TRUE(result->connectionChoicesUnexplored);
EXPECT_FALSE(result->searchTruncated);
```
Require the plan report notice. A single-route gather must not set the flag; a capped route list must additionally report truncation.

- [ ] **Step 2 — run the focused test before changing behavior.**

```sh
cmake --build build --target MappingCoveringSearchTest MappingPlanReportTest -j 6
build/MappingCoveringSearchTest
build/MappingPlanReportTest
```

Expected before the change: the new behavioral assertion fails (or the new interface test cannot compile). Keep existing legal controls green; record the observed failure.

- [ ] **Step 3 — implement this contract.**

Centralize the decision used by plain, fan-out and gather code:
```cpp
// Whenever exact mode retains only one of alternatives.size() > 1:
//   result.connectionChoicesUnexplored = true;
//   add the stable connection_choice_unexplored notice once.
// Propagate route-enumeration truncation independently.
```
Do not imply exhaustive infeasibility when a restricted search finds no plan. B7 later removes the restriction by branching; this task first makes current behavior honest.

- [ ] **Step 4 — rerun the focused command and review the gate.**

Every local exact collapse is visible, including rejected branches and gather feeds. Run relevant neighboring tests when the interface changes. No broad success claim follows merely from a new schema parsing.

- [ ] **Step 5 — commit the independently testable change.**

```sh
git add lib/Mapping/CoveringSearch.cpp test/Mapping/covering_search.cpp test/Mapping/plan_report.cpp
git commit -m "fix(mapping): report unexplored exact gather alternatives"
```

### A7: Evaluate constraints by required facts and explicit roles

**Dependencies:** Baseline; A1 for endpoint-role projections

**Files:**
- Modify: include/LLK/Perf/Legality.h
- Modify: lib/Perf/Legality.cpp
- Modify: lib/Conversion/MicroMapping/SearchBindingLoader.cpp
- Modify: lib/Conversion/LLKToMicro/LLKToMicro.cpp
- Test: test/Conversion/MicroMapping/search_binding_loader.cpp
- Test: test/Perf/legality.cpp

**Interfaces:** Produces BindingFacts/checkConstraint overload from master. Role checks use SearchConstraint.params and SearchParam.role; do not use unique-kind lookup for multi-role declarations.

- [ ] **Step 1 — add the regression and legal controls.**

Register the review's valid vector-only mapping_extent fixture. Also directly test two referenced layout parameters:
```cpp
SearchSpace space;
space.params.push_back(SearchParam{"lhs_layout", "layout", {SearchChoice("row_major")}, "operand0"});
space.params.push_back(SearchParam{"rhs_layout", "layout", {SearchChoice("blocked")}, "operand1"});
Candidate c;
c.symbolicValues["lhs_layout"] = "row_major";
c.symbolicValues["rhs_layout"] = "blocked";
// Machine SRAM supports row_major only. layout_supported referencing both
// parameters must reject rhs_layout, rather than treat ambiguity as absence.
```
Add two-MMAs with different shapes, missing original dimensions for tail checks, and one explicit shape-independent legal control.

- [ ] **Step 2 — run the focused test before changing behavior.**

```sh
cmake --build build --target SearchBindingLoaderTest LegalityTest llk-opt -j 6
build/SearchBindingLoaderTest
build/LegalityTest
ctest --test-dir build -R MicroMappingCandidate --output-on-failure
```

Expected before the change: the new behavioral assertion fails (or the new interface test cannot compile). Keep existing legal controls green; record the observed failure.

- [ ] **Step 3 — implement this contract.**

Dispatch constraint requirements explicitly:
```cpp
// mapping_extent/vector-width/owner availability: no contraction required.
// per-port layout_supported: evaluate each referenced parameter/role.
// contraction dtype/fragment requirements: evaluate each applicable contraction.
// global divisibility/tails: use originalWorkload; absent required facts reject.
```
Collect all contraction facts; export original workload dimensions before tiling. Keep old shape-based checkLegality as a wrapper constructing BindingFacts. Do not manufacture GEMM dimensions for vector kernels. Reject duplicate role declarations and unresolved role references specifically. Layout-kind bridging remains target-data driven.

- [ ] **Step 4 — rerun the focused command and review the gate.**

Constrained vector mapping succeeds; capacity violations and unsupported second layout roles reject; no first-MMA or unique-kind bypass remains. Run relevant neighboring tests when the interface changes. No broad success claim follows merely from a new schema parsing.

- [ ] **Step 5 — commit the independently testable change.**

```sh
git add include/LLK/Perf/Legality.h lib/Perf/Legality.cpp lib/Conversion/MicroMapping/SearchBindingLoader.cpp lib/Conversion/LLKToMicro/LLKToMicro.cpp test/Conversion/MicroMapping/search_binding_loader.cpp test/Perf/legality.cpp
git commit -m "fix(mapping): evaluate persistent constraints from scoped workload facts"
```

### A8: Reject unsupported offset views before identity lowering

**Dependencies:** Baseline

**Files:**
- Modify: lib/Conversion/MicroToLinalg/MicroToLinalg.cpp
- Test: test/Conversion/MicroToLinalg/micro_to_linalg_invalid.mlir
- Test: test/Conversion/MicroToLinalg/micro_to_linalg.mlir

**Interfaces:** No API change. Identity lowering requires equal shape and provably zero offsets; supported extract-slice semantics are added in C3.

- [ ] **Step 1 — add the regression and legal controls.**

Add the review fixture with equal 8x8 source/result shapes and offsets [1,0]; it must fail conversion. Add dynamic-offset rejection and an explicit [0,0] control:
```mlir
%z = arith.constant 0 : index
%v = micro.tile_view %a[%z, %z] {shape = array<i64: 8, 8>} : tensor<8x8xf32> -> !micro.tile<8x8xf32, memory = #micro.memory<sram>>
```
The legal control must replace the view with %a; an unsupported window must produce a diagnostic, not return the whole source.

- [ ] **Step 2 — run the focused test before changing behavior.**

```sh
cmake --build build --target llk-opt -j 6
ctest --test-dir build -R MicroToLinalg --output-on-failure
```

Expected before the change: the new behavioral assertion fails (or the new interface test cannot compile). Keep existing legal controls green; record the observed failure.

- [ ] **Step 3 — implement this contract.**

Use MLIR constant matching for each offset before replacing the op:
```cpp
// identityView = equal source/result shape &&
//   every supplied offset is a constant index zero;
// if !identityView, notifyMatchFailure with unsupported slice reason.
```
Unknown offsets cannot prove identity. Keep dimensionality/bounds verification separate from this safe lowering decision; C3 owns actual slice support.

- [ ] **Step 4 — rerun the focused command and review the gate.**

Nonzero/dynamic offsets no longer silently disappear; zero-offset and offset-free identity controls still lower. Run relevant neighboring tests when the interface changes. No broad success claim follows merely from a new schema parsing.

- [ ] **Step 5 — commit the independently testable change.**

```sh
git add lib/Conversion/MicroToLinalg/MicroToLinalg.cpp test/Conversion/MicroToLinalg/micro_to_linalg_invalid.mlir test/Conversion/MicroToLinalg/micro_to_linalg.mlir
git commit -m "fix(micro): reject offset views in identity-only lowering"
```

### A9: Charge materialized transforms and preserve their dependencies

**Dependencies:** Baseline; A2 for endpoint-resolved cost comparison

**Files:**
- Modify: include/LLK/Perf/MicroDAG.h
- Modify: lib/Perf/MicroDAG.cpp
- Modify: include/LLK/Mapping/CostModel.h
- Modify: lib/Mapping/CostModel.cpp
- Test: test/Perf/l1_resource_dag.cpp
- Test: test/Mapping/cost_model.cpp

**Interfaces:** Adds EventKind::Transform mapped to CostEventKind::Transform. Introduce TransformCostInput and estimateTransformCost in CostModel.h; both planner and perf use it.

- [ ] **Step 1 — add the regression and legal controls.**

Register this transform-only regression using the existing parseKernel/parseMachine helpers, then add a copy -> transform -> vector chain:
```cpp
TEST(L1ResourceDAG, TransformIsAChargedEvent) {
  auto parsed = parseKernel(R"mlir(
module {
  micro.kernel @transform_only {
    %a = tensor.empty() : tensor<8x8xf32>
    %t = micro.transform %a {src_map = affine_map<(d0,d1)->(d0,d1)>, dst_map = affine_map<(d0,d1)->(d1,d0)>} : tensor<8x8xf32> -> tensor<8x8xf32>
    micro.yield
  }
}
)mlir");
  ASSERT_TRUE(parsed);
  auto model = parseMachine(testMachine(1, 1, 1));
  auto dag = buildMicroDAG(parsed->kernel, model);
  ASSERT_TRUE(bool(dag)) << llvm::toString(dag.takeError());
  ASSERT_EQ(dag->events.size(), 1u);
  EXPECT_EQ(dag->events.front().kind, EventKind::Transform);
  EXPECT_EQ(dag->events.front().costKind, mapping::CostEventKind::Transform);
  EXPECT_GT(dag->events.front().minCycles, 0u);
}
```
The chain fixture retains this operation:
```mlir
%t = micro.transform %a {src_map = affine_map<(d0,d1)->(d0,d1)>, dst_map = affine_map<(d0,d1)->(d1,d0)>} : tensor<8x8xf32> -> tensor<8x8xf32>
```
Assert one Transform event, positive cost for a nonidentity conversion, fresh output storage and a vector dependency on that event. Identity-map transforms may have zero arithmetic cost only if modeled explicitly; they must still propagate producers.

- [ ] **Step 2 — run the focused test before changing behavior.**

```sh
cmake --build build --target L1ResourceDAGTest MappingCostModelTest micro-perf -j 6
build/L1ResourceDAGTest
build/MappingCostModelTest
```

Expected before the change: the new behavioral assertion fails (or the new interface test cannot compile). Keep existing legal controls green; record the observed failure.

- [ ] **Step 3 — implement this contract.**

Define the shared cost input with input/output types, source/destination maps, memory node and selected compute resource. Estimate bytes/work from checked static facts and machine capabilities; an unknown resource or footprint is a diagnostic, not zero. Build and register the event:
```cpp
// event.kind = EventKind::Transform;
// event.costKind = CostEventKind::Transform;
// event.deps = producers of transform source;
// event.minCycles = rounded shared estimate;
// producers[transform result] = added event id;
```
Stamp the transform's selected resource during binding. Use an explicit machine/default policy for an unmapped reference kernel; do not branch on target names. B3 completes physical padding/lifetime accounting and B8 compares full plan event streams.

- [ ] **Step 4 — rerun the focused command and review the gate.**

The review transform fixture cannot report an empty event set; consumer schedule starts after transform completion. Run relevant neighboring tests when the interface changes. No broad success claim follows merely from a new schema parsing.

- [ ] **Step 5 — commit the independently testable change.**

```sh
git add include/LLK/Perf/MicroDAG.h lib/Perf/MicroDAG.cpp include/LLK/Mapping/CostModel.h lib/Mapping/CostModel.cpp test/Perf/l1_resource_dag.cpp test/Mapping/cost_model.cpp
git commit -m "fix(perf): model materialized layout transforms"
```

## Stage A release gate

- [ ] Run all named regressions and the full registered suite.
- [ ] Re-run the review's seven probes against the delivery head; retire their defective outputs with explicit evidence.
- [ ] Check static MLIR linking on CI LLVM 22.1.8 and the local LLVM 24 build.
- [ ] Publish scope accurately: A fixes correctness; output associations, complete tile materialization, full exact branching and target execution remain B/C work.
