# Issue #129 Resources and Static Costs Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close G1/G2/G6/G7/G8 by making selected identity, physical feasibility, final search scoring and bound-kernel performance analysis describe the same execution.

**Architecture:** Persist concrete bindings; reserve every route hop; derive live occupancy from execution events rather than a serial topological index. A completion callback in CoveringSearch evaluates a canonical materialized preview above the mapping core, using the same event extraction and scheduling as micro-perf before top-K retention.

**Tech Stack:** C++20, LLKMapping, LLKMicroMapping, LLKPerf, MLIR Micro-IR, GTest, versioned JSON/MLIR metadata.

**Spec:** [Master plan and gap matrix](2026-10-07-issue129-gap-closure.md); normative design §§9–18, 22, 25, 27.6, 29; existing B1/B3/B4/B7/B8 contracts in the October 4 plan.

## Global Constraints

- “`micro` remains the only canonical execution IR.”
- “Target policy lives in target packages.”
- “Repeated runs with identical inputs and configuration produce byte-identical normalized IR and plan reports.”
- All global constraints and baseline pins in the [master plan](2026-10-07-issue129-gap-closure.md) apply. In particular, no LLKMapping -> LLKPerf/Runtime dependency is permitted.
- Strict executable analysis rejects missing facts. Partial analysis preserves a reason and cannot claim resource feasibility.

## Review Focus

1. Same-kind nodes attached to one executor must not collapse to the first node: R1/R2/R8.
2. Renamed files with identical target content must replay; unchanged filenames with changed content must fail: R1/R8.
3. Sequential iterations reuse a buffer; simultaneous stages/owners reserve all live versions: R4/R5.
4. A cheaper infeasible covering must not hide a feasible exact topK=1 result: R7/R8.
5. Unknown engines, dynamic footprints and bounded event expansion must fail strict evaluation with stable reasons: R2/R3/R6/R7.

---

## File responsibilities and shared additions

`MappingPlan.h/.cpp`, metadata/report/verification own persisted selection. `StoragePlan.cpp` owns route allocations and dependency construction. New `StorageLiveness.h/.cpp` owns occupancy from scheduled events and reuse-order validation. `SelectedKernelAnalysis.h/.cpp` in LLKPerf owns canonical static analysis; new `CompletePlanEvaluation.h/.cpp` in LLKMicroMapping owns preview binding and final plan evaluation. CoveringSearch calls an injected interface and knows neither dialect conversion nor perf internals.

The test-support file `test/Mapping/resource_regression_fixture.h/.cpp` is created in R1 and linked only by integration regression tests. It constructs the small scenarios listed below directly from finite MachineModel data, LLKMap text and parsed Micro-IR. It does not call production search/storage to compute expected results.

```cpp
// Test-only, namespace issue129. Created and implemented in R1.
struct ResourceCase {
  std::unique_ptr<mlir::MLIRContext> context;
  mlir::OwningOpRef<mlir::ModuleOp> source;
  mlir::llk::mapping::WorkloadGraph graph;
  std::unique_ptr<mlir::llk::mapping::MappingTarget> target;
};
llvm::Expected<ResourceCase> resourceCase(llvm::StringRef name);
// Allowed names, each owned by its task:
// R1: two-compute; R3: missing-memory, named-ports;
// R4: two-hop; R5: sequential, pipeline-four, parallel-overlap;
// R7: capacity-topk; R8: joint-oracle.
llvm::Expected<mlir::llk::mapping::MappingSearchResult>
searchCase(ResourceCase &c,
           const mlir::llk::mapping::MappingSearchOptions &options);
// searchCase extracts the fixture's actual source graph and supplies the
// completion callback once R7 defines it. Before R7 it uses ordinary search.
```

New production interfaces are introduced only by their owning tasks; declarations here fix their cross-task meaning:

```cpp
// R6: include/LLK/Perf/SelectedKernelAnalysis.h, namespace mlir::llk::perf.
struct SelectedKernelAnalysis {
  mapping::PlanEventDAG events;
  mapping::EventScheduleResult schedule;
  mapping::Cost cost;
  std::map<std::string, uint64_t> trafficBytes;
  std::map<std::string, uint64_t> peakBytes;
  bool complete = false;
  std::vector<std::string> incompleteReasons;
};
llvm::Expected<SelectedKernelAnalysis>
analyzeSelectedKernel(mlir::Operation *kernel,
                      const machine::MachineModel &machine,
                      bool requireComplete);

// R7: include/LLK/Mapping/CoveringSearch.h, namespace mlir::llk::mapping.
struct CompletePlanEvaluation {
  std::optional<CoveringPlan> plan;
  std::optional<Diagnostic> rejection;
};
using CompletePlanEvaluator = std::function<
  llvm::Expected<CompletePlanEvaluation>(const CoveringPlan &)>;
// MappingSearchOptions gains CompletePlanEvaluator evaluateCompletePlan;
// and bool requireCompleteEvaluation = false; both default initialized.

// R7: CompletePlanEvaluation.h, namespace mlir::llk::mapping.
llvm::Expected<CompletePlanEvaluation>
evaluateCompletePlan(mlir::ModuleOp source, const WorkloadGraph &graph,
                     const MappingTarget &target, const CoveringPlan &proposal,
                     BindContract contract);
```

`CompletePlanEvaluation` has exactly one of a finalized `plan` or candidate `rejection`. An `llvm::Error` means invalid invocation/configuration or an infrastructure failure that must stop search. Storage overflow, unresolved strict facts and unsupported materialization are candidate rejections with existing stable diagnostic codes (extend Diagnostics only when no existing code fits). A callback must not recursively invoke CoveringSearch or write files.

### Task R1: Preserve selected compute nodes through every durable boundary

**Files:** Modify `include/LLK/Mapping/MappingPlan.h`, `lib/Mapping/{MappingPlan,CoveringSearch,MappingMetadata,PlanReport,PlanVerification,LatencyProvider,CostEvent}.cpp`; test `test/Mapping/{covering_search,plan_report,plan_binder,stable_hash,cost_model}.cpp`; create the test-support files above; register their integration target in root `CMakeLists.txt`.

**Interfaces:** Consumes existing `CandidateInstance::computeBindings`, `computePlanId`, report and metadata codecs. Produces `PlanPlacement::computeBindings` with the same requirement-key -> concrete-node meaning, sorted canonical serialization, and explicit schema migration diagnostics. Later tasks use this field; no executor-based first-engine fallback is allowed for strict mapped work.

- [ ] **Step 1: Add the two-compute fixture and failing identity assertion.** Use one worker `e0`, two attached vector engines `vpu.a` and `vpu.b`, both concurrency one, an `8x8xf32` add, and a rule requiring `vector_engine`. Enumerate exact with topK=8. Implement the test-support constructor by parsing the fixture and filling these MachineModel nodes; expected resources are literal values, not inferred by a scorer.

```cpp
auto c = issue129::resourceCase("two-compute");
ASSERT_TRUE(c);
MappingSearchOptions options;
options.mode = SearchMode::Exact;
options.topK = 8;
auto result = issue129::searchCase(*c, options);
ASSERT_TRUE(result);
ASSERT_EQ(result->plans.size(), 2u);
std::set<std::string> selected;
for (const auto &plan : result->plans)
  selected.insert(plan.placements.front().computeBindings.lookup("vector_engine"));
EXPECT_EQ(selected, (std::set<std::string>{"vpu.a", "vpu.b"}));
EXPECT_NE(result->plans[0].id, result->plans[1].id);
```

- [ ] **Step 2: Run `cmake --build build --target MappingCoveringSearchTest MappingPlanReportTest MappingPlanBinderTest` and the new filtered regression.** The initial compile fails for the absent placement field; after adding the field, the assertion must fail until propagation is implemented. Record both transitions.
- [ ] **Step 3: Add and propagate the field.** Copy the selected instance's compute map when constructing every placement, including every node of a fused instance. Include sorted, length-delimited compute bindings in canonical plan identity and operation/connection signatures; persist them in both JSON and `micro.mapping`. Give the added container a default value and retain explicit SmallVector inline capacities.

```cpp
// At instance -> placement construction; do not query the machine again.
placement.computeBindings = instance.computeBindings;
// At normalization: validate the recorded node against the required kind,
// attachment/visibility and selected executor before using its id.
```

- [ ] **Step 4: Version incompatible state and remove derived-score identity.** Bump `kPlanReportVersion`, `kMappingMetadataVersion`, `kSupportedMappingMetadataVersion` and `kCostModelVersion` from two to three. Later tasks in this repair use the v3 contract. Remove localCost/totalCost and diagnostic/truncation fields from v3 instance/plan canonical identity; include explicit schema and source/target/binding provenance. Preserve decision-bearing compute/layout/port/storage choices. Scores remain reported and rederived, so a preview does not need its final score in order to acquire a valid ID. Reject old report/metadata at strict replay with an unsupported-schema diagnostic and regeneration guidance; do not guess which attached compute was selected. Unknown IDs, wrong kinds, unattached compute nodes and differing bindings inside one instance fail verification.
- [ ] **Step 5: Extend round-trip/key tests.** Each resource survives report read, metadata decode and canonical binding; fresh source/target verification rejects changed machine content. Reverse engine insertion order and compare canonical selections/reports. Change only selected engine and assert key inequality; change only an estimate/diagnostic and assert instance/plan identity stays unchanged. The earlier first-engine probe must now produce `vpu.a` and `vpu.b` respectively.
- [ ] **Step 6: Run `ctest --test-dir build --output-on-failure -R 'Mapping(CoveringSearch|PlanReport|PlanBinder|StableHash|CostModel)Test'`.** Commit with `fix: persist selected compute bindings in mapping plans`. Review gate: no recorded selection is re-derived from executor ordering.

### Task R2: Schedule each named DMA node with its own concurrency

**Files:** Modify `lib/Mapping/EventSchedule.cpp`, `include/LLK/Mapping/EventSchedule.h`; tests `test/Mapping/cost_model.cpp`, `test/Perf/l1_resource_dag.cpp`.

**Interfaces:** Consumes existing `MachineModel::findTransferEngine` and `TransferEngineNode::count`. Produces the unchanged scheduling API plus `llvm::Error validateEventResources(const PlanEventDAG &, const MachineModel &)` for strict callers. Analysis-only legacy unnamed pools remain clearly partial.

- [ ] **Step 1: Write the direct occupancy regression.** Build `dma.a(count=1)` and unused `dma.b(count=1)` attached to one executor. No links are needed for the scheduler-only test.

```cpp
std::vector<PlanCostEvent> events = {
  makePlanCostEvent(CostEventKind::TransferHop, "dma.a", 10, 0, 256),
  makePlanCostEvent(CostEventKind::TransferHop, "dma.a", 10, 0, 256)};
auto withUnused = scheduleNormalizedEvents(events, machine);
machine.transferEngines.pop_back(); // removes dma.b, not dma.a
auto alone = scheduleNormalizedEvents(events, machine);
EXPECT_EQ(alone.predictedCycles, 20u);
EXPECT_EQ(withUnused.predictedCycles, alone.predictedCycles);
EXPECT_EQ(withUnused.entries[1].start, 10u);
```

- [ ] **Step 2: Run `build/MappingCostModelTest --gtest_filter='*Dma*'`.** Baseline schedules both events at zero when the unused node exists.
- [ ] **Step 3: Replace machine-wide pool slots with the named node's count.** Keep keys `dma/<id>`. Compute/transform pools similarly require their recorded compute nodes; validate positive counts and event dependencies for strict input. Do not make cost scheduling itself decide layout or route legality.

```cpp
// In the TransferHop case of poolFor:
const auto *engine = machine.findTransferEngine(event.event.resource);
return {"dma/" + event.event.resource,
        std::max<uint32_t>(1, engine ? engine->count : 1)};
// validateEventResources rejects a missing engine before a strict call.
```

- [ ] **Step 4: Add controls:** count=2 on the same node gives ten cycles; one event each on `dma.a`/`dma.b` overlaps; a dependency serializes independent engines; unknown selected engine fails strict validation; insertion order does not alter scheduling. Add the corresponding bound-kernel L1 test so both API paths use the repair.
- [ ] **Step 5: Run `ctest --test-dir build --output-on-failure -R 'MappingCostModelTest|L1ResourceDAGTest'`.** Commit with `fix: respect per-node transfer concurrency`. Review gate: no pool uses whole-machine count as the concurrency of one named engine.

### Task R3: Require complete per-port physical memory facts for executable plans

**Files:** Modify `lib/Conversion/MicroMapping/MicroMappingCommon.h`, `lib/Mapping/{StoragePlan,PlanVerification,PlanBinder,Placement}.cpp`, `mapping/{x86-avx2,generic-accelerator}/rules.llkmap`; tests `test/Mapping/{storage_plan,plan_binder,placement}.cpp`; create `test/Conversion/MicroMapping/strict_missing_memory.mlir`.

**Interfaces:** Consumes `PortRef`, `PortMemoryBinding`, explicit tile memory/owner and target visibility. Produces resolved input/output memory bindings for every materialized value and a strict readiness verdict. This is independent of whether an emitter has a hardware backend.

- [ ] **Step 1: Add missing-memory and named-ports fixtures.** Missing-memory has a vector rule with no port memory declaration and a tile memory kind that resolves to two accessible nodes. Named-ports declares two same-kind inputs on different nodes plus its output. Assert partial bind retains a reason and strict bind rejects ambiguity; assert explicit per-port selections survive.

```cpp
auto c = issue129::resourceCase("missing-memory");
ASSERT_TRUE(c);
MappingSearchOptions options;
options.mode = SearchMode::Exact;
auto result = issue129::searchCase(*c, options);
ASSERT_TRUE(result);
ASSERT_FALSE(result->plans.empty());
auto materializer = createCanonicalPlanMaterializer();
auto strict = bindPlan(*c->source, result->plans.front(), *c->target,
                       BindContract::Executable, materializer.get());
EXPECT_FALSE(strict);
if (!strict)
  EXPECT_NE(llvm::toString(strict.takeError()).find("memory"), std::string::npos);
```

- [ ] **Step 2: Run `build/MappingPlanBinderTest --gtest_filter='*PhysicalMemory*'` and the new FileCheck negative.** Baseline can succeed while storage notes say planning was skipped.
- [ ] **Step 3: Resolve memory facts by endpoint.** Prefer named rule requirements; otherwise use the value's explicit memory kind only when the selected executor sees exactly one compatible node, including byte/alignment/layout checks. Treat boundary descriptors as borrowed allocations with named nodes and known spans. Unknown kind, ambiguous nodes, dynamic unknown footprint and overflow are strict rejections; do not silently bind all ports to the first memory of a class.

```cpp
// Strict resolution has three outcomes: exactly one node -> bind;
// zero nodes -> inaccessible-memory diagnostic;
// more than one node -> ambiguous-memory diagnostic requiring named ports.
// Bare instance-wide requirements are compatibility inputs, never authority
// for two endpoint occurrences that selected different nodes.
```

- [ ] **Step 4: Remove the executable storage-skip escape.** Partial analysis records `physical_complete=false` and ordered reasons. Direct strict bindPlan as well as the pass/compiler adapters must demand verified physical completeness; guard this in PlanBinder/PlanVerification rather than only the CLI adapter. `report-only` still records the evaluation contract and incomplete status instead of claiming a fully executable plan.
- [ ] **Step 5: Update shipped rules with the necessary named ports.** Inspect each rule's actual input/output signature; add memory requirements consistent with the input tile memory and machine visibility. Do not force every operation into SRAM or replace symbolic kinds with fixed concrete node names. Version/content hashes invalidate changed rules automatically.
- [ ] **Step 6: Run placement/storage/binder tests and map all existing four public fixtures strictly.** Remaining unsupported fixtures fail with precise reasons until their owning tasks repair them. Commit with `fix: require complete physical memory bindings for executable mapping`. Review gate: no strict success carries “storage planning was skipped”.

### Task R4: Allocate and materialize every movement hop

**Files:** Modify `include/LLK/Mapping/MappingPlan.h`, `lib/Mapping/{StoragePlan,MappingMetadata,PlanReport,PlanVerification}.cpp`, `lib/Conversion/MicroMapping/PlanMaterialization.cpp`; tests `test/Mapping/{storage_plan,plan_binder,plan_report}.cpp`; create `test/Conversion/MicroMapping/physical_two_hop.mlir` and matching machine/rule fixtures under `test/Mapping/Inputs/issue129/`.

**Interfaces:** Add `PlanMovementHop` with `index`, `srcMemory`, `dstMemory`, `engine`, `sourceStorageId`, `destinationStorageId`, `movementStep`, `waitStep` (numeric defaults zero; strings empty). `PlanConnection::hops` is an ordered vector. Add optional hop index to movement/synchronization PlanStep. `storageIds` remains a compatibility projection; `hops` is materialization authority.

- [ ] **Step 1: Turn the audit's SRAM→L2→DRAM probe into a checked-in fixture.** A 256-byte value traverses two links with distinct DMA engines; L2 capacity is 256. A second fixture reduces L2 to 128. Require a selected L2 allocation and reject the undersized intermediate.

```cpp
auto c = issue129::resourceCase("two-hop");
ASSERT_TRUE(c);
MappingSearchOptions options;
options.mode = SearchMode::Exact;
auto result = issue129::searchCase(*c, options);
ASSERT_TRUE(result);
auto plan = result->plans.front();
ASSERT_FALSE(finalizeStoragePlan(c->graph, plan, c->target->machine()));
const auto &route = plan.connectionPlans.front();
ASSERT_EQ(route.hops.size(), 2u);
EXPECT_EQ(route.hops[0].dstMemory, "l2.0");
EXPECT_EQ(route.hops[1].sourceStorageId, route.hops[0].destinationStorageId);
EXPECT_TRUE(llvm::any_of(plan.allocations, [](const auto &a) {
  return a.memory == "l2.0" && a.bytes == 256;
}));
```

- [ ] **Step 2: Run storage/binder filtered two-hop tests.** Baseline reserves only the route destination; the new hop field initially causes a compile failure.
- [ ] **Step 3: Build one destination allocation and movement/wait pair per hop.** Link the intermediate's last use to the next hop's read completion. Include transform allocations at the actual transform position and replicate/gather output allocations. Preserve direct-edge compute dependencies; direct does not mean dependency-free. Use solved physical maps for padded/banked footprints and checked byte arithmetic.
- [ ] **Step 4: Make materialization consume these hop decisions.** Emit the copy into the recorded destination allocation, stamp connection/value/hop/storage IDs and concrete source/destination nodes, and rewire the exact consumer PortRefs. A missing hop allocation or mismatched route length rejects strict replay. Do not allocate a new unrelated buffer simply because the route names the same memory kind.

```cpp
// Movement materialization iterates connection.hops in route order.
// copy[h] reads sourceStorageId, writes destinationStorageId;
// wait[h] orders the next reader; the final reader is the selected consumer.
// Existing allocation materialization maps StorageAllocation::id -> SSA value.
```

- [ ] **Step 5: Add report/metadata round trips, fan-out and capacity controls.** Two routes sharing L2 must retain distinct allocations unless aliasing is proved; same-kind L2 nodes remain separate. Re-finalization is idempotent. Mutating a hop engine/storage ID is rejected. All added fields use v3 serialization established by R1; include physical choices in the canonical identity, while diagnostics/derived timing remain excluded.
- [ ] **Step 6: Run `ctest --test-dir build --output-on-failure -R 'Mapping(StoragePlan|PlanBinder|PlanReport)Test|PhysicalTwoHop'`.** Commit with `fix: reserve and bind storage for every transfer hop`. Review gate: each materialized movement has a corresponding source/destination storage decision.

### Task R5: Verify storage liveness under actual resource overlap

**Files:** Create `include/LLK/Mapping/StorageLiveness.h`, `lib/Mapping/StorageLiveness.cpp`; modify `MappingPlan.h`, `CostEvent.h`, `StoragePlan.cpp`, metadata/report codecs, materialization and root CMake; test `test/Mapping/storage_liveness.cpp` registered as `MappingStorageLivenessTest`.

**Interfaces:** Extend normalized events with default-initialized `owner`, `srcMemory`, `dstMemory`, `sourceNode`, `occurrence`, optional `planStep`/`connectionId` (uint64_t IDs) and `hopIndex=0`, plus `storageUses`. A `StorageUse` has `uint64_t allocationId=0`, `uint64_t occurrence=0`, and `StorageAccess access=StorageAccess::Read`, where `StorageAccess { Read, Write }`. Add `StorageLivenessResult` with `std::map<std::string,uint64_t> peakBytes` and `std::vector<PlanStepEdge> requiredReuseEdges`, and:

```cpp
llvm::Expected<StorageLivenessResult>
analyzeStorageLiveness(const CoveringPlan &plan, const PlanEventDAG &events,
                       const EventScheduleResult &schedule);
```

An occurrence distinguishes loop iteration, owner and pipeline stage deterministically. Sequential occurrences may reuse storage; simultaneous occurrences cannot. Existing beginStep/endStep become report projections of this stronger relation, not a proof of concurrency safety.

- [ ] **Step 1: Add three event/lifetime fixtures.** Sequential four iterations of one 256-byte temporary peak at 256; four simultaneously resident pipeline versions peak at 1024; two parallel 256-byte copies overlapping in 512-byte L2 peak at 512 and fail when capacity is 256. Add a padded layout whose physical bytes exceed logical bytes.

```cpp
auto live = analyzeStorageLiveness(plan, dag, schedule);
ASSERT_TRUE(live);
EXPECT_EQ(live->peakBytes.at("l2.0"), 512u);
// For the sequential fixture:
auto serialLive = analyzeStorageLiveness(serialPlan, serialDag, serialSchedule);
ASSERT_TRUE(serialLive);
EXPECT_EQ(serialLive->peakBytes.at("l2.0"), 256u);
```

- [ ] **Step 2: Build/run `MappingStorageLivenessTest`.** The baseline has no event-based liveness and cannot demonstrate these expected peaks.
- [ ] **Step 3: Compute live intervals from producer start through final reader completion, including wait/barrier, yields and carried values.** Borrowed input/output descriptors are never considered reusable compiler scratch. Enumerate bounded occurrences using the same structural expansion as event extraction; overflow/unknown trip counts are explicit incomplete facts. Pipeline residency multiplies live versions, not all loop trip counts indiscriminately.
- [ ] **Step 4: Use a conservative reuse policy.** Only alias slots when the step/event dependency order proves the prior last use precedes the next write. If serial reuse is chosen, add the necessary dependency edge and materialize that ordering; reschedule and recompute occupancy. Never rely on a single arbitrary topological ordering to make parallel work appear disjoint. Do not add speculative overlap optimization in this repair.

```cpp
// Safe aliasing predicate for the dependency graph:
// all final readers of A happen-before every first writer of B.
// If the chosen reuse introduces order, record A.lastUse -> B.firstWrite
// in the plan and canonical bound Micro-IR, then schedule again.
```

- [ ] **Step 5: Check peaks against concrete memory capacities.** Diagnose memory ID, required bytes, capacity and simultaneous occurrences. Preserve alignment and alias-base ownership. Test double-count prevention for proven aliases and distinct same-kind nodes. Validate the emitted allocation/storage-ID graph matches planned storage; bufferization's generated scratch is addressed separately by E2/E3 and must not be falsely counted as a named target slot.
- [ ] **Step 6: Run storage/liveness/binder plus L1 suites.** Commit with `fix: verify concurrent storage lifetimes and ordered reuse`. Review gate: a serial loop and a four-stage pipeline have different, explained residency, and reuse order exists in the emitted IR.

### Task R6: Use one canonical selected-kernel event analysis

**Files:** Create `include/LLK/Perf/SelectedKernelAnalysis.h`, `lib/Perf/SelectedKernelAnalysis.cpp`; modify `lib/Perf/{MicroDAG,MicroCostModel,MicroPerfReport}.cpp`, `include/LLK/Mapping/CostEvent.h`, `lib/Mapping/{CostEvent,EventSchedule}.cpp`, root CMake; tests `test/Perf/l1_resource_dag.cpp`, `test/Mapping/cost_model.cpp`.

**Interfaces:** Produces `analyzeSelectedKernel` and `SelectedKernelAnalysis` declared above. `PlanEventDAG` carries the extended event fields including owners and accesses; schedulePlanEvents passes the same owner constraints as scheduleL1. `CoveringPlan` gains an optional derived normalized analysis-event snapshot, excluded from identity and never trusted on replay. `buildPlanEvents` reads a verified snapshot for finalized plans; analysis-only legacy hand-built plans retain explicitly labeled accumulation fallback.

- [ ] **Step 1: Replace the test that pins rule estimate five versus perf eight.** The rule-local five may remain a frontier estimate; the final selected add's two consumers and copies must have the same event/resource/work/bytes/dependency/owner tuples and scheduled cost as its bound kernel. Create an equality test for every normalized field, not only total latency.

```cpp
auto analysis = analyzeSelectedKernel(bound.kernel, target.machine(), true);
ASSERT_TRUE(analysis);
auto events = buildPlanEvents(finalizedPlan, target.machine());
ASSERT_TRUE(events);
ASSERT_EQ(events->events.size(), analysis->events.events.size());
for (size_t i = 0; i < events->events.size(); ++i) {
  EXPECT_EQ(events->events[i].event.resource,
            analysis->events.events[i].event.resource);
  EXPECT_EQ(events->events[i].workItems, analysis->events.events[i].workItems);
  EXPECT_EQ(events->events[i].bytes, analysis->events.events[i].bytes);
  EXPECT_EQ(events->events[i].deps, analysis->events.events[i].deps);
  EXPECT_EQ(events->events[i].owner, analysis->events.events[i].owner);
}
```

- [ ] **Step 2: Run `build/L1ResourceDAGTest --gtest_filter='*Plan*'`.** Current public plans disagree with bound IR and the former test enshrines part of the discrepancy.
- [ ] **Step 3: Extract once from canonical mapped Micro-IR.** Reuse buildMicroDAG's semantic handling for loops, copies, waits, transforms, gathers and owners; change mapped compute resolution to recorded R1 bindings. Use normalizedPlanEvent plus concrete route and storage provenance to fill all fields. Strict events name the selected executor as the owner-occupancy pool; add scheduler owner-count resolution through MachineModel::findExecutor and its count. Preserve abstract-kind fallback only for unmapped analysis. Test two engines attached to a single-count executor versus engines on independent executors, so dropping or aggregating owner constraints cannot pass parity. No separate rule-cost formula for final compute events, and no synthesizing a workload copy as a constant-cost compute event.
- [ ] **Step 4: Centralize scheduling, memory traffic and liveness summaries.** Call R2 resource validation and R5 liveness for strict analysis. Both planner and micro-perf consume the same SelectedKernelAnalysis; MicroPerfReport's L0/L1 summaries derive from it. Count per-memory reads/writes under one documented convention, including both hops and external load/store traffic. Compare owner occupancy and synchronization cycles as well as compute/transfer.
- [ ] **Step 5: Keep measured estimates separate.** Apply valid LatencyProvider durations to exactly these event signatures only after static legality; no hit may erase work, traffic or capacity. A miss retains static duration. Default `micro-perf` and planner acceptance comparisons use static mode; calibrated mode must identify its provider/model and is not the parity baseline.
- [ ] **Step 6: Add strict unknown/cap controls and direct-edge ordering tests.** Unknown compute node, missing route provenance, overflow and too many expanded events fail or return explicit partial status according to contract. Repeated operands, direct edges, multiple results, fused group output uses and loop-carried values keep dependencies. Expand past the configured cap and require a cap diagnostic rather than silent single-iteration scoring.
- [ ] **Step 7: Run perf/cost/liveness tests.** Commit with `refactor: share canonical selected-kernel static analysis`. Review gate: planner and perf are two consumers of one semantic extraction/schedule, without a new dependency cycle.

### Task R7: Evaluate complete proposals before exact/beam retention

**Files:** Modify `include/LLK/Mapping/CoveringSearch.h`, `lib/Mapping/CoveringSearch.cpp`, `lib/Conversion/MicroMapping/MicroMappingCommon.h`; create CompletePlanEvaluation header/source; root CMake; tests `test/Mapping/{covering_search,properties}.cpp`, `test/Conversion/MicroMapping/complete_plan_evaluation.cpp` registered as `CompletePlanEvaluationTest`.

**Interfaces:** Produces the completion callback and evaluation functions declared above. `runMappingSearch` and compiler/tuner adapters supply it with the exact source module, graph, target and binding. Plan source/target/binding hashes are populated before preview verification. Final events and costs are derived again for report-replayed proposals.

- [ ] **Step 1: Add the capacity-topK regression.** Carry forward the audit's 8x8xf32/multiplicity-four case; additionally use an unambiguous four-live-version fixture (pipeline-four from R5), SRAM 512/DRAM 4096, SRAM cost one/DRAM two, exact topK=1. If R5 shows ordinary sequential multiplicity reuses one buffer, update the old probe's expected footprint instead of preserving a false 1024-byte model. The four-live-version case must still select legal DRAM.

```cpp
auto c = issue129::resourceCase("capacity-topk");
ASSERT_TRUE(c);
MappingSearchOptions options;
options.mode = SearchMode::Exact;
options.topK = 1;
options.requireCompleteEvaluation = true;
auto result = issue129::searchCase(*c, options);
ASSERT_TRUE(result);
ASSERT_EQ(result->plans.size(), 1u);
EXPECT_EQ(result->plans.front().placements.front().rule, "r.large");
EXPECT_GT(result->frontier.plansRejectedByCapacity, 0u);
EXPECT_FALSE(result->searchTruncated);
```

- [ ] **Step 2: Run `CompletePlanEvaluationTest` and the exact capacity test.** Baseline keeps cheap SRAM before discovering capacity failure.
- [ ] **Step 3: Implement the integration evaluator in this order:** clone proposal; resolve maps/compute/ports; finalize hop storage/steps; set required provenance and decision-only v3 ID; canonical bind preview; verify selected semantics; run selected static analysis; validate occupancy; attach derived event snapshot/final cost and physical readiness. If ordered reuse changes durable choices/dependencies, recompute the ID and rebuild the preview; repeat scheduling/liveness until stable within a bounded deterministic process, rejecting a cycle. Require one final bind/analysis to reproduce the retained choices and events. Scores never enter the v3 identity, avoiding a circular dependency between metadata verification and final scoring.

```cpp
// On each complete search proposal, before insertion into result.plans:
auto evaluated = options_.evaluateCompletePlan(proposal);
if (!evaluated) return evaluated.takeError();
if (evaluated->rejection) {
  // Record stable frontier code/message/count, then continue enumeration.
} else {
  // Rank and retain *evaluated->plan using its final scheduled objective.
}
```

- [ ] **Step 4: Move all top-K and deterministic-first-legal decisions after evaluation.** A failed cheap proposal continues exact enumeration and deterministic search. Beam retains only feasible complete proposals but still honestly reports frontier truncation. Missing evaluator under requireCompleteEvaluation is a configuration error. Remove the post-topK storage filter as the acceptance authority; retain an idempotence assertion/verification for returned plans.
- [ ] **Step 5: Audit every early rejection/pruning bound.** Disable additive rule-cost pruning for overlapped latency, provider overrides, utilization or any objective lacking a proved monotone lower bound. Similarly, summing all candidate storage is not a sound capacity rejection when lifetimes/aliases can reuse it: only a proved minimum-live-footprint bound may reject a partial state. Final R5 liveness decides otherwise. Preserve symmetry reduction only for a demonstrated equivalence that cannot erase a distinct feasible resource/route choice. Leave finite memory/candidate/route/event caps with searchTruncated/status reporting. A no-plan result caused by a cap is distinct from exhaustive infeasibility. Candidate rejections do not set truncation.
- [ ] **Step 6: Test callback contracts and replay.** A callback rejection recovers the next legal plan; fatal error stops; topK=0 returns all feasible complete plans; callback cannot return both plan/rejection; repeated preview leaves source IR unchanged. Replayed reports use this evaluator before score/compile and retain the same durable decisions/ID under the same model.
- [ ] **Step 7: Run covering/evaluation/property tests and the public capacity CLI fixture.** Commit with `fix: validate and score complete coverings before top-k`. Review gate: changing K does not change the best feasible exact plan when enumeration is untruncated.

### Task R8: Verify exact search independently and close durable static parity

**Files:** Modify `test/Mapping/{properties,covering_search,plan_report,plan_binder}.cpp`, `test/Perf/l1_resource_dag.cpp`; create `test/Mapping/exact_resource_oracle.cpp` (`ExactResourceOracleTest`); finalize report/metadata schema documentation and root CMake.

**Interfaces:** Consumes R1–R7. Produces no new production API. The oracle enumerates finite fixture assignments without CoveringSearch, finalizeStoragePlan, analyzeSelectedKernel or production key builders; expected capacities/traffic/ordering use direct fixture arithmetic.

- [ ] **Step 1: Create the joint-oracle fixture.** Two compute nodes, two same-kind memory nodes, two alternative routes (one via undersized L2), one direct dependent edge, optional four-stage residency. Restrict the product to at most 64 combinations. Independently enumerate assignments/routes, sum physical live bytes and calculate fixed-duration resource schedules; sort feasible candidates using the declared objective and stable literal selection tuple.

```python
# Test-oracle arithmetic specification; translate to the C++ test.
feasible = []
for compute in ("vpu.a", "vpu.b"):
    for memory, capacity in (("sram.0", 512), ("dram.0", 4096)):
        for route, middle_capacity in (("direct", None), ("via-l2", 128)):
            needed = 4 * 256
            if needed > capacity:
                continue
            if middle_capacity is not None and 256 > middle_capacity:
                continue
            feasible.append((compute, memory, route))
assert feasible == [("vpu.a", "dram.0", "direct"),
                    ("vpu.b", "dram.0", "direct")]
```

- [ ] **Step 2: Run ExactResourceOracleTest against exact topK=1,2,0 and reversed insertion order.** Require equality of feasible selection tuples and best cost; include sequential temporaries whose summed sizes exceed capacity but whose true live peak fits, to catch unsafe partial-state capacity rejection. Bounded beam only needs legal outputs and explicit truncation, never equality to exhaustive exact.
- [ ] **Step 3: Promote all audit probes into regressions.** Verify selected compute events, per-DMA schedules, top-K recovery, intermediate occupancy and all old public planner/perf mismatches. Include normalized event fields, owner constraints, final cycles, per-memory traffic and capacity peaks; compare exact values from the same static mode, not an arbitrary tolerance that hides semantic drift.
- [ ] **Step 4: Harden durable replay.** Fresh-process report -> source bind -> decode -> static reanalysis -> report produces the same normalized IR and decisions. Alter only graph operand occurrence, compute binding, memory node, rule content, solved map or target profile: reject. Renaming/copying target files without content changes must not invalidate content identity. Run two independent MLIRContexts to catch borrowed-map lifetime errors.
- [ ] **Step 5: Run `ctest --test-dir build --output-on-failure -R 'Mapping|ExactResourceOracle|CompletePlanEvaluation|L1ResourceDAG'` and the whole suite at Gate R.** Save artifact paths plus exact toolchain/revision. Commit with `test: prove feasible exact search and selected-plan perf parity`. Review gate: every G1/G2/G6/G7/G8 counterexample is a checked-in regression, and no cap is mistaken for an infeasibility proof.
