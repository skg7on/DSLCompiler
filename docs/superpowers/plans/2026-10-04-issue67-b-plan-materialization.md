# Issue #67 Stage B — Complete Plans and Materialization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make a selected plan fully explicit, capacity-legal and materializable for canonical tile programs.

**Architecture:** Persist typed endpoint/resource state, build a storage/synchronization plan, and materialize it through canonical Micro builders. Replace restricted exact connection selection with bounded joint branching and align normalized plan/perf events.

**Tech Stack:** C++20, LLVM/MLIR, LLKMap, GoogleTest, FileCheck, CMake/Ninja/CTest.

**Spec:** [Normative design](../specs/2026-09-18-microir-inspired-dslcompiler-enhancement-design.md), [master contracts/dependencies](2026-10-04-issue67-improvement-plan.md), [review evidence](../../reviews/2026-10-04-issue67-pr111-status.md).

## Global Constraints

Apply every global constraint and interface decision in the master plan, including isolated worktrees, canonical Micro-IR, target-owned policy, LLVM 22/24 portability, bounded-solver soundness and legacy compatibility. Proposed APIs below are implemented by their owning task before dependent tasks use them.

## Review Focus

Apply the master's five review-focus cases. Each owning task specifies the negative input, positive control and verification command. Source snippets are implementation/test contracts; adapt includes and registration to existing file conventions without weakening the assertions.

## File responsibilities

Add generic `MappingMetadata.{h,cpp}` for schema encoding/decoding and `PlanVerification.cpp` for layered legality. Move canonical operation construction into `lib/Conversion/MicroMapping/PlanMaterialization.cpp`, implementing the master's PlanMaterializer; add public type builders in MicroHelpers. Storage/lifetime logic lives in `StoragePlan.{h,cpp}` rather than further enlarging CoveringSearch. Keep target-owned policy out of these files.

### B1: Persist schema-v2 endpoint and selected implementation state

**Dependencies:** A1–A5

**Files:**
- Create: include/LLK/Mapping/MappingMetadata.h
- Create: lib/Mapping/MappingMetadata.cpp
- Create: lib/Mapping/PlanVerification.cpp
- Modify: lib/Mapping/PlanBinder.cpp
- Modify: include/LLK/Mapping/PlanReport.h
- Modify: lib/Mapping/PlanReport.cpp
- Modify: include/LLK/Mapping/MappingPlan.h
- Modify: CMakeLists.txt
- Test: test/Mapping/plan_binder.cpp
- Test: test/Mapping/plan_report.cpp

**Interfaces:** Produces encodeSelectedPlan(ModuleOp, const CoveringPlan&, const MappingTarget&) -> llvm::Error and decodeSelectedPlan(ModuleOp, const MappingTarget&) -> llvm::Expected<CoveringPlan>. Add readPlanReport(StringRef, const MappingTarget&, const WorkloadGraph&) for versioned data replay. Persist endpoints, rule params, compute/resource bindings, maps, storage IDs, source-graph identity and target content hashes. The master defines canonical source-graph hashing and its relation to the materialized graph.

- [ ] **Step 1 — add the regression and legal controls.**

Round-trip a plan whose two input uses share a value but have distinct layouts/routes. Decode and compare endpoint/layout/map/resource fields exactly, not only plan ID:
```cpp
// p is the endpoint-rich plan built by the registered repeated-use fixture.
EXPECT_EQ(decoded->connectionPlans[0].consumerPorts,
          p.connectionPlans[0].consumerPorts);
EXPECT_EQ(decoded->placements[0].layoutSolutions.size(),
          p.placements[0].layoutSolutions.size());
```
Also serialize/reload the v2 JSON report and compare plan identity/all selected fields. Add changed input-graph/target hash rejection. Add v1 unambiguous migration, ambiguous v1 rejection in executable mode, unsupported schema-version rejection, malformed endpoint indices and content-hash mismatch tests.

- [ ] **Step 2 — run the focused test before changing behavior.**

```sh
cmake --build build --target MappingPlanBinderTest MappingPlanReportTest -j 6
build/MappingPlanBinderTest
build/MappingPlanReportTest
```

Expected before the change: the new behavioral assertion fails (or the new interface test cannot compile). Keep existing legal controls green; record the observed failure.

- [ ] **Step 3 — implement this contract.**

Use structured dictionaries, with this required schema shape:
```mlir
micro.plan = {schema_version = 2 : i64, id = 42 : i64,
              binding_hash = 0 : i64, materialized = true,
              graph_hash = "2222222222222222", target_hash = "3333333333333333",
              machine_hash = "0123456789abcdef", rule_hash = "fedcba9876543210", layout_hash = "1111111111111111"}
// Each layout entry carries endpoint node/direction/index, family,
// typed parameters and concrete affine map; each connection carries
// source/destination endpoints, ordered route/engines/hops and storage IDs.
```
The values above illustrate the schema; the encoder computes actual content hashes from canonical source/target inputs. Use checked integer width/range, dictionary/array/string/map reads and stable diagnostics. Version the canonical identity when endpoint semantics change. Keep provenance/report-only fields distinct from identity-bearing selections. Share typed serialization with readPlanReport; a frozen plan cannot bypass current graph/target verification. Split verification out of the oversized binder while preserving its public wrapper.

- [ ] **Step 4 — rerun the focused command and review the gate.**

Decoding restores execution-affecting choices; ambiguous legacy data cannot masquerade as executable v2 state. Run relevant neighboring tests when the interface changes. No broad success claim follows merely from a new schema parsing.

- [ ] **Step 5 — commit the independently testable change.**

```sh
git add include/LLK/Mapping/MappingMetadata.h lib/Mapping/MappingMetadata.cpp lib/Mapping/PlanVerification.cpp lib/Mapping/PlanBinder.cpp include/LLK/Mapping/PlanReport.h lib/Mapping/PlanReport.cpp include/LLK/Mapping/MappingPlan.h CMakeLists.txt test/Mapping/plan_binder.cpp test/Mapping/plan_report.cpp
git commit -m "feat(mapping): persist versioned endpoint and resource selections"
```

### B2: Associate every input/output role with its memory and owner choices

**Dependencies:** A7, B1

**Files:**
- Modify: include/LLK/Mapping/MappingRules.h
- Modify: include/LLK/Mapping/MappingPlan.h
- Modify: lib/Mapping/MappingRules.cpp
- Modify: lib/Mapping/Placement.cpp
- Modify: lib/Mapping/CoveringSearch.cpp
- Modify: docs/design/llkmap-rule-grammar.md
- Test: test/Mapping/mapping_rules.cpp
- Test: test/Mapping/covering_search.cpp

**Interfaces:** Produces PortMemoryBinding vectors on CandidateInstance/PlanPlacement. MemoryRequirement gains optional named port role; placement consumes explicit bound owner/memory-path projections rather than provenance-only values.

- [ ] **Step 1 — add the regression and legal controls.**

Parse/print/reparse these proposed clauses:
```text
require memory output "small" kind sram;
require memory output "large" kind dram;
```
Construct outputs 4 and 4096 bytes: SRAM=1024, DRAM=8192 is legal; swapping the assignments rejects. Both memories=1024 rejects. Add same-kind different-memory roles, outputless sink and legacy single-binding controls. A bound owner or memory path must change permitted placements or reject contradictory rules.

- [ ] **Step 2 — run the focused test before changing behavior.**

```sh
cmake --build build --target MappingRulesTest MappingPlacementTest MappingCoveringSearchTest -j 6
build/MappingRulesTest
build/MappingPlacementTest
build/MappingCoveringSearchTest
```

Expected before the change: the new behavioral assertion fails (or the new interface test cannot compile). Keep existing legal controls green; record the observed failure.

- [ ] **Step 3 — implement this contract.**

Extend the formal LLKMap grammar and parser together. Named roles resolve through RulePort/PortRef. Bare legacy `require memory kind sram` remains valid only when its role assignment is unique; ambiguous multi-output/multi-memory combinations keep A's rejection. Capacity is charged to each explicitly selected output memory, not the first output replicated across bindings.
```cpp
// For every output endpoint p:
//   require exactly one selected memory or a declared alias assignment;
//   charge storageFacts(p).bytes to that node;
//   record PortMemoryBinding{p, node.id}.
```
Project owner_mapping to abstract executor requirements and memory_path to allowed endpoint/hop nodes using machine data. Reject unsupported selected axes explicitly; do not silently ignore them. Concrete executor IDs remain selected metadata, not tile type fields.

- [ ] **Step 4 — rerun the focused command and review the gate.**

The original unsafe combination becomes supportable when explicitly associated, while ambiguous declarations still reject. Bound resource choices affect legality. Run relevant neighboring tests when the interface changes. No broad success claim follows merely from a new schema parsing.

- [ ] **Step 5 — commit the independently testable change.**

```sh
git add include/LLK/Mapping/MappingRules.h include/LLK/Mapping/MappingPlan.h lib/Mapping/MappingRules.cpp lib/Mapping/Placement.cpp lib/Mapping/CoveringSearch.cpp docs/design/llkmap-rule-grammar.md test/Mapping/mapping_rules.cpp test/Mapping/covering_search.cpp
git commit -m "feat(mapping): bind memory roles to concrete workload ports"
```

### B3: Build physical storage footprints and live ranges

**Dependencies:** B1/B2; A9

**Files:**
- Create: include/LLK/Mapping/StoragePlan.h
- Create: lib/Mapping/StoragePlan.cpp
- Create: test/Mapping/storage_plan.cpp
- Modify: include/LLK/Mapping/MappingPlan.h
- Modify: lib/Mapping/TileFacts.cpp
- Modify: lib/Mapping/CoveringSearch.cpp
- Modify: CMakeLists.txt
- Test: test/Mapping/tile_facts.cpp
- Test: test/Mapping/covering_search.cpp

**Interfaces:** Produces finalizeStoragePlan(const WorkloadGraph&, CoveringPlan&, const MachineModel&) -> llvm::Error. Adds stable plan-step IDs, allocation begin/end steps and dependency edges; persisted StorageAllocation values use those steps.

- [ ] **Step 1 — add the regression and legal controls.**

Test plain 8x8xf32 = 256 bytes; a padded/block layout may require a larger physical image. Test the exact selected map footprint, not logical elements alone:
```cpp
TEST(StoragePlan, PreservesSourceWhileBothTransformsAreLive) {
  StorageAllocation a; a.id=1; a.memory="sram.0"; a.bytes=256; a.beginStep=0; a.endStep=4;
  StorageAllocation b; b.id=2; b.memory="sram.0"; b.bytes=256; b.beginStep=1; b.endStep=3;
  StorageAllocation c; c.id=3; c.memory="sram.0"; c.bytes=256; c.beginStep=2; c.endStep=4;
  auto peak = computePeakStorage({a,b,c});
  ASSERT_TRUE(bool(peak));
  EXPECT_EQ((*peak)["sram.0"], 768u);
}
```
Add sequential buffer reuse, overlapping consumers, intermediate staging, pipeline-stage multiplication, unused output release, explicit proven alias and arithmetic-overflow rejection cases.

- [ ] **Step 2 — run the focused test before changing behavior.**

```sh
cmake --build build --target MappingTileFactsTest MappingStoragePlanTest MappingCoveringSearchTest -j 6
build/MappingTileFactsTest
build/MappingStoragePlanTest
build/MappingCoveringSearchTest
```

Expected before the change: the new behavioral assertion fails (or the new interface test cannot compile). Keep existing legal controls green; record the observed failure.

- [ ] **Step 3 — implement this contract.**

Compute a checked physical footprint from shape, dtype, affine map and any declared padding. Use bounded finite image enumeration for small static maps or proven affine bounds; missing/dynamic bounds produce a specific unsupported-footprint diagnostic. Do not use zero or an admitted lower bound for capacity legality.
```cpp
// Build deterministic plan-step DAG: compute/hop/transform/wait/barrier/gather.
// Allocation begins before its first writer and ends after its last real reader.
// At each dependency-valid step, add newly live storage, test capacity, then expire.
// Reuse requires nonoverlap; aliasing requires explicit compatible storage proof.
```
Include scope/iteration/pipeline metadata from workload extraction. Unknown execution multiplicity is reported; strict executable planning cannot pretend an unknown loop runs once. Keep conservative fallback for explicit analysis mode only. Store occupancy/resource diagnostics in the report.

- [ ] **Step 4 — rerun the focused command and review the gate.**

Physical padding, transformed replicas and staging buffers cannot escape capacity checks; sequential legal reuse remains admitted. Run relevant neighboring tests when the interface changes. No broad success claim follows merely from a new schema parsing.

- [ ] **Step 5 — commit the independently testable change.**

```sh
git add include/LLK/Mapping/StoragePlan.h lib/Mapping/StoragePlan.cpp test/Mapping/storage_plan.cpp include/LLK/Mapping/MappingPlan.h lib/Mapping/TileFacts.cpp lib/Mapping/CoveringSearch.cpp CMakeLists.txt test/Mapping/tile_facts.cpp test/Mapping/covering_search.cpp
git commit -m "feat(mapping): plan explicit physical storage and value lifetimes"
```

### B4: Materialize canonical tile transfers and transforms

**Dependencies:** A2/A5/A9, B1–B3

**Files:**
- Create: lib/Conversion/MicroMapping/PlanMaterialization.cpp
- Modify: include/LLK/Mapping/PlanBinder.h
- Modify: lib/Mapping/PlanBinder.cpp
- Modify: include/LLK/Dialect/Micro/MicroHelpers.h
- Modify: lib/Dialect/Micro/MicroOps.cpp
- Modify: lib/Conversion/MicroMapping/MicroMappingCommon.h
- Modify: CMakeLists.txt
- Test: test/Mapping/plan_binder.cpp
- Test: test/Conversion/MicroMapping/micro_map_unmaterialized.mlir

**Interfaces:** Implements PlanMaterializer from the master; bindPlan gains final optional PlanMaterializer* argument. Public helpers construct a TileType with new abstract memory while preserving logical shape/dtype/owner.

- [ ] **Step 1 — add the regression and legal controls.**

Change the canonical tile-transfer fixture from warned output to strict successful binding:
```sh
build/llk-opt test/Conversion/MicroMapping/micro_map_unmaterialized.mlir   '--micro-map=target=probe machine=test/Conversion/MicroMapping/probe_machine.yaml layouts=mapping/x86-avx2/layouts.llkmap rules=test/Conversion/MicroMapping/unmaterialized_rules.llkmap emitters=e1 require-executable=1'
```
FileCheck must see tile_async_copy, wait, correct destination tile memory and no unmaterialized warning. Add tile LayoutTransform, TransferAndTransform, two consumers of one value requiring different destinations, and tensor movement regression controls.

- [ ] **Step 2 — run the focused test before changing behavior.**

```sh
cmake --build build --target llk-opt MappingPlanBinderTest -j 6
build/MappingPlanBinderTest
ctest --test-dir build -R MicroMappingUnmaterialized --output-on-failure
```

Expected before the change: the new behavioral assertion fails (or the new interface test cannot compile). Keep existing legal controls green; record the observed failure.

- [ ] **Step 3 — implement this contract.**

Move operation construction out of LLKMapping into the canonical materializer, preserving metadata-only standalone callers. Replace ShapedType-only rejection with canonical tensor/tile adapters:
```cpp
// Inspect source logical tile facts via public Micro helpers.
// Construct destination type from selected memory role and solved layout.
// Emit typed tile_async_copy into the allocation selected by StoragePlan.
// Emit micro.transform with concrete maps and selected transform resource.
// Rewire only consumerPorts, never every use of a value or whole consumer op.
```
Transform outputs are distinct storage unless B3 proves aliasing. Apply type propagation to exactly the affected operand/results; preserve unrelated uses. Do not encode target layout IDs as generic enums. Executable without a canonical materializer or with missing static facts rejects with a named unresolved decision.

- [ ] **Step 4 — rerun the focused command and review the gate.**

Canonical tiles have complete, verified movement/transform dataflow; tensor behavior and partial-analysis reporting remain intact. Run relevant neighboring tests when the interface changes. No broad success claim follows merely from a new schema parsing.

- [ ] **Step 5 — commit the independently testable change.**

```sh
git add lib/Conversion/MicroMapping/PlanMaterialization.cpp include/LLK/Mapping/PlanBinder.h lib/Mapping/PlanBinder.cpp include/LLK/Dialect/Micro/MicroHelpers.h lib/Dialect/Micro/MicroOps.cpp lib/Conversion/MicroMapping/MicroMappingCommon.h CMakeLists.txt test/Mapping/plan_binder.cpp test/Conversion/MicroMapping/micro_map_unmaterialized.mlir
git commit -m "feat(micro): materialize selected canonical tile connections"
```

### B5: Distinguish concrete memory nodes of one abstract space

**Dependencies:** B1/B4

**Files:**
- Modify: include/LLK/Dialect/Micro/MicroOps.td
- Modify: lib/Dialect/Micro/MicroOps.cpp
- Modify: lib/Conversion/MicroMapping/PlanMaterialization.cpp
- Modify: lib/Mapping/PlanVerification.cpp
- Modify: lib/Perf/MicroDAG.cpp
- Test: test/Dialect/Micro/tile_ops.mlir
- Test: test/Mapping/plan_binder.cpp
- Test: test/Perf/route_identity.cpp

**Interfaces:** Movement ops carry generic source/destination node, connection/hop/engine identity. Equal abstract memory kinds are legal only for validated distinct nodes.

- [ ] **Step 1 — add the regression and legal controls.**

Use machine nodes sram.0 and sram.1 with a legal link; bind a tile transfer between them while both TileTypes remain memory=sram. Verify one copy and correct per-hop perf attribution. Add nonexistent link, equal source/destination node, unsupported engine, malformed IDs and a two-hop route with same-kind adjacent memories.

- [ ] **Step 2 — run the focused test before changing behavior.**

```sh
cmake --build build --target MappingPlanBinderTest RouteIdentityTest L1ResourceDAGTest -j 6
build/MappingPlanBinderTest
build/RouteIdentityTest
build/L1ResourceDAGTest
```

Expected before the change: the new behavioral assertion fails (or the new interface test cannot compile). Keep existing legal controls green; record the observed failure.

- [ ] **Step 3 — implement this contract.**

Permit same-kind movement when concrete storage/node identity establishes real work:
```cpp
// Source/destination kind equality is not node identity.
// Same-kind source/destination requires distinct src_node/dst_node attributes.
// Structural verifier checks shape/type/attribute shape;
// machine verifier checks actual nodes, link, engine, transaction/alignment facts.
```
Do not add a generic enum per physical memory. Match perf using connection + hop ID, not destination kind or first matching link. Unknown target memory kinds require an explicit canonical abstract-space mapping; reject missing mappings rather than inventing a dialect enum.

- [ ] **Step 4 — rerun the focused command and review the gate.**

Distinct SRAM nodes are not collapsed; each selected route hop has one correctly attributed movement. Run relevant neighboring tests when the interface changes. No broad success claim follows merely from a new schema parsing.

- [ ] **Step 5 — commit the independently testable change.**

```sh
git add include/LLK/Dialect/Micro/MicroOps.td lib/Dialect/Micro/MicroOps.cpp lib/Conversion/MicroMapping/PlanMaterialization.cpp lib/Mapping/PlanVerification.cpp lib/Perf/MicroDAG.cpp test/Dialect/Micro/tile_ops.mlir test/Mapping/plan_binder.cpp test/Perf/route_identity.cpp
git commit -m "feat(micro): preserve node identity for same-space memory transfers"
```

### B6: Materialize explicit gather semantics and synchronization

**Dependencies:** B1–B5

**Files:**
- Modify: include/LLK/Mapping/MappingPlan.h
- Modify: include/LLK/Dialect/Micro/MicroOps.td
- Modify: include/LLK/Dialect/Micro/MicroEnums.h
- Modify: lib/Dialect/Micro/MicroOps.cpp
- Modify: lib/Conversion/MicroMapping/PlanMaterialization.cpp
- Modify: lib/Mapping/PlanVerification.cpp
- Modify: lib/Perf/MicroDAG.cpp
- Test: test/Mapping/plan_binder.cpp
- Test: test/Perf/l1_resource_dag.cpp
- Test: test/Dialect/Micro/tile_ops.mlir

**Interfaces:** Adds GatherSemantics {Sum, Max, Concatenate} and optional concat axis to gather connections. Canonical micro.gather carries explicit semantics; SynchronizationStep drives waits/barriers. No arithmetic is inferred solely from multi-producer topology.

- [ ] **Step 1 — add the regression and legal controls.**

Test two typed producer tiles with explicit Sum semantics, selected feed routes, staging allocations and one gathered result. Add Max and Concatenate shape controls; missing semantics must reject Executable binding. Test a producer-consumer boundary requiring wait, a collective barrier across workers, and independent copies that must not receive unnecessary global barriers.

- [ ] **Step 2 — run the focused test before changing behavior.**

```sh
cmake --build build --target MappingPlanBinderTest L1ResourceDAGTest llk-opt -j 6
build/MappingPlanBinderTest
build/L1ResourceDAGTest
ctest --test-dir build -R MicroDialectTileOps --output-on-failure
```

Expected before the change: the new behavioral assertion fails (or the new interface test cannot compile). Keep existing legal controls green; record the observed failure.

- [ ] **Step 3 — implement this contract.**

Define/verify the canonical gather and barrier ops alongside their hand-written enum/string tables:
```text
micro.gather inputs, kind=sum|max|concat, axis(for concat) -> typed result
micro.barrier dependency tokens, scope=generic executor group
```
Sum/Max require compatible element types/shapes; Concatenate requires a declared axis and exact resulting extent. Emit feed copies, tokens, staging storage and reduction/assembly under those semantics. Existing synthetic multi-producer graphs without semantics remain Partial-only, with a stable explicit reason.
Place waits before actual data uses and barriers only when machine/operation dependency semantics require them. Preserve region boundaries; reject a plan that needs synchronization the IR scope cannot express. Perf and target lowering must either handle every new op or reject it explicitly.

- [ ] **Step 4 — rerun the focused command and review the gate.**

Every gather has declared semantics and executable dataflow; required synchronization is represented and independent work can still overlap. Run relevant neighboring tests when the interface changes. No broad success claim follows merely from a new schema parsing.

- [ ] **Step 5 — commit the independently testable change.**

```sh
git add include/LLK/Mapping/MappingPlan.h include/LLK/Dialect/Micro/MicroOps.td include/LLK/Dialect/Micro/MicroEnums.h lib/Dialect/Micro/MicroOps.cpp lib/Conversion/MicroMapping/PlanMaterialization.cpp lib/Mapping/PlanVerification.cpp lib/Perf/MicroDAG.cpp test/Mapping/plan_binder.cpp test/Perf/l1_resource_dag.cpp test/Dialect/Micro/tile_ops.mlir
git commit -m "feat(micro): materialize gather and synchronization decisions"
```

### B7: Explore joint connection choices in exact search

**Dependencies:** A6, B2/B3/B6

**Files:**
- Modify: include/LLK/Mapping/CoveringSearch.h
- Modify: lib/Mapping/CoveringSearch.cpp
- Modify: lib/Mapping/Placement.cpp
- Modify: lib/Mapping/PlanReport.cpp
- Test: test/Mapping/covering_search.cpp
- Test: test/Mapping/properties.cpp

**Interfaces:** Produces enumerateConnectionChoices(requests, target, options) as a bounded list of connection combinations. Exact branches over alternatives; cap/truncation and solver-undecided status remain separate.

- [ ] **Step 1 — add the regression and legal controls.**

Use two consumers whose cheapest routes share one-tile staging memory X; either dearer direct route makes the complete covering fit. Exact with caps lifted must find the feasible optimum. Independently enumerate this tiny graph as an oracle:
```python
from itertools import product

def route_oracle(groups, capacity):
    legal = []
    for chosen in product(*groups):
        peak = {}
        for route in chosen:
            for memory, size in route["live_bytes"].items():
                peak[memory] = peak.get(memory, 0) + size
        if all(size <= capacity[memory] for memory, size in peak.items()):
            legal.append((sum(r["cost"] for r in chosen),
                          tuple(r["id"] for r in chosen)))
    return sorted(legal)
```
Use an independent test fixture supplying all simultaneously live route buffers and base placement costs. Compare exact's sorted legal choices/costs to this oracle before B8 introduces scheduled latency; after B8, use the independent tiny schedule oracle too.
Cover fan-out transforms, gathers, maximize objectives, equal-cost lower-ID ties, route/state caps and insertion-order determinism.

- [ ] **Step 2 — run the focused test before changing behavior.**

```sh
cmake --build build --target MappingCoveringSearchTest MappingPropertiesTest -j 6
build/MappingCoveringSearchTest
build/MappingPropertiesTest
```

Expected before the change: the new behavioral assertion fails (or the new interface test cannot compile). Keep existing legal controls green; record the observed failure.

- [ ] **Step 3 — implement this contract.**

Refactor partial extension so connection alternatives create separate states before occupancy checks:
```text
for placement in legal placements:
  for connection combination in deterministic bounded product:
    clone partial state
    apply chosen costs, storage/live-range effects and dependencies
    reject illegal capacity/route combinations
    enqueue legal state
```
Exact uses all combinations within explicitly reported limits; deterministic returns the first complete legal covering; beam ranks bounded states by the declared objective. Apply admissible bounds only; do not prune equal objective cost when the ID tie cannot be proven. Keep A6 notices only when an explicit policy still collapses alternatives; report each cap independently.

- [ ] **Step 4 — rerun the focused command and review the gate.**

Exact agrees with brute-force small-graph results and does not falsely reject a legal expensive route combination. Run relevant neighboring tests when the interface changes. No broad success claim follows merely from a new schema parsing.

- [ ] **Step 5 — commit the independently testable change.**

```sh
git add include/LLK/Mapping/CoveringSearch.h lib/Mapping/CoveringSearch.cpp lib/Mapping/Placement.cpp lib/Mapping/PlanReport.cpp test/Mapping/covering_search.cpp test/Mapping/properties.cpp
git commit -m "feat(mapping): search complete joint connection alternatives"
```

### B8: Unify selected-plan/perf events and route-aware measurement identity

**Dependencies:** A9, B1–B7

**Files:**
- Modify: include/LLK/Mapping/CostEvent.h
- Modify: lib/Mapping/CostEvent.cpp
- Modify: include/LLK/Mapping/LatencyProvider.h
- Modify: lib/Mapping/LatencyProvider.cpp
- Modify: lib/Mapping/CoveringSearch.cpp
- Modify: lib/Perf/MicroDAG.cpp
- Modify: lib/Perf/MicroCostModel.cpp
- Test: test/Mapping/cost_model.cpp
- Test: test/Perf/l1_resource_dag.cpp
- Test: test/Mapping/properties.cpp

**Interfaces:** Adds PlanEventDAG with CostEvent nodes, dependencies and physical occupancy. buildPlanEvents(const CoveringPlan&, const MachineModel&) -> llvm::Expected<PlanEventDAG>; scheduled final cost and perf share event semantics. Add ConnectionSignature and default-null latency lookup overload for connections.

- [ ] **Step 1 — add the regression and legal controls.**

Compare a transform/two-hop/gather selected plan to its materialized kernel: normalized event kind/resource/work/bytes and dependency edges must agree. Distinguish additive work from scheduled latency:
```cpp
// Same work/event identities; final latency follows the same resource scheduler.
// Do not assert sum(event cycles) equals overlapped critical-path latency.
```
Change only route, hop engine, role assignment, map, bundle parameter or storage layout; the corresponding measurement identity must change. Identical reordered descriptions retain one key. Unsupported measurements use static estimates without affecting legality.

- [ ] **Step 2 — run the focused test before changing behavior.**

```sh
cmake --build build --target MappingCostModelTest L1ResourceDAGTest MappingPropertiesTest -j 6
build/MappingCostModelTest
build/L1ResourceDAGTest
build/MappingPropertiesTest
```

Expected before the change: the new behavioral assertion fails (or the new interface test cannot compile). Keep existing legal controls green; record the observed failure.

- [ ] **Step 3 — implement this contract.**

Use shared normalized event construction and checked occupancy for both paths. Preserve selected compute/transform resources and explicit synchronization. Count structural execution multiplicity; unknown strict facts reject instead of silently charging one iteration. Compute final candidate score from the shared schedule; optimistic partial bounds remain separately named.
```cpp
// ConnectionSignature includes kind, value type, endpoints, ordered node/link/
// engine path, concrete maps/parameters and model/content versions.
// TargetContext includes machine/rule/layout content hashes.
// Provider miss -> static estimate; hit -> calibrated cost, never new legality.
```
Bump cost-model/key versions. Serialization is length-delimited or structured, not ambiguous string concatenation. Keep compute identity fields added by PR #111 and add endpoint roles so swapping assignments cannot collide.

- [ ] **Step 4 — rerun the focused command and review the gate.**

The selected plan and materialized perf graph account for the same work and schedule semantics; routes/roles cannot share incorrect measurement keys. Run relevant neighboring tests when the interface changes. No broad success claim follows merely from a new schema parsing.

- [ ] **Step 5 — commit the independently testable change.**

```sh
git add include/LLK/Mapping/CostEvent.h lib/Mapping/CostEvent.cpp include/LLK/Mapping/LatencyProvider.h lib/Mapping/LatencyProvider.cpp lib/Mapping/CoveringSearch.cpp lib/Perf/MicroDAG.cpp lib/Perf/MicroCostModel.cpp test/Mapping/cost_model.cpp test/Perf/l1_resource_dag.cpp test/Mapping/properties.cpp
git commit -m "feat(perf): align mapped event costs and complete latency identity"
```

## Stage B release gate

- [ ] Every execution-affecting selection survives v2 round-trip and semantic verification.
- [ ] Strict tile-transfer/transform/gather/two-hop fixtures materialize without unresolved decisions; unknown facts still reject explicitly.
- [ ] Physical capacity/lifetime tests and exact-vs-oracle property tests pass.
- [ ] Shared plan/perf event and scheduling checks pass, alongside legacy controls.
- [ ] Publish the remaining boundary accurately: backend readiness/numeric target execution are stage C gates, not consequences of successful connection binding.
