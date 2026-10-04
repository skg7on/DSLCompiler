# Issue #67 Improvement Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Resolve the PR #111 review defects and complete the selected-plan execution, target-lowering, and performance contracts needed to close issue #67.

**Architecture:** Preserve canonical Micro-IR and the data-driven mapping core. Use explicit operand/result endpoints and resource assignments throughout search, persistence, verification, materialization and cost events. Deliver independently testable stages, with target-owned lowering and a descriptor-pointer runtime ABI layered on the completed plan model.

**Tech Stack:** C++20, LLVM/MLIR, TableGen, LLKMap, YAML MachineModel, GoogleTest, FileCheck, CMake/Ninja/CTest, ORC JIT.

**Spec:** [Normative enhancement design](../specs/2026-09-18-microir-inspired-dslcompiler-enhancement-design.md); [PR #111 review and reproductions](../../reviews/2026-10-04-issue67-pr111-status.md).

## Global Constraints

The following requirements apply to every task and companion plan:

- “`micro` remains the only canonical execution IR.” (§5.1)
- “Target policy lives in target packages.” (§5.4)
- “Target layout IDs are mapping results, not new enumerants in the Micro dialect.” (§13.4)
- “The implementation must not vendor or create a build-time dependency on the MicroIR reference repository.” (issue #67 architecture boundary)
- “Rule matching shall be side-effect aware. No rule may fuse across an operation with unknown effects, a synchronization boundary, or a region boundary it does not model.” (§14.2)
- Bounded solvers remain dependency-free; exhausted/undecided solves cannot establish legality. No mandatory SMT solver, Micro emulator, or accelerator hardware dependency.
- All writes occur in an isolated worktree below `.claude/worktrees/`, on a category/kebab-case branch. The main checkout is read-only.
- Verify Linux static-library builds using the CI-pinned LLVM 22.1.8 and the local LLVM/MLIR 24 configuration. New code must compile with both; do not rely on the local monolithic MLIR library to supply undeclared static dependencies.
- Preserve legacy compilation/JIT behavior when mapping is disabled. Every schema/API change includes migration/rejection tests and documentation.
- A numeric test cannot pass solely because compilation/lookup succeeded. An unavailable required backend is an explicit skip; an invocation error or wrong result is a failure.

## Review Focus

1. One SSA value used twice with incompatible port layouts: both operand uses must retain their own requirements (A1/A2, B1).
2. Structurally plausible but semantically forged mapping metadata: reject wrong predicates, capabilities, solved parameters and movement stamps (A3–A5).
3. Legal non-MMA workloads and several layout roles: evaluate only required facts, never silently skip declared constraints (A7).
4. Two cheap routes share staging capacity while a dearer combination fits: exact search must find the feasible combination and disclose every cap (A6, B7).
5. Mapped execution across transforms, tails and multiple outputs: numeric results, allocation lifetimes, event dependencies and cache identity must agree (B2–B6, C1–C8).

## Baseline and scope

Planning baseline: PR #111 head `249aa17670ff617bde1f37540075aa2c7f0d3929`, including PR #112. Review evidence: fresh build, 126 CTests, 124 passed, two CPU-specific skips. Earlier LLVM 22 linking failure was addressed by `249aa17`; new remote CI must still be checked by the implementer.

This is a proposed implementation plan, not a claim that its interfaces already exist. The normative design is retained; the concrete interfaces below are the recommended implementation decisions. Before executing, rebase the selected worktree onto the then-current intended base and retire any task already fixed with equivalent evidence.

Included: seven review findings; endpoint/resource identity; canonical tile movement/transforms; complete allocation/synchronization; joint connection search; target-owned AVX2 emission; explicit kernel ABI and numeric invocation; compile/tune integration; fused-rule support and complete acceptance chains.

Production measurement databases, calibration fitting and prediction validation remain owned by #51/#52. This plan provides their executable candidate and complete key/event hooks; it does not require a functional emulator or pretend a generic-accelerator fixture executes on unavailable hardware.

## Delivery structure

| Deliverable | Tasks | Depends on | Merge gate |
|---|---|---|---|
| [A — correctness and legality](2026-10-04-issue67-a-correctness.md) | A1–A9 | Baseline | All confirmed review defects have rejecting/working controls; exact restrictions remain explicit. |
| [B — complete plan and materialization](2026-10-04-issue67-b-plan-materialization.md) | B1–B8 | A1–A5, A9 | Tile transfers/transforms, endpoint rewiring, physical occupancy and synchronization are complete; exact explores joint choices. |
| [C — target execution and acceptance](2026-10-04-issue67-c-execution-acceptance.md) | C1–C10 | B1–B8 | Target-selected AVX2 programs execute numerically; compile/tune and all §29 gates have evidence. |

Use several PRs, each ending at a task or dependency boundary. Do not combine all stages into one unreviewable change. A7/A8 and the A9 perf fix can proceed independently of A1/A2; B4/B5 need B1–B3. C1 can begin after B1, but numeric target execution cannot be accepted before C2–C4 and B's materialization contract are complete.

## Architectural choices

Three approaches were considered:

1. **Incremental explicit contracts — recommended.** Extend the existing model, preserve baseline behavior and complete one tested stage at a time. This retains the investment in the mapper and limits compatibility risk.
2. **Repair only observed reproductions.** Cheapest immediately, but leaves endpoint ambiguity, execution completeness and target/perf divergence. Suitable only for A as a temporary release gate.
3. **Replace the mapper/backend pipeline.** Could unify every layer in a new system, but duplicates tested components, greatly enlarges the review surface and risks the canonical IR/legacy boundary. Not selected.

### Endpoint identity

Define in `include/LLK/Mapping/WorkloadGraph.h`:

```cpp
enum class PortDirection { Input, Output };
struct PortRef {
  WorkloadNodeId node = 0;
  PortDirection direction = PortDirection::Input;
  uint32_t index = 0;
  bool operator==(const PortRef &) const = default;
};
const WorkloadPort *lookupPort(const WorkloadGraph &, const PortRef &);
std::string canonicalPortRefString(const PortRef &);
```

`PortRef` identifies an occurrence, not its SSA value. Resolve it only after graph finalization. `LayoutRequirement`, `SolvedLayout`, `PortSpec`, `ConnectionRequest`, `ConnectionPlan` and `PlanConnection` carry explicit endpoints. Add `producerPort` and sorted `consumerPorts`; node/instance consumer lists can remain compatibility projections, never the rewiring authority. Use endpoint-aware canonical identity; bump plan identity/report schema versions rather than silently replaying old hashes under new semantics.

### Selected state and verification

Split generic metadata encoding/decoding from semantic checking: `MappingMetadata.{h,cpp}` and `PlanVerification.cpp`; retain `PlanBinder.h` entry points. No target-specific semantics enter generic parsing.

Persist `schema_version=2`, source-graph and target/rule/layout/machine hashes, resolved rule parameters, compute/memory bindings, per-port layout family/parameters/map, output storage assignments and per-connection endpoints. Define the source-graph hash over canonical, pre-materialization workload semantics, including operand occurrences, types, access maps and semantic attributes; omit bookkeeping attributes. Retain that source identity when validating a materialized graph through its recorded connection provenance. Read legacy metadata only when its missing endpoint/resource associations are uniquely recoverable; reject ambiguous executable replay. Preserve ordinary v1 parsing/printing for historical analysis.

Shared rule verification interfaces in `MappingRules.h`:

```cpp
llvm::Error verifyRuleSelection(const RuleDef &, const WorkloadNode &,
    const llvm::StringMap<SearchValue> &parameters,
    const machine::MachineModel &, const LayoutContext &);
llvm::Error verifySolvedLayout(const LayoutDef &, const SolvedLayout &,
    const machine::MachineModel &, mlir::MLIRContext &,
    const LayoutContext &, const SolverLimits &);
```

Use existing `evaluatePredicate` and LLKMap evaluation. Verification must validate the **recorded** assignment, not choose a different legal solution. Layout domains, constraints, maps and port associations must all agree. Do not compare only printer shape or rule names.

B1 also adds `readPlanReport(llvm::StringRef json, const MappingTarget &, const WorkloadGraph &) -> llvm::Expected<CoveringPlan>` in PlanReport.h. The v2 report round-trips the selected data needed for replay; graph and target content hashes must match before any executable binding. Reading a report reconstructs data, not executable code; semantic verification/materialization remain mandatory. C5 exposes optional frozen-selection replay through `--plan-report=<path>` while retaining search-based ID replay for historical reports.

### Constraint facts

Add `BindingFacts` in `include/LLK/Perf/Legality.h`:

```cpp
struct BindingFacts {
  std::optional<WorkloadShape> originalWorkload;
  std::vector<WorkloadShape> contractions;
};
LegalityResult checkConstraint(const SearchConstraint &, const SearchSpace &,
    const Candidate &, const BindingFacts &, const machine::MachineModel &);
```

Keep the existing shape overload as a wrapper. Shape-independent constraints operate with no contraction. Shape-dependent constraints identify their scope; missing required facts reject with a specific diagnostic. Export original workload dimensions before tiling as generic provenance. Role constraints evaluate the parameters they reference; absence and ambiguity are different states.

### Resource and storage model

Add in `MappingPlan.h`:

```cpp
struct PortMemoryBinding { PortRef port; MemoryNodeId memory; };
using PlanStepId = uint64_t;
struct StorageAllocation {
  uint64_t id = 0;
  WorkloadValueId value = 0;
  MemoryNodeId memory;
  uint64_t bytes = 0;
  std::optional<uint64_t> aliasOf;
  PlanStepId beginStep = 0;
  PlanStepId endStep = 0; // live through this step
};
struct SynchronizationStep {
  uint64_t id = 0;
  std::vector<ConnectionId> waitsFor;
  std::vector<PortRef> precedes;
  bool requiresBarrier = false;
};
```

Output placement and connection representations name these storage IDs. All arithmetic is checked. Physical layout footprint includes padding; an unknown footprint is not zero. Allocate transforms out of place by default, preserving immutable producer data. Permit reuse/aliasing only with a proof and explicit `aliasOf`. Live ranges end at the last actual consumer use, including route/transform/synchronization dependencies.

### Materialization versus target readiness

Keep partial analysis. Strengthen executable binding to mean all selected connection/storage/synchronization decisions are represented; annotate completion explicitly and validate it. Add a separate target-readiness/lowering gate. A complete connection plan containing unsupported `micro.spatial_for` must fail target readiness, not claim successful target compilation.

Tile construction is exposed through public canonical Micro type helpers. Put dialect-specific materialization in `lib/Conversion/MicroMapping/PlanMaterialization.cpp`, behind a generic `PlanMaterializer` interface, avoiding a new Mapping↔MicroDialect library cycle:

```cpp
class PlanMaterializer {
public:
  virtual ~PlanMaterializer() = default;
  virtual llvm::Error materialize(mlir::ModuleOp,
      const CoveringPlan &, const MappingTarget &, BoundPlan &) = 0;
};
```

`bindPlan` receives an optional materializer pointer after its existing `BindContract` argument. The CLI supplies the canonical implementation; standalone mapping users retain metadata-only Partial binding, while Executable without a materializer fails. Migrate direct executable test callers explicitly. The existing tensor implementation is moved into that canonical materializer, preserving tested behavior.

### Target lowering and runtime ABI

Extend `TargetEmitter` using `TargetLoweringContext` in a new `MappingLowering.h`:

```cpp
struct TargetLoweringContext {
  const machine::MachineModel &machine;
  llvm::ArrayRef<PlanPlacement> placements;
  llvm::ArrayRef<PlanConnection> connections;
};
virtual llvm::Error lower(llvm::ArrayRef<mlir::Operation *> coveredOps,
    const TargetBundle &, const TargetLoweringContext &,
    mlir::RewriterBase &) const;
```

The default returns an unsupported-lowering error; file declarations alone cannot impersonate an executable emitter. AVX2 owns supported bundle semantics and lowering patterns. The generic bridge remains useful for reference execution and must not be mistaken for selected-bundle execution.

Make kernel arguments/results explicit, then bufferize results to caller-owned output parameters and request `llvm.emit_c_interface`. All runtime descriptor pointers enter a generated wrapper; do not reinterpret an expanded LLVM signature as the legacy `KernelFn`. New `MappedExecutable` owns its JIT/module lifetime and exposes:

```cpp
llvm::Error invoke(llvm::ArrayRef<MemRef2D *> inputs,
                   llvm::ArrayRef<MemRef2D *> outputs);
```

The C4 factory is declared in `MappedExecutable.h`:

```cpp
namespace llk {
llvm::Expected<std::unique_ptr<MappedExecutable>> createMappedExecutable(
    mlir::ModuleOp bufferedModule, llvm::StringRef entrySymbol);
}
```

It receives function/memref IR after bufferization and loop lowering, records the typed input/output contract, converts results to caller-owned out-parameters, emits the C-interface wrapper and runs the existing LLVM/ORC compilation stages. Its owned translated LLVM module/context and JIT outlive invocation; it retains no borrowed MLIR operation after creation. C5 calls this factory rather than duplicating the ABI/JIT pipeline.

Use MLIR packed/C-interface invocation or an equivalently tested generated adapter. The implementation must validate arity, element types, strides and ownership. Keep the legacy five-descriptor ABI separate and tested. A raw `lookupTyped` cast is not an ABI proof.

### Additional shared contracts introduced by the task owners

A9 defines in `CostModel.h`:

```cpp
struct TransformCostInput {
  mlir::Type inputType;
  mlir::Type outputType;
  mlir::AffineMap srcMap;
  mlir::AffineMap dstMap;
  std::string memoryNode;
  std::string computeResource;
};
llvm::Expected<Cost> estimateTransformCost(const TransformCostInput &,
                                         const machine::MachineModel &);
```

B3 defines in `StoragePlan.h`:

```cpp
llvm::Expected<std::map<MemoryNodeId, uint64_t>>
computePeakStorage(llvm::ArrayRef<StorageAllocation>);
llvm::Error finalizeStoragePlan(const WorkloadGraph &, CoveringPlan &,
                               const machine::MachineModel &);
```

`computePeakStorage` summarizes a validated deterministic schedule's live ranges; it does not invent a schedule or prove arbitrary concurrent ordering safe. `finalizeStoragePlan` builds/validates dependency-aware intervals first. `CoveringPlan` gains allocations, synchronization steps and stable step/dependency descriptions in B1/B3.

B7 defines in `Placement.h`:

```cpp
struct ConnectionChoiceSet {
  std::vector<std::vector<ConnectionPlan>> combinations;
  bool truncated = false;
  bool undecided = false;
};
llvm::Expected<ConnectionChoiceSet> enumerateConnectionChoices(
    llvm::ArrayRef<ConnectionRequest>, const MappingTarget &,
    const PlacementOptions &);
```

B8 defines normalized events in `CostEvent.h`:

```cpp
struct PlanCostEvent {
  CostEvent event;
  uint64_t workItems = 0;
  uint64_t bytes = 0;
  std::vector<uint32_t> deps;
};
struct PlanEventDAG { std::vector<PlanCostEvent> events; };
llvm::Expected<PlanEventDAG> buildPlanEvents(const CoveringPlan &,
                                          const machine::MachineModel &);
llvm::Expected<Cost> schedulePlanEvents(const PlanEventDAG &,
                                      const machine::MachineModel &);
```

In `LatencyProvider.h`, `ConnectionSignature` stores connection kind, value-type rendering, canonical endpoint/route/engine/map/storage renderings and the cost-model version. Add `lookupCycles(const ConnectionSignature &, const TargetContext &) const` with a default nullopt implementation. Extend TargetContext with rule/layout content hashes, default-initialized for source compatibility. Use schema-aware canonical serialization for keys.

C5 defines in `MappedCompilation.h`:

```cpp
struct MappedCompileOptions {
  std::string entrySymbol;
  bool requireExecutable = true;
};
llvm::Expected<std::unique_ptr<llk::MappedExecutable>> compileMappedKernel(
    mlir::ModuleOp source, const MappingTarget &, const CoveringPlan &,
    const MappedCompileOptions &);
```

The CLI/tuner's schedule-instantiation and search adapter produces the source/target/plan passed to this lower-level compilation API; it does not repeat search inside `compileMappedKernel`. Target/candidate file options use the existing MicroMapOptions in that adapter. MappedExecutable is defined before C5 by C4, in `include/LLK/Runtime/MappedExecutable.h` and `runtime/MappedExecutable.cpp`.

## Verification workflow

In the execution worktree:

```sh
cmake -S . -B build -G Ninja \
  -DLLVM_PROJECT_BUILD_DIR=/Users/skg7on/Workspace/Projects/llvm-project/build
cmake --build build -j 6
ctest --test-dir build --output-on-failure -j 6
```

Each task first adds a behavioral failing regression, runs its named test, makes the smallest coherent change, reruns focused tests, then commits. Before each delivery PR, build relevant tools and run the registered suite once. Repeat broad tests after new failures/changes, not as a substitute for targeted evidence. CI must also build/link with static MLIR; attach the run/revision to the delivery evidence.

## Completion evidence

| Requirement | Owning task / gate |
|---|---|
| Seven fresh review defects | A2–A9 |
| Durable endpoint/resource identity | A1/A2, B1/B2 |
| Tile routes/transforms/gather/allocation/sync | B3–B6 |
| Exhaustive bounded exact choices and tie behavior | A6, B7 |
| Selected target bundle actually determines code | C2/C3 |
| Callable kernel ABI and numeric execution | C1/C4 |
| Global constraints, owner/memory/pipeline/tail choices | A7, B2, C5 |
| Shared events, full cache/route identity, perf parity | A9, B8, C6 |
| Multi-node target rule / fused coverage | C7 |
| All five §25.5 fixture chains and §29 criteria | C8/C10 |
| Production measurement/calibration | Interface hook in C6; implementation continues under #51/#52 |

The twelve normative §29 criteria are tracked individually. Existing coverage must be retained and re-run; a criterion is complete only when its delivery-head evidence is recorded by C10.

| §29 | Required behavior | Task owner and evidence |
|---|---|---|
| 1 | Deterministic search-space and complete-binding round trip | A7/C5/C8: preserve dialect round trips; instantiate and replay full bindings, including non-MMA controls. |
| 2 | Normalized concrete MachineModel topology | B2/B3/B5/C8: retain loader/topology rejection tests; verify selected role resources, physical occupancy and same-kind node identity. |
| 3 | Deterministic direct and multi-hop route ranking | A6/B5/B7/C8: preserve routing tests, test equivalent alternatives and the two-hop acceptance fixture. |
| 4 | Target-owned layout legality without generic target branches | A4/A7/C9: verify recorded assignments and explicit role constraints; scan generic ODS/core policy boundaries. |
| 5 | Rules generate mapping candidates | A3/C7: retain single-op matching and add bounded fused matching with effect/region controls. |
| 6 | Concrete instances, connections and complete coverings | A1/A2/B2–B7: check exact endpoint coverage, explicit storage/sync and joint route feasibility. |
| 7 | Shared deterministic/beam/exact interface and honest limits | A6/B7: compare modes against an independent bounded oracle and verify cap/exhaustion reporting. |
| 8 | Selected placements, routes, transforms and bundles materialize | B1/B4–B6/C2/C3: round-trip metadata, verify actual SSA rewiring and prove the selected emitter changes emitted code. |
| 9 | Shared MachineModel and route/cost primitives | A9/B8/C6: compare normalized selected-plan and MicroDAG events, dependencies, keys and final latency. |
| 10 | Working AVX2 target and available legacy path | C1–C6/C8: invoke mapped programs numerically on a supported host and re-run legacy compile/runtime tests. |
| 11 | Second-target neutrality of generic Micro ODS | B5/B6/C8/C9: run generic-accelerator route/materialization/perf conformance; accelerator execution is not a required substitute for these checks. |
| 12 | Byte-identical normalized IR and reports on repeated runs | A1/B1/B7/C5/C8: repeat fresh-process extraction, search, bind and replay; compare normalized IR/report bytes. |

Do not close #67 when A alone passes, when only compile/lookup succeeds, or when a report declares unsupported work executable. C10 requires an evidence matrix with no unresolved mandatory criterion.
