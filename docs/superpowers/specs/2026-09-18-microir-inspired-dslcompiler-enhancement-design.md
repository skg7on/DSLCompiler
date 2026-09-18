# MicroIR-Inspired DSLCompiler Enhancement Design

**Status:** Draft for review
**Date:** 2026-09-18
**Scope:** M9-M13 Micro-IR roadmap enhancement
**Parent roadmap:** [Canonical DNN Micro-IR roadmap](https://github.com/skg7on/DSLCompiler/issues/41)
**Immediate dependency:** [Issue #44: implement search-space ops](https://github.com/skg7on/DSLCompiler/issues/44)

**Related design documents:**

- [Canonical Micro-IR redesign](2026-08-11-canonical-micro-ir-redesign.md)
- [Micro-IR auto-tuning implementation spec](2026-08-11-micro-ir-autotuning-implementation-spec.md)
- [Micro-IR tile programming model](2026-08-13-micro-ir-tile-programming-model-spec.md)
- [Micro-IR core concepts](../../design/m9-micro-ir-core-concepts.md)

---

## 1. Purpose

This document specifies how DSLCompiler should reuse the strongest concepts from the MicroIR reference prototype while preserving DSLCompiler's existing architecture and roadmap.

The reference comparison showed that DSLCompiler is the broader end-to-end compiler and runtime, while MicroIR has a more developed mapping-search model. The enhancement therefore imports concepts, not a second IR stack:

1. adapt MicroIR's mapping candidate, placed instance, connection, and complete covering concepts;
2. evolve `MachineModel` into a topology graph for placement and multi-hop routing;
3. add a SIRMap-like declarative mapping-rule layer;
4. add a declarative layout constraint language;
5. keep search spaces persistent in MLIR through `micro.search_space` and related ops;
6. keep Ampere, warp, TTGIR, and other target-specific concepts outside the target-independent `micro` dialect.

The intended result is a reusable mapping and tuning architecture for AVX2 first, followed by TTGIR, NPUs, and custom accelerators without redesigning the canonical Micro-IR.

This is a conceptual adaptation from the MicroIR prototype as inspected at commit `e9807ec`. DSLCompiler must implement the design in its own types, naming, tests, and target integrations; it must not vendor or create a build-time dependency on the reference repository.

---

## 2. Problem Statement

DSLCompiler already has the correct top-level separation:

```text
LLK / Linalg semantics
        |
        v
canonical micro execution IR
        |
        +--> MachineModel and micro-perf
        |
        +--> AVX2 / future accelerator backends
```

Its current and planned search model is strong at preserving user-visible choices:

```text
micro.search_space
  +-- micro.param
  +-- micro.constraint
  +-- micro.objective
  +-- micro.candidate
```

However, the roadmap does not yet define a complete bridge from those parameter bindings to a placed, connected, routable execution plan. In particular, it needs explicit answers for:

- how a lowering or target rule becomes a candidate;
- how an abstract owner becomes a concrete executor placement;
- how producer and consumer placements are connected;
- when a transfer or layout conversion must be inserted;
- how a complete set of choices covers the program;
- how topology, capacity, and target rules participate in legality;
- how target-specific layouts are described without leaking into `micro`;
- how a selected plan becomes deterministic concrete Micro-IR.

MicroIR addresses these problems with candidate, instance, connection, covering, topology, rule, and layout abstractions. DSLCompiler should adopt those ideas behind its existing canonical IR boundary.

---

## 3. Goals

### 3.1 Functional goals

The enhanced architecture shall:

- preserve search intent and candidate bindings in MLIR;
- map concrete `micro` operations or subgraphs to target implementation bundles;
- distinguish abstract resource ownership from concrete executor placement;
- select legal layouts using declarative target constraints;
- synthesize direct or multi-hop routes between placed producers and consumers;
- insert required transfer and data-transform steps into a plan;
- search complete program coverings deterministically;
- score plans using static costs and optional calibrated latency data;
- materialize the selected plan as concrete `micro.kernel` IR plus generic placement and route metadata;
- leave the existing AVX2 compile and JIT path working throughout adoption.

### 3.2 Architectural goals

The architecture shall support multiple targets without changing target-independent Micro ODS definitions for each target. It shall provide stable extension points for:

- target machine profiles;
- layout libraries;
- mapping rule libraries;
- latency providers;
- target emitters.

### 3.3 Quality goals

Search and binding must be reproducible. Given the same input IR, target configuration, search options, and calibration database, DSLCompiler shall produce the same ordered candidates, plan identifiers, diagnostics, and selected result.

---

## 4. Non-Goals

This design does not:

- import MicroIR's `SemanticIR` dialect;
- make TTIR, TTGIR, Triton encodings, warp counts, or Ampere layout names part of canonical `micro`;
- replace LLK or Linalg as the semantic input layer;
- require a general-purpose SMT solver for the first implementation;
- require functional emulation of Micro-IR;
- replace the current AVX2 backend before the new path is validated;
- prescribe a single search strategy for every target;
- store every intermediate C++ search object in MLIR.

---

## 5. Normative Design Decisions

The following decisions govern the rest of this specification.

### 5.1 `micro` remains the only canonical execution IR

DSLCompiler shall not add a second SemanticIR-like canonical layer. LLK/Linalg lowering produces searchable or concrete `micro`; the mapping engine consumes `micro` and returns a selected concrete `micro` program.

### 5.2 Persistent intent, transient search mechanics

MLIR stores durable, reproducible intent:

- parameter domains;
- constraints;
- objectives;
- named candidate bindings;
- the selected concrete kernel;
- generic placement, route, and rule-selection metadata required to reproduce lowering.

C++ stores transient search mechanics:

- enumerated rule candidates;
- placed instances;
- synthesized connections;
- partial coverings;
- pruning state;
- cached scores.

This preserves the advantage of issue #44 without forcing a potentially large beam or exact-cover frontier into the IR.

### 5.3 Abstract owner and concrete placement are different concepts

`!micro.tile` owner information describes the class of resource responsible for a value, such as a worker or vector engine. It does not identify a particular core, warp, DMA channel, or engine instance.

Concrete executor identifiers belong to a selected mapping plan and are validated against `MachineModel`. They are represented by generic executor references, never by target-specific fields in `!micro.tile`.

### 5.4 Target policy lives in target packages

Target-independent code defines interfaces and generic data structures. Target packages provide machine data, layout rules, mapping rules, costs, and emitters.

For example, `numWarps`, an NVIDIA MMA encoding, or a TTGIR blocked layout may appear in an NVIDIA target package but not in `MicroTypes.td`, `MicroAttrs.td`, or generic search constraints.

### 5.5 Selected plans are explicit

The mapping engine shall not hide transfer, conversion, route, or placement decisions solely in a report. Binding a selected plan must make execution-affecting decisions visible in concrete Micro-IR or in generic attributes attached to that IR.

---

## 6. Architecture Overview

```text
LLK / Linalg workload
        |
        | LLKToMicro / schedule export
        v
micro.search_space or concrete micro.kernel
        |
        | SearchSpaceLoader
        v
numeric + symbolic search bindings
        |
        | MappingProblemBuilder
        v
target-independent micro workload graph
        |
        +---------------------------+
        |                           |
        v                           v
MappingRuleRegistry          MachineModel topology
        |                    + LayoutConstraintSet
        v                           |
MappingCandidate generation        |
        |                           |
        v                           v
CandidateInstance placement + ConnectionPlan synthesis
        |
        v
CoveringSearch (deterministic / beam / exact)
        |
        v
CostModel + optional LatencyCache
        |
        v
selected CoveringPlan
        |
        | MappingPlanBinder
        v
concrete micro.kernel
  + placements
  + routes
  + transforms
  + selected target bundle identifiers
        |
        +--------------------+---------------------+
        |                    |                     |
        v                    v                     v
micro-perf             AVX2 lowering        future target emitter
                                             (TTGIR / NPU / custom)
```

The mapping subsystem is not a new IR. It is a planner between persistent Micro-IR and target lowering.

---

## 7. Terminology and Concept Mapping

| DSLCompiler concept | MicroIR reference concept | DSLCompiler interpretation |
|---|---|---|
| `!micro.tile` | ranked tensor plus semantic encoding | Canonical execution value; retains shape, element type, layout, memory space, and abstract owner |
| `micro.mma`, `micro.vector`, `micro.reduce` | generic semantic compute plus target bundle | Explicit target-independent compute whose implementations are supplied by mapping rules |
| `micro.tile_async_copy` | transfer plus memory route | Explicit movement op; selected plan supplies concrete route and engines |
| `micro.tile_partition` and layout attrs | data transform plus layout templates | Logical tile/layout transformation constrained by a target layout library |
| `micro.param` / `micro.candidate` | global parameters and mapping choices | Persistent search domain and complete user-visible binding |
| `MappingCandidate` | mapping candidate | A rule-derived implementation option for one op or a compatible subgraph |
| `CandidateInstance` | candidate instance | A mapping candidate with concrete executor and memory placement |
| `ConnectionPlan` | connection plan | The transport and transformation needed between two placed instances |
| `CoveringPlan` | covering plan | A complete non-overlapping implementation of the workload graph |
| `MachineModel` | hardware JSON plus topology | Versioned target capabilities, hierarchy, topology, capacities, links, and costs |

The terms `micro.candidate` and `MappingCandidate` are intentionally distinct:

- `micro.candidate` is a complete persistent binding of named search parameters;
- `MappingCandidate` is an internal rule-derived implementation option for part of the workload.

Code and diagnostics must use the full names to avoid ambiguity.

---

## 8. Persistent Search IR Contract

Issue #44 remains the source of truth for the first search ops. The enhancement does not expand its implementation scope into the full mapping engine.

### 8.1 Required issue #44 behavior

The initial ops shall preserve:

- explicitly typed parameter domains;
- integer and symbolic choices;
- stable constraint and objective kinds;
- complete candidate bindings;
- deterministic ordering and printing;
- local verification of domain values;
- parent-level verification of references and complete bindings.

Representative IR:

```mlir
micro.search_space @gemm_space attributes {workload = "gemm"} {
  micro.param "BM" {kind = "integer", choices = [32 : i64, 64 : i64]}
  micro.param "tile_layout" {
    kind = "layout", choices = ["blocked", "row_major"]
  }
  micro.param "memory_path" {
    kind = "memory_path", choices = ["dram:l2:sram", "dram:sram"]
  }
  micro.constraint "sram_capacity" {params = ["BM", "tile_layout"]}
  micro.objective {direction = "minimize", metric = "latency_cycles"}
  micro.candidate @candidate_17 {
    bindings = {BM = 64 : i64, memory_path = "dram:l2:sram",
                tile_layout = "blocked"}
  }
}
```

### 8.2 Boundary between issue #44 and later work

Issue #44 validates structural and vocabulary-level correctness. It does not determine whether:

- a target has enough memory;
- a fragment is supported by a compute unit;
- a requested route exists;
- a layout satisfies target constraints;
- a candidate covers all operations;
- a candidate has the best predicted latency.

Those checks require `MachineModel`, layout rules, mapping rules, and the mapping engine introduced in later milestones.

### 8.3 Search binding handoff

`SearchSpaceLoader` converts a `micro.candidate` to a typed `SearchBinding`:

```cpp
namespace llk::mapping {

using SearchValue = std::variant<int64_t, std::string>;

struct SearchBinding {
  std::string candidateId;
  llvm::StringMap<SearchValue> values;
  uint64_t stableHash;
};

} // namespace llk::mapping
```

Bindings are immutable after validation. Every later plan records the source binding hash.

---

## 9. Mapping Search Data Model

The mapping data model adapts MicroIR's four useful layers and makes their contracts explicit for DSLCompiler.

### 9.1 Stable identifiers

Search objects shall reference stable IDs rather than owning or relying on raw MLIR pointers for serialization and comparison:

```cpp
using WorkloadNodeId = uint32_t;
using WorkloadValueId = uint32_t;
using RuleId = std::string;
using ExecutorId = std::string;
using MemoryNodeId = std::string;
using LinkId = std::string;
using LayoutId = std::string;
using CandidateId = uint64_t;
using InstanceId = uint64_t;
using ConnectionId = uint64_t;
using PlanId = uint64_t;
```

Pointers may be used internally during a single pass invocation, but stable keys drive hashing, sorting, reports, and cache entries.

### 9.2 `MappingCandidate`

A `MappingCandidate` is an unplaced rule match. It describes a legal implementation shape but not a specific hardware instance.

```cpp
struct MappingCandidate {
  CandidateId id;
  RuleId rule;
  llvm::SmallVector<WorkloadNodeId> coveredNodes;
  std::string targetBundle;
  llvm::SmallVector<PortSpec> ports;
  llvm::SmallVector<ExecutorRequirement> executorRequirements;
  llvm::SmallVector<MemoryRequirement> memoryRequirements;
  llvm::SmallVector<LayoutRequirement> layoutRequirements;
  llvm::StringMap<SearchValue> resolvedParameters;
  Cost lowerBound;
};
```

Rules:

- `coveredNodes` is non-empty, sorted, and unique;
- a workload node may be covered once in a complete plan unless a rule explicitly declares fused coverage;
- ports identify all values crossing the covered subgraph boundary;
- requirements name abstract capabilities, never concrete target IDs;
- `lowerBound` must be optimistic so it is safe for pruning.

### 9.3 `CandidateInstance`

A `CandidateInstance` places one mapping candidate on concrete machine resources.

```cpp
struct CandidateInstance {
  InstanceId id;
  CandidateId candidate;
  llvm::StringMap<ExecutorId> executorBindings;
  llvm::StringMap<MemoryNodeId> memoryBindings;
  llvm::StringMap<LayoutId> layoutBindings;
  ResourceUsage resourceUsage;
  Cost localCost;
};
```

An instance is legal only if:

- each bound resource exists in the selected `MachineModel`;
- capability, containment, concurrency, and placement constraints hold;
- memory capacities and live-range estimates are not exceeded;
- selected layouts satisfy target constraints;
- every binding is deterministic and complete.

### 9.4 `ConnectionPlan`

A `ConnectionPlan` connects one producer port to one or more consumer ports.

```cpp
enum class ConnectionKind {
  Direct,
  Transfer,
  LayoutTransform,
  TransferAndTransform,
  Replicate,
  Reduce
};

struct ConnectionPlan {
  ConnectionId id;
  InstanceId producer;
  llvm::SmallVector<InstanceId> consumers;
  WorkloadValueId value;
  ConnectionKind kind;
  llvm::SmallVector<MemoryNodeId> memoryRoute;
  llvm::SmallVector<ExecutorId> transferEngines;
  std::optional<AffineMap> producerMap;
  llvm::SmallVector<AffineMap> consumerMaps;
  std::optional<LayoutTransform> transform;
  Cost cost;
};
```

Connections are first-class because a locally legal producer and consumer pair may be globally illegal or expensive once movement, conversion, fan-out, and topology are considered.

### 9.5 `CoveringPlan`

A `CoveringPlan` is a complete executable proposal.

```cpp
struct CoveringPlan {
  PlanId id;
  uint64_t sourceBindingHash;
  llvm::SmallVector<InstanceId> instances;
  llvm::SmallVector<ConnectionId> connections;
  llvm::StringMap<SearchValue> globalParameters;
  Cost totalCost;
  PlanDiagnostics diagnostics;
};
```

A valid plan shall:

- cover every required workload node exactly once;
- connect every external candidate input and output;
- obey global layout and target parameter constraints;
- fit resources under the selected scheduling model;
- have no illegal route or unresolved conversion;
- have a stable ID derived from canonical ordered content.

---

## 10. Workload Graph and Affine Mapping Semantics

The mapping engine consumes a target-independent graph extracted from concrete Micro-IR.

```cpp
struct WorkloadPort {
  WorkloadValueId value;
  Type type;
  std::optional<AffineMap> accessMap;
};

struct WorkloadNode {
  WorkloadNodeId id;
  OperationName opName;
  llvm::SmallVector<WorkloadPort> inputs;
  llvm::SmallVector<WorkloadPort> outputs;
  DictionaryAttr attributes;
};
```

### 10.1 Use MLIR affine maps as the common representation

DSLCompiler shall reuse MLIR `AffineMap` and `IntegerSet` representations for index relationships instead of defining target-specific mapping syntax in C++.

Affine mappings express:

- how a target rule consumes a producer tile;
- how a result tile is distributed over logical executor coordinates;
- how a layout transform relates source and destination indices;
- whether two ports can connect directly;
- which dimensions are broadcast, reduced, sliced, or permuted.

### 10.2 Mapping compatibility

Two connected ports are direct-compatible only if their element type, logical tile shape, memory visibility, and affine index relation agree. Otherwise the connection synthesizer must either:

- select a supported layout transform;
- select a supported transfer plus transform;
- reject the pair with a structured diagnostic.

Affine equivalence and composition shall use MLIR canonicalization utilities. Target rule authors must not implement ad hoc string comparison for maps.

---

## 11. MachineModel v2: Topology and Placement

Issue #45 should introduce a versioned MachineModel schema whose graph is sufficient for both performance evaluation and mapping legality.

### 11.1 Node kinds

The model contains stable-ID nodes of four kinds:

- `executor`: a place where work can execute;
- `memory`: a place where data can reside;
- `compute`: a capability attached to an executor;
- `transfer_engine`: a resource that moves data over one or more links.

### 11.2 Edge kinds

The graph contains:

- `contains`: hierarchy and ownership;
- `dominates`: visibility or scope dominance;
- `attached_to`: compute or transfer resources attached to an executor;
- `link`: directed data path with supported traffic and cost properties.

### 11.3 Required properties

Each executor has:

- stable ID and abstract owner kind;
- optional logical coordinates;
- parent executor;
- supported concurrency and scheduling properties.

Each memory has:

- stable ID and abstract memory-space kind;
- capacity and alignment;
- visibility scope;
- supported layouts or layout classes;
- optional banking and transaction properties.

Each compute resource has:

- stable ID and capability kind;
- supported element types;
- fragment or vector shapes;
- throughput, issue, and latency properties;
- concurrency and occupancy limits.

Each link has:

- source and destination node IDs;
- bandwidth, fixed latency, and transaction granularity;
- supported transfer engines;
- directionality and concurrency class.

### 11.4 Representative YAML shape

```yaml
schema: llk.machine.v2
target: x86-avx2

executors:
  - id: package.0
    kind: system
  - id: core.0
    kind: worker
    parent: package.0
    coordinates: [0]

memories:
  - id: dram.0
    kind: dram
    visible_from: package.0
    capacity_bytes: 17179869184
  - id: l1.0
    kind: sram
    visible_from: core.0
    capacity_bytes: 32768

compute:
  - id: avx2.0
    kind: vector_engine
    attached_to: core.0
    element_types: [f32]
    vector_bits: 256

links:
  - id: dram_to_l1.0
    source: dram.0
    destination: l1.0
    bandwidth_bytes_per_cycle: 32
    latency_cycles: 20
```

The concrete schema may split repeated nodes through templates, but loading must produce this normalized graph.

### 11.5 Owner-to-placement rule

An abstract Micro owner matches a machine executor when the executor's `kind` is equal to or refines that owner. A mapping rule may add further capability requirements. Placement never succeeds by comparing a target-specific executor ID with a Micro owner string.

### 11.6 Schema compatibility

The loader shall reject unknown major schema versions. Minor additions must have defaults or be explicitly optional. MachineModel v1 profiles, if created before this design lands, shall be migrated by a standalone converter or compatibility loader rather than silently reinterpreted.

---

## 12. Multi-Hop Routing

Direct source-to-destination copy assumptions are insufficient for accelerator memory hierarchies. The topology service shall enumerate and score legal paths.

### 12.1 Routing API

```cpp
struct RouteRequest {
  MemoryNodeId source;
  MemoryNodeId destination;
  uint64_t bytes;
  Type elementType;
  std::optional<ExecutorId> producerExecutor;
  std::optional<ExecutorId> consumerExecutor;
};

struct MemoryRoute {
  llvm::SmallVector<MemoryNodeId> nodes;
  llvm::SmallVector<LinkId> links;
  llvm::SmallVector<ExecutorId> transferEngines;
  Cost cost;
};

class TopologyService {
public:
  FailureOr<llvm::SmallVector<MemoryRoute>>
  enumerateRoutes(const RouteRequest &, unsigned limit) const;
};
```

### 12.2 Route legality

A route is legal when:

- consecutive nodes are joined by compatible directed links;
- a legal transfer engine is available for every hop that requires one;
- the value type, alignment, layout, and transaction size are supported;
- intermediate capacity and liveness requirements are satisfied;
- the producer and consumer can access their endpoint memories;
- the path contains no repeated node.

### 12.3 Determinism and pruning

Route enumeration shall be cycle-free and deterministic. It sorts paths by:

1. estimated route cost;
2. hop count;
3. lexicographic sequence of stable link IDs.

The first implementation uses bounded best-first search. It returns at most `maxRoutesPerConnection`; it does not enumerate all paths in a large topology.

### 12.4 Materialization

A selected multi-hop route is materialized as either:

- a sequence of existing `micro.tile_async_copy`, `micro.wait`, and allocation operations when intermediate storage is semantically required; or
- a generic route attribute on a single copy op when the backend guarantees atomic route lowering and no intermediate tile is observable.

The binder must choose one representation consistently for a target. Performance evaluation must observe every hop in both forms.

---

## 13. Declarative Layout Constraint Language

Target layout legality must not be a growing set of C++ `if` statements. DSLCompiler shall add a small declarative language named **LLKMap**, with a layout declaration subset.

### 13.1 File ownership

```text
mapping/
  x86-avx2/
    layouts.llkmap
    rules.llkmap
  nvidia-sm80/
    layouts.llkmap
    rules.llkmap
```

Machine topology remains in `machines/*.yaml`. LLKMap files reference capabilities through stable MachineModel queries.

### 13.2 Layout declarations

A layout declaration defines:

- a stable layout ID;
- applicable rank and element types;
- integer and symbolic parameters;
- constraints over tile dimensions, parameters, and machine queries;
- an affine logical-to-physical mapping;
- optional cost hints.

Illustrative syntax:

```text
layout avx2.blocked_2d(M, N, VW) {
  require rank == 2;
  require VW == machine.compute("vector_engine").lanes(element_type);
  require N % VW == 0;
  map (m, n) -> (m, floordiv(n, VW), mod(n, VW));
}
```

The shown declarations and expressions define the required LLKMap surface constructs. Whitespace, comments, escaping, and recovery behavior shall follow LLVM source conventions and be fixed in the LLKMap parser specification.

### 13.3 Constraint evaluator

The first implementation uses a deterministic typed expression evaluator with bounded enumeration over declared finite domains. It supports:

- integer arithmetic and comparison;
- boolean conjunction, disjunction, and negation;
- divisibility and bounded ranges;
- affine expressions and maps;
- finite quantification over declared executors or dimensions;
- read-only MachineModel capability queries.

It does not require Z3. A future solver may implement the same `LayoutSolver` interface without changing rule files or clients.

### 13.4 Canonical Micro layout versus target layout

`#micro.layout` describes target-independent layout properties used by Micro-IR. A target layout ID describes a concrete implementation satisfying those properties.

For example:

```text
#micro.layout<blocked, [16, 8]>       target-independent request
nvidia-sm80.mma_operand_a             target implementation layout
avx2.blocked_2d                       target implementation layout
```

Target layout IDs are mapping results, not new enumerants in the Micro dialect.

---

## 14. Declarative Mapping Rules and Target Bundles

The mapping-rule subset of LLKMap describes how Micro operations or small subgraphs can be implemented on a target.

### 14.1 Rule contents

Each rule declares:

- a stable rule ID and version;
- an input pattern over Micro operations;
- attribute, shape, dtype, and affine-map predicates;
- abstract executor, compute, memory, and layout requirements;
- named input and output ports;
- the target bundle to select;
- parameter domains or derived parameter expressions;
- a static cost lower bound;
- an emitter key understood by the target plugin.

Illustrative syntax:

```text
rule avx2.micro_vector_f32 {
  match micro.vector(kind = "add", element_type = f32);
  require executor kind worker;
  require compute kind vector_engine;
  require layout operand0 satisfies avx2.blocked_2d;
  bundle "avx2.vector.add.f32";
  emit "avx2_vector_add";
}
```

### 14.2 Rule matching

Rules may cover one operation initially. Fused multi-operation patterns are allowed later, but must list every covered workload node and every boundary port explicitly.

Rule matching shall be side-effect aware. No rule may fuse across an operation with unknown effects, a synchronization boundary, or a region boundary it does not model.

### 14.3 Target bundle contract

A target bundle is an opaque target-owned implementation name plus typed parameters. Generic code may compare, hash, report, and pass it to a target plugin; it shall not interpret Ampere, TTGIR, NPU, or AVX2-specific fields.

```cpp
struct TargetBundle {
  std::string name;
  DictionaryAttr parameters;
  std::string emitterKey;
};
```

### 14.4 Rule validation

At load time, the registry rejects:

- duplicate rule IDs;
- unknown Micro operation names;
- missing or duplicate ports;
- unknown machine capability queries;
- unbound parameters;
- invalid affine maps;
- unknown layout IDs;
- unknown emitter keys for the selected target plugin.

This makes target configuration failures startup diagnostics rather than late search failures.

---

## 15. Placement and Connection Synthesis

### 15.1 Placement enumeration

For each `MappingCandidate`, the placement engine:

1. resolves abstract executor requirements against matching MachineModel nodes;
2. enumerates compatible compute and memory attachments;
3. solves layout constraints for each required port;
4. estimates local resource use;
5. emits legal `CandidateInstance` objects in stable order.

Symmetric placements may be canonicalized when the target declares executors equivalent. Symmetry reduction must preserve at least one representative and must be disableable for diagnostics.

### 15.2 Connection synthesis

For each dataflow edge between two instances, the connection engine attempts in order:

1. direct connection;
2. layout-only transform in a mutually visible memory;
3. direct memory transfer;
4. transfer plus layout transform;
5. bounded multi-hop transfer, with transform placed at a legal hop.

Every alternative becomes a `ConnectionPlan`. An absence of alternatives makes that pair of instances incompatible; it does not immediately reject other placements.

### 15.3 Fan-out and fan-in

The engine shall model:

- shared reads when all consumers can legally access one placement;
- replication when consumers require distinct placements;
- reductions or gathers when multiple producers feed one logical value;
- cost and capacity of replicated or intermediate tiles.

---

## 16. Covering Search

### 16.1 Search stages

The end-to-end mapping search is:

```text
load SearchBinding
  -> build workload graph
  -> enumerate MappingCandidates
  -> enumerate CandidateInstances
  -> synthesize ConnectionPlans lazily
  -> select a complete CoveringPlan
  -> score and rank complete plans
  -> bind selected plan to Micro-IR
```

### 16.2 Search modes

The implementation shall provide three modes behind one interface:

- `deterministic`: first legal plan in canonical order, intended for tests and bring-up;
- `beam`: bounded best-first search, the production default;
- `exact`: branch-and-bound exact-cover search for small graphs and validation.

```cpp
struct MappingSearchOptions {
  SearchMode mode = SearchMode::Beam;
  unsigned beamWidth = 64;
  unsigned topK = 8;
  unsigned maxCandidatesPerNode = 64;
  unsigned maxInstancesPerCandidate = 64;
  unsigned maxRoutesPerConnection = 8;
  uint64_t memoryBudgetBytes = 512ULL << 20;
  bool enableLatencyCache = true;
  bool enableSymmetryReduction = true;
};
```

All caps are explicit options and are reported with the result. Reaching a cap shall produce a `search_truncated` diagnostic rather than silently claiming optimality.

### 16.3 Partial-plan state

A partial plan records:

- covered workload nodes;
- selected instances;
- materialized connection choices;
- live resource use;
- accumulated cost;
- admissible lower bound for uncovered work.

The beam is ordered by lower bound, then covered-node count descending, then stable partial-plan ID.

### 16.4 Exact-cover semantics

Fused mapping rules may cover several nodes, so covering selection is a constrained exact-cover problem augmented with connectivity and resource constraints. The exact mode branches on the lowest-ID uncovered node and tries candidates in stable order.

### 16.5 Failure reporting

When no plan exists, the result reports the earliest deterministic failure frontier with counts for:

- nodes with no matching rules;
- candidates with no legal placement;
- instance pairs with no legal connection;
- plans rejected by capacity;
- plans rejected by global layout or parameter constraints.

---

## 17. Cost Model and Latency Cache

### 17.1 Cost structure

Costs remain multi-dimensional until the objective ranks them:

```cpp
struct Cost {
  double latencyCycles = 0.0;
  uint64_t dramBytes = 0;
  uint64_t localBytes = 0;
  uint64_t spillBytes = 0;
  double computeUtilization = 0.0;
  double transferUtilization = 0.0;
};
```

`micro.objective` supplies the primary and secondary comparison order. A target may provide estimates but may not replace the declared objective.

### 17.2 Static cost

Static cost combines:

- rule-local compute cost;
- route transfer cost per hop;
- layout transformation cost;
- synchronization cost;
- resource overlap according to the MachineModel scheduling policy;
- penalties for spills or unsupported fast paths.

Issue #46's L0/L1 performance evaluator should consume the same normalized MachineModel and selected plan events rather than developing a parallel target description.

### 17.3 Latency provider interface

```cpp
class LatencyProvider {
public:
  virtual ~LatencyProvider() = default;
  virtual std::optional<double>
  lookupCycles(const OperationSignature &, const TargetContext &) const = 0;
};
```

Providers may use analytical estimates, calibration records, or measurements. Absence of a cache entry falls back to static cost; it does not make a candidate illegal.

### 17.4 Cache key

The latency cache key includes:

- machine profile content hash;
- rule ID and rule version;
- operation name, operand/result types, and relevant attributes;
- target bundle parameters;
- concrete layout and placement classes;
- route class when movement is measured;
- compiler cost-model version.

This prevents stale measurements from being silently reused after a rule or machine profile changes.

---

## 18. Materializing the Selected Plan

`MappingPlanBinder` clones the source concrete kernel and applies a selected plan without mutating the search-space operation.

### 18.1 Generic mapping metadata

The Micro dialect may add generic attributes with semantics such as:

- executor reference: abstract kind plus MachineModel node ID;
- memory reference: Micro memory kind plus MachineModel node ID;
- selected target layout ID;
- selected rule and target bundle ID;
- ordered memory route.

These attributes must remain target-neutral containers. An attribute may hold `nvidia-sm80.mma_a` as an opaque selected layout ID, but Micro verifiers shall not know what that ID means beyond structure. Machine-aware validation belongs to the selected target plugin.

### 18.2 Inserted operations

The binder inserts or refines:

- tile allocations for required intermediate memories;
- tile copies for every materialized route hop;
- layout conversions where selected connections require them;
- waits or barriers required by transfer dependencies;
- placement metadata on compute, memory, and structured control operations.

### 18.3 Verification phases

Verification is intentionally layered:

1. ordinary Micro dialect verification checks structural correctness;
2. `verify-micro-mapping --machine=<profile>` resolves IDs, rules, layouts, and routes;
3. the target emitter verifies target bundle completeness before lowering.

Generic `llk-opt` must still be able to parse and print mapped Micro-IR without loading a target plugin.

---

## 19. Target Plugin Boundary

Targets implement a common registry contract:

```cpp
class MappingTarget {
public:
  virtual StringRef getTargetName() const = 0;
  virtual const MachineModel &getMachineModel() const = 0;
  virtual const LayoutRegistry &getLayouts() const = 0;
  virtual const MappingRuleRegistry &getRules() const = 0;
  virtual const LatencyProvider *getLatencyProvider() const = 0;
  virtual std::unique_ptr<TargetEmitter> createEmitter() const = 0;
};
```

### 19.1 Generic core owns

- search-space loading;
- workload graph extraction;
- common candidate/instance/connection/covering types;
- topology algorithms;
- layout and rule parsing infrastructure;
- generic search algorithms;
- reports and deterministic IDs;
- plan binding infrastructure.

### 19.2 Target package owns

- MachineModel profile content;
- target layouts and constraints;
- mapping rules and target bundles;
- specialized cost or latency provider;
- backend emitter;
- target-specific diagnostics.

### 19.3 NVIDIA example boundary

An NVIDIA target package may own:

- block, warp, and thread executor nodes;
- global, shared, and register memory nodes;
- `numWarps` and CTA-shape global parameters;
- MMA operand/result encodings;
- swizzled shared-memory layouts;
- TTGIR target bundles and emitter.

None of those become required fields or enumerants in the generic Micro dialect.

---

## 20. Proposed Repository Layout

The following layout keeps generic mapping code separate from performance modeling and target integrations:

```text
include/LLK/Mapping/
  SearchBinding.h
  WorkloadGraph.h
  MappingPlan.h
  MappingRules.h
  LayoutConstraints.h
  Placement.h
  Routing.h
  CoveringSearch.h
  CostModel.h
  MappingTarget.h

lib/Mapping/
  SearchBinding.cpp
  WorkloadGraph.cpp
  MappingPlan.cpp
  MappingRules.cpp
  LayoutConstraints.cpp
  Placement.cpp
  Routing.cpp
  CoveringSearch.cpp
  CostModel.cpp

include/LLK/Machine/
  MachineModel.h
  MachineModelLoader.h
  Topology.h

lib/Machine/
  MachineModel.cpp
  MachineModelLoader.cpp
  Topology.cpp

include/LLK/Target/AVX2/Mapping/
lib/Target/AVX2/Mapping/

machines/
  x86-avx2-cpu.yaml
  generic-accelerator.yaml

mapping/
  x86-avx2/
    layouts.llkmap
    rules.llkmap
```

Target-independent mapping code shall not be placed under the AVX2 backend or the `Perf` namespace. Existing tuning types under `include/LLK/Perf` should delegate to or migrate into `LLK/Mapping` as this design lands, avoiding duplicate `Candidate` models.

---

## 21. Public Passes and Tools

The enhanced pipeline should expose narrow, composable entry points:

```text
--micro-build-workload-graph
--micro-map="target=x86-avx2 mode=beam top-k=8"
--micro-bind-plan="plan-id=<id>"
--micro-verify-mapping="machine=<path>"
```

The exact pass split may combine graph construction and mapping internally, but user-facing behavior shall support:

- emitting a plan report without modifying input IR;
- selecting a plan by stable ID;
- binding the best plan directly;
- verifying already-mapped Micro-IR.

Tool integration:

- `llk-opt` registers generic passes and target plugins built into the binary;
- `llk-compile --emit=micro` emits concrete unmapped or mapped Micro-IR by explicit option;
- `llk-compile --emit=micro-search` preserves issue #44 search spaces;
- `llk-tune` drives search bindings, mapping, ranking, and optional measurement;
- `micro-perf` consumes the same selected plan and MachineModel.

---

## 22. Determinism, Reports, and Diagnostics

### 22.1 Canonical ordering

The implementation shall sort:

- rules by rule ID;
- workload nodes by deterministic graph ID;
- machine nodes and links by stable ID;
- candidates by covered-node sequence, rule ID, then ID;
- placements by executor, memory, and layout binding tuples;
- routes by cost, hop count, then link sequence;
- complete plans by declared objective, then plan ID.

Unordered containers may be used for lookup but never as output iteration order.

### 22.2 Plan report

The mapping tool emits a versioned JSON report containing:

- input module hash;
- source search binding and hash;
- machine, layout-library, and rule-library hashes;
- search options and truncation flags;
- candidate, instance, route, and plan counts;
- rejected counts grouped by stable reason code;
- top-K plans and component costs;
- selected plan ID;
- compiler and cost-model version.

The report is diagnostic and reproducibility metadata. It does not replace `micro.search_space` or the selected concrete Micro-IR.

### 22.3 Diagnostic codes

At minimum, stable codes cover:

```text
no_matching_rule
no_legal_layout
no_legal_executor
memory_capacity_exceeded
unsupported_compute_fragment
no_memory_route
no_layout_transform
global_constraint_failed
search_truncated
latency_cache_miss
target_bundle_invalid
```

Diagnostics include involved workload node IDs, rule IDs, resource IDs, and source locations when available.

---

## 23. Milestone Integration

This design refines the existing #41 roadmap rather than replacing it.

### 23.1 M9 / issue #44: persistent search vocabulary

Implement the existing search ops and verifier boundaries. Add no topology search or rule solver to #44. Ensure symbolic domains can name layouts, owner mappings, memory paths, fragment shapes, and policies without embedding target enumerations.

### 23.2 M10 / issue #45: MachineModel v2 foundation

Implement:

- versioned loader;
- normalized executor/memory/compute/transfer graph;
- stable IDs and validation;
- containment, visibility, and link queries;
- bounded route enumeration;
- generic accelerator and AVX2 profiles.

This is the earliest milestone where explicit executor placement and multi-hop route legality become available.

### 23.3 M10 / issue #46: shared cost events

Make `micro-perf` consume normalized compute, transfer-hop, transform, synchronization, and capacity events. The mapping engine and performance evaluator must share event and cost definitions.

### 23.4 M11 / issues #47 and #48: workload and search export

- #47 emits concrete tile dataflow suitable for `WorkloadGraph` extraction.
- #48 emits persistent search domains and bindings with stable names consumed by `SearchSpaceLoader`.

### 23.5 M12 / issue #49: mapping core

Expand the tuning-core implementation to include:

- LLKMap layout and rule loading;
- mapping candidate generation;
- placement enumeration;
- connection and route synthesis;
- deterministic and beam covering search;
- machine-aware legality.

Exact search may land in the same issue or a focused follow-up, but the common interface and state model must be present.

### 23.6 M12 / issue #50: binding and outputs

Implement:

- plan ranking by Micro objective;
- selected-plan materialization;
- target bundle handoff;
- schedule YAML and plan JSON output;
- stable plan selection by ID.

### 23.7 M13: measurement and calibration

Add latency providers and versioned cache records without changing mapping plan semantics. Measurement enriches cost; it does not redefine legality.

---

## 24. Implementation Slices

Each slice should land with tests and preserve the legacy AVX2 path.

1. **Search IR foundation** — complete issue #44 exactly at its current scope.
2. **Normalized MachineModel graph** — schema, loader, validation, and query API.
3. **Topology routing** — deterministic direct and multi-hop route tests.
4. **Mapping data model** — stable IDs, workload graph, candidates, instances, connections, coverings.
5. **LLKMap layout subset** — parser, type checker, finite-domain solver, AVX2 examples.
6. **LLKMap rule subset** — one-op rules, ports, requirements, bundles, registry validation.
7. **Placement and connection engine** — instances, direct connections, transfers, transforms.
8. **Covering search** — deterministic mode, beam mode, exact validation mode.
9. **Plan binder** — placements, routes, inserted operations, target bundle metadata.
10. **AVX2 integration** — rules, layouts, emitter handoff, cost and tuning integration.
11. **Calibration integration** — latency provider and cache.
12. **Second-target proof** — a small generic-accelerator or NVIDIA configuration proving no generic dialect changes are needed.

A separate implementation plan should break these slices into commits only after this design is approved.

---

## 25. Testing Strategy

### 25.1 Dialect tests

- round-trip all issue #44 ops twice through `llk-opt`;
- reject duplicate, missing, mixed-type, and out-of-domain bindings;
- parse and print generic placement, route, and bundle metadata without a target plugin;
- reject structurally invalid generic metadata.

### 25.2 MachineModel tests

- load valid v2 AVX2 and generic-accelerator profiles;
- reject duplicate IDs, missing parents, dangling links, cycles in containment, invalid capacities, and unsupported versions;
- query owner-to-executor matches and memory visibility;
- enumerate direct and multi-hop routes in deterministic order;
- reject routes with missing engines or inaccessible endpoints.

### 25.3 Layout and rule tests

- parse, print, and reload LLKMap files deterministically;
- reject unknown queries, unbound variables, invalid maps, duplicate IDs, and unknown emitters;
- solve positive and negative AVX2 layout cases;
- prove target-specific names do not appear in generic Micro ODS files.

### 25.4 Mapping engine unit tests

- generate candidates for single-op and fused-rule fixtures;
- enumerate placements across equivalent and non-equivalent executors;
- synthesize direct, transfer, transform, and multi-hop connections;
- reject capacity, capability, and affine-map mismatches;
- find identical best plans in repeated runs;
- compare exact and beam results on small graphs;
- report truncation when configured caps are reached.

### 25.5 End-to-end tests

At minimum, use:

- elementwise vector operation;
- tiled GEMM with memory staging;
- fused SwiGLU or comparable multi-op graph;
- a fixture requiring a layout transform;
- a generic-accelerator fixture requiring a two-hop memory route.

For each fixture, verify:

- search-space round trip;
- selected rule and placement;
- route and inserted transform/copy operations;
- deterministic plan report;
- successful `micro-perf` evaluation;
- unchanged legacy AVX2 compilation when the new mapping path is disabled.

### 25.6 Fuzz and property tests

Add bounded property tests for:

- route paths never containing repeated nodes;
- complete plans covering all required nodes exactly once;
- stable IDs not depending on insertion order;
- serialized then reloaded reports preserving plan identity;
- cost ordering satisfying deterministic tie breaks.

---

## 26. Migration and Compatibility

### 26.1 Opt-in rollout

The new path starts behind explicit flags. Existing LLK-to-AVX2 lowering remains the default until end-to-end correctness and performance validation meet roadmap criteria.

### 26.2 Existing Micro-IR

Unmapped concrete `micro.kernel` remains valid. Placement and route metadata are optional until a pass or emitter requires mapped input.

### 26.3 Existing tuning code

Current `LLK/Perf` search structures should be adapted incrementally:

- `SearchParam`, constraints, and objectives become the loader-side search vocabulary;
- the existing complete parameter `Candidate` becomes `SearchBinding` or a compatibility wrapper;
- mapping candidates use the explicit `MappingCandidate` name;
- schedule YAML remains an output adapter over selected plans.

No two independently evolving candidate, legality, or MachineModel implementations should remain after M12.

### 26.4 Configuration stability

Machine, layout, and rule files are versioned and content-hashed. Reports record those hashes. Renaming a rule or resource is a compatibility change and must either provide an alias during migration or invalidate old plan selections explicitly.

---

## 27. Risks and Mitigations

### 27.1 Search-space explosion

**Risk:** Layout, placement, route, and fusion choices multiply rapidly.

**Mitigation:** Early lower bounds, capability filtering, symmetry reduction, bounded routes, lazy connection synthesis, beam limits, and explicit truncation diagnostics.

### 27.2 A bespoke rule language becomes a compiler project

**Risk:** LLKMap grows beyond mapping needs.

**Mitigation:** Keep the first grammar typed and small, reuse MLIR affine syntax and parsers where practical, use finite domains, and exclude arbitrary functions or general control flow.

### 27.3 MachineModel and Micro-IR duplicate semantics

**Risk:** Owners, memories, and layouts drift between IR and target profiles.

**Mitigation:** Micro declares abstract kinds; MachineModel declares concrete instances and capabilities; a single resolver owns matching and diagnostics.

### 27.4 Target details leak into generic code

**Risk:** The first accelerator integration introduces warp or TTGIR assumptions in core APIs.

**Mitigation:** Opaque target bundle parameters, generic executor and layout IDs, target-plugin validation, and a second-target conformance fixture.

### 27.5 Plans become unreproducible

**Risk:** Unordered enumeration, changing cache contents, or mutable profiles alter results.

**Mitigation:** Canonical ordering, content hashes, stable IDs, explicit search limits, report versions, and cache-version keys.

### 27.6 Mapping and performance models diverge

**Risk:** Search selects a plan using costs that `micro-perf` interprets differently.

**Mitigation:** Share normalized MachineModel nodes, events, routes, and `Cost` primitives; add cross-check tests comparing planner component sums with performance evaluator inputs.

---

## 28. Alternatives Considered

### 28.1 Import MicroIR's SemanticIR directly

Rejected. It would create a second canonical execution abstraction, duplicate Micro-IR concepts, and couple DSLCompiler to Triton/Ampere-oriented structures.

### 28.2 Keep all mapping logic in C++ without persistent MLIR search ops

Rejected. It would lose reproducible search intent, inspectability, FileCheck coverage, and the explicit compiler contract established by issue #44.

### 28.3 Put target-specific layouts in `#micro.layout`

Rejected. It would require dialect changes for each backend and make generic Micro-IR depend on warp, ISA, and backend encoding details.

### 28.4 Encode concrete executor IDs in `!micro.tile`

Rejected. Types would become tied to a loaded machine instance and would be difficult to rewrite during placement search. Abstract ownership remains in the tile type; concrete placement belongs to selected-plan metadata.

### 28.5 Use only YAML for mapping rules and layout constraints

Rejected as the primary format. YAML is appropriate for graph-shaped machine data, but affine maps, typed expressions, and quantified constraints require a language with stronger syntax and validation. YAML may still provide high-level manifests that list LLKMap files.

### 28.6 Require an SMT solver initially

Rejected. Target layout domains are finite for the MVP, and a deterministic bounded solver reduces dependencies and improves diagnostics. The solver interface allows a future SMT-backed implementation.

---

## 29. Acceptance Criteria

The enhancement design is realized when all of the following are true:

1. `micro.search_space` and complete candidate bindings round-trip deterministically.
2. A normalized MachineModel represents concrete executor, memory, compute, transfer, containment, visibility, and link topology.
3. The topology service finds and ranks direct and multi-hop routes deterministically.
4. Target layout legality is expressed in target-owned declarative files and solved without target-specific branches in generic Micro code.
5. Target mapping rules generate `MappingCandidate` objects for Micro operations.
6. The engine produces concrete `CandidateInstance`, `ConnectionPlan`, and complete `CoveringPlan` objects.
7. Deterministic, beam, and exact modes share one search interface and report truncation honestly.
8. The selected plan materializes placements, routes, transforms, and target bundle selections in concrete Micro-IR.
9. `micro-perf` and mapping search consume the same MachineModel and route/cost primitives.
10. AVX2 works as the first target while the legacy path remains available.
11. A second-target fixture demonstrates that Ampere, warp, TTGIR, NPU, or other target-specific concepts are unnecessary in generic Micro ODS definitions.
12. Repeated runs with identical inputs and configuration produce byte-identical normalized IR and plan reports.

---

## 30. Final Design Summary

DSLCompiler should reuse MicroIR's mapping-search concepts as a planner behind canonical Micro-IR, not import MicroIR's dialect stack.

The durable contract is:

```text
Search intent          = micro.search_space + complete bindings
Execution semantics    = concrete micro.kernel
Target facts           = MachineModel topology
Target policy          = LLKMap layouts + mapping rules
Search mechanics       = MappingCandidate -> CandidateInstance
                         -> ConnectionPlan -> CoveringPlan
Selected execution     = concrete micro.kernel + generic mapping metadata
Performance            = selected plan + MachineModel + calibration
```

This fills DSLCompiler's current mapping, placement, routing, layout, and covering gaps while preserving its strongest architectural properties: a target-independent tile execution IR, persistent MLIR search spaces, an end-to-end runtime path, and room for several backend families.
