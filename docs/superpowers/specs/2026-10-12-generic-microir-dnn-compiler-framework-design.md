# Generic MicroIR-Based DNN Compiler Framework — Goal and Design

**Date:** 2026-10-12

**Status:** Project direction revised to the requested generic framework scope; requirements below distinguish design intent from delivered support.

**Scope:** Project goals, component contracts, extension rules, search architecture and framework acceptance.

**Architecture:** [ARCHITECTURE.md](../../../ARCHITECTURE.md)

## 1. Intent and Authority

**Design a generic MicroIR-based, mapping- and search-based DNN compiler framework.**

The deliverable is reusable infrastructure for representing DNN execution, mapping it onto machine resources, exploring legal compilation choices, selecting a reproducible plan, and lowering or analyzing that plan. Kernel authors use it to explore implementations; compiler and target developers extend its semantic adapters, canonical primitives, machine descriptions, rules, analysis and backends.

Matmul and fused SwiGLU are two example operator kernels used to test MicroIR and the compilation framework. Their shapes, contraction arms, epilogues, schedules and performance targets are fixture-specific. They are neither the project goal nor the required form of every future workload. AVX2 is an executable validation target; a generic accelerator profile tests topology and mapping policy independently of hardware execution.

This document supersedes project-goal and scope statements that define DSLCompiler as a SwiGLU compiler, a fixed set of LLM kernels, or a CPU-only kernel optimizer. It governs architecture-level interpretation of existing design specifications. Their detailed IR, grammar and mapping contracts remain applicable where consistent with this goal. Historical implementation plans and revision-pinned reviews retain their original evidence and chronology.

The [feature matrix](../../manual/source/features.rst) and [PR #133 acceptance record](../../reviews/issue67-final-acceptance.md) remain the sources for delivered support. This revision does not implement a general importer, new operators, new hardware backends or a broader runtime ABI.

## 2. Architectural Approach

Use the existing canonical `micro` dialect and mapping core as the framework center. Keep operator semantics at the input boundary and target implementation details at the output boundary. Search consumes the same explicit execution artifacts and legality/analysis contracts used for compilation.

Two alternatives explain this choice:

- Expanding a fixed operator catalog would provide more kernels quickly, but would retain operator dispatch and contraction assumptions in shared infrastructure.
- Building a complete model importer and execution runtime first would enlarge the scope before the execution and search extension contracts are proven.

The selected approach proves generic infrastructure through small, independent conformance workloads, then expands semantic input and target coverage through those boundaries. A full model compiler remains a possible consumer of the framework; complete model import and orchestration are not prerequisites for this scope.

## 3. Canonical Artifacts and Pipeline

```text
Semantic/structured program + numerical contract
    -> decomposition / legal source transformations
    -> MicroIR tile program + declared search space
    -> original-source candidate instantiation
    -> workload graph
    -> match -> place -> connect -> cover
    -> complete-plan materialization / verification / static scoring
    -> selected bound MicroIR + frozen plan + diagnostics
    -> target lowering / backend verification / executable
       or machine-based performance report

Machine topology + layouts/rules + emitters -> target choices
Verified measurements                     -> optional feedback
```

The framework can also start from directly authored MicroIR. LLK matmul/SwiGLU export is one implemented input adapter. General structured/Linalg import is a future adapter, not an existing capability.

The canonical tile contract remains `Tile = (shape, dtype, layout, memory_space, owner)`. Conceptual layers are uTile (dataflow), uMap (placement, movement and schedule), and uHW (canonical fragments and resource events), all in one `micro` namespace. Logical views have no movement cost; materialization, ownership and instruction fragments expose different resource obligations.

Search IR stores domains, constraints and objectives. A concrete `micro.kernel` stores explicit computation and schedule and may initially contain abstract ownership. A final selected executable kernel must have all execution-affecting decisions bound and materialized. High-level DNN operations decompose into execution primitives; search constructs do not reach backend lowering.

## 4. Component Contracts and Extension

### 4.1 Semantic and structured input

An input adapter declares operand/result structure, shapes, dtypes, layout requirements, effects and numerical semantics, and decomposes computation into supported tile primitives. It exposes applicable transformations and finite search choices without relying on an operator-name switch in the generic planner.

The generic workload contract must accommodate elementwise, reduction, broadcast, contraction and composed dataflow as capabilities are added. It must not require every workload to have exactly M/N/K, one shared left-hand operand, one or two RHS arms, or a contraction epilogue. Contraction helpers remain useful as specialized adapters.

The [operator-kernel registry design](2026-10-11-conversion-kernel-lowering-framework-design.md) is an incremental refactor of existing LLK consumers. Its `KernelContract` and `MicroKernelFragment` are bounded by the current contraction export template. Generalizing those interfaces is separate future work; registering more operators alone does not prove a generic workload contract.

### 4.2 MicroIR primitives

Canonical primitives define tile computation, views/partitioning, memory materialization, movement, layout transformation, temporal/spatial execution and synchronization. Their structural verifiers enforce operand, result, shape, dtype, effect and dependency contracts. Numerical approximation stays explicit in semantics.

A new primitive requires a documented coverage path through extraction, legality, mapping, performance interpretation and lowering. Unsupported behavior must be identified at the applicable boundary. A workload label is diagnostic/provenance information, not a replacement for operation semantics.

### 4.3 Machine and target packages

`LLKMachine` owns a typed, validated resource topology: executors, compute, memory, transfer resources and legal links, with capabilities, capacities and costs. LLKMap declares layout constraints, matching predicates, implementation bundles and emitter keys. Target packages implement selected behavior.

Generic machine and mapping libraries must not branch on target names or interpret opaque emitter/bundle names as policy. Canonical MicroIR does not acquire per-backend owner vocabulary or concrete engine IDs in tile types. Abstract owners resolve through a profile; concrete resource selections persist in plan metadata.

Adding a target requires machine/rule conformance and explicit lowering capabilities. A profile with estimated costs establishes modeled mapping support; executing hardware requires a backend and execution evidence.

### 4.4 Mapping and plan materialization

Mapping consumes primitive/dataflow facts and produces candidates, placed instances, producer/consumer connections and coherent graph coverings. Connections include direct use, transfer routes, layout conversion or combined movement. Rules may cover a single operation or a bounded fused group.

The binder retains source associations, concrete compute/memory/transfer identity, per-hop storage and lifetimes, routes, layouts and implementation choices. Materialization turns movement, transforms and synchronization into canonical execution IR. Verification distinguishes a structurally valid kernel, a feasible complete plan and an executable selected backend.

`LLKMapping` remains independent of performance, runtime and JIT libraries. MLIR integration and mapped compilation perform materialization/shared analysis through integration contracts; they do not move runtime dependencies into the generic planner. The [September mapping specification](2026-09-18-microir-inspired-dslcompiler-enhancement-design.md) remains the detailed mapping contract.

### 4.5 Analysis and execution

Shared analysis describes work, bytes, dependencies, selected resources, lifetimes and capacity. L0 supplies static bounds; L1 adds dependency/resource scheduling. Search scores and materialized reports must agree on the same selected facts.

The [injected MicroDAG event builder](2026-10-04-micro-dag-ioc-design.md) is an implemented interpretation extension seam. The core owns traversal and graph invariants; handlers contribute events, storage and aliases. The current zero-event fallback for unregistered leaves does not prove complete semantic/cost coverage. Framework conformance must identify modeled leaves and reject or explicitly classify unsupported computational/effectful leaves before claiming complete analysis. A stricter coverage policy remains future work.

Selected-target lowering must realize the chosen implementation and reject unsupported decisions. Portable reference execution is separately identified. Runtime capability checks and ownership apply to the delivered ABI; the current typed mapped invocation is rank-two and must not be advertised as arbitrary-rank DNN execution.

## 5. Search-Based Compilation

Search has two cooperating levels:

1. **Source/schedule candidates:** supported decomposition, fusion, tiling, ordering, layout or pipeline transformations applied to the original source, preserving its numerical/effect contract.
2. **Mapping candidates:** rule bundles, placements, resource selections, connections, routes and fragments for a particular source candidate.

This is a staged search contract, not a requirement for one exhaustive joint solver. Decisions that change dataflow create a new source candidate. Binding must not substitute a synthetic GEMM for a different original workload or alter semantics through metadata alone.

Each supported choice has a finite domain, legality rule and concrete realization. Shared complete-plan materialization, live-storage/capacity checks and static analysis run before top-K retention. Execution-based ranking additionally requires successful backend verification and typed numerical validation. Rejected candidates carry reasons; missing measurement values never rank as zero.

Deterministic, beam and bounded exact strategies use the same legality and score definitions. Reports preserve search options, limits, truncation and tie-breaking. Bounded exact mode is optimal only within its completely explored declared domain. The architecture does not make every listed search dimension an implemented option.

Search dimensions derive from workload/target capabilities. `BM/BN/BK`, M-buckets and a two-arm contraction template describe the current examples. Non-contraction workloads need suitable parameter schemas; adding them is a framework capability increment.

Frozen replay preserves full source/candidate content, bindings, plan, machine/layout/rule content, compiler version, ABI, numerical contract and execution mode. Measured and predicted metrics remain distinct. Measurement persistence and calibration are separate capability work (#51/#52).

## 6. Scope and Current Gaps

The scope includes reusable MicroIR, machine/target descriptions, mapping, search, selected-plan materialization, shared analysis and backend integration. It includes supported kernels and composed subgraphs with explicit capability limits.

The design does not require an ONNX/framework model importer, global graph partitioning, a Python DSL, a cycle-accurate simulator or production execution on every modeled target. These can consume or extend the framework later. It also does not replace Linalg/Vector or encode vendor instructions in canonical MicroIR.

The architecture review identifies these limits to generic coverage:

| Area | Existing evidence | Framework work still needed |
|---|---|---|
| Semantic export | LLK matmul and fused SwiGLU | General structured import and non-contraction workload contracts |
| Source search | Original-source finite candidates and frozen replay | Transformation/parameter schemas beyond the contraction template |
| Event interpretation | Injected operation builders and built-in defaults | Explicit completeness policy for unsupported computational leaves |
| Mapping portability | AVX2 and generic accelerator policy/topology | Further primitive/target conformance and hardware emitters |
| Invocation | Checked typed rank-two buffers and scratch ownership | Wider rank/port/effect contracts when supported workloads require them |
| Performance | Shared uncalibrated static estimates and optional verified measurement | Production persistence, calibration and prediction-error validation |

These are design priorities, not claims that this documentation change closes them.

## 7. Framework Acceptance

Acceptance is organized by reusable capability rather than a fixed operator count or a single SwiGLU throughput target.

| Contract | Fixture or extension | Required evidence |
|---|---|---|
| Tile/dataflow semantics | Direct MicroIR views, partitions and primitives | Verifier coverage, dependency preservation, correct materialization costs |
| Contraction | Matmul example | Dtype/accumulation, tails, reuse, selected resource identity and numerical checks |
| Multi-branch fusion | SwiGLU example | Shared-input edges, multiple accumulators, live storage and selected fused epilogue realization |
| Non-contraction generality | Direct elementwise/reduction programs plus future semantic adapter | No required contraction template in shared workload/search contracts |
| Movement/layout | Copy/transform chains across memories | Legal multi-hop paths, dependencies, per-hop storage/lifetimes and traffic parity |
| Resource feasibility | Multiple same-kind engines and constrained memories | Concrete identity, occupancy/capacity rejection before retention |
| Search/replay | Legal, infeasible and bounded candidate domains | Stable ordering, truthful truncation, original-source preservation and frozen replay |
| Analysis | Selected plan and materialized IR | Shared cycles/traffic/capacity accounting; explicit modeling coverage |
| Target extension | AVX2 and another topology/rule package | No target-name branches in the generic planner; separate execution capability reporting |
| Semantic extension | A future workload beyond matmul/SwiGLU | Adapter/primitive support without operator-name changes in shared mapping/analysis |
| Backend execution | Selected and reference modes on supported hosts | Chosen implementation realized, no silent fallback, typed invocation and numerical checks |

Existing fixtures satisfy only the capability/revision rows supported by their evidence. The framework acceptance matrix is a design requirement for future increments, not a declaration that every row has passed today.

## 8. Capability Roadmap and Documentation Rules

Retain M1–M6 as CPU validation history and M9–M12/mapping repairs as framework foundation history. Future sequencing should follow these capability dependencies:

1. Define broader workload/parameter contracts and analysis completeness policy.
2. Extend semantic/structured import and search instantiation beyond contractions; prove it with a small non-contraction workload.
3. Extend primitive and target coverage through declarative policy, event builders and emitters; validate storage, resource identity and analysis parity.
4. Add persistent verified measurement and calibration; record prediction error by workload/target regime.
5. Expand frontend, runtime and hardware integration as required by proven workload capabilities.

Each increment needs its own bounded design and implementation plan. This document does not start those code changes. Operator additions are selected to exercise missing framework contracts, and target additions are selected to exercise different resource/topology constraints.

Project entry points and architecture documents use the generic framework goal. Specialized specifications state their subsystem or fixture scope and link here. Historical plans and acceptance records are preserved; they cannot override current goals or imply new support. Future performance reports name the source, plan, machine, backend and measured/predicted status so example-kernel results remain interpretable.
