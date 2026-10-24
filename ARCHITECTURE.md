# Generic MicroIR DNN Compiler Framework — Architecture

## 1. Project Goal

**Design a generic MicroIR-based, mapping- and search-based DNN compiler framework.**

The framework represents DNN computations as tile execution programs, maps them onto explicit machine resources, searches legal implementation choices, and lowers a selected plan for execution or performance analysis. Reusable compilation machinery is the project's central deliverable.

**Matmul and fused SwiGLU are example operator kernels for testing the framework.** Matmul exercises contraction, tiling, placement and movement; SwiGLU exercises shared inputs, multiple contractions and fused elementwise dataflow. Neither operator defines the compiler's scope. Intel AVX2 is the first executable validation backend. The generic accelerator package tests mapping onto a different topology, and the Arm SVE/SME packages supply static mapping and analysis conformance at modeled VL=SVL=512, plus an executable Arm SVE f32 add/multiply subset with an emulated (not native) acceptance path; SME, ZA, other signatures and timing remain non-executable, and none of those machine profiles supplies hardware execution.

The [framework goal and design specification](docs/superpowers/specs/2026-10-12-generic-microir-dnn-compiler-framework-design.md) defines project-wide requirements and acceptance criteria. The [developer manual](docs/manual/source/architecture.rst) describes current components, and the [feature matrix](docs/manual/source/features.rst) states delivered support. The [historical CPU architecture](docs/design/legacy-cpu-architecture.md) preserves M1–M7 design details.

## 2. Compilation Contract

```text
DNN semantics / structured tensor programs
    -> semantic decomposition and legal transformations
    -> MicroIR tile dataflow and schedule choices
    -> source candidate instantiation
    -> target mapping: match / place / connect / cover
    -> materialize, verify and score complete plans
    -> selected bound MicroIR + replayable plan
    -> target lowering and execution
       or static performance analysis

MachineModel + LLKMap + target emitters -> mapping and lowering
Shared resource/event analysis         -> search ranking and reports
Optional verified measurements         -> ranking feedback and calibration
```

This is the generic architecture contract. Current semantic export supports LLK matmul and fused SwiGLU; a general structured/Linalg importer remains future work. Directly authored MicroIR can exercise the framework independently of those export roots. The existing LLK/Linalg CPU pipeline remains a validation and comparison path.

```text
Tile = (shape, dtype, layout, memory_space, owner)
MicroIR = Tile Dataflow + Tile Mapping + Tile Schedule
Predicted performance = F(MicroIR, MachineModel, calibration)
```

One `micro` namespace has three conceptual layers: **uTile** describes tile dataflow, **uMap** describes placement, ownership, movement and scheduling, and **uHW** describes canonical instruction fragments and resource events.

`micro.search_space` describes choices, constraints and objectives. A concrete `micro.kernel` has explicit computation and schedule, but may still need target mapping. A selected executable kernel must have all execution-affecting choices bound, materialized and verified. Concrete kernels contain execution primitives rather than high-level DNN semantic operations.

## 3. Component Responsibilities

| Component | Responsibility | Extension boundary |
|---|---|---|
| Semantic/structured input | Computation, shapes, dtypes, effects and numerical contracts | Frontend adapters and operator decomposition; `llk` is one adapter |
| Micro dialect | Tile dataflow, execution structure, search intent and structural verification | Canonical primitives; no backend vocabulary |
| `LLKMachine` | Typed resource topology, capabilities, capacities and costs | Versioned `machines/*.yaml` profiles |
| `LLKMapping` | Workload extraction, matching, placement, connections, covering and plan verification | Declarative layouts/rules and target interfaces |
| `LLKMicroMapping` | MLIR integration and canonical plan materialization | Source-to-plan associations and movement/sync construction |
| `LLKPerf` | Candidate analysis, event extraction, bounds, resource scheduling and tuning | Injected event builders and cost/measurement providers |
| `LLKTargetMapping` | Target factories, mapping bundles and emitters | `mapping/<target>/` and `lib/Target/<Target>/Mapping/` |
| `LLKMappedCompilation` | Bind, verify, lower and compile selected plans | Explicit selected-target or reference execution mode |
| Runtime | Invocation, buffer ownership, scratch lifetime, JIT and execution | Backend ABI/capability validation |

`LLKMapping` depends on machine facts, not on `LLKPerf`, JIT or runtime. `LLKPerf` consumes shared mapping cost/event primitives. Compilation integrates mapping, analysis and execution without a reverse dependency from the generic planner to the runtime.

## 4. Mapping and Search

Mapping follows **extract → match → place → connect → cover → bind → verify**. It extracts a workload graph, matches single operations or bounded fused subgraphs, places candidates with legal layouts, synthesizes connections, and selects a coherent covering. Complete-plan materialization and shared analysis check live storage, capacity and dependencies before top-K retention. Executable compilation additionally checks backend realizability.

Schedule search and mapping search cooperate. A source candidate describes transformations and parameter choices; mapping search explores implementations, placements and connections for that candidate. Decisions that alter dataflow require a new source candidate. They cannot be introduced by rewriting only a report or workload label.

Search is finite and reproducible. Deterministic, beam and bounded exact strategies share legality and cost contracts; reports record bounds and truncation. Exact mode describes its explored finite domain, not global optimality. Search dimensions come from supported workload transformations and target capabilities. `BM/BN/BK`, two contraction arms and five M-buckets are contraction-example choices, not mandatory framework fields.

## 5. Generic Extension Rules

- New workloads contribute semantic contracts and decomposition into supported MicroIR primitives. Shared mapping and analysis operate on primitives and graph facts rather than `swiglu` or `matmul` workload names.
- New primitive behavior contributes verification, mapping/lowering support and event interpretation. A registry alone does not establish support for arbitrary operations.
- New targets contribute machine profiles, layouts, mapping rules and emitters. Generic `lib/Machine` and `lib/Mapping` never branch on target names.
- Abstract owners belong to tile semantics. Concrete resource IDs belong to selected plan metadata and are validated against the machine.
- Artifact identity preserves source content, shapes/dtypes/layouts, numerical contracts, bindings, plan, machine/policy content, compiler version, ABI and execution mode.
- Unsupported semantic, mapping, modeling or execution capabilities need explicit diagnostics and evidence limits. Reference and selected-target execution are separately reported.

The [operator registration proposal](docs/superpowers/specs/2026-10-11-conversion-kernel-lowering-framework-design.md) improves semantic extensibility, but its M/N/K contract and shared contraction driver are a bounded adapter, not the final generic DNN workload interface. The merged [MicroDAG builder design](docs/superpowers/specs/2026-10-04-micro-dag-ioc-design.md) provides injectable event interpretation while core traversal retains graph invariants.

## 6. Validation and Roadmap

Framework acceptance is organized by capability: tile/dataflow semantics, non-contraction computation, multi-branch fusion, movement/layout conversion, resource topology and capacity, reproducible search/replay, analysis parity, and target extension. Matmul/SwiGLU numerical results establish exercised contracts on their fixtures; framework success also requires reusable extension boundaries. See the [specification's validation matrix](docs/superpowers/specs/2026-10-12-generic-microir-dnn-compiler-framework-design.md#7-framework-acceptance).

M1–M6 delivered CPU validation infrastructure; M7 frontend work is deferred. M9–M12 and the mapping repair series established MicroIR, topology-aware mapping, selected plan materialization, shared static analysis and original-source tuning. Their chronological documents remain implementation history.

Future work is prioritized by framework capability: general workload/structured import, search spaces beyond the contraction template, additional primitive/target conformance, measured feedback and calibration, and broader execution APIs where workloads require them. Each increment names its supported domain and tests extension without operator/target dispatch in generic components.

[PR #133 acceptance](docs/reviews/issue67-final-acceptance.md) remains revision-pinned resource/execution/tuning evidence. Selected execution is verified on compatible x86_64 AVX2 fixtures; generic accelerator coverage is static. Production persistence/calibration (#51/#52), general frontend coverage and direct MicroIR emulation (#54) remain separate work. The goal revision changes documentation and architectural direction; it does not add operator or backend support.
