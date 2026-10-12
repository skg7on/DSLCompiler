# Canonical MicroIR Framework Architecture

**Parent:** [ARCHITECTURE.md](../../ARCHITECTURE.md)

**Status:** Architecture direction aligned on 2026-10-12; delivered coverage remains revision-specific.

**Scope:** Generic MicroIR-based, mapping- and search-based DNN compiler framework.

**Governing spec:** [Framework goal and design](../superpowers/specs/2026-10-12-generic-microir-dnn-compiler-framework-design.md)

## 1. Purpose

The canonical `micro` dialect is the shared execution artifact for DNN compilation, target mapping, search and machine-based analysis. The project develops that reusable framework. Matmul and fused SwiGLU are example operator kernels for validation; AVX2 is the first executable validation backend.

The original M9+ proposal introduced MicroIR alongside the CPU pipeline. The architecture now treats MicroIR and mapping/search as the framework center. The [August redesign](../superpowers/specs/2026-08-11-canonical-micro-ir-redesign.md) preserves the original rollout rationale; its initial implementation sequence does not define the current project scope.

## 2. Pipeline and Artifacts

```text
Semantic/structured adapter or direct MicroIR
    -> tile dataflow + declared schedule/search choices
    -> original-source candidate instantiation
    -> workload extraction / rule matching / placement
    -> connections / complete covering search
    -> materialization / verification / shared static scoring
    -> selected bound MicroIR + frozen plan
    -> target lowering / execution or performance analysis

MachineModel + LLKMap + target emitters -> mapping and lowering
Optional verified measurement           -> feedback and calibration
```

Current semantic export supports LLK matmul and SwiGLU. A general structured/Linalg importer is future work. The established LLK/Linalg CPU path remains a validation/comparison route; Linalg is not a mandatory intermediate for every future MicroIR input.

Search IR describes domains, constraints and objectives. Concrete MicroIR describes computation and schedule and may still require target placement. Selected executable MicroIR has all execution-affecting choices bound, materialized and verified. Source transformations that change dataflow require new source candidates rather than metadata-only substitutions.

## 3. Layer Responsibilities

| Layer | Contract |
|---|---|
| Semantic/structured input | Mathematical computation, ports, shapes/dtypes, effects, numerical semantics and legal decomposition |
| uTile | Canonical tile dataflow and computation; logical views versus materialized values |
| uMap | Memory/owner placement, movement, layout transformation, temporal/spatial schedule, pipeline and synchronization |
| uHW | Canonical instruction fragments and machine-visible resource events |
| Machine/target package | Topology, capabilities, capacities, costs, declarative layouts/rules and selected emitters |
| Mapping/search | Candidates, placements, connections and coherent complete plans; feasibility and reproducible selection |
| Analysis/execution | Shared resource/event accounting, explicit model limits, backend realization and invocation |

uTile/uMap/uHW share one `micro` namespace. The primary value remains `Tile = (shape, dtype, layout, memory_space, owner)`. Concrete kernels contain execution primitives, not DNN semantic operations.

## 4. Extension and Acceptance

Workloads extend through semantic adapters and supported primitives; targets extend through machine profiles, layouts/rules and emitters. Shared mapping and analysis consume operation/dataflow facts rather than workload-name switches. Generic machine/mapping code never branches on target names.

M/N/K, `BM/BN/BK`, two contraction arms, M-buckets and typed rank-two invocation describe current adapter/runtime boundaries. Future non-contraction workloads need appropriate port, iteration/dataflow and search contracts. They must not be forced into a SwiGLU or matmul template.

Framework acceptance covers tile semantics, non-contraction computation, fusion, movement/layout, resource topology/capacity, reproducible search/replay, analysis parity and extension. Example-kernel numerical/performance results are evidence for exercised contracts, not full DNN coverage. See the governing spec for the capability matrix and current genericity gaps.

## 5. Detailed References and Status

- [Core MicroIR semantics](m9-micro-ir-core-concepts.md) and [tile programming model](../superpowers/specs/2026-08-13-micro-ir-tile-programming-model-spec.md).
- [Generic mapping/search contract](../superpowers/specs/2026-09-18-microir-inspired-dslcompiler-enhancement-design.md), [workflow](micro-ir-mapping-workflow.md) and [LLKMap syntax](llkmap-syntax.md).
- [Current component map](../manual/source/architecture.rst), [feature matrix](../manual/source/features.rst) and [PR #133 acceptance](../reviews/issue67-final-acceptance.md).
- [Historical CPU validation architecture](legacy-cpu-architecture.md).

M1–M6 are CPU validation history; M9–M12 and mapping repairs established framework foundations. Future increments prioritize broader workload/import and source-search contracts, modeling completeness, primitive/target conformance and measured feedback/calibration. Generic accelerator coverage is static; selected execution remains limited to delivered backends and tested hosts.
