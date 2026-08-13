# M9+ Canonical Micro-IR Architecture

**Parent:** [ARCHITECTURE.md](../../ARCHITECTURE.md)
**Status:** Draft Design
**Scope:** Project-level redesign for a tile-centric canonical DNN micro-IR, machine models, performance evaluation, and auto-scheduling.

---

## 1. Motivation

The existing LLK compiler is a focused out-of-tree MLIR compiler for CPU LLM kernels. It keeps DNN semantics in a small `llk` dialect, lowers to Linalg/Vector/LLVM, and validates the first backends on AVX2. That design is useful for a compiler pipeline, but it does not yet provide a canonical low-level execution representation suitable for:

- Comparing schedules across hardware targets.
- Evaluating model performance on candidate AI accelerators.
- Searching tile, memory, mapping, and pipeline choices in an Ansor/TVM-style auto-scheduler.
- Representing multi-level tile execution close to AI hardware resources.
- Sharing one artifact between compiler backends, performance models, and architecture exploration.

The new architecture adds a canonical `micro` dialect below LLK and above target-specific lowering. `micro` is not a new DNN semantic dialect. It represents how a tensor program executes on an abstract AI accelerator: tile dataflow, compute engines, memory hierarchy, data movement, spatial mapping, temporal schedule, synchronization, and tunable schedule choices.

---

## 2. High-Level Pipeline

The redesigned project has two related paths:

```text
Frontend / semantic path:

  Python DSL / Triton-like IR / LLK ops
        |
        v
  LLK semantic dialect
        |
        v
  Linalg + tensor + arith + math
        |
        v
  schedule selection and fusion
        |
        +-----------------------------+
        |                             |
        v                             v
  existing CPU lowering          LLKToMicro
  Vector/SCF/MemRef/LLVM             |
        |                            v
        v                       micro search space
  ORC JIT / runtime                  |
                                     v
                              candidate binding
                                     |
                                     v
                              concrete micro IR
                                  /       \
                                 /         \
                                v           v
                         performance     future target
                         evaluator       lowerings
```

The first implementation keeps the existing AVX2/JIT path working. `micro` is introduced as an export, analysis, and auto-search layer. Once it is stable, target lowerings can move to consume concrete `micro.kernel` directly.

Concrete Micro-IR is organized conceptually as:

```text
uTile: algorithmic tile dataflow
    -> uMap: memory placement, owner mapping, pipeline, sync
    -> uHW: instruction fragments and machine-visible events
```

The MVP may implement these as one `micro` dialect rather than three dialects, but the separation is part of the architecture contract.

---

## 3. Layer Responsibilities

### 3.1 Semantic Layer

The semantic layer answers: what operation is being computed?

It includes:

- `llk.fused_swiglu`
- `llk.rope`
- `llk.attention`
- `llk.matmul`
- Python/Triton frontend scaffolding
- Pattern recognition and high-level fusion

This layer should continue to avoid hardware details such as SRAM sizes, DMA queues, PE mapping, or matrix-engine tile shapes.

### 3.2 Structured Optimization Layer

The structured layer answers: what legal loop/tensor transformations expose useful locality and fusion?

It includes:

- LLK to Linalg lowering
- canonicalization
- shape specialization
- schedule selection
- fusion
- packing annotations

This layer can still use Linalg, Tensor, SCF, and Vector to preserve MLIR-native optimization opportunities.

### 3.3 Canonical Execution Layer

The canonical execution layer answers: how does this scheduled tensor program execute on a machine with finite compute, memory, movement, and synchronization resources?

The primary value is a tile:

```text
Tile = (shape, dtype, layout, memory_space, owner)
```

Tiles are the bridge between MLIR tensor semantics and AI hardware. Concrete Micro-IR must distinguish logical tiles, materialized memory tiles, execution tiles assigned to resources, and instruction fragments that match hardware compute granularity.

It is represented by the new `micro` dialect:

- `micro.kernel`
- `micro.for`
- `micro.spatial_for`
- `micro.pipeline`
- `micro.tile_view`
- `micro.tile_partition`
- `micro.alloc`
- `micro.async_copy`
- `micro.wait`
- `micro.mma`
- `micro.vector`
- `micro.reduce`
- `micro.store`
- communication and synchronization ops

No DNN semantic ops such as `attention`, `swiglu`, `rope`, or `conv2d` appear in concrete `micro.kernel`. They must lower into compute, movement, mapping, and sync operations.

### 3.4 Machine Model Layer

The machine model answers: what resources are available, and what are their costs?

Machine profiles are authored as YAML under `machines/` and validated into typed C++ structs. The performance evaluator consumes typed models, not ad hoc YAML maps.

Initial profiles:

- `machines/generic-ai-accel-v1.yaml`
- `machines/x86-avx2-cpu.yaml`

Later profiles can model GPU, NPU, systolic-array, and custom ASIC targets.

### 3.5 Search and Tuning Layer

The search layer answers: which legal schedule candidate should be selected for a workload, shape, and machine?

The search layer uses:

- `micro.search_space`
- `micro.param`
- `micro.constraint`
- `micro.objective`
- `micro.candidate`
- candidate legality verification
- L0/L1 performance estimates
- optional measured feedback on the current AVX2 runtime

This makes `micro` both a performance artifact and a schedule-search substrate.

---

## 4. Core Contract

The project-level contract becomes:

```text
Performance = F(workload semantics, micro schedule, machine model, calibration)
```

In implementation terms:

```text
LLK/Linalg workload
    -> micro.search_space
    -> micro.candidate
    -> concrete micro.kernel
    -> micro-perf + MachineModel
    -> selected schedule + metrics
```

A concrete `micro.kernel` must be fully bound and deterministic. Search constructs must not leak into the final lowering path.

### 4.1 Tile-Centric Contract

The canonical artifact is not only a loop nest with ops. It is a tile execution plan:

```text
DNN Micro-IR = Tile Dataflow + Tile Mapping + Tile Schedule
```

The tile contract is:

- `uTile` represents algorithmic tile dataflow: partition, view, matmul, elementwise, reduce, broadcast, and fusion.
- `uMap` binds tiles to memory spaces, owner resources, copy paths, pipelines, and synchronization.
- `uHW` resolves mapped tiles into instruction fragments such as MMA tiles, vector tiles, DMA transactions, barriers, and communication events.
- Logical tile views have no direct movement cost.
- Memory tiles consume capacity and generate memory/copy events.
- Execution tiles drive occupancy, parallelism, and owner-scoped synchronization.
- Instruction fragments drive compute utilization and target-lowering legality.

This makes the performance contract:

```text
Performance = F(Tile Dataflow, Tile Mapping, Tile Schedule, MachineModel, Calibration)
```

The detailed tile programming model is specified in [2026-08-13-micro-ir-tile-programming-model-spec.md](../superpowers/specs/2026-08-13-micro-ir-tile-programming-model-spec.md).

---

## 5. Repository Layout

New directories:

```text
include/LLK/Dialect/Micro/
  MicroDialect.h
  MicroDialect.td
  MicroOps.td
  MicroTypes.td

lib/Dialect/Micro/
  MicroDialect.cpp
  MicroOps.cpp

include/LLK/Conversion/LLKToMicro/
  LLKToMicro.h

lib/Conversion/LLKToMicro/
  LLKToMicro.cpp

include/LLK/Perf/
  MachineModel.h
  MachineModelLoader.h
  MicroCostModel.h
  MicroDAG.h
  SearchSpace.h
  Candidate.h

lib/Perf/
  MachineModel.cpp
  MachineModelLoader.cpp
  MicroCostModel.cpp
  MicroDAG.cpp
  SearchSpace.cpp
  Candidate.cpp

machines/
  generic-ai-accel-v1.yaml
  x86-avx2-cpu.yaml

tools/micro-opt/
  micro-opt.cpp

tools/micro-perf/
  micro-perf.cpp
```

Existing directories remain valid. In particular, current `llk-opt`, `llk-compile`, `llk-bench`, and `llk-tune` continue to exist.

---

## 6. Pipeline Integration

### 6.1 Initial Integration

Initial integration is non-disruptive:

```text
LLK -> Linalg -> schedule/fuse
        |
        +-> LLKToMicro -> micro-opt -> micro-perf
        |
        +-> existing vector/bufferize/LLVM/JIT path
```

This supports performance analysis without forcing the working CPU compiler to immediately depend on the new dialect.

### 6.2 Stable Integration

After the `micro` verifier, cost model, and lowering tests mature:

```text
LLK -> Linalg -> LLKToMicro -> MicroToVector/LLVM -> ORC JIT
                         |
                         +-> micro-perf
```

At that point, `micro` becomes the primary post-schedule contract for both lowering and evaluation.

---

## 7. Performance Evaluation Strategy

### 7.1 L0 Static Bound

The L0 evaluator computes:

- total MACs/FLOPs
- bytes moved by memory level
- live materialized tile bytes
- compute lower bound
- memory lower bound
- capacity requirements
- simple roofline-style latency bound

It should be fast enough to evaluate large search spaces.

### 7.2 L1 Resource DAG Scheduler

The L1 evaluator lowers tile-producing and tile-consuming `micro` ops into a resource-constrained event DAG:

- DMA events for `micro.tile_async_copy` or `micro.async_copy`
- memory events for `micro.tile_load`, `micro.tile_store`, `micro.load`, and `micro.store`
- matrix-engine events for `micro.tile_mma` or `micro.mma`
- vector-engine events for `micro.tile_vector`, `micro.vector`, `micro.tile_reduce`, and `micro.reduce`
- dependency events for `micro.wait`
- synchronization costs for barriers and fences

It reports:

- predicted cycles and time
- matrix/vector/DMA utilization
- memory bandwidth utilization
- critical path
- bottleneck reason
- copy/compute overlap efficiency
- capacity violations

### 7.3 L2 Calibrated Event Simulation

L2 is intentionally later. It can add:

- bank conflicts
- NoC contention
- DMA queue depth
- pipeline bubbles
- instruction issue effects
- hardware counter calibration

The project should not start with cycle-accurate simulation.

---

## 8. Auto-Search Architecture

The search layer works with two forms of `micro`:

```text
parametric/search micro IR:
  describes legal choices

bound/execution micro IR:
  describes one concrete candidate
```

The tuner flow is:

```text
MicroSearchSpace
    |
candidate generator
    |
candidate binding
    |
legality verifier
    |
L0/L1 performance model
    |
top-K candidates
    |
optional AVX2 JIT measurement
    |
schedule YAML + calibration data
```

Search dimensions include tile sizes at each hierarchy level, loop order, spatial mapping, memory placement, owner mapping, pipeline depth, layout, compute binding, tensorization fragment shape, vector width, reduction strategy, communication strategy, parallel granularity, and fusion boundaries.

---

## 9. Milestone Sequence

### M9: Canonical Micro-IR Foundation

Add the concrete execution dialect, tile type/attributes, tile operation MVP, and tests.

Exit criterion: `llk-opt` can round-trip representative tile-centric `micro.kernel` IR containing logical tile views, materialized memory tiles, owner mapping, pipeline, async copies, waits, and MMA/vector fragments.

### M10: MachineModel and Performance L0/L1

Add YAML machine profiles, schema validation, and `micro-perf`.

Exit criterion: one concrete tile-centric `micro.kernel` can be evaluated on both `generic-ai-accel-v1.yaml` and `x86-avx2-cpu.yaml`.

### M11: LLKToMicro Export Path

Convert scheduled LLK/Linalg patterns into concrete `micro`.

Exit criterion: current SwiGLU schedules can be materialized as tile-centric `micro` and evaluated without breaking the existing AVX2/JIT path.

### M12: Search-Space Micro-IR and Auto Scheduler

Add parametric search constructs and integrate candidate generation with `llk-tune`.

Exit criterion: the tuner searches `BM`, `BN`, `BK`, multi-level tile shapes, pipeline stages, layout, owner mapping, spatial mapping, tensorization fragments, and vector width, then records the chosen candidate and predicted/measured metrics.

### M13: Feedback and Calibration

Store selected schedules as YAML with prediction and measurement metadata.

Exit criterion: predicted vs measured error is tracked across regression tests and can be used to calibrate `x86-avx2-cpu.yaml`.

---

## 10. Non-Goals

- Do not replace StableHLO, TOSA, Linalg, or Vector.
- Do not encode vendor ISA instructions directly in canonical `micro`.
- Do not put DNN semantic ops in concrete `micro.kernel`.
- Do not make `MachineModel` a loose unvalidated config blob.
- Do not build an L2 event simulator before L0/L1 produce useful results.
- Do not force the existing AVX2 compiler path to depend on `micro` in the first milestone.

---

## 11. Design Summary

The redesigned project keeps LLK as the semantic DNN layer and adds `micro` as the tile-centric canonical execution layer. `micro` is low enough to model AI hardware behavior and high enough to stay portable across matrix engines, vector engines, SRAM hierarchies, DMA engines, NoC-like communication, and spatial mappings.

The key shift is that tile dataflow, tile mapping, and tile schedule become explicit executable artifacts and searchable objects, not only JSON fields consumed by transformation passes.
