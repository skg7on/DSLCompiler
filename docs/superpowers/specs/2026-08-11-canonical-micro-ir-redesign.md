# Canonical Micro-IR Redesign Spec

**Status:** Draft
**Date:** 2026-08-11
**Related Design Docs:**
- [M9+ Canonical Micro-IR Architecture](../../design/m9-canonical-micro-ir-architecture.md)
- [Micro-IR Core Concepts](../../design/m9-micro-ir-core-concepts.md)

---

## 1. Goal

Redesign the project around a canonical DNN micro-IR that is lower level than LLK/Linalg, closer to AI hardware execution, and useful for both performance evaluation and auto-search/auto-optimization.

The redesign target is dual-track:

1. Define a generic AI accelerator execution IR and machine model.
2. Validate the MVP on the current AVX2 CPU compiler/runtime path.

The existing LLK semantic dialect and CPU pipeline remain intact while the new layer matures.

---

## 2. Problem Statement

The current project has a strong semantic and lowering pipeline:

```text
LLK dialect -> Linalg -> tiled/fused IR -> Vector -> MemRef/SCF -> LLVM -> ORC JIT
```

It can represent DNN-specific operations such as SwiGLU, RoPE, and Attention, and it can lower them toward CPU execution. However, it lacks a canonical representation of executed tensor programs that explicitly models:

- matrix/tensor engine work
- vector work
- memory hierarchy
- async data movement
- pipeline overlap
- spatial mapping
- synchronization
- communication
- resource occupancy
- auto-scheduling search spaces

As a result, the current schedule database can select tile parameters, but the selected schedule is not materialized as a single canonical execution artifact for performance modeling, hardware comparison, or tuning.

---

## 3. Design Principles

### 3.1 Separate Semantics, Execution, and Machine Cost

```text
DNN semantics:        LLK / Linalg
Execution schedule:   micro
Hardware costs:       MachineModel YAML
Performance:          micro + MachineModel + calibration
```

LLK answers what is computed. `micro` answers how it executes. `MachineModel` answers how expensive it is on a target.

### 3.2 Do Not Build Another StableHLO

`micro` is intentionally below model semantics. Concrete `micro.kernel` must not contain `attention`, `swiglu`, `rope`, or `conv2d`. Those operations lower into `mma`, `vector`, `reduce`, memory movement, mapping, pipeline, and sync.

### 3.3 Be Architecture-Parametric, Not Hardware-Blind

`micro` should avoid vendor ISA names such as a specific NVIDIA WGMMA instruction or a custom NPU opcode. It should still expose hardware-visible execution characteristics such as MMA shape, memory space, dtype, pipeline depth, and mapping target.

### 3.4 Support Search and Concrete Execution

`micro` must represent both:

- parametric search spaces for auto-scheduling
- concrete execution candidates for performance evaluation and lowering

Search constructs are not backend IR. Candidate binding produces deterministic concrete `micro.kernel`.

### 3.5 Preserve the Working Compiler

The first milestone must not break the current AVX2 path. `micro` starts as export and evaluation infrastructure, then later becomes a lowering contract.

---

## 4. Proposed Architecture

```text
Python DSL / Triton-like IR / LLK ops
        |
        v
LLK semantic dialect
        |
        v
Linalg + tensor + arith + math
        |
        v
schedule selection + fusion
        |
        +-----------------------------+
        |                             |
        v                             v
existing AVX2 lowering           LLKToMicro
Vector/SCF/MemRef/LLVM               |
        |                            v
        v                       micro.search_space
ORC JIT / runtime                    |
                                     v
                              auto-scheduler
                                     |
                                     v
                              micro.candidate
                                     |
                                     v
                              concrete micro.kernel
                                  /       \
                                 /         \
                                v           v
                         micro-perf      future target
                         evaluator       lowerings
```

The architecture supports two use cases immediately:

1. Compile and run kernels using the existing CPU path.
2. Export and evaluate concrete/searchable micro-IR for hardware and schedule exploration.

---

## 5. Micro Dialect Scope

### 5.1 Concrete Execution Ops

Structure:

```text
micro.kernel
micro.for
micro.spatial_for
micro.pipeline
micro.stage
```

Memory:

```text
micro.alloc
micro.view
micro.load
micro.store
micro.copy
micro.async_copy
micro.prefetch
```

Compute:

```text
micro.mma
micro.vector
micro.reduce
micro.permute
micro.convert
micro.special
```

Synchronization:

```text
micro.wait
micro.barrier
micro.fence
```

Communication:

```text
micro.broadcast
micro.gather
micro.scatter
micro.exchange
micro.collective
```

MVP implementation should focus on:

```text
micro.kernel
micro.for
micro.spatial_for
micro.pipeline
micro.alloc
micro.async_copy
micro.wait
micro.mma
micro.vector
micro.reduce
micro.store
```

### 5.2 Search Ops

Search-space ops:

```text
micro.search_space
micro.param
micro.constraint
micro.objective
micro.candidate
```

These represent legal schedule choices, constraints, ranking objectives, and selected bindings.

### 5.3 Attributes

Memory spaces:

```text
dram
l2
sram
rf
acc
scratch
```

Mapping targets:

```text
cluster.x
cluster.y
core.x
core.y
pe.x
pe.y
lane
dma
matrix_engine
vector_engine
worker
```

DTypes:

```text
f32
f16
bf16
i32
i8
fp8.e4m3
fp8.e5m2
int4
fp4
block_float
sparse_2_4
```

MVP dtypes are `f32`, `f16`, `bf16`, `i32`, and `i8`.

---

## 6. Search-Space Design

### 6.1 Search Form

Example:

```mlir
micro.search_space @swiglu_space
    attributes { workload = "fused_swiglu" } {
  %BM = micro.param "BM" : i64 {choices = [8, 16, 32, 64]}
  %BN = micro.param "BN" : i64 {choices = [32, 64, 128]}
  %BK = micro.param "BK" : i64 {choices = [32, 64, 128]}
  %stages = micro.param "num_stages" : i64 {choices = [1, 2, 3]}
  %spatial_m = micro.param "spatial_m" : i64 {choices = [1, 2, 4, 8]}
  %spatial_n = micro.param "spatial_n" : i64 {choices = [1, 2, 4, 8]}

  micro.constraint {
    sram_bytes(%BM, %BN, %BK) <= machine.memory.sram.capacity_bytes
    compatible_mma_shape(%BM, %BN, %BK, bf16, f32)
  }

  micro.objective minimize latency
      secondary [maximize matrix_utilization, minimize dram_bytes]
}
```

### 6.2 Search Dimensions

The first search space should cover:

- `BM`, `BN`, `BK`
- pipeline stages
- spatial mapping over M/N
- vector width
- parallel grain
- memory placement
- packed layout selection

Later search dimensions:

- loop order
- split-K and reduction strategy
- prefetch distance
- swizzled layouts
- epilogue fusion
- online softmax block size
- matrix-engine tile selection

### 6.3 Candidate Binding

Example:

```mlir
micro.candidate @candidate_17
    binds {
      BM = 32,
      BN = 64,
      BK = 64,
      num_stages = 2,
      spatial_m = 4,
      spatial_n = 2,
      vector_width = 8
    }
```

Binding a candidate must:

- substitute all parameters
- verify all constraints
- emit concrete `micro.kernel`
- attach candidate metadata for perf reporting

---

## 7. Concrete Execution Design

Example concrete SwiGLU tile:

```mlir
micro.kernel @swiglu_candidate_17
    attributes {
      workload = "fused_swiglu",
      candidate = "candidate_17",
      target = "x86-avx2-cpu"
    } {
  micro.spatial_for %bm = 0 to %M step 32 map = #micro.map<cluster.y> {
    micro.spatial_for %bn = 0 to %N step 64 map = #micro.map<cluster.x> {
      %acc_g = micro.alloc tensor<32x64xf32> memory = #micro.memory<acc>
      %acc_u = micro.alloc tensor<32x64xf32> memory = #micro.memory<acc>

      micro.pipeline stages = 2 {
        micro.for %bk = 0 to %K step 64 {
          %x = micro.async_copy %X[%bm, %bk]
              : #micro.memory<dram> -> #micro.memory<sram>
          %wg = micro.async_copy %WG[%bk, %bn]
              : #micro.memory<dram> -> #micro.memory<sram>
          %wu = micro.async_copy %WU[%bk, %bn]
              : #micro.memory<dram> -> #micro.memory<sram>

          micro.wait %x, %wg, %wu

          micro.mma %x, %wg, %acc_g
              {shape = [16, 16, 32], input = bf16, accumulator = f32}
          micro.mma %x, %wu, %acc_u
              {shape = [16, 16, 32], input = bf16, accumulator = f32}
        }
      }

      %gate = micro.vector "silu" %acc_g {math_mode = "bounded_fast"}
      %out = micro.vector "mul" %gate, %acc_u
      micro.store %out, %Y[%bm, %bn]
          : #micro.memory<acc> -> #micro.memory<dram>
    }
  }
}
```

This IR makes the following directly observable:

- matrix work
- DRAM traffic
- SRAM traffic
- accumulator footprint
- pipeline depth
- copy/compute overlap
- spatial parallelism
- synchronization dependencies
- epilogue fusion

---

## 8. MachineModel YAML

Machine models are authored in YAML and validated strictly into typed C++ structs.

Initial generic accelerator:

```yaml
name: generic-ai-accel-v1
clock_hz: 1000000000

compute:
  matrix_engines:
    - name: mxu
      count: 8
      tile_shapes:
        - [16, 16, 32]
      input_dtypes: [bf16, f16, int8]
      accumulator_dtypes: [f32, i32]
      issue_cycles: 1
      latency_cycles: 8

  vector_engines:
    - name: vec
      count: 8
      lanes: 64

memory:
  dram:
    capacity_bytes: 68719476736
    bandwidth_bytes_per_cycle: 1024
    latency_cycles: 300

  sram:
    capacity_bytes: 1048576
    bandwidth_bytes_per_cycle: 4096
    latency_cycles: 20
    banks: 32

  acc:
    capacity_bytes: 262144
    bandwidth_bytes_per_cycle: 8192
    latency_cycles: 1

dma:
  engines: 4
  max_outstanding: 16
  setup_cycles: 16

sync:
  barrier_cycles: 64
  wait_cycles: 4
```

Validation requirements:

- required top-level sections must exist
- unknown keys are errors unless versioned
- capacities, bandwidths, and cycle counts must be positive
- dtype strings must map to known micro dtypes
- matrix tile shapes must have exactly three dimensions
- memory spaces referenced by `micro` must exist in the model

---

## 9. Performance Evaluator

### 9.1 L0 Static Bound

`micro-perf --level=0` computes:

- operation counts
- memory bytes by source/destination level
- compute lower bound
- memory lower bound
- capacity requirements
- roofline-style latency bound

This level is used for fast candidate pruning.

### 9.2 L1 Resource DAG Scheduler

`micro-perf --level=1` lowers concrete micro IR into a resource DAG.

Events:

```text
micro.async_copy -> DMA event
micro.mma -> matrix-engine event
micro.vector -> vector-engine event
micro.reduce -> vector-engine/reduction event
micro.wait -> dependency edge
micro.barrier -> sync event
micro.load/store -> memory event
```

Reports:

- predicted cycles
- predicted time
- matrix utilization
- vector utilization
- DMA utilization
- DRAM bandwidth utilization
- SRAM bandwidth utilization
- critical path
- bottleneck reason
- overlap efficiency
- capacity violations

### 9.3 L2 Calibrated Event Simulation

L2 is later work. It can model bank conflicts, NoC contention, DMA queues, pipeline bubbles, instruction issue, and hardware counter calibration. It must not block M9-M12.

---

## 10. Schedule Persistence

The current `schedules/schedule_db.json` remains valid during migration.

Longer term, selected schedules should be written as YAML:

```yaml
workload: fused_swiglu
target: x86-avx2-cpu
shape:
  M_bucket: 4
  N: 4096
  K: 4096
dtype: bf16

selected_candidate:
  BM: 32
  BN: 64
  BK: 64
  num_stages: 2
  spatial_m: 4
  spatial_n: 2
  vector_width: 8

metrics:
  predicted_cycles: 123456
  measured_ns: 98765
  matrix_utilization: 0.82
  dram_bandwidth_utilization: 0.61
```

Schedule YAML is a result of tuning. `micro.search_space` is the source of legal choices.

---

## 11. Tooling

### 11.1 `micro-opt`

MLIR optimizer driver for `micro` parse/print and micro-specific passes.

Initial responsibilities:

- register the `micro` dialect
- run parse/print tests
- run verifier-only tests
- run canonical micro passes later

### 11.2 `micro-perf`

Performance evaluator for concrete micro IR.

Example:

```bash
micro-perf --machine=machines/generic-ai-accel-v1.yaml --level=1 kernel.micro.mlir
```

Output should be deterministic and testable with FileCheck or GTest.

### 11.3 `llk-compile --emit=micro`

Add an export mode that runs the current pipeline through schedule/fusion and emits `micro` instead of continuing to LLVM.

### 11.4 `llk-tune`

Evolve `llk-tune` to generate candidates from `micro.search_space`, evaluate them with `micro-perf`, optionally measure them through the current AVX2 runtime, and persist selected schedules.

---

## 12. File Plan

Architecture docs:

```text
docs/design/m9-canonical-micro-ir-architecture.md
docs/design/m9-micro-ir-core-concepts.md
```

Future implementation files:

```text
include/LLK/Dialect/Micro/MicroDialect.td
include/LLK/Dialect/Micro/MicroOps.td
include/LLK/Dialect/Micro/MicroTypes.td
include/LLK/Dialect/Micro/MicroDialect.h

lib/Dialect/Micro/MicroDialect.cpp
lib/Dialect/Micro/MicroOps.cpp

include/LLK/Conversion/LLKToMicro/LLKToMicro.h
lib/Conversion/LLKToMicro/LLKToMicro.cpp

include/LLK/Perf/MachineModel.h
include/LLK/Perf/MachineModelLoader.h
include/LLK/Perf/MicroCostModel.h
include/LLK/Perf/MicroDAG.h
include/LLK/Perf/SearchSpace.h
include/LLK/Perf/Candidate.h

lib/Perf/MachineModel.cpp
lib/Perf/MachineModelLoader.cpp
lib/Perf/MicroCostModel.cpp
lib/Perf/MicroDAG.cpp
lib/Perf/SearchSpace.cpp
lib/Perf/Candidate.cpp

machines/generic-ai-accel-v1.yaml
machines/x86-avx2-cpu.yaml

tools/micro-opt/micro-opt.cpp
tools/micro-perf/micro-perf.cpp
```

Future tests:

```text
test/Dialect/Micro/ops.mlir
test/Dialect/Micro/search_space.mlir
test/Conversion/LLKToMicro/swiglu_to_micro.mlir
test/Conversion/LLKToMicro/matmul_to_micro.mlir
test/Perf/machine_model_loader.cpp
test/Perf/l0_static_bound.cpp
test/Perf/l1_resource_dag.cpp
test/Perf/search_candidate_binding.cpp
```

---

## 13. Milestones

### M9: Canonical Micro-IR Foundation

Build concrete execution ops and verifiers.

Success criteria:

- `micro.kernel` examples parse and print.
- verifier rejects unknown memory spaces, invalid waits, malformed MMA shapes, and invalid mapping attributes.
- representative GEMM/SwiGLU tile examples are committed as FileCheck tests.

### M10: MachineModel and Perf L0/L1

Build YAML loader and performance evaluator.

Success criteria:

- machine YAML profiles load into typed C++ structs.
- invalid profiles produce actionable errors.
- `micro-perf` evaluates one concrete micro kernel on both generic accelerator and AVX2 CPU profiles.
- L0 and L1 outputs are deterministic.

### M11: LLKToMicro Export Path

Materialize current scheduled LLK kernels as concrete micro IR.

Success criteria:

- `llk.fused_swiglu` can emit concrete micro IR.
- basic matmul can emit concrete micro IR.
- `llk-compile --emit=micro` works without breaking existing LLVM/JIT flow.
- generated micro IR is accepted by `micro-perf`.

### M12: Search-Space Micro-IR and Auto Scheduler

Add parametric search representation and candidate binding.

Success criteria:

- search-space IR can express tile sizes, pipeline stages, spatial mapping, and vector width.
- candidate binding emits concrete micro IR.
- legality verifier catches capacity and machine-support violations.
- `llk-tune` can evaluate multiple candidates through `micro-perf`.

### M13: Feedback and Calibration

Persist measured and predicted metrics.

Success criteria:

- selected schedules are written as YAML.
- predicted vs measured latency is tracked for AVX2.
- regression tests expose prediction drift.

---

## 14. Non-Goals

- Do not replace LLK semantic ops.
- Do not replace Linalg/Vector in the first milestone.
- Do not encode vendor-specific ISA operations in canonical `micro`.
- Do not build a cycle-accurate simulator first.
- Do not switch the existing schedule DB to YAML in M9.
- Do not make machine profiles permissive or schema-less.

---

## 15. Risks and Mitigations

### Risk: Micro-IR Becomes Too Abstract

If `micro` only represents generic tensor contraction, performance estimates will be weak.

Mitigation: require explicit memory spaces, movement ops, pipeline structure, and mapping attributes in concrete micro examples.

### Risk: Micro-IR Becomes Too Vendor-Specific

If `micro` encodes vendor ISA details, it loses portability.

Mitigation: use architecture-parametric ops such as `micro.mma` with tile shape and dtype, then let machine profiles decide support and cost.

### Risk: Search Constructs Pollute Lowering

Search metadata could make concrete lowering ambiguous.

Mitigation: require candidate binding to produce fully concrete `micro.kernel`; target lowerings reject search ops.

### Risk: Performance Model Gives False Confidence

Uncalibrated estimates can mislead design decisions.

Mitigation: report model level, bottleneck reason, and prediction-vs-measurement error when measurement exists.

### Risk: Initial Scope Is Too Large

Dialect, machine model, perf engine, and tuner are all substantial.

Mitigation: stage M9-M13 and keep existing AVX2 compiler path independent until `micro` earns its place.

---

## 16. Resolved Design Choices

The implementation plan should use these choices unless later evidence shows a concrete problem.

### 16.1 Constraint Representation

MVP constraints are represented as typed built-in constraint ops or attributes, not as an arbitrary expression language.

Examples:

```text
sram_capacity(BM, BN, BK, dtype)
acc_capacity(BM, BN, accumulator_dtype)
mma_compatible(BM, BN, BK, input_dtype, accumulator_dtype)
mapping_extent_leq(spatial_m, machine.compute.matrix_engines.count)
tail_supported(M, BM)
```

This keeps verification deterministic and avoids building an expression parser in the first search milestone.

### 16.2 Search-Space Ownership

Search spaces are persisted in MLIR using `micro.search_space`, `micro.param`, `micro.constraint`, `micro.objective`, and `micro.candidate`. The tuner lowers those ops into typed C++ search objects for generation, pruning, and ranking.

MLIR is the interchange format. Typed C++ is the execution data structure.

### 16.3 Perf Output Format

`micro-perf` emits YAML by default so results can be stored next to machine and schedule YAML. It may also support a compact human-readable text mode for terminal use, but tests should validate the YAML output.

### 16.4 Schedule Persistence

Selected schedule YAML supplements `schedules/schedule_db.json` through M13. It does not replace the existing JSON database until the current AVX2 pipeline consumes selected YAML schedules reliably.

### 16.5 Attention Lowering Boundary

M11 implements `llk.fused_swiglu` and basic matmul to `micro`. Attention and online softmax lower later, after `micro.reduce`, vector epilogues, SRAM lifetime modeling, and capacity checks are already tested on simpler kernels.

---

## 17. Final Contract

The redesigned compiler should eventually answer:

```text
Given a model/kernel, shape, and machine YAML,
generate or search legal micro schedules,
bind candidates into concrete execution IR,
estimate latency/utilization/bottlenecks,
optionally compile and measure on AVX2,
and persist the best schedule with prediction and measurement metadata.
```

The canonical artifact is no longer only optimized MLIR or object code. It is the pair:

```text
concrete micro.kernel + MachineModel YAML
```

For auto-optimization, the canonical source is:

```text
micro.search_space + workload shape + MachineModel YAML
```
