# Micro-IR Tile Programming Model Spec

**Status:** Draft
**Date:** 2026-08-13
**Parent Spec:** [Canonical Micro-IR Redesign Spec](2026-08-11-canonical-micro-ir-redesign.md)
**Related Docs:**
- [M9+ Canonical Micro-IR Architecture](../../design/m9-canonical-micro-ir-architecture.md)
- [Micro-IR Core Concepts](../../design/m9-micro-ir-core-concepts.md)
- [Micro-IR Dialect Implementation Spec](2026-08-11-micro-ir-dialect-implementation-spec.md)
- [LLK/Linalg to Micro-IR Lowering Implementation Spec](2026-08-11-llk-to-micro-lowering-implementation-spec.md)
- [Micro-IR Auto-Tuning and Auto-Optimization Implementation Spec](2026-08-11-micro-ir-autotuning-implementation-spec.md)
- [Micro-IR AVX2 Simulator and Emulator Implementation Spec](2026-08-11-micro-ir-avx2-simulator-implementation-spec.md)

---

## 1. Goal

Refine canonical DNN Micro-IR into a tile-centric execution IR. The IR must be low-level enough to evaluate how a model kernel uses AI chipset resources, but still architecture-parametric enough to describe AVX2 CPUs, GPUs, NPUs, systolic arrays, and custom accelerators.

The core definition becomes:

```text
DNN Micro-IR = Tile Dataflow + Tile Mapping + Tile Schedule
Performance = F(Tile Dataflow, Mapping, Schedule, MachineModel, Calibration)
```

Tiles are not a convenience syntax. A tile is the primary SSA value that connects MLIR-level tensor programs to hardware-level execution fragments.

---

## 2. Canonical Tile Value

The fundamental Micro-IR value is:

```text
Tile = (shape, dtype, layout, memory_space, owner)
```

Each field has hardware-evaluation meaning:

| Field | Meaning | Performance Role |
|-------|---------|------------------|
| `shape` | Static or symbolic extents for the tile | FLOPs, bytes, fragment compatibility, tail handling |
| `dtype` | Element or accumulator type | compute throughput, memory bytes, conversion cost |
| `layout` | Order, stride, vectorization, swizzle, banking, alignment | coalescing, bank conflicts, vector load legality, tensorization |
| `memory_space` | Logical memory level or register class | capacity, bandwidth, latency, copy path legality |
| `owner` | Execution resource scope that owns or consumes the tile | occupancy, parallelism, synchronization, communication |

The MVP should implement this as a first-class MLIR type:

```mlir
!micro.tile<32x64xbf16,
             layout = #micro.layout<row_major, vector = 8>,
             memory = #micro.memory<sram>,
             owner = #micro.owner<worker>>
```

If TableGen custom type parsing is too large for the first patch, the fallback is:

```mlir
tensor<32x64xbf16>
  attributes {
    micro.layout = #micro.layout<row_major, vector = 8>,
    micro.memory = #micro.memory<sram>,
    micro.owner = #micro.owner<worker>
  }
```

The fallback is only an implementation staging tactic. The design contract is `!micro.tile`.

---

## 3. Tile Kinds

Micro-IR must distinguish four tile kinds because they have different cost semantics.

### 3.1 Logical Tile

A logical tile is a view, slice, partition, reshape, or broadcast of another tensor or tile.

Properties:

- no data movement cost by itself
- no independent storage
- may carry shape, indexing, and intended layout metadata
- becomes materialized only when a load, copy, store, compute, or layout-transform op requires it

Representative ops:

```text
micro.tile_view
micro.tile_slice
micro.tile_partition
micro.tile_reshape
micro.tile_broadcast
```

### 3.2 Memory Tile

A memory tile is materialized in a memory level such as DRAM, L2, SRAM, RF, accumulator, or scratch.

Properties:

- consumes capacity in its memory space
- participates in copy/load/store events
- has lifetime and reuse
- exposes byte count, alignment, layout, and banking constraints

Representative ops:

```text
micro.tile_alloc
micro.tile_load
micro.tile_store
micro.tile_copy
micro.tile_async_copy
micro.tile_prefetch
```

### 3.3 Execution Tile

An execution tile is assigned to a spatial execution entity.

Properties:

- binds work to cluster, core, PE group, worker, lane, matrix engine, vector engine, or DMA engine
- defines occupancy and parallelism
- controls communication between owners
- is the unit of schedule search for spatial mapping

Representative ops:

```text
micro.tile_map
micro.tile_distribute
micro.tile_replicate
micro.tile_exchange
micro.spatial_for
```

### 3.4 Instruction Fragment

An instruction fragment is the smallest tile that maps cleanly to a hardware execution primitive.

Properties:

- matches matrix-engine, vector-engine, DMA, or reduction granularity
- may be much smaller than the algorithmic tile
- drives utilization and tensorization legality
- is the closest canonical Micro-IR level to target ISA

Representative ops:

```text
micro.tile_mma
micro.tile_vector
micro.tile_reduce
micro.tile_permute
micro.tile_convert
```

The MVP may keep existing op spellings such as `micro.mma`, `micro.vector`, and `micro.reduce`, but their operands and results should be tile-typed.

---

## 4. Layered Tile IR

Micro-IR has three conceptual layers. They are not necessarily three dialects in the MVP, but the separation must be visible in the op design, verifiers, and performance model.

### 4.1 `uTile`: Algorithmic Tile Dataflow

`uTile` describes the computation as a graph of multidimensional tile operations.

Examples:

```text
tile matmul:      A[MxK] x B[KxN] -> C[MxN]
tile elementwise: SiLU(G) * U
tile reduce:      reduce K into accumulator
tile broadcast:   expand a scale or mask tile
```

`uTile` answers:

- What are the tile shapes?
- How are tensors partitioned?
- Which tiles are reused?
- Which computation consumes each tile?

It must not decide the exact memory hierarchy, owner resource, or instruction fragment unless already known.

### 4.2 `uMap`: Memory, Owner, and Schedule Mapping

`uMap` binds tile dataflow to a machine-visible execution plan.

It answers:

- Where is each tile materialized?
- Which execution resource owns each tile?
- Which copy path moves each tile?
- Which tiles are pipelined or double-buffered?
- Which synchronization edges are required?
- How are cluster/core/PE/worker scopes used?

This is the most important layer for architecture evaluation. Many AI chipset performance questions are questions about `uMap`: SRAM fit, NoC traffic, DMA overlap, core occupancy, accumulator pressure, and synchronization cost.

### 4.3 `uHW`: Native Execution Fragments

`uHW` resolves mapped tiles into hardware-execution fragments.

Examples:

```text
MMA fragment:      16x16x32 bf16xf32
AVX2 vector tile:  8xf32 lanes
DMA transaction:   aligned 256-byte transfer
Barrier:           core or cluster scope
NoC transfer:      source owner -> destination owner
```

`uHW` is still canonical Micro-IR, not vendor ISA. Backend lowering may then map it to LLVM vector/FMA, GPU MMA, systolic-array commands, or custom accelerator packet formats.

---

## 5. Tile Operation Families

The tile programming model adds the following canonical operation families.

### 5.1 Construction and Transformation

```text
micro.tile_view
micro.tile_slice
micro.tile_partition
micro.tile_reshape
micro.tile_transpose
micro.tile_broadcast
micro.tile_concat
micro.tile_pad
```

Rules:

- `tile_view`, `tile_slice`, `tile_partition`, `tile_reshape`, and `tile_broadcast` are logical unless they require materialization.
- `tile_transpose` is logical only if expressed as layout reinterpretation; it is physical when it changes memory order.
- `tile_pad` must state whether padding is virtual or materialized.

### 5.2 Movement

```text
micro.tile_load
micro.tile_store
micro.tile_copy
micro.tile_async_copy
micro.tile_prefetch
```

Rules:

- movement ops produce memory tiles
- async movement returns a tile and a token
- source/destination memory spaces must be legal in the MachineModel
- byte count must be derivable from tile shape, dtype, layout, and padding

### 5.3 Compute

```text
micro.tile_mma
micro.tile_vector
micro.tile_reduce
micro.tile_scan
micro.tile_permute
micro.tile_convert
```

Rules:

- compute ops consume execution tiles or instruction fragments
- `tile_mma` must state the fragment shape and accumulator dtype
- `tile_vector` must state vector width or derive it from the owner resource
- `tile_reduce` must state reduction axes and accumulator layout
- `tile_convert` must expose conversion cost and resulting dtype

### 5.4 Mapping

```text
micro.tile_map
micro.tile_distribute
micro.tile_replicate
micro.tile_partition_owner
micro.spatial_for
```

Rules:

- mapping ops produce execution tiles
- owners must exist in MachineModel
- owner extent cannot exceed available resources
- replicated tiles create memory pressure and possible communication cost

### 5.5 Schedule and Synchronization

```text
micro.for
micro.spatial_for
micro.pipeline
micro.stage
micro.wait
micro.barrier
micro.fence
```

Rules:

- pipeline operands are actual tile-producing operations, not opaque metadata
- wait operands must be async tokens
- barrier and fence scopes must match tile owners
- double buffering is represented by distinct tile instances per stage

---

## 6. Layout Model

Layout is first-class. It must not be hidden in backend lowering.

MVP layout attribute:

```mlir
#micro.layout<row_major>
#micro.layout<col_major>
#micro.layout<blocked, block = [16, 16]>
#micro.layout<row_major, vector = 8, align = 32>
#micro.layout<swizzled, swizzle = "xor", banks = 32>
```

Required layout properties:

| Property | Purpose |
|----------|---------|
| `order` | logical element traversal |
| `block` | blocked or tiled memory order |
| `vector` | contiguous elements consumed by vector lanes |
| `align` | load/store and DMA alignment |
| `banks` | banking legality and conflict modeling |
| `swizzle` | target-independent memory swizzle description |

Verifier rules:

- reinterpretation between layouts is legal only when byte order is unchanged or explicitly marked as a logical view.
- physical layout transformation must be represented as movement or compute with measurable cost.
- MMA/vector fragments must use layouts supported by their owner compute engine.
- search candidates with unsupported layout or alignment are illegal.

---

## 7. Ownership Model

Owner describes the resource scope responsible for a tile.

Canonical owner names:

```text
cluster
core
warp
wave
subgroup
pe_group
pe
lane
worker
matrix_engine
vector_engine
dma
```

The MachineModel maps these logical owners to target resources. For AVX2, the initial mapping can be:

```text
worker         -> thread-pool worker
lane           -> AVX2 vector lane group
vector_engine  -> AVX2 FMA/vector unit model
dma            -> modeled memory copy path
```

For a systolic-array-like accelerator, the mapping can be:

```text
cluster        -> chiplet or array group
core           -> compute tile
pe_group       -> PE row/column group
pe             -> processing element
matrix_engine  -> systolic/MMA primitive
dma            -> DMA or NoC copy engine
```

Verifier rules:

- every execution tile must have exactly one owner scope
- owner transitions require movement, communication, replication, or synchronization
- a tile cannot be consumed by a compute engine outside its owner scope without an explicit transfer
- nested owners must be compatible with enclosing `spatial_for` scopes

---

## 8. Canonical Example: Tiled Matmul

Search-level tile IR:

```mlir
micro.search_space @matmul_space {
  %BM = micro.param "BM" : i64 {choices = [16, 32, 64, 128]}
  %BN = micro.param "BN" : i64 {choices = [16, 32, 64, 128]}
  %BK = micro.param "BK" : i64 {choices = [32, 64, 128]}
  %stages = micro.param "num_stages" : i64 {choices = [1, 2, 3]}
  %layoutA = micro.param "layout_a" : !micro.layout
      {choices = [#micro.layout<row_major>, #micro.layout<blocked, block = [16, 32]>]}

  micro.constraint {
    tile_divides_or_masks(M, %BM)
    tile_divides_or_masks(N, %BN)
    tile_divides_or_masks(K, %BK)
    materialized_tile_bytes(%BM, %BK, bf16, %layoutA) <= machine.memory.sram.capacity_bytes
    fragment_compatible(%BM, %BN, %BK, machine.compute.matrix_engines)
  }

  micro.objective minimize latency
      secondary [maximize compute_utilization, minimize dram_bytes, minimize sram_spill]
}
```

Concrete tile execution IR:

```mlir
micro.kernel @matmul_candidate_0 attributes {target = "x86-avx2-cpu"} {
  micro.spatial_for %bm = 0 to %M step 32 map = #micro.map<worker> {
    micro.spatial_for %bn = 0 to %N step 64 map = #micro.map<worker> {
      %a_l = micro.tile_view %A[%bm, 0] shape = [32, %K]
          : tensor<?x?xbf16> -> !micro.tile<32x?xbf16, layout = #micro.layout<row_major>>
      %b_l = micro.tile_view %B[0, %bn] shape = [%K, 64]
          : tensor<?x?xbf16> -> !micro.tile<?x64xbf16, layout = #micro.layout<row_major>>
      %c = micro.tile_alloc shape = [32, 64]
          dtype = #micro.dtype<f32>
          memory = #micro.memory<acc>
          owner = #micro.owner<worker>

      micro.pipeline stages = 2 {
        micro.for %bk = 0 to %K step 64 {
          %a, %ta = micro.tile_async_copy %a_l[%bm, %bk]
              to #micro.memory<sram> owner = #micro.owner<worker>
          %b, %tb = micro.tile_async_copy %b_l[%bk, %bn]
              to #micro.memory<sram> owner = #micro.owner<worker>
          micro.wait %ta, %tb

          %af = micro.tile_partition %a shape = [16, 32]
              owner = #micro.owner<vector_engine>
          %bf = micro.tile_partition %b shape = [32, 16]
              owner = #micro.owner<vector_engine>
          micro.tile_mma %af, %bf, %c
              {fragment = [16, 16, 32],
               input = #micro.dtype<bf16>,
               accumulator = #micro.dtype<f32>}
        }
      }

      micro.tile_store %c, %C[%bm, %bn]
          to #micro.memory<dram>
    }
  }
  micro.yield
}
```

This syntax is illustrative. The implementation spec defines the exact MVP spelling. The required semantic points are first-class tiles, explicit materialization, explicit owner mapping, and explicit fragment compatibility.

---

## 9. Lowering from MLIR to Tile Micro-IR

The lowering path should resolve progressively:

```text
Linalg/Tensor operation
    -> logical tile dataflow
    -> multi-level tile hierarchy
    -> layout selection
    -> memory placement
    -> owner mapping
    -> pipeline and synchronization
    -> instruction fragments
    -> target lowering
```

Mapping rules:

| Source Concept | Micro-IR Result |
|----------------|-----------------|
| Linalg loop nest | `micro.for`, `micro.spatial_for`, logical tile views |
| Tiling schedule | `micro.tile_partition` hierarchy |
| Fusion | shared tile producers and consumers |
| Packing | layout attribute or physical layout transform |
| Bufferization | memory tile allocation and movement |
| Vectorization | `micro.tile_vector` fragments |
| Matmul tensorization | `micro.tile_mma` fragments |
| Async pipeline annotation | `micro.pipeline`, `micro.tile_async_copy`, `micro.wait` |
| Parallel mapping | owner attributes and `micro.spatial_for` mappings |

Initial lowering should support matmul and fused SwiGLU because they exercise the important cases: repeated K-reduction, accumulator tiles, double contraction reuse, vector post-processing, and output stores.

---

## 10. Auto-Search and Auto-Optimization

Tile-centric Micro-IR improves auto-search because the scheduler can mutate the exact values that determine hardware behavior.

Required search dimensions:

```text
algorithmic tile shape:       BM, BN, BK
multi-level tile hierarchy:   cluster/core/PE/fragment shapes
layout:                       row, column, blocked, vectorized, swizzled
memory placement:             DRAM/L2/SRAM/RF/ACC/scratch
owner mapping:                worker/core/PE/lane/matrix_engine/vector_engine
pipeline:                     stages, prefetch distance, double buffering
tensorization:                MMA/vector/reduction fragment shape
loop schedule:                order, unroll, split, fuse, reorder
communication:                replicate, shard, exchange, collective strategy
```

Candidate legality must check:

- tile hierarchy divisibility or masked-tail support
- memory capacity by live materialized tile set
- legal copy path between memory spaces
- owner extent within MachineModel resources
- layout compatibility with memory banking and compute engines
- fragment compatibility with matrix/vector engines
- synchronization scope compatibility
- pipeline stage liveness and outstanding async limits

Candidate ranking should use:

- L0 static bound for fast pruning
- L1 resource DAG for top candidates
- optional AVX2 measurement for calibration
- persisted schedule YAML with selected tile hierarchy, layout, owner mapping, and predicted/measured metrics

---

## 11. Simulator and Emulator Contract

The performance evaluator consumes concrete tile Micro-IR and MachineModel YAML.

Event generation rules:

| Micro-IR Tile Op | Event |
|------------------|-------|
| `tile_async_copy` | DMA or memory-copy event with bytes, path, latency, bandwidth, token |
| `tile_load` / `tile_store` | memory event with source/destination spaces |
| `tile_mma` | matrix-engine event with fragment shape, dtype, issue cycles, latency |
| `tile_vector` | vector-engine event with lanes, dtype, operation mix |
| `tile_reduce` | reduction event with data dependency and owner scope |
| `tile_exchange` | communication event between owners |
| `wait` / `barrier` / `fence` | dependency and synchronization event |

The L0 evaluator estimates roofline-style lower bounds from tile bytes and tile FLOPs. The L1 evaluator schedules the event DAG against compute engines, memory ports, DMA queues, owner resources, and synchronization costs. L2 calibration is future work and should remain optional until L0/L1 are useful.

For Intel AVX2, tile fragments map to vector/FMA issue models rather than a real tensor engine. That is acceptable because AVX2 is a validation backend, not the only intended target.

---

## 12. Implementation Plan

### M9a: Tile Type and Attributes

Add:

- `!micro.tile`
- `#micro.layout`
- `#micro.owner`
- verifier support for shape, dtype, memory, layout, and owner fields

Acceptance:

- `llk-opt` can parse, print, and round-trip representative tile types and attributes.
- invalid tile memory spaces, owners, and layout parameters fail verification.

### M9b: Tile Ops MVP

Add:

- `micro.tile_view`
- `micro.tile_alloc`
- `micro.tile_async_copy`
- `micro.tile_store`
- `micro.tile_partition`
- tile-typed `micro.mma`, `micro.vector`, and `micro.reduce`

Acceptance:

- representative matmul and fused SwiGLU tile kernels round-trip and verify.
- logical tile views do not count as materialized memory.
- async copies return explicit tokens.

### M10a: Tile-Aware MachineModel

Extend YAML with:

- owner hierarchy and resource counts
- supported tile layouts
- matrix/vector fragment shapes
- memory-space copy paths
- banking, alignment, and async queue limits

Acceptance:

- `generic-ai-accel-v1.yaml` models matrix fragments and owner hierarchy.
- `x86-avx2-cpu.yaml` maps workers and vector lanes without pretending AVX2 is an accelerator array.

### M10b: Tile-Aware Performance Events

Lower tile ops to L0/L1 events.

Acceptance:

- materialized tile bytes, capacity, and lifetime are visible in reports.
- tile MMA/vector/copy events report utilization and bottlenecks.
- layout transform and communication costs are modeled when present.

### M11a: Tile-Centric Lowering

Update LLK/Linalg lowering to emit tile dataflow first, then mapping and hardware fragments.

Acceptance:

- matmul and fused SwiGLU emit logical views, materialized memory tiles, owner mapping, pipeline, and tile compute ops.
- the existing AVX2/JIT path remains available.

### M12a: Tile Search Space

Extend search-space IR and tuning records with tile hierarchy, layout, memory placement, owner mapping, pipeline, and fragment parameters.

Acceptance:

- tuner can generate and rank legal tile candidates using MachineModel constraints.
- selected schedule YAML records tile decisions and performance predictions.

---

## 13. Design Invariants

- Tile is the primary execution value in Micro-IR.
- Logical tile operations are zero-cost views until materialized.
- Memory tiles consume capacity and participate in byte movement.
- Execution tiles bind work to owner resources.
- Instruction fragments model the closest canonical level to hardware execution.
- Layout is first-class and can be searched, verified, and costed.
- Owner mapping is first-class and can be searched, verified, and costed.
- Concrete `micro.kernel` contains no high-level DNN semantic ops.
- Search IR binds into deterministic concrete tile Micro-IR before performance evaluation or lowering.
