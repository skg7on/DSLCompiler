# Micro-IR Core Concepts

**Parent:** [m9-canonical-micro-ir-architecture.md](m9-canonical-micro-ir-architecture.md)
**Status:** Draft Design
**Scope:** Core semantics of the canonical `micro` dialect.

---

## 1. Definition

`micro` is a tile-centric canonical execution IR for DNN kernels. It represents a scheduled tensor program in terms of:

- tile dataflow
- compute operations
- data movement
- memory placement
- tile layout
- spatial mapping
- temporal scheduling
- synchronization
- communication
- tunable schedule choices

It sits below LLK/Linalg and above target-specific ISA dialects or runtime lowerings.

`micro` should answer:

```text
How is this computation executed on a machine with compute engines,
memories, data movement engines, interconnects, and synchronization costs?
```

It should not answer:

```text
What high-level DNN operator is this?
```

That question belongs to LLK, StableHLO, TOSA, or framework IR.

### 1.1 Tile as the Primary Value

The primary SSA value in `micro` is a tile:

```text
Tile = (shape, dtype, layout, memory_space, owner)
```

`shape` defines the multidimensional execution granularity. `dtype` defines element or accumulator format. `layout` defines order, blocking, vectorization, swizzle, banking, and alignment. `memory_space` defines where the tile is materialized. `owner` defines the execution resource scope that owns or consumes the tile.

The intended MLIR type is:

```mlir
!micro.tile<32x64xbf16,
             layout = #micro.layout<row_major, vector = 8>,
             memory = #micro.memory<sram>,
             owner = #micro.owner<worker>>
```

Concrete Micro-IR distinguishes:

- logical tiles: zero-cost views, slices, partitions, reshapes, and broadcasts
- memory tiles: materialized data in a memory level
- execution tiles: tiles assigned to owner resources
- instruction fragments: tiles matching MMA, vector, DMA, reduction, or communication granularity

### 1.2 Layered Tile IR

The dialect has one textual namespace, `micro`, but the design separates three conceptual layers:

```text
uTile: tile dataflow
    -> uMap: memory placement, owner mapping, pipeline, sync
    -> uHW: instruction fragments and machine-visible events
```

`uTile` describes algorithmic tile operations such as matmul, elementwise, reduce, broadcast, and partition. `uMap` binds tiles to memory levels, owners, pipelines, copy paths, and synchronization. `uHW` resolves mapped tiles into canonical hardware-near fragments such as MMA shapes, vector widths, DMA transactions, barriers, and NoC-like communication events.

The detailed tile model is specified in [2026-08-13-micro-ir-tile-programming-model-spec.md](../superpowers/specs/2026-08-13-micro-ir-tile-programming-model-spec.md).

---

## 2. Two Forms of Micro-IR

`micro` has two intentionally separate forms.

### 2.1 Search Micro-IR

Search IR describes legal schedule choices and optimization objectives.

Example:

```mlir
micro.search_space @swiglu_space
    attributes { workload = "fused_swiglu" } {
  %BM = micro.param "BM" : i64 {choices = [8, 16, 32, 64]}
  %BN = micro.param "BN" : i64 {choices = [32, 64, 128]}
  %BK = micro.param "BK" : i64 {choices = [32, 64, 128]}
  %stages = micro.param "num_stages" : i64 {choices = [1, 2, 3]}

  micro.constraint {
    sram_bytes(%BM, %BN, %BK) <= machine.memory.sram.capacity_bytes
    compatible_mma_shape(%BM, %BN, %BK, bf16, f32)
  }

  micro.objective minimize latency
      secondary [maximize matrix_utilization, minimize dram_bytes]
}
```

Search IR is consumed by auto-schedulers and tuners. It is not lowered to LLVM or a backend directly.

### 2.2 Concrete Execution Micro-IR

Concrete IR describes one bound candidate.

Example:

```mlir
micro.kernel @swiglu_candidate_17
    attributes {
      workload = "fused_swiglu",
      candidate = "candidate_17",
      target = "x86-avx2-cpu"
    } {
  micro.spatial_for %bm = 0 to %M step 32 map = #micro.map<worker> {
    micro.spatial_for %bn = 0 to %N step 64 map = #micro.map<lane> {
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

Concrete IR is consumed by performance models and future target lowerings.

---

## 3. Operation Families

### 3.1 Structure

```text
micro.kernel
micro.for
micro.spatial_for
micro.pipeline
micro.stage
```

Structure ops describe execution hierarchy and schedule organization.

`micro.for` is temporal. Iterations occur over time on the same logical resources.

`micro.spatial_for` is spatial. Iterations are mapped onto hardware resources such as clusters, cores, PEs, lanes, or thread-pool workers.

### 3.2 Memory

```text
micro.alloc
micro.view
micro.tile_view
micro.tile_slice
micro.tile_partition
micro.tile_alloc
micro.load
micro.store
micro.copy
micro.async_copy
micro.tile_load
micro.tile_store
micro.tile_copy
micro.tile_async_copy
micro.prefetch
micro.tile_prefetch
```

Memory ops must carry enough information to compute:

- source and destination memory spaces
- byte counts
- tile shapes
- tile layouts
- lifetimes
- capacity pressure
- reuse opportunities
- overlap potential

Logical tile ops such as `micro.tile_view`, `micro.tile_slice`, and `micro.tile_partition` are zero-cost until materialized. Materialization occurs through allocation, load, copy, async copy, store, layout transform, or compute consumption.

### 3.3 Compute

```text
micro.mma
micro.tile_mma
micro.vector
micro.tile_vector
micro.reduce
micro.tile_reduce
micro.tile_scan
micro.permute
micro.tile_permute
micro.convert
micro.tile_convert
micro.special
```

`micro.mma` and `micro.tile_mma` represent matrix/tensor engine work one level above vendor ISA. They should describe input tile shapes, fragment shape, input dtype, accumulator dtype, layout constraints, and optional resource binding. The MVP may keep the shorter `micro.mma` spelling while requiring tile-typed operands.

`micro.vector` and `micro.tile_vector` represent vector elementwise work such as add, mul, exp, silu, reciprocal, or compare/select.

`micro.reduce` and `micro.tile_reduce` represent reductions such as sum, max, min, or product.

`micro.special` is reserved for hardware-visible functional units that do not fit the other compute ops. It must be used sparingly and with verifier checks.

### 3.4 Synchronization

```text
micro.wait
micro.barrier
micro.fence
```

Synchronization ops make dependencies explicit for both lowering and performance modeling.

### 3.5 Communication

```text
micro.broadcast
micro.gather
micro.scatter
micro.exchange
micro.collective
micro.tile_distribute
micro.tile_replicate
micro.tile_exchange
```

Communication ops describe data movement between spatial resources or memory partitions. MVP support can be limited to parsing and modeling simple byte movement.

---

## 4. Attributes

### 4.1 Memory Spaces

```text
dram
l2
sram
rf
acc
scratch
```

The memory hierarchy is architecture-parametric. A machine profile decides what each space means for capacity, bandwidth, latency, banking, and legal transfers.

### 4.2 Mapping Targets

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

`worker` exists so the current AVX2 thread pool can validate spatial mapping without pretending to be a systolic array.

### 4.3 DTypes

Required MVP dtypes:

```text
f32
f16
bf16
i32
i8
```

Near-term extension dtypes:

```text
fp8.e4m3
fp8.e5m2
int4
fp4
```

Later composite dtypes:

```text
block_float(element, scale, block_size)
sparse_2_4(element)
```

Composite dtypes affect memory traffic, packing/unpacking, scale fetches, tensor-engine selection, and conversion costs.

### 4.4 Layouts

Required MVP layouts:

```text
row_major
col_major
blocked
vectorized
swizzled
```

Layout attributes must carry enough metadata for alignment, banking, vector-load legality, tensorization, and physical layout-transform cost:

```text
order
block
vector
align
banks
swizzle
```

Layout reinterpretation is a logical tile transform only when byte order is unchanged. Physical data rearrangement must be represented as movement or compute with measurable cost.

### 4.5 Owners

Canonical owner scopes:

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

The MachineModel maps these owner scopes to target resources. AVX2 can map `worker` to a thread-pool worker and `vector_engine` to AVX2 vector/FMA resources; an accelerator can map `core`, `pe_group`, `pe`, and `matrix_engine` to its array hierarchy.

---

## 5. Search Concepts

### 5.1 Tunable Parameters

Search parameters must be explicit SSA-like values in `micro.search_space`:

```text
BM, BN, BK
num_stages
spatial_m, spatial_n
vector_width
unroll
prefetch_distance
layout
tile_layout
owner_mapping
fragment_shape
memory_space
reduction_strategy
fusion_boundary
```

### 5.2 Constraints

Constraints define the legal candidate set:

```text
capacity constraints
dtype support constraints
matrix-engine tile compatibility
divisibility or tail handling
mapping resource limits
banking/layout constraints
owner hierarchy constraints
fragment compatibility constraints
pipeline live-tile constraints
parallelism limits
```

Constraints are evaluated against workload shape and `MachineModel`.

### 5.3 Objectives

Objectives define how candidates are ranked:

```text
minimize latency
maximize throughput
maximize matrix utilization
minimize DRAM bytes
minimize energy
minimize capacity spill
```

The MVP objective is latency, with matrix utilization and DRAM bytes as secondary metrics.

### 5.4 Candidates

A candidate is a binding of search parameters:

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

Candidate binding produces concrete `micro.kernel`.

---

## 6. Machine Model YAML

Machine models are authored in YAML and validated strictly.

Example:

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

Unknown keys are errors unless explicitly versioned. This keeps machine profiles readable without making compiler behavior ambiguous.

---

## 7. Lowering Rules

### 7.1 From LLK/Linalg to Micro

The `LLKToMicro` pass consumes scheduled structured IR and emits one or both of:

- `micro.search_space`
- concrete `micro.kernel`

For early milestones, `LLKToMicro` can start with direct scheduled patterns for matmul and fused SwiGLU.

### 7.2 From Search Micro to Concrete Micro

Candidate binding substitutes parameter values, checks constraints, and emits concrete schedule structure:

```text
micro.param -> constants
micro.constraint -> verifier checks
micro.objective -> tuning metadata
micro.candidate -> concrete micro.kernel attributes
```

### 7.3 From Concrete Micro to Perf Events

Concrete micro ops lower to cost-model events:

```text
async_copy -> DMA event
tile_async_copy -> DMA event
mma -> matrix engine event
tile_mma -> matrix engine event
vector/reduce -> vector engine event
tile_vector/tile_reduce -> vector engine event
wait/barrier -> dependency or sync event
load/store -> memory event
tile_load/tile_store -> memory event
```

### 7.4 From Concrete Micro to Target Lowering

Target lowering is future work. The first target can map concrete `micro` to the existing AVX2 path, but the first milestone should focus on export and evaluation.

---

## 8. Verifier Requirements

Concrete `micro.kernel` verifier:

- all search parameters are bound
- every tile has known shape, dtype, layout, memory space, and owner when required by its kind
- logical tile views are zero-cost and do not claim materialized storage
- materialized memory tiles have capacity-visible lifetime and byte size
- instruction fragments divide or legally mask their parent execution tile
- memory spaces are known
- transfer source and destination spaces are legal
- `micro.wait` operands reference async operations
- `micro.mma` dtypes and tile shape are supported by the selected machine when a machine is provided
- tile layouts are compatible with their memory space and compute owner
- owner mappings are valid and nested scopes are compatible
- allocations fit static capacity when shape is static
- `micro.spatial_for` mapping targets are valid

Search verifier:

- parameter names are unique
- choices are non-empty
- constraints reference known parameters
- objective metrics are known
- candidates bind all required parameters exactly once

---

## 9. Design Invariants

- Concrete `micro.kernel` has no DNN semantic ops.
- Tile is the primary execution value.
- Logical tiles, memory tiles, execution tiles, and instruction fragments have distinct cost semantics.
- Layout and owner mapping are first-class, verifiable, and searchable.
- Search constructs are never lowered directly to LLVM.
- Machine-specific costs live in YAML machine models and typed C++ structs.
- Schedule choices become explicit IR structure.
- Existing LLK, Linalg, Vector, and AVX2 paths remain valid during the transition.
- Performance estimates are evidence for search, not proof of hardware behavior until calibrated.
