# Micro-IR AVX2 Simulator and Emulator Implementation Spec

**Status:** Draft
**Date:** 2026-08-11
**Parent Spec:** [Canonical Micro-IR Redesign Spec](2026-08-11-canonical-micro-ir-redesign.md)

---

## 1. Goal

Implement a simulator/emulator path for concrete `micro.kernel` on a specific target, starting with Intel CPU with AVX2. The first version is a performance simulator, not a functional tensor emulator. It estimates latency, utilization, bandwidth pressure, and bottlenecks from `micro` plus `machines/x86-avx2-cpu.yaml`.

The simulator consumes tile-centric Micro-IR. It must derive events from tile shape, dtype, layout, memory space, owner, and fragment metadata rather than treating generic op names as enough information.

Functional emulation can be added later for debugging, but it is not required for the first target.

---

## 2. Definitions

Simulator:

```text
Consumes concrete micro IR and MachineModel, produces predicted timing
and resource metrics without executing tensor math.
```

Emulator:

```text
Consumes concrete micro IR and input tensors, executes equivalent tensor
math on a host implementation, and produces output tensors.
```

MVP scope:

```text
simulator only
```

The word "emulator" in CLI/help text should be reserved for future functional execution to avoid confusing users.

---

## 3. Repository Files

Create:

```text
include/LLK/Perf/MachineModel.h
include/LLK/Perf/MachineModelLoader.h
include/LLK/Perf/MicroDAG.h
include/LLK/Perf/MicroCostModel.h
include/LLK/Perf/MicroPerfReport.h

lib/Perf/MachineModel.cpp
lib/Perf/MachineModelLoader.cpp
lib/Perf/MicroDAG.cpp
lib/Perf/MicroCostModel.cpp
lib/Perf/MicroPerfReport.cpp

machines/x86-avx2-cpu.yaml
machines/generic-ai-accel-v1.yaml

tools/micro-perf/micro-perf.cpp

test/Perf/machine_model_loader.cpp
test/Perf/l0_static_bound.cpp
test/Perf/l1_resource_dag.cpp
test/Perf/micro_perf_cli.mlir
```

Modify:

```text
CMakeLists.txt
```

---

## 4. Machine Model for AVX2 CPU

`machines/x86-avx2-cpu.yaml`:

```yaml
schema_version: 1
name: x86-avx2-cpu
clock_hz: 3000000000

compute:
  owners:
    worker:
      count: 8
      maps_to: worker_threads
    lane:
      count: 8
      maps_to: avx2_lanes
    vector_engine:
      count: 8
      maps_to: avx2-vector

  worker_threads:
    count: 8

  matrix_engines:
    - name: avx2-fma
      count: 8
      tile_shapes:
        - [1, 8, 8]
        - [4, 8, 8]
      input_dtypes: [f32, bf16]
      accumulator_dtypes: [f32]
      issue_cycles: 1
      latency_cycles: 5
      flops_per_cycle: 16

  vector_engines:
    - name: avx2-vector
      count: 8
      lanes:
        f32: 8
        bf16: 16
      issue_cycles: 1
      latency_cycles: 4
      supported_layouts: [row_major, vectorized]

memory:
  dram:
    capacity_bytes: 68719476736
    bandwidth_bytes_per_cycle: 32
    latency_cycles: 220

  l2:
    capacity_bytes: 1048576
    bandwidth_bytes_per_cycle: 64
    latency_cycles: 12

  sram:
    alias: l1
    capacity_bytes: 32768
    bandwidth_bytes_per_cycle: 64
    latency_cycles: 4
    banks: 1
    supported_layouts: [row_major, vectorized, blocked]

  acc:
    alias: registers
    capacity_bytes: 4096
    bandwidth_bytes_per_cycle: 256
    latency_cycles: 1
    supported_layouts: [row_major, vectorized]

dma:
  engines: 1
  max_outstanding: 1
  setup_cycles: 0
  maps_to: load_store_units

sync:
  barrier_cycles: 200
  wait_cycles: 0
```

The exact numerical defaults are calibration seeds. They are not claims about any specific CPU SKU. The machine file should document that users can copy and calibrate it for their host.

---

## 5. Machine Model Loader

### 5.1 Dependencies

Preferred YAML library:

```text
LLVM YAML I/O
```

Reason:

- LLVM is already a dependency
- avoids adding `yaml-cpp`
- integrates with LLVM diagnostics

### 5.2 C++ Types

```cpp
struct MatrixEngineModel {
  std::string name;
  uint32_t count;
  std::vector<std::array<int64_t, 3>> tileShapes;
  std::vector<std::string> inputDTypes;
  std::vector<std::string> accumulatorDTypes;
  uint64_t issueCycles;
  uint64_t latencyCycles;
  std::optional<double> flopsPerCycle;
};

struct VectorEngineModel {
  std::string name;
  uint32_t count;
  std::map<std::string, int64_t> lanes;
  uint64_t issueCycles;
  uint64_t latencyCycles;
  std::vector<std::string> supportedLayouts;
};

struct MemoryLevelModel {
  std::string name;
  std::optional<std::string> alias;
  uint64_t capacityBytes;
  double bandwidthBytesPerCycle;
  uint64_t latencyCycles;
  std::optional<uint32_t> banks;
  std::vector<std::string> supportedLayouts;
};

struct OwnerModel {
  std::string name;
  uint32_t count;
  std::optional<std::string> mapsTo;
};

struct DmaModel {
  uint32_t engines;
  uint32_t maxOutstanding;
  uint64_t setupCycles;
  std::optional<std::string> mapsTo;
};

struct SyncModel {
  uint64_t barrierCycles;
  uint64_t waitCycles;
};

struct MachineModel {
  uint32_t schemaVersion;
  std::string name;
  uint64_t clockHz;
  uint32_t workerThreads;
  std::vector<OwnerModel> owners;
  std::vector<MatrixEngineModel> matrixEngines;
  std::vector<VectorEngineModel> vectorEngines;
  llvm::StringMap<MemoryLevelModel> memory;
  DmaModel dma;
  SyncModel sync;
};
```

### 5.3 Validation

Reject:

- missing `schema_version`
- unsupported schema version
- empty name
- zero clock
- zero engine counts
- non-positive bandwidth
- zero capacity for modeled memory
- matrix tile shapes not length 3
- unknown dtype names
- unknown owner names
- unsupported layout names
- missing required memory spaces used by a concrete kernel

Return `llvm::Expected<MachineModel>`.

---

## 6. Micro DAG

### 6.1 Event Types

```cpp
enum class EventKind {
  TileView,
  TilePartition,
  AsyncCopy,
  Load,
  Store,
  Mma,
  Vector,
  Reduce,
  Wait,
  Barrier
};
```

### 6.2 Resource Types

```cpp
enum class ResourceKind {
  Dma,
  MatrixEngine,
  VectorEngine,
  MemoryRead,
  MemoryWrite,
  Sync
};
```

### 6.3 Event Node

```cpp
struct MicroEvent {
  uint32_t id;
  EventKind kind;
  ResourceKind resource;
  std::string resourceName;
  uint64_t workItems;
  uint64_t bytes;
  uint64_t minCycles;
  std::vector<uint32_t> deps;
  std::string sourceOpName;
  std::string tileShape;
  std::string tileLayout;
  std::string tileMemory;
  std::string tileOwner;
};
```

### 6.4 DAG Builder

Inputs:

- concrete `micro.kernel`
- machine model

Algorithm:

```text
walk concrete micro kernel in block order
track async token producer event ids
for each op:
  read tile shape, dtype, layout, memory, owner, and fragment metadata
  create event with resource and estimated work
  add dependency on prior event in same sequential region
  for wait:
    add dependencies on token producer events
  for pipeline:
    mark overlap window using pipeline stage count
return MicroDAG
```

MVP dependency model:

- logical tile views and partitions create metadata nodes only when needed for dependency tracing; they have zero cycles unless they imply a physical layout transform
- preserve sequential order inside `micro.for`
- allow async copies before `micro.wait` to overlap
- model `pipeline stages > 1` by allowing copy event for iteration i+1 to overlap with MMA for iteration i

This is analytical, not cycle-accurate.

---

## 7. L0 Static Bound

Inputs:

- concrete `micro.kernel`
- machine model

Outputs:

```cpp
struct L0Report {
  uint64_t totalFlops;
  uint64_t totalBytesDram;
  uint64_t totalBytesSram;
  llvm::StringMap<uint64_t> liveTileBytesByMemory;
  uint64_t computeCyclesLowerBound;
  uint64_t memoryCyclesLowerBound;
  uint64_t predictedCycles;
  std::string bottleneck;
};
```

Calculations:

### 7.1 `micro.mma`

For `micro.mma` or `micro.tile_mma` shape `[m, n, k]`:

```text
flops = 2 * m * n * k
```

Multiply by loop trip counts if statically known.

For AVX2:

```text
cycles = ceil(flops / (worker_threads * flops_per_cycle_per_worker))
```

### 7.2 Tile Movement Ops

For `micro.tile_async_copy`, `micro.async_copy`, `micro.tile_load`, `micro.load`, `micro.tile_store`, and `micro.store`:

```text
cycles = latency(src_or_dst_memory) + ceil(bytes / bandwidth_bytes_per_cycle)
```

Bytes are derived from tile shape, dtype, layout, and padding. L0 accumulates bytes by memory space and computes memory lower bound by dominant memory path.

### 7.3 Logical Tile Ops

`micro.tile_view`, `micro.tile_slice`, and logical `micro.tile_partition` contribute zero cycles. They may affect:

- downstream byte counts
- downstream fragment compatibility
- live range attribution
- layout metadata

Physical layout transforms are not zero-cost and must be represented as copy, permute, vector, or target-specific transform events.

### 7.4 Prediction

```text
predictedCycles = max(computeCyclesLowerBound, memoryCyclesLowerBound)
```

This is intentionally optimistic.

---

## 8. L1 Resource DAG Scheduler

Inputs:

- `MicroDAG`
- `MachineModel`

Outputs:

```cpp
struct L1Report {
  uint64_t predictedCycles;
  double predictedNs;
  double matrixUtilization;
  double vectorUtilization;
  double dmaUtilization;
  double dramBandwidthUtilization;
  double sramBandwidthUtilization;
  std::string bottleneck;
  std::vector<std::string> capacityViolations;
  std::vector<std::string> layoutWarnings;
  std::vector<std::string> ownerWarnings;
};
```

### 8.1 Scheduling Algorithm

Use a deterministic list scheduler:

```text
ready = events with no unscheduled deps
time = 0
while unscheduled events remain:
  choose ready event with lowest original id
  resource = event.resource
  start = max(dep_finish_times, resource_next_available[resource])
  finish = start + event.minCycles
  record start/finish
  resource_next_available[resource] = finish
  add newly ready dependents
predictedCycles = max finish
```

Resource multiplicity:

- matrix engine count allows that many concurrent matrix events
- vector engine count allows that many concurrent vector events
- DMA engines allow that many concurrent copy events
- owner count bounds concurrent execution tiles mapped to that owner
- worker thread count scales AVX2 matrix/vector throughput but does not create separate events in MVP

### 8.2 Pipeline Overlap

For a `micro.pipeline stages = S` inside a reduction loop:

- copy events for up to `S - 1` future iterations can be ready before current compute finishes
- wait events constrain use of a specific tile
- the MVP may approximate this by reducing sequential copy+compute sum to resource-scheduled copy and compute streams with token dependencies

The report must include:

```text
overlap_efficiency = 1 - (predicted_cycles / sequential_copy_plus_compute_cycles)
```

Clamp to `[0, 1]`.

### 8.3 Bottleneck Classification

Compute utilization:

```text
busy_cycles(resource) / (predicted_cycles * resource_count)
```

Bottleneck is the resource with highest utilization above 0.75. If none exceeds 0.75, classify by critical path resource.

Supported bottleneck strings:

```text
matrix_engine
vector_engine
dma
dram_bandwidth
sram_bandwidth
owner_occupancy
layout_transform
sync
dependency
unknown
```

---

## 9. Capacity Checks

Before scheduling, compute peak live allocation per memory space.

MVP:

- logical tile views do not count against capacity
- all `micro.alloc` in a loop body are assumed live for the whole loop body
- `micro.tile_alloc` and `micro.alloc` contribute materialized tile bytes
- `micro.tile_async_copy` and `micro.async_copy` result tiles are live until the next `micro.wait` and consuming compute event
- pipeline `stages` multiplies tile storage by stage count for SRAM copies
- physical layout transforms contribute destination tile bytes while the transform is live

Report capacity violations:

```text
memory sram requires 65536 bytes but machine has 32768 bytes
memory acc requires 8192 bytes but machine has 4096 bytes
```

Capacity violations do not crash `micro-perf`; they mark the candidate illegal for tuning.

---

## 10. `micro-perf` CLI

```bash
micro-perf \
  --machine=machines/x86-avx2-cpu.yaml \
  --level=1 \
  --format=yaml \
  input.micro.mlir
```

Options:

```text
--machine=<path>
--level=0|1
--format=yaml|text
--kernel=<symbol>
--fail-on-capacity-violation
```

Default:

```text
--level=1
--format=yaml
```

YAML output:

```yaml
schema_version: 1
machine: x86-avx2-cpu
kernel: swiglu_candidate_17
level: 1
predicted_cycles: 123456
predicted_ns: 41152.0
utilization:
  matrix: 0.82
  vector: 0.12
  dma: 0.61
bandwidth:
  dram: 0.58
  sram: 0.33
tiles:
  live_bytes:
    sram: 32768
    acc: 4096
  layouts:
    input: row_major
    weights: blocked_16x16
  owners:
    outer: worker
    fragment: vector_engine
bottleneck: matrix_engine
capacity_violations: []
```

---

## 11. CMake Integration

Add library:

```cmake
add_library(LLKPerf STATIC
    lib/Perf/MachineModel.cpp
    lib/Perf/MachineModelLoader.cpp
    lib/Perf/MicroDAG.cpp
    lib/Perf/MicroCostModel.cpp
    lib/Perf/MicroPerfReport.cpp
)

target_link_libraries(LLKPerf PUBLIC
    MicroDialect
    LLVMSupport
    MLIRIR
    MLIRParser
)
```

Add tool:

```cmake
add_llvm_executable(micro-perf tools/micro-perf/micro-perf.cpp)
target_link_libraries(micro-perf PRIVATE LLKPerf MicroDialect ${LLK_MLIR_EXEC_LIBS})
```

---

## 12. Tests

### 12.1 Machine Model Loader

`test/Perf/machine_model_loader.cpp`

Cases:

- valid AVX2 YAML loads
- missing clock rejected
- invalid dtype rejected
- invalid tile shape rejected
- invalid owner rejected
- invalid layout rejected
- zero bandwidth rejected

### 12.2 L0 Static Bound

`test/Perf/l0_static_bound.cpp`

Use a tiny micro GEMM:

```text
M=16, N=16, K=32
one MMA
two async copies
one store
```

Assert:

- flops = 16384
- DRAM bytes equals A+B+C tile bytes
- live tile bytes by memory are reported
- predicted cycles is max of compute/memory lower bounds

### 12.3 L1 Resource DAG

`test/Perf/l1_resource_dag.cpp`

Cases:

- copy -> wait -> mma produces ordered schedule
- independent copies can overlap when DMA engines > 1
- pipeline stages reduce predicted cycles compared with sequential model
- logical tile views have zero cycles
- owner resource limits cap concurrent execution tiles
- bottleneck classification is stable

### 12.4 CLI FileCheck

`test/Perf/micro_perf_cli.mlir`

Run:

```bash
micro-perf --machine=machines/x86-avx2-cpu.yaml --level=1 \
  test/Perf/micro_perf_cli.mlir | FileCheck test/Perf/micro_perf_cli.mlir
```

Checks:

- `predicted_cycles`
- `matrix`
- `dma`
- `tiles`
- `bottleneck`

---

## 13. Calibration Path

Initial AVX2 calibration should be file-based and explicit:

```yaml
schema_version: 1
machine: x86-avx2-cpu
calibration:
  matrix_engine_scale: 0.72
  dram_bandwidth_scale: 0.58
  sram_bandwidth_scale: 0.80
```

The simulator applies scale factors only when a calibration file is provided.

CLI:

```text
--calibration=machines/x86-avx2-cpu-calibration.yaml
```

Do not silently learn or overwrite machine YAML during `micro-perf`.

---

## 14. Future Functional Emulator

Functional emulation can be added after L1 performance modeling.

Scope:

- execute `micro.tile_mma`, `micro.mma`, `micro.tile_vector`, `micro.vector`, `micro.tile_reduce`, `micro.reduce`, and memory ops on host tensors
- compare outputs against existing C++ reference kernels
- debug lowering correctness independently from performance estimates

It should be a separate tool or explicit mode:

```text
micro-emulate
```

Do not mix functional emulation with `micro-perf` default behavior.

---

## 15. Implementation Order

1. Add `MachineModel` structs and YAML loader.
2. Add AVX2 and generic machine YAML files.
3. Add L0 static bound cost model.
4. Add MicroDAG event extraction.
5. Add L1 deterministic list scheduler.
6. Add utilization, bandwidth, bottleneck, and capacity reporting.
7. Add `micro-perf` CLI.
8. Add GTest and FileCheck coverage.
9. Add optional calibration scale-file support.

---

## 16. Acceptance Criteria

- `machines/x86-avx2-cpu.yaml` validates.
- `micro-perf` runs on a concrete micro GEMM/SwiGLU example.
- L0 reports operation counts, bytes, and a roofline-style bound.
- L1 reports predicted cycles, utilization, bandwidth pressure, bottleneck, and capacity violations.
- tests cover loader validation, L0 math, L1 scheduling, pipeline overlap, and CLI output.
- no functional tensor emulation is implied by the MVP simulator.
