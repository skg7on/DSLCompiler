# Micro-IR Auto-Tuning and Auto-Optimization Implementation Spec

**Status:** Draft
**Date:** 2026-08-11
**Parent Spec:** [Canonical Micro-IR Redesign Spec](2026-08-11-canonical-micro-ir-redesign.md)

---

## 1. Goal

Implement an auto-tuning framework based on `micro.search_space`, `micro.candidate`, concrete `micro.kernel`, `MachineModel` YAML, and `micro-perf`. The framework should evolve the current `llk-tune` grid-search tool into a micro-IR based schedule search engine.

The tuning framework is tile-centric. It searches tile hierarchy, tile layout, memory placement, owner mapping, pipeline structure, and instruction-fragment binding, not only loop tile sizes.

The first target is Intel CPU with AVX2. The design must remain generic enough for future accelerator profiles.

---

## 2. Existing Starting Point

Current `tools/llk-tune/llk-tune.cpp`:

- generates a nested grid over `BM`, `BN`, `BK`, `VM`, `VN`, `num_threads`, and `grain_size`
- applies an L1 footprint filter
- writes top schedules to JSON
- records measured GFLOPS metadata

The new framework keeps that simple grid as the first candidate generator, but moves legality, evaluation, and persistence onto the `micro` abstraction.

The first grid remains useful, but each candidate must eventually bind:

```text
uTile choices: BM, BN, BK, fusion, reduction tile shape
uMap choices: layout, memory placement, owner mapping, pipeline stages
uHW choices: MMA/vector/reduction fragment shape and target compute binding
```

---

## 3. Architecture

```text
LLK/Linalg workload
    |
    v
LLKToMicroSearchSpace
    |
    v
micro.search_space
    |
    v
SearchSpaceLoader
    |
    v
CandidateGenerator
    |
    v
CandidateBinding
    |
    v
LegalityVerifier + MachineModel
    |
    v
micro-perf L0/L1
    |
    v
Top-K candidate set
    |
    v
optional AVX2 compile and measurement
    |
    v
Schedule YAML + calibration records
```

---

## 4. Repository Files

Create:

```text
include/LLK/Perf/SearchSpace.h
include/LLK/Perf/Candidate.h
include/LLK/Perf/CandidateGenerator.h
include/LLK/Perf/CandidateBinding.h
include/LLK/Perf/Legality.h
include/LLK/Perf/TuningSession.h
include/LLK/Perf/ScheduleRecord.h

lib/Perf/SearchSpace.cpp
lib/Perf/Candidate.cpp
lib/Perf/CandidateGenerator.cpp
lib/Perf/CandidateBinding.cpp
lib/Perf/Legality.cpp
lib/Perf/TuningSession.cpp
lib/Perf/ScheduleRecord.cpp

test/Perf/search_space_loader.cpp
test/Perf/candidate_generator.cpp
test/Perf/candidate_binding.cpp
test/Perf/legality.cpp
test/Perf/tuning_session.cpp
```

Modify:

```text
tools/llk-tune/llk-tune.cpp
CMakeLists.txt
```

Optional later:

```text
tools/micro-tune/micro-tune.cpp
```

MVP should evolve `llk-tune` first to reuse the existing tool name.

---

## 5. Core Data Structures

### 5.1 `SearchParam`

```cpp
struct SearchChoice {
  std::variant<int64_t, std::string> value;
};

struct SearchParam {
  std::string name;
  std::vector<SearchChoice> choices;
};
```

Rules:

- `choices` must be non-empty
- choices must be unique after sorting
- integer choices are used for tile sizes, loop bounds, pipeline stages, and vector widths
- string choices are used for layout, owner mapping, memory path, fragment shape, and policy names

### 5.2 `SearchConstraint`

```cpp
enum class ConstraintKind {
  SramCapacity,
  AccCapacity,
  MmaCompatible,
  MappingExtent,
  TailSupported,
  VectorWidthSupported,
  TileHierarchyCompatible,
  LayoutSupported,
  OwnerSupported,
  FragmentCompatible,
  PipelineLiveTiles
};

struct SearchConstraint {
  ConstraintKind kind;
  std::vector<std::string> params;
  std::map<std::string, std::string> attrs;
};
```

The first implementation uses typed constraint names from `micro.constraint`; it does not evaluate arbitrary expressions.

### 5.3 `SearchObjective`

```cpp
enum class ObjectiveDirection { Minimize, Maximize };

struct SearchObjective {
  ObjectiveDirection direction;
  std::string primaryMetric;
  std::vector<std::string> secondaryMetrics;
};
```

Supported MVP metrics:

```text
latency_cycles
dram_bytes
sram_bytes
matrix_utilization
dma_utilization
capacity_spill_bytes
```

### 5.4 `SearchSpace`

```cpp
struct SearchSpace {
  std::string name;
  std::string workload;
  std::vector<SearchParam> params;
  std::vector<SearchConstraint> constraints;
  SearchObjective objective;
};
```

### 5.5 `Candidate`

```cpp
struct Candidate {
  std::string id;
  std::map<std::string, int64_t> values;
  std::map<std::string, std::string> symbolicValues;
};
```

Candidate IDs should be stable:

```text
candidate_<64-bit-hash-of-sorted-numeric-and-symbolic-bindings>
```

String-valued candidate choices are required for layout, owner mapping, memory path, and fragment-shape names. Store those in `symbolicValues` so integer parameter handling remains simple.

### 5.6 `CandidateMetrics`

```cpp
struct CandidateMetrics {
  uint64_t predictedCycles{0};
  double predictedNs{0.0};
  double matrixUtilization{0.0};
  double dmaUtilization{0.0};
  uint64_t dramBytes{0};
  uint64_t sramBytes{0};
  std::string bottleneck;
  std::optional<double> measuredNs;
  std::optional<double> measuredGflops;
};
```

### 5.7 `TuningResult`

```cpp
struct TuningResult {
  Candidate candidate;
  CandidateMetrics metrics;
  bool legal{false};
  std::string rejectionReason;
};
```

---

## 6. Candidate Generation Algorithms

### 6.1 Grid Search

MVP algorithm:

```text
input: SearchSpace
output: all Cartesian-product candidates, optionally capped

candidates = [{}]
for param in params:
  next = []
  for candidate in candidates:
    for value in param.choices:
      next.push(candidate + {param.name = value})
  candidates = next
return candidates
```

Pruning:

- cheap static constraint checks run as early as possible
- if `--max-candidates=N`, stop after N accepted candidates

Grid search is deterministic and best for FileCheck/GTest validation.

### 6.2 Random Search

Add after grid search:

```text
for i in 0..max_trials:
  sample one value per param using seeded RNG
  reject duplicates
  run legality and performance
```

CLI must take a seed:

```text
--search=random --seed=1234
```

### 6.3 Cost-Model Guided Search

Add after L0/L1 are stable:

```text
generate broad random/grid sample
evaluate with L0
keep top K
evaluate top K with L1
optionally measure top M
fit correction factor by workload/machine
iterate
```

This is not a learned model in the MVP. It is a staged pruning strategy.

### 6.4 Evolutionary Search

Deferred until the candidate record format and metrics are stable.

Mutation dimensions:

- tile sizes
- multi-level tile hierarchy
- tile layouts
- owner mappings
- memory placements
- instruction fragment shapes
- pipeline stages
- vector width
- thread count
- parallel axis
- grain size

---

## 7. Legality Algorithm

Inputs:

- `SearchSpace`
- `Candidate`
- workload shape
- `MachineModel`

Output:

```cpp
struct LegalityResult {
  bool legal;
  std::string reason;
};
```

Algorithm:

```text
for each constraint:
  dispatch by ConstraintKind
  read candidate param values
  read machine resource values
  compute legality condition
  if false, return illegal with stable reason
return legal
```

Constraint behavior:

### 7.1 `SramCapacity`

For fused SwiGLU:

```text
x_bytes = BM * BK * bytes(input_dtype)
wg_bytes = BK * BN * bytes(weight_dtype)
wu_bytes = BK * BN * bytes(weight_dtype)
double_buffer_factor = pipeline_stages > 1 ? 2 : 1
required = double_buffer_factor * (x_bytes + wg_bytes + wu_bytes)
legal if required <= machine.memory.sram.capacity_bytes * sram_utilization_limit
```

MVP `sram_utilization_limit`:

```text
0.80
```

### 7.2 `AccCapacity`

For fused SwiGLU:

```text
gate_acc = BM * BN * bytes(acc_dtype)
up_acc = BM * BN * bytes(acc_dtype)
required = gate_acc + up_acc
legal if required <= machine.memory.acc.capacity_bytes
```

### 7.3 `MmaCompatible`

Legal if:

- machine has a matrix engine supporting requested input and accumulator dtypes
- requested `mma_shape` appears in supported tile shapes
- `BM`, `BN`, `BK` are multiples of the MMA shape or tail handling is enabled

### 7.4 `MappingExtent`

For CPU AVX2:

```text
num_threads <= machine.compute.worker_threads.count
spatial map worker extent <= num_threads
```

For generic accelerator:

```text
spatial_m * spatial_n <= matrix_engine.count or core count
```

### 7.5 `TailSupported`

MVP:

- static shapes can have tails if the generated concrete `micro.kernel` records `tail_policy = "mask"`
- if no tail policy exists, dimensions must divide tile sizes

### 7.6 `VectorWidthSupported`

For AVX2:

```text
f32 vector_width <= 8
bf16 logical vector_width <= 16 for storage, but dot accumulation maps to f32 lanes
```

The initial BF16 path should use the existing project convention `vector_width = 8`.

### 7.7 `TileHierarchyCompatible`

Legal if every child tile divides its parent tile or has explicit mask/tail metadata.

Example:

```text
worker_tile = [BM, BN, BK]
fragment = [fm, fn, fk]
legal if BM % fm == 0 and BN % fn == 0 and BK % fk == 0
or tail_policy == "mask"
```

### 7.8 `LayoutSupported`

Legal if:

- every materialized tile layout is supported by the target memory space
- every compute fragment layout is supported by the selected compute engine
- physical layout transforms fit in the selected memory and pipeline budget
- alignment requirements from the layout are satisfiable

### 7.9 `OwnerSupported`

Legal if:

- every selected owner exists in the MachineModel
- mapped owner extent is within resource count
- nested owner scopes are compatible
- a tile consumed by another owner has an explicit transfer, exchange, or replication step

### 7.10 `FragmentCompatible`

Legal if:

- selected fragment shape is supported by at least one target compute engine
- fragment dtype and accumulator dtype are supported
- fragment shape divides or masks the execution tile
- the selected owner can issue the fragment

### 7.11 `PipelineLiveTiles`

Legal if:

```text
live_materialized_bytes(stage_count, tile_shapes, layouts, memory_spaces)
  <= memory_capacity * utilization_limit
```

Also check async queue limits:

```text
outstanding_async_copies <= machine.dma.max_outstanding
```

---

## 8. Candidate Binding

Candidate binding converts:

```text
micro.search_space + Candidate -> concrete micro.kernel
```

Inputs:

- MLIR module containing one `micro.search_space`
- candidate values
- workload metadata
- schedule template

Outputs:

- MLIR module containing one concrete `micro.kernel`

MVP strategy:

Candidate binding must fill both numeric and symbolic tile decisions:

```text
BM/BN/BK                  -> loop steps and logical tile shapes
tile_layout               -> #micro.layout on tile views/materialized tiles
memory_path               -> tile copy/load/store spaces
owner_mapping             -> #micro.owner and micro.spatial_for maps
fragment_shape            -> micro.tile_partition and micro.tile_mma fragment metadata
pipeline_stages           -> micro.pipeline stages and live tile instances
vector_width              -> micro.tile_vector metadata
```

The output must contain no `micro.param`, `micro.constraint`, `micro.objective`, or unresolved symbolic choice.

Use C++ builders to emit a concrete kernel directly from candidate values. Do not try to textual-substitute MLIR.

API:

```cpp
mlir::LogicalResult bindCandidateToMicroKernel(
    mlir::ModuleOp module,
    const SearchSpace &space,
    const Candidate &candidate,
    const WorkloadShape &shape,
    mlir::OpBuilder &builder);
```

---

## 9. Performance Ranking

Ranking key:

```text
primary: lower predictedCycles
secondary:
  higher matrixUtilization
  lower dramBytes
  lower capacity_spill_bytes
tie-breaker:
  stable candidate id
```

The ranking must be deterministic.

---

## 10. Measurement Loop on AVX2

MVP measurement is optional and controlled by:

```text
--measure
```

Algorithm:

```text
take top K predicted candidates
for each candidate:
  bind candidate to schedule metadata
  run existing compile/JIT benchmark path if available
  collect median runtime over repeat count
  attach measured_ns and measured_gflops
write schedule YAML
```

CLI options:

```text
--measure
--measure-top-k=5
--warmup=5
--repeat=30
```

Use median, not mean, as the primary measured latency to reduce outlier sensitivity.

---

## 11. CLI Design

Evolve `llk-tune`:

```bash
llk-tune \
  --input kernel.mlir \
  --workload=fused_swiglu \
  --M=128 --N=4096 --K=4096 \
  --machine=machines/x86-avx2-cpu.yaml \
  --search=grid \
  --perf-level=1 \
  --top-k=10 \
  --output=schedules/selected/fused_swiglu_x86_avx2.yaml
```

Search modes:

```text
grid
random
staged
```

Performance levels:

```text
0
1
```

The tool should print a concise summary and write full YAML.

---

## 12. Schedule YAML Output

```yaml
schema_version: 1
workload: fused_swiglu
target: x86-avx2-cpu
machine: machines/x86-avx2-cpu.yaml
shape:
  M_bucket: 4
  M: 128
  N: 4096
  K: 4096
dtype:
  input: bf16
  output: bf16
  accumulator: f32
candidate:
  id: candidate_2db9f7c4
  values:
    BM: 32
    BN: 64
    BK: 64
    pipeline_stages: 1
    vector_width: 8
    num_threads: 8
    grain_size: 4
  tile:
    hierarchy:
      worker: [32, 64, 64]
      vector_engine_fragment: [16, 16, 32]
    layout:
      input: row_major
      weights: blocked_16x16
      accumulator: row_major
    memory_path: [dram, sram, acc]
    owner_mapping:
      outer: worker
      fragment: vector_engine
    tail_policy: mask
metrics:
  perf_level: 1
  predicted_cycles: 123456
  predicted_ns: 123456.0
  matrix_utilization: 0.82
  dma_utilization: 0.61
  dram_bytes: 67108864
  sram_bytes: 134217728
  bottleneck: matrix_engine
measurement:
  measured: false
```

If measured:

```yaml
measurement:
  measured: true
  warmup: 5
  repeat: 30
  median_ns: 98765.0
  measured_gflops: 540.0
```

---

## 13. Tests

### 13.1 Search Space Loader

Input:

- `micro.search_space` with params and constraints

Assert:

- param names and choices are parsed
- string-valued choices for layout, owner, and fragment shape are parsed
- constraints are typed
- objective is parsed

### 13.2 Candidate Generator

Input:

- small space with 2 params and 2 choices each

Assert:

- grid emits 4 candidates
- order is deterministic
- `--max-candidates=3` emits 3 candidates

### 13.3 Legality

Cases:

- legal candidate
- SRAM overflow
- ACC overflow
- unsupported MMA shape
- unsupported vector width
- unsupported tile layout
- unsupported owner mapping
- incompatible fragment shape
- pipeline live-tile overflow

### 13.4 Candidate Binding

Input:

- search space plus candidate

Assert:

- concrete `micro.kernel` is emitted
- no search ops remain in output
- bound tile sizes appear in loop steps and tensor shapes
- layout, memory, owner, and fragment decisions appear in tile metadata

### 13.5 Tuning Session

Input:

- small search space
- fake machine model
- deterministic fake performance model

Assert:

- top candidate is selected
- rejected candidates include stable reasons
- output YAML matches expected fields

---

## 14. Implementation Order

1. Add typed C++ search data structures.
2. Add symbolic candidate values for layout, owner, memory path, and fragment choices.
3. Parse `micro.search_space` into `SearchSpace`.
4. Implement deterministic grid candidate generation.
5. Implement tile hierarchy, layout, owner, fragment, and live-tile legality checks against `MachineModel`.
6. Implement candidate binding to tile-centric concrete `micro.kernel`.
7. Connect bound kernels to `micro-perf` L0/L1 APIs.
8. Evolve `llk-tune` CLI to use the new flow.
9. Add YAML schedule output with tile decisions.
10. Add optional AVX2 measurement loop.

---

## 15. Acceptance Criteria

- `llk-tune` can load a micro search space and machine YAML.
- grid search generates deterministic candidates.
- legality checks reject invalid candidates with stable reason strings.
- top candidates can be ranked by L0 or L1 predicted cycles.
- selected schedule YAML is written with tile hierarchy, layout, owner mapping, memory placement, pipeline, and fragment decisions.
- optional AVX2 measurement can annotate top candidates without changing the default compile path.
