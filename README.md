# DSLCompiler - MLIR-Based DNN Micro-IR for AI Chipset Evaluation

[![CI](https://github.com/skg7on/DSLCompiler/actions/workflows/ci.yml/badge.svg)](https://github.com/skg7on/DSLCompiler/actions/workflows/ci.yml)
[![Coverage](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/skg7on/DSLCompiler/main/badges/coverage.json)](https://github.com/skg7on/DSLCompiler/actions/workflows/coverage.yml)

DSLCompiler is an out-of-tree [MLIR](https://mlir.llvm.org/) compiler project centered on a low-level canonical DNN Micro-IR. The main goal is to represent how neural-network kernels execute on real or proposed AI chipsets, then use that representation to evaluate performance, compare hardware targets, drive auto-optimization, and eventually lower into target-specific backends.

The project uses MLIR as the compiler infrastructure. High-level or structured MLIR programs lower through LLK/Linalg-style semantic and tensor layers into `micro`, a hardware-near execution IR that models compute engines, memory hierarchy, data movement, spatial mapping, pipeline overlap, synchronization, and tunable schedule choices.

Intel CPU with AVX2 is the first validation backend and machine profile. It is not the final scope of the project; it is a concrete target used to prove the Micro-IR, performance model, lowering path, and tuning workflow before adding GPU, NPU, systolic-array, and custom AI accelerator profiles.

## Project Mission

Build an AI compiler and architecture-evaluation stack around one canonical artifact:

```text
DNN Micro-IR = Tile Dataflow + Tile Mapping + Tile Schedule
Performance = F(Tile Dataflow, Mapping, Schedule, MachineModel, Calibration)
```

The project should answer:

- How does this model kernel execute on a specific AI chipset?
- Which compute engines, memory levels, DMA/copy paths, and synchronization points are used?
- What are the expected cycles, bandwidth pressure, utilization, bottlenecks, and capacity violations?
- Which legal schedule candidate should an auto-scheduler choose for this model and target?
- How can MLIR compiler pipelines lower into this execution representation and then into real backends?

## Core Concept

```text
DNN / LLM workload
        |
        v
MLIR frontend or semantic IR
        |
        v
LLK / Linalg / Tensor optimization
        |
        v
LLKToMicro lowering
        |
        v
tile-centric micro IR
uTile -> uMap -> uHW
        |
        +---------------------------+----------------------------+
        |                           |                            |
        v                           v                            v
micro.search_space            concrete micro.kernel       current backend path
auto-scheduler input          bound execution schedule    AVX2 validation path
        |                           |                            |
        v                           v                            v
candidate generation          MachineModel YAML           Vector/LLVM/ORC JIT
        |                           |
        v                           v
candidate binding             micro-perf
        |                           |
        +-------------+-------------+
                      v
           selected schedule + predicted/measured metrics
```

`LLK` and Linalg answer what tensor computation is being performed. `micro` answers how the scheduled computation executes. `MachineModel` YAML answers what the target hardware resources and costs are.

## Micro-IR Design Goal

`micro` is intentionally lower-level than LLK, StableHLO, TOSA, or Linalg. A concrete `micro.kernel` should not contain high-level semantic operators such as `attention`, `swiglu`, `rope`, or `conv2d`. Those operations must lower into hardware-near primitives:

- temporal loops and spatial mappings
- matrix/tensor engine work, such as `micro.mma`
- vector elementwise work, such as `micro.vector`
- reductions, stores, loads, and explicit memory placement
- async copies, waits, barriers, and pipeline stages
- memory spaces such as DRAM, L2, SRAM, register file, accumulator, and scratch
- target-independent resource annotations that can be interpreted by different MachineModels

The primary Micro-IR value is a tile:

```text
Tile = (shape, dtype, layout, memory_space, owner)
```

Concrete Micro-IR distinguishes logical tiles, memory tiles, execution tiles, and instruction fragments. This is the bridge between MLIR tensor programs and AI hardware: the same tile dataflow can be remapped across AVX2 workers, GPU warps, NPU cores, systolic-array PEs, DMA engines, and custom matrix engines through MachineModel YAML and backend-specific lowering.

The Micro-IR contract is:

```text
Performance = F(tile dataflow, tile mapping, tile schedule, MachineModel, calibration)
```

## Primary Use Cases

- **AI chipset performance evaluation** - estimate cycles, bandwidth pressure, utilization, bottlenecks, memory capacity pressure, and synchronization cost for a model kernel on a specific MachineModel.
- **MLIR-to-Micro lowering** - lower structured MLIR/LLK/Linalg programs into concrete `micro.kernel` or parametric `micro.search_space`.
- **AI compiler construction** - use Micro-IR as the middle/lower execution layer between semantic MLIR and target-specific backend lowering.
- **Auto-search and auto-optimization** - represent Ansor/TVM-style search spaces, generate legal candidates, bind candidates to concrete Micro-IR, and rank them using performance models and measurements.
- **Hardware architecture exploration** - compare the same concrete Micro-IR across AVX2 CPU, generic accelerator, GPU/NPU, systolic-array, or custom ASIC MachineModels.
- **Schedule persistence and feedback** - store selected schedules, predicted metrics, measured metrics, and calibration data for regression tracking.

## Initial Validation Backend

The first concrete backend/profile is Intel CPU with AVX2:

```text
fused SwiGLU:
Y = SiLU(X * Wg) * (X * Wu)

dtype:
BF16 inputs, FP32 accumulation

target:
x86-64 CPU with AVX2
```

This backend exists to validate the infrastructure:

- explicit vector lowering through MLIR Vector/LLVM
- ORC JIT execution path
- AVX2 MachineModel profile
- Micro-IR performance simulator behavior
- optional predicted-vs-measured calibration loop

Future targets should reuse the same Micro-IR contract with different MachineModel YAML profiles and backend lowerings.

## Project Status

**Main direction:** canonical DNN Micro-IR for AI chipset evaluation and MLIR-based AI compiler lowering.

**Current validation path:** LLK/Linalg to AVX2 CPU lowering and runtime execution.

**Roadmap tracker:** [#41](https://github.com/skg7on/DSLCompiler/issues/41)

| Milestone | Status | Scope |
|-----------|--------|-------|
| M1: Scalar end-to-end pipeline | In progress | LLK dialect, LLKToLinalg, scalar JIT foundation |
| M2: Explicit vector path | In progress | MLIR Vector lowering and AVX2 validation |
| M3: Fused memory lowering | Design approved | Double-contraction fusion, packing, scratch analysis |
| M4: Parallel execution | Design approved | Thread pool and tiled parallel dispatch |
| M5: Specialization and tuning | Design approved | Shape buckets, JIT cache, schedule DB |
| M6: Multi-kernel support | Design approved | RoPE, Attention, shared compiler infrastructure |
| M7+: Python frontend | Future | Triton-like DSL and Python entry points |
| M9: Canonical Micro-IR foundation | Planned | `micro` dialect, execution ops, search ops, verifiers |
| M10: MachineModel and performance evaluation | Planned | YAML profiles, `micro-perf`, L0/L1 simulator |
| M11: MLIR to Micro lowering | Planned | `llk-compile --emit=micro`, `--emit=micro-search` |
| M12: Micro-based auto-optimization | Planned | Candidate generation, legality, binding, ranking |
| M13: Feedback and calibration | Planned | Schedule YAML, predicted vs measured metrics |

## Documentation

| Document | Description |
|----------|-------------|
| [ARCHITECTURE.md](ARCHITECTURE.md) | Existing compiler architecture, lowering pipeline, component map, engineering rules |
| [docs/design/m1-scalar-pipeline.md](docs/design/m1-scalar-pipeline.md) | M1 design: LLK dialect, verifier, LLKToLinalg, JIT cache, ABI |
| [docs/design/m2-explicit-vector.md](docs/design/m2-explicit-vector.md) | M2 design: tiling schedule, vector dialect, AVX2 target, masked tails |
| [docs/design/m3-fused-memory.md](docs/design/m3-fused-memory.md) | M3 design: double-contraction fusion, weight packing, scratch analysis |
| [docs/design/m4-parallel-execution.md](docs/design/m4-parallel-execution.md) | M4 design: thread pool, parallel decomposition, dispatch thresholds |
| [docs/design/m5-specialization-tuning.md](docs/design/m5-specialization-tuning.md) | M5 design: M-bucketing, JIT cache, schedule DB, autotuning foundation |
| [docs/design/m6-multi-kernel.md](docs/design/m6-multi-kernel.md) | M6 design: RoPE, Attention, online softmax, shared infrastructure |
| [docs/design/m9-canonical-micro-ir-architecture.md](docs/design/m9-canonical-micro-ir-architecture.md) | M9+ architecture: canonical Micro-IR, MachineModel, performance evaluation, auto-search |
| [docs/design/m9-micro-ir-core-concepts.md](docs/design/m9-micro-ir-core-concepts.md) | Core `micro` concepts: execution IR, search IR, attributes, verifier invariants |
| [docs/superpowers/specs/2026-08-11-canonical-micro-ir-redesign.md](docs/superpowers/specs/2026-08-11-canonical-micro-ir-redesign.md) | Full canonical Micro-IR redesign spec and milestone contract |
| [docs/superpowers/specs/2026-08-13-micro-ir-tile-programming-model-spec.md](docs/superpowers/specs/2026-08-13-micro-ir-tile-programming-model-spec.md) | Tile-based programming model: `!micro.tile`, tile kinds, `uTile/uMap/uHW`, lowering, search, and performance contract |
| [docs/superpowers/specs/2026-08-11-micro-ir-dialect-implementation-spec.md](docs/superpowers/specs/2026-08-11-micro-ir-dialect-implementation-spec.md) | Detailed implementation spec for the `micro` dialect |
| [docs/superpowers/specs/2026-08-11-llk-to-micro-lowering-implementation-spec.md](docs/superpowers/specs/2026-08-11-llk-to-micro-lowering-implementation-spec.md) | Detailed implementation spec for LLK/Linalg to Micro lowering |
| [docs/superpowers/specs/2026-08-11-micro-ir-autotuning-implementation-spec.md](docs/superpowers/specs/2026-08-11-micro-ir-autotuning-implementation-spec.md) | Detailed implementation spec for Micro-based auto-tuning and auto-optimization |
| [docs/superpowers/specs/2026-08-11-micro-ir-avx2-simulator-implementation-spec.md](docs/superpowers/specs/2026-08-11-micro-ir-avx2-simulator-implementation-spec.md) | Detailed implementation spec for the AVX2 Micro performance simulator |
| [docs/superpowers/plans/](docs/superpowers/plans/) | Implementation plans and task breakdowns |

## Architecture Layers

### 1. Semantic Layer

The semantic layer describes the mathematical workload:

- `llk.fused_swiglu`
- `llk.rope`
- `llk.attention`
- `llk.matmul`
- future Python or Triton-like frontends

This layer should not encode hardware details such as SRAM size, DMA queues, PE mapping, pipeline depth, or matrix-engine tile shapes.

### 2. Structured MLIR Optimization Layer

The structured layer uses MLIR-native IR to expose legal transformations:

- Linalg/Tensor/SCF/Vector
- canonicalization
- shape specialization
- fusion
- packing annotations
- deterministic schedule selection

This keeps the compiler compatible with the MLIR ecosystem before lowering into the lower-level Micro-IR contract.

### 3. Canonical Micro-IR Layer

The Micro-IR layer materializes the execution schedule:

```text
micro.search_space
    -> micro.candidate
    -> concrete micro.kernel
```

It captures tile construction, memory placement, layout, data movement, compute fragments, synchronization, spatial mapping, temporal schedule, and pipeline structure.

The layer is split conceptually into:

- `uTile`: algorithmic tile dataflow, such as tile matmul, elementwise, reduce, broadcast, and partition.
- `uMap`: memory placement, spatial ownership, pipeline structure, copy paths, and synchronization.
- `uHW`: native execution fragments that match hardware granularity, such as MMA tiles, vector tiles, DMA transactions, barriers, and NoC transfers.

### 4. MachineModel Layer

Machine profiles are YAML files under `machines/` and are loaded into typed C++ structures. A MachineModel describes:

- compute engines and throughput
- memory hierarchy, capacity, latency, and bandwidth
- DMA/copy engines
- synchronization costs
- supported dtypes and tile shapes
- target constraints used by legality checks

Initial profiles:

- `machines/x86-avx2-cpu.yaml`
- `machines/generic-ai-accel-v1.yaml`

### 5. Evaluation and Optimization Layer

`micro-perf` and tuning tools consume concrete Micro-IR plus MachineModel YAML:

```text
concrete micro.kernel + MachineModel
    -> L0 static bound
    -> L1 resource DAG schedule
    -> predicted cycles, utilization, bandwidth pressure, bottlenecks
    -> optional measured feedback
```

The auto-optimization flow consumes `micro.search_space`, generates candidates, checks legality, binds candidates into concrete `micro.kernel`, ranks them, and persists selected schedules.

## Planned Tool Surface

```bash
# Emit concrete Micro-IR without running the JIT path
llk-compile --emit=micro input.mlir

# Emit a parametric Micro-IR search space for auto-scheduling
llk-compile --emit=micro-search input.mlir

# Evaluate concrete Micro-IR against a machine profile
micro-perf --machine machines/x86-avx2-cpu.yaml --level l1 input.micro.mlir

# Tune candidates from a Micro search space
llk-tune --search-space swiglu.micro.mlir --machine machines/x86-avx2-cpu.yaml
```

These commands document the intended M9-M13 interface. They become available as the corresponding issues land.

## Tech Stack

- **C++20**, **CMake** >= 3.20
- **MLIR/LLVM** main branch, aligned with LLVM 20+
- **MLIR dialects and passes** for semantic, structured, and Micro-IR lowering
- **Google Test** for C++ tests
- **FileCheck** for MLIR IR tests
- **YAML MachineModel profiles** for hardware target descriptions
- **PyTorch** for numerical reference implementations in tests only

## Getting Started

```bash
# Clone
git clone https://github.com/skg7on/DSLCompiler.git
cd DSLCompiler

# Build, requires LLVM/MLIR built from source
mkdir build && cd build
cmake .. -G Ninja -DLLVM_PROJECT_BUILD_DIR=/path/to/llvm-project/build
ninja

# Run tests
ninja check-llk
```

When `LLVM_PROJECT_BUILD_DIR` is set, `MLIR_DIR` and `LLVM_DIR` are inferred and system-installed LLVM/MLIR are ignored. If omitted, `find_package` searches the standard CMake prefixes.

## Engineering Rules

1. Treat Micro-IR as the core project artifact for execution, evaluation, and optimization.
2. Treat `!micro.tile` as the primary execution value; represent tile shape, dtype, layout, memory space, and owner explicitly.
3. Keep DNN semantics, structured MLIR optimization, Micro-IR tile scheduling, MachineModel cost, and target lowering separate.
4. Lower high-level DNN operators into hardware-near Micro primitives; do not leave semantic ops inside concrete `micro.kernel`.
5. Distinguish zero-cost logical tile views from materialized memory tiles and physical layout transforms.
6. Keep Micro-IR architecture-parametric; model AVX2, GPU, NPU, systolic-array, and custom accelerator targets through MachineModel data and backend-specific lowering.
7. Use verifier-backed legality before auto-tuning or performance claims.
8. Prefer deterministic schedule generation before stochastic search.
9. Measure and model memory movement, synchronization, packing, tile layout transforms, and dispatch overhead, not only compute throughput.
10. Keep the existing AVX2 path working while Micro-IR matures.
11. Persist selected schedules, predicted metrics, measured metrics, and calibration data in structured formats.
12. Use external high-performance libraries as baselines, not as substitutes for the Micro-IR contract.

## License

MIT
