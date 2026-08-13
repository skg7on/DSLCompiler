# DSLCompiler - Domain-Specific LLM Kernel Compiler

[![CI](https://github.com/skg7on/DSLCompiler/actions/workflows/ci.yml/badge.svg)](https://github.com/skg7on/DSLCompiler/actions/workflows/ci.yml)
[![Coverage](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/skg7on/DSLCompiler/main/badges/coverage.json)](https://github.com/skg7on/DSLCompiler/actions/workflows/coverage.yml)

An out-of-tree [MLIR](https://mlir.llvm.org/) compiler for computationally dense LLM kernels such as SwiGLU, RoPE, and Attention.

The current implementation targets high-performance CPU kernels through LLK -> Linalg -> Vector -> LLVM lowering, with explicit AVX2 SIMD and an ORC JIT runtime. The project is now evolving toward a canonical DNN Micro-IR: a lower-level execution IR that sits below LLK/Linalg, is closer to AI hardware, and can drive performance evaluation, schedule search, and future target lowerings.

## Core Concept

```
LLM-domain semantics             <- "what" the model computes
        |
        v
LLK semantic dialect             <- fused_swiglu, rope, attention
        |
        v
structured tensor computation    <- Linalg/Tensor/Arith/Math
        |
        v
schedule selection and fusion    <- tile, pack, specialize, fuse
        |
        +--------------------------+---------------------------+
        |                          |                           |
        v                          v                           v
current AVX2 path          micro.search_space           concrete micro.kernel
Vector/SCF/MemRef/LLVM     auto-search input            hardware-near schedule
        |                          |                           |
        v                          v                           v
ORC JIT machine code       candidate binding            MachineModel + micro-perf
                                                       schedule metrics + feedback
```

`llk` answers what is computed. `micro` answers how a scheduled tensor program executes. `MachineModel` YAML answers what the target costs are.

## Targets

Current execution baseline:

`Y = SiLU(X * Wg) * (X * Wu)` -- fused SwiGLU with BF16 inputs, FP32 accumulation, x86-64 AVX2.

Micro-IR MVP target:

- Canonical `micro` dialect for concrete execution schedules and search spaces.
- Intel CPU with AVX2 as the first validated machine profile.
- YAML `MachineModel` profiles for AVX2 CPU and a generic AI accelerator.
- Performance simulator first; functional Micro emulation is a future debugging feature.

## Project Status

**Current implementation phase:** AVX2 CPU compiler/runtime pipeline.

**M9+ design phase:** canonical Micro-IR, MachineModel, performance evaluation, and auto-tuning roadmap. Planning issue: [#41](https://github.com/skg7on/DSLCompiler/issues/41).

| Milestone | Status | Scope |
|-----------|--------|-------|
| M1: Scalar end-to-end pipeline | In progress | LLK dialect, LLKToLinalg, scalar JIT |
| M2: Explicit vector path (AVX2) | In progress | Vector dialect lowering, AVX2 target path |
| M3: Fused memory lowering | Design approved | Double-contraction fusion, packing, scratch analysis |
| M4: Parallel execution | Design approved | Thread pool and tiled parallel dispatch |
| M5: Specialization and tuning | Design approved | Shape buckets, JIT cache, schedule DB |
| M6: Multi-kernel support | Design approved | RoPE, Attention, shared infrastructure |
| M7+: Python frontend | Future | Triton-like DSL and Python entry points |
| M9: Canonical Micro-IR foundation | Planned | `micro` dialect, concrete execution ops, search ops |
| M10: MachineModel and perf L0/L1 | Planned | YAML machine profiles, `micro-perf`, AVX2 simulator |
| M11: LLK/Linalg to Micro export | Planned | `llk-compile --emit=micro`, `--emit=micro-search` |
| M12: Micro-based auto-tuning | Planned | Candidate generation, legality, binding, ranking |
| M13: Feedback and calibration | Planned | Predicted vs measured AVX2 metrics, schedule YAML |

## Documentation

| Document | Description |
|----------|-------------|
| [ARCHITECTURE.md](ARCHITECTURE.md) | High-level architecture, 7-stage lowering pipeline, component map, engineering rules |
| [docs/design/m1-scalar-pipeline.md](docs/design/m1-scalar-pipeline.md) | M1 design: LLK dialect, verifier, LLKToLinalg, JIT cache, ABI |
| [docs/design/m2-explicit-vector.md](docs/design/m2-explicit-vector.md) | M2 design: tiling schedule, vector dialect, AVX2 target, masked tails |
| [docs/design/m3-fused-memory.md](docs/design/m3-fused-memory.md) | M3 design: double-contraction fusion, weight packing, scratch analysis |
| [docs/design/m4-parallel-execution.md](docs/design/m4-parallel-execution.md) | M4 design: thread pool, parallel decomposition, dispatch thresholds |
| [docs/design/m5-specialization-tuning.md](docs/design/m5-specialization-tuning.md) | M5 design: M-bucketing, 4-level JIT cache, schedule DB, autotuning |
| [docs/design/m6-multi-kernel.md](docs/design/m6-multi-kernel.md) | M6 design: RoPE (shuffles), Attention (online softmax), shared infra |
| [docs/design/m9-canonical-micro-ir-architecture.md](docs/design/m9-canonical-micro-ir-architecture.md) | M9+ architecture: canonical Micro-IR, MachineModel, performance evaluation, auto-search |
| [docs/design/m9-micro-ir-core-concepts.md](docs/design/m9-micro-ir-core-concepts.md) | Core concepts for `micro`: execution IR, search IR, attributes, verifier invariants |
| [docs/superpowers/specs/2026-08-11-canonical-micro-ir-redesign.md](docs/superpowers/specs/2026-08-11-canonical-micro-ir-redesign.md) | Full canonical Micro-IR redesign spec and milestone contract |
| [docs/superpowers/specs/2026-08-11-micro-ir-dialect-implementation-spec.md](docs/superpowers/specs/2026-08-11-micro-ir-dialect-implementation-spec.md) | Detailed implementation spec for the `micro` dialect |
| [docs/superpowers/specs/2026-08-11-llk-to-micro-lowering-implementation-spec.md](docs/superpowers/specs/2026-08-11-llk-to-micro-lowering-implementation-spec.md) | Detailed implementation spec for LLK/Linalg to Micro lowering |
| [docs/superpowers/specs/2026-08-11-micro-ir-autotuning-implementation-spec.md](docs/superpowers/specs/2026-08-11-micro-ir-autotuning-implementation-spec.md) | Detailed implementation spec for Micro-based auto-tuning and auto-optimization |
| [docs/superpowers/specs/2026-08-11-micro-ir-avx2-simulator-implementation-spec.md](docs/superpowers/specs/2026-08-11-micro-ir-avx2-simulator-implementation-spec.md) | Detailed implementation spec for the AVX2 Micro performance simulator |
| [docs/superpowers/plans/](docs/superpowers/plans/) | Implementation plans: ~29 TDD tasks with complete code |

## Architecture

### Current AVX2 Lowering Path

```
LLK Dialect  ->  Linalg Tensor IR  ->  Tiled & Fused  ->  Explicit SIMD (Vector)
      |              |                    |                    |
      v              v                    v                    v
Domain ops    Structured compute    scf.forall tiles    vector.contract
                                  Transform schedule    vector.transfer_*

      |              |                    |
      v              v                    v
Physical Memory  ->  CF + Runtime  ->  LLVM IR  ->  ORC JIT
   memref           llrt.parallel_for      Machine code
  scratch audit     persistent pool
```

This path remains the working compiler/runtime path while Micro-IR is introduced.

### M9+ Canonical Micro-IR Path

```
LLK/Linalg workload
    |
    v
LLKToMicro export
    |
    +-----------------------+
    |                       |
    v                       v
micro.search_space      concrete micro.kernel
    |                       |
    v                       v
auto-scheduler          MachineModel YAML
    |                       |
    v                       v
micro.candidate         micro-perf L0/L1
    |                       |
    v                       v
candidate binding       predicted cycles, bandwidth, utilization
    |                       |
    +-----------+-----------+
                v
       selected schedule + metrics
```

Concrete `micro.kernel` is below DNN semantics. It should contain hardware-near operations such as `micro.mma`, `micro.vector`, `micro.async_copy`, `micro.wait`, `micro.pipeline`, `micro.spatial_for`, and memory-space annotations. It should not contain high-level ops such as `attention`, `swiglu`, `rope`, or `conv2d`.

### Key Design Decisions

- **Separate semantics, execution, and cost** -- LLK/Linalg describes what is computed, `micro` describes how it executes, and MachineModel YAML describes target resources and costs.
- **Concrete Micro-IR is lower than DNN ops** -- high-level operators lower into compute, data movement, memory placement, mapping, pipeline, and synchronization.
- **Search IR and execution IR are separate** -- `micro.search_space` represents legal choices; candidate binding produces deterministic concrete `micro.kernel`.
- **Machine models are data** -- target profiles live in YAML under `machines/`, then load into typed C++ structures for validation and performance modeling.
- **Performance evaluation is first-class** -- `micro-perf` will estimate cycles, bandwidth pressure, utilization, and bottlenecks from `micro.kernel` plus MachineModel.
- **Auto-tuning uses Micro-IR** -- `llk-tune` will evolve from fixed schedule-grid generation into a Micro search engine with legality checks, ranking, optional AVX2 measurement, and YAML schedule output.
- **Preserve the current AVX2 compiler** -- Micro-IR starts as an export, analysis, and search layer before it becomes a backend lowering contract.
- **Explicit SIMD, not LLVM autovec** -- the existing CPU path emits `vector.contract` and `vector.transfer_read/write` directly.
- **Small custom semantic dialect** -- `llk` only retains domain semantics unavailable in generic MLIR; it should not recreate tensor/linalg/vector/memref.
- **Bounded specialization** -- 5 M-buckets {1}, [2,4], [5,16], [17,64], >=65; exact N, K where needed.

## Tech Stack

- **C++20**, **CMake** >= 3.20
- **MLIR/LLVM** main branch (aligned with LLVM 20+)
- **Google Test** for C++ tests
- **FileCheck** for MLIR IR tests
- **PyTorch** for numerical reference implementations (test dependency only)
- **YAML MachineModel profiles** for Micro-IR performance evaluation (planned M10+)

## Getting Started

```bash
# Clone
git clone https://github.com/skg7on/DSLCompiler.git
cd DSLCompiler

# Build (requires LLVM/MLIR built from source)
mkdir build && cd build
cmake .. -G Ninja -DLLVM_PROJECT_BUILD_DIR=/path/to/llvm-project/build
ninja

# Run tests
ninja check-llk
```

When `LLVM_PROJECT_BUILD_DIR` is set, `MLIR_DIR` and `LLVM_DIR` are inferred and
system-installed LLVM/MLIR are ignored. If omitted, `find_package` searches
the standard CMake prefixes.

Planned M9+ Micro-IR tool surface:

```bash
# Emit concrete Micro-IR without running the JIT path
llk-compile --emit=micro input.mlir

# Emit Micro-IR search space for auto-scheduling
llk-compile --emit=micro-search input.mlir

# Evaluate a concrete Micro kernel against a machine profile
micro-perf --machine machines/x86-avx2-cpu.yaml --level l1 input.micro.mlir

# Tune candidates from a Micro search space
llk-tune --search-space swiglu.micro.mlir --machine machines/x86-avx2-cpu.yaml
```

These commands document the intended interface. They become available as the M9-M13 issues land.

## Engineering Rules

1. Keep semantics, execution scheduling, machine cost, and target lowering separate
2. Lower to Linalg before writing custom loop generation
3. Do not lower to memrefs too early -- tile and fuse in tensor form
4. Do not rely on LLVM auto-vectorization for the central microkernel
5. Treat numerical approximation as IR semantics, not command-line flags
6. Specialize on layout and ISA before specializing on every exact shape
7. Use deterministic schedules and verifier-backed legality before auto-tuning
8. Measure packing, parallel dispatch, and memory movement overhead
9. Use external GEMM as baseline, not final implementation
10. Keep Micro-IR architecture-parametric while validating first on Intel CPU with AVX2

## License

MIT
