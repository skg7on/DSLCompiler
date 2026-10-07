# MicroIR developer guide

DSLCompiler is an MLIR compiler and hardware evaluation project built around **MicroIR**, the `micro` dialect. It describes a kernel's tile computation, placement, movement, and schedule so developers can inspect how it executes on a machine profile. AVX2 is the first validation target; the generic accelerator package demonstrates how target policy can change without changing the mapping core.

## Choose a starting point

| Your goal | Read first | Try next |
|---|---|---|
| Understand the project in a few minutes | [Concepts](concepts.md), [features and current limits](features.md) | [Architecture](architecture.md) |
| Build and inspect a kernel | [Getting started](getting-started.md) | [Your first kernel](tutorials/01-first-kernel.md) |
| Choose a hardware mapping | [Mapping tutorial](tutorials/02-mapping.md) | [Optimizer manual](tools/optimizer.md) |
| Estimate hardware costs | [Performance tutorial](tutorials/03-performance.md) | [Performance manual](tools/performance.md) |
| Explore schedule choices | [Tuning tutorial](tutorials/04-tuning.md) | [Tuner manual](tools/tuning.md) |
| Extend a compiler component or target | [Architecture](architecture.md) | [Contributor guide](contributing.md) |

## Tool manuals

| Tool | Main responsibility | Manual |
|---|---|---|
| `llk-compile` | Export IR or compile a kernel, including the shared mapped compilation path | [Compiler](tools/compiler.md) |
| `llk-opt` | Parse, verify, transform, map, and inspect MLIR | [Optimizer](tools/optimizer.md) |
| `micro-perf` | Estimate cycles, traffic, capacity, and resource pressure | [Performance](tools/performance.md) |
| `llk-tune` | Generate and rank schedule candidates; write schedule records | [Tuning](tools/tuning.md) |
| `llk-bench` | Run the current synthetic host benchmark harness | [Benchmarking](tools/benchmark.md) |

The tutorials run from the repository root, use binaries in `build/`, and write generated files under `build/tutorial/`. The checked-in [examples](examples/README.md) contain real MLIR inputs. The sequence is: semantic source → concrete MicroIR → mapped MicroIR → cost report. Search-space export and tuning are a related workflow, with the current integration limits explained explicitly.

## Reading the project accurately

This guide describes the implementation on `main` after PRs #128 and #130, using their source baseline `bb82038`. It introduces the architecture's intended contracts and identifies where implementation evidence is narrower. A successful parse, mapping, cost estimate, JIT compilation, and numerical invocation establish different things.

The [feature guide](features.md) summarizes current capabilities. [Issue #129](https://github.com/skg7on/DSLCompiler/issues/129), its [implementation plan](superpowers/plans/2026-10-07-issue129-gap-closure.md), and the [gap assessment](reviews/2026-10-07-issue67-current-gap-assessment.md) track remaining resource, execution, tuning, and acceptance work. Those plans are not user-facing features already delivered. Measurement calibration is separate work in [#51](https://github.com/skg7on/DSLCompiler/issues/51) and [#52](https://github.com/skg7on/DSLCompiler/issues/52).

## Deeper references

- [Micro dialect semantics](design/m9-micro-ir-core-concepts.md) and [tile programming model](superpowers/specs/2026-08-13-micro-ir-tile-programming-model-spec.md).
- [Mapping workflow](design/micro-ir-mapping-workflow.md), [layout grammar](design/llkmap-layout-grammar.md), and [rule grammar](design/llkmap-rule-grammar.md).
- [Normative mapping design](superpowers/specs/2026-09-18-microir-inspired-dslcompiler-enhancement-design.md) and [historical CPU architecture](../ARCHITECTURE.md).
- [Milestone design documents](design/) and [implementation plans](superpowers/plans/) explain decisions and task history; start with this guide for current usage.
