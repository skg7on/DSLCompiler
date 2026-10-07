# Features and current limits

MicroIR provides a common representation for scheduled tensor kernels, declarative
target mapping, and performance exploration. The established CPU compiler remains
available while the newer mapped path is being strengthened. Use this page to
choose a workflow and understand what a successful run proves.

This summary describes the source tree after PRs #128 and #130, reviewed on
2026-10-07. #130 adds repair plans; it does not implement their proposed runtime,
resource, backend, or tuner changes. The [current gap assessment](reviews/2026-10-07-issue67-current-gap-assessment.md)
is the detailed evidence behind the limits below.

## What you can do today

| Capability | Available behavior | Important limit | Guide |
|---|---|---|---|
| Write and inspect MicroIR | Tile types; kernel signatures; temporal/spatial loops; pipelines; MMA/vector/reduction; movement, transforms, waits and barriers; search-space operations | Syntax and structural verification do not establish a legal target mapping or executable backend support | [Concepts](concepts.md), [first kernel](tutorials/01-first-kernel.md) |
| Compile the established CPU path | LLK-to-Linalg lowering for supported semantic operations, scheduling, tiling/fusion, explicit Vector IR, bufferization and LLVM/JIT infrastructure | `llk.matmul` is left untouched by the legacy conversion and uses the Micro route; kernel, shape, dtype, math-mode and host support remain specific | [Compiler](tools/compiler.md) |
| Export supported semantic kernels to MicroIR | Concrete kernel and search-space export for LLK matmul and fused SwiGLU | General Linalg, RoPE and attention export are not implemented by `LLKToMicro` | [Compiler](tools/compiler.md) |
| Describe a machine | Versioned YAML topology with executor, memory, compute, transfer nodes and links; owner refinements and model validation | Shipped numerical costs are modeling seeds; a profile is not a hardware execution backend | [Architecture](architecture.md), [performance tutorial](tutorials/03-performance.md) |
| Author target policy | LLKMap layouts, affine maps, finite constraints, port predicates, capability requirements, target bundles and emitter keys | Bounded solving can truncate or remain undecided; unsupported facts must be diagnosed | [Formal syntax](design/llkmap-syntax.md) |
| Match single and fused operations | Single-operation rules and bounded fused-group matching | The shipped convert→SiLU→multiply fused rule can be selected and bound but currently fails target lowering; group dispatch is incomplete | [Mapping tutorial](tutorials/02-mapping.md) |
| Search mappings | Deterministic, beam and exact covering modes; placement, route and connection alternatives; search bounds and diagnostics | Physical feasibility is finalized after top-K retention and may lose an available legal plan; beam mode has no optimality guarantee | [Optimizer](tools/optimizer.md) |
| Bind, report and replay selections | Versioned plan reports and selected metadata; canonical movement/transform/sync materialization; layered verification | Concrete selected compute identities and complete physical storage are not yet preserved or checked across every path | [Optimizer](tools/optimizer.md), [architecture](architecture.md) |
| Estimate performance | Level-0 work/bandwidth bounds and level-1 dependency/resource scheduling; cycles, traffic, utilization, bottleneck and capacity reporting | Planner and bound-kernel estimates currently disagree; named DMA concurrency and storage accounting have known defects; predictions are uncalibrated | [Performance analyzer](tools/performance.md) |
| Tune schedule parameters | Candidate enumeration, legality checks, synthetic GEMM/SwiGLU binding, objective ranking and schedule output; optional library measurement callback | The CLI does not map and compile every original-source candidate; the callback is not a production measurement database | [Tuner](tools/tuning.md), [tuning tutorial](tutorials/04-tuning.md) |
| Run supported mapped kernels | Shared mapped compilation and descriptor-pointer ORC invocation; portable numerical GEMM/SwiGLU acceptance fixtures | Selected AVX2 width/ISA realization is unproved; repeated invocations can leak compiler allocations, and descriptor rank/dtype/buffer validation is incomplete | [Compiler](tools/compiler.md) |
| Exercise the benchmark harness | `llk-bench` runs a synthetic host loop and reports timing | It does not parse MLIR, load weights, invoke a compiled kernel or use the JIT; thread-pool setup occurs per invocation | [Benchmark](tools/benchmark.md) |
| Explore a second target | Generic accelerator topology, layouts, rules and mapping through the same canonical IR | No generic accelerator hardware backend or directly interpreting MicroIR emulator is delivered | [Mapping tutorial](tutorials/02-mapping.md) |

The implemented operation inventory is
[MicroOps.td](../include/LLK/Dialect/Micro/MicroOps.td). Broad operation lists in
design specifications describe direction as well as delivered features; they
should not be used as an executable support matrix.

## Choose a starting point

**Learn the IR:** parse a small kernel with `llk-opt`, inspect its tile values,
and follow [tutorial 1](tutorials/01-first-kernel.md). This is the fastest way to
understand the model without needing a semantic frontend or JIT.

**Explore a mapping:** use a small concrete kernel, the shipped AVX2 package,
and a versioned plan report. Follow [tutorial 2](tutorials/02-mapping.md), inspect
the selected rules and routes, then compare the bound IR with the source. A
matching plan ID is evidence of selection consistency, not proof of complete
physical feasibility or target code generation.

**Explore a machine:** change a profile in an isolated worktree and compare
analyzer reports for the same kernel. Follow [tutorial 3](tutorials/03-performance.md).
State which profile and analysis level produced the result, and keep modeled
cycles separate from measured time.

**Explore schedules:** export or load a search space, cap enumeration while
experimenting, inspect rejected candidates, and follow
[tutorial 4](tutorials/04-tuning.md). The synthetic tuning workflow is useful for
search-space exploration; it does not yet establish the best mapped executable
for the original source.

**Explore the timing harness:** read the [benchmark guide](tools/benchmark.md)
before using `llk-bench`. Its synthetic host loop measures the harness rather
than a compiled MLIR kernel, and its per-invocation thread-pool setup is part of
that timing. For actual supported kernel invocation, use the C++
[mapped compilation API](../include/LLK/Conversion/MappedCompilation.h) and
[MappedExecutable](../include/LLK/Runtime/MappedExecutable.h), with
[execution tests](../test/Execution/mapped_acceptance.cpp) as examples. Observe
the mapped runtime limits above and record host, inputs, schedule and timing
conditions; these APIs do not constitute the planned calibrated measurement loop.

## Work planned separately

The [issue #129 gap-closure plan](superpowers/plans/2026-10-07-issue129-gap-closure.md)
defines proposed repairs in three areas:

- **Resources and static scoring:** preserve concrete selections; finalize
  per-hop storage/liveness before retaining plans; respect each resource's own
  concurrency; align selected-plan and bound-kernel static analysis.
- **Execution and runtime:** validate typed invocation descriptors; establish
  output ownership and deterministic scratch release; lower fused selections as
  groups; realize selected vector widths and ISA requirements in backend code.
- **Tuning and acceptance:** instantiate the original source for each binding;
  integrate mapping, selected analysis and optional compilation/measurement;
  strengthen replay and all required numerical/public workflow checks.

These are planned contracts, not available flags or newly delivered APIs.
Production measured-record persistence, calibration fitting and prediction-error
validation remain the separate #51/#52 workstream. A direct MicroIR emulator and
a broad Python/model frontend remain deferred. Existing Python/Triton-related
scaffolding does not establish a complete general frontend.

## Reading status evidence

Historical milestone documents record substantial delivered work, but a merged
component or passing portable numerical test does not settle every end-to-end
contract. In particular:

- Parsing proves the representation can be read; target verification resolves
  selected facts.
- A materialized route proves copies were emitted; complete intermediate
  allocation lifetimes and occupancy need their own checks.
- A portable native-host JIT result proves numerical behavior for its inputs;
  selected AVX2 lowering needs explicit vector-width/ISA evidence.
- Shared event primitives do not by themselves establish planner/analyzer cost
  parity, and static parity does not establish calibrated hardware prediction.

For exact reproduced failures and scope, read the
[current gap assessment](reviews/2026-10-07-issue67-current-gap-assessment.md).
To help close a gap or extend a capability, start with
[contributing](contributing.md).
