# The Micro-IR workflow

This is the contributor-facing guide to generating, evaluating, tuning, and
mapping tile-centric Micro-IR. It assumes the build works (`README.md`) and that
you have read the tile programming model spec
(`docs/superpowers/specs/2026-08-13-micro-ir-tile-programming-model-spec.md`).

## The idea in one line

**DNN Micro-IR = Tile Dataflow + Tile Mapping + Tile Schedule.**

A `micro.kernel` is a concrete schedule, not an intermediate representation of
the model. Its primary value is a **tile** — `(shape, dtype, layout,
memory_space, owner)` — and the dialect says *where* work runs and *what it
costs*, in terms a machine model can price:

| Layer | What it answers | Ops |
|---|---|---|
| tile dataflow | what work happens | `micro.mma`, `micro.vector`, `micro.reduce` |
| tile mapping | where data lives and who owns the work | `micro.tile_view`, `micro.tile_partition`, `micro.tile_alloc`, `#micro.layout`, `#micro.owner`, `#micro.memory` |
| tile schedule | when it happens | `micro.for`, `micro.spatial_for`, `micro.pipeline`, `micro.async_copy`, `micro.wait` |

## Two paths out of LLK/Linalg

```bash
# 1. A concrete kernel: the schedule chosen now, emitted as Micro-IR.
llk-compile --emit=micro input.mlir

# 2. A search space: the tunable parameters, their legality constraints, and the
#    objective a tuner ranks candidates by. Nothing is chosen yet.
llk-compile --emit=micro-search input.mlir
```

`--emit=micro` produces a `micro.kernel` with no search ops in it. `--emit=micro-search`
produces a `micro.search_space` with `micro.param`, `micro.constraint`,
`micro.objective`, and (once a candidate is fixed) `micro.candidate`.

The rest of `llk-compile`'s surface (`--emit=llvm`, the default, and `--emit=mlir`)
is the CPU validation path and is unchanged by anything below:
**AVX2 is the first validation backend, not the only supported target, and the
existing compilation and JIT path remains available.**

## Evaluating a concrete kernel

```bash
micro-perf --machine machines/x86-avx2-v2.yaml --level 1 kernel.micro.mlir
```

- `--level 0` prices the kernel statically: work volume against the machine's
  issue rates and bandwidths.
- `--level 1` also schedules it against the machine's resources, so overlapping
  copies, pipelined loops, and owner limits show up.
- `--format text` prints the same numbers for humans.
- `--kernel <symbol>` picks one kernel out of a file with several.

The output reports totals, bounds, utilization, bottleneck, live tile bytes, and
any warning the machine raised: a layout a memory does not support, an owner the
machine does not model, a copy path it does not declare.

## Tuning

```bash
llk-tune --input swiglu.micro.mlir --machine machines/x86-avx2-v2.yaml \
         --M 128 --N 4096 --K 4096
```

`llk-tune` enumerates the search space, checks each candidate against the
machine, ranks the legal ones by the space's `micro.objective`, and writes the
winner as a schedule record. `--M/--N/--K` supply the problem shape, which the
search space does not carry.

## Describing your machine

Machine models are YAML (`machines/*.yaml`) and are loaded into a typed topology:
executors, memories, compute capabilities, transfer engines, and the directed
links between memories. A profile is data, so a new machine is usually a new
file rather than new code.

What a profile declares (all of it optional except the target name):

```yaml
schema: llk.machine.v2
target: my-machine
clock_hz: 3000000000       # needed only for nanosecond estimates
worker_threads: 8
sync: {barrier_cycles: 200, wait_cycles: 0}
executors: [...]           # each with a Micro owner kind, a parent, a concurrency,
                           #   optional coordinates, and optional equivalent_to
memories: [...]            # capacity, alignment, access bandwidth and latency
compute: [...]             # element types, accumulator types, shapes, lanes
transfer_engines: [...]
links: [...]               # transfer bandwidth, latency, engines
```

Copy `machines/x86-avx2-v2.yaml` and calibrate it for your host before trusting
any cycle estimate. The numbers in the shipped files are calibration seeds, not
claims about any SKU.

An executor may list `equivalent_to: [other, ...]` to declare that it can be
swapped with those executors without changing a mapping. The declaration is a
*target* assertion, so the loader enforces it: every entry must name an existing
executor, must be mutual (the other executor declares this one back), and must
join executors of the same kind. The mapper collapses a declared group to a
single representative; symmetry reduction over executors that declare nothing
falls back to a structural rule (same kind, parent, coordinates, concurrency,
scheduling class, and compute/memory attachments). Both are off-switchable for
diagnostics (`MappingSearchOptions::enableSymmetryReduction`,
`PlacementOptions::reduceSymmetry`), so a caller can still enumerate every
representative.

## Mapping a kernel onto a target

The mapping subsystem takes a concrete kernel and a target and chooses *which*
implementation of each operation to use, where to place it, and how data moves
between the pieces. It is target-independent: nothing about AVX2 appears in the
generic code, and a second target (the generic accelerator in
`machines/generic-ai-accel-v2.yaml`) needs only its own declarative files.

A target packages four things, and only the first three are files:

1. a **machine profile** (`machines/*.yaml`),
2. **layout declarations** (`mapping/<target>/layouts.llkmap`) — which layouts the
   target supports and the constraints under which each is legal,
3. **mapping rules** (`mapping/<target>/rules.llkmap`) — which Micro operations
   map to which target bundles,
4. the **emitter keys** its plugin implements (code, because they name C++
   emitters).

The chain, and where each step lives:

```
concrete micro.kernel
  -> WorkloadGraph          LLK/Mapping/WorkloadGraph.h    what work exists
  -> MappingCandidate       MappingRules.h                 which rule matches
  -> CandidateInstance      Placement.h                    where it runs
  -> ConnectionPlan         Placement.h                    how values move
  -> CoveringPlan           CoveringSearch.h               a complete selection
  -> bound kernel           PlanBinder.h                   selections written down
```

Each step is deterministic: the same kernel and target produce the same ids,
the same plan, and byte-identical metadata.

### What the binder writes

`bindPlan` clones the module — the source is never mutated — and records the
selection as ordinary attributes, so `llk-opt` prints and re-parses mapped IR
with no target plugin loaded:

- `micro.plan` on the kernel: the plan id, the source binding hash, whether a
  cap truncated the search;
- `micro.mapping` on each covered operation: the rule, bundle, emitter,
  executor, memories, and layouts it selected;
- `micro.routes` on the kernel: each connection's ordered memory route and the
  engines that carry it.

A movement the plan routed is emitted as `micro.async_copy` plus `micro.wait`,
and `micro-perf` charges it **hop by hop** rather than endpoint to endpoint.

### Verification is layered

1. the ordinary Micro verifier checks structure;
2. `verifyMappedMicroIR` resolves every id the metadata names — rules, executors,
   memories (and that the executor can see them), layouts, route hops, emitters;
3. the target emitter validates its own bundle before lowering.

## Acceptance chains

The design's acceptance criteria are checked two ways, and the split matters
because they answer different questions.

**Execution.** `test/Execution/mapped_acceptance.cpp` exports the compiler-
generated matmul and SwiGLU, maps them against the shipped AVX2 target,
compiles the selected plan and **invokes it**, checking every element of every
result. This is what a passing build means by "it works": a mapped program that
runs and computes the right numbers.

These chains are host-portable. The exported program is Linalg, SCF and memref
and the JIT compiles it for whatever machine is running, so the numbers are the
same on arm64 and on x86. What it does *not* establish is that AVX2
instructions were emitted -- that needs an AVX2 runner, and no test here
pretends a portable run is evidence about an instruction set.

**Pipelines.** `test/Conversion/MicroMapping/acceptance_pipeline.py` drives the
tools through their public flags only, and asserts the things that are invisible
from inside one process:

- the export and the search round-trip;
- the mapped IR carries `micro.plan`, `micro.mapping` and `micro.routes`;
- the plan report is the versioned schema and names the machine, layout, rule
  and target content hashes it searched;
- two runs produce **byte-identical** report and IR;
- `--micro-verify-mapping` accepts what the mapper wrote;
- the performance model produces events for it.

Its chains are `vector-add`, `staged-gemm`, `fused-swiglu` and `second-target`
(the same concrete kernel mapped on the generic accelerator). The
compiler-generated chains are AVX2-only, because the exported program uses the
tile-level ops that only the AVX2 rule set covers; the accelerator's rules
describe the tensor-level movement and vector family.

## Where to go deeper

- tile model, ops, and verifier rules: `docs/design/m9-micro-ir-core-concepts.md`
- layering and the redesign: `docs/design/m9-canonical-micro-ir-architecture.md`
- the mapping design: `docs/superpowers/specs/2026-09-18-microir-inspired-dslcompiler-enhancement-design.md`
- the LLKMap grammars: `docs/design/llkmap-layout-grammar.md`, `docs/design/llkmap-rule-grammar.md`
