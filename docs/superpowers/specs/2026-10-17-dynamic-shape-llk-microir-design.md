# Dynamic Shapes for LLK Kernels and MicroIR

**Date:** 2026-10-17

**Status:** Proposed design; no implementation or delivered-support claim.

**Reviewed baseline:** `45560cbdebe452dbec2ae1d5859f43bb868cf4aa`.

**User requirement:** Fully dynamic contraction dimensions, including M, N and K, are the first functional milestone. Fixed weights or dynamic M alone are insufficient.
**Plan:** [Implementation sequence](../plans/2026-10-17-dynamic-shape-llk-microir.md).

## 1. Architectural conclusion

Support runtime workload extents while binding schedule choices before execution. A dynamic tensor dimension does not require a dynamically sized hardware fragment. A selected kernel can have runtime loop bounds and boundary lengths with fixed tile capacities, vector widths, fragment shapes, pipeline depths, layouts and concrete resources.

This extends the [generic framework design](2026-10-12-generic-microir-dnn-compiler-framework-design.md), [kernel adapter design](2026-10-11-conversion-kernel-lowering-framework-design.md), [tile model](2026-08-13-micro-ir-tile-programming-model-spec.md) and [LLK export design](2026-08-11-llk-to-micro-lowering-implementation-spec.md). The tile model already allows symbolic extents; the implementation does not supply their end-to-end execution contract. The older kernel-adapter spec describes an earlier revision: current code also has rank-neutral workload contracts and a source-adapter registry. Extend those seams instead of introducing another operator dispatch layer.

```text
Source + symbolic shape contract + numerical contract
    -> runtime dimension reification / relationship checks
    -> MicroIR with runtime loops and fixed-capacity tiles
    -> finite schedule / mapping variants
    -> complete-plan storage proof + backend legality
    -> guarded executable variant + frozen shape domain

Concrete shape samples -> analysis clone -> shared plan analysis / ranking
Actual descriptors    -> shape validation -> deterministic variant dispatch
```

The recommendation is runtime-parametric execution with optional guarded specialization. Exact-shape JIT specialization alone is a useful later optimization, but would not establish that one compiled kernel supports fully dynamic M/N/K. Making every materialized tile dynamically sized would require symbolic layouts, capacity and instruction legality throughout the planner and is unnecessary for this milestone.

## 2. Current-code review

Paths and line numbers below refer to the baseline, not to the proposed implementation.

| Boundary | Current behavior | Required change |
|---|---|---|
| LLK semantics | `lib/Dialect/LLK/LLKOps.cpp:46` skips dimension comparisons when either dimension is dynamic | Preserve compile-time contradiction checks; reify unresolved equalities at execution |
| SwiGLU → Linalg | `lib/Conversion/LLKKernel/SwiGLU.cpp:50` rejects dynamic M/N; its `tensor.empty` operations receive no dynamic sizes | Supply sizes with `tensor.dim`; preserve existing arithmetic and math-mode semantics |
| Existing matmul Linalg support | `lib/Conversion/LLKKernel/Matmul.cpp:61` already collects dynamic sizes for conversion tensors | Reuse this pattern; accepting dynamic tensors locally is not full pipeline support |
| Contraction export | `lib/Conversion/LLKKernel/Contract.cpp:15` rejects every non-static tensor; `validateMicroContraction` uses integer M/N/K and `classifyM` | Represent dimensions and their relationships explicitly; only classify known M |
| Generic source contract | `include/LLK/Conversion/Workload/WorkloadContract.h:52` and its validator require positive static extents, version 1 | Add a versioned symbolic contract without making M/N/K generic framework fields |
| Kernel assembly | `lib/Conversion/LLKToMicro/MicroKernelDriver.cpp:296,396` clamps tiles against integer extents, pads whole tensors, emits constant loop bounds and a whole-output SRAM allocation | Keep tile sizes static, use runtime bounds, bounded boundary movement and explicit output destinations |
| MicroIR structure | `TileType::verify` accepts dynamic dimensions; loops already take SSA index bounds | Reuse loops; parsing `?` alone does not bind a dimension or prove capacity |
| Boundary movement | `include/LLK/Dialect/Micro/MicroOps.td:115,140,216` has static view shapes, allocation without size operands and stores without valid extents | Add explicit bounded loads and clipped stores; do not reinterpret an out-of-range view as padding |
| Reference lowering | `lib/Conversion/MicroToLinalg/MicroToLinalg.cpp:370,779` emits slices/inserts of the full static tile shape | Lower clipped slices, tile-local padding and clipped writes |
| Mapping/storage | `lib/Mapping/TileFacts.cpp`, `StoragePlan.cpp`, `WorkloadGraph.cpp:85` rely on static footprints or constant trip counts | Prove local capacity independently of tensor extents; keep unknown multiplicities explicit |
| Analysis | `lib/Perf/MicroDAG.cpp:965` uses one iteration for an unknown bound and marks analysis incomplete | Require shape samples for complete ranking; preserve truthful incomplete diagnostics otherwise |
| Invocation | `include/LLK/Runtime/KernelAbi.h` and `runtime/MappedInvocation.cpp:191` define static positive rank-two ABI v1 and exact descriptor size checks | Add ABI v2 symbolic dimensions and relationships, while retaining dtype, allocation, layout and alias checks |
| Replay/tuning | Source contract, binding facts, candidate verification and measurement identity contain concrete shapes | Distinguish reusable variant identity from a concrete invocation/sample identity |

Removing the SwiGLU rejection alone leaves unsized allocations, unchecked dimension relationships, unsafe boundary accesses, invalid capacity accounting and misleading performance estimates.

## 3. Scope and invariants

1. First executable acceptance covers independently varying M, N and K, mixed static/dynamic forms, matmul with a nonzero initializer, and two-arm SwiGLU.
2. Add a direct elementwise MicroIR fixture to prove shape/boundary machinery is independent of contractions and operator names.
3. Rank is known at compilation. Generic shape metadata remains rank-neutral; mapped execution initially stays rank two, compact row-major, offset zero, at most 12 descriptors, with read-only inputs and disjoint caller-owned outputs.
4. Local materialized tiles, instruction shapes and maximum concurrent tile count are compile-time bounded. Unbounded dynamic local allocation is rejected. Dynamic external tensors are not charged to SRAM as a whole tensor.
5. Preserve source accumulator types, output conversion points, math mode, initialization and reduction semantics. Dynamic shape support cannot silently enable approximation or reassociation.
6. No workload-name branches in shared mapping/analysis; no target-name branches in generic machine/mapping code. `LLKMapping` remains independent of runtime/JIT/perf libraries.
7. Static version-1 artifacts remain readable and retain their identity. Symbolic contracts and ABI v2 have explicit versions and distinct hashes.
8. Validate capacity and backend realizability before retaining executable candidates. Unsupported boundary capabilities reject the candidate with a reason.
9. Support nonnegative external extents. M=0 or N=0 produces an empty output with no data movement or compute; K=0 preserves matmul initialization and evaluates SwiGLU's epilogue from zero accumulators. Static execution tiles remain strictly positive.
10. Use C++20, modern MLIR APIs, local LLVM 24+ and pinned CI LLVM 22.1.8. No dependency on a new external shape solver.

Dynamic rank, data-dependent output shapes/ragged tensors, arbitrary strides, general dynamic local tiles, new hardware execution backends and production shape-profile persistence are outside this increment. RoPE/attention require their own semantic shape relations before being advertised as supported; the shared infrastructure must accommodate them without contraction fields.

## 4. Shape contract and guards

Introduce plain shape data in `LLKShape`, below workload, mapping and runtime integration. A `Dimension` is either a nonnegative constant or a named symbol. A `ShapeContract` contains ordered named port shapes, a deterministic symbol table and optional range/congruence constraints. Reusing a symbol means equality. A symbol is bound by an actual port dimension, never by the integer sentinel used in MLIR types. Arbitrary symbolic expression solving is not required.

Use namespace `::llk::shape` for the plain data. Symbol constraints are an inclusive nonnegative minimum (default 0), an optional inclusive maximum and a positive `multiple_of` (default 1). Every used symbol must be declared and occur in a port; duplicate declarations, unused declarations, unsatisfiable bounds and invalid divisors fail validation. Workload v2 attribute `shape` arrays contain integer constants or symbol-name strings; a `symbols` dictionary stores these constraints. Named axes use the same constant/string encoding. Canonical encoding tags constants and symbols separately and length-delimits names. It does not infer equality from two unrelated dynamic dimensions.

Workload contract v2 uses these dimensions for port shapes and optional named axes. Contraction adapters associate M/N/K with symbols; elementwise adapters may use d0/d1. Preserve the v1 encoder/canonical rendering for all v1 contracts. Newly exported symbolic or zero-extent contracts use v2. Replace integer-only contraction facts with `ContractionDimensions`; legacy integer helpers are called only after successful static resolution.

For SwiGLU, the adapter declares:

```text
X: [M,K], Wg: [K,N], Wu: [K,N], output: [M,N]
source destination/init shape: [M,N] (its values are not read by SwiGLU)
```

For matmul: A[M,K], B[K,N], initializer[M,N], output[M,N]. The initializer is read. Output storage remains disjoint from input storage under the mapped ABI.

When exporting SwiGLU, its source destination/init maps to the output port because its incoming values have no read effect; the output descriptor supplies its shape witness. It is not an additional read-only mapped input. Tensor entry points still validate the original destination's dimensions. Matmul's read initializer remains a separate input descriptor and is copied tilewise into accumulators before reduction. Binding metadata, rather than a positional inference from the signature, distinguishes these cases.

Static contradictions fail during verification. Unknown equalities are checked before allocation or memory access: all K dimensions agree, all N dimensions agree, and initializer/output shapes match. Reify tensor/memref extents with standard `tensor.dim`/`memref.dim`; use ordinary index arithmetic and comparisons. Compiled tensor entry points contain a guard prologue; typed mapped invocation returns an `llvm::Error` before entry on invalid descriptors. Both consume the same canonical contract. Guard elimination requires proof, not matching `?` type spellings.

Range/congruence predicates restrict optimized variants. The baseline variant accepts every representable nonnegative shape satisfying semantic and descriptor constraints. Exact-shape or divisible variants may be preferred, with stable priority and a baseline final choice. Unsupported shapes must produce a diagnostic if no executable variant applies; never silently switch execution mode.

All descriptor sizes, shape products, byte ranges, trip counts and tile-index products use checked arithmetic. Define the supported host index range as signed 64-bit nonnegative; reject values/products required by execution that exceed it or the backing allocation. An overflow in a hypothetical total FLOP count makes analysis unsupported rather than rejecting otherwise representable execution. Use loop tile numbers `t < ceildiv(extent,B)` and `offset=t*B`; calculate ceildiv as quotient plus a nonzero-remainder bit so `extent+B-1` cannot overflow.

## 5. MicroIR boundary contract

Keep `micro.for`/`micro.spatial_for` SSA bounds and static steps. For axis extent D and tile B, form `offset=t*B` and `valid=min(B,D-offset)`. Local compute types stay `!micro.tile<BMxBKx...>` etc. Avoid runtime-sized vector types.

### 5.1 Bounded load

Add `micro.tile_load`, a materializing boundary operation returning a static-capacity tile and an async token. Its operands are a ranked source, one index offset and one index valid extent per source dimension; its attributes include destination memory, optional owner and a typed padding value. Full result shape is static and positive. Explicit zero padding is used by the contraction adapter.

Semantics: copy the source rectangle `[offset, offset+valid)` into the origin of the result and fill its complement with the padding value. Require `0 <= valid[i] <= tileShape[i]` and a source-contained rectangle. Runtime values need dominating guards/construction proofs. Invalid lanes are never read. The operation is movement with storage, routes and dependencies, not a zero-cost view. It is initially a source-boundary operation on ranked tensor/memref values; existing tile-to-tile movement continues to use `micro.tile_async_copy`.

Use the existing view/copy path when full bounds are proven. Do not weaken `micro.tile_view` into an implicitly padded materialization. `tile_load` mapping must account for the same concrete source/destination nodes, routes, intermediate buffers and wait dependencies as existing copies, plus padding work. A target without bounded transfer support can realize it as a declared tile-local fill plus clipped copy; a rule/profile alone does not establish executable support.

### 5.2 Clipped store

Extend destination-bearing `micro.tile_store` with optional per-axis `valid_sizes` index operands. Absent operands preserve its current full-tile semantics and assembly. Present operands store only the origin rectangle of the source tile at the destination offsets; the result is the same destination shape/type. No write occurs outside the rectangle. Extend parser/printer, operand segments, verifier, lowering, binding and event extraction together.

### 5.3 Exact reduction tails

Extend `micro.mma` with an optional `valid_k` index operand, bounded by its static K capacity. It specifies the logical reduction length; omission means full K. This is primitive semantics and must survive extraction/materialization and selected lowering.

The baseline computes only valid K terms, preserving initialization and the existing source numerical contract. On AVX2, a selected emitter may vectorize N while using the runtime K length. A target that only implements full fragments rejects partial-K plans unless a supported remainder implementation exists. Treating zero-filled K padding as unconditional arithmetic equivalence can alter signed-zero or reduction behavior; padded full-fragment execution is an optional transformation requiring the numerical contract to authorize it. No new fast-math policy is implied here.

### 5.4 Output destination

The dynamic exporter must not use the current whole-output `micro.tile_alloc` in SRAM. Add explicit argument/result bindings on `micro.kernel`: each entry argument identifies an input port or caller-owned output port; each yielded result identifies an output port. Output arguments may be ranked tensors at the semantic boundary, become memrefs in the wrapper, and are carried through the spatial loops. The kernel yields the updated destination.

For v2, output bindings are lowered to caller output memref arguments before ordinary bufferization, with result-to-destination equivalence verified. Read initialization is a separate input port for matmul; SwiGLU's output destination has no read effect. Analysis treats these destinations as external storage, not compiler SRAM. Any bufferization-introduced full-output temporary is an acceptance failure. Existing unannotated static kernels keep the v1 path.

## 6. Mapping and feasibility

Separate static physical facts from runtime work multiplicities. For example, contraction local storage includes `stages * (BM*BK*inputBytes + arms*BK*BN*weightBytes)`, accumulator tiles, selected epilogue temporaries, routing intermediates and layout padding. This is an illustration, not a replacement for shared lifetime analysis: count actual buffers and overlap in the complete materialized plan.

Runtime tile counts must not imply simultaneous allocation of every logical tile. Bound spatial residency by the selected executor pool and declared launch/scheduling policy; prove the maximum number of concurrently live pipeline buffers independently of total M/N/K. If residency cannot be bounded, reject capacity certification. Edge tiles have the same allocated tile capacity even when fewer bytes are copied.

Predicates are classified as proven, disproven or requiring a runtime guard. Symbolic divisibility is not automatically legal. The baseline uses bounded loads/stores and exact K tails; full-tile variants require guards. Machine capabilities and LLKMap declare bounded-load/store/partial-K realization requirements. Static tiles still drive layout/fragment legality. Every new computational/movement leaf participates in complete-analysis coverage checks.

Capacity certification is valid for the full declared shape domain. Cost ranking may cover only specified samples, and must say so. Do not infer domain-wide latency optimality from a sample. A dynamic kernel with no samples can be compiled with a supplied legal schedule and deterministic mapping choices; it cannot claim a complete numeric performance score or participate in latency-based top-K ranking.

## 7. Analysis, search and identity

Initially require finite concrete `ShapeSample`s for dynamic performance analysis/tuning. Each sample binds every symbol to a validated extent and may have a positive weight. Evaluate an analysis-only clone: substitute bound dimension computations, fold arithmetic/guards and run the same workload extraction, complete-plan materialization and shared analysis used for static plans. Executable IR retains runtime dimensions. Do not specialize the source into an unrelated synthetic contraction.

For `micro-perf`, add `--shape-values=M=...,N=...,K=...` (symbol names come from the contract). Without complete bindings, report incomplete analysis and no rankable latency. Extend dynamic tuning with a versioned `--shape-profile=<json>` containing ordered samples, weights, aggregation policy and variant budget. Its initial schema is `{"version":1,"samples":[{"values":{"M":7,"N":13,"K":5},"weight":1.0}],"aggregation":"weighted_mean","max_variants":1}`. Weights must be finite and positive; max_variants must be positive and defaults to 1. Only weighted_mean is accepted initially. Every candidate must be legal for the full executable domain and analyzable at every sample. Invalid or missing samples fail explicitly. Keep finite domain bounds, truncation and tie-breaking in reports.

Counts must handle interior/edge tiles, clipped external traffic, full allocated footprints, valid-K compute and fill work. Plan and materialized-IR reports must agree for each sample. For an event count exceeding the existing simulator limit, return an explicit unsupported/incomplete result; compressed symbolic DAG scheduling is later work. Unknown bounds must never become a valid one-iteration score.

Substituting M/N/K does not make `min(B,D-offset)` constant inside an unexpanded loop. Extend shared shape-fact evaluation with an explicit induction-variable environment for each sampled iteration; `walkLoop` binds the current IV before handlers evaluate valid sizes and valid_k. Support only the checked index arithmetic/comparisons emitted by shape lowering, and reject unsupported expressions. Mapping cost aggregation and MicroDAG handlers consume the same evaluated boundary facts. Do not merely repeat one body event with full-tile bytes for every iteration or flatten away pipeline/dependency structure to obtain constants.

Identity has two levels:

- **Compiled variant:** original source content, symbolic contract/version, numerical contract, ABI/version, guard domain, static schedule, mapping/resources, machine/rules/layouts/emitter versions, compiler and execution mode. Different invocations within that domain reuse it.
- **Sample/measurement:** compiled variant identity plus actual shapes, sample/profile identity and existing measurement protocol. Never reuse a measured latency across shapes merely because code is reused. Packed-weight caches also retain actual shape, layout, dtype and existing content/lifetime identity.

Frozen replay preserves the symbolic artifact and guards, the shape profile used for selection and the deterministic dispatch order. Shape samples are observations used for ranking, not replacements for the executable domain. Exact-shape JIT caching and bounded profile-guided multiversioning follow the baseline milestone.

## 8. Runtime and empty extents

ABI v2 records the canonical symbolic contract and explicit argument/result port bindings. Keep the existing memref descriptor transport; sizes already reside in `MemRef2D`. This changes ABI metadata/validation and generated wrapper signatures, not caller buffer ownership. Version 1 retains its existing static validation and hash. Version 2 checks static dimensions exactly, binds repeated symbols consistently and enforces predicates before JIT entry.

Retain dtype tags, compact strides (`stride1=1`, `stride0=size1`), zero offsets, allocation bounds and disjoint outputs. Negative sizes fail before conversion to unsigned arithmetic. For zero-element ports allow null pointers with `allocationBytes=0` and return an empty touched span; nonempty ports retain existing pointer requirements. Empty spans never alias. All semantic checks still run on empty invocations, so M=0 does not excuse incompatible K dimensions. Nonempty output with K=0 still requires valid output storage and initialized accumulators.

Scratch stays invocation-owned. Fixed local allocations are released on completion and error paths. Packed-weight routines must either accept runtime K/N with correctly keyed packing or select an explicit unpacked implementation; bypassing legacy fixed-shape packing must be deliberate and reported. Dynamic support must not depend on packing an entire unbounded weight matrix into limited local memory.

## 9. Milestones and acceptance

| Milestone | Deliverable | Gate |
|---|---|---|
| A: contracts and primitives | Shape v2, guards, bounded loads/stores, exact-K semantics, explicit destinations | Round-trip/verifier tests and static identity compatibility |
| B: fully dynamic baseline | M/N/K runtime export, portable reference and selected AVX2 execution, legal full-domain mapping | Compile once and invoke multiple unrelated triples, with no exact-shape recompilation |
| C: measured optimization | Sample-aware analysis/search and correct cache/replay identity | Plan/IR analysis parity and deterministic profile ranking |
| D: optional specialization | Guarded interior/edge split, divisible fast paths, bounded exact-shape variants | Same numerical and memory checks; include dispatch/packing cost |

Mandatory fixtures include `(M,N,K)=(1,1,1),(7,13,5),(8,32,32),(9,33,35),(17,65,67)` under one fully dynamic signature; mixed-static variants; each empty dimension; nonzero matmul initialization; mismatched Wu K/N; incorrect output shape; negative/overflowing descriptors; undersized allocation; unsupported dtype/stride/alias; full and partial K; signed zero and nonfinite values under the supported numerical contracts; repeated invocations with different shapes; allocation lifetime; constrained SRAM with large external tensors; and a dynamic direct elementwise kernel.

Reference and selected-target results are separate evidence. Selected AVX2 acceptance requires an AVX2 host and must fail rather than silently report reference success. Generic accelerator conformance proves static feasibility/capability rejection, not hardware execution. Run the full registered suite on pinned Linux LLVM 22.1.8 and compile/reference checks locally with LLVM 24+.

## 10. Upstream mechanisms and limits

The proposed lowering uses existing MLIR mechanisms: [shape inference](https://mlir.llvm.org/docs/ShapeInference/) distinguishes static inference from runtime shape computation/checks; [tensor operations](https://mlir.llvm.org/docs/Dialects/TensorOps/) provide dynamic slices and padding; [vector transfers](https://mlir.llvm.org/docs/Dialects/Vector/) provide padding and bounds/mask semantics. These references support the building blocks, not a claim that the repository's current passes implement this design. Validate concrete APIs and transformations against both supported LLVM revisions rather than copying the rolling documentation's signatures.

Design calls made for this proposal: fixed-capacity local tiles, a new materializing boundary load, an explicit valid-K operand, output destination bindings, ABI/workload v2, support for zero extents, sample-based initial performance analysis, and fully dynamic M/N/K before specialization. These are recommendations awaiting design review, not additional user requirements.
