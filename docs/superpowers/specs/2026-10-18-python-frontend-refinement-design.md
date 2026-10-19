# Python frontend refinement and LLK/MicroIR integration design

**Date:** 2026-10-18

**Status (2026-10-19):** PF01 packaging/registration is implemented in
[PR #233](https://github.com/skg7on/DSLCompiler/pull/233), pending merge. Remaining
capabilities are proposed. The [current frontend guide](../../manual/source/python-frontend.rst)
and [environment evidence](../../reviews/2026-10-19-pf01-python-environment.md)
describe the project `.venv` and verified PF01 behavior; this design does not
establish native execution support.

**Baseline:** `13cef9c` (main, after #200).

**Review:** [Eleven frontend findings](../../reviews/2026-10-18-python-triton-inductor-gap-assessment.md).

**Plan:** [Implementation tasks and acceptance](../plans/2026-10-18-python-frontend-refinement.md).

## 1. Intent, scope and decisions

Make Python a real consumer of the generic MicroIR framework: capture a supported computation, preserve its semantics, select and compile an executable, invoke checked buffers, and show evidence of what executed. Repair the concrete review defects and define separate, testable contracts for the LLK decorator, Dynamo, Inductor and upstream Triton.

The user's requested scope is frontend refinement, detailed implementation planning and a tracking issue. The architecture choices below are proposed engineering defaults, not additional user requirements or claims of approval. This document refines the July Triton frontend proposal where its fake TTIR and CPU-pipeline assumptions conflict with the current framework. The generic framework design remains authoritative.

Choose a shared native frontend service and a versioned typed frontend program. LLK's own AST frontend lowers to this program; the service constructs registered LLK/Linalg operations and uses the existing workload/source-adapter, mapping and mapped-runtime APIs. Do not build another optimizer in Python. Do not require upstream Triton to import `llk`.

Alternatives considered: implementing the full upstream Triton dialect immediately would couple the first Python executable to a separate LLVM toolchain; continuing string-based pseudo-TTIR would retain the current semantics and parser failures. The typed neutral boundary allows a small executable LLK frontend first and a genuine upstream adapter later.

The first Inductor integration uses `torch.library.custom_op` CPU implementations plus a scoped FX-to-LLK rewrite before delegation to Inductor. This delivers externally called LLK/MicroIR kernels inside an Inductor-managed graph. It does not promise fusion across the external call or a new device backend. A full Inductor scheduler/codegen backend is a separate future expansion.

## 2. Milestones and honest closure

| Milestone | Delivered capability | Required evidence |
|---|---|---|
| A: foundations | Real registration, strict diagnostics, repaired legacy passes, versioned program and shared native service | Negative tests, parser/verifier checks, no disabled verification |
| B: executable decorator | CPU add/multiply, then matmul/SwiGLU, caller-owned outputs, bounded specialization | Native invocation, numerical results, cache identity, memory canaries |
| C: framework integration | Independent Dynamo backend and Inductor external-kernel composition | Real Torch tests, FakeTensor/opcheck, fullgraph composition, native-call evidence |
| D: upstream Triton | Real upstream JIT/backend/driver adapter over an audited subset | Actual `@triton.jit` launch, genuine upstream TTIR, LLK execution evidence |
| E: release acceptance | Installable artifacts, reproducible CI, documented capability matrix | Clean-environment install and all required execution lanes |

Review closure means every advertised capability either executes correctly or is rejected at a documented boundary. It does not mean all operations in the old proposal become supported. Atomics, asynchronous/shared-memory programs, arbitrary reduction/control-flow patterns, arbitrary ranks/strides, CUDA execution and model-wide dynamic shapes remain unsupported until separately implemented. The upstream integration milestone remains open until a real upstream kernel runs; importing a wrapper or reusing its Python source does not close it.

## 3. Supported initial envelope

The planned reference envelope below retains Python 3.11/3.12 + Torch 2.9.0.
Current local development uses the project `.venv` with Python 3.14.8, Torch
2.14.1 and NumPy 2.5.4, separately verified for PF01's 21 tests. This additional
package/registration evidence does not qualify the proposed native, Inductor or
upstream Triton milestones on that environment.

- Python 3.11 and 3.12; C++20. CI LLVM/MLIR 22.1.8 and the repository's local LLVM 24+ lane. Use version guards where APIs differ.
- Initial real-framework compatibility baseline: Torch 2.9.0; optional upstream Triton 3.5.0 on Linux x86_64. These are deliberately pinned initial validation versions, not a claim about the newest releases. Their pairing is recorded by [PyTorch's version file](https://github.com/pytorch/pytorch/blob/v2.9.0/.ci/docker/triton_version.txt). Qualify additional releases independently.
- CPU, positive static rank-two shapes, compact row-major, zero storage offset, disjoint input/output storage. Dtypes: strict f32 add/multiply; f32 matmul; bf16 inputs/f32 accumulation only when the selected native adapter/backend certifies it. No automatic input copies or dtype conversion.
- Pointer-output decorator kernels return `None`. Framework functional wrappers allocate a fresh CPU tensor and pass it as a checked output. Reject result-returning block kernels until a separate result contract exists.
- Synchronous calls. Keep input/output owners and the native executable alive until invocation finishes. Initial execution is serialized per executable; concurrent callers may share immutable compiled code but not scratch state.
- `backend="reference"` is explicit for host validation. `backend="selected-target"` requires supported host ISA, realized selected backend groups and zero reference fallback for fixtures claiming selected execution. Never substitute reference execution when a selected target is unavailable.
- First release specializes concrete shapes. A changed concrete shape produces another bounded cache entry. No claim of compile-once symbolic execution; coordinate ABI v2 and shape guards with #191–#199.

## 4. Typed frontend boundary

Add immutable Python records in `python/llk/frontend/_ir.py` and a matching checked JSON schema in `docs/design/python-frontend-v1.schema.json`:

- `TensorSpec(name, shape, dtype, strides, direction, effect)`; strides are element strides, shape is concrete, and effects distinguish read inputs from write-only destinations.
- `LaunchSpec(grid: tuple[int, int, int], block: tuple[int, int, int])`; both contain positive integers. Normalize tuple or constexpr-callable grids exactly once. Axes outside 0–2 fail.
- `SourceSpan(file, line, column)`; every diagnostic-producing node carries one.
- `Value(id, dtype, shape)` and `Operation(opcode, operands, results, attributes, regions, span)`; defined-before-use, unique ids and exact operand/result schemas are required. Scalars/constexprs are typed; no Python repr or AST text is an operand.
- `FrontendProgram(version=1, entry, tensors, constexprs, launch, operations, numerical_contract)`; includes explicit stores and a void return. The finite opcode registry begins with constants, affine indices, arithmetic/comparison/select, broadcast, load/store, dot and the recognized SiLU expression. The only initial region-bearing opcode is `counted_loop`: signed-index lower/upper, a positive constant step, typed induction/block arguments, init/iter values, body operations and a typed yield matching the iter values. Body values cannot escape except through loop results; canonical K-reduction recovery verifies this structure. Other regions/opcodes fail validation.
- `CompileOptions(target, backend, search_mode, max_candidates, top_k, numerical_contract)` and `CompileRequest(program, options)`; initial search defaults are deterministic, max_candidates=64, top_k=1.
- `CompilationReport` holds `native_request_identity` (the full canonical native specialization identity), frontend/compiler identity, workload/source-adapter identity, target/machine/rule content hashes, plan id, ABI hash, selected/reference group counts and reached stages. Counts come from native compilation, never inferred from a requested option.

The numerical contract is structured, not an opaque fast/strict switch: storage/compute/accumulation dtypes; explicit cast sites and rounding (round-to-nearest-even for the initial f32/bf16 modes); activation expression/approximation; reduction-order, reassociation and FMA permissions; and subnormal behavior. Capture and preserve every observable intermediate rounding boundary. Default decorator semantics forbid relaxed activation/reassociation unless the caller explicitly opts into a distinct keyed mode. Torch matching preserves ATen dtype/cast/activation semantics and declines an unsupported numerical realization. A tolerance in an acceptance test never authorizes deleting casts or changing activation/reduction policy.

Canonical serialization sorts mapping keys, preserves port/op order, encodes integers without float conversion and rejects unknown fields/versions. Runtime addresses are excluded. The canonical program and complete native environment identity are retained beside a SHA-256 index; compare full identities on hits. Final cache identity includes the selected source candidate and plan.

The service owns parsing, semantic validation, structured recovery, adapter selection, candidate realization, mapping, plan binding and `compileMappedKernel`. Reuse the contract overload of `runMappedTuningSession` for candidate selection; retain the selected instantiated module/plan through a shared completion helper rather than replaying a synthetic M/N/K workload. Coordinate generic CLI/replay changes with #156 and capability receipts with #157; do not fork those schemas.

## 5. AST, pointer and launch semantics

`inspect.signature(...).bind` binds positional/keyword/default arguments before compilation. Resolve `tl.constexpr` annotations, including postponed annotations, against the actual language sentinel. Support renamed language imports by symbol identity, not the spelling `tl`. Capture only explicitly accepted immutable closure/global constants and include their values in identity; reject dynamic globals, arbitrary function calls and unavailable source with a source-located diagnostic.

The visitor maintains an SSA symbol table and typed expression results. Python arithmetic, comparison, indexing and `None` axes produce IR; unsupported nodes raise errors. Initially accept straight-line expressions, constexpr branches, and canonical counted K loops with a recognized dot accumulator. Runtime branches/while loops, arbitrary loop-carried state and side effects are explicit errors. Positional and keyword mask/other/acc forms have identical binding rules; duplicate/unknown keywords fail.

Recognize affine accesses and recover whole structured kernels only after proving: each logical output has exactly one writer, the grid covers the declared domain, masks preserve bounds, and reductions use the declared accumulator/numerical order. Native mapping owns execution scheduling after this proof. Source BLOCK choices become constrained schedule choices with realization receipts; they cannot be silently discarded or freely retuned. Explicit one-dimensional flattened indexing may recover a rank-two operand only from its declared shape and proven bijective indexing, never by inventing an ABI rank change.

Elementwise B starts with shapes divisible by declared blocks, matching the current adapter. Non-divisible shapes are a separate bounded extension with explicit tails and canary tests; they are rejected until that task passes. Invalid or partial grid coverage fails rather than duplicating/skipping writes. Physical element strides and logical memref subview strides are separate quantities; recovery and legacy pass repair must not multiply both.

## 6. Native API and ownership

Use a small shared C API (`LLKFrontendCAPI`) loaded by ctypes, avoiding an initial pybind/MLIR-Python dependency. Native request handling lives in `LLKFrontend`, above mapping/runtime; generic mapping stays independent of Python and JIT orchestration.

Proposed public C types/functions (the implementation plan fixes their complete fields):

```c
typedef struct LLKExecutable LLKExecutable;
typedef struct LLKError LLKError;
int llk_frontend_compile_v1(const char *json, size_t length,
                            LLKExecutable **out, LLKError **error);
int llk_frontend_invoke_v1(LLKExecutable *handle,
                           const LLKBuffer2D *inputs, size_t input_count,
                           const LLKBuffer2D *outputs, size_t output_count,
                           LLKError **error);
void llk_frontend_release_v1(LLKExecutable *handle);
```

`LLKBuffer2D` mirrors the native checked descriptor and carries element type and allocation byte extent. Validate arity, dtype, shape, strides, offset, allocation bounds and forbidden overlap before entering JIT code. Errors are owned, explicitly freed and carry stable code/message/source span; no C++ exception crosses C. Native handle lifetime is opaque, refcounted internally, and Python closes once via an idempotent context manager/finalizer. A Python call holds a strong handle reference and lock through ctypes invocation.

Use the Python buffer protocol for the no-Torch rank-two f32 path and an explicit Torch CPU tensor adapter for real Torch tensors. Reject unsupported providers, devices, subclasses and invalid metadata. FakeTensor/meta handling must never request a data pointer or invoke the C API. Do not expose raw arbitrary addresses as the default public Python interface.

## 7. Public API and cache behavior

Keep `@llk.jit` and `@llk.jit()`; add keyword-only compile options. `kernel[grid](*args, **kwargs)` is the block-programming form; `kernel(..., grid=...)` is equivalent. Omission of a required grid is an error. Expose `kernel.compile(signature, constexprs, grid)` for precompilation, returning `CompiledKernel`, with `frontend_ir`, `llkir`, `micro_ir`, `report`, and `close()`.

Deprecate fake TTIR output. `kernel.ttir()` returns genuine upstream TTIR only for the upstream adapter and otherwise raises `UnsupportedFeatureError` with guidance to `frontend_ir`. Do not silently relabel private JSON as Triton IR. Assembly is optional and reports unavailability honestly.

Replace the single compiled flag with a process-local bounded LRU (default 128 entries), deduplicate concurrent misses by full request identity, and retain executable ownership independently from the cache so eviction cannot free an active call. Failed compilations do not poison the success cache. Include source/closure content, concrete signature, constexprs, launch, numerical policy, compiler/API version, target content and source-adapter/search choices in identity. Disk/object caching and measured-record persistence are separate work coordinated with #161–#164.

## 8. Torch and Inductor

`import llk.torch` registers `llk` and `llk_inductor` idempotently using named arguments and does not reset Dynamo. `llk` is a strict supported-graph backend: normalize captured graphs through AOTAutograd's ATen interface, recognize accepted graphs, allocate outputs and invoke native handles. Unsupported nodes identify the exact FX node; this backend does not silently return `gm.forward`.

`llk_inductor` clones the captured graph, replaces supported, effect-safe subgraphs with registered `llk` functional custom ops, and delegates the remainder to version-qualified Inductor `compile_fx`. This route deliberately allows Inductor to compile unmatched operations and records partition decisions. Rewrites are scoped to this backend invocation; no global monkeypatches or replacement of the CPU device backend.

Register `llk::add`, `llk::mul`, `llk::matmul` and `llk::swiglu` functional operators with truthful schemas, fresh outputs, CPU implementations and FakeTensor functions. Reuse one native request builder across decorator, Dynamo and custom-op routes. Automatic SwiGLU replacement initially accepts only f32 graphs with a matching native numerical capability. The current `llk.fused_swiglu` operation is BF16-only; PF11 explicitly extends its verifier/type contract and native lowering to uniform f32 before enabling these rewrites, preserves the old BF16 contract and rejects mixed storage dtypes. In particular, bf16 Torch mm→SiLU→mul rounds at the two mm results and SiLU/multiply boundaries; the current f32-accumulator MicroIR epilogue does not preserve that behavior, so decline that Inductor rewrite and reject it in strict LLK until explicit intermediate casts are realized and verified. Even f32 activation/reduction policies must be checked before fusion; add an exact native realization where needed rather than quietly enabling approximation. First training support covers add/multiply/matmul with registered ATen backward formulas; transpose materialization in backward is explicit. SwiGLU training remains rejected until its numerical contract and gradient implementation have a separate acceptance task. Never count eager gradients as LLK-compiled backward evidence.

Required tests include `torch.library.opcheck`, FakeTensor no-pointer checks, `torch.compile(fullgraph=True, backend="llk_inductor")`, supported/unsupported mixed graphs, forward/backward comparisons and scoped compile evidence. AOTInductor export/serialization is not claimed by the external-call milestone; it needs a portable native artifact/runtime contract later.

## 9. Real upstream Triton adapter

Implement this after B/C rather than renaming the AST wrapper. Qualify Triton 3.5.0's real backend and driver interfaces, then provide compiler-stage, target, loader and launcher adapters. Genuine TTIR is obtained through upstream specialization and exported as an `UnboundFrontendTemplate(schema_version=1, required_runtime_grid=True)` containing pointer/scalar parameter types, constexprs, typed affine access/mask/dataflow, source identity and required alignment/argument constraints. Upstream compile-time pointer specialization does not provide complete tensor shapes or the normalized launch grid. The driver retains the template; its CPU launcher binds real tensor metadata, scalar values and the runtime grid, proves coverage, constructs `FrontendProgram` and then compiles/invokes through the shared LLK service. The first accepted upstream kernel is CPU add/multiply over the same compact rank-two storage and proven indexing/coverage rules.

Use process-isolated TTIR export when LLVM dialect/toolchain versions differ. Only neutral serialized values cross this boundary; no borrowed MLIR context/module/type, C++ object or executable pointer does. The parent launcher uses its own LLK handle, buffer validation and execution report. Maintain two cache stages: upstream template identity (TTIR/constexpr/pointer-specialization/toolchain content), then native specialization identity (template plus concrete tensor metadata/scalars, launch, numerical policy and native environment). Never reuse a shape-bound handle based only on upstream pointer signatures. Reports are launch-time records exposed by `last_launch_report()` in the activation context; genuine upstream `.asm["ttir"]` remains compile-time inspection. Serialize scoped activation against Triton's process-global active driver and restore it on exit/error; concurrent independent activation contexts are outside v1. Qualify importing Torch/Triton/LLK in either order, including environments where an upstream Python import loads native components; use subprocess isolation for compilation and keep the supported combination disabled if native-library coexistence fails. Do not link arbitrary Triton wheel LLVM objects into LLK's LLVM 22/24 build.

Toolchain qualification produces a lock manifest with exact source revisions, LLVM revision, exporter schema and supported kernel corpus. Failure leaves D unimplemented and the parent issue open. A source-extraction adapter, simulated `BaseBackend`, subprocess-only parse result, or GPU execution cannot substitute for a successful upstream JIT-to-LLK CPU launch.

## 10. Legacy pass repairs and acceptance

Repair double-applied strides, reject unsupported grid returns, validate exact generic-op schemas, reject missing atomic runtime functions and unsupported await/reduction/broadcast/transpose cases, and enforce no residual `tt.*` at executable boundaries. Keep unknown-dialect parsing limited to the explicitly experimental `llk-opt` fixtures; production frontend JSON goes through its typed importer. `llk-compile` gains an explicit frontend-json input mode instead of claiming that unregistered names are upstream Triton support.

The old staged execution skeleton must become a meaningful executable test for supported input, or an expected unsupported diagnostic test until its actual adapter lands. Remove verifier-disabling success tests. CI must distinguish compiler-free frontend tests, native reference execution, real Torch/Inductor execution and selected AVX2 execution. Required lanes fail on zero collected cases or skipped required execution. macOS lacking an executable ARM target runs reference tests only and never counts as selected ARM evidence.

For strict elementwise f32, require exact output agreement, including signed zero where the source promises it. For f32 matmul use `rtol=1e-4, atol=1e-5`; bf16-input/f32-accumulation fixtures use `rtol=2e-2, atol=2e-2`. NaN/Inf behavior is tested against the stated numerical policy and never hidden by tolerance. Memory tests use guard regions, malformed descriptors and repeated invocation/release.

No product implementation is authorized by this planning document alone. Implementation will use isolated worktrees, independently reviewable commits and the task-level test gates in the linked plan; this turn publishes the proposal and tracking issue.

Planning review resolved three cross-component dependencies: upstream launch-time metadata requires an unbound template and second specialization cache; counted K loops require typed regions and yields; f32 SwiGLU requires a native op/type extension and exact numerical-contract matching. These are implementation requirements, not evidence that the proposed paths already run.
