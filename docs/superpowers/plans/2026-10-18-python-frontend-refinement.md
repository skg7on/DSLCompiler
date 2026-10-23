# Python Frontend Refinement Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. Select execution mode only when implementation is requested.

**Goal:** Close the eleven reviewed Python frontend gaps and deliver verified LLK decorator, Dynamo, Inductor external-kernel and upstream Triton entry paths into the shared LLK/MicroIR compiler/runtime.

**Architecture:** A typed, versioned frontend program feeds one native service using existing source adapters, mapping/search and checked mapped invocation. LLK AST capture, Torch capture and a separately qualified upstream Triton adapter use that boundary. Inductor retains graph ownership and calls registered LLK external kernels.

**Tech Stack:** Python 3.11/3.12, C++20, LLVM/MLIR 22.1.8 and local 24+, ctypes/C ABI, pytest/GTest/CTest, Torch 2.9.0, optional Triton 3.5.0 on Linux x86_64.

**Spec:** [Proposed refined design](../specs/2026-10-18-python-frontend-refinement-design.md).

**Review:** [Revision-pinned gap assessment](../../reviews/2026-10-18-python-triton-inductor-gap-assessment.md).

**Baseline/status (2026-10-23):** Original planning baseline `13cef9c`. PF01
([PR #233](https://github.com/skg7on/DSLCompiler/pull/233)) and PF02
([PR #241](https://github.com/skg7on/DSLCompiler/pull/241)) are merged; PF03 starts
from main `29b00eb`. PF03's checked steps record branch implementation and
verification in the [PF03 evidence](../../reviews/2026-10-23-pf03-typed-frontend.md).
PF04–PF20 remain proposed and unchecked. Paths in those later tasks describe
future files unless marked Modify.

**Current local environment:** use the project `.venv` (Python 3.14.8 / Torch
2.14.1 / NumPy 2.5.4). PF01 passes 21 tests with that interpreter as well as
21 per original Python 3.11/3.12 + Torch 2.9.0 reference configuration. See the
[current frontend guide](../../manual/source/python-frontend.rst) and
[verification record](../../reviews/2026-10-19-pf01-python-environment.md).

## Global constraints

- Python 3.11 and 3.12; C++20. CI LLVM/MLIR 22.1.8 and the repository's local LLVM 24+ lane. Use version guards where APIs differ.
- Initial real-framework compatibility baseline: Torch 2.9.0; optional upstream Triton 3.5.0 on Linux x86_64. Qualify additional releases independently.
- CPU, positive static rank-two shapes, compact row-major, zero storage offset, disjoint input/output storage. No automatic input copies or dtype conversion.
- Synchronous calls. Keep input/output owners and the native executable alive until invocation finishes. Initial execution is serialized per executable.
- `backend="reference"` is explicit for host validation. `backend="selected-target"` requires supported host ISA and never substitutes reference execution.
- First release specializes concrete shapes. Dynamic ABI support remains coordinated with #191–#199; production measurement persistence remains #161–#164.
- Every design/implementation write uses an isolated worktree with its own build directory. Use `Op::create`, `mlir::cast`/`dyn_cast` and existing LLVM guards; do not copy another worktree's binaries.
- No unsupported opcode, syntax, layout, effect, target or runtime function may fall back silently. No success test disables MLIR verification. Additions to the supported envelope require numerical and negative coverage.
- Required execution tests cannot skip or collect zero cases. Reference and selected-target evidence remain separate. The parent issue stays open until the actual upstream launch milestone is met.

## Review focus

Each condition is assigned to a test-owning task; these are additional likely user inputs, not exceptions to the design.

1. Postponed annotations, renamed language imports, immutable closure constants and source-unavailable functions: bind deterministically or report a source-located error (PF03/PF04).
2. Grid callables with side effects, duplicate/partial coverage, zero extents and mismatched masks: evaluate once and reject invalid coverage before invocation (PF03/PF08/PF10).
3. CPU tensor views/subclasses, offset/overlap, malformed buffer sizes and FakeTensor inputs: reject unsupported real buffers and never dereference fake buffers (PF07/PF08/PF14).
4. Concurrent compilation, cache eviction during an active call, close during invocation and failed compilation: preserve ownership and permit a clean retry (PF07/PF09).
5. Mixed supported/unsupported FX graphs and native-library import-order/version conflicts: preserve effects under Inductor and reject unqualified upstream combinations explicitly (PF13/PF15/PF17–PF19).

## File ownership and sequencing

This is one dependency-ordered plan because all four entry paths share the same program, ownership and execution-report contracts. Each task is a separately reviewable commit/PR unit. Do not combine the final upstream adapter with the first native executable milestone.

| Area | New files / responsibility | Existing files modified |
|---|---|---|
| Packaging/diagnostics | `python/pyproject.toml`, `python/llk/_errors.py`, `python/llk/_capabilities.py`, `python/llk/compatibility.json` | `python/llk/__init__.py`, `python/llk/torch/{__init__,_dynamo_backend}.py` |
| Typed frontend | `python/llk/frontend/{__init__,_ir,_binding,_serialize}.py`, `docs/design/python-frontend-v1.schema.json` | `python/llk/language/{__init__,_ast_compiler,_triton_ir}.py` |
| Native import/service | `include/LLK/Frontend/{Program,Importer,Compiler}.h`, `lib/Frontend/{Program,Importer,Compiler}.cpp`, `lib/Frontend/CMakeLists.txt` | root `CMakeLists.txt`, shared candidate-completion seams |
| C ABI/runtime | `include/LLK/CAPI/Frontend.h`, `lib/CAPI/{Frontend.cpp,CMakeLists.txt}`, `python/llk/runtime/{_buffers,_cache}.py` | `python/llk/runtime/{_ffi,_jit}.py` |
| Torch composition | `python/llk/torch/{_compat,_fx_importer,_ops,_inductor_backend,_autograd}.py` | existing Dynamo registration/backend |
| Upstream adapter | `python/llk/triton/{__init__,_backend,_driver,_export_worker}.py`, `tools/triton-export/`, version/toolchain lock | packaging and optional CI only |
| Acceptance | `test/python/` suites, `test/Frontend/`, `test/CAPI/`, `test/Execution/python_frontend.cpp`, `test/python/compat/` | CMake test registration, CI, feature/manual pages |

Exact task paths are authoritative. Existing APIs to reuse: `instantiateCandidate` in `Conversion/MicroMapping/CandidateInstantiation.h`, the `WorkloadContract` overload of `runMappedTuningSession`, `compileMappedKernel`, `MappedExecutable::invoke` and `validateMappedInvocation`. Re-audit #156/#157 before editing shared replay/capability schemas.

Dependency chain: PF01 → PF03 → PF04 → PF05 → PF06 → PF07 → PF08 → PF09. PF02 can proceed independently after PF01. PF10/PF11 extend the executable domain. PF12/PF13 follow PF09; PF14 → PF15 → PF16 follow PF11/PF13. PF17 → PF18 → PF19 follow the established shared API. PF20 integrates all completed milestones.

## Interface contract used by the tasks

All names below are proposed. Later tasks consume them exactly; changing a name requires updating the spec and every consumer.

**Python:** frozen `TensorSpec`, `LaunchSpec`, `SourceSpan`, `Value`, `Operation`, `FrontendProgram`, `CompileOptions`, `CompileRequest`, `CompilationReport` records in `frontend/_ir.py`; its `native_request_identity` is the full canonical native specialization identity; define `NumericalContract` with storage/compute/accumulator dtype, ordered cast sites/rounding, activation expression/approximation, reduction-order/reassociation/FMA flags and subnormal policy. Define `counted_loop` regions with typed induction/block args, lower/upper/positive step, init/iter values, body and yield; schema and native parser share this representation. Other regions remain unsupported; binding returns `KernelBinding` from `_binding.py`. Add `config_root: str` to options alongside the design's target/backend/search/numerical fields. The default is an installed package resource root; development override is explicit and participates in identity by file content, not just path.

```python
bind_kernel(fn, args: tuple, kwargs: dict, grid: tuple | Callable,
            options: CompileOptions) -> KernelBinding
compile_kernel(fn, binding: KernelBinding) -> FrontendProgram
serialize_request(request: CompileRequest) -> bytes
compile_request(request: CompileRequest) -> CompiledKernel
CompiledKernel.invoke(inputs: tuple, outputs: tuple) -> None
CompiledKernel.report: CompilationReport
CompiledKernel.close() -> None
```

`KernelBinding` contains ordered provider objects held by the caller, ordered `TensorSpec` ports, bound constexprs, accepted closure/global constants, normalized launch and options. Providers/addresses never serialize. `CompileOptions` defaults: target=`x86-avx2`, backend=`reference`, search_mode=`deterministic`, max_candidates=64, top_k=1. A selected-target caller opts in explicitly.

**Native service:** namespace `llk::frontend`; `parseRequest(StringRef json) -> Expected<FrontendRequest>`, `importProgram(const FrontendRequest &, MLIRContext &) -> Expected<ImportedProgram>`, `compileFrontend(const FrontendRequest &) -> Expected<FrontendCompilation>`. `ImportedProgram` owns its module plus root selection, canonical `WorkloadContract`, finite `SearchSpace`, and ordered public-to-native port map. `FrontendCompilation` owns `MappedExecutable` and copies of the report/inspection strings; no borrowed MLIR objects escape their owner.

The native destination used by the elementwise source adapter is a no-read input in addition to the written result. Map that internal port to an independently allocated native scratch descriptor, not an alias of the public output/input. The service records this adaptation and owns its lifetime. Do not weaken ABI disjointness or expose an extra unexplained Python argument. Remove the internal destination port only through a separately verified source-adapter/ABI change.

**C API:** opaque `LLKExecutable`, `LLKError`; `LLKBuffer2D {void *allocated; void *aligned; int64_t offset; int64_t sizes[2]; int64_t strides[2]; uint32_t element_type; uint64_t allocation_bytes;}`. Fix element tags F32=1, BF16=2, F16=3, I32=4, I8=5. `LLKString {const char *data; size_t length;}` is borrowed until handle/error release; Python copies immediately. Return 0 on success and 1 on error, set output handles to null on failure, and never throw through C.

```c
uint32_t llk_frontend_api_version(void); /* returns 1 */
int llk_frontend_compile_v1(const char *, size_t, LLKExecutable **, LLKError **);
int llk_frontend_invoke_v1(LLKExecutable *, const LLKBuffer2D *, size_t,
                          const LLKBuffer2D *, size_t, LLKError **);
int llk_frontend_report_v1(LLKExecutable *, LLKString *, LLKError **);
int llk_frontend_inspect_v1(LLKExecutable *, const char *stage,
                           LLKString *, LLKError **);
LLKString llk_frontend_error_json_v1(const LLKError *);
void llk_frontend_error_free_v1(LLKError *);
void llk_frontend_release_v1(LLKExecutable *);
```

Inspection stages: `frontend`, `llk`, `micro`; unavailable `asm` returns `inspection_unavailable`. Stable initial frontend codes include `unsupported_frontend_call`, `invalid_launch_coverage`, `unsupported_training_contract`, `native_backend_unavailable`, `unsupported_tail_policy` and `upstream_toolchain_unqualified`; report unsupported stages separately from malformed schemas. Native errors encode `code`, `message`, optional `file`, `line`, `column`, and failing stage. Python exceptions derive from `LLKError`: `FrontendSyntaxError`, `UnsupportedFeatureError`, `TensorContractError`, `CompilationError`, `ExecutionError`. Compile-cache defaults: 128 entries, full-identity equality, SHA-256 index, one in-flight compilation per identity, no successful entry on failure.

## Implementation build setup

For Python frontend development, use the project `.venv` interpreter by absolute
path from the isolated worktree and install `./python[test]` there as described in
the current guide. Preserve separate Python 3.11/3.12 + Torch 2.9.0 environments
for the reference matrix; `[torch-test]` selects that reference Torch version.
PF01 needs no native build. For later native tasks, configure a fresh worktree
build against the approved LLVM tree and build only the named targets. PF07
introduces `LLK_BUILD_PYTHON_CAPI`; from PF07 onward configure with
`-DLLK_BUILD_PYTHON_CAPI=ON` and export `LLK_FRONTEND_LIBRARY` to that worktree's
actual shared-library output. PF20 registers those native settings in CMake/CI.
The current `.venv` qualification does not establish those future native lanes.

## Milestone A — repair boundaries and establish the shared contract

### PF01: Package the prototype and repair real Torch registration

**Files:** Create `python/pyproject.toml`, `python/llk/_errors.py`, `python/llk/_capabilities.py`, `python/llk/compatibility.json`, `test/python/test_torch_registration.py`, `test/python/test_package_install.py`; modify `python/llk/torch/_dynamo_backend.py`, `python/llk/torch/__init__.py`.

**Interfaces:** Produce named exception classes above and `register() -> None`. Until native compilation lands, backend execution raises `UnsupportedFeatureError(code="native_backend_unavailable")` rather than returning `gm.forward`. Registration uses the same singleton instance for public reporting and the registry.

- [x] Write real-Torch subprocess tests for import, repeated `register()` and reload; spy on `torch._dynamo.reset` and assert zero calls. Assert lookup returns the LLK callable. With Torch absent, `import llk` and `import llk.torch` succeed without registration. Add the package install test in a clean venv.

Pinned regression assertions (local test fixture names):

```python
assert callable(torch._dynamo.lookup_backend("llk"))
assert reset_spy.call_count == 0
```

- [x] Run `python -m pytest test/python/test_torch_registration.py test/python/test_package_install.py -q`; confirm failures include the existing `eager` keyword/API behavior, not missing Torch in the required Torch lane.
- [x] Add setuptools packaging from `python/`, optional Torch test dependencies and the explicit initial compatibility manifest. Use `register_backend(compiler_fn=instance, name="llk")`, an idempotent ownership check and no global reset. Do not overwrite an unrelated registered backend with the same name.
- [x] Run the tests in Torch-present and Torch-absent environments; wheel import and named lookup must pass; invoking the prototype must explicitly report its unavailable native capability.
- [x] Stage this task's files and commit `fix: package frontend and repair Dynamo registration`.

### PF02: Repair unsafe legacy Triton pass behavior

**Files:** Modify `lib/Conversion/TritonToLLK/{BlockPointerToVector,GridToForall,TritonCPUVerifier,AtomicToLLRT,SharedMemToScratch,TritonToStructured}.cpp`, `test/Conversion/TritonToLLK/{block_ptr_to_vector,grid_to_forall_return,cpu_verifier_pass,triton_full_pipeline}.mlir`, `test/FileCheck.cmake`; create `test/Frontend/legacy_typed_contract.cpp`, `test/Frontend/CMakeLists.txt`; register the category in `test/CMakeLists.txt`.

**Interfaces:** Legacy input remains an explicitly experimental generic-op format. Unsupported ops fail with `triton_unsupported_op`; malformed schemas use `triton_invalid_schema`; result-returning grid functions use `triton_grid_result_unsupported`. No new executable support is claimed for atomics/await.

- [x] Add stride assertions (`strided<[64, 1]>`, never `[4096, 1]`), typed operand/attribute arity failures, nonzero/negative offsets, supported masks and invalid mask types. Replace the return fixture's verifier-disabled success with expected diagnostic. Add a guarded-buffer numerical load/store case at zero offsets.

Pinned regression assertions (local test fixture names):

```python
assert "strided<[64, 1]" in lowered_ir
assert "strided<[4096, 1]" not in lowered_ir
```

- [x] Run `cmake --build build --target LLKFrontendTests llk-opt` and `ctest --test-dir build -R 'FrontendLegacy|BlockPointer|GridToForall|TritonCPU' --output-on-failure`; show the old stride/return failures.
- [x] Distinguish physical base layout from logical subview steps: compact memref input uses unit logical steps, validates declared physical strides against the base, and rejects unsupported layouts. Validate exact array lengths, rank, operands/results and mask element types before indexing/casting. Reject unsupported atomic/runtime and await/reduction/broadcast/transpose forms; require residual-op rejection at an executable boundary. Reject escaping grid results before mutation.
- [x] Rerun the targeted tests with verification enabled. The legacy full-pipeline fixture must now separate executable-supported cases from expected unsupported cases. Verify canaries and expected row values, not just operation names.
- [x] Commit `fix: validate legacy Triton lowering and repair block addressing`.

**Implementation evidence:** [PF02 safety verification](../../reviews/2026-10-22-pf02-legacy-triton-safety.md).

### PF03: Define typed program, deterministic serialization and argument binding

**Files:** Create `python/llk/frontend/{__init__,_ir,_binding,_serialize}.py`, `docs/design/python-frontend-v1.schema.json`, `test/python/frontend_cases.py`, `test/python/test_frontend_binding.py`, `test/python/test_frontend_schema.py`.

**Interfaces:** Produce the frozen records, `KernelBinding`, `bind_kernel` and `serialize_request`. `frontend_cases.py` defines `make_add_program(shape=(32,64), block=(8,16))` with two read inputs, one write output, f32 and exact coverage, plus malformed-request variants.

- [x] Test positional/keyword/default binding, postponed constexpr annotations, renamed `llk.language`, closure constants and duplicate/unknown args. Include a nested counted-K-loop fixture with typed init/yield and cast sites; reject escaping region values, yield-type mismatch and erased cast metadata. Pin canonical bytes against a literal fixture; permuted dict order is identical, permuted port order differs. Assert grid callable runs once and its result is a normalized three-tuple. Reject bool grid extents, zero/negative shape, axes >2, unknown schema fields/versions and non-finite constexpr floats.

Pinned regression assertions (local test fixture names):

```python
assert serialize_request(request_a) == serialize_request(same_request_reordered_keys)
assert grid_call_count == 1
```

- [x] Run `python -m pytest test/python/test_frontend_binding.py test/python/test_frontend_schema.py -q`; verify the intended missing records/functions fail.
- [x] Implement records and validation. Use signed 64-bit ranges for serialized indices/extents and checked products for allocation sizes. Bind values by sentinel/symbol identity, preserve port order and never serialize runtime addresses/providers. Emit source-span diagnostics for binding failures.
- [x] Rerun both suites and round-trip the canonical fixture; assert source/constant/launch/numerical changes alter identity, while pointer relocation does not.
- [x] Commit `feat: define typed frontend program and binding contract` (`b78ee15`).

### PF04: Replace lossy AST walking with typed SSA capture

**Files:** Modify `python/llk/language/{__init__,_ast_compiler,_triton_ir}.py`, `test/python/{test_ast_compiler,test_triton_ir_builder,test_jit_invocation}.py`; create `test/python/test_ast_semantics.py`, `test/python/test_frontend_diagnostics.py`.

**Interfaces:** Produce `compile_kernel(fn, binding: KernelBinding) -> FrontendProgram`. `_triton_ir.py` becomes a compatibility facade; private JSON is named frontend IR and cannot be returned as TTIR. Public language stubs exist for accepted syntax and raise outside capture.

- [ ] Add exact operation/value assertions for canonical masked add and multiply, offsets, comparison, `[:, None]`, constexpr branches and positional/keyword masks. Assert unknown math/calls/runtime branches/while loops, source-unavailable functions and mutable captured globals raise with line/column. Add `test_keyword_mask_and_other_survive_capture` and `test_unknown_call_cannot_equal_empty_kernel`.

Pinned regression assertions (local test fixture names):

```python
assert load_op.operands == (pointer_value.id, mask_value.id, other_value.id)
assert unsupported_error.code == "unsupported_frontend_call"
```

- [ ] Run `python -m pytest test/python/test_ast_semantics.py test/python/test_frontend_diagnostics.py -q`; verify old stringification/silent dropping fails.
- [ ] Implement a typed environment and explicit visitor methods; resolve operands recursively and bind assignment results. Type-check broadcast/indexing and mask/other rules. Only constexpr branches are evaluated here; canonical counted K loops are represented as typed regions, not unrolled or walked once. Other runtime control flow fails. Capture explicit conversions/rounding and ordered numerical-policy metadata. Validate the resulting program before returning it.
- [ ] Run the new suites and updated original tests. No assertion may count `tt.load` substrings as valid IR; compare canonical typed operations and meaningful negative diagnostics.
- [ ] Commit `feat: capture typed SSA and preserve frontend semantics`.

### PF05: Import checked frontend programs into registered structured IR

**Files:** Create `include/LLK/Frontend/{Program,Importer}.h`, `lib/Frontend/{Program,Importer}.cpp`, `lib/Frontend/CMakeLists.txt`, `test/Frontend/{program,importer}.cpp`; modify root `CMakeLists.txt`, `test/Frontend/CMakeLists.txt`.

**Interfaces:** Produce `FrontendRequest`, `ImportedProgram`, `parseRequest`, `importProgram`. Parser supports only schema version 1 and uses LLVM JSON with strict field/type/range checks. Imported modules contain registered `func`, `arith`, `tensor`, `linalg` or `llk` operations only.

- [ ] Add GTests importing PF03's canonical fixture: verify the module, root port effects and `linalg.generic` scalar add/mul. Reject undefined/duplicate ids, wrong mask shape, extra effects, multiple writes per element, incomplete/duplicate grid coverage and arbitrary pointer expressions. Assert no operation's name starts with `tt.`.

Pinned regression assertions (local test fixture names):

```cpp
EXPECT_TRUE(succeeded(mlir::verify(*imported.module)));
EXPECT_EQ(unsupportedGridError.code, "invalid_launch_coverage");
```

- [ ] Build/run `LLKFrontendTests --gtest_filter='FrontendProgram.*:FrontendImporter.*'`; record expected red assertions.
- [ ] Implement typed validation and affine access/coverage proof before structured recovery. Recover only full-domain rank-two add/mul accepted by `ElementwiseSourceAdapter`. Carry BLOCK choices into source-owned parameter roles and preserve the numerical contract. Unsupported programs fail before MLIR construction or with verified diagnostics.
- [ ] Rerun GTests plus parse/print/parse/verify of the emitted registered module. Prove each source choice has a declared source or mapping owner; inspect root port order.
- [ ] Commit `feat: import frontend programs into structured workload sources`.

### PF06: Share candidate selection and native mapped compilation

**Files:** Create `include/LLK/Frontend/Compiler.h`, `lib/Frontend/Compiler.cpp`, `test/Frontend/compiler.cpp`; modify `lib/Frontend/CMakeLists.txt`, `include/LLK/Conversion/MicroMapping/MappedTuningSession.h`, `lib/Conversion/MicroMapping/MappedTuningSession.cpp` only as needed for the shared completion seam.

**Interfaces:** Produce `compileFrontend` and `FrontendCompilation`. Add `compileMappedTuningSelection(ModuleOp, const perf::SearchSpace &, const workload::WorkloadContract &, const mapping::MappingTarget &, const MappedTuningCandidate &, const MappedTuningOptions &) -> Expected<llk::MappedCompilation>` to the mapped-tuning header. It realizes the ranked source candidate with `instantiateCandidate`, checks source/binding/plan identities, then calls `compileMappedKernel`; CLI/replay consumers can reuse it.

- [ ] Add full/tiled add/mul native compile tests with a literal expected output and report assertions. A changed source/binding/target content must reject stale selection. Assert no synthetic M/N/K contract enters the elementwise route. Verify selected mode fails on unavailable ISA and reference mode identifies itself.

Pinned regression assertions (local test fixture names):

```python
assert report["backend"] == "reference"
assert report["source_adapter"] == expected_elementwise_adapter_key
```

- [ ] Run `LLKFrontendTests --gtest_filter='FrontendCompiler.*'` and existing candidate/mapped-tuning tests; establish failures for the missing service.
- [ ] Call the contract overload of `runMappedTuningSession`, retain top_k=1 under explicit search limits, and use the shared selection helper. Preserve native counters/hashes. Implement the internal no-read destination descriptor adaptation specified above with owned storage and an explicit port map. Keep `LLKMapping` free of Python/runtime orchestration dependencies.
- [ ] Rerun frontend compile/invoke and existing source-candidate/tuning tests. Native outputs must agree exactly for add/mul and no stale selected state may pass. Confirm internal destination scratch does not alias public buffers.
- [ ] Commit `feat: expose shared frontend compilation through mapped selection`.

## Milestone B — execute and specialize the LLK decorator

### PF07: Expose versioned C API with checked invocation and ownership

**Files:** Create `include/LLK/CAPI/Frontend.h`, `lib/CAPI/{Frontend.cpp,CMakeLists.txt}`, `test/CAPI/{frontend.c,frontend_errors.cpp,CMakeLists.txt}`; modify root/test CMake registration; introduce `LLK_BUILD_PYTHON_CAPI` (default OFF, explicitly ON in native/frontend test builds).

**Interfaces:** Produce the exact C layout/functions in the interface contract and shared library target `LLKFrontendCAPI`. Export only the declared symbols; keep exception/error/handle ownership native.

- [ ] Write a C consumer compiling PF03's serialized fixture and invoking f32 arrays. Assert error-out initialization, ABI tag/layout offsets, wrong arity/dtype/shape/stride/byte extent and forbidden overlap rejection. Add repeated compile/invoke/release and simultaneous invokes on distinct outputs; reject closed Python handles in PF08 rather than dereferencing freed C pointers.

Pinned regression assertions (local test fixture names):

```c
assert(llk_frontend_api_version() == 1);
assert(llk_frontend_invoke_v1(handle, bad_inputs, 2, outputs, 1, &error) == 1);
```

- [ ] Run `cmake --build build --target LLKFrontendCAPI LLKCAPIFrontendTests` and `ctest --test-dir build -R FrontendCAPI --output-on-failure`; verify the missing API is the failure.
- [ ] Marshal to native `InvocationBuffer2D`, call existing validation, and catch exceptions at the C boundary. Implement borrowed report/error string accessors and explicit free/release. Protect invocation/scratch with handle locking and document raw C ownership: no calls after release; raw double release is invalid, Python close is idempotent.
- [ ] Rerun C/C++ consumers under AddressSanitizer/LeakSanitizer on supported Linux tooling. Assert guard regions, no leaks after error paths and no exceptions cross C.
- [ ] Commit `feat: add checked C API for frontend compilation and invocation`.

### PF08: Implement Python buffer adapters, launch and native JIT calls

**Files:** Modify `python/llk/runtime/{_ffi,_jit}.py`, `python/llk/__init__.py`; create `python/llk/runtime/_buffers.py`, `test/python/{test_native_jit,test_buffer_contract,test_launch_contract}.py`.

**Interfaces:** Produce `compile_request`, `CompiledKernel`, `CompiledKernel.invoke/close`, decorator option handling and both launch forms. Implement `LLKBuffer2D` as ctypes with exactly the native field order; keep providers strongly referenced during calls.

- [ ] Add a real rank-two buffer-protocol add kernel test using shapes 32×64, blocks 8×16; assert exact outputs, unchanged inputs, guard bytes and `report.backend`. Test `kernel[grid]` equals `kernel(..., grid=grid)`, precompilation, close twice and close while another thread invokes. Reject missing grid, partial coverage, tensor views/offsets/overlap and mismatched allocation bytes before JIT entry.

Pinned regression assertions (local test fixture names):

```python
assert output_values == expected_add_values
assert input_bytes_after == input_bytes_before
assert native_invoke_spy.call_count == 1
```

- [ ] Run `LLK_FRONTEND_LIBRARY=$PWD/build/lib/libLLKFrontendCAPI.so python -m pytest test/python/test_native_jit.py test/python/test_buffer_contract.py test/python/test_launch_contract.py -q` (use `.dylib` on macOS); confirm the current NotImplementedError fails the real invocation assertion.
- [ ] Load the explicit development path or installed package resource, check API version, marshal Python exceptions and implement context-managed handle ownership. Use a strong reference plus lock through ctypes calls; never cache raw caller buffers. Preserve void pointer-output semantics and keep `.ttir()` explicitly unavailable for this frontend.
- [ ] Rerun all three suites with native-call counters. A test passes only if a caller-owned buffer changes through native code and the compilation report matches the executed handle.
- [ ] Commit `feat: execute llk decorators through the checked native bridge`.

### PF09: Implement bounded specialization cache and inspection reports

**Files:** Create `python/llk/runtime/_cache.py`, `test/python/{test_specialization_cache,test_jit_lifetime,test_inspection}.py`; modify `_jit.py`, `_ffi.py`, native report generation.

**Interfaces:** Produce `CompilationCache(max_entries=128).get_or_compile(request, compile_fn) -> CompiledKernelLease`; leases own handles independently of cache membership. Add `JitFunction.cache_info()` returning hits/misses/compilations/evictions and inspection properties from the actual native artifact.

- [ ] Assert same full identity compiles once; changed shape/dtype/constexpr/closure/grid/numerical/target contents/ABI version compiles separately. Force identical digest buckets with unequal full identities and require separate artifacts. Evict an active entry safely; deduplicate eight concurrent misses; retry after a failed compilation; close does not deadlock invocation.

Pinned regression assertions (local test fixture names):

```python
assert kernel.cache_info().compilations == 1
assert concurrent_successful_invocations == 8
```

- [ ] Run `python -m pytest test/python/test_specialization_cache.py test/python/test_jit_lifetime.py test/python/test_inspection.py -q` with the native library; verify missing cache behavior fails.
- [ ] Implement pending/ready/failed state transitions, a bounded LRU and strong leases. Native environment identity is queried before lookup and final candidate/plan identity is stored in the artifact; changed config files cannot reuse a handle. Expose reached stages and native counters; unavailable disassembly returns its explicit error.
- [ ] Rerun stress tests and assert compile counts, not timings. No active handle may be freed by eviction; no success report may come from an uncompiled request.
- [ ] Commit `feat: add bounded frontend specialization and executable inspection`.

### PF10: Add proven bounded tails for elementwise kernels

**Files:** Modify `lib/Frontend/Importer.cpp`, `lib/Conversion/Workload/ElementwiseSourceAdapter.cpp`, native tile movement/lowering files only when a declared primitive change is necessary; create `test/Frontend/tails.cpp`, `test/python/test_frontend_tails.py` and register tests.

**Interfaces:** Extend the existing elementwise adapter contract with an explicit supported tail policy and receipt; do not silently remove its present divisibility gate. Reuse bounded movement contracts where delivered; otherwise implement static full-tile plus edge-fragment realization with exact static fragment sizes, keeping every access in bounds.

- [ ] Add 33×65 and 1×7 add/mul with blocks 8×16 and declared masks, comparing exact outputs and guard regions. Test all-false masked lanes, negative/outside offsets and invalid masks. Before the extension, non-divisible inputs must raise `unsupported_tail_policy`.

Pinned regression assertions (local test fixture names):

```python
assert output_values_33x65 == expected_values_33x65
assert prefix_guard_after == prefix_guard_before
assert suffix_guard_after == suffix_guard_before
```

- [ ] Run `LLKFrontendTests --gtest_filter='FrontendTails.*'` plus `python -m pytest test/python/test_frontend_tails.py -q`; verify real tail cases initially fail.
- [ ] Realize interior and statically sized edge fragments with explicit source receipts and native mapping verification. Combine semantic masks with valid bounds; do not create out-of-bounds subviews and rely on a later mask to repair them. Limit accepted masks to those whose memory safety and store coverage are proved.
- [ ] Rerun reference and selected-target numerical/canary cases and source-adapter regressions. Selected-target acceptance must realize the chosen edge implementation with zero fallback or reject that target/shape explicitly.
- [ ] Commit `feat: realize bounded elementwise tails in frontend kernels`.

### PF11: Add matmul accumulation and guarded SwiGLU recognition

**Files:** Modify `python/llk/language/_ast_compiler.py`, `python/llk/frontend/_ir.py`, `docs/design/python-frontend-v1.schema.json`, `lib/Frontend/Importer.cpp`, language stubs, `include/LLK/Dialect/LLKOps.td`, `lib/Dialect/LLK/LLKOps.cpp`, `lib/Conversion/LLKKernel/SwiGLU.cpp`, and the selected SiLU/epilogue emitter/lowering when its numerical capability needs extension; create `test/python/test_frontend_contractions.py`, `test/Frontend/contractions.cpp`, `test/Dialect/swiglu_f32.mlir`. Reuse the existing LLK operator registry rather than adding operator switches to the generic planner.

**Interfaces:** Accept canonical counted K loops with one/two dot accumulators, explicit initial values and numerical contract. Produce LLK matmul or fused SwiGLU only when all shape, access, shared-input and effect predicates hold. Nonzero accumulators and every intermediate cast remain explicit; reject if the chosen native adapter cannot preserve them. Initial automatic SwiGLU fusion is f32-only and requires a native activation/reduction capability matching the source; bf16 Torch intermediate-rounding patterns decline the rewrite or raise in strict LLK. Explicit relaxed decorator semantics use a different NumericalContract and cache key.

- [ ] First pin the current BF16-only SwiGLU verifier failure on uniform f32 inputs/results, retain BF16 behavior and reject mixed types. Add f32 matmul cases M,N,K=(32,64,16) and (7,9,5), a nonzero accumulator negative/positive case according to the adapter contract, and SwiGLU with shared X. Negative cases: different X, additional users/stores, wrong reduction extent, ignored `acc=` keyword, unsupported math mode. Add bf16 Torch fixtures whose rounded mm/SiLU intermediates differ from the current f32 fused epilogue; assert rewrite rejection rather than accepting a loose allclose. Add strict-versus-relaxed activation and explicit-cast tests. Pin f32 tolerance `rtol=1e-4, atol=1e-5`; bf16-input/f32-accumulation tolerance `rtol=2e-2, atol=2e-2` only in a certified lane.

Pinned regression assertions (local test fixture names):

```python
torch.testing.assert_close(result, reference, rtol=1e-4, atol=1e-5)
assert bf16_intermediate_rounding_rewrite.accepted is False
```

- [ ] Run contraction GTests and `python -m pytest test/python/test_frontend_contractions.py -q`; verify missing loop/accumulator behavior fails.
- [ ] Extend the native SwiGLU type/verifier and lowering contract to uniform f32 operands/results while preserving BF16 semantics; keep automatic f32 fusion disabled until its native tests pass. Implement explicit loop-carried SSA, accumulator initialization and typed dot validation, then structured kernel recovery/fusion. Preserve reduction/numerical semantics and explicit cast sites; implement a matching strict-f32 activation/native epilogue if missing, and otherwise reject that selected target. Test tolerances do not grant permission to erase casts or reassociate a source that forbids it. Constrain source tile choices through the existing contraction source adapter and receipts. Reject patterns outside the declared family.
- [ ] Rerun numerical/negative tests and compare frontend-derived MicroIR contract/plan evidence with directly authored native fixtures. Exercise NaN/Inf and activation numerical-policy boundaries.
- [ ] Commit `feat: lower captured contractions and verified SwiGLU patterns`.

## Milestone C — Torch capture and actual Inductor composition

### PF12: Expose the shared frontend service in the compiler CLI

**Files:** Modify `tools/llk-compile/llk-compile.cpp` and its CMake target; create `test/Frontend/cli.py`, `test/Frontend/Inputs/{add-v1,bad-version-v1}.json`; update `test/FileCheck.cmake`.

**Interfaces:** Add `--input-format=mlir|frontend-json`, default `mlir`; frontend-json parses a complete CompileRequest and uses `compileFrontend`. Preserve existing emit/mapping flags for MLIR; conflicting JSON/CLI option sources fail `conflicting_compile_options` rather than silently overriding one another.

- [ ] Add CLI tests for registered IR inspection, compile reports and invalid JSON/version/opcode. Legacy `tt.*` input in default production mode must have an actionable unsupported-input message. No unregistered-dialect escape flag is introduced as production Triton support.

Pinned regression assertions (local test fixture names):

```python
assert cli_report["abi_hash"] == capi_report["abi_hash"]
assert cli_report["plan_id"] == capi_report["plan_id"]
```

- [ ] Run `ctest --test-dir build -R FrontendCLI --output-on-failure`; confirm the new input mode initially fails.
- [ ] Route JSON through the same native service as Python, including target/identity and source-selection checks. Remove misleading implicit Triton detection from the production path; preserve legacy fixtures in llk-opt's explicitly experimental flow.
- [ ] Rerun CLI tests; the same canonical request via CLI and C API must report equal workload/target/plan/ABI identities, ignoring process-local handles and timings.
- [ ] Commit `feat: route frontend JSON through the shared compiler service`.

### PF13: Implement strict normalized-ATen Dynamo compilation

**Files:** Modify `python/llk/torch/_dynamo_backend.py`; create `python/llk/torch/{_compat,_fx_importer}.py`, `test/python/{test_dynamo_backend,test_fx_importer}.py`.

**Interfaces:** `import_fx(gm, example_inputs, options) -> CompileRequest`; `LLKBackend.__call__` returns an equivalent callable backed by compiled handles. Normalize through `aot_autograd(fw_compiler=..., inference_compiler=...)`; its compiler returns boxed callables as required by the pinned Torch API.

- [ ] Test real `torch.compile(backend="llk", fullgraph=True)` for add/mul/mm/SwiGLU, tuple/output order, FakeTensor example metadata and changed concrete shapes. Unsupported sin/in-place/view-dependent graphs must identify the node and fail strict compilation; assert no `gm.forward` execution path or graph mutation on failure.

Pinned regression assertions (local test fixture names):

```python
assert native_invoke_spy.call_count == 1
assert unsupported_fx_error.node_name == "sin"
```

- [ ] Run `python -m pytest test/python/test_dynamo_backend.py test/python/test_fx_importer.py -q`; missing native lowering must fail numerical-plus-native-counter assertions.
- [ ] Add version-qualified normalization in `_compat.py`, match exact ATen targets/shapes/effects, construct the shared request and allocate fresh outputs. Preserve AOT boxed calling/output conventions. Inference-only operations reject requires_grad until PF16 supplies a registered path.
- [ ] Rerun fullgraph numerical tests with native reports and explicit unsupported cases. Assert compilation uses fake metadata only and invocation uses real buffers.
- [ ] Commit `feat: compile supported Dynamo graphs through LLK MicroIR`.

### PF14: Register functional LLK operators with truthful fake metadata

**Files:** Create `python/llk/torch/_ops.py`, `test/python/{test_torch_ops,test_fake_tensor_contract}.py`; modify Torch initialization and buffer adapters.

**Interfaces:** Register `llk::add(Tensor a, Tensor b) -> Tensor`, `llk::mul`, `llk::matmul`, `llk::swiglu(Tensor x, Tensor wg, Tensor wu) -> Tensor` with `mutates_args=()`. CPU implementations call the shared request/cache; fake implementations construct shape/dtype/device-correct fresh outputs without reading storage.

- [ ] Run opcheck-schema/fake tests with representative static shapes, transposed/offset tensors and requires_grad policy. Spy on data_ptr/native functions during FakeTensor execution and assert zero calls. Assert real output allocation has no forbidden alias and matches the declared metadata.

Pinned regression assertions (local test fixture names):

```python
assert fake_data_ptr_spy.call_count == 0
assert fake_native_compile_spy.call_count == 0
```

- [ ] Run `python -m pytest test/python/test_torch_ops.py test/python/test_fake_tensor_contract.py -q`; verify missing schemas/fake implementations fail.
- [ ] Register custom ops and fake implementations using Torch 2.9 APIs, with idempotent ownership and version checks. Use the same native buffer validation as the decorator; unsupported views/subclasses fail instead of copying silently.
- [ ] Rerun `torch.library.opcheck` and direct numerical tests for all four operators; preserve explicit inference-only guards until training formulas land.
- [ ] Commit `feat: register checked LLK external operators for PyTorch`.

### PF15: Extend Inductor through scoped LLK external-kernel partitioning

**Files:** Create `python/llk/torch/_inductor_backend.py`, `test/python/{test_inductor_composition,test_inductor_partitioning}.py`; modify registration and `_compat.py`.

**Interfaces:** `rewrite_supported_subgraphs(gm, example_inputs, options) -> (GraphModule, PartitionReport)` clones the graph; `inductor_backend(gm, example_inputs, **kwargs)` rewrites eligible pure regions, then calls the qualified `torch._inductor.compile_fx.compile_fx`. Produce frozen `PartitionReport(llk_regions: int, matched_nodes: tuple[str, ...], declined: tuple[tuple[str, str], ...])`. Register name `llk_inductor`. Unsupported regions stay with Inductor, with an explicit partition report.

- [ ] Test `torch.compile(fullgraph=True, backend="llk_inductor")` on `sin(a+b)` and a SwiGLU-plus-unmatched-op graph. Assert supported nodes call LLK exactly once, unmatched nodes are handled by Inductor, outputs match, shared users/effects are preserved, and no global matcher changes leak to ordinary `backend="inductor"` compilation. Add a graph with no LLK region: report zero partitions honestly.

Pinned regression assertions (local test fixture names):

```python
assert partition_report.llk_regions == 1
assert native_invoke_spy.call_count == 1
assert inductor_compile_spy.call_count == 1
```

- [ ] Run `python -m pytest test/python/test_inductor_composition.py test/python/test_inductor_partitioning.py -q`; require real Inductor and fail on missing dependencies rather than skipping this lane.
- [ ] Implement effect- and numerical-contract-aware FX rewrites to PF14 operators; decline bf16 SwiGLU and any unmatched activation/cast/reduction policy and delegate surrounding graph compilation. Scope all configuration to the backend call. Do not register a replacement CPU device scheduler, claim cross-call fusion or represent this integration as a full Inductor codegen backend.
- [ ] Rerun fullgraph composition/partition tests and inspect compiled graph/code plus native reports. A correct eager result without an Inductor compile and LLK native invocation fails acceptance.
- [ ] Commit `feat: compose LLK external kernels with Inductor`.

### PF16: Add qualified training for add/multiply/matmul

**Files:** Create `python/llk/torch/_autograd.py`, `test/python/test_torch_training.py`; modify `_ops.py`, `_dynamo_backend.py` and `_compat.py`.

**Interfaces:** Register autograd formulas for add/multiply/matmul and use AOTAutograd forward/backward compilers. Backward transpose materialization is explicit in ATen and included in the compiled partition/report. SwiGLU requires_grad remains a concrete unsupported diagnostic until a separate gradient task.

- [ ] Add f32 directional finite-difference comparisons against an independently evaluated f64 reference with fixed `eps=1e-3, atol=1e-2, rtol=1e-2`, direct analytic f32 gradient comparisons, repeated backward, saved-tensor lifetime and opcheck autograd-registration tests. Assert separate forward/backward native reports when claiming LLK-compiled backward. Unsupported SwiGLU training must fail, not silently use eager autograd.

Pinned regression assertions (local test fixture names):

```python
torch.testing.assert_close(native_grad, reference_grad, rtol=1e-2, atol=1e-2)
assert swiglu_training_error.code == "unsupported_training_contract"
```

- [ ] Run `python -m pytest test/python/test_torch_training.py -q`; establish missing gradient/AOT behavior failures.
- [ ] Register mathematically correct formulas, ensure gradients preserve shape/device and materialize transpose layouts explicitly before the compact ABI boundary. Delegate non-LLK backward regions to Inductor only in llk_inductor, recording that partition; llk strict mode rejects unsupported backward graphs.
- [ ] Rerun numeric/registration/fullgraph forward-backward tests; identify which backward kernels are native LLK and which are ATen/Inductor. Never infer training support from forward correctness.
- [ ] Commit `feat: qualify LLK autograd and backward compilation`.

## Milestone D — genuine upstream Triton support

### PF17: Qualify the upstream toolchain and implement neutral TTIR export

**Files:** Create `python/llk/triton/{__init__,_export_worker}.py`, `tools/triton-export/{CMakeLists.txt,TritonExport.cpp}`, `python/llk/triton/toolchain-lock.json`, `test/python/compat/{test_triton_toolchain,test_ttir_export}.py`; modify optional packaging/build configuration only.

**Interfaces:** Export worker protocol v1 accepts source/module locator, upstream pointer/scalar signature and specialization constraints, constexprs and backend options; returns genuine upstream TTIR text plus an `UnboundFrontendTemplate` and toolchain identities, or a structured error. `UnboundFrontendTemplate` fixes `schema_version=1` and `required_runtime_grid=True`; the template carries typed parameters, affine access/mask/dataflow and counted-loop/cast regions, source/TTIR hashes and required argument/alignment constraints; it has no invented concrete tensor shapes or runtime grid. `bind_triton_template(template, args, grid, options) -> CompileRequest` is produced in PF18 at actual launch. Exporter objects never cross processes. The toolchain lock records exact Triton/LLVM revisions and build configuration actually qualified.

- [ ] Add a real upstream frontend capture/export test for canonical add/mul, plus unsupported op/layout/memory-effect/version fixtures. Test spaces/non-ASCII paths and structured errors without parsing stdout log noise. Test import orders Torch→Triton→LLK and LLK→Triton→Torch in subprocesses.

Pinned regression assertions (local test fixture names):

```python
assert toolchain_lock["triton_version"] == "3.5.0"
assert template.schema_version == 1
assert template.required_runtime_grid is True
```

- [ ] Run `python -m pytest test/python/compat/test_triton_toolchain.py test/python/compat/test_ttir_export.py -q` against the pinned real package/exporter; confirm absence or ABI mismatch fails qualification.
- [ ] Build the exporter against Triton's own pinned LLVM/dialect, walk actual typed TTIR and translate accepted affine/mask/dataflow semantics into the unbound template schema; preserve all runtime scalar dependencies for launcher-time binding. Spawn with argv, bounded output and dedicated result framing; retain diagnostics separately. Never parse arbitrary TTIR with LLK's unknown-dialect parser or share C++/MLIR objects across toolchains. Generate and verify the lock from the actual binaries.
- [ ] Rerun capture/export/roundtrip/import-order tests. A failed native-library coexistence check disables upstream support with `upstream_toolchain_unqualified`; record an open blocker rather than claiming D complete. No simulated backend or copied source is qualification evidence.
- [ ] Commit `feat: qualify real Triton TTIR export into frontend programs`.

### PF18: Implement upstream backend, driver and CPU launcher

**Files:** Create `python/llk/triton/{_backend,_driver}.py`, `test/python/compat/test_triton_launch.py`; modify plugin entry points/toolchain manifest and inspection handling.

**Interfaces:** Implement Triton 3.5.0 `BaseBackend` and `DriverBase` subclasses with their audited abstract methods, an `llk_cpu` target and scoped activation context `llk.triton.activate(options)`. Compiler final bytes encode the versioned unbound template, not a machine-code claim; `CPUUtils.load_binary` retains that template and reports honest artifact metadata. `CPULauncher` implements `bind_triton_template`, binds actual CPU tensor shapes/strides/dtypes and scalar values, normalizes the runtime grid, proves coverage, obtains a shape-bound `CompiledKernel` lease from PF09 and marshals through PF08 validation. Key this second cache stage by template content plus the entire concrete native request; upstream pointer/alignment specialization alone is insufficient. Expose launch evidence with `last_launch_report()` rather than pretending native compilation happened before launch. Serialize the activation context and restore the prior process-global active driver on exit/error.

- [ ] Add an actual `@triton.jit` add/mul launched with `[grid]` inside the activation context, writing CPU output and exposing genuine `.asm["ttir"]` plus `last_launch_report()`. Launch the same upstream pointer-specialized kernel on 32×64 and 64×32 buffers and require distinct correct native specializations; vary a runtime scalar extent and grid without changing pointer types. Assert the prior driver is restored; GPU tensor arguments fail; malformed final artifacts/version/hash fail before invoke. Use real upstream classes and specialization, not mocks.

Pinned regression assertions (local test fixture names):

```python
assert report_32x64.native_request_identity != report_64x32.native_request_identity
assert driver_after_context is driver_before_context
```

- [ ] Run `python -m pytest test/python/compat/test_triton_launch.py -q`; establish absent plugin/driver behavior as red.
- [ ] Implement backend options/hash/stages/dialect loading/module map and driver utility/launcher contracts against the exact qualified version. Final artifact validation binds upstream source/signature/options to the unbound template; launcher binding validates every deferred tensor/scalar/alignment/launch constraint before creating the native request. Retain handles and input owners until synchronous CPU completion. Export upstream JIT specialization only through PF17 and route all launch-time LLK compilation through the shared cache/service. Different runtime shapes/grids must not collide in the second-stage cache.
- [ ] Rerun real CPU launch and import/activation-lifetime cases; inspect native-call evidence and output canaries. Reject an unqualified version rather than guessing private API compatibility.
- [ ] Commit `feat: launch genuine Triton kernels through LLK CPU backend`.

### PF19: Establish upstream compatibility corpus and supported boundaries

**Files:** Create `test/python/compat/kernels/{elementwise,matmul,swiglu}.py`, `test/python/compat/test_triton_corpus.py`, `docs/design/triton-compatibility-v1.json`; modify TTIR exporter and backend only for proved semantic extensions.

**Interfaces:** Publish per-kernel frontend/import/mapping/reference/selected/invocation status with reason codes; reuse #157's capability schema where delivered. Declare at least 10 fixed, source-attributed kernel fixtures rather than an unbounded compatibility percentage.

- [ ] Add ten cases: full add, tiled add, full multiply, tiled multiply, masked tail add, masked tail multiply, flattened-index add over declared rank-two storage, f32 matmul, accumulated matmul according to the supported initializer contract, and SwiGLU. Add explicit rejection cases for atomic, shared async/await, arbitrary reduce, runtime while, unsupported strides and GPU-only instructions.

Pinned regression assertions (local test fixture names):

```python
assert len(executed_fixture_ids) == 10
assert all(item.native_invocations > 0 for item in executed_fixture_reports)
```

- [ ] Run `python -m pytest test/python/compat/test_triton_corpus.py -q`; collect the exact kernel count and initial missing-capability failures.
- [ ] Extend only exporter-recognized patterns that reach existing verified source adapters. If accumulated matmul requires an adapter change, implement/test it in this task or keep that required fixture failing and the milestone open; never substitute zero initialization. Record full toolchain/source/numerical/plan identities per executable fixture.
- [ ] Rerun the corpus on reference and supported AVX2 selected-target lanes. Require all declared executable fixtures to pass; unsupported fixtures must yield the specified diagnostic. Do not count rejected/parsed-only cases as executed compatibility successes.
- [ ] Commit `test: qualify real Triton kernel compatibility and rejection boundaries`.

## Milestone E — reproducible release and closure

### PF20: Wire required CI, packaging and user-facing capability documentation

**Files:** Modify `.github/workflows/ci.yml`, root/test CMake, `python/pyproject.toml`, `docs/manual/source/{features.rst,index.rst}`, `README.md`, the July/frontend design cross-references and this plan; create `test/python/conftest.py`, `test/python/check_required_python_execution.py`, `docs/manual/source/tutorials/python-frontend.rst`, `test/Frontend/acceptance_manifest.schema.json`.

**Interfaces:** Add pytest markers `frontend_unit`, `frontend_native`, `torch_integration`, `triton_integration`, `selected_avx2`; add CMake options `LLK_BUILD_PYTHON_CAPI` and `LLK_ENABLE_PYTHON_TESTS` (both OFF for ordinary compiler-only users; ON in required frontend CI). Declare dependencies in the check target and package install native/configuration resources through explicit install rules.

- [ ] Add clean-wheel-install tests, version/API/library mismatch tests and an acceptance-manifest checker that fails on zero executed required cases, a skipped required lane, absent native-call counters or selected fallback. Include a synthetic failing checker fixture to prove enforcement.

Pinned regression assertions (local test fixture names):

```python
assert acceptance_errors_for_zero_execution == ["required_execution_missing"]
assert acceptance_errors_for_selected_fallback == ["selected_reference_fallback"]
```

- [ ] Run the checker tests first; verify missing/skip/zero-count evidence fails. Configure a fresh build with `-DLLK_BUILD_PYTHON_CAPI=ON -DLLK_ENABLE_PYTHON_TESTS=ON` and the repository's LLVM path.
- [ ] Add required Python 3.11/3.12 CPU/Torch jobs, optional-dependency-absent import lane, pinned upstream qualification/launch jobs, Linux AVX2 selected execution and available local LLVM 24+ checks. Run pytest from the configured interpreter, not an ambient `python`. Package installed machine/rule/layout resources and preserve their hashes; reject missing files explicitly. Replace the disabled staged skeleton with a real supported execution test or an active expected-rejection test whose title states its capability.
- [ ] At milestone exit run `cmake --build build --target check-llk`, all required pytest markers, strict Sphinx/reference checks and the clean-wheel examples. Record commit, dependency/toolchain locks, test counts, numerical errors, guard checks, native invocation identities and skip reasons in an acceptance report. Reconcile #156/#157 and #191 work without widening claims.
- [ ] Commit `test: require frontend execution and publish capability acceptance`; update the parent issue with evidence only after these checks pass. Keep unfinished upstream/training/target rows open.

## Review finding → task closure map

| Finding in review | Tasks | Closure evidence |
|---|---|---|
| 1. Registration/API/reset | PF01 | Real installed Torch registration/reload, zero reset calls |
| 2. Invalid IR | PF03–PF05, PF17 | Typed program verification; registered MLIR roundtrip; genuine upstream TTIR only in D |
| 3. Lost semantics | PF03–PF05, PF10–PF11 | Exact masks/SSA/accumulator tests and explicit unsupported syntax |
| 4. No executable JIT/FFI | PF06–PF09 | Checked native invocation, output mutation, lifetime/caching evidence |
| 5. Wrong block addressing | PF02, PF10 | Correct stride plus numerical/canary loads/stores |
| 6. Disconnected compilation | PF05–PF06, PF12 | One shared service, same CLI/C API identities, honest input-format gate |
| 7. Escaping grid values | PF02, PF05/PF08 | Valid void grids; result forms rejected with verification enabled |
| 8. Overbroad whitelist/runtime claims | PF02, PF19–PF20 | Exact schemas, residual checks and required unsupported diagnostics |
| 9. Integration boundary missing | PF13–PF19 | Independent Dynamo, actual Inductor composition and actual upstream launch |
| 10. Tensor/specialization contracts | PF03, PF07–PF11 | Metadata/alias/bounds guards and full-identity cache tests |
| 11. Ineffective test/CI coverage | PF01–PF20 | Parser/verifier/numerical evidence and enforced nonzero required execution |

## Completion and review gates

Every task owns a red/green cycle and one focused commit. Run only affected tests while iterating; run the whole compiler and required integration lanes at each milestone exit or when a change affects shared invariants. Future commands/test names in this plan are implementation requirements, not evidence they already exist.

A/B can be reviewed and released as explicitly bounded decorator capability while C/D remain open. The umbrella issue closes only when all required milestone rows and the eleven finding rows have concrete evidence; a proposal, unsupported diagnostic, passing eager result or frontend substring test does not satisfy an executable row. Optional future features excluded by the design remain explicit follow-up work and never silently become claimed support.

Implementation handoff: review the proposed design and plan before starting product changes. Native inline execution is a reasonable default for this tightly shared interface sequence; subagent-driven execution is available if requested. The present task ends after publishing these planning artifacts and the tracking issue.
