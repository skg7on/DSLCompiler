# PF08 Python buffer adapters and native JIT launch verification

**Date:** 2026-11-01 (Asia/Shanghai).

**Scope:** [issue #219](https://github.com/skg7on/DSLCompiler/issues/219), PF08 in
[the frontend plan](../superpowers/plans/2026-10-18-python-frontend-refinement.md).
Base `main` `b61c1a2` contains merged PF07 (PR #251). Implementation commit
`7bc2b5b` is on `feat/python-native-jit-launch` in the isolated worktree
`.claude/worktrees/feat+dynamic-acceptance`, which owns its own `build/`.

## Delivered behavior

`python/llk/runtime/_buffers.py` mirrors the frozen native `LLKBuffer2D` layout
(72 bytes on 64-bit hosts, field offsets 0/8/16/24/40/56/64) and adapts caller
providers through the Python buffer protocol or an explicit
`__array_interface__`. It reads the address with `PyObject_GetBuffer`/`PyBUF_STRIDES`
and validates rank two, positive extents, native element format and itemsize,
compact row-major strides, writability for outputs, declared allocation at least
as large as the declared extent, and disjoint outputs. Array-interface providers
whose `base` names a different data pointer are rejected as offset views. Raw
byte providers (`B`/`b`, rank one) are reinterpreted only against an explicit
`TensorSpec`, must be unit-stride contiguous and must cover its declared byte
extent. No copy, dtype conversion or provider cache exists. A lease holds the
provider and a live buffer export for one invocation, so a resizable provider
cannot move its storage while the GIL is released in the native call.

`python/llk/runtime/_ffi.py` loads `LLKFrontendCAPI` from
`LLK_FRONTEND_LIBRARY` or the installed package resource, requires API version 1,
translates owned native error JSON into `FrontendSyntaxError`,
`UnsupportedFeatureError`, `TensorContractError`, `CompilationError` and
`ExecutionError` with `code`/`message`/`stage`/source span, and owns handles via
`CompiledKernel`. `invoke` holds an `RLock` across the whole ctypes call;
`close` is idempotent, waits for an in-flight call, and later invocations raise
`invalid_frontend_handle`. `report` parses the native `CompilationReport`,
including `backend` and `source_adapter`. `compile_request` and
`CompiledKernel` are exported from `llk`.

`python/llk/runtime/_jit.py` keeps `@llk.jit` and `@llk.jit()` with keyword-only
`CompileOptions`. `kernel[grid](*args, **kwargs)` and
`kernel(..., grid=grid)` share one build path: bind, prove grid × BLOCK covers
every public tensor exactly, capture, compile, then invoke the caller's
providers. A missing grid or invalid coverage fails with
`invalid_launch_coverage` before native entry. `kernel.compile(...)` returns an
owned handle for precompilation. `ttir()` remains explicitly unavailable.

## Red/green

The three suites were written first. The initial run failed because the feature
did not exist:

```text
ERROR test/python/test_native_jit.py - TypeError: jit() got an unexpected keyword argument 'config_root'
ERROR test/python/test_buffer_contract.py - TypeError: jit() got an unexpected keyword argument 'config_root'
ERROR test/python/test_launch_contract.py - TypeError: jit() got an unexpected keyword argument 'config_root'
```

Before PF08, the decorator call itself raised
`UnsupportedFeatureError(code="native_backend_unavailable")`; that is now the
missing-library/unsupported-ISA diagnostic rather than the decorator's behavior.
After the implementation the same focused command passes 33 cases.

## Accepted tiling and machine-capacity gap

The issue's literal 32×64/8×16 launch is **not compilable** on the stock
`machines/x86-avx2-v2.yaml` profile. Compiling that request through the C API
reports:

```text
frontend_compilation_failed: no executable frontend candidate; capacity_failure:
no feasible covering: memory_capacity_exceeded: storage plan: memory 'sram.0'
over capacity (peak 147456 bytes live over up to 16 simultaneous occurrence(s)
per slot, 32768 byte capacity)
```

The measurements are:

| shape / block | native compile |
| --- | --- |
| 32×64 / 8×16 | fails, 147,456 B peak vs 32,768 B |
| 32×64 / 16×32 | fails, 49,152 B peak |
| 16×16 / 8×16 | compiles and executes (2 tiled programs) |
| 32×64 / 32×64 | compiles and executes (full domain) |

PF06's C API fixture had already been reduced to a full-domain 8×16 shape for the
same reason. PF08 keeps production machine data unchanged and exercises the same
captured kernel, native source adapter and launch path at 16×16/8×16 tiled and
32×64 full domain. Closing the literal 32×64/8×16 case needs a separate shared
mapping/storage-capacity change; lowering the capacity counter is not in PF08's
scope. The plan and manual record this deviation.

## Validation

Commands run from the worktree root. The library was built with
`-DLLK_BUILD_PYTHON_CAPI=ON`; Python is the project `.venv` 3.14.8.

```sh
cmake -S . -B build -G Ninja \
  -DLLVM_PROJECT_BUILD_DIR=/Users/skg7on/Workspace/Projects/llvm-project/build \
  -DLLK_BUILD_PYTHON_CAPI=ON
cmake --build build --target LLKFrontendCAPI LLKFrontendTests LLKCAPIFrontendTests -j6
LLK_FRONTEND_LIBRARY=$PWD/build/lib/libLLKFrontendCAPI.dylib \
  /Users/skg7on/Workspace/Projects/DSLCompiler/.venv/bin/python -m pytest \
    test/python/test_native_jit.py test/python/test_buffer_contract.py \
    test/python/test_launch_contract.py -q
LLK_FRONTEND_LIBRARY=$PWD/build/lib/libLLKFrontendCAPI.dylib \
  /Users/skg7on/Workspace/Projects/DSLCompiler/.venv/bin/python -m pytest -q
ctest --test-dir build -R 'FrontendCAPI|Frontend' --output-on-failure
python scripts/check-license-headers.py
```

| Lane | Actual result |
| --- | --- |
| PF08 focused suites | 33 passed, 0 failed, 0 skipped |
| Full Python suite | 137 passed, 0 failed, 0 skipped |
| Native `Frontend` + `FrontendCAPI` | 58 of 58 passed |
| Full native CTest in this worktree | 1,678 run; 4 failed (host-ISA lane), 4 skipped, 1 disabled |
| Linux CI, [PR #273](https://github.com/skg7on/DSLCompiler/pull/273), LLVM/MLIR 22.1.8 | `build-and-test`, `coverage`, `analyze`, `html` and `license-headers` passed |
| License-header audit | 370 files checked, clean |

The 104-test PF06 baseline becomes 137 with the 33 new cases; no existing Python
test regressed. The native lane includes `FrontendCAPI.Consumer`,
`FrontendCAPI.Exports` and all `FrontendCompiler`/`FrontendImporter` cases.

The four full-suite failures are the `selected-avx2` cases
(`MappedAVX2Acceptance`, `GenericElementwiseAVX2Test`,
`FrameworkPrimitivesAVX2Test`, `DynamicMappedAVX2Acceptance`), each reporting
"required selected AVX2 execution needs an x86_64 AVX2 host" on this arm64
machine. They are the same host-ISA failures seen on the unchanged base; this
change is Python-only and does not touch their binaries. `DocReferences` failed
on the first full run because a `kernel[grid](...)` example was parsed as a
Markdown link; the manual now uses a `launch` variable and the rerun passes.

### Numerical, canary and native-call evidence

- Tiled 16×16/8×16 and full-domain 32×64 adds compare every output element
  against independent `a+b` tables built from position-varying inputs. Both
  input byte snapshots are unchanged.
- Every buffer is a guarded ctypes allocation with two `-777.0` floats on each
  side of the payload; all canaries remain `-777.0` after success and after the
  rejected cases.
- `native_invoke` is wrapped with a counting spy that delegates to the real
  Python-to-native call. Successful tests assert exactly one native invocation;
  missing grid, partial/over coverage and wrong arity/shape assert zero.
- 32×64 full-domain and 16×16 tiled cases report `backend="reference"` and
  `source_adapter="linalg-elementwise-v1"`.
- The report returned by the executed handle is the compilation report:
  `report.frontend_identity` equals the canonical `serialize_request` bytes for
  the same request, and the same handle is the one invoked.
- Close is idempotent; a second close is a no-op. A concurrent close started
  while `native_invoke` is deliberately blocked does not complete until the call
  is released; the in-flight invocation succeeds, and a later invocation raises
  `invalid_frontend_handle`.

Actual native identities from the same run:

| field | tiled 16×16/8×16 | full 32×64 |
| --- | --- | --- |
| backend | `reference` | `reference` |
| source_adapter | `linalg-elementwise-v1` | `linalg-elementwise-v1` |
| target_hash | `0x17bbdd0bb651e319` | `0x17bbdd0bb651e319` |
| machine_hash | `0xe033f896c242e2c4` | `0xe033f896c242e2c4` |
| rules_hash | `0x6d851943096f8682` | `0x6d851943096f8682` |
| plan_id | `0xd97be0129d9e1131` | `0x6dafe90fcc99e7fe` |
| abi_hash | `0x4007b5fdb8b3e09b` | `0xb510ec546d31949b` |
| selected / reference group counts | 0 / 0 | 0 / 0 |
| selected_groups_verified | 4 | 4 |

`reference_group_count` is zero because the target's semantic emitters lower all
four groups before the portable reference backend, matching PF06's documented
counter semantics. It is not reported as a positive count.

## Independent review and fix pass

A fresh reviewer inspected the uncommitted diff and reproduced two safety
defects; both are fixed before commit.

1. **Rank-1 byte providers skipped stride validation.**
   `memoryview(bytearray(1025))[::-1]` exported its pointer at the last byte and
   `nbytes=1025`, so a 16×16 f32 descriptor was accepted with an allocation that
   extended 1,023 bytes past the real buffer; used as an output it would write
   out of bounds. Raw byte providers now require unit strides
   (`strides == (1,)`) and no suboffsets, and the tests cover reversed and strided
   byte views.
2. **No buffer export was held across the native call.** The `Py_buffer` export
   was released while building the descriptor, before `ctypes` released the GIL,
   so a concurrent `bytearray.extend` could relocate storage under the live
   descriptor. `BufferLease` now retains the `memoryview` export and
   `CompiledKernel.invoke` releases it only after the native call returns. A test
   proves `bytearray.extend` raises `BufferError` while the lease is held.

The same pass made providers optional for metadata-only `kernel.compile(...)`,
replaced the close/invoke timing check with explicit events, and keyed element
formats by code plus byte width so NumPy `|i1` and `<f2` map to i8/f16 while
`<f8` is rejected. After the fixes the focused suites pass 33 cases and the full
Python suite passes 137; the native `Frontend`/`CAPI` and documentation lanes are
unchanged.

## Local toolchain and capability limits

Local toolchain: LLVM/MLIR **24.0.0git**, Homebrew clang 20.1.8, macOS arm64,
Python 3.14.8. No ASan/LSan result is claimed here; PF07 documented the host's
inability to run ASan and deferred that lane to PF20's Linux tooling.

Selected-target mode was not executed on this arm64 host. The suite asserts the
explicit `native_backend_unavailable` diagnostic with no reference fallback; on a
compatible x86 host the same test requires `backend="selected-target"` and exact
outputs instead.

PF08 adds no specialization cache, inspection report object, tails, Torch/Inductor
composition, training or upstream Triton launch. Those remain PF09, PF10 and the
later frontend milestones, and PF20 still registers the native Python lane in
CMake/CI. The umbrella #203 remains open, and closing #219 does not imply any
other entry path is supported.
