# PF17 verification record: qualified Triton 3.6.0 TTIR export

Date: 2026-11-05
Issue: [#228](https://github.com/skg7on/DSLCompiler/issues/228)
Parent: [#203](https://github.com/skg7on/DSLCompiler/issues/203)
Plan: `docs/superpowers/plans/2026-10-18-python-frontend-refinement.md` §PF17
Implementation commits: `0baf023` (exporter and tests), `123cc6d` (function-scoped
access attribution), `e99d89b` (clean-wheel packaging assertions)

## Delivered

`llk.triton.capture_kernel` and `llk.triton.export_module` export genuine
upstream-specialized TTIR and a neutral `UnboundFrontendTemplate` from a
process-isolated worker. The calling process never imports Triton and never
links a Triton LLVM object into the LLK build.

The worker is a separate interpreter launched with an explicit argv:

- It loads the qualified `triton` package from `--triton-python` (the
  `LLK_TRITON_PYTHONPATH` directory or the current interpreter's import path).
- It builds an upstream `ASTSource`, applies specialization attributes and
  constexprs, and runs the backend's own `ttir` compiler stage. No simulated
  backend, copied TTIR or LLK unknown-dialect parser is involved.
- It walks the typed module through Triton's own MLIR bindings
  (`triton._C.libtriton.ir`) and emits only JSON. The walker never receives
  arbitrary user TTIR: it always translates the module Triton's parser produced
  from the requested kernel.
- It writes exactly one marker-framed JSON document to stdout
  (`LLK-TRITON-EXPORT-V1`); upstream prints and diagnostics are redirected to
  stderr. The parent reads both streams with hard size caps and a timeout.

`UnboundFrontendTemplate` fixes `schema_version=1` and
`required_runtime_grid=True`. It carries typed pointer/scalar/constexpr
parameters, SSA values with compile-time tile shapes, the dataflow graph,
masked loads/stores, counted-loop and cast regions, required alignment
constraints, source/TTIR hashes, and the observed toolchain identity. It has no
runtime tensor shape and no launch grid. `bind_triton_template` remains PF18's
launch-time contract and is deliberately absent.

## Toolchain lock

`python/llk/triton/toolchain-lock.json` was regenerated through
`llk.triton.generate_toolchain_lock` from the actual binaries and then
qualified against its own output. The observed identity was:

| Field | Value |
|---|---|
| `triton_version` | `3.6.0` |
| `source_revision` | `93f05e19a8024938c320d82854e307d602b5a05d` |
| `llvm_revision` | `27d654c4c4e6eb7c19e46af20500200e793da7c7` |
| `protocol_version` | `1` |
| `target` | `cuda:80` |
| `python_tag` | `cpython-3.14` |
| distribution | source checkout at `~/Workspace/Projects/triton` |

Qualification compares Triton version, LLVM revision, source revision and
target exactly. A missing package, unreadable binary, changed lock or different
target raises `UnsupportedFeatureError(code="upstream_toolchain_unqualified")`
before any translation. `LLK_TRITON_PYTHONPATH` selects a source checkout; the
`triton-test` extra pins `triton==3.6.0` for lanes where a wheel exists.

The local macOS host has no upstream Triton wheel and no GPU. TTIR generation
is a front-end stage and does not require a driver, device or `ptxas`; no GPU
execution is claimed, and PF18 owns the CPU launch contract.

## Accepted subset and rejections

Accepted: program id, integer/boolean constants, ranges, splats, pointer
addition, masked and unmasked loads/stores, integer and float arithmetic,
comparisons, selects, casts, and counted `scf.for` loops with yields. Memory
accesses are attributed back to kernel arguments through `splat`/`addptr`
chains; each pointer argument is classified as read or write.

Rejected with structured codes:

| Case | Code |
|---|---|
| atomics | `unsupported_triton_memory_effect` |
| a pointer argument both loaded and stored | `unsupported_triton_memory_effect` |
| `tt.trans`, `tt.reshape`, `tt.expand_dims`, `tt.broadcast` | `unsupported_triton_layout` |
| reduce/scan/arbitrary control flow/other ops | `unsupported_triton_operation` |
| float or dense literal constants | `unsupported_triton_constant` |
| signature/constexpr/specialization malformed or out of order | `invalid_triton_signature` |
| kernel load or TTIR generation failure | `triton_compilation_failed` |

## Observed template identity

For the canonical masked f32 add (`BLOCK=1024`, 16-byte divisibility on all
three pointers, signature `x_ptr, y_ptr, out_ptr, n, BLOCK`):

```text
template identity  31e1bce5c15a83fc76b0973232bf3b682b1a91015932624b0b0311740175e54f
source hash        78990655326cfc4c26bfb7c9933b6dd611bcef448ee484e0f2a0b25d207ebae9
TTIR hash          d3223fcf9b7e7dd2dd167ec715c94430048469a5aef6717dfcc4a2f6c11d0201
values             21
top-level ops      19
```

Parameters: `x_ptr`/`y_ptr` pointers `f32` access `read` alignment 16,
`out_ptr` pointer `f32` access `write` alignment 16, `n` scalar `i32`,
`BLOCK` constexpr `i32` value 1024. Capturing the same source through two
different temporary directories produces byte-identical TTIR and the same
template identity because the worker prints the module without debug locations.

## Test evidence

Green — focused compat lane (Python 3.14.8, Triton 3.6.0 source checkout,
LLVM revision `27d654c4`, macOS arm64):

```sh
LLK_TRITON_PYTHONPATH=/Users/skg7on/Workspace/Projects/triton/python \
    .venv/bin/python -m pytest test/python/compat/ -q
# 22 passed
```

Covered:

- checked-in lock values and regeneration from the actual binaries;
- real add and multiply capture with typed parameters, masks, alignment and
  scalar dependencies;
- count-ended loop and `bf16`→`f32` cast regions;
- template JSON roundtrip and deterministic identity across paths;
- spaces and non-ASCII path components;
- top-level stdout log noise not corrupting result framing;
- atomic, layout, aliased read/write and compilation-failure rejections;
- missing toolchain, mismatched version/LLVM/source revision/protocol and
  mismatched export target qualification failures;
- Torch→Triton→LLK and LLK→Triton→Torch import orders in subprocesses.

The clean-wheel install lane in `test/python/test_package_install.py` now
imports `llk.triton`, reads the packaged `toolchain-lock.json` and asserts the
`triton-test` extra is declared.

Absence lane — without `LLK_TRITON_PYTHONPATH`:

```sh
env -u LLK_TRITON_PYTHONPATH .venv/bin/python -m pytest test/python/compat/ -q
# 2 passed, 20 skipped   (the lock and missing-toolchain checks still run)
```

Full Python suite:

```sh
CXX=/usr/bin/clang++ CC=/usr/bin/clang \
LLK_FRONTEND_LIBRARY=$PWD/build/lib/libLLKFrontendCAPI.dylib \
LLK_TRITON_PYTHONPATH=/Users/skg7on/Workspace/Projects/triton/python \
    .venv/bin/python -m pytest -q
# 309 passed
```

Observed red/green cycles during implementation:

- Sorting the request JSON keys reordered the kernel signature, so
  `tt.divisibility` was attached to the wrong argument positions and the
  upstream generator raised `IndexError('Function argument index out of
  range')`. The request now preserves signature order, and the worker rejects
  any signature whose order differs from `kernel.arg_names`.
- The translator initially received the AST attribute table instead of the raw
  specialization mapping, so alignment constraints were dropped. The template
  now records `alignment_bytes=16` and the emitted TTIR is checked for
  `tt.divisibility = 16`.
- `str()` on the TTIR module embedded absolute source locations, which made
  hashes path-dependent. The worker uses `str_nodebug()` (still genuine TTIR)
  so identities are deterministic.

The full-suite command requires the Apple clang override for Inductor on this
host, as recorded by the PF16 evidence; without it, Inductor's C++ wrapper
compilation fails on the ambient Homebrew LLVM include order.

## Deviation from the planned file list

The plan named `tools/triton-export/{CMakeLists.txt,TritonExport.cpp}`. The
approved implementation instead walks typed TTIR through Triton's own pinned
MLIR Python bindings inside the isolated worker. This keeps the same isolation
invariant (no Triton LLVM object is linked into or shared with the LLK build),
avoids a second standalone LLVM build dependency, and exercises the same
Triton-owned parser and dialects. The plan and issue file lists were updated to
match.

## Not delivered by PF17

- `bind_triton_template` and CPU launch: PF18.
- The kernel corpus, corpus-wide compatibility and rejection boundaries: PF19.
- Required CI/pytest markers and clean-wheel user documentation: PF20.
- Any GPU execution, Triton runtime/driver activation, Inductor integration
  through Triton, or claim that export proves native execution.
- `llk.capabilities()` still reports `native_execution: unavailable`; it now
  reports `upstream_triton: ttir-export-only` and the qualified version.
