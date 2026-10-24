# PF07 versioned frontend C API verification

**Date:** 2026-10-24 (Asia/Shanghai).

**Scope:** [issue #218](https://github.com/skg7on/DSLCompiler/issues/218), PF07 in
[the frontend plan](../superpowers/plans/2026-10-18-python-frontend-refinement.md).
Base `main` `9cdb3b1` contains merged PF06 (PR #247). Work is on
`feat/issue218-frontend-capi` in the isolated worktree
`.claude/worktrees/feat+issue218-frontend-capi`, which owns its own `build/`.

## Delivered behavior

`LLKFrontendCAPI` is a shared library over the existing `LLKFrontend` service.
It adds no compiler behavior: `llk_frontend_compile_v1` calls `parseRequest`
followed by `compileFrontend`, and `llk_frontend_invoke_v1` marshals
`LLKBuffer2D` into `InvocationBuffer2D` and calls `FrontendCompilation::invoke`.
Arity, element type, extent, compact strides, zero offset, allocation bounds and
output/input disjointness stay with `validateMappedInvocation`, so a C caller is
checked by exactly the same code path as a C++ caller. Nothing in this change
duplicates or weakens that validator.

The frozen v1 surface is the eight declared entry points in
`include/LLK/CAPI/Frontend.h`: `llk_frontend_api_version`,
`compile_v1`, `invoke_v1`, `report_v1`, `inspect_v1`, `error_json_v1`,
`error_free_v1` and `release_v1`. Element tags are fixed ABI values
(F32=1 … I8=5) and are translated explicitly, so a tag the ABI does not define
fails instead of defaulting to f32. `LLKBuffer2D` mirrors `MemRef2D` field for
field and adds the declared tag and allocation extent; the test asserts the
field offsets the Python ctypes structure will mirror.

Return convention is 0/1. Every failing entry point leaves its output handle or
string null/zero and, when the caller supplied an error slot, stores an owned
`LLKError`. Its JSON form carries `code`, `message` and `stage`, plus `file`,
`line` and `column` when the native diagnostic had a source span. `LLKString`
views are borrowed from the owning handle or error; no buffer pointer is
retained past a call.

Ownership is reference counted inside the handle. `release_v1` drops the
owner's reference; a call that is already running holds its own reference, keeps
the compilation alive, and frees it with the last reference. Starting a call on a
released handle and releasing twice remain caller errors and are documented as
such, because no implementation can detect a use-after-free of a raw pointer.

`LLK_BUILD_PYTHON_CAPI` defaults to OFF. Enabling it turns on position
independent code for the project archives, which a shared library needs on ELF
linkers, and registers the `CAPI` test category plus the plain-C consumer. Both
subdirectories stay out of the build when the option is off; a default
configuration of this branch was verified separately. `LLKCAPIFrontendTests` is
also in the `check-llk` dependency list, so `check-llk` builds the consumer
rather than discovering a missing binary.

Only the declared symbols are exported. The project archives are built with
default visibility, so the target uses a linker export list: an
`-exported_symbols_list` for Mach-O and a `--version-script` for ELF. `nm -gU
build/lib/libLLKFrontendCAPI.dylib` reports exactly the eight `llk_frontend_*`
symbols. Hiding the rest matters because the library is loaded into a Python
process where another MLIR copy may already be resident. The JIT is unaffected:
`MappedExecutable` hands `memrefCopy` to LLJIT by address rather than by dynamic
lookup, and the full test suite still executes.

The C boundary reports every failure as `llvm::Error`. With the project's
`-fno-exceptions` setting no throw site exists; for builds that enable exceptions
the entry points run through a guard that converts an escaping exception into a
`frontend_native_error`, so nothing unwinds through the caller's frames.

## Red/green and validation

The consumers were written first against the published header with no
implementation. The red build failed to link with undefined
`llk_frontend_api_version`, `_compile_v1`, `_invoke_v1`, `_report_v1`,
`_inspect_v1`, `_error_json_v1`, `_error_free_v1` and `_release_v1` — the missing
API, not a test defect. `build-red.log` records it.

```sh
cmake -S . -B build -G Ninja \
  -DLLVM_PROJECT_BUILD_DIR=/Users/skg7on/Workspace/Projects/llvm-project/build \
  -DLLK_BUILD_PYTHON_CAPI=ON
cmake --build build --target LLKCAPIFrontendTests LLKCAPITests -j4
ctest --test-dir build -R FrontendCAPI --output-on-failure
cmake --build build -j4
ctest --test-dir build --output-on-failure
python3 scripts/check-license-headers.py
```

| Lane | Actual result |
| --- | --- |
| `ctest -R FrontendCAPI` | 5 passed, 0 failed, 0 skipped |
| Full `ctest` in this worktree | 1,463 registered; 1,462 run; 3 failed, 6 skipped, 1 disabled |
| Frontend category (`LLKFrontendTests`) | 56 passed, 0 failed |
| Default configuration (`LLK_BUILD_PYTHON_CAPI=OFF`) | Configures and generates without CAPI targets |
| License-header and deprecated-API audits | Clean |

The full-suite failures are `MappedAVX2Acceptance`, `GenericElementwiseAVX2Test`
and `FrameworkPrimitivesAVX2Test`, which require x86_64 AVX2 and fail on this
aarch64 host exactly as they did on the unchanged PF06 base. They are not
described as a green suite. The skips are the three legacy JIT and three selected
AVX2 fixtures; the staged Triton test remains disabled.

Local toolchain: LLVM/MLIR **24.0.0git**, Homebrew clang **20.1.8**, macOS arm64.
No Python package was used or changed.

### C consumer coverage

`test/CAPI/frontend.c` is plain C: no C++, no LLVM headers, no native
`InvocationBuffer2D`. It asserts the API version, the five element tags and the
`LLKBuffer2D`/`LLKString` layout on 64-bit hosts, then compiles the PF03 fixture
with its frozen target configuration embedded, so no live checkout path is read.

- Malformed text fails with `invalid_frontend_schema` and a null handle; the
  error slot is initialized from a sentinel on every path.
- Three compile/invoke/release rounds each allocate their own handle and produce
  identical results.
- The report is non-empty and identifies `"backend":"reference"` and
  `"source_adapter":"linalg-elementwise-v1"`; `micro` inspection contains the
  bound `micro.kernel`; `asm` fails with `inspection_unavailable` instead of
  returning empty text.
- Wrong input and output arity, a BF16 tag on an f32 port, a changed extent,
  transposed strides, a nonzero offset, an allocation one element too small, an
  undefined element tag and an output aliasing an input all fail with
  `invalid_frontend_buffer`, and a valid call still succeeds afterwards.
- Successful responses are compared with an independent `258 + 3*i` table for all
  128 elements, both inputs are unchanged, and two guard floats on each side of
  every buffer still hold their sentinel.

### C++ coverage

`test/CAPI/frontend_errors.cpp` uses only the C++ standard library and gtest.

- An unregistered opcode produces `invalid_frontend_schema` with the source span
  (`file`, `line`, `column`), so the Python layer can raise a located error.
- Truncated text and a null text pointer are rejected without dereferencing.
- Every buffer contract violation above is repeated through one handle, and
  output/input aliasing is reported as an overlap.
- `report_v1` returns the same borrowed pointer twice and across an invocation;
  `inspect_v1` borrows from the same owner, and an unavailable stage leaves the
  output zeroed.
- Four threads invoke the same handle concurrently with distinct output buffers:
  all four succeed and each output matches the literal table.

### Sanitizer status

AddressSanitizer cannot run on this host at all: a trivial `-fsanitize=address`
program never completes under Homebrew clang 20.1.8 on this macOS build, and the
same happens for the ASan-instrumented consumer here. That is an environment
limitation, not a property of this change, and no ASan/LSan result is claimed for
it. The sanitizer lane is deferred to the Linux tooling the plan names, which
PF20 registers as a required lane. What is verified in-process today is the guard
regions around every buffer, repeated success and error paths on one handle, and
concurrent invocation.

## Capability limits and handoff

PF07 exposes the C ABI only. There is still no Python buffer adapter, launch
protocol, specialization cache, inspection property or CLI route; those are
PF08, PF09 and PF12. The supported envelope remains the PF05/PF06 one: static
rank-two compact row-major strict-f32 elementwise add/multiply, ABI v1, explicit
`reference` or `selected-target` execution. Dynamic shapes (#191–#199) and
measurement persistence (#161–#164) remain separate. The umbrella #203 stays
open, and closing #218 does not imply that any Python entry path executes.
