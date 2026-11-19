# PF19 verification record: upstream Triton compatibility corpus

Date: 2026-11-18
Issue: [#230](https://github.com/skg7on/DSLCompiler/issues/230)
Parent: [#203](https://github.com/skg7on/DSLCompiler/issues/203)
Plan: `docs/superpowers/plans/2026-10-18-python-frontend-refinement.md` §PF19
Dependencies: [#229](https://github.com/skg7on/DSLCompiler/issues/229) (PF18,
merged in #286), [#221](https://github.com/skg7on/DSLCompiler/issues/221) (PF10,
merged in #276)

## Delivered

The compatibility corpus is declared once in
`docs/design/triton-compatibility-v1.json` and executed by
`test/python/compat/test_triton_corpus.py` through a dedicated interpreter
(`test/python/compat/_triton_corpus_lane.py`). Every executable fixture is a
genuine `@triton.jit` kernel in `test/python/compat/kernels/`, exported through
the qualified upstream compiler (Triton 3.6.0, target `cuda:80`), bound to
concrete rank-two f32 CPU tensors, and compiled/invoked by the LLK native
frontend in a Triton-free worker. The report records per-fixture frontend,
binding, reference, selected and invocation status; rejected fixtures declare
the stable diagnostic instead of counting as compatibility.

Exporter extensions were limited to patterns that reach an already verified
source adapter:

- `tt.expand_dims` / `tt.broadcast` rank views are translated (transposes and
  arbitrary reshapes stay rejected as `unsupported_triton_layout`).
- `tt.dot` is exported only with IEEE input precision; the CUDA default tf32
  mode is rejected as `unsupported_triton_numerics`.
- Dense tile constants are read from the printed module text (the qualified
  MLIR bindings expose no dense accessor) and kept as their true value, so a
  non-zero accumulator seed is preserved rather than substituted with zero.
- The exact `tl.sigmoid(x) * x` lowering becomes one `silu` operation; the
  elided chain is dead-code eliminated and every constant value is checked.
- `tl.range(..., num_stages=...)` — the qualified language's only asynchronous
  shared-memory staging request — is rejected as
  `unsupported_triton_memory_effect`; reductions are reported as
  `unsupported_triton_operation` before their printed rank view.

The launcher gained a rank-two tile binder that replays a neutral template into
the program shape the native importer proves: affine index facts over the two
program ids, the two tile lanes and the counted induction variable; pointer
tiles with an explicit flat offset decomposed by the tensor's row stride; the
counted loop emitted with its concrete K bound and block step; per-axis masks
combined through the framework's `select` conjunction; and the fused `silu`
epilogue.

## Executed corpus

Reference lane, one native invocation per fixture (`LLKFrontendCAPI`, reference
backend). Errors are the maximum absolute difference from an independent NumPy
oracle.

| Fixture | Source adapter | Max abs error |
|---|---|---|
| `elementwise.add.full` | `linalg-elementwise-v1` | 0 |
| `elementwise.add.tiled` | `linalg-elementwise-v1` | 0 |
| `elementwise.multiply.full` | `linalg-elementwise-v1` | 0 |
| `elementwise.multiply.tiled` | `linalg-elementwise-v1` | 0 |
| `elementwise.add.masked-tail` (33x65, 8x16 blocks, rounded grid) | `linalg-elementwise-v1` | 0 |
| `elementwise.multiply.masked-tail` (33x65, 8x16 blocks, rounded grid) | `linalg-elementwise-v1` | 0 |
| `elementwise.add.flattened` (4x16 storage, one 64-lane block) | `linalg-elementwise-v1` | 0 |
| `contraction.matmul.f32` (32x16 * 16x64) | `llk-contraction-v1` | 2.86e-06 |
| `contraction.matmul.accumulated` (caller `[32, 64]` seed) | `llk-contraction-v1` | 3.81e-06 |
| `contraction.swiglu.f32` (16x16 * 16x64, exact SiLU) | `llk-contraction-v1` | 6.10e-05 |

Each record carries the genuine TTIR hash and template identity, the bound
program identity, and the native report's frontend/compiler/workload/
source-adapter identities, target/machine/rules hashes, plan id, ABI hash,
selected/reference group counts and reached stages. The reference lane reports
`backend=reference` and `reference_group_count=0`.

## Rejection boundaries

| Construct | Stage | Diagnostic |
|---|---|---|
| atomic | export | `unsupported_triton_memory_effect` |
| shared async pipeline (`tl.range(num_stages=2)`) | export | `unsupported_triton_memory_effect` |
| arbitrary reduce (`tl.sum`) | export | `unsupported_triton_operation` |
| runtime `while` | export | `unsupported_triton_operation` |
| runtime stride (`offs * step`) | binding | `unsupported_triton_launch` |
| GPU-only inline assembly | export | `unsupported_triton_operation` |

None of these is counted as executed compatibility.

## Selected-target lane

The corpus binds and would execute every fixture with
`backend="selected-target"`. On this host (`arm64`) the x86-avx2 selected lane
is not supported, so each record reports
`selected.status = "unavailable"`, `reason = "host_isa_unsupported"` with the
host machine recorded. Reference results are never substituted for selected
results. The selected lane requires `selected_group_count > 0` and
`reference_group_count == 0` when it runs; the PF20 Linux x86_64/AVX2 CI lane is
the first environment that can execute it.

## Focused commands and results

Local environment: project `.venv` (Python 3.14.8), Triton 3.6.0 source
checkout at `93f05e19a8024938c320d82854e307d602b5a05d`, worktree build with
`-DLLK_BUILD_PYTHON_CAPI=ON`.

```sh
export LLK_TRITON_PYTHONPATH=/Users/skg7on/Workspace/Projects/triton/python
export LLK_FRONTEND_LIBRARY=$PWD/build/lib/libLLKFrontendCAPI.dylib

python -m pytest test/python/compat/test_triton_corpus.py -q
# 2 passed

python -m pytest test/python/compat/test_ttir_export.py -q
# 20 passed

python -m pytest test/python/compat -q
# 33 passed

python -m pytest test/python/compat/_triton_launch_inner.py -q -rs
# 22 passed, 4 skipped (upstream/LLK LLVM coexistence gate)

python -m pytest test/python -q
# 311 passed, 9 failed
```

The nine failures are the pre-existing local Inductor C++ compilation issue
(Homebrew `llvm@20` include ordering), reproduced unchanged on the unmodified
main checkout; they are unrelated to this change and are `test_inductor_*` and
`test_torch_training.py` cases.

## Supported boundary

- Genuine upstream Triton 3.6.0 flat elementwise f32 add/multiply (including
  masked, rounded-grid tails over a declared rank-two domain) and the counted
  f32 matmul, caller-accumulated matmul and fused SwiGLU contractions.
- Contractions require IEEE f32 dot precision, exact output tiling, a K extent
  that is a positive multiple of the block, unmasked (fully in-bounds) loads,
  and either a zero tile or a caller `[M, N]` accumulator seed.
- Elementwise tails require the kernel's own `(rows < M) & (cols < N)` mask;
  the importer proves it and the runtime bounds every access.
- Atomics, asynchronous staging, reductions, runtime control flow, runtime
  strides, layout conversions, reduced-precision dot modes and GPU-only
  instructions remain explicit diagnostics.

## Handoff

- PF20: run the corpus in required CI on Linux x86_64 with AVX2, which
  exercises the selected-target lane and its no-fallback assertion.
- Persistent measurement records, dynamic shapes and additional upstream
  releases remain separate work under #161–#164, #191–#199 and #203.
