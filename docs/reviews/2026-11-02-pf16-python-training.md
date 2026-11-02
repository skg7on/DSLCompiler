# PF16 verification record: qualified training for add, multiply and matmul

Date: 2026-11-02
Issue: [#227](https://github.com/skg7on/DSLCompiler/issues/227)
Parent: [#203](https://github.com/skg7on/DSLCompiler/issues/203)
Plan: `docs/superpowers/plans/2026-10-18-python-frontend-refinement.md` §PF16
Implementation commit: `b9b5ad4`

## Delivered

`import llk.torch` now registers first-order autograd formulas for `llk::add`,
`llk::mul` and `llk::matmul` through `torch.library.register_autograd`. The
registered `llk::swiglu` operator still has no formula: a `requires_grad`
operand raises `TensorContractError(code="unsupported_training_contract")`
before any compilation or native invocation, and strict `backend="llk"` rejects
a trained SwiGLU graph with the same code. No path falls back to eager autograd.

The strict `llk` backend now supplies AOTAutograd forward and backward
compilers. The training forward compiler accepts `requires_grad` placeholders
and returns the saved primals unchanged; the backward compiler compiles an
identity gradient (`add`) as a boxed pass-through and `mul` gradients as native
`mul` kernels. A backward graph that still contains an `aten.t`/view is
rejected with `unsupported_training_contract`, because the compact row-major
ABI cannot read a transposed view.

`backend="llk_inductor"` now permits `requires_grad` operands for
add/multiply/matmul forward rewrites. AOTAutograd traces the registered formula
for the replaced `llk::*` call, so the Inductor-compiled backward invokes two
native LLK matmuls while Inductor owns the graph orchestration and the ATen
transpose copies. A trained SwiGLU region is declined with
`unsupported_training_contract` and left to Inductor; it is never claimed as an
LLK training region.

`_autograd.record_native_calls()` returns a frozen
`TrainingReport(forward, backward, materializations)`. Each forward and
backward entry carries the operator, the phase and the complete
`CompilationReport` of the kernel that ran. Materializations name every layout
copy that remains outside LLK (incoming expanded gradient, transposed operand).

## Backward partition

| Forward operator | Native backward kernels | Non-LLK backward work |
|---|---|---|
| `add` | none — gradient is the incoming tangent | none |
| `mul` | two `llk::mul` calls (`grad * b`, `a * grad`) | one `contiguous` copy when the incoming gradient is an expanded view |
| `matmul` | two `llk::matmul` calls (`grad @ B^T`, `A^T @ grad`) | ATen `transpose(0, 1).contiguous()` for both operands; `contiguous` for a non-compact incoming gradient |
| `swiglu` | none | rejected: `unsupported_training_contract` |

The formulas save both operands with `ctx.save_for_backward`, so repeated
`backward(retain_graph=True)` and backward after the caller drops its tensor
references keep working. The native kernels receive detached storage: the
formula, not the kernel, owns differentiation. Higher-order gradients are
rejected explicitly rather than silently detached.

## Native identities

Observed on the local reference build with the shared `x86-avx2` configuration
(`backend=reference`):

| Phase | Operator (shapes) | `source_adapter` | `plan_id` | `abi_hash` | `target_hash` |
|---|---|---|---|---|---|
| forward | add `4x5` | `linalg-elementwise-v1` | `0x9457ab9122fee833` | `0xfd9593b9fbb809c7` | `0x17bbdd0bb651e319` |
| backward | add identity | — | — | — | — |
| forward | mul `4x5` | `linalg-elementwise-v1` | `0x25f18444ec96f624` | `0xfd9593b9fbb809c7` | `0x17bbdd0bb651e319` |
| backward | mul `grad*b` | `linalg-elementwise-v1` | `0x25f18444ec96f624` | `0xfd9593b9fbb809c7` | `0x17bbdd0bb651e319` |
| backward | mul `a*grad` | `linalg-elementwise-v1` | `0x25f18444ec96f624` | `0xfd9593b9fbb809c7` | `0x17bbdd0bb651e319` |
| forward | matmul `4x6 @ 6x5` | `llk-contraction-v1` | `0x2c80c86edbd4d51d` | `0xfc98f672dd72a80a` | `0x17bbdd0bb651e319` |
| backward | matmul `grad @ B^T` | `llk-contraction-v1` | `0x4fad53b2055b909c` | `0x3abf62facb9e25ab` | `0x17bbdd0bb651e319` |
| backward | matmul `A^T @ grad` | `llk-contraction-v1` | `0x0e69bd582fb90496` | `0xc9c6f5c619f163f4` | `0x17bbdd0bb651e319` |

All entries reach the `frontend`, `llk`, `micro`, `lowered` and `executable`
stages. The forward and backward matmul plans are distinct because the
contraction extents differ; an `add`/`mul` backward reuses the forward plan but
is still reported as a separate backward kernel invocation.

## Test evidence

Red steps observed:

```sh
LLK_FRONTEND_LIBRARY=$PWD/build/lib/libLLKFrontendCAPI.dylib \
    python -m pytest test/python/test_torch_training.py -q
# ImportError: cannot import name '_autograd' from 'llk.torch'
```

Two further red/green cycles were watched during implementation:

- The incoming tangent of `out.sum().backward()` is an expanded view with zero
  strides, so the first native `mul`/`matmul` backward was rejected by
  `TorchBuffer`. The formulas now materialize it with an explicit recorded
  `contiguous()` copy.
- AOTAutograd compiles the backward graph lazily at the first backward, which
  moved the strict matmul rejection from the forward call to `backward()`; the
  test asserts the actual contract. A cold `TORCHINDUCTOR_CACHE_DIR` is required
  to observe the traced transpose materializations instead of reusing a
  previously compiled Inductor artifact.

Green step — focused PF16 lane (Python 3.14.8, Torch 2.14.1, LLVM/MLIR 24):

```sh
LLK_FRONTEND_LIBRARY=$PWD/build/lib/libLLKFrontendCAPI.dylib \
CXX=/usr/bin/clang++ CC=/usr/bin/clang \
    python -m pytest test/python/test_torch_training.py -q
# 25 passed
```

Affected Torch regression sets:

```sh
LLK_FRONTEND_LIBRARY=$PWD/build/lib/libLLKFrontendCAPI.dylib \
CXX=/usr/bin/clang++ CC=/usr/bin/clang python -m pytest \
    test/python/test_torch_ops.py test/python/test_fake_tensor_contract.py \
    test/python/test_dynamo_backend.py test/python/test_fx_importer.py \
    test/python/test_inductor_composition.py \
    test/python/test_inductor_partitioning.py \
    test/python/test_torch_training.py -q
# 105 passed
```

Full Python suite:

```sh
LLK_FRONTEND_LIBRARY=$PWD/build/lib/libLLKFrontendCAPI.dylib \
CXX=/usr/bin/clang++ CC=/usr/bin/clang python -m pytest -q
# 287 passed
```

The pinned checks are exercised literally:

```python
torch.testing.assert_close(native_grad, reference_grad, rtol=1e-2, atol=1e-2)
assert swiglu_training_error.code == "unsupported_training_contract"
```

Covered numerically: f32 directional finite differences against an independent
f64 evaluation for add/mul/matmul (`eps=1e-3`, `atol=rtol=1e-2`), direct
analytic f32 gradients, shape/device/dtype preservation, compact backward
outputs, repeated backward, saved-tensor lifetime after the caller drops its
references, `torch.library.opcheck` `test_autograd_registration`, and
idempotent registration across a module reload.

Covered by execution evidence: one forward and one distinct backward native
report set for matmul; two native `mul` backward kernels under strict
`backend="llk"`; identity backward for `add`; `llk_inductor` running one real
`compile_fx` with one native forward and two native matmul backward calls; and
the declined `unsupported_training_contract` SwiGLU region report.

Pinned Torch 2.9.0 lane — the local reference venv is Python 3.14.8, where
Torch 2.9.0 itself raises `torch.compile is not supported on Python 3.14+`.
The registration/eager subset runs:

```sh
LLK_FRONTEND_LIBRARY=$PWD/build/lib/libLLKFrontendCAPI.dylib \
    /tmp/llk-torch29-venv/bin/python -m pytest \
        test/python/test_torch_training.py test/python/test_torch_ops.py \
        test/python/test_fake_tensor_contract.py -q -k "not llk_inductor"
# 47 passed, 4 failed (the four torch.compile tests; runtime guard above)
```

The four failures are the interpreter's documented `torch.compile` exclusion,
not an LLK behavior difference. The declared Torch 2.9.0 reference matrix is
Python 3.11/3.12 (`python/llk/compatibility.json`); neither interpreter is
installed on this host, so the 2.9 compile lanes are not claimed here.

`CXX`/`CC` are needed only on this macOS host, where the Homebrew LLVM clang
first on `PATH` cannot compile Torch's C++ headers. The Linux CI toolchain
needs no such override.

## Limitations (explicit, not silent)

- First-order gradients only. A `create_graph=True` backward through an LLK
  operator raises `unsupported_training_contract`.
- Reference execution only in this record; selected AVX2 training remains a
  separate qualified lane.
- Float32, positive static rank-two, compact row-major, zero-offset CPU tensors.
  No broadcasting, scalars, dtype conversion or hidden copies.
- `llk_inductor` compiles the backward graph itself; LLK owns only the formula's
  `llk::*` kernels. Transpose/copy regions and the graph orchestration are
  attributed to Inductor/ATen. This is not a full Inductor codegen backend.
- Strict `backend="llk"` rejects a matmul backward because the transpose
  materialization would need a copy kernel outside the LLK ABI. Use
  `backend="llk_inductor"` for that region.
- The native-call recorder is process-visible while a context is active;
  concurrent unrelated training steps should use separate processes rather than
  sharing one recorder.
