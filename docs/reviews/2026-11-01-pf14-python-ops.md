# PF14 verification record: checked functional LLK Torch operators

Date: 2026-11-01
Issue: [#225](https://github.com/skg7on/DSLCompiler/issues/225)
Parent: [#203](https://github.com/skg7on/DSLCompiler/issues/203)
Plan: `docs/superpowers/plans/2026-10-18-python-frontend-refinement.md` §PF14
Implementation commit: `3ec9b45139651ff97802885e7f29594d53985056`

## Delivered

`llk.torch` registers four functional PyTorch custom operators:

| Operator | Schema |
|---|---|
| `llk::add` | `(Tensor a, Tensor b) -> Tensor` |
| `llk::mul` | `(Tensor a, Tensor b) -> Tensor` |
| `llk::matmul` | `(Tensor a, Tensor b) -> Tensor` |
| `llk::swiglu` | `(Tensor x, Tensor wg, Tensor wu) -> Tensor` |

Every schema is functional (`mutates_args=()`, no `!` alias annotation) and
registers a CPU kernel plus a FakeTensor kernel through the Torch 2.9
`torch.library.custom_op`/`register_fake` API. `llk.torch.register_operators()`
is public, runs during package import, and is idempotent: a repeated call or a
module reload preserves the exact registered `OpOverload` objects. A foreign or
partial owner of the `llk` operator namespace raises
`UnsupportedFeatureError(code="operator_registration_conflict")` instead of
being replaced.

The CPU kernels do not re-implement the native contract. They build their
request with the shared builder extracted from the Dynamo importer
(`_fx_importer.op_request`) and invoke the same bounded
`CompilationCache`/`CompileRequest` path. Caller tensors pass through
`TorchBuffer`, the checked adapter the Dynamo route already uses, so the C ABI
still receives compact row-major float32 rank-two CPU storage with zero offset
and no `requires_grad`. Mismatched operator extents fail in the shared builder
with `invalid_frontend_argument` before compilation.

Fake kernels return a fresh `torch.empty` output whose shape, dtype, device and
contiguous strides are derived from metadata alone. They never construct a
`TorchBuffer`, request `data_ptr`, compile a request or call the native runtime,
and reject unsupported layouts, dtypes and training metadata with the same
stable codes as the real path.

## Native identities

Observed on the local reference build with the shared `x86-avx2`
configuration (`backend=reference`):

| Operator (shapes) | `source_adapter` | `plan_id` | `abi_hash` | `target_hash` | groups (selected/reference) |
|---|---|---|---|---|---|
| add `4x5` | `linalg-elementwise-v1` | `0x9457ab9122fee833` | `0xfd9593b9fbb809c7` | `0x17bbdd0bb651e319` | 0 / 0 |
| mul `4x5` | `linalg-elementwise-v1` | `0x25f18444ec96f624` | `0xfd9593b9fbb809c7` | `0x17bbdd0bb651e319` | 0 / 0 |
| matmul `4x6 @ 6x5` | `llk-contraction-v1` | `0x2c80c86edbd4d51d` | `0xfc98f672dd72a80a` | `0x17bbdd0bb651e319` | 0 / 0 |
| swiglu `3x6, 6x4, 6x4` | `llk-contraction-v1` | `0x287959cbfd45786f` | `0xb5a151e1875a6a4b` | `0x17bbdd0bb651e319` | 0 / 0 |

All runs reach the `frontend`, `llk`, `micro`, `lowered` and `executable`
stages. The local host is aarch64 macOS, so the selected AVX2 lane is not
claimed here; `backend="selected-target"` remains a separate target/CI
qualification. The SwiGLU program carries the exact
`x/(1+exp(-x))` activation contract and is float32-only; a bf16 call is
rejected rather than approximated.

## Rejected inputs

| Input | Code |
|---|---|
| transposed, offset or otherwise non-compact operand | `invalid_frontend_buffer` |
| `torch.nn.Parameter` or another tensor subclass | `invalid_frontend_buffer` |
| non-float32 operand, including bf16 SwiGLU | `invalid_frontend_buffer` |
| rank other than two / non-positive extent | `invalid_frontend_buffer` |
| meta-device operand through the fake kernel | `invalid_frontend_buffer` |
| `requires_grad` operand (add and SwiGLU) | `unsupported_training_contract` |
| unequal add/mul extents, mismatched contraction extents or SwiGLU weights | `invalid_frontend_argument` |

A foreign `llk::add` schema fails package import with
`operator_registration_conflict`. Nothing is copied, converted or executed
eagerly on any rejection path.

## Test evidence

Red step — before implementation:

```sh
python -m pytest test/python/test_torch_ops.py -q
# ImportError: cannot import name '_ops' from 'llk.torch'
```

A probe custom op without `register_fake` fails the pinned check:

```text
opcheck(op, ...): test_faketensor failed with There was no fake impl registered
for <CustomOpDef(pf14probe::add)>. This is necessary for torch.compile/export/fx
tracing to work. Please use `impl.register_fake` to add an fake impl.
```

Green step — focused PF14 suites (Python 3.14.8, Torch 2.14.1, LLVM/MLIR 24):

```sh
LLK_FRONTEND_LIBRARY=$PWD/build/lib/libLLKFrontendCAPI.dylib \
    python -m pytest test/python/test_torch_ops.py \
                     test/python/test_fake_tensor_contract.py -q
# 27 passed (21 operator/registration + 6 fake-contract)
```

Pinned Torch 2.9.0 lane (same Python 3.14.8 interpreter, separate reference
venv):

```sh
LLK_FRONTEND_LIBRARY=$PWD/build/lib/libLLKFrontendCAPI.dylib \
    /tmp/llk-torch29-venv/bin/python -m pytest \
        test/python/test_torch_ops.py \
        test/python/test_fake_tensor_contract.py -q
# 27 passed
```

The `custom_op`, `register_fake` and `opcheck` forms used here exist in the
pinned reference release; no Torch 2.14-only behavior is required.

The pinned assertions are exercised literally:

```python
assert fake_data_ptr_spy.call_count == 0
assert fake_native_compile_spy.call_count == 0
```

Covered: functional schemas for all four operators; idempotent registration
across a module reload; foreign-owner rejection in a subprocess; exactly one
native invocation per numerical call with the declared fresh, non-aliased
output; specialization-cache reuse for a repeated call; `torch.library.opcheck`
`test_schema` and `test_faketensor` for every operator; transposed, offset,
subclass, dtype, rank, extent, meta-device and training rejections with zero
native calls; FakeTensor metadata parity; and the Torch-version guard.

Affected regression sets:

```sh
LLK_FRONTEND_LIBRARY=$PWD/build/lib/libLLKFrontendCAPI.dylib \
    python -m pytest -q
# 238 passed
```

The full Python suite includes PF01 package/registration, PF03–PF13 frontend,
cache, inspection, tail and Dynamo coverage.

## Limitations (explicit, not silent)

- Reference execution only in this record. The selected-target backend is
  available through `CompileOptions` but was not exercised on the aarch64 local
  host.
- Inference only. `requires_grad` inputs raise `unsupported_training_contract`;
  PF16 registers add/multiply/matmul autograd formulas and keeps SwiGLU
  training rejected.
- Float32, positive static rank-two, compact row-major, zero-offset CPU tensors
  only. No broadcasting, scalars, dtype conversion or automatic copies.
- Configuration discovery still scans the source tree for `machines/` and
  `mapping/`; packaging those resources is PF20.
- Inductor partitioning/composition (`llk_inductor`), Dynamo rewrites of ATen
  graphs to these operators and upstream Triton entry paths are later tasks.
- `capabilities()` remains the compiler-free PF01 report; this record does not
  widen it.
