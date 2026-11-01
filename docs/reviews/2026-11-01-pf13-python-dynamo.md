# PF13 verification record: strict Dynamo compilation through native MicroIR

Date: 2026-11-01
Issue: [#224](https://github.com/skg7on/DSLCompiler/issues/224)
Parent: [#203](https://github.com/skg7on/DSLCompiler/issues/203)
Plan: `docs/superpowers/plans/2026-10-18-python-frontend-refinement.md` §PF13
Implementation commit: `e399a0e0832183da1117992ed3911abe9b804b5c`

## Delivered

`torch.compile(backend="llk", fullgraph=True)` now compiles the supported ATen
subset through the same typed frontend, native compiler and checked invocation
that `@llk.jit` uses. The backend normalizes captured graphs with
AOTAutograd's ATen interface, imports one frontend request per graph output,
compiles it through the shared bounded cache, allocates fresh outputs and
invokes the native handles over caller-owned CPU tensors.

Nothing executes `gm.forward`. A graph outside the envelope raises
`UnsupportedFXError` carrying the offending FX node's name (and target); a
missing native service still reports `native_backend_unavailable` before any
graph diagnostic, preserving the PF01 registration contract.

## Accepted graph envelope

| Pattern | Program |
|---|---|
| `aten.add.Tensor(a, b)` | two-input elementwise tile program, blocks `8x16`, rounded grids with the proved in-bounds mask |
| `aten.mul.Tensor(a, b)` | the same elementwise family with `mul` |
| `aten.mm.default(a, b)` | one counted-K `dot` loop, one output store, zero accumulator |
| `silu(mm(x, wg)) * mm(x, wu)` | two dots sharing one left tile, exact SiLU contract, `mul` epilogue |

Every port is CPU, float32, positive rank-two, compact row-major and zero
offset. Elementwise operands have the identical shape. Matmul contraction
extents agree. Fused SwiGLU requires the identical left operand and two
distinct weight tensors of one shape. Contractions tile the output exactly with
the whole output as the launch block; the native importer derives the mapping
anchor and schedule from the target machine as it does for authored kernels.

The imported program is byte-for-byte the same tile program shape the source
frontend emits: `program_id`/`range`/`address`/`div`/`rem` index facts,
`broadcast` masks, `load`/`store`, and a typed `counted_loop` carrying
`dot`/`yield`. The native elementwise and contraction adapters therefore prove
the same access, mask and reduction contracts; Torch matching never bypasses
the importer.

## Rejected by node identity

| Graph | Code | `node_name` |
|---|---|---|
| `torch.sin(a + b)` | `unsupported_fx_operation` | `sin` |
| `(a + b).view(-1)`, `t`, `transpose`, `reshape`, `expand`, … | `unsupported_fx_view` | `view` (or the method) |
| `a.add_(b)`, `copy_`, other mutable targets | `unsupported_fx_mutation` | `add_` |
| scalar operand, broadcasting, composed arithmetic | `unsupported_fx_operand` / `unsupported_fx_shape` / `unsupported_fx_graph` | the offending node |
| standalone `silu` or a differently fused projection | `unsupported_fx_graph` | `silu` / the root node |
| symbolic example extents | `unsupported_symbolic_shape` | the placeholder |
| `requires_grad` inputs and the AOT forward compiler | `unsupported_training_contract` | placeholder / no node |
| non-contiguous, offset or non-float32 example metadata | `unsupported_tensor_layout` / `unsupported_fx_dtype` | the placeholder |
| non-CPU example metadata | `unsupported_fx_device` | the placeholder |
| tensor subclass | `unsupported_fx_subclass` | the placeholder |

The pinned checks are exercised literally:

```python
assert native_invoke_spy.call_count == 1
assert unsupported_fx_error.node_name == "sin"
```

A failed import is also asserted to call `gm.forward` zero times and leave
`gm.code` unchanged.

## Calling convention

`LLKBackend.__call__` returns `aot_autograd(fw_compiler=..., inference_compiler=...)`.
The inference compiler is the strict native compiler. The forward compiler is a
concrete `unsupported_training_contract` rejection until PF16 registers
backward formulas; AOTAutograd only selects it when autograd is required.

Compilers return `functorch.compile.make_boxed_func` callables, matching the
pinned Torch boxed contract; the tests also assert no "doesn't take boxed
arguments" warning. Compilation receives `FakeTensor` metadata only and never
reads storage. Invocation wraps each real tensor in an explicit
`TorchBuffer` provider that re-checks CPU/strided/float32/rank-two/contiguity/
offset/`requires_grad` immediately before the C ABI call; outputs are fresh
`torch.empty` allocations.

Multiple graph outputs are imported in graph order and each is backed by its
own compiled native handle; the boxed callable returns the outputs in the same
order. Outputs are never fused across graph results.

## Native identities

Observed on the reference build with the shared `x86-avx2` configuration:

| Graph | `source_adapter` | `plan_id` | `abi_hash` | `target_hash` | groups (selected/reference) |
|---|---|---|---|---|---|
| add `4x5` | `linalg-elementwise-v1` | `0xd97be0129d9e1131` | `0xc4a811ca1d1fa77b` | `0x17bbdd0bb651e319` | 0 / 0 |
| mm `4x6 @ 6x5` | `llk-contraction-v1` | `0x2c80c86edb…` | `0xfc98f672dd…` | `0x17bbdd0bb651e319` | 0 / 0 |
| SwiGLU `3x4x6` | `llk-contraction-v1` | `0x287959cbfd…` | `0xb5a151e187…` | `0x17bbdd0bb651e319` | 0 / 0 |

All runs report `backend=reference` and reach the `frontend`, `llk`, `micro`,
`lowered` and `executable` stages. The local host is aarch64 macOS, so the
selected AVX2 lane is not claimed here; `backend="selected-target"` remains a
separate target/CI qualification.

## Test evidence

Red step — before implementation:

```sh
python -m pytest test/python/test_fx_importer.py -q
# ModuleNotFoundError: No module named 'llk.torch._fx_importer'
```

Green step — focused PF13 suites (Python 3.14.8, Torch 2.14.1, LLVM/MLIR 24):

```sh
LLK_FRONTEND_LIBRARY=$PWD/build/lib/libLLKFrontendCAPI.dylib \
    python -m pytest test/python/test_dynamo_backend.py \
                     test/python/test_fx_importer.py -q
# 29 passed
```

Covered: real fullgraph add/multiply/mm/SwiGLU numerics with exactly one native
invocation each, tuple output order with two native invocations, changed
concrete shapes with two native invocations, the plain registered
`backend="llk"` form without explicit options, unsupported sin/in-place/view
node identity, training rejection, boxed-call conformance, and a direct
AOTAutograd lane that runs add/mm/SwiGLU without Dynamo.

Pinned Torch 2.9.0 lane (separate `/tmp` venv, Python 3.14.8):

```sh
LLK_FRONTEND_LIBRARY=$PWD/build/lib/libLLKFrontendCAPI.dylib \
    /tmp/llk-torch29-venv/bin/python -m pytest \
        test/python/test_fx_importer.py \
        'test/python/test_dynamo_backend.py::test_aot_boxed_conventions_run_natively_without_dynamo' -q
# 17 passed (14 importer + 3 direct AOT add/mm/SwiGLU)
```

Torch 2.9.0 itself refuses `torch.compile` on Python 3.14
(`RuntimeError: torch.compile is not supported on Python 3.14+`), so that lane
qualifies normalization and the boxed AOT calling convention directly rather
than claiming the Dynamo entry point. The pinned Python 3.11/3.12 fullgraph
lane is part of PF20's required-CI work.

Affected regression sets:

```sh
LLK_FRONTEND_LIBRARY=$PWD/build/lib/libLLKFrontendCAPI.dylib \
    python -m pytest test/python -q --ignore=test/python/test_package_install.py
# 203 passed

python -m pytest test/python/test_package_install.py -q
# 8 passed (clean Torch-absent wheel)
```

The registration subprocess tests now pin `LLK_FRONTEND_LIBRARY` to a missing
path, so they assert the unavailable-backend diagnostic deterministically
instead of depending on whether the outer test process had a library.

## Limitations (explicit, not silent)

- Concrete shapes only. The first release specializes the example metadata; a
  symbolic graph raises `unsupported_symbolic_shape`. Callers that need a new
  concrete specialization for a changed shape can pass `dynamic=False`, as the
  PF13 tests do; symbolic execution belongs with ABI v2 (#191–#199).
- Reference execution only in this local record. The selected-target backend is
  available through `options={"backend": "selected-target"}` but was not
  exercised on the aarch64 local host.
- Configuration discovery scans the source tree for `machines/` and
  `mapping/`. Packaging those resources into the wheel is PF20.
- One native kernel per graph output; independent outputs are not fused.
- No backward or training path, no bf16 graph input, no broadcasting, scalar
  operands, arbitrary elementwise chains or layout-dependent views.
- `capabilities()` remains the compiler-free PF01 report; this record does not
  widen it.
