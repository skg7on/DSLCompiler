# PF15 verification record: scoped LLK external kernels inside Inductor

Date: 2026-11-02
Issue: [#226](https://github.com/skg7on/DSLCompiler/issues/226)
Parent: [#203](https://github.com/skg7on/DSLCompiler/issues/203)
Plan: `docs/superpowers/plans/2026-10-18-python-frontend-refinement.md` §PF15
Implementation commit: `d3dbbd65028aa4cc9942466d6f737325227a4bcf`

## Delivered

`import llk.torch` now registers `llk_inductor` beside the strict `llk`
backend. `torch.compile(fullgraph=True, backend="llk_inductor")` clones the
Dynamo-captured graph, replaces effect-safe regions that satisfy the native
contract with the PF14 `llk::*` custom operators, and delegates the whole
rewritten graph to `torch._inductor.compile_fx.compile_fx`. The registered
backend callable is the `_InductorBackend` singleton; repeated registration and
module reload preserve it, and a foreign owner of the name raises
`UnsupportedFeatureError(code="backend_registration_conflict")`.

`rewrite_supported_subgraphs(gm, example_inputs, options)` never mutates its
input. It clones through `torch.fx.Graph.graph_copy`, reads per-node
`meta["val"]`/`meta["example_value"]` fake tensors, and — for graphs without
that metadata — replays the graph on fake tensors built from `example_inputs`.
It returns a frozen
`PartitionReport(llk_regions, matched_nodes, declined)` where `matched_nodes`
names the accepted region roots in graph order and `declined` pairs every
remaining candidate with a stable reason. Declined nodes stay in the graph for
Inductor; nothing is copied, converted or executed eagerly.

The backend is external-kernel composition, not a device scheduler or an
Inductor codegen backend. It does not claim fusion across an external call,
does not patch Inductor or Dynamo globally, and passes `torch.compile`
`options` to Inductor as per-compilation `config_patches` (verified unchanged
in `torch._inductor.config` afterwards).

## Partition policy

| Pattern | LLK operator | Requirements |
|---|---|---|
| `operator.add` / `torch.add` / `aten.add.Tensor`, alpha=1 | `llk::add` | equal shapes, both operands native-eligible |
| `operator.mul` / `torch.mul` / `aten.mul.Tensor` | `llk::mul` | equal shapes, both operands native-eligible |
| `operator.matmul` / `torch.matmul` / `aten.mm` / `aten.matmul` | `llk::matmul` | rank-two operands, contraction extents agree |
| `silu(mm(x, wg)) * mm(x, wu)` (either order) | `llk::swiglu` | one shared left operand, two distinct weights, exact f32 SiLU contract, no internal node with an external user |

Both Dynamo-level callables and normalized ATen overloads are recognized, so a
graph can be partitioned before AOTAutograd (the Inductor route) or after it
(direct ATen fixtures). An operand is native-eligible only when it is a graph
placeholder or the output of an already-accepted LLK region. Chained regions
read the replacement node, so `a + b + c` partitions into two native calls and
one entry point.

## Declined inputs

| Condition | Reason |
|---|---|
| not a recognized pattern / standalone `silu`, `sin`, `exp`, `sum`, `tanh` | `unmatched_operation` |
| operand produced by a non-native node; incompatible activation fusion | `operand_not_native` |
| dtype other than float32, including bf16 SwiGLU | `unsupported_dtype` |
| non-CPU device | `unsupported_device` |
| rank not two, non-positive extent or disagreeing extents | `unsupported_shape` |
| symbolic (`SymInt`) extents | `unsupported_dynamic_shape` |
| non-compact row-major or non-zero storage offset | `unsupported_layout` |
| `requires_grad` operand | `unsupported_training_contract` |
| any in-place tensor effect anywhere in the graph | `unsupported_effect` |
| internal SwiGLU node used outside the region | `shared_user` |

The `unsupported_effect` rule is deliberately conservative: without alias
analysis a mutating node anywhere makes the compact, copy-free operand
contract unprovable, so the graph is left to Inductor with an explicit reason.
The rejected-path policy is local to partitioning; unlike the strict `llk`
backend, unmatched nodes do not fail the graph.

## Native identities

Identities of the operators the partitioned graphs invoke, observed on the
local reference build with the shared `x86-avx2` configuration
(`backend=reference`):

| Operator (shapes) | `source_adapter` | `plan_id` | `abi_hash` | `target_hash` | stages |
|---|---|---|---|---|---|
| add `4x5` | `linalg-elementwise-v1` | `0x9457ab9122fee833` | `0xfd9593b9fbb809c7` | `0x17bbdd0bb651e319` | frontend, llk, micro, lowered, executable |
| matmul `4x6 @ 6x5` | `llk-contraction-v1` | `0x2c80c86edbd4d51d` | `0xfc98f672dd72a80a` | `0x17bbdd0bb651e319` | frontend, llk, micro, lowered, executable |
| swiglu `3x6, 6x4, 6x4` | `llk-contraction-v1` | `0x287959cbfd45786f` | `0xb5a151e1875a6a4b` | `0x17bbdd0bb651e319` | frontend, llk, micro, lowered, executable |

The rewritten graph for `torch.compile(lambda a, b: torch.sin(a + b),
backend="llk_inductor")` is:

```text
graph():
    %l_a_ : torch.Tensor [num_users=1] = placeholder[target=L_a_]
    %l_b_ : torch.Tensor [num_users=1] = placeholder[target=L_b_]
    %add_default : [num_users=1] = call_function[target=torch.ops.llk.add.default](args = (%l_a_, %l_b_), kwargs = {})
    %sin : [num_users=1] = call_function[target=torch.sin](args = (%add_default,), kwargs = {})
    return (sin,)
```

with `PartitionReport(llk_regions=1, matched_nodes=('add',),
declined=(('sin', 'unmatched_operation'),))`, exactly one `native_invoke`, and
one `compile_fx` call.

## Test evidence

Red step — before implementation:

```sh
python -m pytest test/python/test_inductor_partitioning.py \
                 test/python/test_inductor_composition.py -q
# ImportError: cannot import name '_inductor_backend' from 'llk.torch'
```

Two further red/green cycles were watched during implementation: chained
regions failed with a stale-node error until operands were remapped to the
replacement nodes, and the mutation test failed until the graph-level effect
rule was added.

Green step — focused PF15 lane, including real Inductor C++ code generation
(cold `TORCHINDUCTOR_CACHE_DIR`, Python 3.14.8, Torch 2.14.1):

```sh
LLK_FRONTEND_LIBRARY=$PWD/build/lib/libLLKFrontendCAPI.dylib \
CXX=/usr/bin/clang++ CC=/usr/bin/clang \
    python -m pytest test/python/test_inductor_composition.py \
                     test/python/test_inductor_partitioning.py -q
# 23 passed (7 composition + 16 partitioning)
```

The pinned assertions are exercised literally:

```python
assert partition_report.llk_regions == 1
assert native_invoke_spy.call_count == 1
assert inductor_compile_spy.call_count == 1
```

Covered composition cases: `sin(a+b)` (one LLK add, one Inductor compile, exact
output), SwiGLU plus unmatched `tanh` (one LLK SwiGLU call, Inductor compiles
the remainder, `1e-4/1e-5`), shared users of one native result feeding two
Inductor activations, a graph with no native region (`exp(a)+exp(b)` reports
zero partitions and never calls native code), bf16 SwiGLU (zero partitions,
Inductor-owned, `1e-2/1e-2`), scoped Inductor config patches, and the same
function compiled afterwards with `backend="inductor"` showing zero native
calls. Rewrite cases cover report freezing, cloning, accepted/declined node
names and reasons, chained regions, symbolic extents, non-compact inputs,
missing metadata, mutation, foreign-backend rejection and option validation.

Affected regression sets:

```sh
LLK_FRONTEND_LIBRARY=$PWD/build/lib/libLLKFrontendCAPI.dylib \
CXX=/usr/bin/clang++ CC=/usr/bin/clang python -m pytest -q
# 261 passed
```

`CXX`/`CC` are needed only on this macOS host, where the Homebrew LLVM clang
first on `PATH` cannot compile Torch's C++ headers (`NAN`/`INFINITY`
undeclared); the project's compiler build already selects its own toolchain.
The Linux CI toolchain needs no such override.

## Limitations (explicit, not silent)

- Reference execution only in this record; selected-target execution remains a
  separately qualified lane.
- Inference only. `requires_grad` operands decline partitioning; PF16
  registers the trained formulas.
- Float32, positive static rank-two, compact row-major, zero-offset CPU
  operands. No broadcasting, scalars, dtype conversion or automatic copies.
- No cross-call fusion, no AOTInductor export/serialization, no Triton or GPU
  route. The strict `llk` backend and the `@llk.jit` decorator are unchanged.
- `capabilities()` remains the compiler-free PF01 report; this record does not
  widen it.
