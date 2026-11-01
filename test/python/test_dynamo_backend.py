"""PF13: real ``torch.compile(backend="llk")`` execution over native MicroIR.

Every numerical assertion runs on caller tensors and counts the native
invocations that produced the result. A result computed eagerly, with a missing
native lowering, or through ``gm.forward`` cannot satisfy both checks.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from unittest.mock import patch

import llk
import pytest
import torch
import torch.nn.functional as F
from torch._dynamo.exc import BackendCompilerFailed

from llk.runtime import _ffi
from llk.torch import get_backend
from llk.torch._fx_importer import UnsupportedFXError

REPO_ROOT = Path(__file__).resolve().parents[2]
COMPILE_OPTIONS = {"config_root": str(REPO_ROOT)}
MM_TOLERANCE = {"rtol": 1e-4, "atol": 1e-5}


@pytest.fixture(autouse=True)
def reset_dynamo():
    torch._dynamo.reset()
    yield
    torch._dynamo.reset()


def compile(fn):
    # First-release specialization is concrete: changed shapes must trigger a
    # new concrete compilation rather than a symbolic graph.
    return torch.compile(
        fn,
        backend="llk",
        fullgraph=True,
        dynamic=False,
        options=COMPILE_OPTIONS,
    )


def values(*shapes, seed=0):
    generator = torch.Generator().manual_seed(seed)
    return tuple(
        torch.randn(shape, generator=generator, dtype=torch.float32)
        for shape in shapes
    )


def invoke_spy():
    return patch.object(_ffi, "native_invoke", wraps=_ffi.native_invoke)


def backend_error(call):
    with pytest.raises(BackendCompilerFailed) as caught:
        call()
    return caught.value.inner_exception


def _direct_module(targets, shapes):
    graph = torch.fx.Graph()
    arguments = [graph.placeholder(f"arg{index}") for index in range(len(shapes))]
    graph.output((targets(graph, *arguments),))
    with torch._subclasses.fake_tensor.FakeTensorMode():
        example = tuple(
            torch.empty(shape, dtype=torch.float32) for shape in shapes
        )
    return torch.fx.GraphModule({}, graph), example


def _direct_add(graph, a, b):
    return graph.call_function(torch.ops.aten.add.Tensor, (a, b))


def _direct_matmul(graph, a, b):
    return graph.call_function(torch.ops.aten.mm.default, (a, b))


def _direct_swiglu(graph, x, wg, wu):
    gate = graph.call_function(torch.ops.aten.mm.default, (x, wg))
    return graph.call_function(
        torch.ops.aten.mul.Tensor,
        (
            graph.call_function(torch.ops.aten.silu.default, (gate,)),
            graph.call_function(torch.ops.aten.mm.default, (x, wu)),
        ),
    )


@pytest.mark.parametrize(
    ("targets", "shapes", "reference", "tolerance"),
    [
        (_direct_add, ((4, 5), (4, 5)), lambda a, b: a + b, {"rtol": 0, "atol": 0}),
        (
            _direct_matmul,
            ((4, 6), (6, 5)),
            lambda a, b: a @ b,
            MM_TOLERANCE,
        ),
        (
            _direct_swiglu,
            ((3, 6), (6, 4), (6, 4)),
            lambda x, wg, wu: F.silu(x @ wg) * (x @ wu),
            MM_TOLERANCE,
        ),
    ],
)
def test_aot_boxed_conventions_run_natively_without_dynamo(
    targets, shapes, reference, tolerance
):
    # This lane qualifies the AOTAutograd calling convention on a Torch release
    # whose torch.compile wrapper may not support the local interpreter.
    module, example = _direct_module(targets, shapes)
    compiled = get_backend()(module, example)
    arguments = values(*shapes)
    with invoke_spy() as calls:
        result = compiled(*arguments)
    assert calls.call_count == 1
    assert isinstance(result, (list, tuple)) and len(result) == 1
    torch.testing.assert_close(result[0], reference(*arguments), **tolerance)


def test_registered_backend_compiles_without_explicit_options():
    # The plain public form must discover the repository configuration tree.
    function = torch.compile(
        lambda a, b: a + b, backend="llk", fullgraph=True, dynamic=False
    )
    a, b = values((4, 5), (4, 5))
    with invoke_spy() as calls:
        result = function(a, b)
    assert calls.call_count == 1
    assert torch.equal(result, a + b)


def test_fullgraph_add_and_multiply_execute_natively():
    add = compile(lambda a, b: a + b)
    mul = compile(lambda a, b: a * b)
    a, b = values((4, 5), (4, 5))
    with invoke_spy() as add_calls:
        added = add(a, b)
    with invoke_spy() as mul_calls:
        multiplied = mul(a, b)
    assert add_calls.call_count == 1
    assert mul_calls.call_count == 1
    assert torch.equal(added, a + b)
    assert torch.equal(multiplied, a * b)
    assert added.data_ptr() not in (a.data_ptr(), b.data_ptr())


def test_fullgraph_matmul_executes_natively():
    mm = compile(lambda a, b: a @ b)
    a, b = values((4, 6), (6, 5))
    with invoke_spy() as calls:
        result = mm(a, b)
    assert calls.call_count == 1
    torch.testing.assert_close(result, a @ b, **MM_TOLERANCE)


def test_fullgraph_swiglu_executes_natively():
    swiglu = compile(lambda x, wg, wu: F.silu(x @ wg) * (x @ wu))
    x, wg, wu = values((3, 6), (6, 4), (6, 4))
    with invoke_spy() as calls:
        result = swiglu(x, wg, wu)
    assert calls.call_count == 1
    torch.testing.assert_close(result, F.silu(x @ wg) * (x @ wu), **MM_TOLERANCE)


def test_tuple_output_order_is_preserved():
    pair = compile(lambda a, b: (a + b, a * b))
    a, b = values((4, 5), (4, 5))
    with invoke_spy() as calls:
        added, multiplied = pair(a, b)
    assert calls.call_count == 2
    assert torch.equal(added, a + b)
    assert torch.equal(multiplied, a * b)


def test_changed_concrete_shapes_recompile_and_execute():
    add = compile(lambda a, b: a + b)
    first = values((4, 5), (4, 5), seed=1)
    second = values((6, 7), (6, 7), seed=2)
    with invoke_spy() as calls:
        added_first = add(*first)
        added_second = add(*second)
    assert calls.call_count == 2
    assert torch.equal(added_first, first[0] + first[1])
    assert torch.equal(added_second, second[0] + second[1])


def test_compiled_callable_is_boxed_without_the_unboxed_fallback_warning():
    function = compile(lambda a, b: a + b)
    a, b = values((4, 5), (4, 5))
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        result = function(a, b)
    assert torch.equal(result, a + b)
    assert not any("boxed arguments" in str(item.message) for item in recorded)


def test_unsupported_sin_fails_strictly_with_node_identity():
    function = compile(lambda a, b: torch.sin(a + b))
    error = backend_error(lambda: function(*values((4, 5), (4, 5))))
    assert isinstance(error, UnsupportedFXError)
    assert error.node_name == "sin"
    assert error.code == "unsupported_fx_operation"


def test_in_place_graph_fails_with_node_identity():
    def inplace(a, b):
        a.add_(b)
        return a

    function = compile(inplace)
    error = backend_error(lambda: function(*values((4, 5), (4, 5))))
    assert isinstance(error, UnsupportedFXError)
    assert error.node_name == "add_"
    assert error.code == "unsupported_fx_mutation"


def test_view_dependent_graph_fails_with_node_identity():
    function = compile(lambda a, b: (a + b).view(-1))
    error = backend_error(lambda: function(*values((4, 5), (4, 5))))
    assert isinstance(error, UnsupportedFXError)
    assert error.node_name == "view"
    assert error.code == "unsupported_fx_view"


def test_training_graphs_are_rejected_until_pf16():
    function = compile(lambda a, b: a + b)
    a, b = values((4, 5), (4, 5))
    a.requires_grad_(True)
    b.requires_grad_(True)
    error = backend_error(lambda: function(a, b))
    assert isinstance(error, llk.UnsupportedFeatureError)
    assert error.code == "unsupported_training_contract"


def test_invocation_reads_real_buffers_after_fake_metadata_compilation():
    # Compilation sees only FakeTensor metadata (see test_fx_importer). The
    # compiled callable must still read the buffers supplied at each call.
    function = compile(lambda a, b: a + b)
    function(*values((4, 5), (4, 5), seed=1))
    a, b = values((4, 5), (4, 5), seed=2)
    a = a + 1.0
    b = b * 3.0
    with invoke_spy() as calls:
        result = function(a, b)
    assert calls.call_count == 1
    assert torch.equal(result, a + b)
    assert get_backend() is not None
