"""PF16: registered autograd formulas and LLK-backed backward execution.

Every gradient assertion is tied to an independently evaluated f64 finite
difference or analytic reference. A correct forward result alone never counts
as training support: the tests also require a distinct native report for each
backward kernel that claims LLK execution and record the explicit transpose
materializations that remain outside the compact ABI.
"""

from __future__ import annotations

import gc
import importlib
import weakref
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
import torch.nn.functional as F
from torch._dynamo.exc import BackendCompilerFailed

from llk import TensorContractError, UnsupportedFeatureError
from llk.runtime import _ffi
from llk.torch import _autograd

REPO_ROOT = Path(__file__).resolve().parents[2]
COMPILE_OPTIONS = {"config_root": str(REPO_ROOT)}

# The plan pins these checks rather than accepting a looser default.
GRAD_RTOL = 1e-2
GRAD_ATOL = 1e-2
FD_EPS = 1e-3
MM_TOLERANCE = {"rtol": 1e-4, "atol": 1e-5}

SHAPES = {
    "add": ((4, 5), (4, 5)),
    "mul": ((4, 5), (4, 5)),
    "matmul": ((4, 6), (6, 5)),
}

REFERENCES = {
    "add": lambda a, b: a + b,
    "mul": lambda a, b: a * b,
    "matmul": lambda a, b: a @ b,
}


@pytest.fixture(autouse=True)
def reset_dynamo():
    torch._dynamo.reset()
    yield
    torch._dynamo.reset()


@pytest.fixture
def cold_inductor_cache(tmp_path, monkeypatch):
    """Force Inductor to trace the backward graph instead of reusing a cache."""
    cache = tmp_path / "inductor"
    cache.mkdir()
    monkeypatch.setenv("TORCHINDUCTOR_CACHE_DIR", str(cache))
    yield


def op(name):
    return getattr(torch.ops.llk, name).default


def values(*shapes, seed=0, requires_grad=False):
    generator = torch.Generator().manual_seed(seed)
    tensors = tuple(
        torch.randn(shape, generator=generator, dtype=torch.float32)
        for shape in shapes
    )
    if requires_grad:
        for tensor in tensors:
            tensor.requires_grad_(True)
    return tensors


def invoke_spy():
    return patch.object(_ffi, "native_invoke", wraps=_ffi.native_invoke)


def backend_error(call):
    with pytest.raises(BackendCompilerFailed) as caught:
        call()
    return caught.value.inner_exception


def native_grads(name, inputs):
    output = op(name)(*inputs)
    return torch.autograd.grad(output.sum(), inputs)


def direction(tensor, seed=11):
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(tensor.shape, generator=generator, dtype=torch.float32)


def directional_derivative(name, inputs, index, vector):
    """Evaluate one directional derivative in f64, independent of LLK."""
    reference = REFERENCES[name]

    def evaluate(offset):
        arguments = [value.detach().double() for value in inputs]
        arguments[index] = arguments[index] + offset * vector.double()
        return reference(*arguments).sum()

    return ((evaluate(FD_EPS) - evaluate(-FD_EPS)) / (2 * FD_EPS)).item()


# --- numeric qualification -------------------------------------------------


@pytest.mark.parametrize("name", sorted(SHAPES))
def test_directional_finite_differences_match_native_gradients(name):
    inputs = values(*SHAPES[name], requires_grad=True)
    gradients = native_grads(name, inputs)
    assert len(gradients) == len(inputs)

    for index, (gradient, value) in enumerate(zip(gradients, inputs)):
        vector = direction(value)
        analytic = (gradient * vector).sum().item()
        finite_difference = directional_derivative(name, inputs, index, vector)
        torch.testing.assert_close(
            torch.tensor(analytic),
            torch.tensor(finite_difference),
            rtol=GRAD_RTOL,
            atol=GRAD_ATOL,
        )


@pytest.mark.parametrize("name", sorted(SHAPES))
def test_analytic_gradients_match_the_f32_reference(name):
    inputs = values(*SHAPES[name], requires_grad=True)
    native = native_grads(name, inputs)
    reference_output = REFERENCES[name](*inputs)
    reference = torch.autograd.grad(reference_output.sum(), inputs)

    for native_grad, reference_grad, value in zip(native, reference, inputs):
        torch.testing.assert_close(
            native_grad, reference_grad, rtol=GRAD_RTOL, atol=GRAD_ATOL
        )
        assert native_grad.shape == value.shape
        assert native_grad.dtype is torch.float32
        assert native_grad.device.type == "cpu"
        if name != "add":
            # Native kernels return compact outputs; transposes were
            # materialized before the ABI boundary.
            assert native_grad.is_contiguous()
            assert int(native_grad.storage_offset()) == 0


@pytest.mark.parametrize("name", sorted(SHAPES))
def test_repeated_backward_reuses_saved_tensors(name):
    inputs = values(*SHAPES[name], requires_grad=True)
    output = op(name)(*inputs)
    loss = output.sum()

    loss.backward(retain_graph=True)
    first = tuple(value.grad.clone() for value in inputs)
    for value in inputs:
        value.grad = None

    loss.backward(retain_graph=True)
    second = tuple(value.grad.clone() for value in inputs)

    for first_grad, second_grad in zip(first, second):
        torch.testing.assert_close(first_grad, second_grad, rtol=0, atol=0)


def test_saved_tensors_outlive_the_caller_tensors():
    inputs = values(*SHAPES["matmul"], requires_grad=True)
    output = op("matmul")(*inputs)
    references = tuple(weakref.ref(value) for value in inputs)
    del inputs
    gc.collect()

    # The autograd formula saves both operands, so the graph keeps them alive.
    assert all(reference() is not None for reference in references)
    output.sum().backward()
    assert all(reference().grad is not None for reference in references)


# --- registration ----------------------------------------------------------


@pytest.mark.parametrize("name", sorted(SHAPES))
def test_registered_autograd_formula_passes_opcheck(name):
    inputs = values(*SHAPES[name], requires_grad=True)
    torch.library.opcheck(
        op(name), inputs, test_utils=("test_autograd_registration",)
    )


def test_autograd_registration_is_idempotent_across_module_reload():
    before = native_grads("mul", values(*SHAPES["mul"], requires_grad=True))
    importlib.reload(_autograd)
    _autograd.register_autograd_formulas()
    after = native_grads("mul", values(*SHAPES["mul"], requires_grad=True))
    for first, second in zip(before, after):
        torch.testing.assert_close(first, second, rtol=0, atol=0)


def test_swiglu_training_reports_the_unsupported_contract():
    inputs = (
        torch.randn(3, 6, requires_grad=True),
        torch.randn(6, 4),
        torch.randn(6, 4),
    )
    with invoke_spy() as calls:
        with pytest.raises(TensorContractError) as caught:
            op("swiglu")(*inputs)
    assert caught.value.code == "unsupported_training_contract"
    assert calls.call_count == 0


# --- native forward/backward reports --------------------------------------


def test_matmul_records_separate_forward_and_backward_native_reports():
    inputs = values(*SHAPES["matmul"], requires_grad=True)
    with _autograd.record_native_calls() as calls:
        output = op("matmul")(*inputs)
        output.sum().backward()

    assert [report.phase for report in calls.forward] == ["forward"]
    assert [report.phase for report in calls.backward] == [
        "backward",
        "backward",
    ]
    assert [report.operator for report in calls.backward] == ["matmul", "matmul"]
    assert all(report.report.stages for report in calls.all)
    assert calls.forward[0].report.native_request_identity
    assert len({id(report) for report in calls.all}) == 3

    materialized = {(item.operand, item.operation) for item in calls.materializations}
    assert materialized == {
        ("grad", "contiguous"),
        ("a", "transpose_contiguous"),
        ("b", "transpose_contiguous"),
    }


@pytest.mark.parametrize(
    ("name", "forward", "backward"),
    [("add", 1, 0), ("mul", 1, 2), ("matmul", 1, 2)],
)
def test_native_report_counts_follow_the_gradient_formula(name, forward, backward):
    inputs = values(*SHAPES[name], requires_grad=True)
    with _autograd.record_native_calls() as calls:
        output = op(name)(*inputs)
        output.sum().backward()

    assert len(calls.forward) == forward
    assert len(calls.backward) == backward
    assert all(report.operator == name for report in calls.all)


# --- strict AOTAutograd forward/backward compilers ------------------------


def compile_strict(fn):
    return torch.compile(
        fn,
        backend="llk",
        fullgraph=True,
        dynamic=False,
        options=COMPILE_OPTIONS,
    )


def test_strict_backend_trains_add_with_an_identity_backward():
    function = compile_strict(lambda a, b: a + b)
    inputs = values((4, 5), (4, 5), requires_grad=True)
    with invoke_spy() as calls:
        output = function(*inputs)
        output.sum().backward()

    assert calls.call_count == 1  # native forward only; the backward is identity
    torch.testing.assert_close(inputs[0].grad, torch.ones(4, 5), rtol=0, atol=0)
    torch.testing.assert_close(inputs[1].grad, torch.ones(4, 5), rtol=0, atol=0)


def test_strict_backend_compiles_two_native_mul_backward_kernels():
    function = compile_strict(lambda a, b: a * b)
    inputs = values((4, 5), (4, 5), requires_grad=True, seed=3)
    with (
        _autograd.record_native_calls() as reports,
        invoke_spy() as calls,
    ):
        output = function(*inputs)
        output.sum().backward()

    assert calls.call_count == 3  # one forward, two backward kernels
    assert [report.phase for report in reports.forward] == ["forward"]
    assert [report.phase for report in reports.backward] == [
        "backward",
        "backward",
    ]
    assert all(report.operator == "mul" for report in reports.all)
    torch.testing.assert_close(inputs[0].grad, inputs[1], rtol=0, atol=0)
    torch.testing.assert_close(inputs[1].grad, inputs[0], rtol=0, atol=0)


def test_strict_backend_rejects_a_matmul_backward_without_materialization():
    function = compile_strict(lambda a, b: a @ b)
    inputs = values((4, 6), (6, 5), requires_grad=True)
    output = function(*inputs)
    # AOTAutograd compiles the backward graph lazily when it is first needed.
    with pytest.raises(UnsupportedFeatureError) as caught:
        output.sum().backward()
    assert caught.value.code == "unsupported_training_contract"
    assert "transpose" in caught.value.message


def test_strict_backend_rejects_swiglu_training():
    function = compile_strict(lambda x, wg, wu: F.silu(x @ wg) * (x @ wu))
    inputs = values((3, 6), (6, 4), (6, 4), requires_grad=True)
    error = backend_error(lambda: function(*inputs))
    assert isinstance(error, UnsupportedFeatureError)
    assert error.code == "unsupported_training_contract"


# --- Inductor composition --------------------------------------------------


def test_llk_inductor_compiles_forward_and_backward_with_native_kernels(
    cold_inductor_cache,
):
    from torch._inductor import compile_fx as inductor_module

    function = torch.compile(
        lambda a, b: a @ b,
        backend="llk_inductor",
        fullgraph=True,
        dynamic=False,
    )
    inputs = values((4, 6), (6, 5), requires_grad=True, seed=5)
    with (
        _autograd.record_native_calls() as calls,
        patch.object(
            inductor_module, "compile_fx", wraps=inductor_module.compile_fx
        ) as inductor,
    ):
        output = function(*inputs)
        output.sum().backward()

    assert inductor.call_count == 1
    assert len(calls.forward) == 1
    assert len(calls.backward) == 2
    assert all(report.operator == "matmul" for report in calls.all)

    reference = (torch.ones(4, 5) @ inputs[1].detach().t()).float()
    torch.testing.assert_close(inputs[0].grad, reference, **MM_TOLERANCE)
    torch.testing.assert_close(
        inputs[1].grad, inputs[0].detach().t() @ torch.ones(4, 5), **MM_TOLERANCE
    )
    materialized = {(item.operand, item.operation) for item in calls.materializations}
    # The compiled partition materializes both transposes in ATen before the
    # native matmul; the traced tangent is already compact.
    assert materialized == {
        ("a", "transpose_contiguous"),
        ("b", "transpose_contiguous"),
    }


def test_llk_inductor_declines_swiglu_training_and_records_the_reason():
    def fn(x, wg, wu):
        return torch.tanh(F.silu(x @ wg) * (x @ wu))

    function = torch.compile(
        fn, backend="llk_inductor", fullgraph=True, dynamic=False
    )
    inputs = values((3, 6), (6, 4), (6, 4), requires_grad=True, seed=7)

    from llk.torch import _inductor_backend

    reports = []
    real_rewrite = _inductor_backend.rewrite_supported_subgraphs

    def capture(module, examples, options):
        result = real_rewrite(module, examples, options)
        reports.append(result[1])
        return result

    with patch.object(
        _inductor_backend, "rewrite_supported_subgraphs", side_effect=capture
    ):
        output = function(*inputs)
        output.sum().backward()

    assert len(reports) == 1
    declined = dict(reports[0].declined)
    assert declined["mul"] == "unsupported_training_contract"
    expected = torch.tanh(F.silu(inputs[0] @ inputs[1]) * (inputs[0] @ inputs[2]))
    torch.testing.assert_close(output, expected, **MM_TOLERANCE)
