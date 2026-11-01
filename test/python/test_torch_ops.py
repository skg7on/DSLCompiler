"""PF14: functional ``llk`` Torch operators backed by the shared native service.

Every numerical assertion counts the native invocations that produced the
result, so eager Torch arithmetic or a missing native lowering cannot pass.
Unsupported views, subclasses and training inputs must fail before the JIT is
entered rather than being copied into a silently different execution.
"""

from __future__ import annotations

import importlib
import os
import subprocess
import sys
import textwrap
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.nn.functional as F

from llk import TensorContractError, UnsupportedFeatureError
from llk.runtime import _ffi
from llk.torch import _compat, _ops

PYTHON_ROOT = Path(__file__).resolve().parents[2] / "python"
MM_TOLERANCE = {"rtol": 1e-4, "atol": 1e-5}

SCHEMAS = {
    "add": "llk::add(Tensor a, Tensor b) -> Tensor",
    "mul": "llk::mul(Tensor a, Tensor b) -> Tensor",
    "matmul": "llk::matmul(Tensor a, Tensor b) -> Tensor",
    "swiglu": "llk::swiglu(Tensor x, Tensor wg, Tensor wu) -> Tensor",
}

SHAPES = {
    "add": ((4, 5), (4, 5)),
    "mul": ((4, 5), (4, 5)),
    "matmul": ((4, 6), (6, 5)),
    "swiglu": ((3, 6), (6, 4), (6, 4)),
}

REFERENCES = {
    "add": (lambda a, b: a + b, {"rtol": 0, "atol": 0}),
    "mul": (lambda a, b: a * b, {"rtol": 0, "atol": 0}),
    "matmul": (lambda a, b: a @ b, MM_TOLERANCE),
    "swiglu": (lambda x, wg, wu: F.silu(x @ wg) * (x @ wu), MM_TOLERANCE),
}


def op(name):
    return getattr(torch.ops.llk, name)


def values(*shapes, seed=0):
    generator = torch.Generator().manual_seed(seed)
    return tuple(
        torch.randn(shape, generator=generator, dtype=torch.float32)
        for shape in shapes
    )


def invoke_spy():
    return patch.object(_ffi, "native_invoke", wraps=_ffi.native_invoke)


def counters():
    info = _ops.cache_info()
    return info or SimpleNamespace(hits=0, misses=0, compilations=0, evictions=0)


# --- registration ---------------------------------------------------------


def test_all_four_operators_are_registered_with_functional_schemas():
    for name, schema in SCHEMAS.items():
        overload = op(name).default
        assert str(overload._schema) == schema
        assert not overload._schema.is_mutable
    assert _ops.register_operators() is None


def test_registration_is_idempotent_across_module_reload():
    before = {name: op(name).default for name in SCHEMAS}
    importlib.reload(_ops)
    _ops.register_operators()
    for name, overload in before.items():
        assert op(name).default is overload


def test_foreign_operator_owner_is_not_replaced(tmp_path):
    code = """
        import torch
        from llk import UnsupportedFeatureError

        torch.library.define("llk::add", "(Tensor a, Tensor b) -> Tensor")
        try:
            import llk.torch
        except UnsupportedFeatureError as error:
            assert error.code == "operator_registration_conflict"
        else:
            raise AssertionError("Foreign llk::add was silently adopted")
    """
    env = dict(
        os.environ,
        PYTHONPATH=str(PYTHON_ROOT),
        PYTHONDONTWRITEBYTECODE="1",
    )
    result = subprocess.run(
        [sys.executable, "-u", "-c", textwrap.dedent(code)],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert result.returncode == 0, result.stdout + result.stderr


# --- native execution -----------------------------------------------------


@pytest.mark.parametrize("name", sorted(SHAPES))
def test_operator_executes_natively_with_a_fresh_declared_output(name):
    inputs = values(*SHAPES[name])
    reference, tolerance = REFERENCES[name]
    with invoke_spy() as calls:
        output = op(name)(*inputs)
    assert calls.call_count == 1
    expected = reference(*inputs)
    torch.testing.assert_close(output, expected, **tolerance)
    assert type(output) is torch.Tensor
    assert output.dtype is torch.float32
    assert output.device.type == "cpu"
    assert tuple(output.shape) == tuple(expected.shape)
    assert output.is_contiguous()
    # A fresh result, never a view or an alias of a caller operand.
    assert output._base is None
    assert output.data_ptr() not in {tensor.data_ptr() for tensor in inputs}


def test_repeated_calls_reuse_the_shared_specialization_cache():
    a, b = values((7, 9), (7, 9))
    before = counters()
    with invoke_spy() as calls:
        first = op("add")(a, b)
        second = op("add")(a, b)
    after = counters()
    assert calls.call_count == 2
    assert torch.equal(first, a + b)
    assert torch.equal(second, a + b)
    assert after.compilations == before.compilations + 1
    assert after.hits == before.hits + 1


# --- unsupported inputs fail before native entry --------------------------


def rejected(code, name, factory):
    inputs = factory()
    before = counters()
    with invoke_spy() as calls:
        with pytest.raises(TensorContractError) as caught:
            op(name)(*inputs)
    assert calls.call_count == 0
    assert caught.value.code == code
    assert counters().compilations == before.compilations
    return caught.value


def test_transposed_operand_is_rejected_instead_of_copied():
    rejected(
        "invalid_frontend_buffer",
        "add",
        lambda: (torch.randn(5, 4).t(), torch.randn(4, 5)),
    )


def test_storage_offset_view_is_rejected():
    rejected(
        "invalid_frontend_buffer",
        "add",
        lambda: (torch.randn(6, 5)[1:], torch.randn(5, 5)),
    )


def test_requires_grad_operand_reports_the_training_contract():
    def binary():
        a = torch.randn(4, 5)
        a.requires_grad_(True)
        return (a, torch.randn(4, 5))

    rejected("unsupported_training_contract", "add", binary)
    rejected(
        "unsupported_training_contract",
        "swiglu",
        lambda: (
            torch.randn(3, 6, requires_grad=True),
            torch.randn(6, 4),
            torch.randn(6, 4),
        ),
    )


def test_tensor_subclass_is_rejected_instead_of_copied():
    parameter = torch.nn.Parameter(torch.randn(4, 5), requires_grad=False)
    rejected(
        "invalid_frontend_buffer",
        "add",
        lambda: (parameter, torch.randn(4, 5)),
    )


def test_non_float32_operands_are_rejected():
    rejected(
        "invalid_frontend_buffer",
        "add",
        lambda: (
            torch.randn(4, 5, dtype=torch.bfloat16),
            torch.randn(4, 5, dtype=torch.bfloat16),
        ),
    )


def test_bf16_swiglu_is_declined():
    rejected(
        "invalid_frontend_buffer",
        "swiglu",
        lambda: (
            torch.randn(3, 6, dtype=torch.bfloat16),
            torch.randn(6, 4, dtype=torch.bfloat16),
            torch.randn(6, 4, dtype=torch.bfloat16),
        ),
    )


def test_rank_and_extent_mismatches_fail():
    rejected(
        "invalid_frontend_buffer",
        "add",
        lambda: (torch.randn(4), torch.randn(4)),
    )
    rejected(
        "invalid_frontend_argument",
        "add",
        lambda: (torch.randn(4, 5), torch.randn(4, 6)),
    )
    rejected(
        "invalid_frontend_argument",
        "matmul",
        lambda: (torch.randn(4, 6), torch.randn(5, 5)),
    )
    rejected(
        "invalid_frontend_argument",
        "swiglu",
        lambda: (torch.randn(3, 6), torch.randn(6, 4), torch.randn(6, 5)),
    )


def test_meta_tensor_is_rejected_without_native_work():
    rejected(
        "invalid_frontend_buffer",
        "add",
        lambda: (
            torch.empty(4, 5, device="meta"),
            torch.empty(4, 5, device="meta"),
        ),
    )


# --- Torch registration checks -------------------------------------------


@pytest.mark.parametrize("name", sorted(SHAPES))
def test_opcheck_schema_and_fake(name):
    torch.library.opcheck(
        op(name),
        values(*SHAPES[name]),
        test_utils=("test_schema", "test_faketensor"),
    )


def test_registration_rejects_an_unqualified_torch_version(monkeypatch):
    monkeypatch.setattr(_compat, "torch_version", lambda: (2, 8))
    with pytest.raises(UnsupportedFeatureError) as caught:
        _ops.register_operators()
    assert caught.value.code == "unsupported_torch_version"
