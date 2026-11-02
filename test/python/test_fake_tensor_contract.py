"""PF14: FakeTensor kernels report metadata without touching storage.

A truthful fake kernel builds a fresh output from shape/dtype/device metadata.
It must never request a data pointer, compile a request or enter the native
runtime; those three events are spied on around every fake execution.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode

import llk.torch  # noqa: F401 - importing the package registers the operators
from llk import TensorContractError
from llk.runtime import _ffi

SHAPES = {
    "add": ((4, 5), (4, 5)),
    "mul": ((4, 5), (4, 5)),
    "matmul": ((4, 6), (6, 5)),
    "swiglu": ((3, 6), (6, 4), (6, 4)),
}

OUTPUT_SHAPES = {
    "add": (4, 5),
    "mul": (4, 5),
    "matmul": (4, 5),
    "swiglu": (3, 4),
}


def op(name):
    return getattr(torch.ops.llk, name)


def fake_inputs(*shapes):
    return tuple(
        torch.empty(shape, dtype=torch.float32, device="cpu") for shape in shapes
    )


@pytest.mark.parametrize("name", sorted(SHAPES))
def test_fake_execution_never_reads_storage_or_enters_the_native_runtime(name):
    with patch.object(
        torch.Tensor, "data_ptr", wraps=torch.Tensor.data_ptr
    ) as fake_data_ptr_spy:
        with patch.object(
            _ffi, "compile_request", wraps=_ffi.compile_request
        ) as fake_native_compile_spy:
            with patch.object(
                _ffi, "native_invoke", wraps=_ffi.native_invoke
            ) as fake_native_invoke_spy:
                with FakeTensorMode():
                    inputs = fake_inputs(*SHAPES[name])
                    output = op(name)(*inputs)
                    assert isinstance(output, FakeTensor)
                    assert tuple(output.shape) == OUTPUT_SHAPES[name]
                    assert output.dtype is torch.float32
                    assert output.device.type == "cpu"
                    assert output.is_contiguous()
                    assert output.requires_grad is False
    assert fake_data_ptr_spy.call_count == 0
    assert fake_native_compile_spy.call_count == 0
    assert fake_native_invoke_spy.call_count == 0


def test_fake_kernels_reject_unsupported_metadata_without_native_work():
    with patch.object(
        torch.Tensor, "data_ptr", wraps=torch.Tensor.data_ptr
    ) as fake_data_ptr_spy:
        with patch.object(
            _ffi, "compile_request", wraps=_ffi.compile_request
        ) as fake_native_compile_spy:
            with FakeTensorMode():
                transposed = torch.empty(5, 4).t()
                contiguous = torch.empty(4, 5)
                with pytest.raises(TensorContractError) as layout:
                    op("add")(transposed, contiguous)
                assert layout.value.code == "invalid_frontend_buffer"

                training = torch.empty(3, 6, requires_grad=True)
                with pytest.raises(TensorContractError) as gradient:
                    op("swiglu")(
                        training, torch.empty(6, 4), torch.empty(6, 4)
                    )
                assert gradient.value.code == "unsupported_training_contract"

                with pytest.raises(TensorContractError) as dtype:
                    op("add")(torch.empty(4, 5, dtype=torch.bfloat16), contiguous)
                assert dtype.value.code == "invalid_frontend_buffer"
    assert fake_data_ptr_spy.call_count == 0
    assert fake_native_compile_spy.call_count == 0


def test_fake_metadata_matches_the_real_native_result():
    inputs = tuple(
        torch.randn(shape, dtype=torch.float32) for shape in SHAPES["matmul"]
    )
    real = op("matmul")(*inputs)
    with FakeTensorMode():
        fake = op("matmul")(*fake_inputs(*SHAPES["matmul"]))
    assert type(fake) is FakeTensor
    assert tuple(fake.shape) == tuple(real.shape)
    assert fake.stride() == real.stride()
    assert fake.dtype is real.dtype
    assert fake.device == real.device
