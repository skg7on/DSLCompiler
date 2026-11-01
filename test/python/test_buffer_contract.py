"""PF08: Python buffer adapters checked before native JIT entry."""

from __future__ import annotations

import ctypes

import native_cases as cases
import pytest
from llk import TensorContractError
from llk.frontend import TensorSpec
from llk.runtime import _buffers


def port(shape=(16, 16), dtype="f32", direction="input", effect="read"):
    return TensorSpec("tensor", shape, dtype, (shape[1], 1), direction, effect)


def output_port(shape=(16, 16)):
    return port(shape, direction="output", effect="write")


class ArrayProvider:
    """Minimal PEP 3118-adjacent provider with an explicit owner base."""

    def __init__(
        self,
        pointer,
        shape=(2, 2),
        base=None,
        read_only=False,
        strides=None,
        typestr="<f4",
    ):
        self.base = base
        item = int(typestr[-1])
        self.__array_interface__ = {
            "version": 3,
            "shape": tuple(shape),
            "typestr": typestr,
            "data": (pointer, read_only),
            "strides": strides or (shape[1] * item, item),
            "desc": "PF08 buffer contract provider",
        }


def test_native_descriptor_layout_is_frozen():
    assert ctypes.sizeof(_buffers.LLKBuffer2D) == 72
    assert _buffers.LLKBuffer2D.allocated.offset == 0
    assert _buffers.LLKBuffer2D.aligned.offset == 8
    assert _buffers.LLKBuffer2D.offset.offset == 16
    assert _buffers.LLKBuffer2D.sizes.offset == 24
    assert _buffers.LLKBuffer2D.strides.offset == 40
    assert _buffers.LLKBuffer2D.element_type.offset == 56
    assert _buffers.LLKBuffer2D.allocation_bytes.offset == 64


def test_element_tags_match_the_abi():
    assert _buffers.ELEMENT_TAGS == {
        "f32": 1,
        "bf16": 2,
        "f16": 3,
        "i32": 4,
        "i8": 5,
    }


def test_compact_buffer_becomes_a_zero_offset_descriptor():
    guarded = cases.Guarded(cases.TILED_SHAPE)
    lease = _buffers.describe(port(), guarded.view)
    descriptor = lease.descriptor
    payload = ctypes.addressof(guarded.storage) + cases.GUARD * 4
    assert descriptor.allocated == payload
    assert descriptor.aligned == payload
    assert descriptor.offset == 0
    assert (descriptor.sizes[0], descriptor.sizes[1]) == cases.TILED_SHAPE
    assert (descriptor.strides[0], descriptor.strides[1]) == (cases.TILED_SHAPE[1], 1)
    assert descriptor.element_type == _buffers.ELEMENT_TAGS["f32"]
    assert descriptor.allocation_bytes == cases.TILED_SHAPE[0] * cases.TILED_SHAPE[1] * 4
    assert lease.owner is not None


def test_non_contiguous_and_negative_stride_views_are_rejected():
    guarded = cases.Guarded(cases.TILED_SHAPE)
    reversed_rows = guarded.view[::-1]
    with pytest.raises(TensorContractError):
        _buffers.describe(port(), reversed_rows)
    storage = (ctypes.c_float * (16 * 16))()
    transposed = ArrayProvider(
        ctypes.addressof(storage), (16, 16), strides=(4, 64)
    )
    with pytest.raises(TensorContractError):
        _buffers.describe(port(), transposed)


def test_array_interface_offset_view_is_rejected():
    storage = (ctypes.c_float * 20)()
    base = ArrayProvider(ctypes.addressof(storage))
    offset_view = ArrayProvider(ctypes.addressof(storage) + 4 * 4, base=base)
    with pytest.raises(TensorContractError) as caught:
        _buffers.describe(port((2, 2)), offset_view)
    assert caught.value.code == "invalid_frontend_buffer"
    lease = _buffers.describe(port((2, 2)), base)
    assert lease.descriptor.allocated == ctypes.addressof(storage)


def test_read_only_provider_cannot_be_an_output():
    read_only = memoryview(bytes(16 * 16 * 4)).cast("f", (16, 16))
    assert read_only.readonly
    _buffers.describe(port(), read_only)
    with pytest.raises(TensorContractError):
        _buffers.describe(output_port(), read_only)


def test_wrong_shape_dtype_and_extent_are_rejected():
    guarded = cases.Guarded(cases.TILED_SHAPE)
    with pytest.raises(TensorContractError):
        _buffers.describe(port((8, 16)), guarded.view)
    with pytest.raises(TensorContractError):
        _buffers.describe(port(dtype="i32"), guarded.view)
    short = bytearray(16 * 16 * 4 - 4)
    with pytest.raises(TensorContractError) as caught:
        _buffers.describe(port(), short)
    assert caught.value.code == "invalid_frontend_buffer"
    doubles = (ctypes.c_double * (16 * 16))()
    double_view = memoryview(doubles).cast("B").cast("d", (16, 16))
    with pytest.raises(TensorContractError):
        _buffers.describe(port(), double_view)


def test_byte_provider_with_exact_extent_is_accepted():
    storage = bytearray(16 * 16 * 4)
    lease = _buffers.describe(port(), storage)
    assert lease.descriptor.allocation_bytes == len(storage)
    assert lease.descriptor.allocated == ctypes.addressof(ctypes.c_char.from_buffer(storage))


def test_rank_one_byte_providers_must_be_unit_stride():
    storage = bytearray(16 * 16 * 4)
    with pytest.raises(TensorContractError):
        _buffers.describe(port(), memoryview(storage)[::-1])
    strided = memoryview(bytearray(16 * 16 * 4 * 2))[::2]
    with pytest.raises(TensorContractError):
        _buffers.describe(port(), strided)


def test_buffer_lease_pins_resizable_storage():
    storage = bytearray(16 * 16 * 4)
    lease = _buffers.describe(port(), storage)
    with pytest.raises(BufferError):
        storage.extend(b"x")
    del lease
    storage.extend(b"x")


def test_integer_and_half_precision_array_types_are_accepted():
    int8_storage = (ctypes.c_int8 * (16 * 16))()
    int8_lease = _buffers.describe(
        port(dtype="i8"), ArrayProvider(ctypes.addressof(int8_storage), (16, 16), typestr="|i1")
    )
    assert int8_lease.descriptor.element_type == _buffers.ELEMENT_TAGS["i8"]
    half_storage = (ctypes.c_uint16 * (16 * 16))()
    half_lease = _buffers.describe(
        port(dtype="f16"), ArrayProvider(ctypes.addressof(half_storage), (16, 16), typestr="<f2")
    )
    assert half_lease.descriptor.element_type == _buffers.ELEMENT_TAGS["f16"]
    double_storage = (ctypes.c_double * (16 * 16))()
    with pytest.raises(TensorContractError):
        _buffers.describe(
            port(), ArrayProvider(ctypes.addressof(double_storage), (16, 16), typestr="<f8")
        )


def test_unaligned_provider_is_rejected():
    storage = bytearray(16 * 16 * 4 + 1)
    unaligned = memoryview(storage).cast("B")[1:].cast("f", (16, 16))
    with pytest.raises(TensorContractError):
        _buffers.describe(port(), unaligned)


def test_unrelated_objects_are_not_buffers():
    with pytest.raises(TensorContractError):
        _buffers.describe(port(), object())


def test_outputs_must_not_overlap_inputs_or_each_other():
    inputs = cases.Guarded(cases.TILED_SHAPE)
    output = cases.Guarded(cases.TILED_SHAPE)
    with pytest.raises(TensorContractError):
        _buffers.prepare_invocation(
            (inputs.view,), (port(),), (inputs.view,), (output_port(),)
        )
    with pytest.raises(TensorContractError):
        _buffers.prepare_invocation(
            (inputs.view,), (port(),), (output.view, output.view), (output_port(), output_port())
        )


def test_arity_is_checked_before_descriptors():
    guarded = cases.Guarded(cases.TILED_SHAPE)
    with pytest.raises(TensorContractError):
        _buffers.prepare_invocation((guarded.view,), (port(), port()), (), ())
