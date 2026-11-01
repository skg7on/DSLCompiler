"""Checked Python buffer adapters for the frontend C ABI v1.

The native contract accepts rank-two compact row-major storage with a zero
descriptor offset. A provider is read through the Python buffer protocol (or an
explicit ``__array_interface__``) without copying, and every failure here happens
before the JIT is entered. Providers are never cached: a lease keeps the caller's
object strongly referenced only for the duration of one invocation.
"""

from __future__ import annotations

import ctypes
from dataclasses import dataclass

from llk._errors import TensorContractError
from llk.frontend._ir import TensorSpec

ELEMENT_TAGS = {"f32": 1, "bf16": 2, "f16": 3, "i32": 4, "i8": 5}

# PEP 3118 type codes are only meaningful together with their byte width:
# NumPy spells int8 as '|i1' and float16 as '<f2'.
_FORMATS = {
    "b": {1: "i8"},
    "i": {1: "i8", 4: "i32"},
    "l": {4: "i32"},
    "f": {2: "f16", 4: "f32"},
    "e": {2: "f16"},
}
_ELEMENT_WIDTH = {"f32": 4, "f16": 2, "i32": 4, "i8": 1}

_PyBUF_STRIDES = 0x0018


class LLKBuffer2D(ctypes.Structure):
    """Field-for-field mirror of the native ``LLKBuffer2D``."""

    _fields_ = [
        ("allocated", ctypes.c_void_p),
        ("aligned", ctypes.c_void_p),
        ("offset", ctypes.c_int64),
        ("sizes", ctypes.c_int64 * 2),
        ("strides", ctypes.c_int64 * 2),
        ("element_type", ctypes.c_uint32),
        ("allocation_bytes", ctypes.c_uint64),
    ]


class _PyBuffer(ctypes.Structure):
    _fields_ = [
        ("buf", ctypes.c_void_p),
        ("obj", ctypes.c_void_p),
        ("len", ctypes.c_ssize_t),
        ("itemsize", ctypes.c_ssize_t),
        ("readonly", ctypes.c_int),
        ("ndim", ctypes.c_int),
        ("format", ctypes.c_char_p),
        ("shape", ctypes.POINTER(ctypes.c_ssize_t)),
        ("strides", ctypes.POINTER(ctypes.c_ssize_t)),
        ("suboffsets", ctypes.POINTER(ctypes.c_ssize_t)),
        ("internal", ctypes.c_void_p),
    ]


def _fail(message):
    raise TensorContractError(message, code="invalid_frontend_buffer")


def _dtype_from_descriptor(text, width):
    """Resolve a PEP 3118 type code and byte width to a supported element."""
    if not text:
        _fail("buffer has no element format")
    code = text[1:] if text[0] in "@=<>!|" else text
    if code and code[-1].isdigit():
        declared = int(code[-1])
        code = code[:-1]
        if width is not None and declared != width:
            _fail(f"element type {text} does not match its {width}-byte item")
        width = declared
    if width is None:
        _fail(f"element type {text} has no byte width")
    dtype = _FORMATS.get(code, {}).get(width)
    if dtype is None:
        _fail(f"unsupported element type: {text}")
    return dtype


def _dtype_from_format(text, itemsize):
    return _dtype_from_descriptor(text, itemsize)


def _dtype_from_typestr(text):
    if not isinstance(text, str) or len(text) < 2:
        _fail("array interface has no element type")
    return _dtype_from_descriptor(text, None)


_GET_BUFFER = ctypes.pythonapi.PyObject_GetBuffer
_GET_BUFFER.argtypes = [ctypes.py_object, ctypes.POINTER(_PyBuffer), ctypes.c_int]
_GET_BUFFER.restype = ctypes.c_int
_RELEASE_BUFFER = ctypes.pythonapi.PyBuffer_Release
_RELEASE_BUFFER.argtypes = [ctypes.POINTER(_PyBuffer)]
_RELEASE_BUFFER.restype = None


def _address_of(provider):
    view = _PyBuffer()
    if _GET_BUFFER(provider, ctypes.byref(view), _PyBUF_STRIDES) != 0:
        _fail("provider did not export a readable buffer")
    try:
        if not view.buf:
            _fail("provider exported a null buffer")
        return int(view.buf)
    finally:
        _RELEASE_BUFFER(ctypes.byref(view))


def _array_interface_layout(provider):
    info = provider.__array_interface__
    if not isinstance(info, dict):
        _fail("array interface must be a mapping")
    shape = tuple(info.get("shape", ()))
    if len(shape) != 2 or any(type(value) is not int or value < 1 for value in shape):
        _fail("provider must expose a positive rank-two shape")
    dtype = _dtype_from_typestr(info.get("typestr"))
    data = info.get("data")
    if isinstance(data, tuple) and len(data) >= 2:
        address, read_only = data[0], bool(data[1])
    elif isinstance(data, int):
        address, read_only = data, False
    else:
        _fail("array interface has no data pointer")
    if type(address) is not int or address <= 0:
        _fail("array interface has no valid data pointer")
    base = getattr(provider, "base", None)
    base_info = getattr(base, "__array_interface__", None)
    if isinstance(base_info, dict):
        base_data = base_info.get("data")
        base_address = base_data[0] if isinstance(base_data, tuple) else base_data
        if isinstance(base_address, int) and base_address != address:
            _fail("tensor views with a storage offset are unsupported")
    strides = info.get("strides")
    item = _ELEMENT_WIDTH[dtype]
    compact = (shape[1] * item, item)
    if strides is None:
        strides = compact
    else:
        strides = tuple(strides)
        if len(strides) != 2 or any(type(value) is not int for value in strides):
            _fail("array interface strides must be two integers")
    allocation = shape[0] * shape[1] * item
    try:
        exported = memoryview(provider)
    except TypeError:
        exported = None
    if exported is not None and exported.nbytes < allocation:
        _fail("array interface allocation is shorter than its declared shape")
    return address, shape, strides, dtype, read_only, allocation, exported


def _buffer_layout(provider):
    try:
        view = memoryview(provider)
    except TypeError:
        _fail("provider does not export the buffer protocol")
    address = _address_of(provider)
    read_only = bool(view.readonly)
    if view.ndim == 1 and view.itemsize == 1 and view.format in ("B", "b"):
        # A raw byte buffer is reinterpreted only against an explicit
        # TensorSpec, so it must be a plain unit-stride byte region; a reversed
        # or strided view would put the exported pointer at the wrong end.
        if tuple(view.strides) != (1,) or view.suboffsets:
            _fail("byte providers must expose unit-stride contiguous bytes")
        return address, None, None, None, read_only, view.nbytes, view
    if view.ndim != 2:
        _fail("provider must expose rank two")
    dtype = _dtype_from_format(view.format, view.itemsize)
    shape = tuple(view.shape)
    if any(size < 1 for size in shape):
        _fail("provider must expose a positive rank-two shape")
    item = _ELEMENT_WIDTH[dtype]
    strides = tuple(stride for stride in view.strides)
    if len(strides) != 2 or strides != (shape[1] * item, item):
        _fail("provider must expose compact row-major storage")
    return address, shape, strides, dtype, read_only, view.nbytes, view


def _layout(provider):
    if hasattr(provider, "__array_interface__"):
        return _array_interface_layout(provider)
    return _buffer_layout(provider)


@dataclass(frozen=True, slots=True)
class BufferLease:
    """A checked descriptor plus the references that keep it addressable.

    ``export`` is a live buffer export (a ``memoryview``) held until the lease is
    dropped, so a resizable provider cannot move its storage during the native
    call. It is ``None`` only for array-interface-only providers, which cannot
    export a buffer protocol view.
    """

    descriptor: LLKBuffer2D
    owner: object
    export: object | None = None


@dataclass(frozen=True, slots=True)
class Invocation:
    inputs: ctypes.Array
    outputs: ctypes.Array
    leases: tuple[BufferLease, ...]

    def release(self):
        """Drop the held buffer exports once the native call has returned."""
        for lease in self.leases:
            if lease.export is not None:
                lease.export.release()


def describe(spec, provider) -> BufferLease:
    """Validate one provider against ``spec`` and build the native descriptor."""
    if not isinstance(spec, TensorSpec):
        _fail("buffer adaptation requires TensorSpec metadata")
    width = _ELEMENT_WIDTH.get(spec.dtype)
    if width is None or spec.dtype not in ELEMENT_TAGS:
        _fail(f"element type {spec.dtype} has no buffer protocol representation")
    address, shape, strides, dtype, read_only, allocation, export = _layout(provider)
    if dtype is not None and dtype != spec.dtype:
        _fail(
            f"provider element type {dtype} does not match declared {spec.dtype}"
        )
    required = spec.shape[0] * spec.shape[1] * width
    if allocation < required:
        _fail(
            f"provider allocation of {allocation} bytes is shorter than the "
            f"{required} bytes the declared extent requires"
        )
    if shape is not None:
        if shape != spec.shape:
            _fail("provider shape does not match the declared tensor extent")
        if strides != (spec.shape[1] * width, width):
            _fail("provider must expose compact row-major strides")
    if spec.effect == "write" and read_only:
        _fail("output provider is read-only")
    if address % width:
        _fail(f"provider pointer is not aligned to its {width}-byte element")
    descriptor = LLKBuffer2D()
    descriptor.allocated = address
    descriptor.aligned = address
    descriptor.offset = 0
    descriptor.sizes[0] = spec.shape[0]
    descriptor.sizes[1] = spec.shape[1]
    descriptor.strides[0] = spec.shape[1]
    descriptor.strides[1] = 1
    descriptor.element_type = ELEMENT_TAGS[spec.dtype]
    descriptor.allocation_bytes = allocation
    return BufferLease(descriptor, provider, export)


def _span(lease):
    begin = int(lease.descriptor.allocated)
    return begin, begin + int(lease.descriptor.allocation_bytes)


def _overlaps(left, right):
    return left[0] < right[1] and right[0] < left[1]


def prepare_invocation(inputs, input_specs, outputs, output_specs) -> Invocation:
    """Build checked descriptors for one synchronous invocation.

    Arity, extent, layout, writability, allocation and disjoint-output checks all
    run here, so a rejected request never reaches native JIT code.
    """
    inputs, input_specs = tuple(inputs), tuple(input_specs)
    outputs, output_specs = tuple(outputs), tuple(output_specs)
    if len(inputs) != len(input_specs) or len(outputs) != len(output_specs):
        _fail(
            f"invocation arity mismatch: {len(inputs)} inputs, {len(outputs)} "
            f"outputs for {len(input_specs)} and {len(output_specs)} ports"
        )
    leases = tuple(
        [describe(spec, provider) for spec, provider in zip(input_specs, inputs)]
        + [describe(spec, provider) for spec, provider in zip(output_specs, outputs)]
    )
    input_leases = leases[: len(inputs)]
    output_leases = leases[len(inputs) :]
    output_spans = [_span(lease) for lease in output_leases]
    for index, span in enumerate(output_spans):
        for other in output_spans[index + 1 :]:
            if _overlaps(span, other):
                _fail("output buffers must be disjoint")
        for input_span in (_span(lease) for lease in input_leases):
            if _overlaps(span, input_span):
                _fail("output buffers must not overlap inputs")
    descriptors = [lease.descriptor for lease in leases]
    return Invocation(
        inputs=(LLKBuffer2D * len(inputs))(*descriptors[: len(inputs)]),
        outputs=(LLKBuffer2D * len(outputs))(*descriptors[len(inputs) :]),
        leases=leases,
    )
