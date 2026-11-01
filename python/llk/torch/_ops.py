"""Checked functional LLK operators and truthful FakeTensor kernels.

``llk::add``, ``llk::mul``, ``llk::matmul`` and ``llk::swiglu`` are functional
PyTorch custom operators. Their CPU kernels validate caller tensors with the
same adapter the decorator and Dynamo routes use, build one shared native
request and invoke the shared specialization cache. Their fake kernels build a
fresh output from shape/dtype/device metadata only: they never request a data
pointer, compile a request or enter the native runtime.

Registration is idempotent and owns the ``llk`` operator namespace: an
existing operator with the same schema is adopted, while a foreign or partial
registration is reported instead of silently replaced.
"""

from __future__ import annotations

import threading

from llk._errors import TensorContractError, UnsupportedFeatureError
from llk.runtime import _ffi
from llk.runtime._cache import CompilationCache

try:
    import torch
except ModuleNotFoundError as error:
    # Missing dependencies inside an installed Torch indicate a broken install.
    if error.name != "torch":
        raise
    torch = None

OP_SCHEMAS = {
    "add": "(Tensor a, Tensor b) -> Tensor",
    "mul": "(Tensor a, Tensor b) -> Tensor",
    "matmul": "(Tensor a, Tensor b) -> Tensor",
    "swiglu": "(Tensor x, Tensor wg, Tensor wu) -> Tensor",
}

_OUTPUTS = {
    "add": lambda a, b: a,
    "mul": lambda a, b: a,
    "matmul": lambda a, b: (a[0], b[1]),
    "swiglu": lambda x, wg, wu: (x[0], wg[1]),
}

_OPERAND_NAMES = {
    "add": ("a", "b"),
    "mul": ("a", "b"),
    "matmul": ("a", "b"),
    "swiglu": ("x", "wg", "wu"),
}

if "_registration_lock" not in globals():
    _registration_lock = threading.RLock()
if "_llk_ops_cache" not in globals():
    _llk_ops_cache = CompilationCache()
if "_llk_ops_options" not in globals():
    _llk_ops_options = None


def cache_info():
    """Return the shared operator cache counters."""
    return _llk_ops_cache.info()


def _options():
    global _llk_ops_options
    if _llk_ops_options is None:
        from ._fx_importer import compile_options

        _llk_ops_options = compile_options()
    return _llk_ops_options


def _execute(kind, tensors):
    """Validate, compile and invoke one operator over caller-owned tensors."""
    from ._fx_importer import TorchBuffer, op_request

    # TorchBuffer is the same checked adapter the decorator and Dynamo routes
    # use. It rejects unsupported devices, dtypes, layouts, subclasses and
    # training inputs before anything is compiled or invoked.
    providers = tuple(TorchBuffer(tensor) for tensor in tensors)
    shapes = tuple(tuple(int(size) for size in tensor.shape) for tensor in tensors)
    request = op_request(kind, shapes, _options())
    kernel = _llk_ops_cache.get_or_compile(request, _ffi.compile_request)
    try:
        output = torch.empty(
            _OUTPUTS[kind](*shapes), dtype=torch.float32, device="cpu"
        )
        kernel.invoke(providers, (TorchBuffer(output),))
    finally:
        kernel.close()
    return output


def _fake_shape(tensor, name):
    """Validate metadata only; never read storage, pointers or native state."""
    from torch._subclasses.fake_tensor import FakeTensor

    if type(tensor) is not torch.Tensor and not isinstance(tensor, FakeTensor):
        raise TensorContractError(
            f"llk operator operand {name} must be a plain CPU torch.Tensor",
            code="invalid_frontend_buffer",
        )
    if tensor.device.type != "cpu" or tensor.layout is not torch.strided:
        raise TensorContractError(
            f"llk operator operand {name} must be a CPU strided tensor",
            code="invalid_frontend_buffer",
        )
    if tensor.dtype is not torch.float32:
        raise TensorContractError(
            f"llk operator operand {name} must be float32",
            code="invalid_frontend_buffer",
        )
    if tensor.dim() != 2 or any(int(size) < 1 for size in tensor.shape):
        raise TensorContractError(
            f"llk operator operand {name} must be a positive rank-two tensor",
            code="invalid_frontend_buffer",
        )
    if not tensor.is_contiguous() or int(tensor.storage_offset()) != 0:
        raise TensorContractError(
            f"llk operator operand {name} must be compact row-major with zero "
            "storage offset",
            code="invalid_frontend_buffer",
        )
    if tensor.requires_grad:
        raise TensorContractError(
            "LLK native invocation is inference only; requires_grad operands "
            "need the registered training path (PF16)",
            code="unsupported_training_contract",
        )
    return tuple(int(size) for size in tensor.shape)


def _fake_metadata(kind, tensors):
    from ._fx_importer import op_output_shape

    names = _OPERAND_NAMES[kind]
    if len(tensors) != len(names):
        raise TensorContractError(
            f"llk::{kind} requires {len(names)} tensor operands",
            code="invalid_frontend_argument",
        )
    shapes = tuple(
        _fake_shape(tensor, name) for name, tensor in zip(names, tensors)
    )
    return op_output_shape(kind, shapes)


def _fake(kind, tensors):
    return torch.empty(
        _fake_metadata(kind, tensors), dtype=torch.float32, device="cpu"
    )


# --- CPU kernels ----------------------------------------------------------


def _add(a, b):
    return _execute("add", (a, b))


def _mul(a, b):
    return _execute("mul", (a, b))


def _matmul(a, b):
    return _execute("matmul", (a, b))


def _swiglu(x, wg, wu):
    return _execute("swiglu", (x, wg, wu))


_IMPLEMENTATIONS = {
    "add": _add,
    "mul": _mul,
    "matmul": _matmul,
    "swiglu": _swiglu,
}

# --- fake kernels ---------------------------------------------------------


def _fake_add(a, b):
    return _fake("add", (a, b))


def _fake_mul(a, b):
    return _fake("mul", (a, b))


def _fake_matmul(a, b):
    return _fake("matmul", (a, b))


def _fake_swiglu(x, wg, wu):
    return _fake("swiglu", (x, wg, wu))


_FAKE_IMPLEMENTATIONS = {
    "add": _fake_add,
    "mul": _fake_mul,
    "matmul": _fake_matmul,
    "swiglu": _fake_swiglu,
}


def _existing_operators():
    namespace = getattr(torch.ops, "llk", None)
    if namespace is None:
        return {}
    return {
        name: overload
        for name in OP_SCHEMAS
        if (overload := getattr(namespace, name, None)) is not None
    }


def register_operators() -> None:
    """Register the functional LLK operators once without replacing an owner."""
    if torch is None:
        return

    from . import _compat

    _compat.require_supported_version()

    with _registration_lock:
        existing = _existing_operators()
        if existing:
            if len(existing) != len(OP_SCHEMAS):
                raise UnsupportedFeatureError(
                    "The 'llk' operator namespace is partially registered by "
                    "another owner; refusing to mix implementations.",
                    code="operator_registration_conflict",
                )
            for name, overload in existing.items():
                expected = f"llk::{name}{OP_SCHEMAS[name]}"
                if str(overload.default._schema) != expected:
                    raise UnsupportedFeatureError(
                        f"The Torch operator 'llk::{name}' is registered by "
                        "another owner with a different schema.",
                        code="operator_registration_conflict",
                    )
            return

        definitions = {
            name: torch.library.custom_op(
                f"llk::{name}",
                _IMPLEMENTATIONS[name],
                mutates_args=(),
                device_types="cpu",
                schema=schema,
            )
            for name, schema in OP_SCHEMAS.items()
        }
        for name, definition in definitions.items():
            definition.register_fake(_FAKE_IMPLEMENTATIONS[name])
