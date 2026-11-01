"""ctypes bridge to the versioned ``LLKFrontendCAPI`` library.

The bridge owns the native handle lifetime and translates owned native
diagnostics into the Python exception hierarchy. Caller buffers are adapted and
checked by :mod:`llk.runtime._buffers` immediately before every native call; no
pointer is retained after the call returns.
"""

from __future__ import annotations

import ctypes
import json
import os
import threading
from importlib.resources import files
from pathlib import Path

from llk._errors import (
    CompilationError,
    ExecutionError,
    FrontendSyntaxError,
    LLKError,
    TensorContractError,
    UnsupportedFeatureError,
)
from llk.frontend import CompilationReport, CompileRequest
from llk.frontend._serialize import serialize_request

from . import _buffers

API_VERSION = 1
_LIBRARY = None
_LIBRARY_PATH = None
_LOAD_LOCK = threading.Lock()


class LLKString(ctypes.Structure):
    _fields_ = [("data", ctypes.c_void_p), ("length", ctypes.c_size_t)]


def _library_file():
    explicit = os.environ.get("LLK_FRONTEND_LIBRARY")
    if explicit:
        path = Path(explicit)
        if not path.is_file():
            raise UnsupportedFeatureError(
                f"LLK_FRONTEND_LIBRARY={path} does not name a file",
                code="native_backend_unavailable",
            )
        return path
    root = Path(str(files("llk")))
    for name in (
        "libLLKFrontendCAPI.dylib",
        "libLLKFrontendCAPI.so",
        "LLKFrontendCAPI.dll",
    ):
        for candidate in (root / name, root / "lib" / name):
            if candidate.is_file():
                return candidate
    return None


def _configure(library):
    library.llk_frontend_api_version.restype = ctypes.c_uint32
    library.llk_frontend_api_version.argtypes = []
    library.llk_frontend_compile_v1.argtypes = [
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(ctypes.c_void_p),
    ]
    library.llk_frontend_compile_v1.restype = ctypes.c_int
    library.llk_frontend_invoke_v1.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(_buffers.LLKBuffer2D),
        ctypes.c_size_t,
        ctypes.POINTER(_buffers.LLKBuffer2D),
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_void_p),
    ]
    library.llk_frontend_invoke_v1.restype = ctypes.c_int
    library.llk_frontend_report_v1.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(LLKString),
        ctypes.POINTER(ctypes.c_void_p),
    ]
    library.llk_frontend_report_v1.restype = ctypes.c_int
    library.llk_frontend_inspect_v1.argtypes = [
        ctypes.c_void_p,
        ctypes.c_char_p,
        ctypes.POINTER(LLKString),
        ctypes.POINTER(ctypes.c_void_p),
    ]
    library.llk_frontend_inspect_v1.restype = ctypes.c_int
    library.llk_frontend_error_json_v1.argtypes = [ctypes.c_void_p]
    library.llk_frontend_error_json_v1.restype = LLKString
    library.llk_frontend_error_free_v1.argtypes = [ctypes.c_void_p]
    library.llk_frontend_error_free_v1.restype = None
    library.llk_frontend_release_v1.argtypes = [ctypes.c_void_p]
    library.llk_frontend_release_v1.restype = None
    return library


def library():
    """Load and qualify the shared library once per process."""
    global _LIBRARY, _LIBRARY_PATH
    with _LOAD_LOCK:
        if _LIBRARY is not None:
            return _LIBRARY
        path = _library_file()
        if path is None:
            raise UnsupportedFeatureError(
                "LLKFrontendCAPI is unavailable; set LLK_FRONTEND_LIBRARY to a "
                "built libLLKFrontendCAPI shared library",
                code="native_backend_unavailable",
            )
        try:
            loaded = _configure(ctypes.CDLL(str(path)))
        except (OSError, AttributeError) as error:
            raise UnsupportedFeatureError(
                f"cannot load LLKFrontendCAPI from {path}: {error}",
                code="native_backend_unavailable",
            ) from error
        version = loaded.llk_frontend_api_version()
        if version != API_VERSION:
            raise UnsupportedFeatureError(
                f"LLKFrontendCAPI reports API version {version}, expected "
                f"{API_VERSION}",
                code="frontend_api_version_mismatch",
            )
        _LIBRARY, _LIBRARY_PATH = loaded, str(path)
        return loaded


def library_path():
    library()
    return _LIBRARY_PATH


def _string_value(value):
    if not value.data or not value.length:
        return ""
    return ctypes.string_at(value.data, value.length).decode("utf-8", "replace")


def _exception_for(code, stage):
    if code in (
        "native_backend_unavailable",
        "frontend_api_version_mismatch",
        "inspection_unavailable",
        "upstream_toolchain_unqualified",
        "source_unavailable",
    ) or code.startswith("unsupported_"):
        return UnsupportedFeatureError
    if code in (
        "invalid_frontend_buffer",
        "invalid_frontend_argument",
        "invalid_launch_coverage",
    ):
        return TensorContractError
    if code in (
        "invalid_frontend_schema",
        "invalid_argument_binding",
        "unsupported_annotation",
    ):
        return FrontendSyntaxError
    if code == "invalid_frontend_handle" or stage == "invoke":
        return ExecutionError
    return CompilationError


def raise_native(error, *, stage):
    """Copy and free one owned native diagnostic, then raise it."""
    text = ""
    if error:
        text = _string_value(library().llk_frontend_error_json_v1(error))
        library().llk_frontend_error_free_v1(error)
    data = {}
    if text:
        try:
            data = json.loads(text)
        except ValueError:
            data = {}
    code = str(data.get("code", "frontend_native_error"))
    message = str(data.get("message", text or "native frontend operation failed"))
    raise _exception_for(code, stage)(
        message,
        code=code,
        file=data.get("file"),
        line=data.get("line"),
        column=data.get("column"),
        stage=data.get("stage", stage),
    )


def native_compile(payload):
    """Compile canonical request bytes and return an owned native handle."""
    out = ctypes.c_void_p()
    error = ctypes.c_void_p()
    text = ctypes.create_string_buffer(payload, len(payload))
    status = library().llk_frontend_compile_v1(
        ctypes.cast(text, ctypes.c_void_p),
        len(payload),
        ctypes.byref(out),
        ctypes.byref(error),
    )
    if status != 0:
        raise_native(error.value, stage="compile")
    return out.value


def native_invoke(handle, inputs, outputs):
    """Invoke one handle with checked descriptor arrays."""
    error = ctypes.c_void_p()
    status = library().llk_frontend_invoke_v1(
        handle,
        inputs,
        len(inputs),
        outputs,
        len(outputs),
        ctypes.byref(error),
    )
    if status != 0:
        raise_native(error.value, stage="invoke")


def native_report(handle):
    out = LLKString()
    error = ctypes.c_void_p()
    status = library().llk_frontend_report_v1(
        handle, ctypes.byref(out), ctypes.byref(error)
    )
    if status != 0:
        raise_native(error.value, stage="compile")
    return _string_value(out)


def native_inspect(handle, stage):
    out = LLKString()
    error = ctypes.c_void_p()
    status = library().llk_frontend_inspect_v1(
        handle, stage.encode("utf-8"), ctypes.byref(out), ctypes.byref(error)
    )
    if status != 0:
        raise_native(error.value, stage="inspect")
    return _string_value(out)


def native_release(handle):
    if handle:
        library().llk_frontend_release_v1(handle)


def _parse_report(text):
    try:
        data = json.loads(text)
        return CompilationReport(
            native_request_identity=str(data["native_request_identity"]),
            frontend_identity=str(data["frontend_identity"]),
            compiler_identity=str(data["compiler_identity"]),
            workload_identity=str(data["workload_identity"]),
            source_adapter_identity=str(data["source_adapter_identity"]),
            target_hash=str(data["target_hash"]),
            machine_hash=str(data["machine_hash"]),
            rules_hash=str(data["rules_hash"]),
            plan_id=str(data["plan_id"]),
            abi_hash=str(data["abi_hash"]),
            selected_group_count=data["selected_group_count"],
            reference_group_count=data["reference_group_count"],
            stages=tuple(data["stages"]),
            backend=str(data["backend"]),
            source_adapter=str(data["source_adapter"]),
        )
    except (TypeError, ValueError, KeyError) as error:
        raise CompilationError(
            f"native frontend report is malformed: {error}",
            code="frontend_compilation_failed",
            stage="compile",
        ) from error


class CompiledKernel:
    """An owned native executable and the port metadata it was compiled for.

    ``invoke`` holds the instance lock for the whole ctypes call, so ``close``
    cannot release the handle underneath an in-flight invocation and concurrent
    calls are serialized. Closing twice is idempotent.
    """

    def __init__(self, handle, report, input_specs, output_specs):
        self._handle = handle
        self._lock = threading.RLock()
        self._closed = False
        self._report = report
        self._input_specs = tuple(input_specs)
        self._output_specs = tuple(output_specs)

    @property
    def report(self):
        return self._report

    @property
    def closed(self):
        return self._closed

    def invoke(self, inputs, outputs):
        inputs, outputs = tuple(inputs), tuple(outputs)
        with self._lock:
            if self._closed:
                raise ExecutionError(
                    "compiled kernel is closed", code="invalid_frontend_handle"
                )
            invocation = _buffers.prepare_invocation(
                inputs, self._input_specs, outputs, self._output_specs
            )
            try:
                native_invoke(self._handle, invocation.inputs, invocation.outputs)
            finally:
                invocation.release()

    def inspect(self, stage):
        with self._lock:
            if self._closed:
                raise ExecutionError(
                    "compiled kernel is closed", code="invalid_frontend_handle"
                )
            return native_inspect(self._handle, stage)

    def close(self):
        with self._lock:
            if self._closed:
                return
            self._closed = True
            handle = self._handle
            self._handle = None
            native_release(handle)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        return False

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def __repr__(self):
        state = "closed" if self._closed else "open"
        return f"<CompiledKernel {state} backend={self._report.backend}>"


def compile_request(request):
    """Compile one validated ``CompileRequest`` into an owned kernel."""
    if not isinstance(request, CompileRequest):
        raise LLKError(
            "compile_request requires a CompileRequest",
            code="invalid_frontend_schema",
            stage="compile",
        )
    payload = serialize_request(request)
    handle = native_compile(payload)
    try:
        report = _parse_report(native_report(handle))
    except BaseException:
        native_release(handle)
        raise
    inputs = tuple(
        tensor for tensor in request.program.tensors if tensor.direction == "input"
    )
    outputs = tuple(
        tensor for tensor in request.program.tensors if tensor.direction == "output"
    )
    return CompiledKernel(handle, report, inputs, outputs)
