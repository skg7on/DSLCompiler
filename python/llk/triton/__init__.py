"""Process-isolated export of genuine upstream Triton TTIR.

This package never imports ``triton`` in the calling process and never links a
Triton LLVM object into the LLK build. Every capture runs in a separate
interpreter that loads the qualified upstream package, generates TTIR with
Triton's own compiler stages, walks the typed module with Triton's own MLIR
bindings, and returns only JSON: the TTIR text, an
:class:`UnboundFrontendTemplate`, and the observed toolchain identity.

Runtime binding of the template to concrete tensors and a launch grid is PF18's
``bind_triton_template``; it is deliberately absent here.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
import threading
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

from llk._errors import (
    CompilationError,
    FrontendSyntaxError,
    UnsupportedFeatureError,
)

from ._template import (
    SCHEMA_VERSION,
    ToolchainIdentity,
    TemplateOperation,
    TemplateParameter,
    TemplateRegion,
    TemplateValue,
    UnboundFrontendTemplate,
)
from ._toolchain import (
    PROTOCOL_VERSION,
    default_lock_path,
    resolve_triton_python,
    toolchain_lock,
)

__all__ = [
    "ExportRequest",
    "TritonExport",
    "ToolchainIdentity",
    "TemplateOperation",
    "TemplateParameter",
    "TemplateRegion",
    "TemplateValue",
    "UnboundFrontendTemplate",
    "capture_kernel",
    "export_module",
    "generate_toolchain_lock",
    "qualify_toolchain",
    "toolchain_lock",
    "PROTOCOL_VERSION",
    "SCHEMA_VERSION",
]

MARKER = "LLK-TRITON-EXPORT-V1"
MAX_STDOUT_BYTES = 8 << 20
MAX_STDERR_BYTES = 1 << 20
_DEFAULT_TIMEOUT = 300.0

_ERROR_TYPES = {
    "upstream_toolchain_unqualified": UnsupportedFeatureError,
    "unsupported_triton_operation": UnsupportedFeatureError,
    "unsupported_triton_layout": UnsupportedFeatureError,
    "unsupported_triton_memory_effect": UnsupportedFeatureError,
    "unsupported_triton_constant": UnsupportedFeatureError,
    "invalid_triton_signature": FrontendSyntaxError,
    "invalid_triton_request": FrontendSyntaxError,
    "triton_compilation_failed": CompilationError,
    "triton_export_failed": CompilationError,
    "triton_export_protocol_error": CompilationError,
    "triton_export_timeout": CompilationError,
}


def _freeze_mapping(value, name):
    if not isinstance(value, Mapping):
        raise FrontendSyntaxError(f"{name} must be a mapping", code="invalid_triton_request")
    frozen = {}
    for key, item in value.items():
        if not isinstance(key, str) or not key:
            raise FrontendSyntaxError(
                f"{name} keys must be nonempty strings", code="invalid_triton_request"
            )
        frozen[key] = item
    return MappingProxyType(frozen)


@dataclass(frozen=True, slots=True)
class ExportRequest:
    """The protocol-v1 request sent to the isolated upstream worker."""

    module_path: str
    function: str
    signature: Mapping
    constexprs: Mapping = field(default_factory=dict)
    specialization: Mapping = field(default_factory=dict)
    options: Mapping = field(default_factory=dict)

    def __post_init__(self):
        if not isinstance(self.module_path, str) or not self.module_path:
            raise FrontendSyntaxError(
                "module_path must be a nonempty string", code="invalid_triton_request"
            )
        if not isinstance(self.function, str) or not self.function:
            raise FrontendSyntaxError(
                "function must be a nonempty string", code="invalid_triton_request"
            )
        signature = _freeze_mapping(self.signature, "signature")
        if not signature:
            raise FrontendSyntaxError(
                "signature must not be empty", code="invalid_triton_signature"
            )
        for name, spec in signature.items():
            if not isinstance(spec, str) or not spec:
                raise FrontendSyntaxError(
                    f"signature for {name} must be a string",
                    code="invalid_triton_signature",
                )
        object.__setattr__(self, "signature", signature)
        object.__setattr__(
            self, "constexprs", _freeze_mapping(self.constexprs, "constexprs")
        )
        object.__setattr__(
            self, "specialization", _freeze_mapping(self.specialization, "specialization")
        )
        object.__setattr__(self, "options", _freeze_mapping(self.options, "options"))

    def to_json(self):
        return {
            "protocol_version": PROTOCOL_VERSION,
            "module_path": self.module_path,
            "function": self.function,
            "signature": dict(self.signature),
            "constexprs": dict(self.constexprs),
            "specialization": {name: dict(value) for name, value in self.specialization.items()},
            "options": dict(self.options),
        }

    @classmethod
    def from_value(cls, value):
        if isinstance(value, cls):
            return value
        if not isinstance(value, Mapping):
            raise FrontendSyntaxError(
                "an export request must be an ExportRequest or a mapping",
                code="invalid_triton_request",
            )
        try:
            return cls(
                module_path=value["module_path"],
                function=value["function"],
                signature=value["signature"],
                constexprs=value.get("constexprs", {}),
                specialization=value.get("specialization", {}),
                options=value.get("options", {}),
            )
        except KeyError as error:
            raise FrontendSyntaxError(
                f"export request is missing {error.args[0]}",
                code="invalid_triton_request",
            ) from error


@dataclass(frozen=True, slots=True)
class TritonExport:
    """Genuine TTIR, its neutral template and the observed toolchain identity."""

    ttir: str
    template: UnboundFrontendTemplate
    toolchain: ToolchainIdentity

    def __post_init__(self):
        if not isinstance(self.ttir, str) or not self.ttir:
            raise FrontendSyntaxError(
                "export requires nonempty TTIR text", code="triton_export_failed"
            )
        if not isinstance(self.template, UnboundFrontendTemplate):
            raise FrontendSyntaxError(
                "export requires an UnboundFrontendTemplate",
                code="triton_export_failed",
            )
        if not isinstance(self.toolchain, ToolchainIdentity):
            raise FrontendSyntaxError(
                "export requires a ToolchainIdentity", code="triton_export_failed"
            )
        if self.template.toolchain != self.toolchain:
            raise FrontendSyntaxError(
                "the template and export toolchain identities disagree",
                code="triton_export_failed",
            )
        digest = hashlib.sha256(self.ttir.encode("utf-8")).hexdigest()
        if digest != self.template.ttir_hash:
            raise FrontendSyntaxError(
                "the template TTIR hash does not match the returned TTIR",
                code="triton_export_failed",
            )

    def to_json(self):
        return {
            "ttir": self.ttir,
            "template": self.template.to_json(),
            "toolchain": self.toolchain.to_json(),
        }


def _worker_path():
    return Path(__file__).resolve().parent / "_export_worker.py"


def _read_bounded(stream, limit):
    chunks = []
    total = 0
    overflow = False
    while True:
        chunk = stream.read(65536)
        if not chunk:
            break
        total += len(chunk)
        if total > limit:
            overflow = True
            break
        chunks.append(chunk)
    return b"".join(chunks), total, overflow


def _run_bounded(argv, timeout):
    process = subprocess.Popen(
        argv,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=dict(os.environ),
    )
    captured = {}

    def drain(name, stream, limit):
        captured[name] = _read_bounded(stream, limit)

    threads = [
        threading.Thread(
            target=drain, args=("stdout", process.stdout, MAX_STDOUT_BYTES), daemon=True
        ),
        threading.Thread(
            target=drain, args=("stderr", process.stderr, MAX_STDERR_BYTES), daemon=True
        ),
    ]
    for thread in threads:
        thread.start()
    timed_out = False
    try:
        process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
        process.kill()
        process.wait()
    for thread in threads:
        thread.join()
    stdout, stdout_size, stdout_overflow = captured.get("stdout", (b"", 0, False))
    stderr, stderr_size, stderr_overflow = captured.get("stderr", (b"", 0, False))
    return {
        "returncode": process.returncode,
        "stdout": stdout,
        "stderr": stderr,
        "stdout_size": stdout_size,
        "stderr_size": stderr_size,
        "stdout_overflow": stdout_overflow,
        "stderr_overflow": stderr_overflow,
        "timed_out": timed_out,
    }


def _stderr_tail(stderr):
    text = stderr.decode("utf-8", "replace").strip()
    if len(text) > 2000:
        text = "..." + text[-2000:]
    return text


def _raise_worker_error(error, stderr):
    if not isinstance(error, Mapping):
        raise CompilationError(
            "the Triton export worker returned a malformed error",
            code="triton_export_protocol_error",
            stage="export",
        )
    code = error.get("code")
    message = error.get("message")
    if not isinstance(code, str) or not isinstance(message, str):
        raise CompilationError(
            "the Triton export worker returned a malformed error",
            code="triton_export_protocol_error",
            stage="export",
        )
    tail = _stderr_tail(stderr)
    if tail:
        message = f"{message}\n{tail}"
    error_type = _ERROR_TYPES.get(code, CompilationError)
    raise error_type(
        message,
        code=code,
        stage=error.get("stage") if isinstance(error.get("stage"), str) else "export",
    )


def _run_worker(extra_args, *, triton_python, lock_path, timeout):
    argv = [sys.executable, "-I", str(_worker_path()), *extra_args]
    if triton_python:
        argv.extend(["--triton-python", str(triton_python)])
    if lock_path is not None:
        argv.extend(["--lock", str(lock_path)])
    outcome = _run_bounded(argv, timeout)
    if outcome["timed_out"]:
        raise CompilationError(
            "the Triton export worker exceeded its time limit",
            code="triton_export_timeout",
            stage="export",
        )
    if outcome["stdout_overflow"]:
        raise CompilationError(
            "the Triton export worker produced oversized output",
            code="triton_export_protocol_error",
            stage="export",
        )
    stdout = outcome["stdout"].decode("utf-8", "replace")
    stderr = outcome["stderr"]
    lines = stdout.split("\n", 1)
    if len(lines) != 2 or lines[0] != MARKER:
        raise CompilationError(
            "the Triton export worker did not produce a framed result"
            + (f": {_stderr_tail(stderr)}" if stderr else ""),
            code="triton_export_protocol_error",
            stage="export",
        )
    try:
        frame = json.loads(lines[1])
    except json.JSONDecodeError as error:
        raise CompilationError(
            f"the Triton export worker returned malformed JSON: {error}",
            code="triton_export_protocol_error",
            stage="export",
        ) from error
    if not isinstance(frame, dict) or frame.get("protocol_version") != PROTOCOL_VERSION:
        raise CompilationError(
            "the Triton export worker returned an unsupported protocol version",
            code="triton_export_protocol_error",
            stage="export",
        )
    if frame.get("ok") is not True:
        _raise_worker_error(frame.get("error"), stderr)
    result = frame.get("result")
    if not isinstance(result, dict):
        raise CompilationError(
            "the Triton export worker returned no result object",
            code="triton_export_protocol_error",
            stage="export",
        )
    return result, stderr


def _lock_path(lock_path):
    return default_lock_path() if lock_path is None else Path(lock_path)


def qualify_toolchain(*, triton_python=None, lock_path=None, timeout=_DEFAULT_TIMEOUT):
    """Qualify the installed upstream package against the checked-in lock."""
    result, _ = _run_worker(
        ["--check"],
        triton_python=resolve_triton_python() if triton_python is None else triton_python,
        lock_path=_lock_path(lock_path),
        timeout=timeout,
    )
    return ToolchainIdentity.from_json(result["toolchain"])


def generate_toolchain_lock(
    path,
    *,
    triton_python=None,
    target="cuda:80",
    qualified_kernels=("add", "multiply"),
    timeout=_DEFAULT_TIMEOUT,
):
    """Regenerate the lock from the actual binaries, then qualify against it.

    This is how the checked-in manifest is produced. The generated file is
    written before qualification, so an edited or stale lock cannot hide a
    mismatch between the recorded and observed toolchains.
    """
    result, _ = _run_worker(
        ["--identity", "--target", target],
        triton_python=resolve_triton_python() if triton_python is None else triton_python,
        lock_path=None,
        timeout=timeout,
    )
    identity = ToolchainIdentity.from_json(result["toolchain"])
    lock = {
        "schema_version": 1,
        "protocol_version": identity.protocol_version,
        "triton_version": identity.triton_version,
        "source_revision": identity.source_revision,
        "llvm_revision": identity.llvm_revision,
        "target": identity.target,
        "template_schema": SCHEMA_VERSION,
        "build": {
            "distribution": "source-checkout"
            if identity.source_revision != "unknown"
            else "installed-package",
            "python_tag": identity.python_tag,
            "platform": identity.platform_tag,
            "notes": "Generated by llk.triton.generate_toolchain_lock from the "
            "actual upstream binaries. TTIR export only; no GPU execution.",
        },
        "qualified_kernels": list(qualified_kernels),
        "notes": "The worker verifies Triton version, LLVM revision, source "
        "revision and target exactly before export. A missing, different or "
        "unreadable toolchain raises upstream_toolchain_unqualified; no "
        "simulated backend or copied TTIR is accepted.",
    }
    destination = Path(path)
    destination.write_text(
        json.dumps(lock, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    regenerate_identity = qualify_toolchain(
        triton_python=triton_python, lock_path=destination, timeout=timeout
    )
    if regenerate_identity != identity:
        raise CompilationError(
            "the generated lock does not reproduce the observed toolchain",
            code="upstream_toolchain_unqualified",
            stage="qualification",
        )
    return lock


def export_module(request, *, triton_python=None, lock_path=None, timeout=_DEFAULT_TIMEOUT):
    """Export genuine upstream TTIR and its neutral unbound template.

    The request is written to a private temporary file and passed by argv. The
    worker's stdout is a single framed JSON document; its stderr diagnostics are
    captured separately and never parsed as result data.
    """
    request = ExportRequest.from_value(request)
    with tempfile.TemporaryDirectory(prefix="llk-triton-export-") as directory:
        request_path = Path(directory) / "request.json"
        # Signature order is kernel argument order; never sort these keys.
        request_path.write_text(
            json.dumps(request.to_json(), sort_keys=False), encoding="utf-8"
        )
        result, _ = _run_worker(
            ["--request", str(request_path)],
            triton_python=resolve_triton_python()
            if triton_python is None
            else triton_python,
            lock_path=_lock_path(lock_path),
            timeout=timeout,
        )
    ttir = result.get("ttir")
    if not isinstance(ttir, str):
        raise CompilationError(
            "the Triton export worker returned no TTIR text",
            code="triton_export_protocol_error",
            stage="export",
        )
    try:
        template = UnboundFrontendTemplate.from_json(result.get("template"))
        toolchain = ToolchainIdentity.from_json(result.get("toolchain"))
        return TritonExport(ttir=ttir, template=template, toolchain=toolchain)
    except FrontendSyntaxError as error:
        raise CompilationError(
            f"the Triton export worker returned an invalid result: {error.message}",
            code="triton_export_protocol_error",
            stage="export",
        ) from error


def capture_kernel(
    module_path,
    function,
    signature,
    *,
    constexprs=None,
    specialization=None,
    options=None,
    triton_python=None,
    lock_path=None,
    timeout=_DEFAULT_TIMEOUT,
):
    """Convenience wrapper around :func:`export_module`."""
    return export_module(
        ExportRequest(
            module_path=str(module_path),
            function=function,
            signature=signature,
            constexprs=constexprs or {},
            specialization=specialization or {},
            options=options or {},
        ),
        triton_python=triton_python,
        lock_path=lock_path,
        timeout=timeout,
    )
