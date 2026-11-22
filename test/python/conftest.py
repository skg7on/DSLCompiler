"""PF20 lane markers, native-call counting and acceptance evidence.

The required lanes are declared in ``test/Frontend/acceptance_manifest.json``.
When a lane run sets ``LLK_PYTHON_EVIDENCE_DIR`` and ``LLK_PYTHON_LANE``, this
module counts every ``llk.runtime._ffi.native_invoke`` call and writes one
evidence JSON file for ``check_required_python_execution.py``. Without those
variables the plugin is inert, so ordinary test runs keep their behavior.
"""

from __future__ import annotations

import pytest

from _acceptance import evidence_dir, lane, record, record_if_missing

MARKERS = {
    "frontend_unit": (
        "compiler-free capture, schema, packaging and error contracts"
    ),
    "frontend_native": (
        "reference-backend native compilation and invocation through the C API"
    ),
    "torch_integration": (
        "real Torch registration, Dynamo, Inductor, operators and training"
    ),
    "triton_integration": (
        "pinned upstream Triton qualification, TTIR export and launch"
    ),
    "selected_avx2": (
        "Linux x86_64/AVX2 selected-target execution; reference is not accepted"
    ),
}

_STATE = {"count": 0, "original": None}


def pytest_configure(config):
    for name, description in MARKERS.items():
        config.addinivalue_line("markers", f"{name}: {description}")


def pytest_sessionstart(session):
    if evidence_dir() is None:
        return
    from llk.runtime import _ffi

    original = _ffi.native_invoke
    if getattr(original, "_llk_acceptance_counted", False):
        return

    def counted(*args, **kwargs):
        _STATE["count"] += 1
        return original(*args, **kwargs)

    counted._llk_acceptance_counted = True
    _ffi.native_invoke = counted
    _STATE["original"] = original


def pytest_sessionfinish(session, exitstatus):
    if evidence_dir() is None:
        return
    record(
        lane=lane(),
        in_process_native_invocations=_STATE["count"],
        exit_status=int(exitstatus),
    )
    # Lanes that execute natives in a subprocess (the Triton corpus worker)
    # report their own count; the in-process counter must not erase it.
    record_if_missing(native_invocations=_STATE["count"])
    if _STATE["original"] is not None:
        from llk.runtime import _ffi

        _ffi.native_invoke = _STATE["original"]
