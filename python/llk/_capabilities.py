"""Compiler-free reporting of the Python package's current capabilities."""

import json
from importlib import resources
import platform


def capabilities() -> dict:
    """Return a fresh report; registration alone never implies execution."""
    manifest = json.loads(
        resources.files("llk").joinpath("compatibility.json").read_text(encoding="utf-8")
    )
    return {
        "schema_version": 1,
        "runtime_python": platform.python_version(),
        "compatibility": manifest,
        "native_backend": {
            "available": False,
            "reason": "native_backend_unavailable",
        },
    }
