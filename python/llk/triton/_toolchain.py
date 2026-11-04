"""Toolchain lock discovery and parsing for the isolated TTIR exporter."""

from __future__ import annotations

import importlib.util
import json
import os
from importlib.resources import files
from pathlib import Path
from types import MappingProxyType

from llk._errors import UnsupportedFeatureError

PROTOCOL_VERSION = 1
TRITON_PYTHON_ENV = "LLK_TRITON_PYTHONPATH"
TOOLCHAIN_LOCK_ENV = "LLK_TRITON_TOOLCHAIN_LOCK"


def _freeze(value):
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return value


def default_lock_path() -> Path:
    """Return the qualified lock, honouring an explicit test/embedded override."""
    explicit = os.environ.get(TOOLCHAIN_LOCK_ENV)
    if explicit:
        path = Path(explicit)
        if not path.is_file():
            raise UnsupportedFeatureError(
                f"{TOOLCHAIN_LOCK_ENV}={path} does not name a file",
                code="upstream_toolchain_unqualified",
            )
        return path
    resource = files("llk.triton").joinpath("toolchain-lock.json")
    return Path(str(resource))


def toolchain_lock(path: Path | None = None) -> MappingProxyType:
    """Read and lightly validate the checked-in qualification manifest."""
    lock_path = default_lock_path() if path is None else Path(path)
    try:
        data = json.loads(lock_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise UnsupportedFeatureError(
            f"cannot read the Triton toolchain lock at {lock_path}: {error}",
            code="upstream_toolchain_unqualified",
        ) from error
    if not isinstance(data, dict):
        raise UnsupportedFeatureError(
            "the Triton toolchain lock must be a JSON object",
            code="upstream_toolchain_unqualified",
        )
    if data.get("schema_version") != 1:
        raise UnsupportedFeatureError(
            "only Triton toolchain lock schema version 1 is supported",
            code="upstream_toolchain_unqualified",
        )
    if data.get("protocol_version") != PROTOCOL_VERSION:
        raise UnsupportedFeatureError(
            "the Triton toolchain lock names an unsupported export protocol",
            code="upstream_toolchain_unqualified",
        )
    for key in ("triton_version", "llvm_revision", "target", "source_revision"):
        if not isinstance(data.get(key), str) or not data[key]:
            raise UnsupportedFeatureError(
                f"the Triton toolchain lock is missing {key}",
                code="upstream_toolchain_unqualified",
            )
    return _freeze(data)


def resolve_triton_python() -> str | None:
    """Locate the directory that contains the upstream ``triton`` package.

    An explicit ``LLK_TRITON_PYTHONPATH`` wins, which is how a source checkout
    is qualified without installing it. Otherwise the current interpreter's
    import path is used. The worker receives this as an explicit argument, so
    it never depends on ambient ``PYTHONPATH`` state.
    """
    explicit = os.environ.get(TRITON_PYTHON_ENV)
    if explicit:
        return str(Path(explicit).resolve())
    spec = importlib.util.find_spec("triton")
    if spec is None or not spec.origin:
        return None
    return str(Path(spec.origin).resolve().parent.parent)
