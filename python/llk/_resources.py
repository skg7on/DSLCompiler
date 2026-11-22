"""Installed machine/rule/layout resources and their content hashes.

The wheel carries the same ``machines/*.yaml`` and ``mapping/*/*.llkmap`` files
the repository ships, under ``llk/config``. ``resources.json`` records each
file's SHA-256 so an incomplete or locally modified installation fails before a
compile request is built, instead of silently compiling against different
target policy.
"""

from __future__ import annotations

import hashlib
import json
from importlib import resources
from pathlib import Path

from llk._errors import UnsupportedFeatureError

MANIFEST_NAME = "resources.json"
CONFIG_DIRECTORY = "config"
_MISSING = "native_configuration_missing"
_CORRUPT = "native_configuration_corrupt"


def _package_root() -> Path:
    return Path(str(resources.files("llk")))


CONFIG_ROOT = _package_root() / CONFIG_DIRECTORY


def packaged_config_root() -> str:
    """Return the configuration directory shipped inside the package."""
    return str(CONFIG_ROOT)


def manifest() -> dict:
    """Load and shape-check the packaged resource manifest."""
    path = _package_root() / MANIFEST_NAME
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise UnsupportedFeatureError(
            f"cannot read the packaged resource manifest at {path}: {error}",
            code=_MISSING,
        ) from error
    if (
        not isinstance(data, dict)
        or data.get("schema_version") != 1
        or data.get("root") != CONFIG_DIRECTORY
        or not isinstance(data.get("files"), dict)
        or not data["files"]
        or any(
            not isinstance(name, str)
            or not name
            or Path(name).is_absolute()
            or ".." in Path(name).parts
            or not isinstance(digest, str)
            or len(digest) != 64
            for name, digest in data["files"].items()
        )
    ):
        raise UnsupportedFeatureError(
            f"the packaged resource manifest at {path} is malformed", code=_CORRUPT
        )
    return data


def validate_configuration(root: str | Path | None = None) -> dict:
    """Verify every declared resource exists and matches its recorded hash."""
    root = Path(packaged_config_root() if root is None else root)
    data = manifest()
    for relative, digest in data["files"].items():
        path = root / relative
        if not path.is_file():
            raise UnsupportedFeatureError(
                f"missing packaged configuration file {relative} under {root}",
                code=_MISSING,
            )
        try:
            actual = hashlib.sha256(path.read_bytes()).hexdigest()
        except (OSError, UnicodeError) as error:
            raise UnsupportedFeatureError(
                f"cannot read packaged configuration file {relative} under "
                f"{root}: {error}",
                code=_MISSING,
            ) from error
        if actual != digest:
            raise UnsupportedFeatureError(
                f"packaged configuration file {relative} under {root} does not "
                f"match its recorded content hash",
                code=_CORRUPT,
            )
    return data
