"""Per-lane evidence recorded for the PF20 acceptance checker.

A lane run sets ``LLK_PYTHON_LANE`` and ``LLK_PYTHON_EVIDENCE_DIR``. The
session hooks in ``conftest.py`` merge the native-call counter, and tests that
exercise selected-target execution add the executed backend and group counts.
``check_required_python_execution.py`` consumes the resulting
``<evidence-dir>/<lane>.json`` files.
"""

from __future__ import annotations

import json
import os
from pathlib import Path


def evidence_dir() -> Path | None:
    value = os.environ.get("LLK_PYTHON_EVIDENCE_DIR")
    return Path(value) if value else None


def lane() -> str:
    return os.environ.get("LLK_PYTHON_LANE", "")


def _path() -> Path | None:
    root = evidence_dir()
    if root is None:
        return None
    root.mkdir(parents=True, exist_ok=True)
    return root / f"{lane() or 'unlabelled'}.json"


def _load(path: Path) -> dict:
    if path.is_file():
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            loaded = {}
        if isinstance(loaded, dict):
            return loaded
    return {}


def record(**fields) -> None:
    """Merge fields into this lane's evidence file; no-op without a lane dir."""
    path = _path()
    if path is None:
        return
    data = _load(path)
    data.update(fields)
    path.write_text(
        json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def record_if_missing(**fields) -> None:
    """Record only fields no test has already reported (subprocess evidence)."""
    path = _path()
    if path is None:
        return
    data = _load(path)
    for name, value in fields.items():
        data.setdefault(name, value)
    path.write_text(
        json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
