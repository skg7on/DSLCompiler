"""Fixtures for the pinned upstream Triton compatibility lane."""

import os
from pathlib import Path

import pytest


def triton_python_path():
    """Return the directory that contains the qualified ``triton`` package."""
    explicit = os.environ.get("LLK_TRITON_PYTHONPATH")
    if explicit:
        return Path(explicit).resolve()
    try:
        import triton
    except Exception:
        return None
    return Path(triton.__file__).resolve().parent.parent


@pytest.fixture(scope="session")
def triton_path():
    path = triton_python_path()
    if path is None:
        pytest.skip(
            "no qualified upstream Triton 3.6.0; set LLK_TRITON_PYTHONPATH "
            "to the directory containing the triton package"
        )
    return path
