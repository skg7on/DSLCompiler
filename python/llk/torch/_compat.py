"""Torch-version-qualified adapters used by the strict LLK Dynamo backend.

The pinned reference environment is Torch 2.9.0; development also runs on newer
releases. Only stable, public Dynamo/AOTAutograd interfaces are used here, and
the version gate names the first release whose boxed AOT calling contract is
supported instead of guessing across incompatible versions.
"""

from __future__ import annotations

import torch

from .._errors import UnsupportedFeatureError

MINIMUM_TORCH = (2, 9)


def torch_version() -> tuple[int, int]:
    """Return the major/minor release without local build suffixes."""
    release = torch.__version__.split("+", 1)[0]
    parts = release.split(".")
    try:
        return int(parts[0]), int(parts[1])
    except (IndexError, ValueError):
        return (0, 0)


def require_supported_version() -> tuple[int, int]:
    """Reject a Torch release whose AOT boxed contract is unqualified."""
    version = torch_version()
    if version < MINIMUM_TORCH:
        raise UnsupportedFeatureError(
            "the strict LLK Dynamo backend requires Torch 2.9 or newer, "
            f"found {torch.__version__}",
            code="unsupported_torch_version",
            stage="compile",
        )
    return version


def aot_autograd_compiler(fw_compiler, inference_compiler):
    """Return the version-qualified AOTAutograd backend callable.

    Both compilers are the same strict native compiler. Training graphs are
    rejected during import until PF16 registers a backward path, so the
    forward compiler is never asked to produce a gradient.
    """
    from torch._dynamo.backends.common import aot_autograd

    return aot_autograd(
        fw_compiler=fw_compiler,
        inference_compiler=inference_compiler,
    )


def boxed_function(unboxed):
    """Return a callable taking the AOT boxed list of arguments.

    AOTAutograd calls compiler results with one list argument when
    ``_boxed_call`` is set. Older releases expose ``make_boxed_func``; the
    local fallback preserves the same contract without depending on it.
    """
    try:
        from functorch.compile import make_boxed_func
    except ImportError:  # pragma: no cover - retained for narrower installs
        def boxed(args):
            return unboxed(*tuple(args))

        boxed._boxed_call = True
        return boxed
    return make_boxed_func(unboxed)
