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


def aot_autograd_compiler(fw_compiler, inference_compiler, bw_compiler=None):
    """Return the version-qualified AOTAutograd backend callable.

    ``fw_compiler`` handles the forward of a training graph, ``bw_compiler``
    handles the AOTAutograd-generated backward graph, and
    ``inference_compiler`` handles graphs that need no autograd. When no
    backward compiler is supplied AOTAutograd reuses the forward compiler.
    """
    from torch._dynamo.backends.common import aot_autograd

    options = {
        "fw_compiler": fw_compiler,
        "inference_compiler": inference_compiler,
    }
    if bw_compiler is not None:
        options["bw_compiler"] = bw_compiler
    return aot_autograd(**options)


def in_autograd_backward() -> bool:
    """Whether the autograd engine is currently executing a backward task.

    Custom-op CPU kernels run with grad mode disabled in both directions, so
    grad mode cannot identify the phase. Torch exposes the active graph task
    for exactly this distinction; when the internal probe is unavailable the
    caller falls back to ``forward`` rather than guessing.
    """
    probe = getattr(torch._C, "_current_graph_task_id", None)
    if probe is None:
        return False
    try:
        return probe() >= 0
    except Exception:  # pragma: no cover - defensive around an internal API
        return False


def inductor_compile_fx():
    """Return Torch's qualified Inductor graph compiler.

    ``torch._inductor`` pulls in the whole code generation stack, so it is
    resolved when the backend is invoked rather than at package import. A
    missing Inductor is an explicit unsupported capability, never a silent
    fallback to eager execution.
    """
    try:
        import torch._inductor.compile_fx as module
    except ModuleNotFoundError as error:
        raise UnsupportedFeatureError(
            "Torch Inductor is unavailable in this Torch installation.",
            code="native_backend_unavailable",
            stage="compile",
        ) from error
    return module.compile_fx


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
