"""Optional Torch integration: strict and Inductor-backed compilation.

Import registers ``backend="llk"`` and ``backend="llk_inductor"`` idempotently
when Torch is installed. ``llk`` compiles supported normalized ATen graphs
through the shared native frontend; ``llk_inductor`` replaces scoped native
regions with the registered ``llk::*`` operators and delegates the rest to
Torch Inductor. Without Torch, import succeeds and registration is a no-op.
"""

from ._dynamo_backend import LLKBackend, get_backend, register
from ._inductor_backend import (
    PartitionReport,
    inductor_backend,
    register as register_inductor_backend,
    rewrite_supported_subgraphs,
)
from ._ops import register_operators

register()
register_operators()
register_inductor_backend()

__all__ = [
    "LLKBackend",
    "PartitionReport",
    "get_backend",
    "inductor_backend",
    "register",
    "register_inductor_backend",
    "register_operators",
    "rewrite_supported_subgraphs",
]

_LAZY_EXPORTS = (
    "UnsupportedFXError",
    "import_fx",
    "import_fx_outputs",
    "import_fx_requests",
)


def __getattr__(name):
    """Expose the Torch-only importer without importing Torch at package import."""
    if name in _LAZY_EXPORTS:
        from . import _fx_importer

        return getattr(_fx_importer, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
