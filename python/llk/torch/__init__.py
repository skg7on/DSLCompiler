"""Optional Torch integration: strict Dynamo backend over the native service.

Import registers ``backend="llk"`` idempotently when Torch is installed. The
backend compiles supported normalized ATen graphs through the shared native
frontend; unsupported graphs raise a structured diagnostic naming the FX node.
Without Torch, import succeeds and registration is a no-op.
"""

from ._dynamo_backend import LLKBackend, get_backend, register

register()

__all__ = ["LLKBackend", "get_backend", "register"]

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
