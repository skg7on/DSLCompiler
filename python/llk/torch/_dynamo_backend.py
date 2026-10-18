"""Optional, idempotent registration of LLK's unavailable native backend."""

from __future__ import annotations

import threading

from .._errors import UnsupportedFeatureError

try:
    import torch
except ModuleNotFoundError as error:
    # Missing dependencies inside an installed Torch indicate a broken install.
    if error.name != "torch":
        raise
    torch = None


class LLKBackend:
    """Dynamo entry point; native compilation is a later frontend milestone."""

    def __call__(self, gm, example_inputs, **kwargs):
        raise UnsupportedFeatureError(
            "LLK native frontend compilation is not available in this package.",
            code="native_backend_unavailable",
        )


# reload retains the module dictionary. Preserve the lock and callable: a
# redefined class is not an identity check for an already registered instance.
if "_registration_lock" not in globals():
    _registration_lock = threading.RLock()
if "_llk_backend" not in globals():
    _llk_backend = LLKBackend()


def get_backend() -> LLKBackend:
    """Return the exact singleton used for public reporting and registration."""
    return _llk_backend


def register() -> None:
    """Register once without resetting Dynamo or replacing another owner."""
    if torch is None:
        return

    from torch._dynamo.backends.registry import lookup_backend, register_backend
    from torch._dynamo.exc import InvalidBackend

    with _registration_lock:
        try:
            registered = lookup_backend("llk")
        except InvalidBackend:
            register_backend(compiler_fn=_llk_backend, name="llk")
            return
        if registered is not _llk_backend:
            raise UnsupportedFeatureError(
                "The Torch backend name 'llk' is already registered by another owner.",
                code="backend_registration_conflict",
            )
