"""Optional Dynamo registration; native LLK compilation remains unavailable.

Import registers ``backend="llk"`` idempotently when Torch is installed.
Calling it raises ``UnsupportedFeatureError`` with ``native_backend_unavailable``.
Without Torch, import succeeds and registration is a no-op.
"""

from ._dynamo_backend import LLKBackend, get_backend, register

register()

__all__ = ["LLKBackend", "get_backend", "register"]
