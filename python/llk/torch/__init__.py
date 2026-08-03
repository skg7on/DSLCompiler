"""PyTorch Dynamo backend for the LLK compiler.

Usage:
    import torch
    import llk.torch  # registers backend="llk"
    model = torch.compile(model, backend="llk")

Importing this package also calls ``register()`` so the backend is available
without an explicit call. If PyTorch is not installed the import still
succeeds and registration is a no-op.
"""

from ._dynamo_backend import LLKBackend, register

register()

__all__ = ["LLKBackend", "register"]
