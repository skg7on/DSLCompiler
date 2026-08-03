"""Registers 'llk' as a torch.compile backend.

The backend detects ATen patterns that map to llk.* ops:

- SiLU(mm(X, Wg)) * mm(X, Wu) -> llk.fused_swiglu
- RoPE pattern -> llk.rope
- scaled_dot_product_attention -> llk.attention

The torch import is optional so that ``import llk.torch`` works in
environments without PyTorch installed; backend registration only happens
when torch is available.
"""

from __future__ import annotations

try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only without torch
    torch = None  # type: ignore[assignment]
    _TORCH_AVAILABLE = False


class LLKBackend:
    """torch.compile backend that lowers to LLK dialect."""

    def __init__(self) -> None:
        self._compiled = 0

    def __call__(self, gm: "torch.fx.GraphModule",
                 example_inputs: "list[torch.Tensor]"):
        """Entry point for torch.compile.

        For now, return the graph unchanged (no lowering). M8e+ full
        integration will add ATen pattern matching here.
        """
        self._compiled += 1
        return gm.forward


def register() -> None:
    """Register the LLK backend with PyTorch Dynamo.

    No-op when PyTorch is not importable. After this call,
    ``torch.compile(model, backend="llk")`` is available.
    """
    if not _TORCH_AVAILABLE:
        return
    torch._dynamo.reset()
    torch._dynamo.register_backend("llk", LLKBackend(), eager=False)


_llk_backend = LLKBackend()
