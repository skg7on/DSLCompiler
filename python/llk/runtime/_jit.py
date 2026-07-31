import textwrap
from typing import Any, Callable

from ..language._ast_compiler import compile_kernel
from ..language._triton_ir import TritonIRBuilder


class JitFunction:
    """Wrapper around a JIT-compiled Triton kernel."""

    def __init__(self, fn: Callable):
        self._fn = fn
        self._compiled = False
        self._ttir: str | None = None

    def __call__(self, *args, **kwargs):
        if not self._compiled:
            self._compile(*args, **kwargs)
            self._compiled = True
        raise NotImplementedError("JIT execution not yet implemented")

    def _compile(self, *args, **kwargs) -> None:
        """Compile the kernel to Triton IR."""
        self._ttir = compile_kernel(self._fn)

    def ttir(self) -> str:
        """Return the Triton IR for this kernel (compile if needed)."""
        if not self._compiled:
            self._compile()
            self._compiled = True
        return self._ttir

    def __repr__(self) -> str:
        return f"<JitFunction {self._fn.__name__}>"


def jit(fn: Callable | None = None) -> JitFunction | Callable:
    """Decorator: mark a function for JIT compilation.

    Usage:
        @llk.jit
        def my_kernel(...): ...

        @llk.jit()
        def my_kernel(...): ...
    """
    if fn is not None:
        return JitFunction(fn)

    def decorator(fn: Callable) -> JitFunction:
        return JitFunction(fn)
    return decorator
