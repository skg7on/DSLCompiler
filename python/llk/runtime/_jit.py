from typing import Callable

from llk._errors import UnsupportedFeatureError

from ..language._ast_compiler import compile_kernel


class JitFunction:
    """Kernel definition with typed capture; native invocation is a later milestone."""

    def __init__(self, fn: Callable):
        self._fn = fn

    def __call__(self, *args, **kwargs):
        raise UnsupportedFeatureError(
            "native frontend execution is unavailable",
            code="native_backend_unavailable",
        )

    def frontend_ir(self, binding):
        """Return checked frontend metadata for an explicit frozen binding."""
        return compile_kernel(self._fn, binding)

    def ttir(self) -> str:
        raise UnsupportedFeatureError(
            "frontend metadata is not upstream TTIR; the upstream adapter is unqualified",
            code="upstream_toolchain_unqualified",
        )

    def __repr__(self) -> str:
        return f"<JitFunction {self._fn.__name__}>"


def jit(fn: Callable | None = None) -> JitFunction | Callable:
    """Mark a kernel definition for typed capture and future native compilation."""
    if fn is not None:
        return JitFunction(fn)

    def decorator(fn: Callable) -> JitFunction:
        return JitFunction(fn)

    return decorator
