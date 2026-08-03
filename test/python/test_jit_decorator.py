import pytest


def test_jit_decorator_returns_jit_function():
    """llk.jit decorator should wrap a function in JitFunction."""
    import llk

    @llk.jit
    def simple_kernel(x_ptr, y_ptr, N: int):
        pass

    from llk import JitFunction
    assert isinstance(simple_kernel, JitFunction)
    assert hasattr(simple_kernel, '__call__')
