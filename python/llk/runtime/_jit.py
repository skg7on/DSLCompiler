class JitFunction:
    """Wrapper around a JIT-compiled Triton kernel."""
    def __init__(self, fn):
        self._fn = fn
        self._compiled = False

    def __call__(self, *args, **kwargs):
        raise NotImplementedError("JIT compilation not yet implemented")

def jit(fn):
    return JitFunction(fn)
