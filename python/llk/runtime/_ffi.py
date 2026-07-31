"""ctypes bridge to the C++ LLK runtime."""


class Tensor:
    """Python-side tensor descriptor matching C Tensor2D."""
    def __init__(self, data, shape, strides=None, dtype=None):
        self.data = data
        self.shape = shape
        self.strides = strides or _default_strides(shape)
        self.dtype = dtype


def _default_strides(shape):
    strides = [1]
    for d in reversed(shape[1:]):
        strides.insert(0, strides[0] * d)
    return tuple(strides)
