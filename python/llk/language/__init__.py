"""Symbols recognized by typed AST capture; never an eager tensor interpreter."""

from llk._errors import UnsupportedFeatureError


class constexpr:
    """Annotation sentinel for compile-time constant parameters."""


float32 = "float32"
bfloat16 = "bfloat16"
float16 = "float16"
int32 = "int32"
int8 = "int8"


def _outside():
    raise UnsupportedFeatureError(
        "language operations require frontend capture", code="outside_frontend_capture"
    )


def program_id(axis):
    _outside()


def arange(start, end):
    _outside()


def load(pointer, mask=True, other=None):
    _outside()


def store(pointer, value, mask=True):
    _outside()


def zeros(shape, dtype=float32):
    _outside()


def dot(a, b, acc=None):
    _outside()


def where(condition, x, y):
    _outside()


def cast(value, dtype, rounding="rtne"):
    _outside()


def cdiv(a, b):
    _outside()


def silu(value, approximation="exact"):
    _outside()


def exp(value):
    """Accepted only inside the recognized exact SiLU expression."""
    _outside()
