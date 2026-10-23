"""Compatibility imports for the former string builder, now frontend metadata.

No object in this module represents upstream Triton IR or an MLIR module.
"""

from llk._errors import UnsupportedFeatureError
from llk.frontend import FrontendProgram, Operation, Value

IRValue = Value


class TritonIRBuilder:
    """Reject the removed pseudo-TTIR builder instead of emitting unchecked text."""

    def __init__(self):
        raise UnsupportedFeatureError(
            "the string builder is retired; use compile_kernel(fn, binding) for FrontendProgram",
            code="legacy_ir_builder_unavailable",
        )


__all__ = ["FrontendProgram", "IRValue", "Operation", "TritonIRBuilder", "Value"]
