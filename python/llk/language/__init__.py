from ._triton_ir import TritonIRBuilder, IRValue

_builder: TritonIRBuilder | None = None

# Sentinel-type objects recognized by the AST compiler
class constexpr:
    """Annotation sentinel for compile-time constant parameters."""
    pass


float32 = "float32"
float16 = "float16"
int32 = "int32"


def _get_builder() -> TritonIRBuilder:
    global _builder
    if _builder is None:
        _builder = TritonIRBuilder()
    return _builder


def arange(start: int, end: int) -> IRValue:
    return _get_builder().make_range(start, end)


def program_id(axis: int) -> IRValue:
    return _get_builder().get_program_id(axis)
