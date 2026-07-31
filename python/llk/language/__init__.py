from ._triton_ir import TritonIRBuilder, IRValue

_builder: TritonIRBuilder | None = None


def _get_builder() -> TritonIRBuilder:
    global _builder
    if _builder is None:
        _builder = TritonIRBuilder()
    return _builder


def arange(start: int, end: int) -> IRValue:
    return _get_builder().make_range(start, end)


def program_id(axis: int) -> IRValue:
    return _get_builder().get_program_id(axis)
