import pytest


def test_arange_emits_make_range():
    """tl.arange should produce a Triton IR make_range op."""
    from llk.language._triton_ir import TritonIRBuilder

    builder = TritonIRBuilder()
    result = builder.make_range(0, 32)

    assert result is not None
    assert "tt.make_range" in str(builder.module)


def test_program_id_emits_get_program_id():
    """tl.program_id should produce tt.get_program_id."""
    from llk.language._triton_ir import TritonIRBuilder

    builder = TritonIRBuilder()
    result = builder.get_program_id(axis=0)

    assert result is not None
    assert "tt.get_program_id" in str(builder.module)
