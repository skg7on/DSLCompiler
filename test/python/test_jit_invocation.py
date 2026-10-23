import llk
import pytest
from ast_cases import binding, masked_add
from llk.frontend import FrontendProgram


def test_jit_metadata_is_explicitly_named_frontend_ir():
    kernel = llk.jit(masked_add)
    program = kernel.frontend_ir(binding(masked_add))
    assert isinstance(program, FrontendProgram)
    assert program.entry == "masked_add"
    assert any(op.opcode == "store" for op in program.operations)


def test_frontend_ir_cannot_be_reported_as_upstream_ttir():
    with pytest.raises(llk.UnsupportedFeatureError) as caught:
        llk.jit(masked_add).ttir()
    assert caught.value.code == "upstream_toolchain_unqualified"


def test_invocation_stays_explicitly_unavailable():
    with pytest.raises(llk.UnsupportedFeatureError) as caught:
        llk.jit(masked_add)()
    assert caught.value.code == "native_backend_unavailable"
