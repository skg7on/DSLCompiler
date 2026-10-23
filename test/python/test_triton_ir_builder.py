import pytest
from llk import UnsupportedFeatureError
from llk.frontend import Value


def test_compatibility_value_is_a_typed_frontend_value():
    from llk.language._triton_ir import IRValue

    assert IRValue("index", "index", (4,)) == Value("index", "index", (4,))


def test_legacy_string_builder_is_explicitly_unavailable():
    from llk.language._triton_ir import TritonIRBuilder

    with pytest.raises(UnsupportedFeatureError) as caught:
        TritonIRBuilder()
    assert caught.value.code == "legacy_ir_builder_unavailable"
