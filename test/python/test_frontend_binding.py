from __future__ import annotations

from dataclasses import replace

import llk.language as renamed_language
import pytest
from frontend_cases import make_add_program, request
from llk import LLKError
from llk.frontend import (
    CompileOptions,
    LaunchSpec,
    TensorArgument,
    bind_kernel,
    serialize_request,
)
from llk.language import constexpr as compile_time

SHIFT = 3
MUTABLE = []


def signature(a, b, out, BLOCK: renamed_language.constexpr = 16):
    return SHIFT


def tensors():
    return make_add_program().tensors


def test_positional_keyword_defaults_and_postponed_annotation():
    a, b, out = tensors()
    binding = bind_kernel(
        signature, (a,), {"b": b, "out": out}, (4, 4), CompileOptions()
    )
    assert binding.constexprs == {"BLOCK": 16}
    assert binding.captured_constants == {"SHIFT": 3}
    assert tuple(t.name for t in binding.tensors) == ("a", "b", "out")
    assert binding.launch.grid == (4, 4, 1)
    explicit = bind_kernel(signature, (a, b, out, 8), {}, (4, 4, 1), CompileOptions())
    assert explicit.constexprs["BLOCK"] == 8


def test_renamed_sentinel_identity_and_immutable_closure_capture():
    shift = (2, 3)

    def kernel(a, BLOCK: compile_time):
        return shift

    binding = bind_kernel(kernel, (tensors()[0], 8), {}, (1,), CompileOptions())
    assert binding.constexprs == {"BLOCK": 8}
    assert binding.captured_constants == {"shift": (2, 3)}


def test_grid_callable_evaluated_once_after_binding_and_normalized():
    calls = []

    def grid(meta):
        calls.append(meta["BLOCK"])
        return (2, 4)

    binding = bind_kernel(signature, tensors(), {}, grid, CompileOptions())
    assert calls == [16]
    assert binding.launch.grid == (2, 4, 1)


@pytest.mark.parametrize("grid", [(), (True,), (0,), (-1,), (1, 1, 1, 1), (2**63,), 1])
def test_invalid_grid_is_source_located(grid):
    with pytest.raises(LLKError) as caught:
        bind_kernel(signature, tensors(), {}, grid, CompileOptions())
    assert caught.value.file == __file__
    assert caught.value.line > 0
    assert caught.value.column >= 0


@pytest.mark.parametrize("kwargs", [{"a": None}, {"unknown": 1}])
def test_duplicate_and_unknown_arguments_fail_before_grid(kwargs):
    calls = []
    with pytest.raises(LLKError):
        bind_kernel(
            signature,
            tensors(),
            kwargs,
            lambda meta: calls.append(meta) or (1,),
            CompileOptions(),
        )
    assert calls == []


@pytest.mark.parametrize("value", [float("nan"), float("inf"), 2**63, [1], {"x": 1}])
def test_invalid_constexpr_values(value):
    with pytest.raises(LLKError):
        bind_kernel(signature, tensors() + (value,), {}, (1,), CompileOptions())


def test_mutable_capture_and_unknown_calls_are_explicit_errors():
    def mutable(a):
        return MUTABLE

    def unsupported(a):
        return print(a)

    for fn in (mutable, unsupported):
        with pytest.raises(LLKError):
            bind_kernel(fn, (tensors()[0],), {}, (1,), CompileOptions())


def test_annotation_resolution_never_evaluates_code():
    called = []

    def dangerous():
        called.append(True)
        return renamed_language.constexpr

    def kernel(a, BLOCK):
        pass

    kernel.__annotations__["BLOCK"] = "dangerous()"
    with pytest.raises(LLKError):
        bind_kernel(kernel, (tensors()[0], 8), {}, (1,), CompileOptions())
    assert called == []


def test_source_unavailable_has_a_structured_diagnostic():
    namespace = {}
    exec(compile("def kernel(a): pass", "<unavailable>", "exec"), namespace)
    with pytest.raises(LLKError) as caught:
        bind_kernel(namespace["kernel"], (tensors()[0],), {}, (1,), CompileOptions())
    assert caught.value.code == "source_unavailable"


def test_providers_stay_owned_and_never_enter_canonical_bytes():
    class Provider:
        def data_ptr(self):
            raise AssertionError("PF03 must not inspect storage")

        def __repr__(self):
            raise AssertionError("providers never serialize")

    first, second = Provider(), Provider()
    ports = tensors()
    owners = tuple(TensorArgument(port, first) for port in ports)
    moved = tuple(TensorArgument(port, second) for port in ports)
    a = bind_kernel(
        signature, owners, {}, LaunchSpec((4, 4, 1), (8, 16, 1)), CompileOptions()
    )
    b = bind_kernel(
        signature, moved, {}, LaunchSpec((4, 4, 1), (8, 16, 1)), CompileOptions()
    )
    assert all(provider is first for provider in a.providers)
    p = make_add_program()
    assert serialize_request(
        request(replace(p, tensors=a.tensors))
    ) == serialize_request(request(replace(p, tensors=b.tensors)))


def test_dtype_named_immutable_capture_is_not_dropped_as_a_language_symbol():
    dtype_name = renamed_language.float32

    def kernel(a):
        return dtype_name

    bound = bind_kernel(kernel, (tensors()[0],), {}, (1,), CompileOptions())
    assert bound.captured_constants == {"dtype_name": "float32"}


def test_annotation_call_expressions_do_not_execute_during_inspection(tmp_path):
    import importlib.util
    import sys

    # 3.14 defers annotations by default. Earlier supported Pythons require
    # postponed annotations to make the same no-evaluation contract observable.
    prefix = (
        "from __future__ import annotations\n" if sys.version_info < (3, 14) else ""
    )
    path = tmp_path / "annotation_kernel.py"
    path.write_text(
        prefix
        + """import llk.language as tl
calls=[]
def dangerous():
    calls.append(True)
    return tl.constexpr
def kernel(a, BLOCK: dangerous()):
    pass
"""
    )
    spec = importlib.util.spec_from_file_location("annotation_kernel", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with pytest.raises(LLKError):
        bind_kernel(module.kernel, (tensors()[0], 8), {}, (1,), CompileOptions())
    assert module.calls == []


def test_annotation_only_local_alias_and_keyword_only_default(tmp_path):
    import importlib.util

    path = tmp_path / "alias_kernel.py"
    path.write_text("""from llk.language import constexpr
def outer():
    compile_time = constexpr
    def kernel(a, /, *, BLOCK: compile_time = 8):
        pass
    return kernel
kernel = outer()
""")
    spec = importlib.util.spec_from_file_location("alias_kernel", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    bound = bind_kernel(module.kernel, (tensors()[0],), {}, (1,), CompileOptions())
    assert bound.constexprs == {"BLOCK": 8}
    overridden = bind_kernel(
        module.kernel, (tensors()[0],), {"BLOCK": 16}, (1,), CompileOptions()
    )
    assert overridden.constexprs == {"BLOCK": 16}
