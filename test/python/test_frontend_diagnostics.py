# Fixtures intentionally include unused assignments that capture must validate.
# ruff: noqa: F841
import inspect

import llk.language as lang
import pytest
from ast_cases import binding, ports
from llk import LLKError
from llk.language._ast_compiler import compile_kernel


def test_unknown_call_cannot_equal_empty_kernel():
    def unknown(a, b, out, N: lang.constexpr):
        lang.unrecognized(a)

    with pytest.raises(LLKError) as caught:
        compile_kernel(unknown, binding(unknown))
    assert caught.value.code == "unsupported_frontend_call"
    assert caught.value.file == __file__
    assert caught.value.line == inspect.getsourcelines(unknown)[1] + 1
    assert caught.value.column == 8


@pytest.mark.parametrize(
    "kind", ["while", "branch", "return", "loop_state", "duplicate", "missing_other"]
)
def test_unsupported_semantics_have_source_locations(kind):
    def while_loop(a, b, out, N: lang.constexpr):
        while True:
            pass

    def runtime_branch(a, b, out, N: lang.constexpr):
        if lang.program_id(0) == 0:
            pass

    def returned(a, b, out, N: lang.constexpr):
        return 1

    def state(a, b, out, N: lang.constexpr):
        total = 0
        for k in range(N):
            total += k

    def duplicate(a, b, out, N: lang.constexpr):
        x = lang.load(a, True, mask=True, other=0.0)

    def missing(a, b, out, N: lang.constexpr):
        x = lang.load(a, mask=True)

    fn = {
        "while": while_loop,
        "branch": runtime_branch,
        "return": returned,
        "loop_state": state,
        "duplicate": duplicate,
        "missing_other": missing,
    }[kind]
    with pytest.raises(LLKError) as caught:
        compile_kernel(fn, binding(fn))
    assert caught.value.file == __file__
    assert caught.value.line > inspect.getsourcelines(fn)[1]
    assert caught.value.column >= 8


def test_source_unavailable_is_structured():
    namespace = {}
    exec(compile("def kernel(a): pass", "<unavailable>", "exec"), namespace)
    with pytest.raises(LLKError) as caught:
        binding(namespace["kernel"], (ports()[0],))
    assert caught.value.code == "source_unavailable"
    assert caught.value.file == "<unavailable>"


def test_mutable_global_capture_is_rejected():
    mutable = []

    def kernel(a, b, out, N: lang.constexpr):
        x = mutable[0]

    with pytest.raises(LLKError) as caught:
        binding(kernel)
    assert caught.value.code == "unsupported_frontend_constant"
    assert caught.value.file == __file__


def test_unknown_math_is_rejected_at_its_call_site():
    import math

    def kernel(a, b, out, N: lang.constexpr):
        x = math.sin(1.0)

    with pytest.raises(LLKError) as caught:
        compile_kernel(kernel, binding(kernel))
    assert caught.value.code == "unsupported_frontend_call"
    assert caught.value.line == inspect.getsourcelines(kernel)[1] + 1
    assert caught.value.column == 12


def test_discarded_dot_cannot_validate_unrelated_loop_carried_state():
    def kernel(a, b, out, N: lang.constexpr):
        r = lang.arange(0, 8)
        offsets = r[:, None] * 8 + r[None, :]
        acc = lang.zeros((8, 8), lang.float32)
        for k in range(N):
            x = lang.load(a + offsets, True, 0.0)
            y = lang.load(b + offsets, True, 0.0)
            unused = lang.dot(x, y, acc)
            acc = acc + 1.0
        lang.store(out + offsets, acc, True)

    with pytest.raises(LLKError) as caught:
        compile_kernel(kernel, binding(kernel))
    assert caught.value.code == "unsupported_frontend_control_flow"


def test_nonfinite_source_constants_fail_even_without_a_use():
    def kernel(a, b, out, N: lang.constexpr):
        x = 1e999

    with pytest.raises(LLKError):
        compile_kernel(kernel, binding(kernel))


def test_not_on_runtime_tiles_is_not_a_constexpr_condition():
    def kernel(a, b, out, N: lang.constexpr):
        mask = not (lang.arange(0, N) < N)

    with pytest.raises(LLKError):
        compile_kernel(kernel, binding(kernel))


def test_loop_induction_cannot_restore_a_preexisting_binding():
    def kernel(a, b, out, N: lang.constexpr):
        r = lang.arange(0, 8)
        offsets = r[:, None] * 8 + r[None, :]
        acc = lang.zeros((8, 8), lang.float32)
        k = 99
        for k in range(0, N, 2):
            x = lang.load(a + offsets, True, 0.0)
            y = lang.load(b + offsets, True, 0.0)
            acc = lang.dot(x, y, acc)
        lang.store(out + offsets, acc + k, True)

    with pytest.raises(LLKError) as caught:
        compile_kernel(kernel, binding(kernel))
    assert caught.value.code == "unsupported_frontend_control_flow"


def test_nested_loops_cannot_reuse_the_outer_induction_binding():
    def kernel(a, b, out, N: lang.constexpr):
        r = lang.arange(0, 8)
        offsets = r[:, None] * 8 + r[None, :]
        acc = lang.zeros((8, 8), lang.float32)
        for k in range(0, N, 4):
            for k in range(0, 4, 2):
                x = lang.load(a + offsets, True, 0.0)
                y = lang.load(b + offsets, True, 0.0)
                acc = lang.dot(x, y, acc)
        lang.store(out + offsets, acc, True)

    with pytest.raises(LLKError) as caught:
        compile_kernel(kernel, binding(kernel))
    assert caught.value.code == "unsupported_frontend_control_flow"
