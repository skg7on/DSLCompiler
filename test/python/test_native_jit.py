"""PF08: execute LLK decorators through caller-owned buffers and native code."""

from __future__ import annotations

import threading
from unittest import mock

import llk
import native_cases as cases
import pytest
from llk import ExecutionError, LLKError
from llk.frontend import LaunchSpec
from llk.runtime import _ffi


def spy_on_native_invoke():
    return mock.patch.object(_ffi, "native_invoke", wraps=_ffi.native_invoke)


def test_tiled_add_writes_exact_caller_owned_output_through_native_code():
    args, buffers = cases.arguments()
    a, b, out = buffers
    compiled = cases.add_kernel.compile(*args, grid=cases.launch())
    assert isinstance(compiled, llk.CompiledKernel)
    assert compiled.report.backend == "reference"
    input_bytes_before = a.bytes(), b.bytes()

    with spy_on_native_invoke() as native_invoke_spy:
        compiled.invoke(*cases.providers(buffers))

    assert native_invoke_spy.call_count == 1
    assert out.values() == cases.expected(cases.TILED_SHAPE[0] * cases.TILED_SHAPE[1])
    assert (a.bytes(), b.bytes()) == input_bytes_before
    assert a.guards_intact() and b.guards_intact() and out.guards_intact()
    assert compiled.report.plan_id
    assert compiled.report.abi_hash
    compiled.close()


def test_subscript_and_keyword_grid_launch_forms_agree():
    first_args, first_buffers = cases.arguments()
    second_args, second_buffers = cases.arguments()

    with spy_on_native_invoke() as native_invoke_spy:
        cases.add_kernel[cases.launch()](*first_args)
        cases.add_kernel(*second_args, grid=cases.launch())

    expected = cases.expected(cases.TILED_SHAPE[0] * cases.TILED_SHAPE[1])
    assert native_invoke_spy.call_count == 2
    assert first_buffers[2].values() == expected
    assert second_buffers[2].values() == expected
    assert first_buffers[2].guards_intact()
    assert second_buffers[2].guards_intact()


def test_full_domain_32x64_add_executes_exact_values():
    shape = cases.FULL_SHAPE
    args, buffers = cases.arguments(shape, shape)
    a, b, out = buffers
    compiled = cases.add_kernel.compile(*args, grid=cases.launch(shape, shape))
    compiled.invoke(*cases.providers(buffers))
    assert out.values() == cases.expected(shape[0] * shape[1])
    assert a.guards_intact() and b.guards_intact() and out.guards_intact()
    assert compiled.report.source_adapter == "linalg-elementwise-v1"
    compiled.close()


def test_precompiled_kernel_reports_and_closes_idempotently():
    args, buffers = cases.arguments()
    compiled = cases.add_kernel.compile(*args, grid=cases.launch())
    assert compiled.report.compiler_identity
    assert compiled.report.stages
    with spy_on_native_invoke() as native_invoke_spy:
        compiled.invoke(*cases.providers(buffers))
        compiled.close()
        compiled.close()
        with pytest.raises(ExecutionError) as caught:
            compiled.invoke(*cases.providers(buffers))
    assert caught.value.code == "invalid_frontend_handle"
    assert native_invoke_spy.call_count == 1


def test_close_waits_for_an_in_flight_invocation():
    args, buffers = cases.arguments()
    compiled = cases.add_kernel.compile(*args, grid=cases.launch())
    started = threading.Event()
    release = threading.Event()
    close_entered = threading.Event()
    close_done = threading.Event()
    failures = []
    real_invoke = _ffi.native_invoke

    def blocking_invoke(*call_args, **call_kwargs):
        started.set()
        if not release.wait(10):
            raise AssertionError("in-flight invocation was never released")
        return real_invoke(*call_args, **call_kwargs)

    def call():
        try:
            compiled.invoke(*cases.providers(buffers))
        except BaseException as error:  # pragma: no cover - diagnostic path
            failures.append(error)

    def close():
        close_entered.set()
        try:
            compiled.close()
        except BaseException as error:  # pragma: no cover - diagnostic path
            failures.append(error)
        finally:
            close_done.set()

    with mock.patch.object(_ffi, "native_invoke", side_effect=blocking_invoke):
        invoking = threading.Thread(target=call)
        invoking.start()
        assert started.wait(10)
        closing = threading.Thread(target=close)
        closing.start()
        assert close_entered.wait(10)
        assert not close_done.wait(0.2), "close must wait for the native call"
        release.set()
        invoking.join(10)
        closing.join(10)

    assert not invoking.is_alive() and not closing.is_alive()
    assert close_done.is_set()
    assert failures == []
    assert buffers[2].values() == cases.expected(cases.TILED_SHAPE[0] * cases.TILED_SHAPE[1])


def test_missing_grid_fails_before_native_entry():
    args, _ = cases.arguments()
    with spy_on_native_invoke() as native_invoke_spy:
        with pytest.raises(LLKError) as caught:
            cases.add_kernel(*args)
    assert caught.value.code == "invalid_launch_coverage"
    assert native_invoke_spy.call_count == 0


def test_partial_coverage_fails_before_native_entry():
    args, _ = cases.arguments()
    partial = LaunchSpec((1, 1, 1), cases.TILED_BLOCK + (1,))
    with spy_on_native_invoke() as native_invoke_spy:
        with pytest.raises(LLKError) as caught:
            cases.add_kernel(*args, grid=partial)
    assert caught.value.code == "invalid_launch_coverage"
    assert native_invoke_spy.call_count == 0
