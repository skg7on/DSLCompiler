"""PF09: leases, cache eviction and close/invoke races hold native ownership."""

from __future__ import annotations

import threading
from unittest import mock

import frontend_cases as frontend
import llk
import native_cases as cases
import pytest
from llk import ExecutionError
from llk.runtime import _ffi
from llk.runtime._cache import CompilationCache


def fresh_kernel():
    return llk.jit(**{"config_root": ".", "config_files": cases.config_files()})(
        cases.add
    )


def add_request(shape, block):
    return frontend.request(
        frontend.make_add_program(shape, block),
        config_root=".",
        config_files=cases.config_files(),
    )


def test_active_lease_survives_eviction_of_its_cache_entry():
    cache = CompilationCache(max_entries=1)
    lease = cache.get_or_compile(
        add_request((16, 16), (8, 16)), _ffi.compile_request
    )
    _, buffers = cases.arguments((16, 16), (8, 16))
    other = cache.get_or_compile(
        add_request((32, 64), (32, 64)), _ffi.compile_request
    )
    try:
        assert cache.info().evictions == 1
        lease.invoke(*cases.providers(buffers))
        assert buffers[2].values() == cases.expected(16 * 16)
    finally:
        other.close()
        lease.close()


def test_closing_a_lease_keeps_the_cache_owned_handle_alive():
    kernel = fresh_kernel()
    args, buffers = cases.arguments()
    first = kernel.compile(*args, grid=cases.launch())
    first.close()
    second = kernel.compile(*args, grid=cases.launch())
    try:
        second.invoke(*cases.providers(buffers))
        assert buffers[2].values() == cases.expected(16 * 16)
        assert kernel.cache_info().hits == 1
    finally:
        second.close()


def test_close_waits_for_an_in_flight_invocation_through_a_lease():
    kernel = fresh_kernel()
    args, buffers = cases.arguments()
    lease = kernel.compile(*args, grid=cases.launch())
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
            lease.invoke(*cases.providers(buffers))
        except BaseException as error:  # pragma: no cover - diagnostic path
            failures.append(error)

    def close():
        close_entered.set()
        try:
            lease.close()
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
    assert buffers[2].values() == cases.expected(16 * 16)
    with pytest.raises(ExecutionError) as caught:
        lease.invoke(*cases.providers(buffers))
    assert caught.value.code == "invalid_frontend_handle"


def test_failed_compilation_is_not_poisoned_for_retry():
    kernel = fresh_kernel()
    args, _ = cases.arguments()
    real_compile = _ffi.native_compile
    attempts = []

    def flaky(payload):
        attempts.append(1)
        if len(attempts) == 1:
            raise RuntimeError("injected native compile failure")
        return real_compile(payload)

    with mock.patch.object(_ffi, "native_compile", side_effect=flaky):
        with pytest.raises(RuntimeError):
            kernel(*args, grid=cases.launch())
        retry_args, retry_buffers = cases.arguments()
        kernel(*retry_args, grid=cases.launch())

    assert len(attempts) == 2
    info = kernel.cache_info()
    assert info.compilations == 1
    assert info.misses == 2
    assert retry_buffers[2].values() == cases.expected(16 * 16)


def test_eight_concurrent_invocations_share_one_native_compilation():
    from llk.runtime import _jit

    kernel = fresh_kernel()
    barrier = threading.Barrier(8)
    outputs = []
    failures = []
    real_compile = _jit.compile_request
    calls = []

    def counting_compile(request):
        calls.append(1)
        return real_compile(request)

    def worker():
        try:
            args, buffers = cases.arguments()
            barrier.wait(10)
            kernel(*args, grid=cases.launch())
            outputs.append(buffers[2].values())
        except BaseException as error:  # pragma: no cover - diagnostic path
            failures.append(error)

    with mock.patch.object(_jit, "compile_request", side_effect=counting_compile):
        threads = [threading.Thread(target=worker) for _ in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(20)

    assert failures == []
    assert len(calls) == 1
    expected = cases.expected(cases.TILED_SHAPE[0] * cases.TILED_SHAPE[1])
    assert len(outputs) == 8
    assert all(values == expected for values in outputs)
    assert kernel.cache_info().compilations == 1


def test_one_native_compilation_feeds_repeated_invocations():
    kernel = fresh_kernel()
    args, buffers = cases.arguments()
    with mock.patch.object(
        _ffi, "native_compile", wraps=_ffi.native_compile
    ) as compile_spy:
        for _ in range(3):
            kernel(*args, grid=cases.launch())

    assert compile_spy.call_count == 1
    info = kernel.cache_info()
    assert info.compilations == 1
    assert info.hits == 2
    assert buffers[2].values() == cases.expected(16 * 16)
