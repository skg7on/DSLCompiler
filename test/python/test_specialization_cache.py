"""PF09: bounded full-identity specialization cache behavior."""

from __future__ import annotations

import threading
from dataclasses import replace

import frontend_cases as frontend
import llk
import native_cases as cases
import pytest
from llk.frontend import LaunchSpec, Operation, Region, Value
from llk.runtime import _cache, _ffi
from llk.runtime._cache import CompilationCache, request_identity


def fresh_kernel():
    """A kernel with its own cache so counters never leak across tests."""
    return llk.jit(**{"config_root": ".", "config_files": cases.config_files()})(
        cases.add
    )


def make_request(shape=(16, 16), block=(8, 16), **options):
    settings = {"config_root": ".", "config_files": cases.config_files()}
    settings.update(options)
    return frontend.request(frontend.make_add_program(shape, block), **settings)


class FakeKernel:
    """Leaseable artifact with the same shared-ownership rule as the runtime."""

    def __init__(self, label):
        self.label = label
        self.invocations = 0
        self.references = 1
        self.closed = False

    def lease(self):
        assert not self.closed
        self.references += 1
        return FakeLease(self)

    def close(self):
        self.references -= 1
        if self.references == 0:
            self.closed = True


class FakeLease:
    def __init__(self, kernel):
        self.kernel = kernel
        self.closed = False

    def invoke(self):
        assert not self.closed
        assert not self.kernel.closed, "a lease must keep its artifact alive"
        self.kernel.invocations += 1

    def close(self):
        if self.closed:
            return
        self.closed = True
        self.kernel.close()


def environment():
    return b"pf09-test-environment"


def _rewrite_type(value, dtype):
    if isinstance(value, Value):
        return replace(value, dtype=value.dtype.replace("f32", dtype))
    if isinstance(value, Operation):
        return replace(
            value,
            results=tuple(_rewrite_type(item, dtype) for item in value.results),
            regions=tuple(_rewrite_type(item, dtype) for item in value.regions),
        )
    if isinstance(value, Region):
        return replace(
            value,
            arguments=tuple(_rewrite_type(item, dtype) for item in value.arguments),
            operations=tuple(_rewrite_type(item, dtype) for item in value.operations),
        )
    return value


def f16_variant(program):
    contract = replace(
        program.numerical_contract,
        storage_dtype="f16",
        compute_dtype="f16",
        accumulator_dtype="f16",
    )
    return replace(
        program,
        tensors=tuple(replace(tensor, dtype="f16") for tensor in program.tensors),
        operations=tuple(_rewrite_type(op, "f16") for op in program.operations),
        numerical_contract=contract,
    )


def test_default_capacity_is_128():
    assert CompilationCache().max_entries == 128
    assert CompilationCache(max_entries=3).max_entries == 3
    for invalid in (0, -1, True, 1.5, "8"):
        with pytest.raises(ValueError):
            CompilationCache(max_entries=invalid)


def test_same_full_identity_compiles_once_and_hits_afterwards():
    kernel = fresh_kernel()
    first_args, _ = cases.arguments()
    kernel(*first_args, grid=cases.launch())
    second_args, second_buffers = cases.arguments()
    kernel(*second_args, grid=cases.launch())

    info = kernel.cache_info()
    assert info.compilations == 1
    assert info.hits == 1
    assert info.misses == 1
    assert second_buffers[2].values() == cases.expected(
        cases.TILED_SHAPE[0] * cases.TILED_SHAPE[1]
    )

    first = kernel.compile(*first_args, grid=cases.launch())
    try:
        assert isinstance(first, llk.CompiledKernelLease)
        assert isinstance(first, llk.CompiledKernel)
    finally:
        first.close()


def test_changed_shape_specializes_separately():
    kernel = fresh_kernel()
    for shape, block in (((16, 16), (8, 16)), ((32, 64), (32, 64))):
        args, _ = cases.arguments(shape, block)
        kernel(*args, grid=cases.launch(shape, block))

    info = kernel.cache_info()
    assert info.compilations == 2
    assert info.hits == 0


def test_full_identity_covers_every_specialization_input():
    base = frontend.make_add_program((16, 16), (8, 16))
    settings = {"config_root": ".", "config_files": cases.config_files()}
    changed_config = dict(cases.config_files())
    key = "machines/x86-avx2-v2.yaml"
    changed_config[key] = changed_config[key] + "\n# changed target content\n"

    def identity(program, environment=b"env", **changes):
        options = dict(settings)
        options.update(changes)
        return request_identity(frontend.request(program, **options), environment)

    base_identity = identity(base)
    identities = {
        "shape": identity(frontend.make_add_program((32, 64), (8, 16))),
        "dtype": identity(f16_variant(base)),
        "constexpr": identity(
            replace(base, constexprs={"BLOCK_M": 8, "BLOCK_N": 8})
        ),
        "closure": identity(replace(base, captured_constants={"scale": 2.0})),
        "grid": identity(replace(base, launch=LaunchSpec((1, 2, 1), (8, 8, 1)))),
        "numerical": identity(
            replace(
                base,
                numerical_contract=replace(base.numerical_contract, fma=True),
            )
        ),
        "target": identity(base, target="generic-accelerator"),
        "target_contents": identity(base, config_files=changed_config),
        "abi_environment": identity(base, environment=b"env-v2"),
    }

    assert len(set(identities.values())) == len(identities)
    assert base_identity not in identities.values()


def test_equal_digest_buckets_keep_full_identities_separate(monkeypatch):
    monkeypatch.setattr(_cache, "_digest", lambda identity: "0" * 64)
    cache = CompilationCache(environment=environment)
    compiled = []

    def compile_fn(label):
        def run(request):
            compiled.append(label)
            return FakeKernel(label)

        return run

    first = cache.get_or_compile(make_request(), compile_fn("first"))
    second = cache.get_or_compile(
        make_request(shape=(32, 64), block=(32, 64)), compile_fn("second")
    )

    assert compiled == ["first", "second"]
    assert first.kernel.label == "first"
    assert second.kernel.label == "second"
    assert cache.info().compilations == 2


def test_eight_concurrent_misses_compile_once():
    cache = CompilationCache(environment=environment)
    request = make_request()
    entered = threading.Event()
    release = threading.Event()
    attempts = []
    successful_invocations = []
    failures = []

    def compile_fn(request):
        attempts.append(1)
        entered.set()
        assert release.wait(10), "the compiling leader was never released"
        return FakeKernel("shared")

    def worker():
        try:
            lease = cache.get_or_compile(request, compile_fn)
            try:
                lease.invoke()
                successful_invocations.append(lease)
            finally:
                lease.close()
        except BaseException as error:  # pragma: no cover - diagnostic path
            failures.append(error)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for thread in threads:
        thread.start()
    assert entered.wait(10)
    assert len(attempts) == 1, "only one leader may enter compilation"
    release.set()
    for thread in threads:
        thread.join(10)

    assert failures == []
    assert len(attempts) == 1
    concurrent_successful_invocations = len(successful_invocations)
    assert concurrent_successful_invocations == 8
    info = cache.info()
    assert info.compilations == 1
    # Every lookup is accounted for. A worker that arrives after the leader
    # finished legitimately observes a hit instead of a miss, so the split
    # between hits and misses is not timing-independent.
    assert info.hits + info.misses == 8
    assert info.misses >= 1


def test_eviction_keeps_an_active_lease_alive():
    cache = CompilationCache(max_entries=1, environment=environment)
    artifact = FakeKernel("active")
    lease = cache.get_or_compile(make_request(), lambda request: artifact)
    other = cache.get_or_compile(
        make_request(shape=(32, 64), block=(32, 64)),
        lambda request: FakeKernel("replacement"),
    )

    assert cache.info().evictions == 1
    # Eviction drops only the cache's reference; the active lease still owns
    # the artifact and nothing is freed underneath it.
    assert artifact.references == 1
    assert artifact.closed is False
    lease.invoke()
    assert artifact.invocations == 1
    other.invoke()
    lease.close()
    other.close()
    assert artifact.closed is True


def test_failed_compilation_is_not_cached_and_can_retry():
    cache = CompilationCache(environment=environment)
    request = make_request()
    attempts = []

    def compile_fn(request):
        attempts.append(1)
        if len(attempts) == 1:
            raise RuntimeError("injected compilation failure")
        return FakeKernel("retried")

    with pytest.raises(RuntimeError):
        cache.get_or_compile(request, compile_fn)
    assert cache.info().compilations == 0
    assert cache.info().hits == 0

    lease = cache.get_or_compile(request, compile_fn)
    assert lease.kernel.label == "retried"
    info = cache.info()
    assert info.compilations == 1
    assert info.misses == 2
    lease.close()


def test_native_environment_identity_is_queried_before_every_lookup(monkeypatch):
    environment = bytearray(b"environment-1")
    monkeypatch.setattr(_ffi, "native_environment", lambda: bytes(environment))
    cache = CompilationCache()
    request = make_request()

    first = cache.get_or_compile(request, _ffi.compile_request)
    environment[:] = b"environment-2"
    second = cache.get_or_compile(request, _ffi.compile_request)
    try:
        assert cache.info().compilations == 2
        assert (
            first.report.native_environment["compiler"]
            == second.report.native_environment["compiler"]
        )
    finally:
        first.close()
        second.close()


def test_unusable_compiled_artifact_does_not_poison_the_cache():
    cache = CompilationCache(environment=environment)
    request = make_request()

    class UnusableKernel(FakeKernel):
        def lease(self):
            raise RuntimeError("unusable artifact")

    with pytest.raises(RuntimeError):
        cache.get_or_compile(request, lambda request: UnusableKernel("broken"))
    assert cache.info().hits == 0

    lease = cache.get_or_compile(request, lambda request: FakeKernel("healthy"))
    assert lease.kernel.label == "healthy"
    # Both artifacts were compiled; only the healthy one stayed cached.
    assert cache.info().compilations == 2
    assert cache.info().misses == 2
    lease.close()


def test_changed_target_contents_cannot_reuse_a_cached_handle():
    cache = CompilationCache()
    base = cases.config_files()
    changed = dict(base)
    changed["machines/x86-avx2-v2.yaml"] = (
        base["machines/x86-avx2-v2.yaml"] + "\n# PF09 cache identity probe\n"
    )
    first = cache.get_or_compile(
        make_request(config_files=base), _ffi.compile_request
    )
    second = cache.get_or_compile(
        make_request(config_files=changed), _ffi.compile_request
    )
    try:
        assert cache.info().compilations == 2
        assert (
            first.report.native_request_identity
            != second.report.native_request_identity
        )
    finally:
        first.close()
        second.close()


def test_cache_hit_returns_the_artifact_for_the_same_full_identity():
    kernel = fresh_kernel()
    args, _ = cases.arguments()
    first = kernel.compile(*args, grid=cases.launch())
    second = kernel.compile(*args, grid=cases.launch())
    try:
        assert (
            second.report.native_request_identity
            == first.report.native_request_identity
        )
        assert kernel.cache_info().hits == 1
    finally:
        first.close()
        second.close()
