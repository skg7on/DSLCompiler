"""PF18: launch genuine upstream Triton kernels through the LLK CPU backend.

The launcher needs the upstream Triton runtime and the LLK frontend C API in
one process.  Both ship their own copy of LLVM, so activation first qualifies
native-library coexistence in a subprocess; where the two images cannot be
loaded together the lane records ``upstream_toolchain_unqualified`` instead of
guessing.  The binding, artifact and activation tests below run without the C
API and remain required.
"""

from __future__ import annotations

import ctypes
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import types
from pathlib import Path
from unittest import mock

import pytest

import llk
import llk.triton
from llk._errors import (
    CompilationError,
    TensorContractError,
    UnsupportedFeatureError,
)
from llk.frontend import CompileOptions
from llk.runtime import _ffi

ADD_SOURCE = """\
import triton
import triton.language as tl


@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    y = tl.load(y_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, x + y, mask=mask)


@triton.jit
def mul_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    y = tl.load(y_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, x * y, mask=mask)
"""

SIGNATURE = {
    "x_ptr": "*fp32",
    "y_ptr": "*fp32",
    "out_ptr": "*fp32",
    "n": "i32",
    "BLOCK": "constexpr",
}
SPECIALIZATION = {
    "x_ptr": {"divisibility": 16},
    "y_ptr": {"divisibility": 16},
    "out_ptr": {"divisibility": 16},
}

CONFIG_FILES = (
    "machines/x86-avx2-v2.yaml",
    "mapping/x86-avx2/layouts.llkmap",
    "mapping/x86-avx2/rules.llkmap",
)

_MISSING = object()


def _root():
    return Path(__file__).resolve().parents[3]


def _options(**changes):
    root = _root()
    data = {
        "config_root": str(root),
        "config_files": {
            name: (root / name).read_text(encoding="utf-8")
            for name in CONFIG_FILES
        },
    }
    data.update(changes)
    return CompileOptions(**data)


def _kernel_path(tmp_path):
    path = tmp_path / "triton launch kernels.py"
    path.write_text(ADD_SOURCE, encoding="utf-8")
    return path


def _load_kernel_module(triton_path, path):
    if str(triton_path) not in sys.path:
        sys.path.insert(0, str(triton_path))
    name = f"llk_triton_launch_{abs(hash(path))}"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _template(path, function="add_kernel", block=64):
    return llk.triton.capture_kernel(
        path,
        function,
        SIGNATURE,
        constexprs={"BLOCK": block},
        specialization=SPECIALIZATION,
        triton_python=str(_triton_path_from_sys()),
    )


def _triton_path_from_sys():
    import triton

    return Path(triton.__file__).resolve().parent.parent


def _torch_pair(torch, shape, block):
    rows, columns = shape
    base = torch.arange(rows * columns, dtype=torch.float32).reshape(rows, columns)
    x = base * 0.5 - 3.0
    y = base * 0.25 + 1.0
    out = torch.full(shape, -999.0, dtype=torch.float32)
    return x, y, out


def _native_spy():
    return mock.patch.object(_ffi, "native_invoke", wraps=_ffi.native_invoke)


class _Guarded:
    """ctypes f32 storage with two payload guards on each side."""

    def __init__(self, shape, payload=None):
        rows, columns = shape
        count = rows * columns
        self.rows, self.columns = rows, columns
        self.guard = 2
        self.storage = (ctypes.c_float * (count + 2 * self.guard))()
        for index in range(len(self.storage)):
            self.storage[index] = -777.0
        byte_view = memoryview(self.storage).cast("B")
        start = self.guard * 4
        self.view = byte_view[start : start + count * 4].cast("f", (rows, columns))
        for row in range(rows):
            for column in range(columns):
                index = row * columns + column
                self.view[row, column] = 0.0 if payload is None else payload[index]

    def values(self):
        return [
            self.view[row, column]
            for row in range(self.rows)
            for column in range(self.columns)
        ]

    def guards_intact(self):
        return (
            self.storage[0] == -777.0
            and self.storage[1] == -777.0
            and self.storage[-1] == -777.0
            and self.storage[-2] == -777.0
        )


@pytest.fixture(scope="session")
def torch(triton_path):
    try:
        import torch
    except ImportError:  # pragma: no cover - required Torch lane
        pytest.skip("torch is required for the Triton launch lane")
    return torch


def _coexistence_skip_reason():
    status = llk.triton.coexistence_status()
    if status.qualified:
        return None
    return "upstream Triton and LLK native libraries do not coexist: " + "; ".join(
        status.reasons
    )


# ---------------------------------------------------------------------------
# Artifact, backend and binding evidence (no LLK C API required).
# ---------------------------------------------------------------------------


def test_artifact_is_a_versioned_unbound_template(triton_path, tmp_path):
    from llk.triton._backend import CPUUtils, UnboundLaunchArtifact

    artifact_bytes = _compile_artifact(triton_path, tmp_path)
    utils = CPUUtils()
    module, function, n_regs, n_spills, n_max_threads = utils.load_binary(
        "add_kernel", artifact_bytes, 0, 0
    )
    assert function is module
    assert module.artifact.entry == "add_kernel"
    assert n_regs == n_spills == 0
    assert n_max_threads >= 1
    assert isinstance(module.artifact, UnboundLaunchArtifact)
    assert module.artifact.schema_version == 1
    assert module.artifact.kind == "llk-unbound-triton-template"
    assert module.metadata["machine_code"] is False
    assert module.metadata["kind"] == "llk-unbound-triton-template"
    assert "tt.addptr" in module.artifact.ttir
    assert "loc(" not in module.artifact.ttir


def test_artifact_loader_rejects_malformed_version_and_hash(triton_path, tmp_path):
    from llk.triton._backend import CPUUtils

    artifact_bytes = _compile_artifact(triton_path, tmp_path)
    utils = CPUUtils()
    payload = json.loads(artifact_bytes)

    def corrupt(**changes):
        data = json.loads(json.dumps(payload))
        data.update(changes)
        return json.dumps(data).encode("utf-8")

    cases = [
        b"{}",
        b"not json",
        corrupt(schema_version=99),
        corrupt(kind="cubin"),
        corrupt(backend_version="999"),
        corrupt(ttir_hash="0" * 64),
        corrupt(ttir="module {\n}\n"),
        corrupt(entry="other_kernel"),
        corrupt(upstream_version="9.9.9"),
    ]
    for data in cases:
        with pytest.raises(CompilationError) as caught:
            utils.load_binary("add_kernel", data, 0, 0)
        assert caught.value.code == "triton_artifact_mismatch"
    with pytest.raises(CompilationError):
        utils.load_binary("add_kernel", artifact_bytes[:128], 0, 0)
    with pytest.raises(CompilationError):
        utils.load_binary("wrong_name", artifact_bytes, 0, 0)


def test_bind_translates_the_flat_kernel_into_a_checked_program(
    triton_path, tmp_path, torch
):
    from llk.triton._driver import CPULauncher

    template = _template(_kernel_path(tmp_path)).template
    x, y, out = _torch_pair(torch, (32, 64), 64)
    request = CPULauncher.bind_triton_template(
        template,
        (x, y, out, 2048, 64),
        (32, 1, 1),
        _options(),
    )
    program = request.program
    assert program.entry == "add_kernel"
    assert [tensor.name for tensor in program.tensors] == ["x_ptr", "y_ptr", "out_ptr"]
    assert all(tensor.shape == (32, 64) for tensor in program.tensors)
    assert program.launch.grid == (32, 1, 1)
    assert program.launch.block == (1, 64, 1)
    opcodes = [operation.opcode for operation in program.operations]
    assert opcodes[-1] == "return"
    assert "address" in opcodes and "load" in opcodes and "store" in opcodes
    assert opcodes.count("store") == 1
    assert opcodes.count("load") == 2
    assert "add" in opcodes
    assert program.captured_constants["triton_template_identity"] == template.identity()
    assert program.captured_constants["triton_source_hash"] == template.source_hash
    assert request.options.numerical_contract == program.numerical_contract


def test_bind_uses_a_second_shape_bound_identity(triton_path, tmp_path, torch):
    from llk.runtime._cache import request_identity

    from llk.triton._driver import CPULauncher

    path = _kernel_path(tmp_path)
    template = _template(path).template
    tall_template = _template(path, block=32).template
    wide = _torch_pair(torch, (32, 64), 64)
    tall = _torch_pair(torch, (64, 32), 32)
    short = _torch_pair(torch, (16, 64), 64)
    first = CPULauncher.bind_triton_template(
        template, (*wide, 2048, 64), (32, 1, 1), _options()
    )
    second = CPULauncher.bind_triton_template(
        tall_template, (*tall, 2048, 32), (64, 1, 1), _options()
    )
    third = CPULauncher.bind_triton_template(
        template, (*short, 1024, 64), (16, 1, 1), _options()
    )
    identity = lambda request: request_identity(request, b"test-environment")
    assert identity(first) != identity(second)
    assert identity(first) != identity(third)
    assert identity(second) != identity(third)
    assert third.program.launch.grid == (16, 1, 1)


def test_bind_covers_the_declared_domain_exactly(triton_path, tmp_path, torch):
    from llk.triton._driver import CPULauncher

    template = _template(_kernel_path(tmp_path)).template
    x, y, out = _torch_pair(torch, (32, 64), 64)
    with pytest.raises(TensorContractError) as caught:
        CPULauncher.bind_triton_template(
            template, (x, y, out, 2048, 64), (31, 1, 1), _options()
        )
    assert caught.value.code == "invalid_launch_coverage"
    with pytest.raises(TensorContractError) as caught:
        CPULauncher.bind_triton_template(
            template, (x, y, out, 2048, 64), (32, 2, 1), _options()
        )
    assert caught.value.code == "invalid_launch_coverage"


def test_bind_rejects_gpu_tensors_before_any_native_request(
    triton_path, tmp_path, torch
):
    from llk.triton._driver import CPULauncher

    template = _template(_kernel_path(tmp_path)).template

    class FakeCudaTensor:
        device = types.SimpleNamespace(type="cuda")
        dtype = torch.float32
        shape = (32, 64)
        strides = (64, 1)

        def data_ptr(self):  # pragma: no cover - must never be reached
            raise AssertionError("a GPU pointer must never be requested")

    with pytest.raises(TensorContractError) as caught:
        CPULauncher.bind_triton_template(
            template,
            (FakeCudaTensor(), FakeCudaTensor(), FakeCudaTensor(), 2048, 64),
            (32, 1, 1),
            _options(),
        )
    assert caught.value.code == "invalid_frontend_buffer"
    assert "CPU" in caught.value.message


def test_bind_rejects_misaligned_and_non_compact_buffers(
    triton_path, tmp_path, torch
):
    from llk.triton._driver import CPULauncher

    template = _template(_kernel_path(tmp_path)).template
    rows, columns = 32, 64
    count = rows * columns
    storage = (ctypes.c_float * (count + 8))()
    base = ctypes.addressof(storage)
    offset = 0 if base % 16 else 1  # element-aligned but not 16-byte aligned
    view = memoryview(storage).cast("B")[
        offset * 4 : (offset + count) * 4
    ].cast("f", (rows, columns))
    other = memoryview(storage).cast("B")[: count * 4].cast("f", (rows, columns))
    with pytest.raises(TensorContractError) as caught:
        CPULauncher.bind_triton_template(
            template, (view, other, other, 2048, 64), (32, 1, 1), _options()
        )
    assert caught.value.code == "invalid_frontend_buffer"
    assert "alignment" in caught.value.message.lower()

    transposed = torch.arange(count, dtype=torch.float32).reshape(columns, rows).t()
    with pytest.raises(TensorContractError) as caught:
        CPULauncher.bind_triton_template(
            template, (transposed, transposed, transposed, 2048, 64), (32, 1, 1),
            _options(),
        )
    assert caught.value.code == "invalid_frontend_buffer"


def test_bind_rejects_a_scalar_that_does_not_match_the_artifact(
    triton_path, tmp_path, torch
):
    from llk.triton._driver import CPULauncher

    template = _template(_kernel_path(tmp_path)).template
    x, y, out = _torch_pair(torch, (32, 64), 64)
    with pytest.raises(TensorContractError):
        CPULauncher.bind_triton_template(
            template, (x, y, out, 2**40, 64), (32, 1, 1), _options()
        )
    with pytest.raises(TensorContractError):
        CPULauncher.bind_triton_template(
            template, (x, y, out, 2048, 32), (32, 1, 1), _options()
        )


def test_bound_program_executes_natively_in_a_triton_free_process(
    triton_path, tmp_path
):
    """The bound request is the native contract; execute it without Triton.

    Activation is gated on native-library coexistence, but the binding half is
    independently checkable: the exact serialized ``CompileRequest`` is
    compiled and invoked by the LLK C API in a Triton-free subprocess, with
    real inputs and outputs, and must produce one distinct native identity per
    concrete tensor/launch shape.
    """
    import numpy as np

    from llk.frontend._serialize import serialize_request
    from llk.triton._driver import CPULauncher

    template = _template(_kernel_path(tmp_path), block=1024).template
    cases = []
    expected = {}
    for name, shape, grid, extent in (
        ("wide", (32, 64), (2,), 2048),
        ("tall", (64, 32), (2,), 2048),
        ("short", (16, 64), (1,), 1024),
    ):
        count = shape[0] * shape[1]
        base = np.arange(count, dtype=np.float32).reshape(shape)
        x = np.ascontiguousarray(base * np.float32(0.5) - np.float32(3.0))
        y = np.ascontiguousarray(base * np.float32(0.25) + np.float32(1.0))
        out = np.zeros(shape, dtype=np.float32)
        request = CPULauncher.bind_triton_template(
            template, (x, y, out, extent, 1024), grid, _options()
        )
        directory = tmp_path / name
        directory.mkdir()
        (directory / "request.json").write_bytes(serialize_request(request))
        np.save(directory / "x.npy", x)
        np.save(directory / "y.npy", y)
        np.save(directory / "out.npy", out)
        cases.append(
            json.dumps(
                {
                    "name": name,
                    "request": str(directory / "request.json"),
                    "x": str(directory / "x.npy"),
                    "y": str(directory / "y.npy"),
                    "out": str(directory / "out.npy"),
                }
            )
        )
        expected[name] = (x + y, directory / "out.npy")
    llk_python = str(Path(llk.__file__).resolve().parent.parent)
    argv = [
        sys.executable,
        "-I",
        str(Path(__file__).with_name("_native_request_worker.py")),
        "--llk-python",
        llk_python,
    ]
    for case in cases:
        argv.extend(["--case", case])
    completed = subprocess.run(
        argv,
        capture_output=True,
        text=True,
        timeout=600,
        env=dict(os.environ),
    )
    assert completed.returncode == 0, completed.stderr[-2000:]
    lines = completed.stdout.split("\n", 1)
    assert lines[0] == "LLK-TRITON-NATIVE-CASES-V1", completed.stderr[-2000:]
    frame = json.loads(lines[1])
    assert frame["ok"] is True, frame
    results = {item["name"]: item for item in frame["results"]}
    assert set(results) == set(expected)
    for name, (wanted, path) in expected.items():
        np.testing.assert_array_equal(np.load(path), wanted)
        assert results[name]["backend"] == "reference"
        assert results[name]["source_adapter"] == "linalg-elementwise-v1"
        assert results[name]["plan_id"]
        assert results[name]["native_request_identity"]
    identities = {
        item["native_request_identity"] for item in results.values()
    }
    assert len(identities) == 3


def test_bind_rejects_a_relaxed_numerical_policy(triton_path, tmp_path, torch):
    from dataclasses import replace

    from llk.frontend import NumericalContract
    from llk.triton._driver import CPULauncher

    template = _template(_kernel_path(tmp_path)).template
    x, y, out = _torch_pair(torch, (32, 64), 64)
    relaxed = replace(
        _options(),
        numerical_contract=NumericalContract(
            activation_expression="x/(1+exp(-x))",
            activation_approximation="exact",
        ),
    )
    with pytest.raises(UnsupportedFeatureError) as caught:
        CPULauncher.bind_triton_template(
            template, (x, y, out, 2048, 64), (32, 1, 1), relaxed
        )
    assert caught.value.code == "unsupported_triton_launch"


def test_backend_compiles_the_qualified_ttir_pipeline(triton_path, tmp_path):
    from llk.triton._backend import LLKCPUBackend

    artifact_bytes = _compile_artifact(triton_path, tmp_path)
    payload = json.loads(artifact_bytes)
    exported = llk.triton.capture_kernel(
        _kernel_path(tmp_path),
        "add_kernel",
        SIGNATURE,
        constexprs={"BLOCK": 64},
        specialization=SPECIALIZATION,
        triton_python=str(triton_path),
    )
    assert payload["ttir"] == exported.ttir
    assert payload["ttir_hash"] == exported.template.ttir_hash
    assert LLKCPUBackend.BINARY_EXT == "llktpl"


# ---------------------------------------------------------------------------
# Activation lifetime.
# ---------------------------------------------------------------------------


class _SentinelDriver:
    def __getattr__(self, name):  # pragma: no cover - never used by the test
        raise AssertionError(f"prior driver was used: {name}")


def _qualified_status():
    from llk.triton._driver import CoexistenceStatus

    return CoexistenceStatus(qualified=True, reasons=())


@pytest.fixture
def qualified_activation(monkeypatch):
    from triton.runtime.driver import driver as driver_config

    from llk.triton import _driver

    monkeypatch.setattr(_driver, "coexistence_status", _qualified_status)
    monkeypatch.setattr(
        _driver, "_qualify_version", lambda: mock.Mock(triton_version="3.6.0")
    )
    previous = getattr(driver_config, "_active", _MISSING)
    yield driver_config
    if previous is _MISSING:
        driver_config._active = None
    else:
        driver_config._active = previous


def test_plugin_shims_expose_concrete_upstream_classes(triton_path):
    import inspect

    from triton.backends import Backend, _find_concrete_subclasses
    from triton.backends.compiler import BaseBackend
    from triton.backends.driver import DriverBase

    from llk.triton._plugin import compiler, driver

    assert issubclass(compiler.LLKCPUBackend, BaseBackend)
    assert issubclass(driver.LLKCPUDriver, DriverBase)
    assert not inspect.isabstract(compiler.LLKCPUBackend)
    assert not inspect.isabstract(driver.LLKCPUDriver)
    assert _find_concrete_subclasses(compiler, BaseBackend) is compiler.LLKCPUBackend
    assert _find_concrete_subclasses(driver, DriverBase) is driver.LLKCPUDriver
    entry = Backend(compiler.LLKCPUBackend, driver.LLKCPUDriver)
    assert entry.compiler is compiler.LLKCPUBackend


def test_missing_native_library_fails_before_driver_installation(
    triton_path, monkeypatch
):
    from llk.runtime import _ffi
    from llk.triton import _driver

    monkeypatch.setattr(
        _driver, "_qualify_version", lambda: mock.Mock(triton_version="3.6.0")
    )
    monkeypatch.setattr(_ffi, "_library_file", lambda: None)
    with pytest.raises(UnsupportedFeatureError) as caught:
        with llk.triton.activate(_options()):
            pass
    assert caught.value.code == "native_backend_unavailable"


def test_activation_restores_the_prior_driver_on_success(qualified_activation):
    driver_config = qualified_activation
    sentinel = _SentinelDriver()
    driver_config.set_active(sentinel)
    with llk.triton.activate(_options()) as activation:
        assert driver_config.active is not sentinel
        assert activation.last_launch_report() is None
    assert driver_config.active is sentinel


def test_activation_restores_the_prior_driver_on_error(qualified_activation):
    driver_config = qualified_activation
    sentinel = _SentinelDriver()
    driver_config.set_active(sentinel)
    with pytest.raises(RuntimeError):
        with llk.triton.activate(_options()):
            raise RuntimeError("activation body failed")
    assert driver_config.active is sentinel


def test_nested_activation_restores_the_outer_driver(qualified_activation):
    driver_config = qualified_activation
    sentinel = _SentinelDriver()
    driver_config.set_active(sentinel)
    with llk.triton.activate(_options()):
        outer = driver_config.active
        with llk.triton.activate(_options()):
            assert driver_config.active is not outer
        assert driver_config.active is outer
    assert driver_config.active is sentinel


def test_unqualified_version_is_rejected_before_activation(
    triton_path, tmp_path, monkeypatch
):
    lock = json.loads(
        (Path(llk.triton.__file__).parent / "toolchain-lock.json").read_text(
            encoding="utf-8"
        )
    )
    lock["triton_version"] = "9.9.9"
    altered = tmp_path / "altered-lock.json"
    altered.write_text(json.dumps(lock), encoding="utf-8")
    monkeypatch.setenv("LLK_TRITON_TOOLCHAIN_LOCK", str(altered))
    with pytest.raises(UnsupportedFeatureError) as caught:
        with llk.triton.activate(_options()):
            pass
    assert caught.value.code == "upstream_toolchain_unqualified"


# ---------------------------------------------------------------------------
# Genuine launch lane (requires qualified native-library coexistence).
# ---------------------------------------------------------------------------


def _qualified_lane_or_skip():
    reason = _coexistence_skip_reason()
    if reason is not None:
        pytest.skip(reason)


def test_real_add_kernel_writes_cpu_output_and_exposes_evidence(
    triton_path, tmp_path, torch
):
    _qualified_lane_or_skip()
    module = _load_kernel_module(triton_path, _kernel_path(tmp_path))
    x, y, out = _torch_pair(torch, (32, 64), 1024)
    with llk.triton.activate(_options()):
        with _native_spy() as spy:
            compiled = module.add_kernel[(2,)](x, y, out, 2048, BLOCK=1024)
            report = llk.triton.last_launch_report()
            assert spy.call_count == 1
    torch.testing.assert_close(out, x + y, rtol=0, atol=0)
    assert "tt.addptr" in compiled.asm["ttir"]
    assert "loc(" not in compiled.asm["ttir"]
    assert report is not None
    assert report.entry == "add_kernel"
    assert report.runtime_grid == (2, 1, 1)
    assert report.native_grid == (2, 1, 1)
    assert report.block == (16, 64, 1)
    assert report.native_invocations == 1
    assert report.ttir_hash == hashlib.sha256(
        compiled.asm["ttir"].encode("utf-8")
    ).hexdigest()
    assert report.triton_version == "3.6.0"
    assert report.backend == "reference"
    assert report.native_request_identity


def test_real_multiply_kernel_writes_cpu_output(triton_path, tmp_path, torch):
    _qualified_lane_or_skip()
    module = _load_kernel_module(triton_path, _kernel_path(tmp_path))
    x, y, out = _torch_pair(torch, (32, 64), 1024)
    with llk.triton.activate(_options()):
        compiled = module.mul_kernel[(2,)](x, y, out, 2048, BLOCK=1024)
        report = llk.triton.last_launch_report()
    torch.testing.assert_close(out, x * y, rtol=0, atol=0)
    assert "tt.addptr" in compiled.asm["ttir"]
    assert report is not None and report.entry == "mul_kernel"


def test_pointer_specialized_kernel_has_distinct_native_identities(
    triton_path, tmp_path, torch
):
    _qualified_lane_or_skip()
    module = _load_kernel_module(triton_path, _kernel_path(tmp_path))
    with llk.triton.activate(_options()):
        wide = _torch_pair(torch, (32, 64), 1024)
        module.add_kernel[(2,)](*wide, 2048, BLOCK=1024)
        report_32x64 = llk.triton.last_launch_report()
        tall = _torch_pair(torch, (64, 32), 1024)
        module.add_kernel[(2,)](*tall, 2048, BLOCK=1024)
        report_64x32 = llk.triton.last_launch_report()
        short = _torch_pair(torch, (16, 64), 1024)
        module.add_kernel[(1,)](*short, 1024, BLOCK=1024)
        report_16x64 = llk.triton.last_launch_report()
    torch.testing.assert_close(wide[2], wide[0] + wide[1], rtol=0, atol=0)
    torch.testing.assert_close(tall[2], tall[0] + tall[1], rtol=0, atol=0)
    torch.testing.assert_close(short[2], short[0] + short[1], rtol=0, atol=0)
    assert (
        report_32x64.native_request_identity
        != report_64x32.native_request_identity
    )
    assert (
        report_32x64.native_request_identity
        != report_16x64.native_request_identity
    )
    assert report_16x64.runtime_grid == (1, 1, 1)
    assert report_16x64.native_grid == (1, 1, 1)
    assert report_16x64.scalars[0]["value"] == 1024
    assert report_32x64.template_identity == report_64x32.template_identity
    assert report_32x64.template_identity == report_16x64.template_identity


def test_real_launch_accepts_guarded_buffer_providers(triton_path, tmp_path):
    _qualified_lane_or_skip()
    module = _load_kernel_module(triton_path, _kernel_path(tmp_path))
    values = [float(index % 17) - 8.0 for index in range(32 * 64)]
    other = [float((index * 3) % 11) + 0.5 for index in range(32 * 64)]
    x = _Guarded((32, 64), values)
    y = _Guarded((32, 64), other)
    out = _Guarded((32, 64))
    with llk.triton.activate(_options()):
        module.add_kernel[(2,)](x.view, y.view, out.view, 2048, BLOCK=1024)
        report = llk.triton.last_launch_report()
    assert out.values() == [left + right for left, right in zip(values, other)]
    assert x.guards_intact() and y.guards_intact() and out.guards_intact()
    assert report is not None and report.native_invocations == 1


# ---------------------------------------------------------------------------
# Artifact construction helper (upstream compiler stages without the C API).
# ---------------------------------------------------------------------------


def _compile_artifact(triton_path, tmp_path):
    import triton
    from triton.backends.compiler import GPUTarget
    from triton.compiler.compiler import ASTSource, make_backend

    from llk.triton._driver import register_backend

    path = _kernel_path(tmp_path)
    module = _load_kernel_module(triton_path, path)
    register_backend()
    target = GPUTarget("llk_cpu", "native", 1)
    backend = make_backend(target)
    options = backend.parse_options({})
    attrs = {
        (index,): [["tt.divisibility", constraint["divisibility"]]]
        for index, name in enumerate(SIGNATURE)
        if (constraint := SPECIALIZATION.get(name))
    }
    source = ASTSource(
        fn=module.add_kernel,
        signature=SIGNATURE,
        constexprs={"BLOCK": 64},
        attrs=attrs,
    )
    context = triton._C.libtriton.ir.context()
    triton._C.libtriton.ir.load_dialects(context)
    backend.load_dialects(context)
    ir = source.make_ir(
        target,
        options,
        backend.get_codegen_implementation(options),
        backend.get_module_map(),
        context,
    )
    stages = {}
    backend.add_stages(stages, options, source.language)
    metadata = {"hash": "llk-test", "triton_version": triton.__version__}
    result = ir
    for stage in stages.values():
        result = stage(result, metadata)
    assert isinstance(result, bytes)
    return result
