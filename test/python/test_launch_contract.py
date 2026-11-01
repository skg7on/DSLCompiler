"""PF08: declarative launch, option handling and precompiled-kernel lifetime."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from unittest import mock

import llk
import llk.language as tl
import native_cases as cases
import pytest
from llk import TensorContractError, UnsupportedFeatureError
from llk.frontend import CompileRequest, LaunchSpec, bind_kernel, serialize_request
from llk.language._ast_compiler import compile_kernel
from llk.runtime import _ffi

def spy_on_native_invoke():
    return mock.patch.object(_ffi, "native_invoke", wraps=_ffi.native_invoke)


def test_callable_grid_runs_once_and_launches():
    calls = []

    def grid(metadata):
        calls.append(dict(metadata))
        return LaunchSpec((2, 1, 1), (8, 16, 1))

    args, buffers = cases.arguments()
    cases.add_kernel(*args, grid=grid)
    assert len(calls) == 1
    assert calls[0]["BLOCK_M"] == 8 and calls[0]["BLOCK_N"] == 16
    assert buffers[2].values() == cases.expected(16 * 16)


def test_over_coverage_is_rejected_before_native_entry():
    args, _ = cases.arguments()
    over = LaunchSpec((3, 1, 1), (8, 16, 1))
    with spy_on_native_invoke() as native_invoke_spy:
        with pytest.raises(llk.LLKError) as caught:
            cases.add_kernel(*args, grid=over)
    assert caught.value.code == "invalid_launch_coverage"
    assert native_invoke_spy.call_count == 0


def test_precompilation_accepts_metadata_only_tensor_arguments():
    compiled = cases.add_kernel.compile(
        *(cases.specs() + cases.constexprs()), grid=cases.launch()
    )
    _, buffers = cases.arguments()
    compiled.invoke(*cases.providers(buffers))
    assert buffers[2].values() == cases.expected(16 * 16)
    compiled.close()


def test_precompiled_kernel_uses_the_callers_current_buffers():
    compile_args, compiled_buffers = cases.arguments()
    compiled = cases.add_kernel.compile(*compile_args, grid=cases.launch())
    replacement = (
        cases.Guarded(cases.TILED_SHAPE, [1.0] * (16 * 16)),
        cases.Guarded(cases.TILED_SHAPE, [2.0] * (16 * 16)),
        cases.Guarded(cases.TILED_SHAPE),
    )
    compiled.invoke(*cases.providers(replacement))
    assert replacement[2].values() == [3.0] * (16 * 16)
    assert compiled_buffers[2].values() == [0.0] * (16 * 16)
    compiled.close()


def test_invoke_rejects_wrong_arity_and_shape_before_native_entry():
    args, buffers = cases.arguments()
    compiled = cases.add_kernel.compile(*args, grid=cases.launch())
    with spy_on_native_invoke() as native_invoke_spy:
        with pytest.raises(TensorContractError):
            compiled.invoke((buffers[0].view,), (buffers[2].view,))
        smaller = cases.Guarded((8, 16))
        with pytest.raises(TensorContractError):
            compiled.invoke((smaller.view, buffers[1].view), (buffers[2].view,))
    assert native_invoke_spy.call_count == 0
    compiled.close()


def test_module_level_compile_request_returns_an_invocable_handle():
    ports = cases.specs()
    options = cases.options()
    binding = bind_kernel(
        cases.add,
        ports + cases.constexprs(),
        {},
        cases.launch(),
        options,
    )
    program = compile_kernel(cases.add, binding)
    request = CompileRequest(
        program,
        cases.options(numerical_contract=program.numerical_contract),
    )
    compiled = llk.compile_request(request)
    assert isinstance(compiled, llk.CompiledKernel)
    assert compiled.report.native_request_identity
    assert compiled.report.frontend_identity == serialize_request(request).decode()
    _, buffers = cases.arguments()
    compiled.invoke(*cases.providers(buffers))
    assert buffers[2].values() == cases.expected(16 * 16)
    compiled.close()


def test_jit_returns_a_decorator_with_options():
    decorator = llk.jit(config_root=".", config_files=cases.config_files())
    wrapped = decorator(cases.add)
    assert isinstance(wrapped, llk.JitFunction)
    assert wrapped is not cases.add_kernel


def test_selected_target_never_falls_back_to_reference():
    def selected_kernel(
        a,
        b,
        out,
        M: tl.constexpr,
        N: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
        cols = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
        offsets = rows[:, None] * N + cols[None, :]
        mask = (rows[:, None] < M) & (cols[None, :] < N)
        x = tl.load(a + offsets, mask=mask, other=0.0)
        y = tl.load(b + offsets, mask, 0.0)
        tl.store(out + offsets, x + y, mask=mask)

    kernel = llk.jit(
        config_root=".",
        config_files=cases.config_files(),
        backend="selected-target",
    )(selected_kernel)
    args, buffers = cases.arguments()
    try:
        compiled = kernel.compile(*args, grid=cases.launch())
    except UnsupportedFeatureError as error:
        assert error.code == "native_backend_unavailable"
        assert buffers[2].values() == [0.0] * (16 * 16)
    else:
        assert compiled.report.backend == "selected-target"
        compiled.invoke(*cases.providers(buffers))
        assert buffers[2].values() == cases.expected(16 * 16)
        compiled.close()


def test_default_reference_option_is_reported():
    args, _ = cases.arguments()
    compiled = cases.add_kernel.compile(*args, grid=cases.launch())
    assert compiled.report.backend == "reference"
    compiled.close()


def test_native_compile_failure_is_marshalled_with_its_code_and_stage():
    args, _ = cases.arguments(cases.FULL_SHAPE, cases.TILED_BLOCK)
    with pytest.raises(llk.CompilationError) as caught:
        cases.add_kernel.compile(
            *args, grid=cases.launch(cases.FULL_SHAPE, cases.TILED_BLOCK)
        )
    assert caught.value.code == "frontend_compilation_failed"
    assert caught.value.stage == "compile"
    assert "memory_capacity_exceeded" in caught.value.message


def test_missing_explicit_library_reports_unavailable_without_fallback(tmp_path):
    root = Path(__file__).resolve().parents[2]
    environment = dict(os.environ)
    environment["LLK_FRONTEND_LIBRARY"] = str(tmp_path / "missing.dylib")
    environment["PYTHONPATH"] = str(root / "python")
    script = (
        "import llk\n"
        "from llk.runtime import _ffi\n"
        "try:\n"
        "    _ffi.library()\n"
        "except llk.UnsupportedFeatureError as error:\n"
        "    print(error.code)\n"
        "else:\n"
        "    raise SystemExit('library unexpectedly loaded')\n"
    )
    completed = subprocess.run(
        [sys.executable, "-c", script],
        env=environment,
        capture_output=True,
        text=True,
        check=True,
    )
    assert completed.stdout.strip() == "native_backend_unavailable"
