"""PF09: reports and inspection come from the actual native artifact."""

from __future__ import annotations

import json

import llk
import native_cases as cases
import pytest
from llk import ExecutionError, UnsupportedFeatureError


def fresh_kernel():
    return llk.jit(**{"config_root": ".", "config_files": cases.config_files()})(
        cases.add
    )


def compiled_kernel():
    kernel = fresh_kernel()
    args, buffers = cases.arguments()
    return kernel, kernel.compile(*args, grid=cases.launch()), buffers


def test_inspection_text_is_the_artifact_compiled_for_the_report():
    kernel, compiled, _ = compiled_kernel()
    try:
        report = compiled.report
        request = json.loads(compiled.frontend_ir)
        assert set(request) == {"program", "options"}
        assert request["program"]["entry"] == "add"
        assert compiled.frontend_ir == report.frontend_identity
        assert "llk.frontend.request" in compiled.llkir
        assert "micro.kernel" in compiled.micro_ir
        assert report.target_hash.removeprefix("0x") in compiled.micro_ir

        native = json.loads(report.native_request_identity)
        assert native["abi"] == report.abi_hash
        assert native["target"] == report.target_hash
        assert report.plan_id
    finally:
        compiled.close()


def test_report_exposes_reached_stages_and_native_counters():
    _, compiled, _ = compiled_kernel()
    try:
        report = compiled.report
        assert report.stages == (
            "frontend",
            "llk",
            "micro",
            "lowered",
            "executable",
        )
        assert report.inspection_stages == ("frontend", "llk", "micro")
        assert isinstance(report.native_counters["generated"], int)
        assert report.native_counters["generated"] >= 1
        assert report.native_counters["max_candidates"] == 64
        assert report.native_counters["mode"] == "deterministic"
        assert report.native_environment["compiler"] == report.compiler_identity
        assert report.native_environment["host_triple"]
    finally:
        compiled.close()


def test_unavailable_disassembly_reports_inspection_unavailable():
    _, compiled, _ = compiled_kernel()
    try:
        with pytest.raises(UnsupportedFeatureError) as caught:
            compiled.asm
        assert caught.value.code == "inspection_unavailable"
        with pytest.raises(UnsupportedFeatureError) as caught:
            compiled.inspect("asan")
        assert caught.value.code == "inspection_unavailable"
    finally:
        compiled.close()


def test_inspection_after_close_raises_invalid_frontend_handle():
    _, compiled, _ = compiled_kernel()
    compiled.close()
    with pytest.raises(ExecutionError) as caught:
        compiled.frontend_ir
    assert caught.value.code == "invalid_frontend_handle"
    with pytest.raises(ExecutionError) as caught:
        compiled.asm
    assert caught.value.code == "invalid_frontend_handle"


def test_a_cache_hit_never_reports_an_uncompiled_artifact():
    kernel = fresh_kernel()
    args, _ = cases.arguments()
    first = kernel.compile(*args, grid=cases.launch())
    try:
        second = kernel.compile(*args, grid=cases.launch())
        try:
            assert kernel.cache_info().compilations == 1
            assert (
                second.report.native_request_identity
                == first.report.native_request_identity
            )
            assert second.frontend_ir == first.frontend_ir
        finally:
            second.close()
    finally:
        first.close()
