"""PF19: the fixed upstream Triton compatibility corpus and its boundaries.

The corpus is declared in ``docs/design/triton-compatibility-v1.json``: ten
genuine ``@triton.jit`` kernels with source attribution together with the six
constructs that must be rejected by name. Execution happens in a dedicated
interpreter because it imports the qualified upstream Triton package; the
native compilation and invocation run in a Triton-free worker so this host's
LLVM coexistence limitation cannot weaken the evidence.
"""


from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import llk
from _acceptance import record as record_acceptance

import pytest

pytestmark = [pytest.mark.triton_integration, pytest.mark.selected_avx2]


ROOT = Path(__file__).resolve().parents[3]
MANIFEST = ROOT / "docs/design/triton-compatibility-v1.json"
LANE = Path(__file__).with_name("_triton_corpus_lane.py")
MARKER = "LLK-TRITON-CORPUS-V1"
REQUIRED_CONSTRUCTS = {
    "atomic",
    "shared_async",
    "reduce",
    "runtime_while",
    "runtime_stride",
    "gpu_only",
}


def _manifest():
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def test_manifest_declares_ten_source_attributed_fixtures():
    manifest = _manifest()
    assert manifest["schema_version"] == 1
    assert manifest["kind"] == "llk-triton-compatibility-corpus"
    assert manifest["toolchain"]["triton_version"] == "3.6.0"
    fixtures = manifest["fixtures"]
    assert len(fixtures) == 10
    assert len({fixture["id"] for fixture in fixtures}) == 10
    for fixture in fixtures:
        source = ROOT / fixture["source"]
        assert source.is_file(), fixture["source"]
        text = source.read_text(encoding="utf-8")
        assert f"def {fixture['function']}(" in text
        assert fixture["grid"] and len(fixture["grid"]) == 3
        assert set(fixture["tensors"]).issubset(set(fixture["signature"]))
    operations = {fixture["operation"] for fixture in fixtures}
    assert {"add", "multiply", "matmul", "matmul_accumulated", "swiglu"} <= (
        operations
    )
    # Both elementwise operations carry a masked-tail case, and add carries the
    # flattened-index case over rank-two storage.
    tails = [fixture for fixture in fixtures if fixture.get("tail")]
    assert len(tails) == 2
    assert {fixture["operation"] for fixture in tails} == {"add", "multiply"}
    flattened = [fixture for fixture in fixtures if fixture["id"].endswith("flattened")]
    assert len(flattened) == 1

    rejections = manifest["rejections"]
    assert {item["construct"] for item in rejections} == REQUIRED_CONSTRUCTS
    for rejection in rejections:
        source = ROOT / rejection["source"]
        assert source.is_file(), rejection["source"]
        assert f"def {rejection['function']}(" in source.read_text(encoding="utf-8")
        assert rejection["expected_code"]


def _run_lane(triton_path, tmp_path):
    llk_python = str(Path(llk.__file__).resolve().parent.parent)
    environment = dict(os.environ)
    environment["LLK_TRITON_PYTHONPATH"] = str(triton_path)
    completed = subprocess.run(
        [
            sys.executable,
            "-I",
            str(LANE),
            "--llk-python",
            llk_python,
            "--root",
            str(ROOT),
            "--manifest",
            str(MANIFEST),
            "--triton-python",
            str(triton_path),
            "--workdir",
            str(tmp_path / "corpus"),
        ],
        capture_output=True,
        text=True,
        timeout=3600,
        env=environment,
    )
    output = completed.stdout + completed.stderr
    assert completed.returncode == 0, output[-8000:]
    lines = completed.stdout.split("\n", 1)
    assert lines[0] == MARKER, output[-8000:]
    frame = json.loads(lines[1])
    assert frame.get("ok") is not False, frame.get("message")
    return frame


def test_corpus_executes_ten_fixtures_natively_and_rejects_the_boundaries(
    triton_path, tmp_path
):
    frame = _run_lane(triton_path, tmp_path)
    manifest = _manifest()
    expected_ids = [fixture["id"] for fixture in manifest["fixtures"]]

    executed_fixture_ids = frame["executed_fixture_ids"]
    executed_fixture_reports = [
        record for record in frame["fixtures"] if record["class"] == "executable"
    ]
    reference_invocations = sum(
        int(item["reference"]["native_invocations"])
        for item in executed_fixture_reports
    )
    selected_records = [item["selected"] for item in executed_fixture_reports]
    if frame["host"]["selected_target_supported"]:
        selected_ok = all(
            item["status"] == "ok" and item["backend"] == "selected-target"
            for item in selected_records
        )
        record_acceptance(
            native_invocations=reference_invocations
            + sum(int(item.get("native_invocations", 0)) for item in selected_records),
            selected_backend="selected-target" if selected_ok else "failed",
            selected_group_count=sum(
                int(item.get("selected_group_count", 0)) for item in selected_records
            ),
            reference_group_count=sum(
                int(item.get("reference_group_count", 0)) for item in selected_records
            ),
        )
    else:
        # Natives ran in the Triton-free worker; this in-process lane can see
        # them only through the frame, so record the worker's count.
        record_acceptance(
            native_invocations=reference_invocations,
            selected_backend="unavailable",
            selected_group_count=0,
            reference_group_count=0,
        )

    assert len(executed_fixture_ids) == 10
    assert all(
        item["reference"]["native_invocations"] > 0
        for item in executed_fixture_reports
    )
    assert executed_fixture_ids == expected_ids

    for record in executed_fixture_reports:
        assert record["frontend"]["status"] == "ok"
        assert record["binding"]["status"] == "ok"
        assert record["frontend"]["toolchain"]["triton_version"] == "3.6.0"
        reference = record["reference"]
        assert reference["status"] == "ok"
        assert reference["backend"] == "reference"
        assert reference["native_invocations"] == 1
        for name in (
            "plan_id",
            "abi_hash",
            "native_request_identity",
            "frontend_identity",
            "compiler_identity",
            "workload_identity",
            "source_adapter_identity",
            "target_hash",
            "machine_hash",
            "rules_hash",
        ):
            assert reference[name], (record["id"], name)
        assert reference["source_adapter"]
        # The lane applies the declared combined tolerance and fails on any
        # violation; the recorded error is the observed numerical evidence.
        assert reference["numerical"]["max_abs_error"] < 1e-2
        selected = record["selected"]
        if frame["host"]["selected_target_supported"]:
            assert selected["status"] == "ok", record["id"]
            assert selected["backend"] == "selected-target"
            assert selected["reference_group_count"] == 0
            assert selected["selected_group_count"] > 0
        else:
            assert selected["status"] == "unavailable"
            assert selected["reason"] == "host_isa_unsupported"

    by_construct = {item["construct"]: item for item in frame["rejections"]}
    assert set(by_construct) == REQUIRED_CONSTRUCTS
    expected_codes = {
        item["construct"]: item["expected_code"] for item in manifest["rejections"]
    }
    for construct, record in by_construct.items():
        assert record["status"] == "rejected", (construct, record)
        assert record["code"] == expected_codes[construct], (construct, record)
