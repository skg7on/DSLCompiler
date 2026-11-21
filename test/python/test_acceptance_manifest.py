"""PF20: the required Python execution lanes and their enforcement checker.

The acceptance manifest is the single declaration of which lanes must execute.
The checker consumes JUnit results plus per-lane evidence, and must fail on a
zero-executed lane, a skipped required lane, a failed lane, absent native-call
counters and reference results standing in for selected-target execution.
"""


from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import jsonschema

from check_required_python_execution import acceptance_errors, summarize_junit

import pytest

pytestmark = pytest.mark.frontend_unit


ROOT = Path(__file__).resolve().parents[2]
FRONTEND = ROOT / "test" / "Frontend"
MANIFEST = FRONTEND / "acceptance_manifest.json"
SCHEMA = FRONTEND / "acceptance_manifest.schema.json"
CHECKER = Path(__file__).with_name("check_required_python_execution.py")

MARKERS = {
    "frontend_unit",
    "frontend_native",
    "torch_integration",
    "triton_integration",
    "selected_avx2",
}


def _manifest():
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def _healthy_results(manifest):
    return {
        lane["id"]: {"executed": 3, "failed": 0, "skipped": 0}
        for lane in manifest["lanes"]
    }


def _healthy_evidence(manifest):
    evidence = {}
    for lane in manifest["lanes"]:
        item = {"native_invocations": 2 if lane["require_native_invocations"] else 0}
        if lane["require_selected_target"]:
            item["selected_backend"] = "selected-target"
            item["selected_group_count"] = 1
            item["reference_group_count"] = 0
        evidence[lane["id"]] = item
    return evidence


def _write_lane(junit_dir, evidence_dir, lane_id, results, evidence):
    junit_dir.mkdir(parents=True, exist_ok=True)
    evidence_dir.mkdir(parents=True, exist_ok=True)
    junit = junit_dir / f"{lane_id}.xml"
    junit.write_text(
        '<?xml version="1.0" encoding="utf-8"?>\n'
        f'<testsuite name="pytest" tests="{results["executed"]}" '
        f'failures="{results["failed"]}" errors="0" '
        f'skipped="{results["skipped"]}" />\n',
        encoding="utf-8",
    )
    (evidence_dir / f"{lane_id}.json").write_text(
        json.dumps(evidence), encoding="utf-8"
    )


def _write_healthy_lanes(manifest, root, faults=None):
    """Write every lane healthy, then apply ``faults[lane_id]`` overrides."""
    faults = faults or {}
    results = _healthy_results(manifest)
    evidence = _healthy_evidence(manifest)
    results.update(faults.get("results", {}))
    evidence.update(faults.get("evidence", {}))
    missing = set(faults.get("missing", ()))
    for lane in manifest["lanes"]:
        if lane["id"] in missing:
            continue
        _write_lane(
            root / "junit",
            root,
            lane["id"],
            results[lane["id"]],
            evidence[lane["id"]],
        )
    return results, evidence


def _run_checker(root):
    return subprocess.run(
        [
            sys.executable,
            "-I",
            str(CHECKER),
            "--manifest",
            str(MANIFEST),
            "--junit-dir",
            str(root / "junit"),
            "--evidence-dir",
            str(root),
        ],
        capture_output=True,
        text=True,
    )


def test_manifest_matches_the_schema_and_declares_every_marker():
    manifest = _manifest()
    schema = json.loads(SCHEMA.read_text(encoding="utf-8"))
    jsonschema.validate(manifest, schema)
    ids = [lane["id"] for lane in manifest["lanes"]]
    assert len(ids) == len(set(ids))
    assert set(ids) == MARKERS
    for lane in manifest["lanes"]:
        assert lane["required"] is True
        assert lane["minimum_executed"] >= 1
        assert lane["marker_expression"]
        if lane["id"] == "selected_avx2":
            assert lane["require_selected_target"] is True
            assert lane["require_native_invocations"] is True
        if lane["id"] == "frontend_native":
            assert lane["require_native_invocations"] is True
        if lane["id"] == "frontend_unit":
            assert "native_library" not in lane["requires"]
        else:
            assert "native_library" in lane["requires"]


def test_healthy_lanes_have_no_acceptance_errors():
    manifest = _manifest()
    assert acceptance_errors(
        manifest, _healthy_results(manifest), _healthy_evidence(manifest)
    ) == []


def test_zero_execution_fixture_reports_required_execution_missing(tmp_path):
    manifest = _manifest()
    results, evidence = _write_healthy_lanes(
        manifest,
        tmp_path,
        {"results": {"frontend_unit": {"executed": 0, "failed": 0, "skipped": 0}}},
    )
    acceptance_errors_for_zero_execution = acceptance_errors(
        manifest, results, evidence
    )
    assert acceptance_errors_for_zero_execution == ["required_execution_missing"]


def test_selected_reference_fallback_fixture_reports_the_fallback_code(tmp_path):
    manifest = _manifest()
    results, evidence = _write_healthy_lanes(
        manifest,
        tmp_path,
        {"evidence": {"selected_avx2": {"native_invocations": 2,
                                        "selected_backend": "reference"}}},
    )
    acceptance_errors_for_selected_fallback = acceptance_errors(
        manifest, results, evidence
    )
    assert acceptance_errors_for_selected_fallback == ["selected_reference_fallback"]


def test_a_skipped_required_lane_fails_even_when_other_lanes_pass():
    manifest = _manifest()
    results = _healthy_results(manifest)
    results["triton_integration"] = {"executed": 0, "failed": 0, "skipped": 2}
    errors = acceptance_errors(manifest, results, _healthy_evidence(manifest))
    assert errors == ["required_lane_skipped"]


def test_a_failed_required_lane_is_not_accepted():
    manifest = _manifest()
    results = _healthy_results(manifest)
    results["torch_integration"] = {"executed": 2, "failed": 1, "skipped": 0}
    errors = acceptance_errors(manifest, results, _healthy_evidence(manifest))
    assert errors == ["required_execution_failed"]


def test_absent_native_counters_fail_the_native_lane():
    manifest = _manifest()
    evidence = _healthy_evidence(manifest)
    evidence["frontend_native"] = {}
    errors = acceptance_errors(manifest, _healthy_results(manifest), evidence)
    assert errors == ["native_invocation_missing"]

    evidence["frontend_native"] = {"native_invocations": 0}
    errors = acceptance_errors(manifest, _healthy_results(manifest), evidence)
    assert errors == ["native_invocation_missing"]


def test_missing_junit_evidence_is_zero_execution():
    manifest = _manifest()
    results = _healthy_results(manifest)
    del results["frontend_unit"]
    errors = acceptance_errors(manifest, results, _healthy_evidence(manifest))
    assert errors == ["required_execution_missing"]


def test_junit_summary_counts_executed_failed_and_skipped(tmp_path):
    xml = """<?xml version="1.0" encoding="utf-8"?>
<testsuites>
  <testsuite name="pytest" tests="4" failures="1" errors="0" skipped="1">
    <testcase name="a" classname="m" />
    <testcase name="b" classname="m"><failure message="boom" /></testcase>
    <testcase name="c" classname="m"><skipped message="missing" /></testcase>
    <testcase name="d" classname="m" />
  </testsuite>
</testsuites>
"""
    path = tmp_path / "lane.xml"
    path.write_text(xml, encoding="utf-8")
    assert summarize_junit(path) == {"executed": 3, "failed": 1, "skipped": 1}


def test_checker_cli_fails_on_zero_executed_lane(tmp_path):
    manifest = _manifest()
    _write_healthy_lanes(
        manifest,
        tmp_path,
        {"results": {"frontend_unit": {"executed": 0, "failed": 0, "skipped": 0}}},
    )
    completed = _run_checker(tmp_path)
    assert completed.returncode == 1, completed.stdout + completed.stderr
    assert completed.stdout.split() == ["required_execution_missing"]


def test_checker_cli_rejects_selected_reference_fallback(tmp_path):
    manifest = _manifest()
    _write_healthy_lanes(
        manifest,
        tmp_path,
        {"evidence": {"selected_avx2": {"native_invocations": 2,
                                        "selected_backend": "reference"}}},
    )
    completed = _run_checker(tmp_path)
    assert completed.returncode == 1, completed.stdout + completed.stderr
    assert completed.stdout.split() == ["selected_reference_fallback"]


def test_checker_cli_accepts_complete_evidence(tmp_path):
    _write_healthy_lanes(_manifest(), tmp_path)
    completed = _run_checker(tmp_path)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert completed.stdout.strip() == ""
