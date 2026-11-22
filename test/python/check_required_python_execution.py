#!/usr/bin/env python3
"""Require every declared Python frontend lane to execute for real.

``test/Frontend/acceptance_manifest.json`` declares the required pytest lanes.
Each lane is run with ``--junitxml`` and writes one evidence JSON file (the
native-call counter and, for the selected lane, the executed backend). This
checker joins the two and fails when a required lane collected or executed
nothing, skipped or failed, omitted its native-call counter, or reported
reference results for selected-target execution.

The JUnit and evidence directories mirror the lane ids::

    junit/<lane>.xml
    <evidence-dir>/<lane>.json

The command prints one failure code per line and exits 1 when any code is
present. Codes are stable identifiers so CI can assert exactly which contract
was violated.
"""

from __future__ import annotations

import argparse
import json
import sys
import xml.etree.ElementTree as ElementTree
from pathlib import Path

REQUIRED_EXECUTION_MISSING = "required_execution_missing"
REQUIRED_LANE_SKIPPED = "required_lane_skipped"
REQUIRED_EXECUTION_FAILED = "required_execution_failed"
NATIVE_INVOCATION_MISSING = "native_invocation_missing"
SELECTED_REFERENCE_FALLBACK = "selected_reference_fallback"

ALL_CODES = (
    REQUIRED_EXECUTION_MISSING,
    REQUIRED_LANE_SKIPPED,
    REQUIRED_EXECUTION_FAILED,
    NATIVE_INVOCATION_MISSING,
    SELECTED_REFERENCE_FALLBACK,
)


def _count(attribute, element):
    try:
        return int(element.get(attribute, "0"))
    except (TypeError, ValueError):
        return 0


def summarize_junit(path):
    """Return executed/failed/skipped counts from one JUnit XML document.

    ``executed`` excludes skipped cases; errors count as failures. A missing or
    unreadable document is treated as zero execution by the caller.
    """
    path = Path(path)
    try:
        root = ElementTree.parse(path).getroot()
    except (OSError, ElementTree.ParseError):
        return {"executed": 0, "failed": 0, "skipped": 0}
    suites = (
        [root]
        if root.tag == "testsuite"
        else [element for element in root.iter("testsuite")]
    )
    tests = failures = errors = skipped = 0
    if suites:
        for suite in suites:
            tests += _count("tests", suite)
            failures += _count("failures", suite)
            errors += _count("errors", suite)
            skipped += _count("skipped", suite)
    else:
        for case in root.iter("testcase"):
            tests += 1
            if case.find("skipped") is not None:
                skipped += 1
            elif case.find("failure") is not None or case.find("error") is not None:
                failures += 1
    return {
        "executed": max(tests - skipped, 0),
        "failed": failures + errors,
        "skipped": skipped,
    }


def collect_results(manifest, junit_dir, evidence_dir):
    """Load JUnit results and evidence for every lane in ``manifest``."""
    junit_dir = Path(junit_dir)
    evidence_dir = Path(evidence_dir)
    results = {}
    evidence = {}
    for lane in manifest["lanes"]:
        lane_id = lane["id"]
        junit = junit_dir / f"{lane_id}.xml"
        if junit.is_file():
            results[lane_id] = summarize_junit(junit)
        evidence_file = evidence_dir / f"{lane_id}.json"
        if evidence_file.is_file():
            try:
                loaded = json.loads(evidence_file.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError):
                loaded = {}
            evidence[lane_id] = loaded if isinstance(loaded, dict) else {}
    return results, evidence


def _positive_count(value):
    try:
        return int(value) > 0
    except (TypeError, ValueError):
        return False


def acceptance_errors(manifest, results, evidence):
    """Return the sorted failure codes for the declared required lanes."""
    errors = set()
    for lane in manifest["lanes"]:
        if not lane.get("required", True):
            continue
        lane_id = lane["id"]
        result = results.get(lane_id)
        evidence_item = evidence.get(lane_id) or {}
        if result is None:
            errors.add(REQUIRED_EXECUTION_MISSING)
            continue
        executed = int(result.get("executed", 0))
        failed = int(result.get("failed", 0))
        skipped = int(result.get("skipped", 0))
        if executed < int(lane.get("minimum_executed", 1)):
            errors.add(
                REQUIRED_LANE_SKIPPED if skipped > 0 else REQUIRED_EXECUTION_MISSING
            )
            continue
        if skipped > 0:
            errors.add(REQUIRED_LANE_SKIPPED)
        if failed > 0:
            errors.add(REQUIRED_EXECUTION_FAILED)
        if lane.get("require_native_invocations") and not _positive_count(
            evidence_item.get("native_invocations")
        ):
            errors.add(NATIVE_INVOCATION_MISSING)
        if lane.get("require_selected_target"):
            selected_ok = (
                evidence_item.get("selected_backend") == "selected-target"
                and int(evidence_item.get("reference_group_count", -1)) == 0
                and _positive_count(evidence_item.get("selected_group_count"))
            )
            if not selected_ok:
                errors.add(SELECTED_REFERENCE_FALLBACK)
    return sorted(errors, key=ALL_CODES.index)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--junit-dir", required=True, type=Path)
    parser.add_argument("--evidence-dir", required=True, type=Path)
    parser.add_argument("--json", type=Path, default=None,
                        help="also write the result summary as JSON")
    args = parser.parse_args(argv)

    try:
        manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        print(f"cannot read acceptance manifest: {error}", file=sys.stderr)
        return 2
    if not isinstance(manifest, dict) or not isinstance(manifest.get("lanes"), list):
        print("acceptance manifest requires a lanes array", file=sys.stderr)
        return 2

    results, evidence = collect_results(manifest, args.junit_dir, args.evidence_dir)
    errors = acceptance_errors(manifest, results, evidence)
    if args.json:
        args.json.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "errors": errors,
                    "results": results,
                    "evidence": evidence,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
    for code in errors:
        print(code)
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
