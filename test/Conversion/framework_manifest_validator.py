#!/usr/bin/env python3
"""Validate the versioned framework selected-evidence manifest.

Every row must name its source, plan, target, compiler and mode, carry strict
analysis, and state each compiler stage as passed, not checked, or unsupported
with a concrete reason. A missing row field or an unsupported claim without a
reason is a hard failure, so the acceptance report cannot omit a limitation.
"""

import json
import pathlib
import sys

STATUSES = {"supported", "unsupported", "not_checked"}
REQUIRED_STAGES = ("mapping", "physical_plan", "selected_lowering",
                   "reference_lowering", "invocation")
REQUIRED_FIELDS = ("source", "symbol", "target", "mode", "compiler",
                   "replay_identity", "plan_id", "counters", "stages",
                   "analysis")


def stage_verdict(stage):
    """Returns (status, reason) for a stage value."""
    if isinstance(stage, str):
        return stage, ""
    if isinstance(stage, dict):
        return stage.get("status", ""), stage.get("reason", "")
    return "", ""


def validate_row(row, index):
    errors = []
    prefix = f"row {index} ({row.get('symbol', '<unknown>')})"
    for field in REQUIRED_FIELDS:
        if field not in row or not row[field]:
            errors.append(f"{prefix}: missing or empty '{field}'")
    if errors:
        return errors

    for key in ("name", "machine"):
        if not row["target"].get(key):
            errors.append(f"{prefix}: target.{key} is empty")
    for key in ("backend", "stop"):
        if not row["mode"].get(key):
            errors.append(f"{prefix}: mode.{key} is empty")

    stages = row["stages"]
    for name in REQUIRED_STAGES:
        if name not in stages:
            errors.append(f"{prefix}: missing stage '{name}'")
            continue
        status, reason = stage_verdict(stages[name])
        if status not in STATUSES:
            errors.append(
                f"{prefix}: stage '{name}' has unknown status '{status}'")
        if status == "unsupported" and not reason:
            errors.append(
                f"{prefix}: stage '{name}' is unsupported without a reason")

    analysis = row["analysis"]
    if analysis.get("format") != "micro-perf-yaml":
        errors.append(f"{prefix}: analysis format is not micro-perf-yaml")
    yaml = analysis.get("yaml", "")
    if "complete: true" not in yaml:
        errors.append(f"{prefix}: analysis is not strictly complete")
    if "bytes:" not in yaml or "live_bytes:" not in yaml:
        errors.append(f"{prefix}: analysis omits traffic/storage")

    selected, _ = stage_verdict(stages.get("selected_lowering", ""))
    reference, _ = stage_verdict(stages.get("reference_lowering", ""))
    if selected != "supported":
        errors.append(f"{prefix}: selected lowering is not supported")
    counters = row.get("counters", {})
    if counters.get("target_lowered_ops", 0) <= 0:
        errors.append(f"{prefix}: no target-lowered operations")
    if counters.get("reference_lowered_ops") != 0:
        errors.append(f"{prefix}: selected run lowered operations by reference")
    if counters.get("reference_groups_lowered") != 0:
        errors.append(f"{prefix}: a selected group fell back to reference")
    if reference == "supported" and counters.get("reference_lowered_ops") != 0:
        errors.append(
            f"{prefix}: reference lowering is claimed without zero-fallback "
            "evidence")
    return errors


def main(manifest_directory):
    directory = pathlib.Path(manifest_directory)
    manifests = sorted(directory.glob("*.manifest.json"))
    if not manifests:
        print(f"manifest: no manifest found in {directory}")
        return 1
    errors = []
    rows = 0
    for path in manifests:
        try:
            manifest = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            errors.append(f"{path.name}: cannot read: {error}")
            continue
        if manifest.get("schema_version") != 1:
            errors.append(f"{path.name}: unsupported schema version")
            continue
        if manifest.get("kind") != "framework-selected-evidence":
            errors.append(f"{path.name}: unexpected manifest kind")
        manifest_rows = manifest.get("rows", [])
        if not manifest_rows:
            errors.append(f"{path.name}: no rows")
        for index, row in enumerate(manifest_rows):
            rows += 1
            errors.extend(validate_row(row, index))

    if errors:
        for error in errors:
            print(f"manifest: {error}")
        print(f"framework manifest: {len(errors)} violation(s)")
        return 1
    print(f"framework manifest: {rows} row(s) validated")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "."))
