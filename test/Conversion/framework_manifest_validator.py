#!/usr/bin/env python3
"""Validate complete, artifact-associated C3 selected-lowering evidence.

All three primitive cases must occur exactly once. Static portability/resource
rows have their own CTests; this manifest records selected lowering, not proof
of reference execution or numerical invocation.
"""

import json
import pathlib
import re
import sys

from framework_perf_report import validate_analysis

STATUSES = {"supported", "unsupported", "not_checked"}
REQUIRED_STAGES = ("mapping", "physical_plan", "selected_lowering",
                   "reference_lowering", "invocation")
REQUIRED_FIELDS = ("source", "symbol", "target", "mode", "compiler",
                   "replay_identity", "plan_id", "counters", "stages",
                   "analysis", "output_shape", "numeric")
CASES = {
    "reduce_sum": ("reduce.mlir", 1, [8]),
    "add_mul": ("add_mul.mlir", 2, [8, 8]),
    "add_mul_reduce": ("add_mul_reduce.mlir", 3, [8]),
}
TARGET = {"name": "x86-avx2", "machine": "machines/x86-avx2-v2.yaml"}
MODE = {"backend": "selected-target", "stop": "lowered"}


def stage_verdict(stage):
    if isinstance(stage, str):
        return stage, ""
    if isinstance(stage, dict):
        return stage.get("status", ""), stage.get("reason", "")
    return "", ""


def identity_fields(identity):
    if not isinstance(identity, str):
        raise ValueError("identity must be text")
    fields = {}
    for part in identity.split("|"):
        key, separator, value = part.partition("=")
        if not separator or not key or not value or key in fields:
            raise ValueError("malformed or duplicate identity field")
        fields[key] = value
    return fields


def validate_row(row, index):
    prefix = f"row {index}"
    if not isinstance(row, dict):
        return [f"{prefix}: row must be an object"]
    errors = []
    prefix += f" ({row.get('symbol', '<unknown>')})"
    for field in REQUIRED_FIELDS:
        if field not in row or not row[field]:
            errors.append(f"{prefix}: missing or empty '{field}'")
    for field in ("target", "mode", "counters", "stages", "analysis", "numeric"):
        if not isinstance(row.get(field), dict):
            errors.append(f"{prefix}: '{field}' must be an object")
    if errors:
        return errors

    symbol = row["symbol"]
    if not isinstance(symbol, str) or symbol not in CASES:
        return [f"{prefix}: unexpected framework case"]
    filename, op_count, shape = CASES[symbol]
    if row["source"] != "test/Execution/Inputs/framework/" + filename:
        errors.append(f"{prefix}: source does not match case")
    if row["target"] != TARGET or row["mode"] != MODE:
        errors.append(f"{prefix}: target/mode does not match selected-lowering case")
    if not isinstance(row["output_shape"], list) or \
            any(type(extent) is not int for extent in row["output_shape"]) or \
            row["output_shape"] != shape:
        errors.append(f"{prefix}: output shape does not match case")

    try:
        replay = identity_fields(row["replay_identity"])
        if set(replay) != {"graph", "binding", "plan"} or any(
                not re.fullmatch(r"0x[0-9a-fA-F]+", value)
                for value in replay.values()) or replay["plan"] != row["plan_id"]:
            raise ValueError("plan id does not match replay identity")
        compiler = identity_fields(row["compiler"])
        if any(not compiler.get(key) for key in (
                "compiler", "machine-content", "cpu", "math-mode")) or \
                compiler.get("backend") != MODE["backend"] or \
                compiler.get("target") != TARGET["name"] or \
                compiler.get("architecture") != "x86_64" or \
                "avx2" not in compiler.get("features", "").split(","):
            raise ValueError("compiler identity does not match selected target/mode")
        analyzed_compiler = identity_fields(row["analysis"].get("compiler"))
        if {key: value for key, value in compiler.items() if key != "math-mode"} != \
                {key: value for key, value in analyzed_compiler.items()
                 if key != "math-mode"} or (
                    "math-mode" in analyzed_compiler and
                    analyzed_compiler["math-mode"] != compiler["math-mode"]):
            raise ValueError("analyzed artifact has different compiler/target identity")
        for key in ("plan_id", "replay_identity"):
            if row["analysis"].get(key) != row[key]:
                raise ValueError(f"analyzed artifact has different {key}")
    except ValueError as error:
        errors.append(f"{prefix}: {error}")

    stages = row["stages"]
    for name in REQUIRED_STAGES:
        if name not in stages:
            errors.append(f"{prefix}: missing stage '{name}'")
    for name, stage in stages.items():
        status, reason = stage_verdict(stage)
        if not isinstance(status, str) or status not in STATUSES:
            errors.append(f"{prefix}: stage '{name}' has unknown status")
        if status == "unsupported" and (not isinstance(reason, str) or
                                         not reason.strip()):
            errors.append(f"{prefix}: stage '{name}' is unsupported without a reason")
    for name in ("mapping", "physical_plan", "selected_lowering"):
        if stage_verdict(stages.get(name))[0] != "supported":
            errors.append(f"{prefix}: {name} is not supported")
    if stage_verdict(stages.get("reference_lowering"))[0] != "not_checked":
        errors.append(f"{prefix}: selected-only artifact cannot prove reference lowering")
    invocation, reason = stage_verdict(stages.get("invocation"))
    if len(shape) == 1:
        if invocation != "unsupported" or not isinstance(reason, str) or \
                "rank-two" not in reason:
            errors.append(f"{prefix}: rank-one invocation must name its ABI blocker")
    elif invocation != "not_checked":
        errors.append(f"{prefix}: lowering artifact cannot prove invocation")
    numeric = row["numeric"]
    if numeric.get("status") != "not_checked" or "verified_by" in numeric:
        errors.append(f"{prefix}: lowering artifact cannot prove numerical execution")
    expected_elements = 8 if len(shape) == 1 else 64
    if type(numeric.get("elements")) is not int or \
            numeric["elements"] != expected_elements:
        errors.append(f"{prefix}: numerical output extent does not match case")

    counters = row["counters"]
    counters_valid = True
    for key in ("target_lowered_ops", "selected_groups_verified",
                "backend_groups_realized", "reference_lowered_ops",
                "reference_groups_lowered"):
        if type(counters.get(key)) is not int or counters[key] < 0:
            errors.append(f"{prefix}: counter '{key}' must be unsigned")
            counters_valid = False
    if counters_valid:
        if counters["target_lowered_ops"] != op_count:
            errors.append(f"{prefix}: not every case operation was target-lowered")
        if counters["selected_groups_verified"] <= 0 or \
                counters["backend_groups_realized"] != counters["selected_groups_verified"]:
            errors.append(f"{prefix}: selected groups were not verified and realized")
        if counters["reference_lowered_ops"] or counters["reference_groups_lowered"]:
            errors.append(f"{prefix}: selected work fell back to reference")

    analysis = row["analysis"]
    try:
        if analysis.get("format") != "micro-perf-yaml":
            raise ValueError("analysis format is not micro-perf-yaml")
        validate_analysis(analysis.get("yaml"), symbol, TARGET["name"])
    except ValueError as error:
        errors.append(f"{prefix}: {error}")
    return errors


def main(manifest_directory):
    manifests = sorted(pathlib.Path(manifest_directory).glob("*.manifest.json"))
    errors = []
    seen = set()
    if not manifests:
        errors.append("no manifest found")
    for path in manifests:
        try:
            manifest = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            errors.append(f"{path.name}: cannot read: {error}")
            continue
        if not isinstance(manifest, dict) or \
                type(manifest.get("schema_version")) is not int or \
                manifest["schema_version"] != 1:
            errors.append(f"{path.name}: unsupported manifest/schema version")
            continue
        if manifest.get("kind") != "framework-selected-evidence":
            errors.append(f"{path.name}: unexpected manifest kind")
        rows = manifest.get("rows")
        if not isinstance(rows, list) or not rows:
            errors.append(f"{path.name}: rows must be a nonempty array")
            continue
        for index, row in enumerate(rows):
            errors.extend(validate_row(row, index))
            if isinstance(row, dict) and isinstance(row.get("symbol"), str):
                symbol = row["symbol"]
                if symbol in seen:
                    errors.append(f"{path.name}: duplicate case '{symbol}'")
                seen.add(symbol)
    for symbol in sorted(CASES.keys() - seen):
        errors.append(f"missing required case '{symbol}'")
    if errors:
        for error in errors:
            print(f"manifest: {error}")
        print(f"framework manifest: {len(errors)} violation(s)")
        return 1
    print(f"framework manifest: {len(seen)} unique case(s) validated")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "."))
