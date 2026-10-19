#!/usr/bin/env python3
# Part of the LLK Compiler Project, under the Apache License v2.0 with LLVM Exceptions.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Deterministic record/report and holdout boundary tests; no JIT required."""
import copy
import json
import pathlib
import subprocess
import sys
import tempfile


def content_hash(identity):
    value = 14695981039346656037
    for byte in identity.encode():
        value = ((value ^ byte) * 1099511628211) & ((1 << 64) - 1)
    return str(value)


def record(identity, measured, predicted=100):
    return {
        "version": 1, "executable_identity": identity,
        "executable_hash": content_hash(identity),
        "target_identity": "synthetic-target-complete-content-v1",
        "backend_identity": "synthetic-selected-backend-complete-content-v1",
        "environment": {"architecture": "synthetic", "cpu_identity": "fixture-cpu",
                        "isa": "fixture-isa", "threads": 1, "affinity": "fixture",
                        "clock_policy": "deterministic-fixture"},
        "protocol": {"version": 1, "warmup": 2, "repeat": 3,
                     "timer": "steady_clock", "statistic": "median", "scope": "per_invocation"},
        "verification": {"fixture_content": "complete-arithmetic-test-fixture",
                         "verifier_version": "fixture-v1", "numerical_contract": "strict-f32"},
        "verified": True, "samples_ns": [measured] * 3, "median_ns": measured,
        "prediction": {"predicted_cycles": 300, "predicted_ns": predicted},
        "source_adapter": "synthetic-statistics-fixture-v1",
        "workload_contract": "fixture-contract:f32:shape=[8,16]",
        "candidate_bindings": "T0=4,T1=8", "bottleneck": "fixture-compute",
        "timestamp_utc": "2026-10-19T00:00:00Z", "compiler_version": "fixture-compiler-v1"}


def main():
    validate = pathlib.Path(sys.argv[1]).resolve()
    with tempfile.TemporaryDirectory(prefix="llk-validation-") as temporary:
        root = pathlib.Path(temporary)
        records = root / "records"
        records.mkdir()

        def write(name, value):
            path = root / name
            path.write_text(json.dumps(value))
            return path

        a = record("complete-fixture-identity-a", 80)
        b = record("complete-fixture-identity-b", 200)
        a_path = write("records/a.json", a)
        b_path = write("records/b.json", b)
        holdout = write("holdout.json", {"version": 1, "identities": [a["executable_identity"], b["executable_identity"]]})

        def run(*arguments, success=True):
            result = subprocess.run([str(validate), *map(str, arguments)], text=True, capture_output=True)
            assert (result.returncode == 0) == success, result.stdout + result.stderr
            return result

        report = json.loads(run(records, f"--holdout-manifest={holdout}").stdout)
        assert report["held_out"] is True
        assert report["calibrated"] is False
        assert report["record_count"] == 2
        assert len(report["groups"]) == 1
        group = report["groups"][0]
        assert group["median_absolute_relative_error"] == 0.375, group
        assert group["p90_absolute_relative_error"] == 0.5, group
        assert [r["signed_relative_error"] for r in group["rows"]] == [0.25, -0.5]
        assert group["regime"]["workload_contract"] == a["workload_contract"]
        assert group["regime"]["protocol"]["repeat"] == 3
        assert "median_absolute_relative_error=0.375" in run(records, "--format=text").stdout
        assert json.loads(run(a_path, b_path).stdout)["held_out"] is False
        # Mixed full target/host/backend/protocol/metadata form separate groups.
        for field in ("target_identity", "backend_identity", "compiler_version",
                      "source_adapter", "workload_contract", "candidate_bindings", "bottleneck"):
            changed = copy.deepcopy(b)
            changed[field] += "-different"
            b_path.write_text(json.dumps(changed))
            assert len(json.loads(run(records).stdout)["groups"]) == 2, field
        for field in ("cpu_identity", "affinity", "clock_policy"):
            changed = copy.deepcopy(b)
            changed["environment"][field] += "-different"
            b_path.write_text(json.dumps(changed))
            assert len(json.loads(run(records).stdout)["groups"]) == 2, field
        changed = copy.deepcopy(b)
        changed["protocol"]["warmup"] += 1
        b_path.write_text(json.dumps(changed))
        assert len(json.loads(run(records).stdout)["groups"]) == 2
        b_path.write_text(json.dumps(b))
        # Unknown time conversion is explicit, never a zero-error observation.
        missing = record("unknown-clock", 80, 0)
        missing_path = write("missing.json", missing)
        excluded = json.loads(run(missing_path).stdout)["groups"][0]
        assert excluded["status"] == "no_comparable_samples"
        assert excluded["median_absolute_relative_error"] is None
        assert excluded["p90_absolute_relative_error"] is None
        assert "no_prediction_time_basis" in excluded["excluded"][0]
        empty = root / "empty"
        empty.mkdir()
        no_samples = json.loads(run(empty).stdout)
        assert no_samples["status"] == "no_comparable_samples"
        # A duplicate path/record and any corrupt verified structure reject.
        assert "duplicate" in run(a_path, a_path, success=False).stderr
        duplicate = write("duplicate.json", a)
        assert "duplicate" in run(a_path, duplicate, success=False).stderr
        corrupt = root / "corrupt.json"
        corrupt.write_text('{"version":')
        assert "corrupt.json" in run(corrupt, success=False).stderr
        for bad in (float("inf"), float("nan"), -1):
            invalid = record("bad", 80, bad)
            corrupt.write_text(json.dumps(invalid))
            run(corrupt, success=False)
        # Full identity manifests cannot overlap or duplicate entries.
        training = write("training.json", {"version": 1, "identities": [a["executable_identity"]]})
        for option in ("training-manifest", "calibration-manifest"):
            result = run(records, f"--holdout-manifest={holdout}", f"--{option}={training}", success=False)
            assert "overlap" in result.stderr
        write("holdout.json", {"version": 1, "identities": [a["executable_identity"]] * 2})
        assert "duplicate" in run(records, f"--holdout-manifest={holdout}", success=False).stderr
        write("holdout.json", {"version": 1, "identities": ["unprovided-full-identity"]})
        assert "missing" in run(records, f"--holdout-manifest={holdout}", success=False).stderr
        write("holdout.json", {"version": 2, "identities": []})
        run(records, f"--holdout-manifest={holdout}", success=False)
        # Training sets need an explicit disjoint holdout, not an accuracy claim on themselves.
        run(records, f"--training-manifest={training}", success=False)
        write("holdout.json", {"version": 1, "identities": [b["executable_identity"]]})
        selected = json.loads(run(records, f"--holdout-manifest={holdout}", f"--training-manifest={training}").stdout)
        assert selected["record_count"] == 1
        assert selected["not_in_holdout_count"] == 1
        assert selected["groups"][0]["median_absolute_relative_error"] == 0.5
        # Reporting never changes source records.
        assert json.loads(a_path.read_text()) == a
        assert json.loads(b_path.read_text()) == b
        # Output aliases must not truncate evidence through a hard link.
        alias = root / "output-alias.json"
        alias.hardlink_to(a_path)
        assert "overwrite" in run(a_path, f"--output={alias}", success=False).stderr
        assert json.loads(a_path.read_text()) == a
        print("PredictionValidationCLI: deterministic statistics, regimes, holdouts and exclusions passed")


if __name__ == "__main__":
    main()
