#!/usr/bin/env python3
# Part of the LLK Compiler Project, under the Apache License v2.0 with LLVM Exceptions.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Publish an uncalibrated real-record baseline without an accuracy threshold."""
import json
import pathlib
import platform
import subprocess
import sys
import tempfile


def main():
    tune, validate, repo, evidence = [pathlib.Path(arg).resolve() for arg in sys.argv[1:5]]
    evidence.mkdir(parents=True, exist_ok=True)
    run = pathlib.Path(tempfile.mkdtemp(prefix="run-", dir=evidence))
    fixture = run / "fixture.json"
    fixture.write_text(json.dumps({
        "version": 1, "warmup": 2, "repeat": 5,
        "inputs": [{"dtype": "f32", "shape": [8, 16], "data": value} for value in (1, 2, 0)],
        "outputs": [{"dtype": "f32", "shape": [8, 16], "data": 3}]}))
    backend = "selected-target" if platform.machine().lower() in ("x86_64", "amd64") else "reference"
    store = run / "records"
    subprocess.run([
        str(tune), f"--input={repo / 'test/Tuning/Inputs/framework/elementwise_search.mlir'}",
        "--mapping-target=x86-avx2", f"--mapping-root={repo}",
        f"--machine={repo / 'machines/x86-avx2-v2.yaml'}", "--mapping-mode=exact",
        f"--mapping-backend={backend}", "--candidate-source=semantic", "--source-symbol=elementwise",
        "--source-adapter=linalg-elementwise-v1", "--top-k=1", "--measure-top=1",
        f"--measurement-inputs={fixture}", f"--measurement-store={store}",
        "--measurement-reuse=off", f"--output={run / 'schedule.yaml'}",
        f"--mapping-report={run / 'tuning.json'}"], check=True, capture_output=True, text=True)
    paths = list(store.glob("*.json"))
    assert len(paths) == 1
    record = json.loads(paths[0].read_text())
    holdout = run / "holdout.json"
    holdout.write_text(json.dumps({"version": 1, "identities": [record["executable_identity"]]}))
    report_path = run / "baseline.json"
    subprocess.run([str(validate), str(store), f"--holdout-manifest={holdout}",
                    f"--output={report_path}"], check=True, capture_output=True, text=True)
    report = json.loads(report_path.read_text())
    assert report["held_out"] and not report["calibrated"]
    assert report["record_count"] == report["compared_record_count"] == 1
    assert report["sample_count"] == 5
    group = report["groups"][0]
    assert group["rows"][0]["identity"] == record["executable_identity"]
    assert group["rows"][0]["predicted_ns"] == record["prediction"]["predicted_ns"]
    assert group["rows"][0]["measured_ns"] == record["median_ns"]
    assert group["regime"]["target_identity"] == record["target_identity"]
    assert group["regime"]["compiler_version"] == record["compiler_version"]
    assert group["prediction_origin"] == "predicted_static"
    assert group["regime"]["backend_identity"] == record["backend_identity"]
    print(f"PredictionValidationMappedCLI: {backend} verified record -> held-out uncalibrated baseline")
    print(f"baseline={report_path} median_absolute_relative_error={group['median_absolute_relative_error']}")


if __name__ == "__main__":
    main()
