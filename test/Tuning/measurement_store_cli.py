#!/usr/bin/env python3
# Part of the LLK Compiler Project, under the Apache License v2.0 with LLVM Exceptions.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Real process-restart persistence, exact misses and verification gates."""
import json
import pathlib
import platform
import subprocess
import sys
import tempfile


def main():
    tune, repo = map(pathlib.Path, sys.argv[1:3])
    with tempfile.TemporaryDirectory(prefix="llk-measurement-cli-") as temporary:
        root = pathlib.Path(temporary)
        fixture = root / "fixture.json"
        fixture.write_text(json.dumps({
            "version": 1, "warmup": 1, "repeat": 3,
            "inputs": [
                {"dtype": "f32", "shape": [8, 16], "data": 1},
                {"dtype": "f32", "shape": [8, 16], "data": 2},
                {"dtype": "f32", "shape": [8, 16], "data": 0}],
            "outputs": [{"dtype": "f32", "shape": [8, 16], "data": 3}]}))
        source = root / "source.mlir"
        source.write_text((repo / "test/Tuning/Inputs/framework/elementwise_search.mlir").read_text())
        report = root / "report.json"
        store = root / "store"
        common = [str(tune), f"--input={source}",
                  "--mapping-target=x86-avx2", f"--mapping-root={repo}",
                  f"--machine={repo / 'machines/x86-avx2-v2.yaml'}",
                  "--mapping-mode=exact", "--candidate-source=semantic",
                  "--source-symbol=elementwise", "--source-adapter=linalg-elementwise-v1",
                  "--top-k=1", f"--output={root / 'schedule.yaml'}",
                  f"--mapping-report={report}"]

        def run(*extra, success=True):
            result = subprocess.run([*common, *extra], capture_output=True, text=True)
            assert (result.returncode == 0) == success, result.stdout + result.stderr
            return result

        # Defaults and explicit static runs have no persistence side effects.
        run("--mapping-backend=reference")
        assert not store.exists()
        run("--mapping-backend=reference", f"--measurement-store={store}", success=False)
        assert not store.exists()
        run("--mapping-backend=reference", "--measurement-reuse=approximate", success=False)
        run("--mapping-backend=reference", "--measurement-reuse=exact", success=False)

        measured = ["--measure-top=1", f"--measurement-inputs={fixture}",
                    f"--measurement-store={store}", "--measurement-reuse=exact"]

        def measure(backend="reference", hits=0, misses=1, writes=1, attempts=1, *extra):
            run(f"--mapping-backend={backend}", *measured, *extra)
            data = json.loads(report.read_text())
            assert data["measurement_record_hits"] == hits, data
            assert data["measurement_record_misses"] == misses, data
            assert data["measurement_records_written"] == writes, data
            assert data["measurement_attempts"] == attempts, data
            sidecar = json.loads(pathlib.Path(str(report) + ".measurements.json").read_text())
            observed = sidecar["records"][0]
            assert observed["measured_ns"] > 0
            assert observed["measured_ns_origin"] == ("measured_record" if hits else "measured_runtime")
            return observed

        first = measure()
        second = measure(hits=1, misses=0, writes=0, attempts=0)
        assert first["measured_ns"] == second["measured_ns"]
        records = list(store.glob("*.json"))
        assert len(records) == 1
        record = json.loads(records[0].read_text())
        assert record["verified"] is True
        assert len(record["samples_ns"]) == 3
        assert record["source_adapter"] == "linalg-elementwise-v1"
        assert record["workload_contract"]
        assert record["environment"]["threads"] > 0
        # Fixture and protocol content are identity, even if arithmetic agrees.
        original = fixture.read_text()
        data = json.loads(original)
        data["outputs"][0]["absolute_tolerance"] = 0.001
        fixture.write_text(json.dumps(data))
        measure()
        fixture.write_text(original)
        data["repeat"] = 4
        fixture.write_text(json.dumps(data))
        measure()
        fixture.write_text(original)
        # Complete source content, not a filename or a workload label, distinguishes records.
        original_source = source.read_text()
        source.write_text(original_source.replace("module {", 'module attributes {llk.test_identity = "changed"} {', 1))
        measure()
        source.write_text(original_source)
        # A different environment cannot use the record at the same digest path.
        original_record = records[0].read_text()
        record["environment"]["threads"] += 1
        records[0].write_text(json.dumps(record))
        run("--mapping-backend=reference", *measured, success=False)  # collision refuses overwrite
        records[0].write_text(original_record)
        # A bad fixture never writes a verified record.
        bad_store = root / "bad-store"
        data = json.loads(original)
        data["outputs"][0]["data"] = 99
        fixture.write_text(json.dumps(data))
        result = run("--mapping-backend=reference", "--measure-top=1",
                     f"--measurement-inputs={fixture}", f"--measurement-store={bad_store}", success=False)
        assert "measurement verification" in result.stderr, result.stderr
        assert not list(bad_store.glob("*.json"))
        fixture.write_text(original)
        if platform.machine().lower() in ("x86_64", "amd64"):
            measure(backend="selected-target")
            measure(backend="selected-target", hits=1, misses=0, writes=0, attempts=0)
            print("MeasurementStoreCLI: selected-target write and exact hit passed")
        else:
            result = run("--mapping-backend=selected-target", *measured, success=False)
            assert "host" in result.stderr.lower(), result.stderr
            print("MeasurementStoreCLI: reference write/hit passed; selected host rejected explicitly")


if __name__ == "__main__":
    main()
