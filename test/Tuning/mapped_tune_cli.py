#!/usr/bin/env python3
"""Public target-aware tune and frozen plan replay smoke test."""

import json
import platform
import pathlib
import shutil
import subprocess
import sys
import tempfile

tune, compile_tool, repo = map(pathlib.Path, sys.argv[1:4])
source = repo / "test/Tuning/Inputs/issue129/mapped_tune.mlir"
with tempfile.TemporaryDirectory(prefix="llk-mapped-tune-") as temp:
    root = pathlib.Path(temp)
    report = root / "report.json"
    schedule = root / "schedule.yaml"
    artifacts = root / "artifacts"
    common = [
        "--mapping-target=x86-avx2",
        f"--mapping-root={repo}",
        f"--machine={repo / 'machines/x86-avx2-v2.yaml'}",
    ]
    subprocess.run(
        [
            str(tune), f"--input={source}", "--M=8", "--N=16", "--K=32",
            "--top-k=1", f"--output={schedule}", *common,
            "--mapping-mode=exact", "--mapping-backend=reference",
            "--candidate-source=semantic", "--source-symbol=matmul",
            f"--mapping-report={report}", f"--candidate-artifacts={artifacts}",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    data = json.loads(report.read_text())
    assert data["execution_mode"] == "lowering-validation"
    assert len(data["candidates"]) == 1
    selected = data["candidates"][0]
    assert selected["bindings"]["BM"] == 8
    assert selected["source_mode"] == "semantic"
    assert selected["predicted_cycles"] > 0
    assert selected["predicted_ns"] > 0
    assert selected["dram_bytes"] >= 0 and selected["sram_bytes"] >= 0
    assert selected["metric_origins"]["predicted_cycles"] == "static-model"
    assert data["mapping_plan_count"] >= 1
    assert isinstance(data["mapping_truncated"], bool)
    assert isinstance(data["generator_truncated"], bool)
    assert '"selectedState"' in selected["plan_report"]
    candidate_dir = artifacts / selected["candidate_id"]
    frozen = candidate_dir / "plan.json"
    instantiated = candidate_dir / "source.mlir"
    assert frozen.read_text() == selected["plan_report"]
    assert "micro.search_space" not in instantiated.read_text()
    frozen_state = json.loads(frozen.read_text())["selectedState"]
    assert any(p.get("bundleParameters", {}).get("VW") == 8
               for p in frozen_state["placements"])
    yaml = schedule.read_text()
    assert selected["plan_id"] in yaml and selected["binding_hash"] in yaml
    copied_profile = root / "same-machine.yaml"
    shutil.copyfile(repo / "machines/x86-avx2-v2.yaml", copied_profile)
    replay_common = [arg for arg in common if not arg.startswith("--machine=")]
    replay_common.append(f"--machine={copied_profile}")
    replay = subprocess.run(
        [
            str(compile_tool), *replay_common, "--mapping-backend=reference",
            "--mapping-stop=lowered", f"--plan-report={frozen}",
            str(instantiated), "--emit=mlir",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert replay.returncode == 0, replay.stderr
    assert "selected-groups-verified=8" in replay.stdout
    assert "micro.selected_connections" in replay.stdout
    assert "arith.extf" in replay.stdout
    replay_again = subprocess.run(
        [
            str(compile_tool), *replay_common, "--mapping-backend=reference",
            "--mapping-stop=lowered", f"--plan-report={frozen}",
            str(instantiated), "--emit=mlir",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert replay.stdout == replay_again.stdout

    changed_source = root / "changed-source.mlir"
    source_text = instantiated.read_text()
    changed_source.write_text(
        source_text.replace(
            "{accumulator = #micro.dtype<f32>, input = #micro.dtype<bf16>",
            "{test_marker = 1 : i64, accumulator = #micro.dtype<f32>, "
            "input = #micro.dtype<bf16>",
            1,
        )
    )
    stale_source = subprocess.run(
        [
            str(compile_tool), *replay_common, "--mapping-backend=reference",
            "--mapping-stop=lowered", f"--plan-report={frozen}",
            str(changed_source), "--emit=mlir",
        ],
        capture_output=True,
        text=True,
    )
    assert stale_source.returncode != 0, stale_source.stdout + stale_source.stderr
    assert "graph_hash does not match" in stale_source.stderr, stale_source.stderr

    if platform.machine().lower() not in ("x86_64", "amd64"):
        unsupported_host = subprocess.run(
            [str(tune), f"--input={source}", "--M=8", "--N=16", "--K=32",
             "--mapping-target=x86-avx2", f"--mapping-root={repo}",
             f"--machine={repo / 'machines/x86-avx2-v2.yaml'}",
             "--mapping-backend=selected-target", "--candidate-source=semantic",
             "--source-symbol=matmul", "--measure-top=1",
             f"--measurement-inputs={repo / 'test/Tuning/Inputs/issue129/measurement_inputs.json'}"],
            capture_output=True,
            text=True,
        )
        assert unsupported_host.returncode != 0
        assert ("unsupported host" in unsupported_host.stderr or
                "host does not support" in unsupported_host.stderr or
                "requires architecture" in unsupported_host.stderr or
                "requires x86" in unsupported_host.stderr), unsupported_host.stderr

    changed_profile = root / "changed-machine.yaml"
    changed_profile.write_text(
        copied_profile.read_text().replace("clock_hz: 3000000000",
                                           "clock_hz: 3000000001"))
    stale = subprocess.run(
        [
            str(compile_tool), *replay_common[:-1],
            f"--machine={changed_profile}", "--mapping-backend=reference",
            "--mapping-stop=lowered", f"--plan-report={frozen}",
            str(instantiated), "--emit=mlir",
        ],
        capture_output=True,
        text=True,
    )
    assert stale.returncode != 0 and "target_hash does not match" in stale.stderr

    missing_fixture = subprocess.run(
        [str(tune), f"--input={source}", *common, "--measure-top=1"],
        capture_output=True,
        text=True,
    )
    assert missing_fixture.returncode != 0
    assert "--measure-top requires --measurement-inputs" in missing_fixture.stderr

    missing_target = subprocess.run(
        [str(tune), f"--input={source}", "--mapping-root=" + str(repo),
         f"--machine={repo / 'machines/x86-avx2-v2.yaml'}",
         "--mapping-report=unused.json", "--candidate-source=semantic",
         "--source-symbol=matmul"],
        capture_output=True,
        text=True,
    )
    assert missing_target.returncode != 0, missing_target.stdout + missing_target.stderr
    assert "target-aware options require --mapping-target" in missing_target.stderr, missing_target.stderr

    bad_expected = root / "bad-expected.json"
    bad_fixture = json.loads(
        (repo / "test/Tuning/Inputs/issue129/measurement_inputs.json").read_text()
    )
    bad_fixture["outputs"][0]["data"] = 68
    bad_expected.write_text(json.dumps(bad_fixture))
    rejected_output = subprocess.run(
        [str(tune), f"--input={source}", "--M=8", "--N=16", "--K=32",
         "--mapping-target=x86-avx2", f"--mapping-root={repo}",
         f"--machine={repo / 'machines/x86-avx2-v2.yaml'}",
         "--candidate-source=semantic", "--source-symbol=matmul",
         "--measure-top=1", f"--measurement-inputs={bad_expected}"],
        capture_output=True,
        text=True,
    )
    assert rejected_output.returncode != 0
    assert "measurement verification" in rejected_output.stderr

    unknown_source = subprocess.run(
        [str(tune), f"--input={source}", *common,
         "--candidate-source=not-a-source"],
        capture_output=True,
        text=True,
    )
    assert unknown_source.returncode != 0
    assert "unsupported --candidate-source" in unknown_source.stderr

    unknown_root = subprocess.run(
        [str(tune), f"--input={source}", "--mapping-target=x86-avx2",
         f"--mapping-root={root / 'missing-config'}",
         f"--machine={repo / 'machines/x86-avx2-v2.yaml'}"],
        capture_output=True,
        text=True,
    )
    assert unknown_root.returncode != 0
    assert "cannot read" in unknown_root.stderr or "cannot open" in unknown_root.stderr

    missing_source_root = subprocess.run(
        [str(tune), f"--input={source}", *common,
         "--candidate-source=semantic", "--source-symbol=matmul",
         "--source-root=2"],
        capture_output=True,
        text=True,
    )
    assert missing_source_root.returncode != 0
    assert "out of range" in missing_source_root.stderr

    measured_report = root / "measured.json"
    measured_schedule = root / "measured.yaml"
    static_report = root / "static_again.json"
    subprocess.run(
        [
            str(tune), f"--input={source}", "--M=8", "--N=16", "--K=32",
            "--top-k=1", f"--output={root / 'static_again.yaml'}", *common,
            "--mapping-mode=exact", "--mapping-backend=reference",
            "--candidate-source=semantic", "--source-symbol=matmul",
            f"--mapping-report={static_report}",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        [
            str(tune), f"--input={source}", "--M=8", "--N=16", "--K=32",
            "--top-k=1", f"--output={measured_schedule}", *common,
            "--mapping-mode=exact", "--mapping-backend=reference",
            "--candidate-source=semantic", "--source-symbol=matmul",
            "--measure-top=1",
            f"--measurement-inputs={repo / 'test/Tuning/Inputs/issue129/measurement_inputs.json'}",
            f"--mapping-report={measured_report}",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    measured = json.loads(measured_report.read_text())
    assert measured_report.read_bytes() == static_report.read_bytes()
    measured_candidate = measured["candidates"][0]
    assert "measured_ns" not in measured_candidate
    sidecar = json.loads(pathlib.Path(str(measured_report) + ".measurements.json").read_text())
    assert sidecar["records"][0]["measured_ns"] > 0
    assert sidecar["records"][0]["identity"]["content_hash"]
    assert sidecar["records"][0]["warmup"] == 1
    assert sidecar["records"][0]["repeat"] == 5
    assert "measured: true" in measured_schedule.read_text()
    assert "warmup: 1" in measured_schedule.read_text()
    assert "repeat: 5" in measured_schedule.read_text()
