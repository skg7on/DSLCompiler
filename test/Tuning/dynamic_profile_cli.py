#!/usr/bin/env python3
"""Concrete-sample dynamic tuning and analysis smoke test (issue #198).

The fixture is a fully dynamic M/N/K matmul. Tuning is legal only on explicit
samples, and the report must record the profile plus each observation's cost.
`micro-perf --shape-values` then analyzes one sample through the same shape
specialization; a partial binding is an explicit incomplete analysis, never a
rankable one-iteration score.
"""

import json
import pathlib
import subprocess
import sys
import tempfile

tune, micro_perf, repo = map(pathlib.Path, sys.argv[1:4])
source = repo / "test/Tuning/Inputs/issue198/dynamic_matmul.mlir"

with tempfile.TemporaryDirectory(prefix="llk-dynamic-profile-") as temp:
    root = pathlib.Path(temp)
    machine = root / "machine.yaml"
    # The bounded baseline keeps eight workers' accumulators live at once and
    # loads the read initializer straight into the accumulator. The fixture
    # raises the local capacities and declares that modeled external path.
    text = (repo / "machines/x86-avx2-v2.yaml").read_text()
    text = text.replace("capacity_bytes: 8192", "capacity_bytes: 1048576")
    text = text.replace("capacity_bytes: 32768", "capacity_bytes: 1048576")
    text += (
        "  - id: dram_to_acc.0\n"
        "    source: dram.0\n"
        "    destination: acc.0\n"
        "    bandwidth_bytes_per_cycle: 64\n"
        "    latency_cycles: 220\n"
        "    transaction_bytes: 64\n"
        "    transfer_engines: [dma.0]\n"
        "  - id: acc_to_dram.0\n"
        "    source: acc.0\n"
        "    destination: dram.0\n"
        "    bandwidth_bytes_per_cycle: 64\n"
        "    latency_cycles: 220\n"
        "    transaction_bytes: 64\n"
        "    transfer_engines: [dma.0]\n"
    )
    machine.write_text(text)

    profile = root / "profile.json"
    profile.write_text(
        json.dumps(
            {
                "version": 1,
                "samples": [
                    {"values": {"M": 7, "N": 13, "K": 5}, "weight": 1.0}
                ],
                "aggregation": "weighted_mean",
                "max_variants": 1,
            }
        )
    )
    report = root / "report.json"
    artifacts = root / "artifacts"
    common = [
        f"--mapping-target=x86-avx2",
        f"--mapping-root={repo}",
        f"--machine={machine}",
        "--candidate-source=semantic",
        "--source-symbol=matmul",
        "--mapping-mode=exact",
        "--mapping-backend=reference",
        "--top-k=1",
    ]
    subprocess.run(
        [
            str(tune),
            f"--input={source}",
            f"--output={root / 'schedule.yaml'}",
            f"--mapping-report={report}",
            f"--candidate-artifacts={artifacts}",
            f"--shape-profile={profile}",
            *common,
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    data = json.loads(report.read_text())
    recorded = data["shape_profile"]
    assert recorded["aggregation"] == "weighted_mean"
    assert recorded["max_variants"] == 1
    assert recorded["samples"][0]["values"] == {"K": 5, "M": 7, "N": 13}
    selected = data["candidates"][0]
    assert selected["predicted_cycles"] > 0
    assert selected["shape_aggregation"] == "weighted_mean"
    samples = selected["shape_samples"]
    assert len(samples) == 1
    assert samples[0]["values"] == {"K": 5, "M": 7, "N": 13}
    assert samples[0]["latency_cycles"] == selected["predicted_cycles"]

    source_artifact = artifacts / selected["candidate_id"] / "source.mlir"
    assert "micro.workload_contract" in source_artifact.read_text()

    full = subprocess.run(
        [
            str(micro_perf),
            f"--machine={machine}",
            "--shape-values=M=7,N=13,K=5",
            str(source_artifact),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "sampled: true" in full.stdout
    # The artifact is the unmapped instantiated kernel, so its abstract `dma`
    # pool is honestly reported incomplete; the sampled work itself is exact.
    assert "flops: 1280" in full.stdout, full.stdout

    partial = subprocess.run(
        [
            str(micro_perf),
            f"--machine={machine}",
            "--shape-values=M=7,N=13",
            str(source_artifact),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "complete: false" in partial.stdout
    assert "missing_value: 'K'" in partial.stdout
    assert "predicted_cycles" not in partial.stdout

    bad = root / "bad.json"
    bad.write_text(
        json.dumps(
            {
                "version": 1,
                "samples": [
                    {"values": {"M": 7, "N": 13, "K": 5}, "weight": 0.0}
                ],
            }
        )
    )
    rejected = subprocess.run(
        [
            str(tune),
            f"--input={source}",
            f"--output={root / 'rejected.yaml'}",
            f"--shape-profile={bad}",
            *common,
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert rejected.returncode != 0
    assert "finite and positive" in rejected.stderr
