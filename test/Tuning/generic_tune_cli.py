#!/usr/bin/env python3
"""Generic source CLI, frozen replay and identity acceptance."""

import json
import pathlib
import shutil
import subprocess
import sys
import tempfile

tune, compile_tool, repo = map(pathlib.Path, sys.argv[1:4])
source = repo / "test/Tuning/Inputs/framework/elementwise_search.mlir"
adapter = "linalg-elementwise-v1"


def run(command, check=True):
    return subprocess.run(
        [str(argument) for argument in command],
        check=check,
        capture_output=True,
        text=True,
    )


def field(stdout, name):
    for line in stdout.splitlines():
        for token in line.split():
            if token.startswith(name + "="):
                return token.split("=", 1)[1]
    return None


with tempfile.TemporaryDirectory(prefix="llk-generic-tune-") as temp:
    root = pathlib.Path(temp)
    report = root / "report.json"
    schedule = root / "schedule.yaml"
    artifacts = root / "artifacts"
    common = [
        "--mapping-target=x86-avx2",
        f"--mapping-root={repo}",
        f"--machine={repo / 'machines/x86-avx2-v2.yaml'}",
    ]

    run([
        tune, f"--input={source}", "--mapping-mode=exact",
        "--mapping-backend=reference", "--candidate-source=semantic",
        "--source-symbol=elementwise", f"--source-adapter={adapter}",
        "--top-k=1", f"--output={schedule}", f"--mapping-report={report}",
        f"--candidate-artifacts={artifacts}", *common,
    ])
    data = json.loads(report.read_text())

    # The generic CLI reports the inferred, versioned source contract and every
    # generated tile binding, not just the retained one.
    assert data["generated"] == 4, data
    assert data["source_contract"]["version"] == 1, data
    assert data["source_contract"]["canonical"], data
    assert data["source_adapter"] == adapter, data
    assert data["source_adapter_version"] == 1, data

    assert len(data["candidates"]) == 1, data
    selected = data["candidates"][0]
    assert selected["source_adapter"] == adapter, selected
    assert selected["source_receipt"]["source"] == ["T0", "T1"], selected
    assert selected["source_receipt"]["mapping"] == [], selected
    selected_identity = selected["replay_identity"]
    assert selected_identity, selected

    # Every stage is present, and the reference-backend run claims only what it
    # established: selected lowering was never attempted.
    capabilities = selected["capabilities"]
    assert capabilities["version"] == 1, capabilities
    for stage in ("structural", "interpretation", "mapping", "physical_plan",
                  "reference_lowering", "selected_lowering", "invocation"):
        assert stage in capabilities, capabilities
    assert capabilities["mapping"] == "supported", capabilities
    assert capabilities["physical_plan"] == "supported", capabilities
    assert capabilities["reference_lowering"] == "supported", capabilities
    assert capabilities["selected_lowering"] == "not_checked", capabilities
    assert capabilities["invocation"] == "not_checked", capabilities

    # The frozen record keeps the source, contract and selected plan identity
    # together, so the artifact does not need the reporting tool to interpret.
    frozen_yaml = schedule.read_text()
    assert "source_provenance:" in frozen_yaml, frozen_yaml
    assert f"adapter: {adapter}" in frozen_yaml, frozen_yaml
    assert "adapter_version: 1" in frozen_yaml, frozen_yaml
    assert "contract_version: 1" in frozen_yaml, frozen_yaml
    assert selected_identity in frozen_yaml, frozen_yaml
    assert selected["plan_id"] in frozen_yaml, frozen_yaml

    candidate_dir = artifacts / selected["candidate_id"]
    frozen = candidate_dir / "plan.json"
    instantiated = candidate_dir / "source.mlir"
    assert frozen.read_text() == selected["plan_report"]
    instantiated_text = instantiated.read_text()
    assert "micro.search_space" not in instantiated_text
    assert 'source_adapter = "linalg-elementwise-v1"' in instantiated_text, (
        instantiated_text)
    assert "micro.workload_contract" in instantiated_text, instantiated_text

    # Replaying the frozen source and plan recovers exactly the identity the
    # tuner selected: source graph, binding and plan id.
    replay = run([
        compile_tool, *common, "--mapping-backend=reference",
        "--mapping-stop=lowered", f"--plan-report={frozen}",
        instantiated, "--emit=mlir",
    ])
    replay_identity = field(replay.stdout, "replay-identity")
    assert replay_identity == selected_identity, (replay.stdout,
                                                  selected_identity)
    assert field(replay.stdout, "source-adapter") == adapter, replay.stdout
    assert field(replay.stdout, "source-contract-version") == "1", replay.stdout

    # The compile sidecar carries the same stage vocabulary and leaves the
    # stages this run did not attempt not_checked.
    capability_line = next(
        (line for line in replay.stderr.splitlines()
         if line.startswith("capabilities: ")), None)
    assert capability_line, replay.stderr
    replay_capabilities = json.loads(capability_line.split(":", 1)[1].strip())
    assert replay_capabilities["mapping"] == "supported", replay_capabilities
    assert replay_capabilities["selected_lowering"] == "not_checked", (
        replay_capabilities)
    assert replay_capabilities["invocation"] == "not_checked", (
        replay_capabilities)

    # A changed source attribute is a different semantics, not a different
    # spelling of the same plan.
    changed_semantics = root / "changed-semantics.mlir"
    changed_semantics.write_text(
        instantiated_text.replace('micro.vector "add"', 'micro.vector "mul"', 1))
    stale = run([
        compile_tool, *common, "--mapping-backend=reference",
        "--mapping-stop=lowered", f"--plan-report={frozen}",
        changed_semantics, "--emit=mlir",
    ], check=False)
    assert stale.returncode != 0, stale.stdout + stale.stderr
    assert "graph_hash does not match" in stale.stderr, stale.stderr

    # A changed port extent is a different workload contract.
    changed_port = root / "changed-port.mlir"
    changed_port.write_text(
        instantiated_text.replace("tensor<8x16xf32>", "tensor<8x8xf32>")
        .replace("!micro.tile<8x16xf32", "!micro.tile<8x8xf32")
        .replace("array<i64: 8, 16>", "array<i64: 8, 8>"))
    stale = run([
        compile_tool, *common, "--mapping-backend=reference",
        "--mapping-stop=lowered", f"--plan-report={frozen}",
        changed_port, "--emit=mlir",
    ], check=False)
    assert stale.returncode != 0, stale.stdout + stale.stderr
    assert "graph_hash does not match" in stale.stderr, stale.stderr

    # A changed target rule is a different target, even with the same machine.
    changed_config = root / "changed-config"
    shutil.copytree(repo / "mapping", changed_config / "mapping")
    rules = changed_config / "mapping/x86-avx2/rules.llkmap"
    rules.write_text(rules.read_text().replace(
        "rule avx2.vector_add v1 {", "rule avx2.vector_add v2 {", 1))
    stale = run([
        compile_tool, "--mapping-target=x86-avx2",
        f"--mapping-root={changed_config}",
        f"--machine={repo / 'machines/x86-avx2-v2.yaml'}",
        "--mapping-backend=reference", "--mapping-stop=lowered",
        f"--plan-report={frozen}", instantiated, "--emit=mlir",
    ], check=False)
    assert stale.returncode != 0, stale.stdout + stale.stderr
    assert "target_hash does not match" in stale.stderr, stale.stderr

    # Changing the execution mode never reuses the previous identity: the
    # selected backend names different code generation for the same plan.
    selected_mode = run([
        compile_tool, *common, "--mapping-backend=selected-target",
        "--mapping-stop=lowered", f"--plan-report={frozen}",
        instantiated, "--emit=mlir",
    ])
    reference_identity = field(replay.stdout, "identity")
    selected_mode_identity = field(selected_mode.stdout, "identity")
    assert reference_identity, replay.stdout
    assert selected_mode_identity, selected_mode.stdout
    assert selected_mode_identity != reference_identity, (
        reference_identity, selected_mode_identity)

    # Explicit M/N/K belong to the contraction adapter; a source family that
    # describes its own contract must not receive them.
    mixed_shape = run([
        tune, f"--input={source}", "--M=8", "--N=16", "--K=32", *common,
        "--candidate-source=semantic", "--source-symbol=elementwise",
        f"--source-adapter={adapter}",
    ], check=False)
    assert mixed_shape.returncode != 0, mixed_shape.stdout + mixed_shape.stderr
    assert "contraction adapter" in mixed_shape.stderr, mixed_shape.stderr

    unknown_adapter = run([
        tune, f"--input={source}", *common, "--candidate-source=semantic",
        "--source-symbol=elementwise", "--source-adapter=missing-family",
    ], check=False)
    assert unknown_adapter.returncode != 0, (
        unknown_adapter.stdout + unknown_adapter.stderr)
    assert "unknown_source_adapter" in unknown_adapter.stderr, (
        unknown_adapter.stderr)
