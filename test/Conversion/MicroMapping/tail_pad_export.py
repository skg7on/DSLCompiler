#!/usr/bin/env python3
"""Tail-padding export acceptance for issue #129 E9."""

import json
import os
import subprocess
import sys


def main():
    llk_opt, llk_compile, source_dir, work_dir = sys.argv[1:]
    os.makedirs(work_dir, exist_ok=True)
    cases = {
        "tail_m": (2, 8, 8),
        "tail_n": (1, 9, 8),
        "tail_k": (1, 8, 7),
        "tail_all": (2, 9, 7),
    }
    entries = []
    for name, (m_bucket, n, k) in cases.items():
        entries.append({
            "operation": "matmul",
            "shape": {"M_bucket": m_bucket, "N": n, "K": k},
            "schedule": {
                "BM": 8, "BN": 16, "BK": 32,
                "mma_shape": "8x8x8", "fragment_shape": "8x8x8",
                "memory_path": "dram:sram:acc", "tile_layout": "row_major",
                "owner_mapping": "worker/lane", "fragment_owner": "vector_engine",
                "tail_policy": "pad", "enable_tile_masks": False,
            },
        })
    schedule = os.path.join(work_dir, "tail-schedule.json")
    with open(schedule, "w", encoding="utf-8") as handle:
        json.dump({"entries": entries}, handle)

    for name in cases:
        fixture = os.path.join(source_dir, "test/Execution/Inputs/issue129/" + name + ".mlir")
        result = subprocess.run(
            [llk_opt, "--llk-to-micro=schedule-db=" + schedule, fixture],
            text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        assert result.returncode == 0, (
            name + " pad export failed\n" + result.stderr + result.stdout)
        assert 'tail_policy = "pad"' in result.stdout, result.stdout
        assert "original_workload" in result.stdout, result.stdout
        assert "valid_extents" in result.stdout, (
            name + " did not preserve generic valid extents\n" + result.stdout)
        lowered = subprocess.run(
            [llk_opt, "--micro-to-linalg"], input=result.stdout, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        assert lowered.returncode == 0, (
            name + " padded kernel did not lower\n" + lowered.stderr + lowered.stdout)
        assert "tensor.pad" in lowered.stdout, (
            name + " did not materialize zero-filled source padding\n" + lowered.stdout)
        assert "tensor.extract_slice" in lowered.stdout, (
            name + " did not restrict stores to valid output extents\n" + lowered.stdout)
        assert "micro.tile_partition" not in lowered.stdout, (
            name + " retained a Micro partition after lowering\n" + lowered.stdout)
        if name == "tail_all":
            selected = subprocess.run(
                [llk_compile, "--mapping-target=x86-avx2",
                 "--mapping-root=" + source_dir, "--mapping-mode=exact",
                 "--mapping-backend=selected-target", "--mapping-stop=lowered",
                 "--emit=mlir", "-"], input=result.stdout, text=True,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            assert selected.returncode == 0, (
                name + " padded kernel has no selected x86 plan\n" +
                selected.stderr + selected.stdout)
            assert "backend=selected-target" in selected.stdout, selected.stdout
            assert "reference-lowered-ops=0" in selected.stdout, selected.stdout
            assert "target-lowered-ops=0" not in selected.stdout, selected.stdout

    tail_fixture = os.path.join(
        source_dir, "test/Execution/Inputs/issue129/tail_all.mlir")
    for policy, expected in (("none", "tail_policy is 'none'"),
                             ("mask", "tail_policy 'mask' is unsupported")):
        rejected_entries = json.loads(json.dumps(entries))
        for entry in rejected_entries:
            entry["schedule"]["tail_policy"] = policy
            entry["schedule"].update({"BM": 4, "BN": 8, "BK": 4})
        rejected_schedule = os.path.join(work_dir, "tail-" + policy + ".json")
        with open(rejected_schedule, "w", encoding="utf-8") as handle:
            json.dump({"entries": rejected_entries}, handle)
        rejected = subprocess.run(
            [llk_opt, "--llk-to-micro=schedule-db=" + rejected_schedule,
             tail_fixture], text=True, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE)
        assert rejected.returncode != 0, (
            "tail_policy='" + policy + "' unexpectedly accepted non-divisible extents")
        assert expected in rejected.stderr, rejected.stderr


if __name__ == "__main__":
    main()
