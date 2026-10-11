#!/usr/bin/env python3
"""Exercise the shipped fused AVX2 rule through search and target lowering."""

import json
import os
import subprocess
import sys


def run(command):
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(
            "command failed (%d): %s\n%s" % (
                result.returncode, " ".join(command), result.stderr.strip()))
    return result.stdout


def main():
    llk_opt, llk_compile, source_dir, work_dir = sys.argv[1:]
    os.makedirs(work_dir, exist_ok=True)
    fixture = os.path.join(
        source_dir, "test/Execution/Inputs/issue129/fused_convert_silu_mul.mlir")
    machine_path = os.path.join(source_dir, "machines/x86-avx2-v2.yaml")
    layouts = os.path.join(source_dir, "mapping/x86-avx2/layouts.llkmap")
    rules = os.path.join(source_dir, "mapping/x86-avx2/rules.llkmap")
    emitters = (
        "avx2_vector_add,avx2_vector_convert,avx2_vector_silu,"
        "avx2_vector_mul,avx2_mma,avx2_reduce,avx2_copy,"
        "avx2_tile_copy,avx2_tile_store,avx2_fused_convert_silu_mul")
    def check_selection(machine, width):
        report_path = os.path.join(work_dir, "fused-report-%d.json" % width)
        mapped_path = os.path.join(work_dir, "fused-mapped-%d.mlir" % width)
        map_options = " ".join((
            "target=x86-avx2", "machine=" + machine, "layouts=" + layouts,
            "rules=" + rules, "emitters=" + emitters, "mode=exact",
            "report=" + report_path))
        mapped = run([llk_opt, "--micro-map=" + map_options, fixture])
        with open(mapped_path, "w") as handle:
            handle.write(mapped)

        with open(report_path) as handle:
            report = json.load(handle)
        selected = report["selectedState"]
        rules_selected = {placement["rule"] for placement in selected["placements"]}
        assert {"avx2.vector_silu", "avx2.vector_mul",
                "avx2.vector_convert_row_major", "avx2.tile_store"} <= rules_selected, (
            "exact search did not select every mapped epilogue and output op at VW%d" % width)
        convert = next(p for p in selected["placements"]
                       if p["rule"] == "avx2.vector_convert_row_major")
        assert convert["ruleParameters"].get("VW") == width, (
            "the selected conversion VW was not recorded")

    check_selection(machine_path, 8)

    # Before the bundle/layout contract was completed this public pipeline
    # failed with: "no vector width was selected".
    lowered = run([
        llk_compile, "--mapping-target=x86-avx2",
        "--mapping-root=" + source_dir, "--mapping-mode=exact",
        "--mapping-stop=target-lowered", "--emit=mlir", fixture])
    assert "target-lowered-ops=4" in lowered
    assert "reference-lowered-ops=0" in lowered
    assert "vector = 8" in lowered


if __name__ == "__main__":
    main()
