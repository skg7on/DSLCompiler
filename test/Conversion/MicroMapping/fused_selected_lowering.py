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
        fused = [placement for placement in selected["placements"]
                 if placement["rule"] == "avx2.fused_convert_silu_mul"]
        assert len(fused) == 3, (
            "exact search did not select the whole fused group at VW%d" % width)
        instances = {placement["instance"] for placement in fused}
        assert len(instances) == 1, "the fused placements do not share one instance"
        assert all(placement["bundle"] == "avx2.fused.convert_silu_mul"
                   for placement in fused)
        assert all(placement["ruleParameters"].get("VW") == width
                   for placement in fused), "the selected VW was not recorded"

        group_nodes = {placement["node"] for placement in fused}
        endpoints = set()
        widths = set()
        for placement in fused:
            for solution in placement["solutions"]:
                widths.add(solution["parameters"].get("VW"))
                port = solution.get("port")
                if port:
                    assert port["node"] in group_nodes, (
                        "layout endpoint escaped the selected fused instance")
                    endpoints.add((port["node"], port["direction"], port["index"]))
        assert widths == {width}, "endpoint layouts do not share the selected VW"
        assert any(direction == "input" for _, direction, _ in endpoints)
        assert any(direction == "output" for _, direction, _ in endpoints)

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
