#!/usr/bin/env python3
"""Selected-lowering acceptance evidence for the C3 framework primitives.

Runs the real AVX2 target package over the reduction and composed-vector
fixtures, requires vector-reduction/vector IR evidence and zero reference
fallback, records the strict static analysis produced by micro-perf, and
writes a versioned manifest keyed by source/plan/target/compiler/mode.

The reduction is rank-one, so its invocation row is explicitly *unsupported*
with the mapped ABI v1 reason; no callable is claimed for it.
"""

import argparse
import json
import os
import platform
import re
import subprocess
import sys

MACHINE = "machines/x86-avx2-v2.yaml"


def run(args, *, check=True, input_text=None):
    result = subprocess.run(args, input=input_text, text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if check and result.returncode:
        raise RuntimeError("command failed: %s\n%s\n%s" %
                           (" ".join(args), result.stdout, result.stderr))
    return result


def token(text, name):
    for line in text.splitlines():
        for value in line.split():
            if value.startswith(name + "="):
                return value.split("=", 1)[1]
    return None


def compile_selected(llk_compile, root, fixture, stop, check=True):
    return run([
        llk_compile,
        "--mapping-target=x86-avx2",
        "--mapping-root=" + root,
        "--mapping-mode=exact",
        "--mapping-backend=selected-target",
        "--mapping-stop=" + stop,
        "--emit=mlir",
        fixture,
    ], check=check)


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("llk_compile")
    parser.add_argument("micro_perf")
    parser.add_argument("source_dir")
    parser.add_argument("work_dir")
    args = parser.parse_args()

    root = os.path.abspath(args.source_dir)
    work = os.path.abspath(args.work_dir)
    os.makedirs(work, exist_ok=True)
    machine = os.path.join(root, MACHINE)
    rows = []

    def analyzed_bound_ir(fixture, symbol):
        mapped = compile_selected(args.llk_compile, root, fixture,
                                  "mapped-micro")
        require("\n" in mapped.stdout,
                "mapped-micro output has no module: " + mapped.stdout)
        micro_ir = mapped.stdout.split("\n", 1)[1]
        bound_path = os.path.join(work, symbol + ".bound.mlir")
        with open(bound_path, "w", encoding="utf-8") as stream:
            stream.write(micro_ir)
        analysis = run([
            args.micro_perf,
            "--machine=" + machine,
            "--level=1",
            "--format=yaml",
            bound_path,
        ])
        require("complete: true" in analysis.stdout,
                "strict analysis is incomplete:\n" + analysis.stdout)
        require("bytes:" in analysis.stdout and
                "live_bytes:" in analysis.stdout,
                "analysis did not report traffic/storage:\n" +
                analysis.stdout)
        return mapped, analysis.stdout

    def base_row(fixture, symbol, lowered, analysis_yaml):
        mapping_line = lowered.stdout.splitlines()[0]
        require(mapping_line.startswith("mapping: "),
                "missing mapping line: " + mapping_line)
        replay_identity = token(mapping_line, "replay-identity")
        require(replay_identity and "|plan=" in replay_identity,
                "mapping line records no replay identity: " + mapping_line)
        plan_id = replay_identity.split("|plan=", 1)[1]
        require(plan_id, "mapping line records no plan id")
        return {
            "source": os.path.relpath(fixture, root),
            "symbol": symbol,
            "target": {
                "name": "x86-avx2",
                "machine": MACHINE,
            },
            "mode": {
                "backend": "selected-target",
                "stop": "lowered",
            },
            "compiler": token(mapping_line, "identity"),
            "replay_identity": replay_identity,
            "plan_id": plan_id,
            "stages": {
                "mapping": "supported",
                "physical_plan": "supported",
                "selected_lowering": "supported",
                "reference_lowering": "supported",
                "invocation": "not_checked",
            },
            "analysis": {
                "format": "micro-perf-yaml",
                "yaml": analysis_yaml,
            },
        }

    # --- direct reduction --------------------------------------------------
    reduce_fixture = os.path.join(root, "test/Execution/Inputs/framework/reduce.mlir")
    reduce_lowered = compile_selected(args.llk_compile, root, reduce_fixture,
                                      "lowered")
    reduce_ir = reduce_lowered.stdout
    require("vector.multi_reduction" in reduce_ir or
            "vector.reduction" in reduce_ir,
            "selected sum did not reach Vector reduction IR:\n" + reduce_ir)
    require("vector<8xf32>" in reduce_ir,
            "selected reduction width is not 8xf32:\n" + reduce_ir)
    require("micro.reduce" not in reduce_ir,
            "micro.reduce survived selected lowering:\n" + reduce_ir)
    require("target-lowered-ops=1" in reduce_lowered.stdout,
            "the selected reduce was not counted as target-lowered")
    require("reference-lowered-ops=0" in reduce_lowered.stdout,
            "the selected reduce fell back to reference lowering")
    require("reference-groups-lowered=0" in reduce_lowered.stdout,
            "a selected group fell back to the reference bridge")
    _, reduce_analysis = analyzed_bound_ir(reduce_fixture, "reduce_sum")
    reduce_row = base_row(reduce_fixture, "reduce_sum", reduce_lowered,
                          reduce_analysis)
    reduce_row["ir_evidence"] = {
        "vector_reduction": True,
        "selected_width": "vector<8xf32>",
        "micro_reduce_remaining": False,
    }

    refused = compile_selected(args.llk_compile, root, reduce_fixture,
                               "executable", check=False)
    refusal = refused.stdout + refused.stderr
    require(refused.returncode != 0,
            "the rank-one reduction must not prepare an executable")
    require("runtime_capability:" in refusal and "rank-two" in refusal,
            "the reduction refusal did not name the rank-two ABI:\n" + refusal)
    reduce_row["stages"]["invocation"] = {
        "status": "unsupported",
        "reason": "runtime_capability: mapped ABI v1 requires rank-two ports",
    }
    rows.append(reduce_row)

    # --- composed add -> multiply (and add -> multiply -> reduce) ----------
    for fixture_name, symbol, node_count in (
            ("add_mul.mlir", "add_mul", 2),
            ("add_mul_reduce.mlir", "add_mul_reduce", 3)):
        fixture = os.path.join(root, "test/Execution/Inputs/framework",
                               fixture_name)
        lowered = compile_selected(args.llk_compile, root, fixture, "lowered")
        ir = lowered.stdout
        require("vector<8xf32>" in ir,
                "selected width is not 8xf32 in %s:\n%s" % (fixture_name, ir))
        require("target-lowered-ops=%d" % node_count in lowered.stdout,
                "not every selected operation was target-lowered in %s" %
                fixture_name)
        require("reference-lowered-ops=0" in lowered.stdout,
                "selected work fell back to reference lowering in %s" %
                fixture_name)
        require("reference-groups-lowered=0" in lowered.stdout,
                "a selected group fell back to the reference bridge in %s" %
                fixture_name)
        if symbol == "add_mul_reduce":
            require("vector.multi_reduction" in ir or
                    "vector.reduction" in ir,
                    "the composed reduction did not reach Vector IR")
        _, analysis_yaml = analyzed_bound_ir(fixture, symbol)
        row = base_row(fixture, symbol, lowered, analysis_yaml)
        row["ir_evidence"] = {
            "selected_width": "vector<8xf32>",
            "vector_reduction": "vector.multi_reduction" in ir or
                                "vector.reduction" in ir,
        }
        row["numeric"] = {
            "formula": "(left[i] + right[i]) * left[i]",
            "elements": 64,
            "verified_by": "FrameworkPrimitivesTest/SelectedAvx2AddMul",
        }
        rows.append(row)

    manifest = {
        "schema_version": 1,
        "kind": "framework-selected-evidence",
        "host": platform.machine(),
        "rows": rows,
    }
    manifest_path = os.path.join(work, "framework_selected_evidence.manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print("framework selected evidence: %d rows -> %s" %
          (len(rows), manifest_path))


if __name__ == "__main__":
    sys.exit(main())
