#!/usr/bin/env python3
"""Exercise the community tutorials with a build from the same source tree.

Checks compilation artifacts and predicted reports, not numerical invocation
or selected-target instruction fidelity. Uses only the Python standard library.
"""

import argparse
import json
from pathlib import Path
import re
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=Path, default=Path("build"))
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    source = Path(__file__).resolve().parents[2]
    build = args.build_dir.resolve()
    output = (args.output_dir or build / "tutorial-check").resolve()
    output.mkdir(parents=True, exist_ok=True)
    tools = {name: build / name for name in
             ("llk-opt", "llk-compile", "micro-perf", "llk-tune", "llk-bench")}
    for name, path in tools.items():
        if not path.is_file():
            raise RuntimeError("missing %s: %s; build the five tutorial tools" % (name, path))

    steps = 0

    def run(name, *arguments, artifact=None):
        nonlocal steps
        command = [str(tools[name]), *map(str, arguments)]
        result = subprocess.run(command, cwd=source, capture_output=True,
                                text=True, timeout=120)
        if result.returncode:
            raise RuntimeError("command failed (%d): %s\n%s\n%s" %
                               (result.returncode, " ".join(command),
                                result.stdout[-8000:], result.stderr[-8000:]))
        if artifact:
            (output / artifact).write_text(result.stdout)
        steps += 1
        print("ok %02d: %s %s" % (steps, name, arguments[0] if arguments else ""))
        return result.stdout

    def require(condition, message):
        if not condition:
            raise RuntimeError(message)

    def pass_path(path):
        # These paths live inside an MLIR pass-option string. argv quoting
        # alone cannot protect whitespace from that second parser.
        return json.dumps(str(path), ensure_ascii=False)

    matmul = source / "docs/examples/matmul.mlir"
    swiglu = source / "docs/examples/swiglu.mlir"
    tile = source / "docs/examples/gemm-tile.micro.mlir"
    space = source / "docs/examples/swiglu.search.mlir"
    run("llk-opt", matmul, artifact="matmul.parsed.mlir")
    structured = run("llk-opt", "--llk-to-linalg", swiglu,
                     artifact="swiglu.linalg.mlir")
    require("linalg.matmul" in structured and "llk.fused_swiglu" not in structured,
            "SwiGLU structured lowering did not replace its semantic operation")
    concrete = run("llk-compile", "--emit=micro", matmul,
                   artifact="matmul.micro.mlir")
    require("micro.kernel @matmul_M16_N64_K64" in concrete and "micro.mma" in concrete,
            "matmul Micro export missing the expected kernel/work")
    run("llk-opt", output / "matmul.micro.mlir", artifact="matmul.roundtrip.mlir")
    fused = run("llk-compile", "--emit=micro", swiglu, artifact="swiglu.micro.mlir")
    require('micro.vector "silu"' in fused and 'micro.vector "mul"' in fused,
            "SwiGLU export missing its epilogue")
    exported_space = run("llk-compile", "--emit=micro-search", swiglu,
                         artifact="swiglu.exported.search.mlir")
    require("micro.search_space" in exported_space and "micro.objective" in exported_space,
            "search export missing decisions or objective")
    run("llk-opt", output / "swiglu.exported.search.mlir",
        artifact="swiglu.search.roundtrip.mlir")

    missing_db = output / "no-schedule.json"
    require(not missing_db.exists(), "fallback exercise requires absent %s" % missing_db)
    run("llk-opt", "--llk-to-micro=schedule-db=%s" % pass_path(missing_db), matmul,
        artifact="matmul.fallback.micro.mlir")
    emitters = ("avx2_vector_add,avx2_vector_convert,avx2_vector_silu,avx2_vector_mul,"
                "avx2_mma,avx2_reduce,avx2_copy,avx2_tile_copy,avx2_tile_store,"
                "avx2_fused_convert_silu_mul")
    target = ("target=x86-avx2 machine=machines/x86-avx2-v2.yaml "
              "layouts=mapping/x86-avx2/layouts.llkmap "
              "rules=mapping/x86-avx2/rules.llkmap emitters=" + emitters)
    options = target + " mode=exact top-k=8"
    report = output / "matmul.plan.json"
    mapped = run("llk-opt", "--micro-map=%s report=%s" % (options, pass_path(report)),
                 output / "matmul.fallback.micro.mlir", artifact="matmul.mapped.mlir")
    plan = json.loads(report.read_text())
    require(plan.get("selectedPlanId") and "micro.plan" in mapped,
            "mapping did not produce a selected plan")
    verified = run("llk-opt", "--micro-verify-mapping=" + target,
                   output / "matmul.mapped.mlir", artifact="matmul.verified.mlir")
    require(verified == mapped, "mapping verification changed the IR")
    replayed = run("llk-opt", "--micro-bind-plan=plan-id=%s %s" %
                   (plan["selectedPlanId"], options), output / "matmul.fallback.micro.mlir",
                   artifact="matmul.replayed.mlir")
    require(replayed == mapped, "plan-ID replay did not reproduce mapped IR")
    run("llk-opt", "--micro-map=target=generic-ai-accel "
        "machine=machines/generic-ai-accel-v2.yaml "
        "layouts=mapping/generic-ai-accel/layouts.llkmap "
        "rules=mapping/generic-ai-accel/rules.llkmap "
        "emitters=accel_vector_add,accel_mxu,accel_copy mode=deterministic "
        "report=%s" % pass_path(output / "accelerator.plan.json"),
        source / "test/Conversion/MicroMapping/micro_map.mlir",
        artifact="accelerator.mapped.mlir")

    frozen = run("llk-compile", "--mapping-target=x86-avx2", "--mapping-root=.",
                 "--plan-report=%s" % report, "--mapping-stop=mapped-micro",
                 output / "matmul.fallback.micro.mlir", artifact="matmul.frozen.log")
    require(frozen.startswith("mapping: target=x86-avx2"),
            "mapped compiler did not print its realization status")
    (output / "matmul.frozen.mlir").write_text(frozen.split("\n", 1)[1])
    run("llk-opt", output / "matmul.frozen.mlir", artifact="matmul.frozen.roundtrip.mlir")
    lowered = run("llk-compile", "--mapping-target=x86-avx2", "--mapping-root=.",
                  "--mapping-mode=exact", "--mapping-stop=lowered", matmul,
                  artifact="matmul.lowered.txt")
    require("scf.for" in lowered and "micro.kernel" not in lowered,
            "mapped compilation did not reach the loop lowering stage")
    executable = run("llk-compile", "--mapping-target=x86-avx2", "--mapping-root=.",
                     "--mapping-mode=exact", matmul, artifact="matmul.compilation.txt")
    require("Compilation successful" in executable,
            "mapped compiler did not complete native JIT compilation")

    for level in (0, 1):
        prediction = run("micro-perf", "--machine=machines/x86-avx2-v2.yaml",
                         "--level=%d" % level, tile, artifact="gemm.l%d.yaml" % level)
        for field in ("flops: 16384", "dram: 3072", "sram: 2048", "acc: 1024"):
            require(field in prediction, "tile report missing " + field)
        require(re.search(r"predicted_cycles: [1-9][0-9]*", prediction),
                "tile report has no positive prediction")
    run("micro-perf", "--machine=machines/x86-avx2-v2.yaml", "--level=1",
        "--format=text", tile, artifact="gemm.text.txt")
    run("micro-perf", "--machine=machines/x86-avx2-v2.yaml", "--level=1",
        output / "matmul.mapped.mlir", artifact="matmul.mapped.perf.yaml")
    workload_report = run("micro-perf", "--machine=machines/x86-avx2-v2.yaml", "--level=1",
                          output / "swiglu.micro.mlir", artifact="swiglu.perf.yaml")
    require("flops: 262144" in workload_report, "SwiGLU projection work count changed")

    tuning_args = ["--machine=machines/x86-avx2-v2.yaml", "-M=8", "-N=64", "-K=64"]
    schedule = output / "swiglu.schedule.yaml"
    run("llk-tune", "--input=%s" % space, *tuning_args, "--search=grid",
        "--max-candidates=1", "--top-k=1", "--perf-level=1", "--output=%s" % schedule)
    text = schedule.read_text()
    require("predicted_cycles:" in text and "measured: false" in text,
            "Micro tuner did not persist a predicted schedule")
    experiment = output / "swiglu.experiment.search.mlir"
    experiment.write_text(space.read_text().replace(
        'micro.param "num_threads" {kind = "integer", choices = [8 : i64]}',
        'micro.param "num_threads" {kind = "integer", choices = [1 : i64, 8 : i64]}'))
    run("llk-tune", "--input=%s" % experiment, *tuning_args, "--search=grid",
        "--max-candidates=2", "--top-k=2", "--output=%s" % (output / "expanded.yaml"))
    run("llk-tune", "--input=%s" % experiment, *tuning_args, "--search=random", "--seed=17",
        "--max-candidates=2", "--top-k=2", "--output=%s" % (output / "random.yaml"))
    run("llk-tune", "-M=8", "-N=64", "-K=64", "--dry-run",
        "-o=%s" % (output / "legacy.json"))
    require(json.loads((output / "legacy.json").read_text()).get("entries"),
            "legacy tuner did not write schedule entries")
    run("llk-bench", "-M=2", "-N=16", "-K=16", "--threads=1",
        "--warmup-ms=10", "--measure-ms=20", "--reps=2", artifact="synthetic-bench.txt")
    print("tutorial checks passed: %d commands; artifacts in %s" % (steps, output))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, RuntimeError, subprocess.TimeoutExpired) as error:
        print("tutorial check failed: %s" % error, file=sys.stderr)
        sys.exit(1)
