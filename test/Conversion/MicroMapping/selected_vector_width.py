#!/usr/bin/env python3
"""Prove selected AVX2 widths and fused groups reach explicit Vector IR."""

import os
import re
import shutil
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
    llk_compile, source_dir, work_dir = sys.argv[1:]
    os.makedirs(work_dir, exist_ok=True)
    width4_root = os.path.join(work_dir, "width4-target")
    os.makedirs(os.path.join(width4_root, "machines"), exist_ok=True)
    os.makedirs(os.path.join(width4_root, "mapping", "x86-avx2"),
                exist_ok=True)
    machine = os.path.join(source_dir, "machines/x86-avx2-v2.yaml")
    with open(machine) as handle:
        profile = handle.read()
    narrow_profile = profile.replace(
        "lanes: {f32: 8,", "lanes: {f32: 4,", 1)
    assert narrow_profile != profile, "the f32 lane profile was not found"
    with open(os.path.join(width4_root, "machines/x86-avx2-v2.yaml"),
              "w") as handle:
        handle.write(narrow_profile)
    for relative in ("mapping/x86-avx2/layouts.llkmap",
                     "mapping/x86-avx2/rules.llkmap"):
        destination = os.path.join(width4_root, relative)
        os.makedirs(os.path.dirname(destination), exist_ok=True)
        shutil.copy2(os.path.join(source_dir, relative), destination)

    results = {}
    for width, root in ((4, width4_root), (8, source_dir)):
        fixture = os.path.join(
            source_dir, "test/Execution/Inputs/issue129/width%d.mlir" % width)
        output = run([
            llk_compile, "--mapping-target=x86-avx2",
            "--mapping-root=" + root, "--mapping-mode=exact",
            "--mapping-backend=selected-target", "--mapping-stop=lowered",
            "--emit=mlir", fixture])
        assert "backend=selected-target" in output
        assert "backend-groups-realized=1" in output
        assert "reference-groups-lowered=0" in output
        vector_type = re.compile(r"vector<%dxf32>" % width)
        assert vector_type.search(output), (
            "selected VW%d did not reach explicit Vector arithmetic\n%s" %
            (width, output))
        assert re.search(r"arith\.addf .* : vector<%dxf32>" % width, output), (
            "selected VW%d did not control the arithmetic vector type\n%s" %
            (width, output))
        assert "vector.transfer_read" in output
        assert "vector.transfer_write" in output
        results[width] = output
        with open(os.path.join(work_dir, "selected-width-%d.mlir" % width),
                  "w") as handle:
            handle.write(output)

    assert results[4] != results[8], "different selected widths emitted identical IR"

    fused_fixture = os.path.join(
        source_dir, "test/Execution/Inputs/issue129/selected_fused.mlir")
    fused = run([
        llk_compile, "--mapping-target=x86-avx2",
        "--mapping-root=" + source_dir, "--mapping-mode=exact",
        "--mapping-backend=selected-target", "--mapping-stop=lowered",
        "--emit=mlir", fused_fixture])
    assert "backend-groups-realized=1" in fused
    assert fused.count("vector.transfer_write") == 1, (
        "fused convert/silu/mul must produce one vector output body\n%s" % fused)
    assert fused.count("vector.transfer_read") == 2, (
        "fused convert/silu/mul must read its input and gate once\n%s" % fused)
    assert "vector<1x8xf32>" in fused
    assert re.search(r"math\.exp .* : vector<8xf32>", fused)
    assert re.search(r"arith\.mulf .* : vector<8xf32>", fused)
    with open(os.path.join(work_dir, "selected-fused.mlir"), "w") as handle:
        handle.write(fused)


if __name__ == "__main__":
    main()
