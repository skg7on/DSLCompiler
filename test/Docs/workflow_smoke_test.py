#!/usr/bin/env python3
"""Verify WorkflowSmoke fails closed for broken fixtures and CLI options."""

import argparse
import copy
import json
import pathlib
import subprocess
import sys
import tempfile


def run_smoke(args, manifest):
    with tempfile.TemporaryDirectory(prefix="llk-workflow-smoke-negative-") as temp:
        root = pathlib.Path(temp)
        manifest_path = root / "manifest.json"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        binary_dir = root / "build"
        return subprocess.run(
            [
                sys.executable,
                args.smoke,
                str(manifest_path),
                args.repo,
                str(binary_dir),
                args.llk_opt,
                args.llk_compile,
                args.llk_tune,
                args.micro_perf,
            ],
            cwd=args.repo,
            capture_output=True,
            text=True,
            check=False,
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest")
    parser.add_argument("repo")
    parser.add_argument("smoke")
    parser.add_argument("llk_opt")
    parser.add_argument("llk_compile")
    parser.add_argument("llk_tune")
    parser.add_argument("micro_perf")
    args = parser.parse_args()

    manifest = json.loads(pathlib.Path(args.manifest).read_text(encoding="utf-8"))
    missing_fixture = copy.deepcopy(manifest)
    missing_fixture["commands"][0]["argv"][5] = "{artifacts}/missing-source-root"
    result = run_smoke(args, missing_fixture)
    assert result.returncode != 0, "WorkflowSmoke accepted a missing fixture root"
    assert "missing fixture" in result.stderr, result.stderr

    invalid_option = copy.deepcopy(manifest)
    invalid_option["commands"] = [
        {
            "name": "invalid-pass-option",
            "argv": [
                "{llk_opt}",
                "--not-a-real-pass-option",
                "{repo}/test/Conversion/MicroMapping/matmul_e2e.mlir",
            ],
        }
    ]
    result = run_smoke(args, invalid_option)
    assert result.returncode != 0, "WorkflowSmoke accepted an invalid pass option"
    assert "invalid-pass-option failed" in result.stderr, result.stderr


if __name__ == "__main__":
    main()
