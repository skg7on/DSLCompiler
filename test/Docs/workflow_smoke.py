#!/usr/bin/env python3
"""Run an allowlisted set of documented compiler workflows with argv arrays."""

import argparse
import json
import pathlib
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest")
    parser.add_argument("repo")
    parser.add_argument("binary_dir")
    parser.add_argument("llk_opt")
    parser.add_argument("llk_compile")
    parser.add_argument("llk_tune")
    parser.add_argument("micro_perf")
    args = parser.parse_args()

    repo = pathlib.Path(args.repo).resolve()
    artifacts = pathlib.Path(args.binary_dir).resolve() / "WorkflowSmoke"
    artifacts.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(pathlib.Path(args.manifest).read_text(encoding="utf-8"))
    documentation = (repo / "docs/design/micro-ir-mapping-workflow.md").read_text(
        encoding="utf-8")
    for token in manifest["documentation_tokens"]:
        if token not in documentation:
            raise SystemExit("documented workflow drift: missing %r" % token)

    values = {
        "python": sys.executable,
        "repo": str(repo),
        "artifacts": str(artifacts),
        "llk_opt": str(pathlib.Path(args.llk_opt).resolve()),
        "llk_compile": str(pathlib.Path(args.llk_compile).resolve()),
        "llk_tune": str(pathlib.Path(args.llk_tune).resolve()),
        "micro_perf": str(pathlib.Path(args.micro_perf).resolve()),
    }
    for command in manifest["commands"]:
        argv = [part.format_map(values) for part in command["argv"]]
        result = subprocess.run(argv, cwd=repo, capture_output=True,
                                text=True, shell=False)
        if result.returncode:
            raise SystemExit("%s failed (%d):\n%s" %
                             (command["name"], result.returncode,
                              result.stderr.strip()))
        if "capture" in command:
            captured = artifacts / command["capture"]
            captured.parent.mkdir(parents=True, exist_ok=True)
            captured.write_text(result.stdout, encoding="utf-8")
        for relative in command.get("outputs", []):
            output = artifacts / relative
            if not output.is_file():
                raise SystemExit("%s did not produce %s" %
                                 (command["name"], relative))
    print("workflow smoke commands and documented options passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
