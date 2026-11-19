#!/usr/bin/env python3
"""Compile and invoke serialized frontend requests with the LLK C API only.

This worker never imports upstream Triton.  It is how the compatibility lane
proves that a request bound from a genuine PF17 template is the exact native
contract, even on hosts where Triton's LLVM and LLK's LLVM cannot share one
process.

Each case names a serialized ``CompileRequest``, its concrete input arrays and
its output arrays. Every field of the native compilation report that identifies
the frontend, the plan and the selected/reference groups is returned, so a
report is evidence of what executed rather than of what was requested.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

MARKER = "LLK-TRITON-NATIVE-CASES-V1"


def _emit(payload):
    body = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    stream = sys.__stdout__
    stream.write(f"{MARKER}\n{body}\n")
    stream.flush()


def _paths(case, key, legacy):
    values = case.get(key)
    if values is None:
        values = [case[name] for name in legacy]
    return [str(value) for value in values]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--llk-python", required=True)
    parser.add_argument("--case", action="append", default=[])
    args = parser.parse_args(argv)
    sys.stdout = sys.stderr
    try:
        sys.path.insert(0, args.llk_python)
        import numpy as np

        import llk.runtime._ffi as ffi
        from llk.frontend._serialize import deserialize_request

        results = []
        for raw in args.case:
            spec = json.loads(raw)
            request = deserialize_request(
                Path(spec["request"]).read_bytes()
            )
            inputs = tuple(
                np.load(path) for path in _paths(spec, "inputs", ("x", "y"))
            )
            outputs = tuple(
                np.load(path) for path in _paths(spec, "outputs", ("out",))
            )
            kernel = ffi.compile_request(request)
            try:
                kernel.invoke(inputs, outputs)
            finally:
                report = kernel.report
                kernel.close()
            for path, value in zip(_paths(spec, "outputs", ("out",)), outputs):
                np.save(path, value)
            results.append(
                {
                    "name": spec["name"],
                    "backend": report.backend,
                    "source_adapter": report.source_adapter,
                    "plan_id": report.plan_id,
                    "abi_hash": report.abi_hash,
                    "native_request_identity": report.native_request_identity,
                    "frontend_identity": report.frontend_identity,
                    "compiler_identity": report.compiler_identity,
                    "workload_identity": report.workload_identity,
                    "source_adapter_identity": report.source_adapter_identity,
                    "target_hash": report.target_hash,
                    "machine_hash": report.machine_hash,
                    "rules_hash": report.rules_hash,
                    "selected_group_count": report.selected_group_count,
                    "reference_group_count": report.reference_group_count,
                    "stages": list(report.stages),
                }
            )
        _emit({"ok": True, "results": results})
        return 0
    except BaseException as error:  # never escape unframed
        _emit({"ok": False, "message": f"{type(error).__name__}: {error}"})
        return 1


if __name__ == "__main__":
    sys.exit(main())
