#!/usr/bin/env python3
"""Compile and invoke serialized frontend requests with the LLK C API only.

This worker never imports upstream Triton.  It is how the compatibility lane
proves that a request bound from a genuine PF17 template is the exact native
contract, even on hosts where Triton's LLVM and LLK's LLVM cannot share one
process.
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
            kernel = ffi.compile_request(request)
            inputs = (np.load(spec["x"]), np.load(spec["y"]))
            outputs = (np.load(spec["out"]),)
            try:
                kernel.invoke(inputs, outputs)
            finally:
                report = kernel.report
                kernel.close()
            np.save(spec["out"], outputs[0])
            results.append(
                {
                    "name": spec["name"],
                    "backend": report.backend,
                    "source_adapter": report.source_adapter,
                    "plan_id": report.plan_id,
                    "abi_hash": report.abi_hash,
                    "native_request_identity": report.native_request_identity,
                }
            )
        _emit({"ok": True, "results": results})
        return 0
    except BaseException as error:  # never escape unframed
        _emit({"ok": False, "message": f"{type(error).__name__}: {error}"})
        return 1


if __name__ == "__main__":
    sys.exit(main())
