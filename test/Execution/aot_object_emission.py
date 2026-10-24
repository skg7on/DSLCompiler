#!/usr/bin/env python3
"""AOT object emission: ELF identity and a complete sidecar (E10 task 1)."""

import json
import os
import struct
import subprocess
import sys


def run(command, expect_success=True):
    result = subprocess.run(command, capture_output=True, text=True)
    if expect_success and result.returncode != 0:
        raise SystemExit("command failed: %s\n%s\n%s" %
                         (" ".join(command), result.stdout, result.stderr))
    return result


def main():
    llk_compile, source_dir, work_dir = sys.argv[1:]
    os.makedirs(work_dir, exist_ok=True)
    source = os.path.join(source_dir,
                          "test/Conversion/MicroMapping/Inputs/arm",
                          "vector_add_f32.mlir")
    obj = os.path.join(work_dir, "arm_vector_add_f32.o")
    identity = os.path.join(work_dir, "identity.json")
    common = [
        llk_compile,
        "--mapping-target=arm-sve",
        "--mapping-root=" + source_dir,
        "--mapping-backend=reference",
        "--mapping-stop=lowered",
        "--emit=object",
        "--target-features=+sve",
    ]
    run(common + ["--target-triple=aarch64-unknown-linux-gnu",
                  "--output=" + obj, "--emit-identity=" + identity, source])

    data = open(obj, "rb").read()
    assert data[:4] == b"\x7fELF", "output is not an ELF object"
    assert struct.unpack_from("<H", data, 18)[0] == 183, "not EM_AARCH64"

    record = json.load(open(identity))
    for key in ("schemaVersion", "compilerVersion", "llvmVersion",
                "targetTriple", "targetFeatures", "entrySymbol",
                "wrapperSymbol", "abiVersion", "abiHash", "sourceGraphHash",
                "planId", "machineHash", "targetHash", "selectedGroupsVerified",
                "backendGroupsRealized", "referenceGroupsLowered",
                "executionIdentity"):
        assert key in record, "identity has no " + key
    assert record["targetTriple"] == "aarch64-unknown-linux-gnu"
    assert record["targetFeatures"] == ["+sve"]
    assert record["entrySymbol"] == "arm_vector_add_f32"
    assert record["wrapperSymbol"] == "_mlir_ciface_arm_vector_add_f32"
    assert record["wrapperSymbol"].encode() in data, "wrapper symbol not exported"
    # `sve_vector_add` implements a lowering (the selected SVE backend pass), so
    # this reference-backend run verifies the selected group and carries it
    # through generic lowering: nothing falls back to the reference bridge and
    # no selected backend group is realized. The identity must record that
    # split rather than a blanket "reference-lowered" count.
    assert record["selectedGroupsVerified"] == 1
    assert record["referenceGroupsLowered"] == 0
    assert record["backendGroupsRealized"] == 0

    missing = run(common[:-1] + ["--output=" + obj, "--emit-identity=" + identity,
                                source], expect_success=False)
    assert missing.returncode != 0
    assert "--target-triple" in missing.stderr, missing.stderr
    print("aot object emission: ok")


if __name__ == "__main__":
    sys.exit(main())
