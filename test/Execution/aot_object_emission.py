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
    # `sve_vector_add` implements a lowering, so this reference-backend run
    # verifies the selected group and hands it to the target emitter; no group
    # falls back to the bridge and the selected backend pass does not run. The
    # unimplemented dtypes (bf16, i8) are the ones the bridge carries, and they
    # are covered by the ArmSVEBackend tests.
    assert record["selectedGroupsVerified"] == 1
    assert record["referenceGroupsLowered"] == 0
    assert record["backendGroupsRealized"] == 0

    # The selected backend is the executable subset this target advertises. Its
    # artifact is prepared through the same ownership stage as the reference
    # one, which is what catches a masked-vector form that stage cannot hold.
    selected_obj = os.path.join(work_dir, "arm_vector_add_f32_selected.o")
    selected_identity = os.path.join(work_dir, "identity_selected.json")
    selected = [
        llk_compile,
        "--mapping-target=arm-sve",
        "--mapping-root=" + source_dir,
        "--mapping-backend=selected-target",
        "--mapping-stop=lowered",
        "--emit=object",
        "--target-features=+sve",
        "--target-triple=aarch64-unknown-linux-gnu",
        "--output=" + selected_obj,
        "--emit-identity=" + selected_identity,
        source,
    ]
    run(selected)

    selected_data = open(selected_obj, "rb").read()
    assert selected_data[:4] == b"\x7fELF", "selected output is not an ELF object"
    assert struct.unpack_from("<H", selected_data, 18)[0] == 183, \
        "selected output is not EM_AARCH64"
    selected_record = json.load(open(selected_identity))
    assert selected_record["wrapperSymbol"] == "_mlir_ciface_arm_vector_add_f32"
    assert selected_record["wrapperSymbol"].encode() in selected_data, \
        "selected wrapper symbol not exported"
    assert selected_record["selectedGroupsVerified"] == 1
    assert selected_record["backendGroupsRealized"] == 1
    assert selected_record["referenceGroupsLowered"] == 0

    missing = run(common[:-1] + ["--output=" + obj, "--emit-identity=" + identity,
                                source], expect_success=False)
    assert missing.returncode != 0
    assert "--target-triple" in missing.stderr, missing.stderr
    print("aot object emission: ok")


if __name__ == "__main__":
    sys.exit(main())
