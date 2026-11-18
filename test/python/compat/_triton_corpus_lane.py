#!/usr/bin/env python3
"""Execute the fixed upstream Triton compatibility corpus (PF19).

This script runs in its own interpreter because it imports the qualified
upstream Triton package. It exports each declared genuine kernel through the
PF17 worker, binds the neutral template to concrete rank-two CPU tensors with
:meth:`CPULauncher.bind_triton_template`, and hands the serialized
``CompileRequest`` to a Triton-free native worker that compiles and invokes it
through the LLK C API. Nothing here claims machine code: the evidence is the
native report the compiler produced for the exact bound request.

The emitted frame carries per-fixture frontend, binding, reference, selected and
invocation status with reason codes. Rejected fixtures report the stable
diagnostic instead of counting as compatibility.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from pathlib import Path

MARKER = "LLK-TRITON-CORPUS-V1"
WORKER = Path(__file__).with_name("_native_request_worker.py")
CONFIG_FILES = (
    "machines/x86-avx2-v2.yaml",
    "mapping/x86-avx2/layouts.llkmap",
    "mapping/x86-avx2/rules.llkmap",
)
CANARY = -999.0
TOLERANCE = {"abs": 1e-5, "rel": 1e-4}


def _emit(payload):
    body = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    stream = sys.__stdout__
    stream.write(f"{MARKER}\n{body}\n")
    stream.flush()


def _values(name, shape):
    import numpy as np

    count = int(shape[0]) * int(shape[1])
    index = np.arange(count, dtype=np.float64)
    offset = 0.5 + (sum(ord(character) for character in name) % 7) * 0.25
    values = np.mod(index * 0.37 + offset, 5.0) - 2.0
    return np.ascontiguousarray(values.astype(np.float32).reshape(tuple(shape)))


def _oracle(operation, arrays, scalars):
    import numpy as np

    if operation == "add":
        return arrays["x_ptr"] + arrays["y_ptr"]
    if operation == "multiply":
        return arrays["x_ptr"] * arrays["y_ptr"]
    if operation == "matmul":
        return arrays["a_ptr"] @ arrays["b_ptr"]
    if operation == "matmul_accumulated":
        return arrays["a_ptr"] @ arrays["b_ptr"] + arrays["init_ptr"]
    if operation == "swiglu":
        gate = arrays["x_ptr"] @ arrays["gate_ptr"]
        up = arrays["x_ptr"] @ arrays["up_ptr"]
        activated = gate / (np.float32(1.0) + np.exp(-gate))
        return activated * up
    raise ValueError(f"no numerical oracle for {operation!r}")


def _check(actual, expected):
    import numpy as np

    got = actual.reshape(-1)
    want = expected.reshape(-1)
    if got.size != want.size:
        raise AssertionError("the output extent changed")
    if np.any(got == np.float32(CANARY)):
        raise AssertionError("a declared output element was never written")
    error = 0.0
    for index in range(want.size):
        difference = abs(float(got[index]) - float(want[index]))
        error = max(error, difference)
        if difference > TOLERANCE["abs"] + TOLERANCE["rel"] * abs(float(want[index])):
            raise AssertionError(
                f"element {index}: got {got[index]!r}, expected {want[index]!r}"
            )
    return error


def _options(root, backend):
    from llk.frontend import CompileOptions

    return CompileOptions(
        backend=backend,
        config_root=str(root),
        config_files={
            name: (root / name).read_text(encoding="utf-8")
            for name in CONFIG_FILES
        },
    )


def _argument_order(fixture, arrays):
    order = []
    for name, spec in fixture["signature"].items():
        if spec == "constexpr":
            order.append(fixture["constexprs"][name])
        elif spec.startswith("*"):
            order.append(arrays[name])
        else:
            order.append(fixture["scalars"][name])
    return tuple(order)


def _fixture_arrays(fixture, outputs=()):
    import numpy as np

    arrays = {}
    for name, shape in fixture["tensors"].items():
        if name in outputs:
            arrays[name] = np.full(tuple(shape), CANARY, dtype=np.float32)
        else:
            arrays[name] = _values(name, shape)
    return arrays


def _read_ports(template):
    inputs = [p.name for p in template.parameters if p.kind == "pointer" and p.access == "read"]
    outputs = [p.name for p in template.parameters if p.kind == "pointer" and p.access == "write"]
    return inputs, outputs


def _export(fixture, root, triton_python):
    import llk.triton

    return llk.triton.capture_kernel(
        root / fixture["source"],
        fixture["function"],
        fixture["signature"],
        constexprs=fixture["constexprs"],
        triton_python=triton_python,
    )


def _prepare(fixture, root, triton_python, backend, directory):
    from llk.frontend._serialize import serialize_request
    from llk.triton._driver import CPULauncher

    export = _export(fixture, root, triton_python)
    template = export.template
    inputs, outputs = _read_ports(template)
    if not outputs:
        raise AssertionError("a fixture must declare one write output")
    arrays = _fixture_arrays(fixture, outputs)
    request = CPULauncher.bind_triton_template(
        template,
        _argument_order(fixture, arrays),
        tuple(fixture["grid"]),
        _options(root, backend),
    )
    directory.mkdir(parents=True, exist_ok=True)
    request_path = directory / "request.json"
    serialized = serialize_request(request)
    request_path.write_bytes(serialized)
    input_paths = []
    for name in inputs:
        path = directory / f"input-{name}.npy"
        import numpy as np

        np.save(path, arrays[name])
        input_paths.append(str(path))
    output_paths = []
    for name in outputs:
        path = directory / f"output-{name}.npy"
        import numpy as np

        np.save(path, arrays[name])
        output_paths.append(str(path))
    return {
        "export": export,
        "arrays": arrays,
        "inputs": inputs,
        "outputs": outputs,
        "case": json.dumps(
            {
                "name": fixture["id"],
                "request": str(request_path),
                "inputs": input_paths,
                "outputs": output_paths,
            }
        ),
        "request": request,
        "serialized": serialized,
        "output_paths": output_paths,
    }


def _run_worker(cases, llk_python):
    argv = [
        sys.executable,
        "-I",
        str(WORKER),
        "--llk-python",
        llk_python,
    ]
    for case in cases:
        argv.extend(["--case", case])
    completed = subprocess.run(
        argv, capture_output=True, text=True, timeout=1800
    )
    if completed.returncode != 0:
        raise AssertionError(
            "the native request worker failed: " + completed.stderr[-4000:]
        )
    lines = completed.stdout.split("\n", 1)
    if len(lines) != 2 or lines[0] != "LLK-TRITON-NATIVE-CASES-V1":
        raise AssertionError(
            "the native request worker produced no framed result: "
            + completed.stdout[-2000:]
            + completed.stderr[-2000:]
        )
    frame = json.loads(lines[1])
    if frame.get("ok") is not True:
        raise AssertionError(f"native execution failed: {frame.get('message')}")
    return {item["name"]: item for item in frame["results"]}


def _selected_supported():
    machine = platform.machine().lower()
    return machine in ("x86_64", "amd64"), machine


def _run_rejections(manifest, root, triton_python, llk_python, workdir):
    from llk.frontend._serialize import serialize_request
    from llk.triton._driver import CPULauncher

    reports = []
    for rejection in manifest["rejections"]:
        entry = {
            "id": rejection["id"],
            "construct": rejection["construct"],
            "stage": rejection["stage"],
            "source": rejection["source"],
            "function": rejection["function"],
            "expected_code": rejection["expected_code"],
        }
        try:
            export = _export(rejection, root, triton_python)
            if rejection["stage"] == "frontend":
                entry.update(status="failed", code=None,
                             message="the construct was exported")
                reports.append(entry)
                continue
            arrays = _fixture_arrays(rejection)
            CPULauncher.bind_triton_template(
                export.template,
                _argument_order(rejection, arrays),
                tuple(rejection["grid"]),
                _options(root, "reference"),
            )
            entry.update(status="failed", code=None,
                         message="the construct was bound")
        except Exception as error:  # structured diagnostics only
            entry.update(
                status="rejected",
                code=getattr(error, "code", None),
                message=str(error),
            )
        reports.append(entry)
    return reports


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--llk-python", required=True)
    parser.add_argument("--root", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--triton-python", required=True)
    parser.add_argument("--workdir", required=True)
    args = parser.parse_args(argv)
    sys.stdout = sys.stderr
    try:
        sys.path.insert(0, args.llk_python)
        import numpy as np

        root = Path(args.root).resolve()
        manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
        workdir = Path(args.workdir)
        llk_python = args.llk_python

        prepared = {}
        cases = []
        for fixture in manifest["fixtures"]:
            directory = workdir / "reference" / fixture["id"]
            prepared[fixture["id"]] = _prepare(
                fixture, root, args.triton_python, "reference", directory
            )
            cases.append(prepared[fixture["id"]]["case"])
        reference_reports = _run_worker(cases, llk_python)

        selected_ok, machine = _selected_supported()
        selected_entries = {}
        if selected_ok:
            selected_cases = []
            for fixture in manifest["fixtures"]:
                directory = workdir / "selected" / fixture["id"]
                prepared[fixture["id"]]["selected"] = _prepare(
                    fixture, root, args.triton_python, "selected-target", directory
                )
                selected_cases.append(prepared[fixture["id"]]["selected"]["case"])
            selected_reports = _run_worker(selected_cases, llk_python)

        fixtures_out = []
        executed_ids = []
        for fixture in manifest["fixtures"]:
            item = prepared[fixture["id"]]
            export = item["export"]
            record = {
                "id": fixture["id"],
                "class": "executable",
                "operation": fixture["operation"],
                "source": fixture["source"],
                "function": fixture["function"],
                "coverage": fixture["coverage"],
                "frontend": {
                    "status": "ok",
                    "ttir_hash": export.template.ttir_hash,
                    "template_identity": export.template.identity(),
                    "toolchain": export.toolchain.to_json(),
                },
                "binding": {
                    "status": "ok",
                    "program_identity": hashlib_hex(item["serialized"]),
                    "grid": list(fixture["grid"]),
                    "block": list(item["request"].program.launch.block),
                    "native_grid": list(item["request"].program.launch.grid),
                },
            }
            expected = _oracle(
                fixture["operation"], item["arrays"], fixture.get("scalars", {})
            )
            report = reference_reports[fixture["id"]]
            actual = np.load(item["output_paths"][0])
            error = _check(actual, expected)
            record["reference"] = _report_entry(report, 1, error)
            if selected_ok:
                selected = selected_reports[fixture["id"]]
                selected_output = np.load(item["selected"]["output_paths"][0])
                selected_error = _check(selected_output, expected)
                record["selected"] = _report_entry(selected, 1, selected_error)
                record["selected"]["status"] = (
                    "ok" if selected.get("backend") == "selected-target" else "failed"
                )
            else:
                record["selected"] = {
                    "status": "unavailable",
                    "reason": "host_isa_unsupported",
                    "host_machine": machine,
                    "target": "x86-avx2",
                }
            fixtures_out.append(record)
            executed_ids.append(fixture["id"])

        rejections = _run_rejections(
            manifest, root, args.triton_python, llk_python, workdir / "rejections"
        )
        _emit(
            {
                "protocol": 1,
                "corpus": manifest["kind"],
                "manifest_schema_version": manifest["schema_version"],
                "toolchain": {
                    "triton_version": export.toolchain.triton_version,
                    "target": export.toolchain.target,
                    "source_revision": export.toolchain.source_revision,
                },
                "host": {
                    "machine": machine,
                    "selected_target_supported": selected_ok,
                },
                "fixtures": fixtures_out,
                "rejections": rejections,
                "executed_fixture_ids": executed_ids,
            }
        )
        return 0
    except BaseException as error:
        _emit({"ok": False, "message": f"{type(error).__name__}: {error}"})
        return 1


def hashlib_hex(value):
    import hashlib

    return hashlib.sha256(value).hexdigest()


def _report_entry(report, invocations, error):
    return {
        "status": "ok",
        "backend": report["backend"],
        "source_adapter": report["source_adapter"],
        "plan_id": report["plan_id"],
        "abi_hash": report["abi_hash"],
        "native_request_identity": report["native_request_identity"],
        "frontend_identity": report["frontend_identity"],
        "compiler_identity": report["compiler_identity"],
        "workload_identity": report["workload_identity"],
        "source_adapter_identity": report["source_adapter_identity"],
        "target_hash": report["target_hash"],
        "machine_hash": report["machine_hash"],
        "rules_hash": report["rules_hash"],
        "selected_group_count": report["selected_group_count"],
        "reference_group_count": report["reference_group_count"],
        "stages": report["stages"],
        "native_invocations": invocations,
        "numerical": {"max_abs_error": error, "tolerance": TOLERANCE},
    }


if __name__ == "__main__":
    sys.exit(main())
