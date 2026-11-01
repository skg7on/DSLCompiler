#!/usr/bin/env python3
"""End-to-end checks for ``llk-compile --input-format=frontend-json``.

The compiler CLI must route a canonical frontend request through the same
native service the Python package and the C API use. That claim is tested by
compiling one materialized request both ways -- CLI and ``LLKFrontendCAPI`` --
and requiring the identity-bearing report fields to agree. The same test also
pins the boundary: malformed requests fail with the service's own error codes,
CLI options that would compete with the request fail loudly, and legacy
``tt.*`` input gets an actionable unsupported-input message instead of being
silently treated as production Triton support.

Usage: cli.py <llk-compile> <LLKFrontendCAPI library> <source-root> <work-dir>
"""

import ctypes
import json
import shutil
import subprocess
import sys
from pathlib import Path

# The frontend report fields that identify what was compiled. They must not
# depend on the entry path: a process-local handle or a timing may differ, but
# the request, workload, target, plan and ABI identities may not.
IDENTITY_FIELDS = (
    "native_request_identity",
    "frontend_identity",
    "workload_identity",
    "source_adapter_identity",
    "target_hash",
    "machine_hash",
    "rules_hash",
    "plan_id",
    "abi_hash",
)

CONFIG_FILES = (
    "machines/x86-avx2-v2.yaml",
    "mapping/x86-avx2/layouts.llkmap",
    "mapping/x86-avx2/rules.llkmap",
)


class CheckFailure(Exception):
    pass


def check(condition, message):
    if not condition:
        raise CheckFailure(message)


def contains_code(text, code):
    """Recognize a diagnostic code in the plain or JSON error forms."""
    return f"{code}:" in text or f'"{code}"' in text


class LLKString(ctypes.Structure):
    _fields_ = [("data", ctypes.c_void_p), ("length", ctypes.c_size_t)]


class CAPI:
    """Minimal ctypes binding for the frontend v1 ABI."""

    def __init__(self, library):
        self.lib = ctypes.CDLL(str(library))
        self.lib.llk_frontend_api_version.restype = ctypes.c_uint32
        self.lib.llk_frontend_compile_v1.argtypes = [
            ctypes.c_char_p,
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_void_p),
        ]
        self.lib.llk_frontend_compile_v1.restype = ctypes.c_int
        self.lib.llk_frontend_report_v1.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(LLKString),
            ctypes.POINTER(ctypes.c_void_p),
        ]
        self.lib.llk_frontend_report_v1.restype = ctypes.c_int
        self.lib.llk_frontend_inspect_v1.argtypes = [
            ctypes.c_void_p,
            ctypes.c_char_p,
            ctypes.POINTER(LLKString),
            ctypes.POINTER(ctypes.c_void_p),
        ]
        self.lib.llk_frontend_inspect_v1.restype = ctypes.c_int
        self.lib.llk_frontend_error_json_v1.argtypes = [ctypes.c_void_p]
        self.lib.llk_frontend_error_json_v1.restype = LLKString
        self.lib.llk_frontend_error_free_v1.argtypes = [ctypes.c_void_p]
        self.lib.llk_frontend_release_v1.argtypes = [ctypes.c_void_p]
        check(self.lib.llk_frontend_api_version() == 1,
              "frontend C API is not version 1")

    @staticmethod
    def _text(value):
        if not value.data:
            return ""
        return ctypes.string_at(value.data, value.length).decode("utf-8")

    def _error(self, error):
        if not error:
            return None
        try:
            return json.loads(self._text(self.lib.llk_frontend_error_json_v1(error)))
        finally:
            self.lib.llk_frontend_error_free_v1(error)

    def compile(self, text):
        """Return (executable or None, error or None)."""
        handle = ctypes.c_void_p()
        error = ctypes.c_void_p()
        encoded = text.encode("utf-8")
        status = self.lib.llk_frontend_compile_v1(
            encoded, len(encoded),
            ctypes.byref(handle), ctypes.byref(error))
        diagnostic = self._error(error)
        check((status == 0) == (bool(handle.value) and diagnostic is None),
              "frontend C API returned an inconsistent compile status")
        return (handle if status == 0 else None), diagnostic

    def report(self, handle):
        out = LLKString()
        error = ctypes.c_void_p()
        status = self.lib.llk_frontend_report_v1(
            handle, ctypes.byref(out), ctypes.byref(error))
        check(status == 0, f"frontend report failed: {self._error(error)}")
        return self._text(out)

    def inspect(self, handle, stage):
        out = LLKString()
        error = ctypes.c_void_p()
        status = self.lib.llk_frontend_inspect_v1(
            handle, stage.encode("utf-8"), ctypes.byref(out),
            ctypes.byref(error))
        check(status == 0, f"frontend inspect failed: {self._error(error)}")
        return self._text(out)

    def release(self, handle):
        self.lib.llk_frontend_release_v1(handle)


class Harness:
    def __init__(self, compiler, library, source_root, work_dir):
        self.compiler = Path(compiler)
        self.capi = CAPI(library)
        self.source_root = Path(source_root)
        self.work_dir = Path(work_dir)
        if self.work_dir.exists():
            shutil.rmtree(self.work_dir)
        self.work_dir.mkdir(parents=True)
        self.checks = 0

    def check(self, condition, message):
        self.checks += 1
        check(condition, message)

    def run(self, *args, timeout=300):
        return subprocess.run(
            [str(self.compiler), *args],
            capture_output=True, text=True, timeout=timeout)

    def materialize(self):
        """A complete request: the checked-in program plus frozen target files.

        The request embeds the machine profile and target policy because the
        native service deliberately never reads a live checkout path.
        """
        fixture = self.source_root / "test/Frontend/Inputs/add-v1.json"
        request = json.loads(fixture.read_text())
        request["options"]["config_files"] = {
            name: (self.source_root / name).read_text()
            for name in CONFIG_FILES
        }
        path = self.work_dir / "add-v1.complete.json"
        path.write_text(json.dumps(request, sort_keys=True,
                                   separators=(",", ":")) + "\n")
        return path, path.read_text()

    def compile_cli(self, request, *extra):
        result = self.run("--input-format=frontend-json", *extra, str(request))
        self.check(result.returncode == 0,
                   f"CLI compilation failed: {result.stderr}")
        try:
            return json.loads(result.stdout)
        except json.JSONDecodeError as error:
            raise CheckFailure(
                f"CLI did not print a JSON report: {error}\n{result.stdout}")

    def expect_cli_failure(self, expected_code, *args):
        result = self.run(*args)
        self.check(result.returncode != 0,
                   f"CLI accepted a request it must reject: {' '.join(args)}")
        self.check(contains_code(result.stderr, expected_code),
                   f"CLI stderr does not name {expected_code}:\n{result.stderr}")
        return result


def main():
    if len(sys.argv) != 5:
        print(__doc__, file=sys.stderr)
        return 2
    harness = Harness(*sys.argv[1:])
    request_path, request_text = harness.materialize()

    # 1. The CLI compiles the request through the shared service and prints its
    #    report. Registered stages and real artifact identities are present.
    cli_report = harness.compile_cli(request_path)
    harness.check(cli_report.get("backend") == "reference",
                  "CLI report backend is not reference")
    harness.check(cli_report.get("source_adapter") == "linalg-elementwise-v1",
                  "CLI report source adapter is not the imported family")
    harness.check(tuple(cli_report.get("inspection_stages", ())) ==
                  ("frontend", "llk", "micro"),
                  "CLI report does not register the three inspection stages")
    for stage in ("llk", "micro", "lowered", "executable"):
        harness.check(stage in cli_report.get("stages", ()),
                      f"CLI report omits the {stage} stage")
    for field in IDENTITY_FIELDS:
        harness.check(bool(cli_report.get(field)),
                      f"CLI report field {field} is empty")

    # 2. The C API compiles the identical canonical request. Every identity
    #    field must agree: one service, one answer.
    handle, diagnostic = harness.capi.compile(request_text)
    harness.check(handle is not None,
                  f"C API compilation failed: {diagnostic}")
    capi_report = json.loads(harness.capi.report(handle))
    harness.check(cli_report["abi_hash"] == capi_report["abi_hash"],
                  "CLI and C API report different ABI hashes")
    harness.check(cli_report["plan_id"] == capi_report["plan_id"],
                  "CLI and C API report different plan ids")
    for field in IDENTITY_FIELDS:
        harness.check(cli_report[field] == capi_report[field],
                      f"CLI and C API disagree on {field}")

    # 3. Inspection is the same registered IR at every stage.
    for stage in ("frontend", "llk", "micro"):
        native = harness.capi.inspect(handle, stage)
        cli = harness.run("--input-format=frontend-json", f"--inspect={stage}",
                          str(request_path))
        harness.check(cli.returncode == 0,
                      f"CLI inspection of {stage} failed: {cli.stderr}")
        harness.check(cli.stdout == native,
                      f"CLI and C API inspection differ at stage {stage}")
    micro = harness.run("--input-format=frontend-json", "--inspect=micro",
                        str(request_path))
    harness.check("micro.kernel" in micro.stdout,
                  "micro inspection does not contain a micro.kernel")
    harness.capi.release(handle)

    # 4. Malformed schema, version and opcode fail with the service's code on
    #    both entry paths; the CLI does not invent its own parsing.
    fixture = json.loads(request_text)
    malformed = harness.work_dir / "malformed.json"
    malformed.write_text("{ not json")
    unknown_opcode = json.loads(request_text)
    unknown_opcode["program"]["operations"][0]["opcode"] = "not_an_opcode"
    opcode_path = harness.work_dir / "bad-opcode.json"
    opcode_path.write_text(json.dumps(unknown_opcode))
    bad_version = harness.source_root / "test/Frontend/Inputs/bad-version-v1.json"
    for path, detail in ((malformed, "malformed JSON"),
                         (bad_version, "unsupported schema version"),
                         (opcode_path, "unknown opcode")):
        text = path.read_text()
        _, diagnostic = harness.capi.compile(text)
        harness.check(diagnostic is not None and
                      diagnostic.get("code") == "invalid_frontend_schema",
                      f"C API did not reject {detail}: {diagnostic}")
        result = harness.expect_cli_failure(
            "invalid_frontend_schema", "--input-format=frontend-json", str(path))
        # The CLI reports the service's own message, not a wrapper's.
        harness.check(diagnostic["message"] in result.stderr,
                      f"CLI rejection of {detail} lost the native message:\n"
                      f"{result.stderr}")

    # 5. A stage the service does not provide stays unavailable.
    harness.expect_cli_failure(
        "inspection_unavailable", "--input-format=frontend-json",
        "--inspect=asm", str(request_path))

    # 6. The request is the single source of compile options. CLI flags that
    #    would override or be silently dropped conflict with it.
    for flag in ("--mapping-target=x86-avx2", "--mapping-mode=exact",
                 "--emit=mlir", "--mapping-stop=lowered"):
        harness.expect_cli_failure(
            "conflicting_compile_options", "--input-format=frontend-json",
            flag, str(request_path))

    # 7. Default MLIR mode still works and there is no production escape hatch
    #    for unregistered dialects.
    legacy = harness.source_root / "test/Conversion/LLKToMicro/search_space_from_schedule.mlir"
    result = harness.run("--input-format=mlir", "--emit=micro-search",
                         str(legacy))
    harness.check(result.returncode == 0,
                  f"default MLIR mode broke: {result.stderr}")
    harness.check("micro.search_space" in result.stdout,
                  "MLIR micro-search output lost its search space")

    triton = harness.work_dir / "legacy-triton.mlir"
    triton.write_text("""
module {
  func.func @kernel() {
    %0 = tt.get_program_id x : i32
    return
  }
}
""")
    result = harness.expect_cli_failure("unsupported_input", str(triton))
    harness.check("--input-format=frontend-json" in result.stderr,
                  "tt.* rejection does not name the supported entry path")
    escape = harness.run("--allow-unregistered-dialect", str(triton))
    harness.check(escape.returncode != 0,
                  "llk-compile still exposes an unregistered-dialect escape")

    print(f"FrontendCLI: {harness.checks} checks passed")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except CheckFailure as failure:
        print(f"FrontendCLI: FAIL: {failure}", file=sys.stderr)
        sys.exit(1)
