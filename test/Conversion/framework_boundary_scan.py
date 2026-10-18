#!/usr/bin/env python3
"""Verify the generic mapping/machine libraries stay target-neutral.

Checks two boundaries that unit tests cannot express:

* no target name and no semantic operator label is read as policy in generic
  ``LLKMapping``/``LLKMachine`` code (comments are not code); and
* neither library links the performance model, runtime or JIT.

A finding is a hard failure: target policy belongs in YAML/LLKMap and target
emitters, and semantic adapters belong above the generic core.
"""

import pathlib
import re
import sys

# Strip line/block comments while preserving string literals, which are where a
# name used as policy would actually live.
_COMMENT_OR_STRING = re.compile(
    r'//[^\n]*|/\*.*?\*/|"(?:\\.|[^"\\])*"', re.S)


def strip_comments(text: str) -> str:
    def replace(match: re.Match) -> str:
        value = match.group(0)
        if value.startswith('"'):
            return value
        return "\n" * value.count("\n") or " "

    return _COMMENT_OR_STRING.sub(replace, text)


TARGET_TOKEN = re.compile(
    r"\b(avx2|aarch64|sve|sme|x86|generic-ai-accel)\b|accel_[a-z]",
    re.IGNORECASE)
SEMANTIC_TOKEN = re.compile(
    r"\b(swiglu|matmul|attention|rope)\b", re.IGNORECASE)
FORBIDDEN_LINK = re.compile(
    r"\b(LLKPerf|LLKRuntime|LLKJit|LLKMappedCompilation|LLKMappedTuning)\b")

GENERIC_DIRECTORIES = (
    "lib/Mapping",
    "lib/Machine",
    "include/LLK/Mapping",
    "include/LLK/Machine",
)
GENERIC_CMAKELISTS = (
    "lib/Mapping/CMakeLists.txt",
    "lib/Machine/CMakeLists.txt",
)


def line_of(text: str, offset: int) -> int:
    return text.count("\n", 0, offset) + 1


def main(root_argument: str) -> int:
    root = pathlib.Path(root_argument).resolve()
    errors: list[str] = []

    for directory_name in GENERIC_DIRECTORIES:
        directory = root / directory_name
        if not directory.is_dir():
            errors.append(f"missing generic directory: {directory_name}")
            continue
        for path in sorted(directory.rglob("*")):
            if path.suffix not in (".cpp", ".h") or not path.is_file():
                continue
            relative = path.relative_to(root)
            code = strip_comments(path.read_text(encoding="utf-8"))
            for match in TARGET_TOKEN.finditer(code):
                errors.append(
                    f"{relative}:{line_of(code, match.start())}: "
                    f"target-name policy '{match.group(0)}' in generic code")
            for match in SEMANTIC_TOKEN.finditer(code):
                errors.append(
                    f"{relative}:{line_of(code, match.start())}: "
                    f"semantic operator label '{match.group(0)}' in generic code")

    for cmake_name in GENERIC_CMAKELISTS:
        path = root / cmake_name
        if not path.is_file():
            errors.append(f"missing library declaration: {cmake_name}")
            continue
        text = path.read_text(encoding="utf-8")
        for match in FORBIDDEN_LINK.finditer(text):
            errors.append(
                f"{cmake_name}: forbidden link dependency '{match.group(0)}'")

    if errors:
        for error in errors:
            print(f"boundary: {error}")
        print(f"framework boundary scan: {len(errors)} violation(s)")
        return 1
    print("framework boundary scan: generic mapping/machine libraries are "
          "target-neutral and free of perf/runtime/JIT links")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "."))
