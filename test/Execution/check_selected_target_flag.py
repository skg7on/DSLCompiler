#!/usr/bin/env python3
"""Exercise the required selected-target command-line contract."""

import os
import subprocess
import sys


def run(binary, *arguments, env=None):
    environment = os.environ.copy()
    environment.update(env or {})
    return subprocess.run(
        [binary, *arguments], check=False, capture_output=True, text=True,
        env=environment
    )


def check_pair(binary, suite, gtest_filter):
    reference = run(binary, "--gtest_list_tests",
                    env={"LLK_TEST_DISABLE_AVX2": "1"})
    assert reference.returncode == 0, reference.stderr
    assert suite in reference.stdout

    required = run(binary, "--require-selected-target", "--gtest_list_tests")
    assert required.returncode == 0, required.stderr

    unsupported = run(
        binary,
        "--require-selected-target",
        f"--gtest_filter={gtest_filter}",
        env={"LLK_TEST_DISABLE_AVX2": "1"},
    )
    assert unsupported.returncode != 0, (
        "required selected execution accepted a disabled AVX2 feature"
    )
    assert "test override disabled required selected AVX2 execution" in (
        unsupported.stdout + unsupported.stderr
    )


def main():
    arguments = sys.argv[1:]
    assert len(arguments) in (1, 2), "expected one or two collapsed binaries"
    check_pair(
        arguments[0],
        "MappedAcceptance.",
        "MappedAcceptance.SelectedAvx2BackendExecutesNumerically",
    )
    if len(arguments) == 2:
        check_pair(
            arguments[1],
            "DynamicMappedAcceptance.",
            "DynamicMappedAcceptance.CompilesOnceAndExecutesUnrelatedMatmulShapes",
        )


if __name__ == "__main__":
    main()
