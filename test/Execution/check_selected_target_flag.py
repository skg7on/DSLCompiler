#!/usr/bin/env python3
"""Exercise the required selected-target command-line contract."""

import subprocess
import sys
import os


def run(binary, *arguments, env=None):
    environment = os.environ.copy()
    environment.update(env or {})
    return subprocess.run(
        [binary, *arguments], check=False, capture_output=True, text=True,
        env=environment
    )


def check_pair(selected, reference, suite, gtest_filter):
    missing = run(selected, "--gtest_list_tests")
    assert missing.returncode != 0, "selected executable accepted a missing gate flag"
    assert "requires --require-selected-target" in missing.stderr

    enabled = run(selected, "--require-selected-target", "--gtest_list_tests")
    assert enabled.returncode == 0, enabled.stderr
    assert suite in enabled.stdout

    unsupported = run(
        selected,
        "--require-selected-target",
        f"--gtest_filter={gtest_filter}",
        env={"LLK_TEST_DISABLE_AVX2": "1"},
    )
    assert unsupported.returncode != 0, "required selected execution accepted a disabled AVX2 feature"
    assert "test override disabled required selected AVX2 execution" in (
        unsupported.stdout + unsupported.stderr
    )

    wrong_binary = run(reference, "--require-selected-target", "--gtest_list_tests")
    assert wrong_binary.returncode != 0, "reference executable accepted selected-target flag"
    assert "only valid for the selected-target executable" in wrong_binary.stderr


def main():
    arguments = sys.argv[1:]
    assert len(arguments) in (2, 4), "expected one or two selected/reference pairs"
    check_pair(
        arguments[0],
        arguments[1],
        "MappedAcceptance.",
        "MappedAcceptance.SelectedAvx2BackendExecutesNumerically",
    )
    if len(arguments) == 4:
        check_pair(
            arguments[2],
            arguments[3],
            "DynamicMappedAcceptance.",
            "DynamicMappedAcceptance.CompilesOnceAndExecutesUnrelatedMatmulShapes",
        )


if __name__ == "__main__":
    main()
