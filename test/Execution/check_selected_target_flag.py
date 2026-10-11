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


def main():
    selected, reference = sys.argv[1:]

    missing = run(selected, "--gtest_list_tests")
    assert missing.returncode != 0, "selected executable accepted a missing gate flag"
    assert "requires --require-selected-target" in missing.stderr

    enabled = run(selected, "--require-selected-target", "--gtest_list_tests")
    assert enabled.returncode == 0, enabled.stderr
    assert "MappedAcceptance." in enabled.stdout

    unsupported = run(
        selected,
        "--require-selected-target",
        "--gtest_filter=MappedAcceptance.SelectedAvx2BackendExecutesNumerically",
        env={"LLK_TEST_DISABLE_AVX2": "1"},
    )
    assert unsupported.returncode != 0, "required selected execution accepted a disabled AVX2 feature"
    assert "test override disabled required selected AVX2 execution" in (
        unsupported.stdout + unsupported.stderr
    )

    wrong_binary = run(reference, "--require-selected-target", "--gtest_list_tests")
    assert wrong_binary.returncode != 0, "reference executable accepted selected-target flag"
    assert "only valid for the selected-target executable" in wrong_binary.stderr


if __name__ == "__main__":
    main()
