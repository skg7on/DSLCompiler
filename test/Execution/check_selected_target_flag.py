#!/usr/bin/env python3
"""Exercise the required selected-target command-line contract."""

import subprocess
import sys


def run(binary, *arguments):
    return subprocess.run(
        [binary, *arguments], check=False, capture_output=True, text=True
    )


def main():
    selected, reference = sys.argv[1:]

    missing = run(selected, "--gtest_list_tests")
    assert missing.returncode != 0, "selected executable accepted a missing gate flag"
    assert "requires --require-selected-target" in missing.stderr

    enabled = run(selected, "--require-selected-target", "--gtest_list_tests")
    assert enabled.returncode == 0, enabled.stderr
    assert "MappedAcceptance." in enabled.stdout

    wrong_binary = run(reference, "--require-selected-target", "--gtest_list_tests")
    assert wrong_binary.returncode != 0, "reference executable accepted selected-target flag"
    assert "only valid for the selected-target executable" in wrong_binary.stderr


if __name__ == "__main__":
    main()
