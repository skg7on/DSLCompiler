#!/usr/bin/env python3
"""Enforce the LLK source-file header convention.

Every tracked C++ or TableGen source file must begin with a single-line banner
followed by the project license notice:

    //===- Foo.h - Short description -------------*- C++ -*-===//
    //
    // Part of the LLK Compiler Project, under the Apache License v2.0 with
    // LLVM Exceptions.
    // See https://llvm.org/LICENSE.txt for license information.
    // SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
    //
    //===----------------------------------------------------------------------===//

The banner must stay on one line and fit the 80-column limit. `.clang-format`
sets `ReflowComments: false` so clang-format does not reflow it; this script is
what catches a banner that was authored wrapped or over-long.

Usage:
    scripts/check-license-headers.py            # check every tracked source
    scripts/check-license-headers.py a.h b.cpp  # check specific files

Exits non-zero and lists the offending files when a header is missing or
malformed. Vendored code under third_party/ is exempt.
"""
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
EXTENSIONS = (".h", ".cpp", ".td")
EXEMPT_PREFIXES = ("third_party/", ".claude/")
COLUMN_LIMIT = 80
SPDX = "SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception"
PROVENANCE = "Part of the LLK Compiler Project"
BANNER_OPEN = "//===- "
BANNER_CLOSE = "===//"


def tracked_sources():
    """Every tracked C++/TableGen source, excluding vendored trees."""
    out = subprocess.run(
        ["git", "-C", str(REPO), "ls-files", *[f"*{e}" for e in EXTENSIONS]],
        capture_output=True, text=True, check=True,
    ).stdout
    files = []
    for line in out.split("\n"):
        line = line.strip()
        if not line or line.startswith(EXEMPT_PREFIXES):
            continue
        if Path(line).suffix in EXTENSIONS:
            files.append(line)
    return files


def check(rel_path):
    """Return an error string, or None when the header conforms."""
    path = REPO / rel_path
    try:
        lines = path.read_text(encoding="utf-8").split("\n")
    except (OSError, UnicodeDecodeError) as exc:
        return f"unreadable: {exc}"

    if not lines or not lines[0].strip():
        return "file is empty"

    banner = lines[0].rstrip("\n")
    if not banner.startswith(BANNER_OPEN):
        return f"missing banner: line 1 must start with {BANNER_OPEN!r}"
    if not banner.endswith(BANNER_CLOSE):
        return (f"banner must be a single line ending in {BANNER_CLOSE!r} "
                f"(found {banner!r})")
    if " - " not in banner[len(BANNER_OPEN):-len(BANNER_CLOSE)]:
        return "banner must separate file name and description with ' - '"
    if len(banner) > COLUMN_LIMIT:
        return f"banner is {len(banner)} columns (limit {COLUMN_LIMIT})"

    head = "\n".join(lines[:12])
    if SPDX not in head:
        return f"missing {SPDX!r} within the first 12 lines"
    if PROVENANCE not in head:
        return f"missing {PROVENANCE!r} within the first 12 lines"
    return None


def main(argv):
    if argv:
        files = [a for a in argv if Path(a).suffix in EXTENSIONS
                 and not a.startswith(EXEMPT_PREFIXES)]
    else:
        files = tracked_sources()

    failures = [(f, check(f)) for f in files]
    failures = [(f, why) for f, why in failures if why]

    for rel, why in sorted(failures):
        print(f"{rel}: {why}")

    checked = len(files)
    if failures:
        print(f"\n{len(failures)} of {checked} file(s) have a non-conforming "
              f"header. See scripts/check-license-headers.py for the format.")
        return 1
    print(f"license headers OK ({checked} file(s) checked)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
