#!/usr/bin/env python3
"""Write a self-contained frontend request for the C API tests.

The PF03 fixture describes the program and options; the frozen target policy
travels inside the request as ``config_files`` so compilation never reads a live
checkout path. Keep this a plain filesystem transformation: it must not import
``llk`` or depend on a configured Python package.
"""

import json
import sys
from pathlib import Path

CONFIG_FILES = (
    "machines/x86-avx2-v2.yaml",
    "mapping/x86-avx2/layouts.llkmap",
    "mapping/x86-avx2/rules.llkmap",
)

# The serialized fixture describes a 32x64 tiled launch. That tiling needs more
# SRAM than the AVX2 machine profile declares, so the C consumer runs the
# full-domain 8x16 shape PF06 qualified. The program's operations already use
# BLOCK_M=8/BLOCK_N=16, so only the public shapes and the launch change.
ROWS, COLUMNS = 8, 16


def main() -> int:
    if len(sys.argv) != 3:
        print(f"usage: {sys.argv[0]} <source-root> <output-path>", file=sys.stderr)
        return 2
    root = Path(sys.argv[1])
    output = Path(sys.argv[2])
    request = json.loads((root / "test/python/frontend_add_v1.json").read_text())
    assert request["program"]["constexprs"] == {
        "BLOCK_M": ROWS,
        "BLOCK_N": COLUMNS,
    }, "fixture block constants no longer match the full-domain shape"
    for tensor in request["program"]["tensors"]:
        tensor["shape"] = [ROWS, COLUMNS]
        tensor["strides"] = [COLUMNS, 1]
    request["program"]["launch"] = {
        "grid": [1, 1, 1],
        "block": [ROWS, COLUMNS, 1],
    }
    request["options"]["config_root"] = "."
    request["options"]["config_files"] = {
        name: (root / name).read_text() for name in CONFIG_FILES
    }
    output.write_text(
        json.dumps(request, sort_keys=True, separators=(",", ":")) + "\n"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
