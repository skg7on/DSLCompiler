"""PF18 launch lane, isolated from the rest of the Python suite.

Upstream Triton statically links its own LLVM; the LLK frontend C API loads
LLVM as a shared library.  Loading both images into one process is fatal on
Mach-O hosts, so the actual launch tests must not run in the same interpreter
as the other frontend tests.  This wrapper runs the real lane in a dedicated
pytest process; the lane itself then qualifies coexistence before installing
the driver.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

INNER = Path(__file__).with_name("_triton_launch_inner.py")


def test_triton_launch_lane_runs_in_a_dedicated_process(triton_path):
    completed = subprocess.run(
        [sys.executable, "-m", "pytest", str(INNER), "-q", "-rs"],
        capture_output=True,
        text=True,
        timeout=1800,
    )
    output = completed.stdout + completed.stderr
    assert completed.returncode == 0, output[-8000:]
    assert " passed" in output, output[-8000:]
