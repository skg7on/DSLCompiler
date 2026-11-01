"""Run a static masked-edge add through the shared native frontend.

Run from the worktree root with llk installed and LLK_FRONTEND_LIBRARY set.
"""

from array import array
from pathlib import Path

import llk
import llk.language as tl
from llk.frontend import LaunchSpec, TensorArgument, TensorSpec


@llk.jit(config_root=str(Path.cwd()), backend="reference")
def add(a, b, out, M: tl.constexpr, N: tl.constexpr,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    offsets = rows[:, None] * N + cols[None, :]
    mask = (rows[:, None] < M) & (cols[None, :] < N)
    x = tl.load(a + offsets, mask=mask, other=0.0)
    y = tl.load(b + offsets, mask=mask, other=0.0)
    tl.store(out + offsets, x + y, mask=mask)


shape = (1, 7)
storage = [array("f", range(7)), array("f", [2.0] * 7), array("f", [0.0] * 7)]
views = [memoryview(buf).cast("B").cast("f", shape) for buf in storage]
ports = [TensorSpec(name, shape, "f32", (7, 1), direction, effect)
         for name, direction, effect in (
             ("a", "input", "read"), ("b", "input", "read"),
             ("out", "output", "write"))]
args = tuple(TensorArgument(spec, view) for spec, view in zip(ports, views))
grid = LaunchSpec((1, 1, 1), (8, 16, 1))

with add.compile(*args, 1, 7, 8, 16, grid=grid) as compiled:
    compiled.invoke((views[0], views[1]), (views[2],))
    assert list(storage[2]) == [float(i + 2) for i in range(7)]
    assert "micro.tile_load" in compiled.micro_ir
    print("backend:", compiled.report.backend)

add[grid](*args, 1, 7, 8, 16)
assert list(storage[2]) == [float(i + 2) for i in range(7)]
print("output:", list(storage[2]))
print("cache:", add.cache_info())
