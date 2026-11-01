"""Fixtures for PF08 native launch and buffer-contract tests.

The kernel is a real typed-add capture. Buffers are guarded ctypes storage
exposed through the Python buffer protocol, so the tests do not depend on NumPy
or Torch and can observe writes outside the declared payload.
"""

from __future__ import annotations

import ctypes
from pathlib import Path

import llk
import llk.language as tl
from llk.frontend import CompileOptions, LaunchSpec, TensorArgument, TensorSpec

TILED_SHAPE = (16, 16)
TILED_BLOCK = (8, 16)
FULL_SHAPE = (32, 64)
GUARD = 2
GUARD_VALUE = -777.0
CONFIG_FILES = (
    "machines/x86-avx2-v2.yaml",
    "mapping/x86-avx2/layouts.llkmap",
    "mapping/x86-avx2/rules.llkmap",
)


def config_files() -> dict[str, str]:
    root = Path(__file__).resolve().parents[2]
    return {name: (root / name).read_text(encoding="utf-8") for name in CONFIG_FILES}


def options(**changes):
    data = {"config_root": ".", "config_files": config_files()}
    data.update(changes)
    return CompileOptions(**data)


def add(
    a,
    b,
    out,
    M: tl.constexpr,
    N: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    offsets = rows[:, None] * N + cols[None, :]
    mask = (rows[:, None] < M) & (cols[None, :] < N)
    x = tl.load(a + offsets, mask=mask, other=0.0)
    y = tl.load(b + offsets, mask, 0.0)
    tl.store(out + offsets, x + y, mask=mask)


add_kernel = llk.jit(**{"config_root": ".", "config_files": config_files()})(add)


def specs(shape=TILED_SHAPE):
    width = shape[1]
    return tuple(
        TensorSpec(name, shape, "f32", (width, 1), direction, effect)
        for name, direction, effect in (
            ("a", "input", "read"),
            ("b", "input", "read"),
            ("out", "output", "write"),
        )
    )


def launch(shape=TILED_SHAPE, block=TILED_BLOCK):
    return LaunchSpec(
        (shape[0] // block[0], shape[1] // block[1], 1),
        (block[0], block[1], 1),
    )


def constexprs(shape=TILED_SHAPE, block=TILED_BLOCK):
    return shape[0], shape[1], block[0], block[1]


def values(count):
    return [float(index % 17) - 8.0 for index in range(count)]


def other_values(count):
    return [float((index * 3) % 11) + 0.5 for index in range(count)]


def expected(count):
    return [left + right for left, right in zip(values(count), other_values(count))]


class Guarded:
    """ctypes storage with two floats of canary on each side of the payload."""

    def __init__(self, shape, payload=None):
        rows, columns = shape
        count = rows * columns
        self.rows, self.columns = rows, columns
        self.storage = (ctypes.c_float * (count + 2 * GUARD))()
        for index in range(len(self.storage)):
            self.storage[index] = GUARD_VALUE
        byte_view = memoryview(self.storage).cast("B")
        self.view = byte_view[GUARD * 4 : (GUARD + count) * 4].cast("f", shape)
        for row in range(rows):
            for column in range(columns):
                index = row * columns + column
                self.view[row, column] = 0.0 if payload is None else payload[index]

    def values(self):
        return [
            self.view[row, column]
            for row in range(self.rows)
            for column in range(self.columns)
        ]

    def bytes(self):
        return memoryview(self.view).tobytes()

    def guards(self):
        tail = len(self.storage) - GUARD
        return [self.storage[index] for index in range(GUARD)] + [
            self.storage[tail + index] for index in range(GUARD)
        ]

    def guards_intact(self):
        return self.guards() == [GUARD_VALUE] * (2 * GUARD)


def arguments(shape=TILED_SHAPE, block=TILED_BLOCK):
    """Return binding arguments plus the three guarded buffers."""
    a = Guarded(shape, values(shape[0] * shape[1]))
    b = Guarded(shape, other_values(shape[0] * shape[1]))
    out = Guarded(shape)
    ports = specs(shape)
    bound = (
        TensorArgument(ports[0], a.view),
        TensorArgument(ports[1], b.view),
        TensorArgument(ports[2], out.view),
    )
    return bound + constexprs(shape, block), (a, b, out)


def providers(buffers):
    a, b, out = buffers
    return (a.view, b.view), (out.view,)
