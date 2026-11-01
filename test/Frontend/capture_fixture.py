"""Build-time interop fixtures from the actual PF04 producer (no Torch needed)."""

from dataclasses import replace
from pathlib import Path
import sys

import llk.language as lang
from llk.frontend import (
    CompileOptions,
    CompileRequest,
    LaunchSpec,
    TensorSpec,
    bind_kernel,
    serialize_request,
)
from llk.language._ast_compiler import compile_kernel


def pointwise(a, b, out, N: lang.constexpr, ADD: lang.constexpr):
    rows = lang.program_id(0) * 8 + lang.arange(0, 8)
    cols = lang.program_id(1) * 16 + lang.arange(0, 16)
    offsets = rows[:, None] * N + cols[None, :]
    mask = (rows[:, None] < 32) & (cols[None, :] < N)
    x = lang.load(a + offsets, mask=mask, other=0.0)
    y = lang.load(b + offsets, mask, 0.0)
    if ADD:
        result = x + y
    else:
        result = x * y
    lang.store(out + offsets, result, mask)


def bounded_pointwise(
    a,
    b,
    out,
    M: lang.constexpr,
    N: lang.constexpr,
    BLOCK_M: lang.constexpr,
    BLOCK_N: lang.constexpr,
    ADD: lang.constexpr,
    DECLARED: lang.constexpr = True,
    ROW_BOUND: lang.constexpr = 0,
):
    """A tail-shaped add/multiply whose launch grid rounds the extent up.

    ``DECLARED`` selects whether the loads and stores carry the in-bounds mask.
    The remaining parameters let a test produce the same program with an
    all-false mask (``DECLARED = False``) or a mask that stops one row short
    (``ROW_BOUND = -1``), so the importer rejects them for the stated reason.
    """
    rows = lang.program_id(0) * BLOCK_M + lang.arange(0, BLOCK_M)
    cols = lang.program_id(1) * BLOCK_N + lang.arange(0, BLOCK_N)
    offsets = rows[:, None] * N + cols[None, :]
    bound = M + ROW_BOUND
    mask = (rows[:, None] < bound) & (cols[None, :] < N)
    if not DECLARED:
        mask = False
    x = lang.load(a + offsets, mask=mask, other=0.0)
    y = lang.load(b + offsets, mask, 0.0)
    if ADD:
        result = x + y
    else:
        result = x * y
    lang.store(out + offsets, result, mask)


def unmasked_pointwise(
    a,
    b,
    out,
    M: lang.constexpr,
    N: lang.constexpr,
    BLOCK_M: lang.constexpr,
    BLOCK_N: lang.constexpr,
):
    """The same non-divisible launch with no declared tail mask."""
    rows = lang.program_id(0) * BLOCK_M + lang.arange(0, BLOCK_M)
    cols = lang.program_id(1) * BLOCK_N + lang.arange(0, BLOCK_N)
    offsets = rows[:, None] * N + cols[None, :]
    x = lang.load(a + offsets, True, 0.0)
    y = lang.load(b + offsets, True, 0.0)
    lang.store(out + offsets, x + y, True)


def negative_offset_pointwise(
    a,
    b,
    out,
    M: lang.constexpr,
    N: lang.constexpr,
    BLOCK_M: lang.constexpr,
    BLOCK_N: lang.constexpr,
):
    """A masked tail whose element addresses start one element too early."""
    rows = lang.program_id(0) * BLOCK_M + lang.arange(0, BLOCK_M)
    cols = lang.program_id(1) * BLOCK_N + lang.arange(0, BLOCK_N)
    offsets = rows[:, None] * N + cols[None, :] - 1
    mask = (rows[:, None] < M) & (cols[None, :] < N)
    x = lang.load(a + offsets, mask, 0.0)
    y = lang.load(b + offsets, mask, 0.0)
    lang.store(out + offsets, x + y, mask)


def counted(a, b, out, K: lang.constexpr):
    r = lang.arange(0, 8)
    offsets = r[:, None] * 8 + r[None, :]
    acc = lang.zeros((8, 8), lang.float32)
    for outer in range(0, K, 2):
        for inner in range(0, K, 2):
            x = lang.load(a + offsets, True, 0.0)
            y = lang.load(b + offsets, True, 0.0)
            acc = lang.dot(x, y, acc)
            acc = acc.to(lang.bfloat16).to(lang.float32)
    lang.store(out + offsets, acc, True)


def emit(path, fn, shape, meta, launch):
    tensors = tuple(
        TensorSpec(name, shape, "f32", (shape[1], 1), direction, effect)
        for name, direction, effect in [
            ("a", "input", "read"),
            ("b", "input", "read"),
            ("out", "output", "write"),
        ]
    )
    options = CompileOptions(config_files={})
    bound = bind_kernel(fn, tensors + meta, {}, launch, options)
    program = compile_kernel(fn, bound)
    path.write_bytes(
        serialize_request(
            CompileRequest(
                program, replace(options, numerical_contract=program.numerical_contract)
            )
        )
    )


if __name__ == "__main__":
    destination = Path(sys.argv[1])
    destination.mkdir(parents=True, exist_ok=True)
    for name, add in [("add", True), ("mul", False)]:
        emit(
            destination / (name + ".json"),
            pointwise,
            (32, 64),
            (64, add),
            LaunchSpec((4, 4, 1), (8, 16, 1)),
        )
    emit(
        destination / "loop.json",
        counted,
        (8, 8),
        (8,),
        LaunchSpec((1, 1, 1), (8, 8, 1)),
    )
    # PF10 bounded tails: a grid that rounds the domain up to whole blocks.
    for name, add in [("add", True), ("mul", False)]:
        emit(
            destination / ("tail_" + name + "_33x65.json"),
            bounded_pointwise,
            (33, 65),
            (33, 65, 8, 16, add),
            LaunchSpec((5, 5, 1), (8, 16, 1)),
        )
        emit(
            destination / ("tail_" + name + "_1x7.json"),
            bounded_pointwise,
            (1, 7),
            (1, 7, 8, 16, add),
            LaunchSpec((1, 1, 1), (8, 16, 1)),
        )
    emit(
        destination / "tail_unmasked_33x65.json",
        unmasked_pointwise,
        (33, 65),
        (33, 65, 8, 16),
        LaunchSpec((5, 5, 1), (8, 16, 1)),
    )
    emit(
        destination / "tail_false_mask_33x65.json",
        bounded_pointwise,
        (33, 65),
        (33, 65, 8, 16, True, False),
        LaunchSpec((5, 5, 1), (8, 16, 1)),
    )
    emit(
        destination / "tail_short_mask_33x65.json",
        bounded_pointwise,
        (33, 65),
        (33, 65, 8, 16, True, True, -1),
        LaunchSpec((5, 5, 1), (8, 16, 1)),
    )
    emit(
        destination / "tail_wide_mask_33x65.json",
        bounded_pointwise,
        (33, 65),
        (33, 65, 8, 16, True, True, 1),
        LaunchSpec((5, 5, 1), (8, 16, 1)),
    )
    emit(
        destination / "tail_negative_offset_33x65.json",
        negative_offset_pointwise,
        (33, 65),
        (33, 65, 8, 16),
        LaunchSpec((5, 5, 1), (8, 16, 1)),
    )
