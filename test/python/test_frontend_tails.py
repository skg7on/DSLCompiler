"""PF10: bounded elementwise tails through the shared native frontend.

The launch grid rounds a 33x65 (and 1x7) domain up to 8x16 blocks. The kernel's
declared in-bounds mask is what makes the rounded blocks legal; the importer
proves it, the elementwise adapter realizes interior and static edge rectangles
with bounded movement, and the native executable writes exactly the declared
domain. Guard words on both sides of every payload catch an out-of-bounds write.
"""

from __future__ import annotations

import llk
import llk.language as tl
import native_cases as cases
import pytest
from llk import LLKError, UnsupportedFeatureError
from llk.frontend import LaunchSpec, TensorArgument, TensorSpec

SHAPE_33X65 = (33, 65)
SHAPE_1X7 = (1, 7)
BLOCK = (8, 16)


def bounded_add(
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


def bounded_mul(
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
    tl.store(out + offsets, x * y, mask=mask)


def unmasked_add(
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
    x = tl.load(a + offsets, True, 0.0)
    y = tl.load(b + offsets, True, 0.0)
    tl.store(out + offsets, x + y, True)


def all_false_add(
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
    x = tl.load(a + offsets, False, 0.0)
    y = tl.load(b + offsets, False, 0.0)
    tl.store(out + offsets, x + y, False)


def negative_offset_add(
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
    offsets = rows[:, None] * N + cols[None, :] - 1
    mask = (rows[:, None] < M) & (cols[None, :] < N)
    x = tl.load(a + offsets, mask, 0.0)
    y = tl.load(b + offsets, mask, 0.0)
    tl.store(out + offsets, x + y, mask)


def kernel(fn):
    return llk.jit(**{"config_root": ".", "config_files": cases.config_files()})(fn)


def launch(shape, block=BLOCK):
    return LaunchSpec(
        (
            -(-shape[0] // block[0]),
            -(-shape[1] // block[1]),
            1,
        ),
        (block[0], block[1], 1),
    )


def ports(shape):
    return tuple(
        TensorSpec(name, shape, "f32", (shape[1], 1), direction, effect)
        for name, direction, effect in (
            ("a", "input", "read"),
            ("b", "input", "read"),
            ("out", "output", "write"),
        )
    )


def arguments(shape, block=BLOCK):
    count = shape[0] * shape[1]
    a = cases.Guarded(shape, cases.values(count))
    b = cases.Guarded(shape, cases.other_values(count))
    out = cases.Guarded(shape)
    declared = ports(shape)
    bound = (
        TensorArgument(declared[0], a.view),
        TensorArgument(declared[1], b.view),
        TensorArgument(declared[2], out.view),
    )
    return bound + (shape[0], shape[1], block[0], block[1]), (a, b, out)


def expected(shape, multiply):
    count = shape[0] * shape[1]
    left, right = cases.values(count), cases.other_values(count)
    if multiply:
        return [first * second for first, second in zip(left, right)]
    return [first + second for first, second in zip(left, right)]


@pytest.mark.parametrize(
    ("shape", "fn", "multiply"),
    [
        (SHAPE_33X65, bounded_add, False),
        (SHAPE_33X65, bounded_mul, True),
        (SHAPE_1X7, bounded_add, False),
        (SHAPE_1X7, bounded_mul, True),
    ],
)
def test_rounded_blocks_write_exact_values_and_keep_guards(shape, fn, multiply):
    args, buffers = arguments(shape)
    a, b, out = buffers
    inputs_before = a.bytes(), b.bytes()
    compiled = kernel(fn).compile(*args, grid=launch(shape))
    try:
        compiled.invoke(*cases.providers(buffers))
        assert out.values() == expected(shape, multiply)
        assert (a.bytes(), b.bytes()) == inputs_before
        assert a.guards_intact() and b.guards_intact() and out.guards_intact()
        assert compiled.report.backend == "reference"
        assert compiled.report.selected_group_count == 0
        assert compiled.report.source_adapter == "linalg-elementwise-v1"
        assert "micro.tile_load" in compiled.micro_ir
        assert "valid[" in compiled.micro_ir
    finally:
        compiled.close()


def test_selected_target_realizes_edges_or_reports_unavailable_host():
    args, buffers = arguments(SHAPE_33X65)
    a, b, out = buffers
    jit = llk.jit(
        config_root=".", config_files=cases.config_files(), backend="selected-target"
    )(bounded_add)
    try:
        compiled = jit.compile(*args, grid=launch(SHAPE_33X65))
    except UnsupportedFeatureError as error:
        # A host without the qualified ISA must say so; it must never run the
        # reference backend under a selected-target request.
        assert error.code == "native_backend_unavailable"
        return
    try:
        compiled.invoke(*cases.providers(buffers))
        assert out.values() == expected(SHAPE_33X65, False)
        assert a.guards_intact() and b.guards_intact() and out.guards_intact()
        assert compiled.report.backend == "selected-target"
        assert compiled.report.selected_group_count > 0
        assert compiled.report.reference_group_count == 0
    finally:
        compiled.close()


@pytest.mark.parametrize("fn", [unmasked_add, all_false_add])
def test_unproved_tail_policy_is_rejected_before_execution(fn):
    args, _ = arguments(SHAPE_33X65)
    with pytest.raises(UnsupportedFeatureError) as caught:
        kernel(fn).compile(*args, grid=launch(SHAPE_33X65))
    assert caught.value.code == "unsupported_tail_policy"


def test_negative_offsets_are_rejected_as_invalid_coverage():
    args, _ = arguments(SHAPE_33X65)
    with pytest.raises(LLKError) as caught:
        kernel(negative_offset_add).compile(*args, grid=launch(SHAPE_33X65))
    assert caught.value.code == "invalid_launch_coverage"


def test_under_covering_grid_is_still_invalid():
    args, _ = arguments(SHAPE_33X65)
    with pytest.raises(LLKError) as caught:
        kernel(bounded_add).compile(*args, grid=LaunchSpec((4, 5, 1), (8, 16, 1)))
    assert caught.value.code == "invalid_launch_coverage"
