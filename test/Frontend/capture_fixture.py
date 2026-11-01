"""Build-time interop fixtures from the actual PF04 producer (no Torch needed)."""

from dataclasses import replace
from pathlib import Path
import sys

import llk.language as lang
from llk.frontend import (
    CompileOptions,
    CompileRequest,
    LaunchSpec,
    NumericalContract,
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


def matmul(
    a,
    b,
    out,
    M: lang.constexpr,
    N: lang.constexpr,
    K: lang.constexpr,
    BM: lang.constexpr,
    BN: lang.constexpr,
    BK: lang.constexpr,
):
    """A canonical counted-K contraction with a zero accumulator."""
    pm = lang.program_id(0)
    pn = lang.program_id(1)
    rm = pm * BM + lang.arange(0, BM)
    rn = pn * BN + lang.arange(0, BN)
    rk = lang.arange(0, BK)
    acc = lang.zeros((BM, BN), lang.float32)
    for kk in range(0, K, BK):
        at = lang.load(a + rm[:, None] * K + (kk + rk)[None, :], True, 0.0)
        bt = lang.load(b + (kk + rk)[:, None] * N + rn[None, :], True, 0.0)
        acc = lang.dot(at, bt, acc)
    lang.store(out + rm[:, None] * N + rn[None, :], acc, True)


def matmul_accumulator(
    a,
    b,
    init,
    out,
    M: lang.constexpr,
    N: lang.constexpr,
    K: lang.constexpr,
    BM: lang.constexpr,
    BN: lang.constexpr,
    BK: lang.constexpr,
):
    """The same contraction seeded from a caller-provided accumulator tile."""
    pm = lang.program_id(0)
    pn = lang.program_id(1)
    rm = pm * BM + lang.arange(0, BM)
    rn = pn * BN + lang.arange(0, BN)
    rk = lang.arange(0, BK)
    acc = lang.load(init + rm[:, None] * N + rn[None, :], True, 0.0)
    for kk in range(0, K, BK):
        at = lang.load(a + rm[:, None] * K + (kk + rk)[None, :], True, 0.0)
        bt = lang.load(b + (kk + rk)[:, None] * N + rn[None, :], True, 0.0)
        acc = lang.dot(at, bt, acc)
    lang.store(out + rm[:, None] * N + rn[None, :], acc, True)


def swiglu(
    x,
    wg,
    wu,
    out,
    M: lang.constexpr,
    N: lang.constexpr,
    K: lang.constexpr,
    BM: lang.constexpr,
    BN: lang.constexpr,
    BK: lang.constexpr,
):
    """Two projections over one shared X tile and a gated silu epilogue."""
    pm = lang.program_id(0)
    pn = lang.program_id(1)
    rm = pm * BM + lang.arange(0, BM)
    rn = pn * BN + lang.arange(0, BN)
    rk = lang.arange(0, BK)
    gate = lang.zeros((BM, BN), lang.float32)
    up = lang.zeros((BM, BN), lang.float32)
    for kk in range(0, K, BK):
        xt = lang.load(x + rm[:, None] * K + (kk + rk)[None, :], True, 0.0)
        gt = lang.load(wg + (kk + rk)[:, None] * N + rn[None, :], True, 0.0)
        ut = lang.load(wu + (kk + rk)[:, None] * N + rn[None, :], True, 0.0)
        gate = lang.dot(xt, gt, gate)
        up = lang.dot(xt, ut, up)
    activated = lang.silu(gate)
    lang.store(out + rm[:, None] * N + rn[None, :], activated * up, True)


def swiglu_native(
    x,
    wg,
    wu,
    out,
    M: lang.constexpr,
    N: lang.constexpr,
    K: lang.constexpr,
    BM: lang.constexpr,
    BN: lang.constexpr,
    BK: lang.constexpr,
):
    """The same SwiGLU with an explicitly relaxed activation policy."""
    pm = lang.program_id(0)
    pn = lang.program_id(1)
    rm = pm * BM + lang.arange(0, BM)
    rn = pn * BN + lang.arange(0, BN)
    rk = lang.arange(0, BK)
    gate = lang.zeros((BM, BN), lang.float32)
    up = lang.zeros((BM, BN), lang.float32)
    for kk in range(0, K, BK):
        xt = lang.load(x + rm[:, None] * K + (kk + rk)[None, :], True, 0.0)
        gt = lang.load(wg + (kk + rk)[:, None] * N + rn[None, :], True, 0.0)
        ut = lang.load(wu + (kk + rk)[:, None] * N + rn[None, :], True, 0.0)
        gate = lang.dot(xt, gt, gate)
        up = lang.dot(xt, ut, up)
    activated = lang.silu(gate, approximation="native")
    lang.store(out + rm[:, None] * N + rn[None, :], activated * up, True)


def swiglu_distinct_inputs(
    x0,
    x1,
    wg,
    wu,
    out,
    M: lang.constexpr,
    N: lang.constexpr,
    K: lang.constexpr,
    BM: lang.constexpr,
    BN: lang.constexpr,
    BK: lang.constexpr,
):
    """The projections read different left operands, so they are not SwiGLU."""
    pm = lang.program_id(0)
    pn = lang.program_id(1)
    rm = pm * BM + lang.arange(0, BM)
    rn = pn * BN + lang.arange(0, BN)
    rk = lang.arange(0, BK)
    gate = lang.zeros((BM, BN), lang.float32)
    up = lang.zeros((BM, BN), lang.float32)
    for kk in range(0, K, BK):
        at = lang.load(x0 + rm[:, None] * K + (kk + rk)[None, :], True, 0.0)
        bt = lang.load(x1 + rm[:, None] * K + (kk + rk)[None, :], True, 0.0)
        gt = lang.load(wg + (kk + rk)[:, None] * N + rn[None, :], True, 0.0)
        ut = lang.load(wu + (kk + rk)[:, None] * N + rn[None, :], True, 0.0)
        gate = lang.dot(at, gt, gate)
        up = lang.dot(bt, ut, up)
    activated = lang.silu(gate)
    lang.store(out + rm[:, None] * N + rn[None, :], activated * up, True)


def swiglu_rounded_activation(
    x,
    wg,
    wu,
    out,
    M: lang.constexpr,
    N: lang.constexpr,
    K: lang.constexpr,
    BM: lang.constexpr,
    BN: lang.constexpr,
    BK: lang.constexpr,
):
    """A bf16 Torch intermediate-rounding pattern: the activation is rounded."""
    pm = lang.program_id(0)
    pn = lang.program_id(1)
    rm = pm * BM + lang.arange(0, BM)
    rn = pn * BN + lang.arange(0, BN)
    rk = lang.arange(0, BK)
    gate = lang.zeros((BM, BN), lang.float32)
    up = lang.zeros((BM, BN), lang.float32)
    for kk in range(0, K, BK):
        xt = lang.load(x + rm[:, None] * K + (kk + rk)[None, :], True, 0.0)
        gt = lang.load(wg + (kk + rk)[:, None] * N + rn[None, :], True, 0.0)
        ut = lang.load(wu + (kk + rk)[:, None] * N + rn[None, :], True, 0.0)
        gate = lang.dot(xt, gt, gate)
        up = lang.dot(xt, ut, up)
    activated = lang.silu(gate).to(lang.bfloat16).to(lang.float32)
    lang.store(out + rm[:, None] * N + rn[None, :], activated * up, True)


def matmul_extra_store(
    a,
    b,
    out,
    M: lang.constexpr,
    N: lang.constexpr,
    K: lang.constexpr,
    BM: lang.constexpr,
    BN: lang.constexpr,
    BK: lang.constexpr,
):
    """A second writer of the same output is outside the accepted family."""
    pm = lang.program_id(0)
    pn = lang.program_id(1)
    rm = pm * BM + lang.arange(0, BM)
    rn = pn * BN + lang.arange(0, BN)
    rk = lang.arange(0, BK)
    acc = lang.zeros((BM, BN), lang.float32)
    for kk in range(0, K, BK):
        at = lang.load(a + rm[:, None] * K + (kk + rk)[None, :], True, 0.0)
        bt = lang.load(b + (kk + rk)[:, None] * N + rn[None, :], True, 0.0)
        acc = lang.dot(at, bt, acc)
    lang.store(out + rm[:, None] * N + rn[None, :], acc, True)
    lang.store(out + rm[:, None] * N + rn[None, :], acc + acc, True)


def matmul_wrong_extent(
    a,
    b,
    out,
    M: lang.constexpr,
    N: lang.constexpr,
    K: lang.constexpr,
    BM: lang.constexpr,
    BN: lang.constexpr,
    BK: lang.constexpr,
):
    """The counted loop runs one block past the declared reduction extent."""
    pm = lang.program_id(0)
    pn = lang.program_id(1)
    rm = pm * BM + lang.arange(0, BM)
    rn = pn * BN + lang.arange(0, BN)
    rk = lang.arange(0, BK)
    acc = lang.zeros((BM, BN), lang.float32)
    for kk in range(0, K + BK, BK):
        at = lang.load(a + rm[:, None] * K + (kk + rk)[None, :], True, 0.0)
        bt = lang.load(b + (kk + rk)[:, None] * N + rn[None, :], True, 0.0)
        acc = lang.dot(at, bt, acc)
    lang.store(out + rm[:, None] * N + rn[None, :], acc, True)


def emit_tensors(path, fn, tensors, meta, launch, options=None):
    options = options or CompileOptions(config_files={})
    bound = bind_kernel(fn, tensors + meta, {}, launch, options)
    program = compile_kernel(fn, bound)
    path.write_bytes(
        serialize_request(
            CompileRequest(
                program, replace(options, numerical_contract=program.numerical_contract)
            )
        )
    )


def emit_matmul(path, fn, shapes, meta, launch, options=None):
    tensors = tuple(
        TensorSpec(name, shape, "f32", (shape[1], 1), direction, effect)
        for name, shape, direction, effect in shapes
    )
    emit_tensors(path, fn, tensors, meta, launch, options)


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
    # PF11: canonical counted contractions. The launch tiles the output
    # domain; the kernel reads its own K extent from the loop bounds.
    matmul_tensors = [
        ("a", (32, 16), "input", "read"),
        ("b", (16, 64), "input", "read"),
        ("out", (32, 64), "output", "write"),
    ]
    matmul_meta = (32, 64, 16, 8, 16, 8)
    matmul_launch = LaunchSpec((4, 4, 1), (8, 16, 1))
    emit_matmul(
        destination / "matmul_32x64x16.json",
        matmul,
        matmul_tensors,
        matmul_meta,
        matmul_launch,
    )
    emit_matmul(
        destination / "matmul_7x9x5.json",
        matmul,
        [
            ("a", (7, 5), "input", "read"),
            ("b", (5, 9), "input", "read"),
            ("out", (7, 9), "output", "write"),
        ],
        (7, 9, 5, 7, 9, 5),
        LaunchSpec((1, 1, 1), (7, 9, 1)),
    )
    emit_matmul(
        destination / "matmul_acc_32x64x16.json",
        matmul_accumulator,
        [
            ("a", (32, 16), "input", "read"),
            ("b", (16, 64), "input", "read"),
            ("init", (32, 64), "input", "read"),
            ("out", (32, 64), "output", "write"),
        ],
        matmul_meta,
        matmul_launch,
    )
    # Two f32 accumulators have to fit the machine's accumulator capacity, so
    # the fused source uses one spatial tile per axis (16x64x16).
    swiglu_tensors = [
        ("x", (16, 16), "input", "read"),
        ("wg", (16, 64), "input", "read"),
        ("wu", (16, 64), "input", "read"),
        ("out", (16, 64), "output", "write"),
    ]
    swiglu_meta = (16, 64, 16, 16, 64, 16)
    swiglu_launch = LaunchSpec((1, 1, 1), (16, 64, 1))
    emit_matmul(
        destination / "swiglu_16x64x16.json",
        swiglu,
        swiglu_tensors,
        swiglu_meta,
        swiglu_launch,
    )
    emit_matmul(
        destination / "swiglu_native_16x64x16.json",
        swiglu_native,
        swiglu_tensors,
        swiglu_meta,
        swiglu_launch,
        options=CompileOptions(
            config_files={},
            numerical_contract=NumericalContract(activation_approximation="native"),
        ),
    )
    emit_matmul(
        destination / "swiglu_distinct_x_16x64x16.json",
        swiglu_distinct_inputs,
        [
            ("x0", (16, 16), "input", "read"),
            ("x1", (16, 16), "input", "read"),
            ("wg", (16, 64), "input", "read"),
            ("wu", (16, 64), "input", "read"),
            ("out", (16, 64), "output", "write"),
        ],
        swiglu_meta,
        swiglu_launch,
    )
    emit_matmul(
        destination / "swiglu_rounded_16x64x16.json",
        swiglu_rounded_activation,
        swiglu_tensors,
        swiglu_meta,
        swiglu_launch,
    )
    emit_matmul(
        destination / "matmul_extra_store_32x64x16.json",
        matmul_extra_store,
        matmul_tensors,
        matmul_meta,
        matmul_launch,
    )
    emit_matmul(
        destination / "matmul_wrong_extent_32x64x16.json",
        matmul_wrong_extent,
        matmul_tensors,
        matmul_meta,
        matmul_launch,
    )
