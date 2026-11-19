"""Genuine upstream Triton contraction fixtures for the compatibility corpus.

Both matrices are read with the documented row-major addresses
``rows * K + rk`` and ``(kk + rk) * N + cols``; the output is written with
``rows * N + cols``. There are deliberately no masks: the LLK counted
contraction family requires a launch grid that tiles the declared output
exactly, and the corpus manifest declares shapes divisible by the block sizes.

``matmul_kernel`` seeds its accumulator with ``tl.zeros`` (a zero tile in TTIR).
``matmul_accumulate_kernel`` instead reads the caller's ``[M, N]`` initializer,
which the framework accepts as the accumulator contract; the corpus never
substitutes a zero seed for it.
"""

import triton
import triton.language as tl


@triton.jit
def matmul_kernel(
    a_ptr,
    b_ptr,
    out_ptr,
    M,
    N,
    K,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    rows = pid_m * BM + tl.arange(0, BM)
    cols = pid_n * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)
    acc = tl.zeros((BM, BN), tl.float32)
    for kk in range(0, K, BK):
        a_tile = tl.load(a_ptr + rows[:, None] * K + (kk + rk)[None, :])
        b_tile = tl.load(b_ptr + (kk + rk)[:, None] * N + cols[None, :])
        acc = tl.dot(a_tile, b_tile, acc, input_precision="ieee")
    tl.store(out_ptr + rows[:, None] * N + cols[None, :], acc)


@triton.jit
def matmul_accumulate_kernel(
    a_ptr,
    b_ptr,
    init_ptr,
    out_ptr,
    M,
    N,
    K,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    rows = pid_m * BM + tl.arange(0, BM)
    cols = pid_n * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)
    acc = tl.load(init_ptr + rows[:, None] * N + cols[None, :])
    for kk in range(0, K, BK):
        a_tile = tl.load(a_ptr + rows[:, None] * K + (kk + rk)[None, :])
        b_tile = tl.load(b_ptr + (kk + rk)[:, None] * N + cols[None, :])
        acc = tl.dot(a_tile, b_tile, acc, input_precision="ieee")
    tl.store(out_ptr + rows[:, None] * N + cols[None, :], acc)
