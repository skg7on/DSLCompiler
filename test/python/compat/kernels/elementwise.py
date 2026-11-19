"""Genuine upstream Triton elementwise fixtures for the compatibility corpus.

These are ordinary ``@triton.jit`` kernels: the corpus exports them with the
qualified upstream compiler and binds their neutral templates to concrete
rank-two CPU tensors. Each fixture in ``docs/design/triton-compatibility-v1.json``
names one function here, its concrete shapes and its launch grid.

The kernels accept one flat program id, exactly two read-only inputs and one
write-only output, which is the elementwise source-adaptor contract. The masked
form is the same kernel as the dense form: the corpus chooses an extent that
leaves a partial last block only for the declared tail fixtures.
"""

import triton
import triton.language as tl


@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    y = tl.load(y_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, x + y, mask=mask)


@triton.jit
def multiply_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    y = tl.load(y_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, x * y, mask=mask)


@triton.jit
def bounded_add_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    M,
    N,
    BM: tl.constexpr,
    BN: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    rows = pid_m * BM + tl.arange(0, BM)
    cols = pid_n * BN + tl.arange(0, BN)
    mask = (rows[:, None] < M) & (cols[None, :] < N)
    offsets = rows[:, None] * N + cols[None, :]
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    tl.store(out_ptr + offsets, x + y, mask=mask)


@triton.jit
def bounded_multiply_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    M,
    N,
    BM: tl.constexpr,
    BN: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    rows = pid_m * BM + tl.arange(0, BM)
    cols = pid_n * BN + tl.arange(0, BN)
    mask = (rows[:, None] < M) & (cols[None, :] < N)
    offsets = rows[:, None] * N + cols[None, :]
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    tl.store(out_ptr + offsets, x * y, mask=mask)
