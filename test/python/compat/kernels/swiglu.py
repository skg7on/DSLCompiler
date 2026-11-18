"""Genuine upstream Triton fused-SwiGLU fixture for the compatibility corpus.

The two projections share the same left tile ``xt`` and accumulate the gate and
up products separately. The stored value is exactly ``silu(gate) * up`` where
``silu(g) = g / (1 + exp(-g))``; Triton's ``tl.sigmoid`` lowers to that
expression, which the exporter recognizes as the framework's fused activation
semantics. Nothing is approximated: the qualified numerics are strict f32.

As with the matmul fixtures there are no masks, so the declared launch grid
tiles ``[M, N]`` exactly.
"""

import triton
import triton.language as tl


@triton.jit
def swiglu_kernel(
    x_ptr,
    gate_ptr,
    up_ptr,
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
    gate = tl.zeros((BM, BN), tl.float32)
    up = tl.zeros((BM, BN), tl.float32)
    for kk in range(0, K, BK):
        x_tile = tl.load(x_ptr + rows[:, None] * K + (kk + rk)[None, :])
        gate_tile = tl.load(gate_ptr + (kk + rk)[:, None] * N + cols[None, :])
        up_tile = tl.load(up_ptr + (kk + rk)[:, None] * N + cols[None, :])
        gate = tl.dot(x_tile, gate_tile, gate, input_precision="ieee")
        up = tl.dot(x_tile, up_tile, up, input_precision="ieee")
    tl.store(
        out_ptr + rows[:, None] * N + cols[None, :],
        tl.sigmoid(gate) * gate * up,
    )
