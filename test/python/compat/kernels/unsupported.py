"""Genuine upstream Triton kernels at the documented rejection boundaries.

Each function is a valid ``@triton.jit`` kernel that the qualified upstream
compiler accepts. The corpus exports or binds it and requires the named,
stable LLK diagnostic instead of a silent approximation, a fallback or a
zero substitution.
"""

import triton
import triton.language as tl


@triton.jit
def atomic_kernel(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    tl.atomic_add(out_ptr + offs, x, mask=mask)


@triton.jit
def pipelined_kernel(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    acc = tl.zeros((BLOCK,), tl.float32)
    for i in tl.range(0, 4, num_stages=2):
        acc += tl.load(x_ptr + offs + i * BLOCK)
    tl.store(out_ptr + offs, acc)


@triton.jit
def reduce_kernel(x_ptr, out_ptr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    tl.store(out_ptr + pid, tl.sum(tl.load(x_ptr + offs)))


@triton.jit
def while_kernel(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    i = 0
    acc = tl.zeros((BLOCK,), tl.float32)
    while i < n:
        acc += tl.load(x_ptr + offs + i)
        i += 1
    tl.store(out_ptr + offs, acc)


@triton.jit
def strided_kernel(x_ptr, out_ptr, n, step, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs * step, mask=mask)
    tl.store(out_ptr + offs, x, mask=mask)


@triton.jit
def inline_assembly_kernel(x_ptr, out_ptr, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(x_ptr + offs)
    y = tl.inline_asm_elementwise(
        "mov.b32 $0, $1;", "=r,r", [x], dtype=tl.float32, is_pure=True, pack=1
    )
    tl.store(out_ptr + offs, y)
