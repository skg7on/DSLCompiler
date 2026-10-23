"""Source fixtures shared by PF04 semantic and diagnostic tests."""

from __future__ import annotations

import llk.language as lang
from llk.frontend import CompileOptions, LaunchSpec, TensorSpec, bind_kernel


def ports(dtype="f32", shape=(8, 8)):
    return tuple(
        TensorSpec(name, shape, dtype, (shape[1], 1), direction, effect)
        for name, direction, effect in [
            ("a", "input", "read"),
            ("b", "input", "read"),
            ("out", "output", "write"),
        ]
    )


def binding(fn, args=None, *, options=None, kwargs=None):
    return bind_kernel(
        fn,
        ports() + (8,) if args is None else args,
        {} if kwargs is None else kwargs,
        LaunchSpec((2, 2, 1), (4, 4, 1)),
        options or CompileOptions(config_files={}),
    )


def masked_add(a, b, out, N: lang.constexpr):
    rows = lang.program_id(0) * 4 + lang.arange(0, 4)
    cols = lang.program_id(1) * 4 + lang.arange(0, 4)
    offsets = rows[:, None] * N + cols[None, :]
    mask = (rows[:, None] < 8) & (cols[None, :] < N)
    x = lang.load(a + offsets, mask=mask, other=2.0)
    y = lang.load(b + offsets, mask, 3.0)
    result = x + y
    lang.store(out + offsets, result, mask=mask)


def counted_dot(a, b, out, K: lang.constexpr):
    r = lang.arange(0, 8)
    offsets = r[:, None] * 8 + r[None, :]
    acc = lang.zeros((8, 8), lang.float32)
    for k in range(0, K, 2):
        x = lang.load(a + offsets, True, 0.0)
        y = lang.load(b + offsets, mask=True, other=0.0)
        acc = lang.dot(x, y, acc=acc)
        acc = acc.to(lang.bfloat16).to(lang.float32)
    lang.store(out + offsets, acc, True)
