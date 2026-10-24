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
