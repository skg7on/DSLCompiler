"""Genuine upstream TTIR export into neutral unbound templates."""

import hashlib
import json
from pathlib import Path

import pytest

import llk.triton as triton_export
from llk._errors import CompilationError, FrontendSyntaxError, UnsupportedFeatureError
from llk.triton import UnboundFrontendTemplate

ADD_SOURCE = """\
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
"""

MUL_SOURCE = """\
import triton
import triton.language as tl


@triton.jit
def mul_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    y = tl.load(y_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, x * y, mask=mask)
"""

LOOP_CAST_SOURCE = """\
import triton
import triton.language as tl


@triton.jit
def scale_kernel(x_ptr, out_ptr, n, alpha, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    start = pid * BLOCK
    for i in tl.range(0, 2):
        offs = start + i * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n
        x = tl.load(x_ptr + offs, mask=mask)
        y = x.to(tl.float32) * alpha
        tl.store(out_ptr + offs, y, mask=mask)
"""

ATOMIC_SOURCE = """\
import triton
import triton.language as tl


@triton.jit
def atomic_kernel(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    tl.atomic_add(out_ptr + offs, x, mask=mask)
"""

LAYOUT_SOURCE = """\
import triton
import triton.language as tl


@triton.jit
def layout_kernel(x_ptr, out_ptr, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    x = tl.load(x_ptr + offs)
    tile = tl.reshape(x, (8, BLOCK // 8))
    transposed = tl.trans(tile)
    tl.store(out_ptr + offs, tl.reshape(transposed, (BLOCK,)))
"""

ALIAS_SOURCE = """\
import triton
import triton.language as tl


@triton.jit
def alias_kernel(ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(ptr + offs, mask=mask)
    tl.store(ptr + offs, x, mask=mask)
"""

BROKEN_SOURCE = """\
import triton
import triton.language as tl


@triton.jit
def broken_kernel(x_ptr, BLOCK: tl.constexpr):
    return x_ptr +
"""

ADD_SIGNATURE = {
    "x_ptr": "*fp32",
    "y_ptr": "*fp32",
    "out_ptr": "*fp32",
    "n": "i32",
    "BLOCK": "constexpr",
}
SPECIALIZATION = {
    "x_ptr": {"divisibility": 16},
    "y_ptr": {"divisibility": 16},
    "out_ptr": {"divisibility": 16},
}


def _write(tmp_path, name, source):
    directory = tmp_path / "triton export café"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    path.write_text(source, encoding="utf-8")
    return path


def _capture(tmp_path, triton_path, name, source, function, signature, **kwargs):
    path = _write(tmp_path, name, source)
    export = triton_export.capture_kernel(
        path,
        function,
        signature,
        triton_python=str(triton_path),
        **kwargs,
    )
    return path, export


def _flatten(operations):
    for operation in operations:
        yield operation
        for region in operation.regions:
            yield from _flatten(region.operations)


def test_add_export_preserves_typed_parameters_and_masks(tmp_path, triton_path):
    path, export = _capture(
        tmp_path,
        triton_path,
        "add kernel.py",
        ADD_SOURCE,
        "add_kernel",
        ADD_SIGNATURE,
        constexprs={"BLOCK": 1024},
        specialization=SPECIALIZATION,
        options={"target": "cuda", "arch": 80},
    )
    ttir = export.ttir
    assert ttir.startswith("module {")
    assert "tt.func public @add_kernel" in ttir
    for op in ("tt.get_program_id", "tt.load", "tt.store", "arith.addf"):
        assert op in ttir
    assert "tt.divisibility = 16" in ttir
    assert "loc(" not in ttir

    template = export.template
    assert template.schema_version == 1
    assert template.required_runtime_grid is True
    parameters = {parameter.name: parameter for parameter in template.parameters}
    assert parameters["x_ptr"].kind == "pointer"
    assert parameters["x_ptr"].dtype == "f32"
    assert parameters["x_ptr"].access == "read"
    assert parameters["x_ptr"].alignment_bytes == 16
    assert parameters["y_ptr"].access == "read"
    assert parameters["out_ptr"].access == "write"
    assert parameters["out_ptr"].alignment_bytes == 16
    assert parameters["n"].kind == "scalar"
    assert parameters["n"].dtype == "i32"
    assert parameters["n"].access == "none"
    assert parameters["BLOCK"].kind == "constexpr"
    assert parameters["BLOCK"].value == 1024
    assert parameters["BLOCK"].value_id == ""
    assert template.constexprs == {"BLOCK": 1024}

    operations = list(_flatten(template.operations))
    opcodes = {operation.opcode for operation in operations}
    assert {
        "program_id",
        "range",
        "splat",
        "add_ptr",
        "load",
        "store",
        "addf",
        "cmpi",
        "return",
    } <= opcodes
    for operation in operations:
        if operation.opcode in ("load", "store"):
            assert operation.attributes["masked"] is True
        if operation.opcode == "program_id":
            assert operation.attributes["axis"] == 0
    masks = [
        operation
        for operation in operations
        if operation.opcode == "cmpi"
        and operation.attributes["predicate"] == "slt"
    ]
    assert masks

    assert template.source_hash == hashlib.sha256(path.read_bytes()).hexdigest()
    assert template.ttir_hash == hashlib.sha256(ttir.encode("utf-8")).hexdigest()
    assert export.toolchain == template.toolchain
    assert export.toolchain.triton_version == "3.6.0"
    assert export.toolchain.target == "cuda:80"

    payload = template.to_json()
    assert all("shape" not in parameter for parameter in payload["parameters"])
    assert "grid" not in payload
    assert (1024,) in {tuple(value.shape) for value in template.values}


def test_multiply_export_uses_multiplication(tmp_path, triton_path):
    _, export = _capture(
        tmp_path,
        triton_path,
        "mul kernel.py",
        MUL_SOURCE,
        "mul_kernel",
        ADD_SIGNATURE,
        constexprs={"BLOCK": 1024},
        specialization=SPECIALIZATION,
    )
    opcodes = {operation.opcode for operation in _flatten(export.template.operations)}
    assert "mulf" in opcodes
    assert "addf" not in opcodes
    assert "tt.func public @mul_kernel" in export.ttir


def test_template_roundtrip_preserves_identity(tmp_path, triton_path):
    _, export = _capture(
        tmp_path,
        triton_path,
        "add kernel.py",
        ADD_SOURCE,
        "add_kernel",
        ADD_SIGNATURE,
        constexprs={"BLOCK": 1024},
        specialization=SPECIALIZATION,
    )
    payload = export.template.to_json()
    restored = UnboundFrontendTemplate.from_json(json.loads(json.dumps(payload)))
    assert restored == export.template
    assert restored.identity() == export.template.identity()


def test_capture_is_deterministic_across_paths(tmp_path, triton_path):
    first = _capture(
        tmp_path / "first",
        triton_path,
        "add kernel.py",
        ADD_SOURCE,
        "add_kernel",
        ADD_SIGNATURE,
        constexprs={"BLOCK": 1024},
        specialization=SPECIALIZATION,
    )[1]
    second = _capture(
        tmp_path / "second",
        triton_path,
        "add kernel.py",
        ADD_SOURCE,
        "add_kernel",
        ADD_SIGNATURE,
        constexprs={"BLOCK": 1024},
        specialization=SPECIALIZATION,
    )[1]
    assert first.ttir == second.ttir
    assert first.template.identity() == second.template.identity()


def test_counted_loop_and_cast_regions_are_preserved(tmp_path, triton_path):
    _, export = _capture(
        tmp_path,
        triton_path,
        "scale kernel.py",
        LOOP_CAST_SOURCE,
        "scale_kernel",
        {
            "x_ptr": "*bf16",
            "out_ptr": "*fp32",
            "n": "i32",
            "alpha": "fp32",
            "BLOCK": "constexpr",
        },
        constexprs={"BLOCK": 64},
    )
    operations = list(_flatten(export.template.operations))
    loop = [operation for operation in operations if operation.opcode == "for"]
    assert len(loop) == 1
    assert loop[0].regions and loop[0].regions[0].operations
    assert loop[0].regions[0].arguments
    casts = [operation for operation in operations if operation.opcode == "cast"]
    assert any(
        operation.attributes["from"] == "bf16"
        and operation.attributes["to"] == "f32"
        for operation in casts
    )
    parameters = {parameter.name: parameter for parameter in export.template.parameters}
    assert parameters["alpha"].kind == "scalar"
    assert parameters["alpha"].dtype == "f32"


def test_atomic_memory_effect_is_rejected(tmp_path, triton_path):
    path = _write(tmp_path, "atomic kernel.py", ATOMIC_SOURCE)
    with pytest.raises(UnsupportedFeatureError) as caught:
        triton_export.capture_kernel(
            path,
            "atomic_kernel",
            {"x_ptr": "*fp32", "out_ptr": "*fp32", "n": "i32", "BLOCK": "constexpr"},
            constexprs={"BLOCK": 64},
            triton_python=str(triton_path),
        )
    assert caught.value.code == "unsupported_triton_memory_effect"


def test_unsupported_layout_is_rejected(tmp_path, triton_path):
    path = _write(tmp_path, "layout kernel.py", LAYOUT_SOURCE)
    with pytest.raises(UnsupportedFeatureError) as caught:
        triton_export.capture_kernel(
            path,
            "layout_kernel",
            {"x_ptr": "*fp32", "out_ptr": "*fp32", "BLOCK": "constexpr"},
            constexprs={"BLOCK": 64},
            triton_python=str(triton_path),
        )
    assert caught.value.code == "unsupported_triton_layout"


def test_read_write_alias_is_rejected(tmp_path, triton_path):
    path = _write(tmp_path, "alias kernel.py", ALIAS_SOURCE)
    with pytest.raises(UnsupportedFeatureError) as caught:
        triton_export.capture_kernel(
            path,
            "alias_kernel",
            {"ptr": "*fp32", "n": "i32", "BLOCK": "constexpr"},
            constexprs={"BLOCK": 64},
            triton_python=str(triton_path),
        )
    assert caught.value.code == "unsupported_triton_memory_effect"


def test_compilation_failure_is_structured(tmp_path, triton_path):
    path = _write(tmp_path, "broken kernel.py", BROKEN_SOURCE)
    with pytest.raises(CompilationError) as caught:
        triton_export.capture_kernel(
            path,
            "broken_kernel",
            {"x_ptr": "*fp32", "BLOCK": "constexpr"},
            constexprs={"BLOCK": 64},
            triton_python=str(triton_path),
        )
    assert caught.value.code == "triton_compilation_failed"


def test_signature_order_must_match_kernel_arguments(tmp_path, triton_path):
    path = _write(tmp_path, "add kernel.py", ADD_SOURCE)
    with pytest.raises(FrontendSyntaxError) as caught:
        triton_export.capture_kernel(
            path,
            "add_kernel",
            {
                "n": "i32",
                "x_ptr": "*fp32",
                "y_ptr": "*fp32",
                "out_ptr": "*fp32",
                "BLOCK": "constexpr",
            },
            constexprs={"BLOCK": 1024},
            triton_python=str(triton_path),
        )
    assert caught.value.code == "invalid_triton_signature"


def test_export_target_mismatch_fails_qualification(tmp_path, triton_path):
    path = _write(tmp_path, "add kernel.py", ADD_SOURCE)
    with pytest.raises(UnsupportedFeatureError) as caught:
        triton_export.capture_kernel(
            path,
            "add_kernel",
            ADD_SIGNATURE,
            constexprs={"BLOCK": 1024},
            options={"target": "cuda", "arch": 90},
            triton_python=str(triton_path),
        )
    assert caught.value.code == "upstream_toolchain_unqualified"


def test_stdout_log_noise_does_not_corrupt_the_result(tmp_path, triton_path):
    source = 'print("upstream log noise on stdout")\n' + ADD_SOURCE
    _, export = _capture(
        tmp_path,
        triton_path,
        "noisy kernel.py",
        source,
        "add_kernel",
        ADD_SIGNATURE,
        constexprs={"BLOCK": 1024},
        specialization=SPECIALIZATION,
    )
    assert "upstream log noise" not in export.ttir
    assert "tt.func public @add_kernel" in export.ttir
