"""Genuine upstream TTIR export into neutral unbound templates."""


import hashlib
import json
from pathlib import Path


import llk.triton as triton_export
from llk._errors import CompilationError, FrontendSyntaxError, UnsupportedFeatureError
from llk.triton import UnboundFrontendTemplate

import pytest

pytestmark = pytest.mark.triton_integration


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


# ---------------------------------------------------------------------------
# PF19: contraction and fused-activation export extensions.
# ---------------------------------------------------------------------------

KERNELS = Path(__file__).with_name("kernels")

MATMUL_SIGNATURE = {
    "a_ptr": "*fp32",
    "b_ptr": "*fp32",
    "out_ptr": "*fp32",
    "M": "i32",
    "N": "i32",
    "K": "i32",
    "BM": "constexpr",
    "BN": "constexpr",
    "BK": "constexpr",
}
MATMUL_CONSTEXPRS = {"BM": 16, "BN": 16, "BK": 16}

SWIGLU_SIGNATURE = {
    "x_ptr": "*fp32",
    "gate_ptr": "*fp32",
    "up_ptr": "*fp32",
    "out_ptr": "*fp32",
    "M": "i32",
    "N": "i32",
    "K": "i32",
    "BM": "constexpr",
    "BN": "constexpr",
    "BK": "constexpr",
}


def _export(tmp_path, triton_path, source, name, function, signature, constexprs=None):
    path = _write(tmp_path, name, source)
    return triton_export.capture_kernel(
        path,
        function,
        signature,
        constexprs=constexprs or {},
        triton_python=str(triton_path),
    )


def _module_export(triton_path, module, function, signature, constexprs):
    return triton_export.capture_kernel(
        KERNELS / module,
        function,
        signature,
        constexprs=constexprs,
        triton_python=str(triton_path),
    )


def test_contraction_export_preserves_dot_and_rank_two_views(triton_path):
    export = _module_export(
        triton_path, "matmul.py", "matmul_kernel", MATMUL_SIGNATURE, MATMUL_CONSTEXPRS
    )
    assert "tt.dot" in export.ttir
    assert "inputPrecision = tf32" not in export.ttir
    operations = list(_flatten(export.template.operations))
    opcodes = {operation.opcode for operation in operations}
    assert {"dot", "expand_dims", "broadcast", "for", "load", "store"} <= opcodes
    zeroes = [
        operation
        for operation in operations
        if operation.opcode == "constant"
        and operation.attributes.get("value") == 0.0
    ]
    assert zeroes, "the zero accumulator tile must survive as a typed constant"
    loops = [operation for operation in operations if operation.opcode == "for"]
    assert len(loops) == 1 and loops[0].regions[0].arguments


def test_accumulated_matmul_seed_is_a_caller_tile(triton_path):
    export = _module_export(
        triton_path,
        "matmul.py",
        "matmul_accumulate_kernel",
        {
            "a_ptr": "*fp32",
            "b_ptr": "*fp32",
            "init_ptr": "*fp32",
            "out_ptr": "*fp32",
            "M": "i32",
            "N": "i32",
            "K": "i32",
            "BM": "constexpr",
            "BN": "constexpr",
            "BK": "constexpr",
        },
        MATMUL_CONSTEXPRS,
    )
    parameters = {parameter.name: parameter for parameter in export.template.parameters}
    assert parameters["init_ptr"].access == "read"
    operations = list(_flatten(export.template.operations))
    loop = next(operation for operation in operations if operation.opcode == "for")
    assert loop.operands, "the caller tile must be the loop-carried seed"


def test_fused_activation_export_preserves_silu(triton_path):
    export = _module_export(
        triton_path, "swiglu.py", "swiglu_kernel", SWIGLU_SIGNATURE, MATMUL_CONSTEXPRS
    )
    assert "math.exp" in export.ttir
    operations = list(_flatten(export.template.operations))
    opcodes = [operation.opcode for operation in operations]
    assert "silu" in opcodes
    # The recognized chain is replaced, not duplicated: no residual elementwise
    # sigmoid survives in the neutral template.
    assert "mulf" in opcodes
    silus = [operation for operation in operations if operation.opcode == "silu"]
    assert len(silus) == 1 and len(silus[0].operands) == 1


TF32_SOURCE = """\
import triton
import triton.language as tl


@triton.jit
def matmul_default_precision(a_ptr, b_ptr, out_ptr, M, N, K,
                             BM: tl.constexpr, BN: tl.constexpr,
                             BK: tl.constexpr):
    pm = tl.program_id(0)
    pn = tl.program_id(1)
    rows = pm * BM + tl.arange(0, BM)
    cols = pn * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)
    acc = tl.zeros((BM, BN), tl.float32)
    for kk in range(0, K, BK):
        at = tl.load(a_ptr + rows[:, None] * K + (kk + rk)[None, :])
        bt = tl.load(b_ptr + (kk + rk)[:, None] * N + cols[None, :])
        acc = tl.dot(at, bt, acc)
    tl.store(out_ptr + rows[:, None] * N + cols[None, :], acc)
"""

PIPELINE_SOURCE = """\
import triton
import triton.language as tl


@triton.jit
def pipelined_kernel(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    acc = tl.zeros((BLOCK,), tl.float32)
    for i in tl.range(0, 4, num_stages=2):
        acc += tl.load(x_ptr + offs + i * BLOCK)
    tl.store(out_ptr + offs, acc)
"""

REDUCE_SOURCE = """\
import triton
import triton.language as tl


@triton.jit
def reduce_kernel(x_ptr, out_ptr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    tl.store(out_ptr + pid, tl.sum(tl.load(x_ptr + offs)))
"""

INLINE_ASM_SOURCE = """\
import triton
import triton.language as tl


@triton.jit
def asm_kernel(x_ptr, out_ptr, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(x_ptr + offs)
    y = tl.inline_asm_elementwise(
        "mov.b32 $0, $1;", "=r,r", [x], dtype=tl.float32, is_pure=True, pack=1
    )
    tl.store(out_ptr + offs, y)
"""


def test_tf32_dot_precision_is_rejected(tmp_path, triton_path):
    with pytest.raises(UnsupportedFeatureError) as caught:
        _export(
            tmp_path,
            triton_path,
            TF32_SOURCE,
            "tf32 matmul.py",
            "matmul_default_precision",
            MATMUL_SIGNATURE,
            MATMUL_CONSTEXPRS,
        )
    assert caught.value.code == "unsupported_triton_numerics"


def test_software_pipeline_request_is_rejected(tmp_path, triton_path):
    with pytest.raises(UnsupportedFeatureError) as caught:
        _export(
            tmp_path,
            triton_path,
            PIPELINE_SOURCE,
            "pipeline kernel.py",
            "pipelined_kernel",
            {"x_ptr": "*fp32", "out_ptr": "*fp32", "n": "i32", "BLOCK": "constexpr"},
            {"BLOCK": 64},
        )
    assert caught.value.code == "unsupported_triton_memory_effect"


def test_reduction_is_rejected(tmp_path, triton_path):
    with pytest.raises(UnsupportedFeatureError) as caught:
        _export(
            tmp_path,
            triton_path,
            REDUCE_SOURCE,
            "reduce kernel.py",
            "reduce_kernel",
            {"x_ptr": "*fp32", "out_ptr": "*fp32", "BLOCK": "constexpr"},
            {"BLOCK": 64},
        )
    assert caught.value.code == "unsupported_triton_operation"


def test_gpu_inline_assembly_is_rejected(tmp_path, triton_path):
    with pytest.raises(UnsupportedFeatureError) as caught:
        _export(
            tmp_path,
            triton_path,
            INLINE_ASM_SOURCE,
            "asm kernel.py",
            "asm_kernel",
            {"x_ptr": "*fp32", "out_ptr": "*fp32", "BLOCK": "constexpr"},
            {"BLOCK": 64},
        )
    assert caught.value.code == "unsupported_triton_operation"


SEED_ONE_SOURCE = """\
import triton
import triton.language as tl


@triton.jit
def matmul_seeded_with_one(a_ptr, b_ptr, out_ptr, M, N, K,
                           BM: tl.constexpr, BN: tl.constexpr,
                           BK: tl.constexpr):
    pm = tl.program_id(0)
    pn = tl.program_id(1)
    rows = pm * BM + tl.arange(0, BM)
    cols = pn * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)
    acc = tl.full((BM, BN), 1.0, tl.float32)
    for kk in range(0, K, BK):
        at = tl.load(a_ptr + rows[:, None] * K + (kk + rk)[None, :])
        bt = tl.load(b_ptr + (kk + rk)[:, None] * N + cols[None, :])
        acc = tl.dot(at, bt, acc, input_precision="ieee")
    tl.store(out_ptr + rows[:, None] * N + cols[None, :], acc)
"""


def test_nonzero_accumulator_seed_is_preserved_not_zeroed(tmp_path, triton_path):
    export = _export(
        tmp_path,
        triton_path,
        SEED_ONE_SOURCE,
        "seeded matmul.py",
        "matmul_seeded_with_one",
        MATMUL_SIGNATURE,
        MATMUL_CONSTEXPRS,
    )
    operations = list(_flatten(export.template.operations))
    tiles = [
        operation
        for operation in operations
        if operation.opcode == "constant"
        and operation.attributes.get("value") == 1.0
    ]
    assert tiles, "a one-initialized accumulator must not become a zero seed"
