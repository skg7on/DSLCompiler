"""Typed capture tests assert dataflow, never textual opcode substrings."""

from dataclasses import replace

import llk.language as lang
import pytest
from ast_cases import binding, counted_dot, masked_add, ports
from llk.frontend import (
    CompileRequest,
    FrontendProgram,
    deserialize_request,
    serialize_request,
)
from llk.language._ast_compiler import compile_kernel


def capture(fn, args=None):
    return compile_kernel(fn, binding(fn, args))


def test_keyword_mask_and_other_survive_capture():
    program = capture(masked_add)
    assert isinstance(program, FrontendProgram)
    operations = program.operations
    loads = [op for op in operations if op.opcode == "load"]
    store = next(op for op in operations if op.opcode == "store")
    definitions = {result.id: op for op in operations for result in op.results}
    x, y = loads
    assert x.operands[1] == y.operands[1] == store.operands[2]
    assert definitions[x.operands[2]].attributes == {"value": 2.0}
    assert definitions[y.operands[2]].attributes == {"value": 3.0}
    addition = definitions[store.operands[1]]
    assert addition.opcode == "add"
    assert addition.operands == (x.results[0].id, y.results[0].id)
    assert x.results[0].shape == (4, 4)
    pointer = definitions[x.operands[0]]
    assert pointer.opcode == "address" and pointer.operands[0] == "a"
    assert any(op.opcode == "mul" for op in operations)
    assert any(op.opcode == "compare" for op in operations)
    assert any(op.opcode == "expand_dims" for op in operations)
    assert operations[-1].opcode == "return"


def test_assignment_aliases_keep_ssa_identity_and_recursive_multiply():
    def multiply(a, b, out, N: lang.constexpr):
        offsets = lang.arange(0, N)
        x = lang.load(a + offsets, True, 0.0)
        alias = x
        y = lang.load(b + offsets, True, 0.0)
        lang.store(out + offsets, alias * (y + 2.0), True)

    p = capture(multiply)
    ops = p.operations
    loads = [o for o in ops if o.opcode == "load"]
    mul = next(o for o in ops if o.opcode == "mul")
    plus = next(o for o in ops if o.opcode == "add" and o.results[0].dtype == "f32")
    assert mul.operands == (loads[0].results[0].id, plus.results[0].id)
    assert plus.operands[0] == loads[1].results[0].id


def test_constexpr_branch_only_captures_selected_computation():
    def choose(a, b, out, N: lang.constexpr):
        offsets = lang.arange(0, 8)
        x = lang.load(a + offsets, True, 0.0)
        if N == 8:
            result = x * 2.0
        else:
            result = x + 3.0
        lang.store(out + offsets, result, True)

    first = capture(choose)
    second = capture(choose, ports() + (4,))
    assert [
        o.opcode for o in first.operations if o.results and o.results[0].dtype == "f32"
    ] == ["constant", "load", "constant", "mul"]
    assert "add" in [o.opcode for o in second.operations]
    assert "mul" not in [o.opcode for o in second.operations]


def test_counted_dot_uses_typed_loop_yield_and_preserves_casts():
    p = capture(counted_dot)
    loop = next(op for op in p.operations if op.opcode == "counted_loop")
    assert loop.attributes == {"step": 2}
    assert len(loop.operands) == 3 and len(loop.results) == 1
    body = loop.regions[0]
    assert [(v.dtype, v.shape) for v in body.arguments] == [
        ("index", ()),
        ("f32", (8, 8)),
    ]
    dot = next(op for op in body.operations if op.opcode == "dot")
    assert dot.operands[2] == body.arguments[1].id
    casts = [op for op in body.operations if op.opcode == "cast"]
    assert [
        (c.source_dtype, c.target_dtype) for c in p.numerical_contract.cast_sites
    ] == [("f32", "bf16"), ("bf16", "f32")]
    assert body.operations[-1].operands == (casts[-1].results[0].id,)
    store = next(op for op in p.operations if op.opcode == "store")
    assert store.operands[1] == loop.results[0].id
    opts = replace(
        binding(counted_dot).options, numerical_contract=p.numerical_contract
    )
    assert deserialize_request(
        serialize_request(CompileRequest(p, opts))
    ) == CompileRequest(p, opts)


def test_capture_does_not_execute_language_stubs():
    for fn, args in [(lang.arange, (0, 8)), (lang.program_id, (0,))]:
        with pytest.raises(Exception) as caught:
            fn(*args)
        assert caught.value.code == "outside_frontend_capture"


def test_nested_counted_loops_preserve_accumulator_region_connections():
    def nested(a, b, out, N: lang.constexpr):
        r = lang.arange(0, 8)
        offsets = r[:, None] * 8 + r[None, :]
        acc = lang.zeros((8, 8), lang.float32)
        for outer in range(0, N, 4):
            for inner in range(0, 4, 2):
                x = lang.load(a + offsets, True, 0.0)
                y = lang.load(b + offsets, True, 0.0)
                acc = lang.dot(x, y, acc)
        lang.store(out + offsets, acc, True)

    p = capture(nested)
    outer = next(o for o in p.operations if o.opcode == "counted_loop")
    inner = next(o for o in outer.regions[0].operations if o.opcode == "counted_loop")
    assert inner.operands[2] == outer.regions[0].arguments[1].id
    assert outer.regions[0].operations[-1].operands == (inner.results[0].id,)
    dot = next(o for o in inner.regions[0].operations if o.opcode == "dot")
    assert dot.operands[2] == inner.regions[0].arguments[1].id
    assert inner.regions[0].operations[-1].operands == (dot.results[0].id,)


def test_recognized_silu_expression_has_exact_numerical_metadata():
    def activation(a, b, out, N: lang.constexpr):
        offsets = lang.arange(0, N)
        x = lang.load(a + offsets, True, 0.0)
        y = x / (1 + lang.exp(-x))
        lang.store(out + offsets, y, True)

    p = capture(activation)
    silu = next(o for o in p.operations if o.opcode == "silu")
    load = next(o for o in p.operations if o.opcode == "load")
    assert silu.operands == (load.results[0].id,)
    assert silu.attributes == {"approximation": "exact"}
    assert p.numerical_contract.activation_expression == "x/(1+exp(-x))"
    assert p.numerical_contract.activation_approximation == "exact"
    assert not p.numerical_contract.reassociation and not p.numerical_contract.fma


def test_binding_freezes_closure_values_used_by_capture():
    shift = 3.0

    def kernel(a, b, out, N: lang.constexpr):
        offsets = lang.arange(0, N)
        x = lang.load(a + offsets, True, 0.0)
        lang.store(out + offsets, x + shift, True)

    bound = binding(kernel)
    shift = 9.0
    p = compile_kernel(kernel, bound)
    assert p.captured_constants == {"shift": 3.0}
    assert any(
        o.opcode == "constant" and o.attributes["value"] == 3.0 for o in p.operations
    )
    assert not any(
        o.opcode == "constant" and o.attributes["value"] == 9.0 for o in p.operations
    )


def test_where_cdiv_mask_or_and_explicit_cast_call_preserve_operands():
    def kernel(a, b, out, N: lang.constexpr):
        offsets = lang.arange(0, lang.cdiv(N, 3))
        mask = (offsets < 2) | (offsets == 2)
        x = lang.load(a + offsets, mask, 0.0)
        rounded = lang.cast(x, lang.bfloat16, rounding="rtne")
        restored = lang.cast(rounded, lang.float32)
        result = lang.where(mask, restored, 2.0)
        lang.store(out + offsets, result, mask)

    p = capture(kernel)
    ops = p.operations
    extent = next(o for o in ops if o.opcode == "range")
    assert extent.attributes == {"start": 0, "end": 3}
    casts = [o for o in ops if o.opcode == "cast"]
    assert casts[1].operands == (casts[0].results[0].id,)
    selects = [o for o in ops if o.opcode == "select"]
    loaded = next(o for o in ops if o.opcode == "load")
    assert loaded.operands[1] == selects[0].results[0].id
    assert selects[1].operands[:2] == (selects[0].results[0].id, casts[1].results[0].id)
    assert [
        (c.source_dtype, c.target_dtype) for c in p.numerical_contract.cast_sites
    ] == [("f32", "bf16"), ("bf16", "f32")]


def test_unary_negation_uses_exact_operation_and_preserves_literal_signed_zero():
    import json
    import math
    from pathlib import Path

    import jsonschema

    def kernel(a, b, out, N: lang.constexpr):
        offsets = lang.arange(0, N)
        x = lang.load(a + offsets, True, -0.0)
        lang.store(out + offsets, -x, True)

    p = capture(kernel)
    load = next(o for o in p.operations if o.opcode == "load")
    padding = next(
        o for o in p.operations if o.results and o.results[0].id == load.operands[2]
    )
    assert math.copysign(1.0, padding.attributes["value"]) == -1.0
    neg = next(o for o in p.operations if o.opcode == "neg")
    assert neg.operands == (load.results[0].id,)
    assert neg.results[0].dtype == "f32"
    store = next(o for o in p.operations if o.opcode == "store")
    assert store.operands[1] == neg.results[0].id
    opts = replace(binding(kernel).options, numerical_contract=p.numerical_contract)
    request = CompileRequest(p, opts)
    assert serialize_request(
        deserialize_request(serialize_request(request))
    ) == serialize_request(request)
    schema = json.loads(
        (
            Path(__file__).parents[2] / "docs/design/python-frontend-v1.schema.json"
        ).read_text()
    )
    jsonschema.Draft202012Validator(schema).validate(
        json.loads(serialize_request(request))
    )
