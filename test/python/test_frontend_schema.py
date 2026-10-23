"""PF03 wire contract and semantic validation, without any native execution."""

import json
from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import pytest
from frontend_cases import make_add_program, make_loop_program, op, request
from llk import LLKError
from llk.frontend import (
    CompileOptions,
    CompileRequest,
    LaunchSpec,
    NumericalContract,
    TensorSpec,
    Value,
    deserialize_request,
    serialize_request,
)


def test_canonical_request_matches_literal_fixture():
    expected = (
        Path(__file__).with_name("frontend_add_v1.json").read_bytes().rstrip(b"\n")
    )
    assert serialize_request(request()) == expected
    assert serialize_request(deserialize_request(expected)) == expected


def test_mapping_order_is_canonical_but_port_order_is_semantic():
    program = make_add_program()
    reordered = replace(program, constexprs={"BLOCK_N": 16, "BLOCK_M": 8})
    assert serialize_request(request(program)) == serialize_request(request(reordered))
    assert serialize_request(request(program)) != serialize_request(
        request(replace(program, tensors=program.tensors[::-1]))
    )


def test_records_are_deeply_immutable_and_snapshot_callers():
    attrs = {"value": 1}
    operation = op("constant", results=(Value("n", "index", ()),), attributes=attrs)
    attrs["value"] = 9
    assert operation.attributes["value"] == 1
    with pytest.raises(TypeError):
        operation.attributes["value"] = 2
    with pytest.raises(FrozenInstanceError):
        operation.opcode = "return"
    program = make_add_program()
    with pytest.raises(TypeError):
        program.constexprs["BLOCK_M"] = 32


@pytest.mark.parametrize(
    "field,value", [("version", True), ("version", 2), ("extra", 1)]
)
def test_unknown_fields_and_versions_rejected(field, value):
    data = json.loads(serialize_request(request()))
    data["program"][field] = value
    with pytest.raises(LLKError):
        deserialize_request(json.dumps(data))


@pytest.mark.parametrize(
    "grid,block",
    [
        ((True, 1, 1), (1, 1, 1)),
        ((0, 1, 1), (1, 1, 1)),
        ((1, 1, 1), (0, 1, 1)),
        ((1, 1, 1, 1), (1, 1, 1)),
        ((2**63, 1, 1), (1, 1, 1)),
        ((2**62, 4, 1), (1, 1, 1)),
    ],
)
def test_invalid_launch_metadata(grid, block):
    with pytest.raises(LLKError):
        LaunchSpec(grid, block)


@pytest.mark.parametrize(
    "shape,strides",
    [
        ((0, 64), (64, 1)),
        ((-1, 64), (64, 1)),
        ((32, 64), (65, 1)),
        ((32,), (1,)),
        ((2**62, 8), (8, 1)),
        ((True, 64), (64, 1)),
    ],
)
def test_tensor_contract_rejects_rank_layout_and_overflow(shape, strides):
    with pytest.raises(LLKError):
        TensorSpec("a", shape, "f32", strides, "input", "read")


def test_integer_indices_are_exact_and_nonfinite_constants_fail():
    p = make_loop_program()
    first = replace(p.operations[0], attributes={"value": 2**53 + 1})
    assert str(2**53 + 1).encode() in serialize_request(
        request(replace(p, operations=(first,) + p.operations[1:]))
    )
    for x in (float("nan"), float("inf"), float("-inf"), 2**63, -(2**63) - 1):
        with pytest.raises(LLKError):
            request(replace(make_add_program(), constexprs={"bad": x}))


def test_source_capture_launch_and_numerical_policy_change_identity():
    p = make_add_program()
    baseline = serialize_request(request(p))
    changes = [
        replace(p, source=p.source + "# source changed"),
        replace(p, captured_constants={"shift": 1}),
        replace(p, constexprs={"BLOCK_M": 8, "BLOCK_N": 16, "tag": 1}),
        replace(p, launch=LaunchSpec((2, 4, 1), (16, 16, 1))),
        replace(p, numerical_contract=NumericalContract(fma=True)),
    ]
    for changed in changes:
        assert serialize_request(request(changed)) != baseline


def test_nested_loop_regions_and_cast_metadata_roundtrip():
    p = make_loop_program()
    result = deserialize_request(serialize_request(request(p)))
    assert result.program.operations[3] == p.operations[3]
    assert (
        result.program.numerical_contract.cast_sites == p.numerical_contract.cast_sites
    )


def test_loop_local_values_cannot_escape():
    p = make_loop_program()
    with pytest.raises(LLKError):
        replace(
            p,
            operations=p.operations[:-1]
            + (
                op("add", ("restored", "zero"), (Value("escape", "f32", ()),)),
                op("return"),
            ),
        )


def test_loop_yield_types_and_cast_sites_are_required():
    p = make_loop_program()
    outer = p.operations[3]
    region = outer.regions[0]
    bad = replace(region, operations=region.operations[:-1] + (op("yield", ("lo",)),))
    with pytest.raises(LLKError):
        replace(
            p,
            operations=p.operations[:3]
            + (replace(outer, regions=(bad,)), op("return")),
        )
    with pytest.raises(LLKError):
        replace(p, numerical_contract=NumericalContract())
    with pytest.raises(LLKError):
        replace(
            p,
            operations=p.operations[:3]
            + (replace(outer, attributes={"step": 0}), op("return")),
        )


def test_undefined_duplicate_ids_masks_and_opcode_schemas_fail():
    p = make_add_program()
    cases = [
        replace(p.operations[15], operands=("missing", "mask", "zero")),
        replace(p.operations[15], operands=("pa", "zero", "zero")),
        replace(p.operations[15], operands=("pa",)),
        replace(p.operations[15], opcode="unknown"),
        replace(p.operations[0], attributes={"axis": 3}),
        replace(p.operations[0], attributes={"axis": 0, "ignored": True}),
        replace(p.operations[0], results=(Value("a", "index", ()),)),
    ]
    for bad in cases:
        ops = list(p.operations)
        ops[15 if bad.opcode in ("load", "unknown") else 0] = bad
        with pytest.raises(LLKError):
            replace(p, operations=tuple(ops))


def test_json_rejects_duplicate_keys_and_unknown_nested_fields():
    with pytest.raises(LLKError):
        deserialize_request('{"program":{},"program":{},"options":{}}')
    data = json.loads(serialize_request(request()))
    data["program"]["operations"][0]["span"]["extra"] = 1
    with pytest.raises(LLKError):
        deserialize_request(json.dumps(data))


def test_config_content_is_preserved_in_request_identity(tmp_path):
    (tmp_path / "machine.yaml").write_text("nodes: first\n")
    first = serialize_request(request(config_root=str(tmp_path)))
    (tmp_path / "machine.yaml").write_text("nodes: second\n")
    assert first != serialize_request(request(config_root=str(tmp_path)))


def test_checked_json_schema_matches_canonical_fixture():
    import jsonschema

    schema = json.loads(
        (
            Path(__file__).parents[2] / "docs/design/python-frontend-v1.schema.json"
        ).read_text()
    )
    validator = jsonschema.Draft202012Validator(schema)
    validator.check_schema(schema)
    data = json.loads(serialize_request(request()))
    validator.validate(data)
    for path, value in [("version", 2), ("version", True), ("unknown", 0)]:
        bad = json.loads(json.dumps(data))
        bad["program"][path] = value
        assert list(validator.iter_errors(bad))
    validator.validate(json.loads(serialize_request(request(make_loop_program()))))


def test_request_cannot_contradict_program_numerical_policy():
    program = make_add_program()
    with pytest.raises(LLKError):
        CompileRequest(
            program, CompileOptions(numerical_contract=NumericalContract(fma=True))
        )


def test_config_snapshot_decode_errors_are_structured(tmp_path):
    (tmp_path / "broken.yaml").write_bytes(b"\xff")
    with pytest.raises(LLKError):
        CompileOptions(config_root=str(tmp_path))


def test_wire_config_snapshot_is_required_without_filesystem_reads(monkeypatch):
    import llk.frontend._ir as ir

    def unexpected_snapshot(root):
        raise AssertionError("decoding must never snapshot the local filesystem")

    monkeypatch.setattr(ir, "config_snapshot", unexpected_snapshot)
    data = json.loads(serialize_request(request()))
    data["options"]["config_files"] = None
    with pytest.raises(LLKError):
        deserialize_request(json.dumps(data))


def test_config_snapshot_rejects_empty_paths():
    with pytest.raises(LLKError):
        CompileOptions(config_root=".", config_files={"": "content"})
