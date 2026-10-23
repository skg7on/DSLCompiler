"""Canonical JSON v1 with strict, non-executing decoding."""

import json
from collections.abc import Mapping
from dataclasses import fields, is_dataclass

from ._ir import (
    CastSite,
    CompileOptions,
    CompileRequest,
    FrontendProgram,
    LaunchSpec,
    NumericalContract,
    Operation,
    Region,
    SourceSpan,
    TensorSpec,
    Value,
    fail,
)


def to_data(value):
    if is_dataclass(value) and not isinstance(value, type):
        return {f.name: to_data(getattr(value, f.name)) for f in fields(value)}
    if isinstance(value, Mapping):
        return {k: to_data(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return [to_data(v) for v in value]
    if value is None or type(value) in (str, bool, int, float):
        return value
    fail("opaque objects cannot be serialized")


def serialize_request(request: CompileRequest) -> bytes:
    if not isinstance(request, CompileRequest):
        fail("serialize_request requires a CompileRequest")
    return json.dumps(
        to_data(request),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def record(cls, data, converters=None):
    if not isinstance(data, dict):
        fail(f"{cls.__name__} must be an object")
    declared = {f.name: f for f in fields(cls)}
    if data.keys() - declared.keys():
        fail(f"unknown {cls.__name__} field")
    if declared.keys() - data.keys():
        fail(f"missing {cls.__name__} field")
    values = dict(data)
    for name, converter in (converters or {}).items():
        if name in values:
            values[name] = converter(values[name])
    return cls(**values)


def sequence(converter):
    def convert(values):
        if not isinstance(values, list):
            fail("expected an ordered JSON array")
        return tuple(converter(value) for value in values)

    return convert


def numerical(data):
    return record(
        NumericalContract, data, {"cast_sites": sequence(lambda x: record(CastSite, x))}
    )


def config_files(data):
    # The Python constructor can take a local snapshot for convenience. Wire
    # input must supply that snapshot explicitly and never consult local files.
    if not isinstance(data, dict):
        fail("config_files must be an explicit JSON object")
    return data


def operation(data):
    return record(
        Operation,
        data,
        {
            "results": sequence(lambda x: record(Value, x)),
            "regions": sequence(
                lambda x: record(
                    Region,
                    x,
                    {
                        "arguments": sequence(lambda y: record(Value, y)),
                        "operations": sequence(operation),
                    },
                )
            ),
            "span": lambda x: record(SourceSpan, x),
        },
    )


def deserialize_request(data: bytes | str) -> CompileRequest:
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                fail(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    def nonfinite(value):
        fail(f"non-finite JSON number: {value}")

    if not isinstance(data, (bytes, str)):
        fail("request JSON must be UTF-8 bytes or text")
    try:
        decoded = json.loads(data, object_pairs_hook=pairs, parse_constant=nonfinite)
    except (ValueError, UnicodeError, RecursionError) as error:
        fail(f"invalid request JSON: {error}")

    def program(value):
        return record(
            FrontendProgram,
            value,
            {
                "tensors": sequence(lambda x: record(TensorSpec, x)),
                "launch": lambda x: record(LaunchSpec, x),
                "operations": sequence(operation),
                "numerical_contract": numerical,
                "constexprs": json_constants,
                "captured_constants": json_constants,
            },
        )

    return record(
        CompileRequest,
        decoded,
        {
            "program": program,
            "options": lambda x: record(
                CompileOptions,
                x,
                {"numerical_contract": numerical, "config_files": config_files},
            ),
        },
    )


def json_constants(value):
    # JSON arrays represent immutable literal tuples, never mutable Python lists.
    if isinstance(value, list):
        return tuple(json_constants(x) for x in value)
    if isinstance(value, dict):
        return {k: json_constants(v) for k, v in value.items()}
    return value
