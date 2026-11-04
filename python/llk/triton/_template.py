"""Neutral, unbound TTIR export template (schema version 1).

The template is the only value that crosses the upstream Triton process
boundary besides the genuine TTIR text and toolchain identity. It carries typed
parameters, the SSA dataflow graph, masks, counted-loop and cast regions, and
content hashes. It never carries a concrete runtime tensor shape or launch grid:
those are bound by the launcher at actual invocation time.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping

from llk._errors import FrontendSyntaxError

SCHEMA_VERSION = 1

ELEMENT_DTYPES = frozenset(
    {
        "i1",
        "i8",
        "i16",
        "i32",
        "i64",
        "index",
        "f16",
        "bf16",
        "f32",
        "f64",
    }
)
POINTER_DTYPE = re.compile(r"ptr<([a-z0-9]+)>")
_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z_0-9]*")
_HEX_DIGEST = re.compile(r"[0-9a-f]{64}")

NEUTRAL_OPCODES = frozenset(
    {
        "program_id",
        "constant",
        "range",
        "splat",
        "add_ptr",
        "load",
        "store",
        "return",
        "for",
        "yield",
        "cast",
        "cmpi",
        "cmpf",
        "select",
        "undef",
        "addi",
        "subi",
        "muli",
        "divsi",
        "divui",
        "remsi",
        "remui",
        "andi",
        "ori",
        "xori",
        "shli",
        "shrsi",
        "shrui",
        "addf",
        "subf",
        "mulf",
        "divf",
        "remf",
        "negf",
    }
)

_KINDS = frozenset({"pointer", "scalar", "constexpr"})
_ACCESS = frozenset({"read", "write", "readwrite", "none"})


def fail(message, *, code="invalid_frontend_schema", error=FrontendSyntaxError):
    raise error(message, code=code)


def _identifier(value, name):
    if not isinstance(value, str) or not _IDENTIFIER.fullmatch(value):
        fail(f"{name} must be an identifier")
    return value


def _digest(value, name):
    if not isinstance(value, str) or not _HEX_DIGEST.fullmatch(value):
        fail(f"{name} must be a lowercase SHA-256 digest")
    return value


def _dtype(value, name="dtype"):
    if value in ELEMENT_DTYPES:
        return value
    match = POINTER_DTYPE.fullmatch(value) if isinstance(value, str) else None
    if match is not None and match.group(1) in ELEMENT_DTYPES:
        return value
    fail(f"{name} must be an element or pointer dtype")
    return value


def _shape(value, name="shape"):
    if not isinstance(value, (tuple, list)):
        fail(f"{name} must be an ordered array")
    dims = tuple(value)
    for dim in dims:
        if type(dim) is not int or dim < 1:
            fail(f"{name} extents must be positive integers")
    return dims


def _literal(value, name="value"):
    if type(value) in (bool, int, float, str):
        return value
    if value is None:
        return None
    fail(f"{name} must be a JSON literal")
    return value


def freeze_attributes(value, name="attributes"):
    """Copy a JSON-compatible attribute mapping into an immutable mapping."""
    if not isinstance(value, Mapping):
        fail(f"{name} must be a mapping")
    frozen = {}
    for key, item in value.items():
        if not isinstance(key, str) or not key:
            fail(f"{name} keys must be nonempty strings")
        if isinstance(item, Mapping):
            frozen[key] = freeze_attributes(item, f"{name}.{key}")
        elif isinstance(item, (tuple, list)):
            items = tuple(_literal(x, f"{name}.{key}") for x in item)
            if any(x is None for x in items):
                fail(f"{name}.{key} cannot contain null")
            frozen[key] = items
        else:
            frozen[key] = _literal(item, f"{name}.{key}")
    return MappingProxyType(frozen)


@dataclass(frozen=True, slots=True)
class ToolchainIdentity:
    """Exact upstream identity observed by the isolated export worker."""

    triton_version: str
    llvm_revision: str
    source_revision: str
    protocol_version: int
    target: str
    python_tag: str
    platform_tag: str

    def __post_init__(self):
        for name in (
            "triton_version",
            "llvm_revision",
            "source_revision",
            "target",
            "python_tag",
            "platform_tag",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                fail(f"toolchain {name} must be a nonempty string")
        if type(self.protocol_version) is not int or self.protocol_version < 1:
            fail("toolchain protocol_version must be a positive integer")

    def to_json(self):
        return {
            "triton_version": self.triton_version,
            "llvm_revision": self.llvm_revision,
            "source_revision": self.source_revision,
            "protocol_version": self.protocol_version,
            "target": self.target,
            "python_tag": self.python_tag,
            "platform_tag": self.platform_tag,
        }

    @classmethod
    def from_json(cls, data):
        if not isinstance(data, Mapping):
            fail("toolchain identity must be a mapping")
        try:
            return cls(**{name: data[name] for name in cls.__dataclass_fields__})
        except KeyError as error:
            fail(f"toolchain identity is missing {error.args[0]}")


@dataclass(frozen=True, slots=True)
class TemplateParameter:
    """One upstream kernel argument, typed but without a runtime shape."""

    name: str
    kind: str
    dtype: str
    access: str = "none"
    alignment_bytes: int = 1
    value_id: str = ""
    value: object = None

    def __post_init__(self):
        _identifier(self.name, "parameter name")
        if self.kind not in _KINDS:
            fail("parameter kind must be pointer, scalar or constexpr")
        _dtype(self.dtype, f"parameter {self.name} dtype")
        if self.access not in _ACCESS:
            fail("parameter access must be read, write, readwrite or none")
        if type(self.alignment_bytes) is not int or self.alignment_bytes < 1:
            fail("parameter alignment must be a positive integer")
        if self.kind == "constexpr":
            if self.value_id:
                fail("constexpr parameters do not have an SSA value")
            if self.access != "none":
                fail("constexpr parameters are not accessed at runtime")
            _literal(self.value, f"constexpr {self.name}")
        else:
            _identifier(self.value_id, f"parameter {self.name} value_id")
            if self.value is not None:
                fail("runtime parameters cannot carry a compile-time value")

    def to_json(self):
        return {
            "name": self.name,
            "kind": self.kind,
            "dtype": self.dtype,
            "access": self.access,
            "alignment_bytes": self.alignment_bytes,
            "value_id": self.value_id,
            "value": self.value,
        }

    @classmethod
    def from_json(cls, data):
        if not isinstance(data, Mapping):
            fail("template parameter must be a mapping")
        try:
            return cls(**{name: data[name] for name in cls.__dataclass_fields__})
        except KeyError as error:
            fail(f"template parameter is missing {error.args[0]}")


@dataclass(frozen=True, slots=True)
class TemplateValue:
    """Typed SSA value. A shape is a compile-time tile shape, never a runtime shape."""

    id: str
    dtype: str
    shape: tuple[int, ...] = ()

    def __post_init__(self):
        _identifier(self.id, "value id")
        _dtype(self.dtype, f"value {self.id} dtype")
        object.__setattr__(self, "shape", _shape(self.shape, f"value {self.id} shape"))

    def to_json(self):
        return {"id": self.id, "dtype": self.dtype, "shape": list(self.shape)}

    @classmethod
    def from_json(cls, data):
        if not isinstance(data, Mapping):
            fail("template value must be a mapping")
        try:
            return cls(id=data["id"], dtype=data["dtype"], shape=data.get("shape", ()))
        except KeyError as error:
            fail(f"template value is missing {error.args[0]}")


@dataclass(frozen=True, slots=True)
class TemplateRegion:
    arguments: tuple[str, ...]
    operations: tuple[TemplateOperation, ...]

    def __post_init__(self):
        if not isinstance(self.arguments, (tuple, list)):
            fail("region arguments must be an ordered array")
        object.__setattr__(
            self, "arguments", tuple(_identifier(x, "region argument") for x in self.arguments)
        )
        if not isinstance(self.operations, (tuple, list)) or any(
            not isinstance(op, TemplateOperation) for op in self.operations
        ):
            fail("region operations must be TemplateOperation records")
        object.__setattr__(self, "operations", tuple(self.operations))

    def to_json(self):
        return {
            "arguments": list(self.arguments),
            "operations": [op.to_json() for op in self.operations],
        }

    @classmethod
    def from_json(cls, data):
        if not isinstance(data, Mapping):
            fail("template region must be a mapping")
        return cls(
            arguments=data.get("arguments", ()),
            operations=tuple(TemplateOperation.from_json(x) for x in data.get("operations", ())),
        )


@dataclass(frozen=True, slots=True)
class TemplateOperation:
    id: str
    opcode: str
    operands: tuple[str, ...] = ()
    results: tuple[str, ...] = ()
    attributes: Mapping = field(default_factory=dict)
    regions: tuple[TemplateRegion, ...] = ()

    def __post_init__(self):
        _identifier(self.id, "operation id")
        if self.opcode not in NEUTRAL_OPCODES:
            fail(f"unsupported neutral opcode {self.opcode!r}")
        for field_name in ("operands", "results"):
            values = getattr(self, field_name)
            if not isinstance(values, (tuple, list)):
                fail(f"operation {field_name} must be an ordered array")
            object.__setattr__(
                self,
                field_name,
                tuple(_identifier(x, f"operation {field_name}") for x in values),
            )
        object.__setattr__(self, "attributes", freeze_attributes(self.attributes))
        if not isinstance(self.regions, (tuple, list)) or any(
            not isinstance(region, TemplateRegion) for region in self.regions
        ):
            fail("operation regions must be TemplateRegion records")
        object.__setattr__(self, "regions", tuple(self.regions))

    def to_json(self):
        return {
            "id": self.id,
            "opcode": self.opcode,
            "operands": list(self.operands),
            "results": list(self.results),
            "attributes": _plain(self.attributes),
            "regions": [region.to_json() for region in self.regions],
        }

    @classmethod
    def from_json(cls, data):
        if not isinstance(data, Mapping):
            fail("template operation must be a mapping")
        try:
            return cls(
                id=data["id"],
                opcode=data["opcode"],
                operands=data.get("operands", ()),
                results=data.get("results", ()),
                attributes=data.get("attributes", {}),
                regions=tuple(TemplateRegion.from_json(x) for x in data.get("regions", ())),
            )
        except KeyError as error:
            fail(f"template operation is missing {error.args[0]}")


def _plain(value):
    if isinstance(value, Mapping):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_plain(item) for item in value]
    return value


def _collect_operations(operations):
    for operation in operations:
        yield operation
        for region in operation.regions:
            yield from _collect_operations(region.operations)


@dataclass(frozen=True, slots=True)
class UnboundFrontendTemplate:
    """Schema-v1 unbound template; the runtime grid and tensor shapes are deferred."""

    entry: str
    parameters: tuple[TemplateParameter, ...]
    values: tuple[TemplateValue, ...]
    operations: tuple[TemplateOperation, ...]
    source_hash: str
    ttir_hash: str
    toolchain: ToolchainIdentity
    schema_version: int = SCHEMA_VERSION
    required_runtime_grid: bool = True

    def __post_init__(self):
        if type(self.schema_version) is not int or self.schema_version != SCHEMA_VERSION:
            fail("only template schema version 1 is supported")
        if self.required_runtime_grid is not True:
            fail("template schema v1 requires a runtime launch grid")
        _identifier(self.entry, "template entry")
        if not isinstance(self.parameters, (tuple, list)) or any(
            not isinstance(parameter, TemplateParameter) for parameter in self.parameters
        ):
            fail("template parameters must be TemplateParameter records")
        object.__setattr__(self, "parameters", tuple(self.parameters))
        if not isinstance(self.values, (tuple, list)) or any(
            not isinstance(value, TemplateValue) for value in self.values
        ):
            fail("template values must be TemplateValue records")
        object.__setattr__(self, "values", tuple(self.values))
        if not isinstance(self.operations, (tuple, list)) or any(
            not isinstance(operation, TemplateOperation) for operation in self.operations
        ):
            fail("template operations must be TemplateOperation records")
        object.__setattr__(self, "operations", tuple(self.operations))
        _digest(self.source_hash, "source_hash")
        _digest(self.ttir_hash, "ttir_hash")
        if not isinstance(self.toolchain, ToolchainIdentity):
            fail("template requires a ToolchainIdentity")
        self._validate_graph()

    def _validate_graph(self):
        value_ids = set()
        for value in self.values:
            if value.id in value_ids:
                fail(f"duplicate template value {value.id}")
            value_ids.add(value.id)
        parameter_names = set()
        for parameter in self.parameters:
            if parameter.name in parameter_names:
                fail(f"duplicate template parameter {parameter.name}")
            parameter_names.add(parameter.name)
            if parameter.value_id:
                if parameter.value_id not in value_ids:
                    fail(f"parameter {parameter.name} names an unknown value")
        operation_ids = set()
        for operation in _collect_operations(self.operations):
            if operation.id in operation_ids:
                fail(f"duplicate template operation {operation.id}")
            operation_ids.add(operation.id)
            for value_id in operation.operands + operation.results:
                if value_id not in value_ids:
                    fail(f"operation {operation.id} names an unknown value {value_id}")
            for region in operation.regions:
                for argument in region.arguments:
                    if argument not in value_ids:
                        fail(f"operation {operation.id} region names an unknown value")
        constexpr_names = {p.name for p in self.parameters if p.kind == "constexpr"}
        if set(self.constexprs) != constexpr_names:
            fail("constexpr parameters and constexpr values must agree")

    @property
    def constexprs(self):
        return MappingProxyType(
            {
                parameter.name: parameter.value
                for parameter in self.parameters
                if parameter.kind == "constexpr"
            }
        )

    def to_json(self):
        return {
            "schema_version": self.schema_version,
            "required_runtime_grid": self.required_runtime_grid,
            "entry": self.entry,
            "parameters": [parameter.to_json() for parameter in self.parameters],
            "values": [value.to_json() for value in self.values],
            "operations": [operation.to_json() for operation in self.operations],
            "source_hash": self.source_hash,
            "ttir_hash": self.ttir_hash,
            "toolchain": self.toolchain.to_json(),
        }

    def identity(self) -> str:
        """Deterministic content identity for the first cache stage."""
        encoded = json.dumps(
            self.to_json(), sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    @classmethod
    def from_json(cls, data):
        if not isinstance(data, Mapping):
            fail("template must be a mapping")
        try:
            return cls(
                entry=data["entry"],
                parameters=tuple(
                    TemplateParameter.from_json(item) for item in data["parameters"]
                ),
                values=tuple(TemplateValue.from_json(item) for item in data["values"]),
                operations=tuple(
                    TemplateOperation.from_json(item) for item in data["operations"]
                ),
                source_hash=data["source_hash"],
                ttir_hash=data["ttir_hash"],
                toolchain=ToolchainIdentity.from_json(data["toolchain"]),
                schema_version=data.get("schema_version", SCHEMA_VERSION),
                required_runtime_grid=data.get("required_runtime_grid", True),
            )
        except KeyError as error:
            fail(f"template is missing {error.args[0]}")
