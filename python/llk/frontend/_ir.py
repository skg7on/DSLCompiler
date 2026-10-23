"""Immutable schema-v1 records. These describe programs, not executables."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from importlib.resources import files
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

from llk._errors import FrontendSyntaxError, TensorContractError

I64_MIN, I64_MAX = -(2**63), 2**63 - 1
STORAGE_DTYPES = ("f32", "bf16", "f16", "i32", "i8")
VALUE_DTYPES = (
    STORAGE_DTYPES + ("i1", "index") + tuple(f"ptr<{d}>" for d in STORAGE_DTYPES)
)
BYTE_WIDTH = {"f32": 4, "bf16": 2, "f16": 2, "i32": 4, "i8": 1}


def fail(
    message, span=None, *, code="invalid_frontend_schema", error=FrontendSyntaxError
):
    location = (
        {}
        if span is None
        else {"file": span.file, "line": span.line, "column": span.column}
    )
    raise error(message, code=code, **location)


def integer(value, name="integer", *, minimum=I64_MIN):
    if type(value) is not int or not minimum <= value <= I64_MAX:
        fail(f"{name} must be a signed 64-bit integer >= {minimum}")
    return value


def text(value, name="string"):
    if not isinstance(value, str) or not value or "\x00" in value:
        fail(f"{name} must be a nonempty string without NUL")
    return value


def identifier(value):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]*", value):
        fail("expected an identifier")
    return value


def shape_tuple(shape, *, rank=None):
    if not isinstance(shape, (tuple, list)):
        fail("shape must be an array")
    result = tuple(shape)
    if (rank is not None and len(result) != rank) or len(result) > 3:
        fail("invalid shape rank")
    count = 1
    for dim in result:
        integer(dim, "extent", minimum=1)
        if count > I64_MAX // dim:
            fail("shape product exceeds signed 64-bit range")
        count *= dim
    return result


def freeze(value):
    """Copy JSON-compatible immutable metadata; never stringify opaque objects."""
    if value is None or type(value) in (str, bool):
        return value
    if type(value) is int:
        return integer(value)
    if type(value) is float:
        if not math.isfinite(value):
            fail("non-finite constants cannot be serialized")
        return value
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            fail("metadata keys must be strings")
        return MappingProxyType({key: freeze(item) for key, item in value.items()})
    if isinstance(value, (tuple, list)):
        return tuple(freeze(item) for item in value)
    fail("metadata must contain only JSON values")


def constants(value):
    if not isinstance(value, Mapping):
        fail("constants must be a mapping")

    def literal(item):
        if type(item) is tuple:
            return tuple(literal(x) for x in item)
        if type(item) not in (type(None), bool, int, float, str):
            fail("only immutable literal constants are supported")
        return freeze(item)

    return MappingProxyType(
        {identifier(name): literal(item) for name, item in value.items()}
    )


def typed_tuple(values, cls, name):
    if not isinstance(values, (tuple, list)) or any(
        not isinstance(v, cls) for v in values
    ):
        fail(f"{name} must contain {cls.__name__} records")
    return tuple(values)


@dataclass(frozen=True, slots=True)
class SourceSpan:
    file: str
    line: int
    column: int

    def __post_init__(self):
        text(self.file, "source file")
        integer(self.line, "source line", minimum=1)
        integer(self.column, "source column", minimum=0)


UNKNOWN_SPAN = SourceSpan("<unknown>", 1, 0)


@dataclass(frozen=True, slots=True)
class TensorSpec:
    name: str
    shape: tuple[int, int]
    dtype: str
    strides: tuple[int, int]
    direction: str = "input"
    effect: str = "read"

    def __post_init__(self):
        identifier(self.name)
        shape = shape_tuple(self.shape, rank=2)
        if self.dtype not in STORAGE_DTYPES:
            fail("unsupported tensor element type", error=TensorContractError)
        if (
            not isinstance(self.strides, (tuple, list))
            or tuple(self.strides) != (shape[1], 1)
            or any(type(x) is not int for x in self.strides)
        ):
            fail(
                "tensor must have compact row-major element strides",
                error=TensorContractError,
            )
        if (self.direction, self.effect) not in (
            ("input", "read"),
            ("output", "write"),
        ):
            fail(
                "tensor effects must be read inputs or write-only outputs",
                error=TensorContractError,
            )
        if shape[0] > I64_MAX // shape[1] // BYTE_WIDTH[self.dtype]:
            fail("tensor allocation byte extent overflows", error=TensorContractError)
        object.__setattr__(self, "shape", shape)
        object.__setattr__(self, "strides", tuple(self.strides))


@dataclass(frozen=True, slots=True)
class LaunchSpec:
    grid: tuple[int, int, int]
    block: tuple[int, int, int]

    def __post_init__(self):
        object.__setattr__(self, "grid", shape_tuple(self.grid, rank=3))
        object.__setattr__(self, "block", shape_tuple(self.block, rank=3))
        for grid, block in zip(self.grid, self.block):
            if grid > I64_MAX // block:
                fail("launch coverage extent overflows", code="invalid_launch_coverage")


@dataclass(frozen=True, slots=True)
class Value:
    id: str
    dtype: str
    shape: tuple[int, ...] = ()

    def __post_init__(self):
        identifier(self.id)
        if self.dtype not in VALUE_DTYPES:
            fail("unsupported value dtype")
        dims = shape_tuple(self.shape)
        if len(dims) > 2:
            fail("schema v1 values have rank at most two")
        object.__setattr__(self, "shape", dims)


@dataclass(frozen=True, slots=True)
class CastSite:
    value_id: str
    source_dtype: str
    target_dtype: str
    rounding: str = "rtne"

    def __post_init__(self):
        identifier(self.value_id)
        if (
            self.source_dtype not in VALUE_DTYPES[:7]
            or self.target_dtype not in VALUE_DTYPES[:7]
            or self.rounding != "rtne"
        ):
            fail("unsupported explicit cast contract")


@dataclass(frozen=True, slots=True)
class NumericalContract:
    storage_dtype: str = "f32"
    compute_dtype: str = "f32"
    accumulator_dtype: str = "f32"
    cast_sites: tuple[CastSite, ...] = ()
    rounding: str = "rtne"
    activation_expression: str = "none"
    activation_approximation: str = "none"
    reduction_order: str = "source"
    reassociation: bool = False
    fma: bool = False
    subnormal: str = "preserve"

    def __post_init__(self):
        for dtype in (self.storage_dtype, self.compute_dtype, self.accumulator_dtype):
            if dtype not in STORAGE_DTYPES:
                fail("unsupported numerical dtype")
        if self.rounding != "rtne" or self.subnormal not in (
            "preserve",
            "flush-to-zero",
        ):
            fail("unsupported rounding/subnormal contract")
        if self.activation_expression not in (
            "none",
            "x/(1+exp(-x))",
        ) or self.activation_approximation not in ("none", "exact", "native"):
            fail("unsupported activation contract")
        if (
            self.reduction_order not in ("source", "unspecified")
            or type(self.reassociation) is not bool
            or type(self.fma) is not bool
        ):
            fail("invalid numerical permissions")
        object.__setattr__(
            self, "cast_sites", typed_tuple(self.cast_sites, CastSite, "cast sites")
        )


@dataclass(frozen=True, slots=True)
class Region:
    arguments: tuple[Value, ...]
    operations: tuple[Operation, ...]

    def __post_init__(self):
        object.__setattr__(
            self, "arguments", typed_tuple(self.arguments, Value, "region arguments")
        )
        object.__setattr__(
            self,
            "operations",
            typed_tuple(self.operations, Operation, "region operations"),
        )


@dataclass(frozen=True, slots=True)
class Operation:
    opcode: str
    operands: tuple[str, ...] = ()
    results: tuple[Value, ...] = ()
    attributes: Mapping = field(default_factory=dict)
    regions: tuple[Region, ...] = ()
    span: SourceSpan = UNKNOWN_SPAN

    def __post_init__(self):
        identifier(self.opcode)
        if not isinstance(self.operands, (list, tuple)):
            fail("operands must be an ordered array")
        object.__setattr__(
            self, "operands", tuple(identifier(x) for x in self.operands)
        )
        object.__setattr__(
            self, "results", typed_tuple(self.results, Value, "operation results")
        )
        if not isinstance(self.attributes, Mapping):
            fail("attributes must be a mapping")
        object.__setattr__(self, "attributes", freeze(self.attributes))
        object.__setattr__(
            self, "regions", typed_tuple(self.regions, Region, "regions")
        )
        if not isinstance(self.span, SourceSpan):
            fail("operation requires a source span")


@dataclass(frozen=True, slots=True)
class FrontendProgram:
    entry: str
    tensors: tuple[TensorSpec, ...]
    constexprs: Mapping
    launch: LaunchSpec
    operations: tuple[Operation, ...]
    numerical_contract: NumericalContract = field(default_factory=NumericalContract)
    version: int = 1
    source: str = ""
    captured_constants: Mapping = field(default_factory=dict)

    def __post_init__(self):
        if type(self.version) is not int or self.version != 1:
            fail("only frontend schema version 1 is supported")
        identifier(self.entry)
        object.__setattr__(
            self, "tensors", typed_tuple(self.tensors, TensorSpec, "tensors")
        )
        object.__setattr__(self, "constexprs", constants(self.constexprs))
        object.__setattr__(
            self, "captured_constants", constants(self.captured_constants)
        )
        object.__setattr__(
            self, "operations", typed_tuple(self.operations, Operation, "operations")
        )
        if (
            not isinstance(self.launch, LaunchSpec)
            or not isinstance(self.numerical_contract, NumericalContract)
            or not isinstance(self.source, str)
        ):
            fail("invalid program launch, numerical contract or source")
        from ._validation import validate_program

        validate_program(self)


def config_snapshot(root):
    root = Path(root).resolve()
    if not root.is_dir():
        fail("config_root must name an existing directory")
    snapshot = {}
    for pattern in ("*.yaml", "*.llkmap"):
        for path in sorted(root.rglob(pattern)):
            if not path.is_file():
                continue
            if not path.resolve().is_relative_to(root):
                fail("config files must stay within config_root")
            try:
                snapshot[path.relative_to(root).as_posix()] = path.read_text(
                    encoding="utf-8"
                )
            except (OSError, UnicodeError) as error:
                fail(f"cannot read config snapshot: {error}")
    return snapshot


@dataclass(frozen=True, slots=True)
class CompileOptions:
    target: str = "x86-avx2"
    backend: str = "reference"
    search_mode: str = "deterministic"
    max_candidates: int = 64
    top_k: int = 1
    numerical_contract: NumericalContract = field(default_factory=NumericalContract)
    config_root: str = field(default_factory=lambda: str(files("llk")))
    config_files: Mapping | None = None

    def __post_init__(self):
        text(self.target, "target")
        text(self.config_root, "config root")
        if (
            self.backend not in ("reference", "selected-target")
            or self.search_mode != "deterministic"
        ):
            fail("unsupported backend or search mode")
        integer(self.max_candidates, "max candidates", minimum=1)
        integer(self.top_k, "top k", minimum=1)
        if self.top_k > self.max_candidates or not isinstance(
            self.numerical_contract, NumericalContract
        ):
            fail("invalid selection bounds or numerical contract")
        data = (
            config_snapshot(self.config_root)
            if self.config_files is None
            else self.config_files
        )
        if not isinstance(data, Mapping) or any(
            not isinstance(k, str)
            or not k
            or Path(k).is_absolute()
            or ".." in Path(k).parts
            or not isinstance(v, str)
            for k, v in data.items()
        ):
            fail("config snapshot requires relative paths and text contents")
        object.__setattr__(self, "config_files", freeze(data))


@dataclass(frozen=True, slots=True)
class CompileRequest:
    program: FrontendProgram
    options: CompileOptions = field(default_factory=CompileOptions)

    def __post_init__(self):
        if not isinstance(self.program, FrontendProgram) or not isinstance(
            self.options, CompileOptions
        ):
            fail("compile request requires a program and compile options")
        if self.program.numerical_contract != self.options.numerical_contract:
            fail("compile options must preserve the program numerical contract")


@dataclass(frozen=True, slots=True)
class CompilationReport:
    native_request_identity: str
    frontend_identity: str
    compiler_identity: str
    workload_identity: str
    source_adapter_identity: str
    target_hash: str
    machine_hash: str
    rules_hash: str
    plan_id: str
    abi_hash: str
    selected_group_count: int
    reference_group_count: int
    stages: tuple[str, ...]

    def __post_init__(self):
        for name in self.__dataclass_fields__:
            value = getattr(self, name)
            if name.endswith("_count"):
                integer(value, name, minimum=0)
            elif name == "stages":
                if not isinstance(value, (tuple, list)):
                    fail("stages must be an ordered array")
                object.__setattr__(self, name, tuple(text(x, "stage") for x in value))
            else:
                text(value, name)
