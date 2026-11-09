"""Upstream Triton driver and CPU launcher bound to the LLK frontend service.

The launcher never invents a machine-code artifact.  The compiler's final bytes
are the :class:`~llk.triton._backend.UnboundLaunchArtifact`; at launch time the
owner exports the genuine upstream specialization through PF17, binds concrete
tensor shapes/strides/dtypes, scalar values and the runtime grid into a checked
``FrontendProgram``, and compiles/invokes through the shared PF08/PF09 service.

The upstream ``BaseBackend`` and ``DriverBase`` subclasses are real: activation
installs ``LLKCPUDriver`` as the process-global Triton driver for the duration of
one scoped context and restores the previous driver on success or error.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping

from llk._errors import (
    CompilationError,
    LLKError,
    TensorContractError,
    UnsupportedFeatureError,
)
from llk.frontend import (
    CompileOptions,
    CompileRequest,
    FrontendProgram,
    LaunchSpec,
    NumericalContract,
    Operation,
    SourceSpan,
    TensorSpec,
    Value,
)
from llk.frontend._validation import broadcast_shape
from llk.runtime import _buffers
from llk.triton._backend import (
    ARTIFACT_SCHEMA_VERSION,
    BACKEND_NAME,
    BACKEND_VERSION,
    CPUUtils,
    LLKCPUBackend,
    LoadedArtifact,
)
from llk.triton._template import UnboundFrontendTemplate
from llk.triton._toolchain import ensure_triton_importable

# Upstream imports stay limited to the base-class modules.  ``triton.runtime``
# is imported lazily inside activation so package entry-point discovery cannot
# observe a partially initialized ``triton.backends`` module.
ensure_triton_importable()
from triton.backends.compiler import GPUTarget
from triton.backends.driver import DriverBase

MARKER = "LLK-TRITON-COEXIST-V1"
MAX_PROBE_OUTPUT = 1 << 20
PROBE_TIMEOUT = 300.0
TEMPLATE_CACHE_ENTRIES = 32

_ACTIVATION_LOCK = threading.RLock()
_ACTIVATIONS: list["Activation"] = []
_LAST_REPORT: "TritonLaunchReport | None" = None
_COEXISTENCE: dict[tuple, "CoexistenceStatus"] = {}
_QUALIFIED: dict[str, Any] = {}
_TEMPLATES = OrderedDict()
_COMPILATION_CACHE = None

_SPAN = SourceSpan("triton", 1, 0)
_INDEX_DTYPES = frozenset({"index", "i32", "i64", "i16", "i8"})
_INT_RANGES = {
    "index": (-(2**63), 2**63 - 1),
    "i32": (-(2**31), 2**31 - 1),
    "i64": (-(2**63), 2**63 - 1),
    "i16": (-(2**15), 2**15 - 1),
    "i8": (-(2**7), 2**7 - 1),
}


@dataclass(frozen=True, slots=True)
class CoexistenceStatus:
    """Whether upstream Triton and the LLK C API can share one process."""

    qualified: bool
    reasons: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class TritonLaunchReport:
    """Native evidence recorded at one actual launch."""

    entry: str
    template_identity: str
    artifact_identity: str
    source_hash: str
    ttir_hash: str
    native_request_identity: str
    frontend_identity: str
    compiler_identity: str
    plan_id: str
    abi_hash: str
    backend: str
    target_hash: str
    triton_version: str
    runtime_grid: tuple[int, int, int]
    native_grid: tuple[int, int, int]
    block: tuple[int, int, int]
    tensors: tuple[dict, ...]
    scalars: tuple[dict, ...]
    constexprs: Mapping
    native_invocations: int

    def to_json(self):
        return {
            "entry": self.entry,
            "template_identity": self.template_identity,
            "artifact_identity": self.artifact_identity,
            "source_hash": self.source_hash,
            "ttir_hash": self.ttir_hash,
            "native_request_identity": self.native_request_identity,
            "frontend_identity": self.frontend_identity,
            "compiler_identity": self.compiler_identity,
            "plan_id": self.plan_id,
            "abi_hash": self.abi_hash,
            "backend": self.backend,
            "target_hash": self.target_hash,
            "triton_version": self.triton_version,
            "runtime_grid": list(self.runtime_grid),
            "native_grid": list(self.native_grid),
            "block": list(self.block),
            "tensors": [dict(item) for item in self.tensors],
            "scalars": [dict(item) for item in self.scalars],
            "constexprs": dict(self.constexprs),
            "native_invocations": self.native_invocations,
        }


@dataclass(frozen=True, slots=True)
class _PreparedLaunch:
    request: CompileRequest
    inputs: tuple
    outputs: tuple
    runtime_grid: tuple[int, int, int]
    native_grid: tuple[int, int, int]
    block: tuple[int, int, int]
    tensors: tuple[dict, ...]
    scalars: tuple[dict, ...]


# ---------------------------------------------------------------------------
# Affine index model for the accepted flat elementwise kernel.
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _Expr:
    """``pid_coefficient * flat_triton_pid + lane * flat_lane + constant``."""

    pid: int = 0
    lane: int = 0
    constant: int = 0

    def uniform(self):
        return self.pid == 0 and self.lane == 0


def _expr_add(left: _Expr, right: _Expr) -> _Expr:
    return _Expr(
        left.pid + right.pid,
        left.lane + right.lane,
        left.constant + right.constant,
    )


def _expr_scale(expr: _Expr, factor: int) -> _Expr:
    return _Expr(expr.pid * factor, expr.lane * factor, expr.constant * factor)


@dataclass(slots=True)
class _Bound:
    kind: str
    expr: _Expr | None = None
    value: Value | None = None
    port: str | None = None
    tile: bool = False


class _Binder:
    """Translate the neutral PF17 template into a checked 2-D frontend program."""

    def __init__(self, template, layouts, scalars, block, mode, runtime_grid,
                 source):
        self.template = template
        self.layouts = layouts
        self.scalars = scalars
        self.block = block
        self.mode = mode
        self.runtime_grid = runtime_grid
        self.source = source
        self.columns = next(iter(layouts.values())).shape[1]
        self.ops: list[Operation] = []
        self.values: dict[str, _Bound] = {}
        self._counter = 0
        self._constants: dict[tuple, Value] = {}
        self._pids: dict[int, Value] = {}
        self._lanes: dict[str, Value] = {}
        self._loads = 0
        self._stores = 0
        self._returned = False
        self.specs = {value.id: value for value in template.values}
        for parameter in template.parameters:
            if parameter.kind == "pointer":
                self.values[parameter.value_id] = _Bound(
                    "ptr", port=parameter.name
                )
            elif parameter.kind == "scalar":
                self.values[parameter.value_id] = _Bound(
                    "index", _Expr(0, 0, scalars[parameter.name])
                )

    # -- emission ---------------------------------------------------------

    def _fresh(self, dtype, shape):
        self._counter += 1
        return Value(f"_t{self._counter}", dtype, tuple(shape))

    def _emit(self, opcode, operands=(), results=(), attributes=None):
        operation = Operation(
            opcode,
            tuple(value.id for value in operands),
            tuple(results),
            attributes or {},
            (),
            _SPAN,
        )
        self.ops.append(operation)

    def _constant(self, value, dtype):
        key = (dtype, value)
        if key not in self._constants:
            result = self._fresh(dtype, ())
            self._emit("constant", (), (result,), {"value": value})
            self._constants[key] = result
        return self._constants[key]

    def _pid(self, axis):
        if axis not in self._pids:
            result = self._fresh("index", ())
            self._emit("program_id", (), (result,), {"axis": axis})
            self._pids[axis] = result
        return self._pids[axis]

    def _lane(self, name, extent):
        if name not in self._lanes:
            flat = self._fresh("index", (extent,))
            self._emit("range", (), (flat,), {"start": 0, "end": extent})
            axis = 1 if name == "rows" else 0
            expanded = self._fresh("index", (extent, 1) if axis == 1 else (1, extent))
            self._emit("expand_dims", (flat,), (expanded,), {"axis": axis})
            self._lanes[name] = expanded
        return self._lanes[name]

    def _binary(self, opcode, left, right):
        shape = broadcast_shape(left.shape, right.shape)
        result = self._fresh("index", shape)
        self._emit(opcode, (left, right), (result,))
        return result

    def _linear(self, terms):
        """Combine ``(value, coefficient)`` terms into one index value."""
        result = self._constant(0, "index")
        for value, coefficient in terms:
            if coefficient == 0:
                continue
            term = value
            if coefficient != 1:
                term = self._binary("mul", value, self._constant(coefficient, "index"))
            result = self._binary("add", result, term)
        return result

    # -- index materialization -------------------------------------------

    def _pid_terms(self, coefficient):
        """Express an upstream flat-program-id coefficient as LLK program ids."""
        if coefficient == 0:
            return []
        if self.mode == "rows":
            return [(0, coefficient)]
        quotient = self.columns // self.block[1]
        return [(0, coefficient * quotient), (1, coefficient)]

    def _scalar_value(self, expr):
        if expr.lane != 0:
            raise UnsupportedFeatureError(
                "a lane-dependent index cannot be used as a scalar",
                code="unsupported_triton_launch",
            )
        return self._linear(
            [(self._pid(axis), coefficient) for axis, coefficient in self._pid_terms(expr.pid)]
            + [(self._constant(expr.constant, "index"), 1)]
        )

    def _lane_terms(self):
        if self.mode == "rows":
            return self._lane("rows", self.block[0]), self._lane("columns",
                                                                 self.block[1])
        return None, self._lane("columns", self.block[1])

    def _row_col(self, expr):
        if expr.lane != 1:
            raise UnsupportedFeatureError(
                "memory addressing requires a full flat lane",
                code="unsupported_triton_launch",
            )
        row_lane, col_lane = self._lane_terms()
        terms = self._pid_terms(expr.pid)
        rows = [
            (self._pid(axis), coefficient // self.columns)
            for axis, coefficient in terms
        ]
        columns = [
            (self._pid(axis), coefficient % self.columns)
            for axis, coefficient in terms
        ]
        rows.append((self._constant(expr.constant // self.columns, "index"), 1))
        columns.append(
            (self._constant(expr.constant % self.columns, "index"), 1)
        )
        if row_lane is not None:
            rows.append((row_lane, 1))
        columns.append((col_lane, 1))
        row = self._linear(rows)
        column = self._linear(columns)
        return row, column

    def _flat_value(self, expr):
        if expr.lane == 0:
            return self._scalar_value(expr)
        row, column = self._row_col(expr)
        scaled = self._binary(
            "mul", row, self._constant(self.columns, "index")
        )
        return self._binary("add", scaled, column)

    def _address(self, port, expr):
        row, column = self._row_col(expr)
        dtype = f"ptr<{self.layouts[port].dtype}>"
        shape = broadcast_shape(row.shape, column.shape)
        pointer = self._fresh(dtype, shape)
        base = Value(port, dtype, ())
        self._emit("address", (base, row, column), (pointer,))
        return pointer

    def _true_mask(self, shape):
        result = self._fresh("i1", shape)
        self._emit("constant", (), (result,), {"value": True})
        return result

    # -- template replay --------------------------------------------------

    def _lookup(self, value_id):
        bound = self.values.get(value_id)
        if bound is None:
            raise UnsupportedFeatureError(
                f"template value {value_id!r} has no definition",
                code="unsupported_triton_launch",
            )
        return bound

    def _index(self, bound):
        if bound.kind != "index" or bound.expr is None:
            raise UnsupportedFeatureError(
                "the accepted lower requires a scalar/integer index operand",
                code="unsupported_triton_launch",
            )
        return bound.expr

    def run(self):
        for operation in self.template.operations:
            self._operation(operation)
        if not self._returned:
            raise UnsupportedFeatureError(
                "the exported template has no return terminator",
                code="unsupported_triton_launch",
            )
        if self._stores != 1 or self._loads != 2:
            raise UnsupportedFeatureError(
                "the accepted launch corpus has exactly two loads and one store",
                code="unsupported_triton_launch",
            )
        self._emit("return")
        return tuple(self.ops)

    def _operation(self, operation):
        code = operation.opcode
        if self._returned:
            raise UnsupportedFeatureError(
                "operations after return are unsupported",
                code="unsupported_triton_launch",
            )
        operands = [self._lookup(name) for name in operation.operands]
        results = operation.results
        attributes = operation.attributes
        if code == "constant":
            self._constant_operation(results[0], attributes)
        elif code == "program_id":
            if attributes.get("axis") != 0:
                raise UnsupportedFeatureError(
                    "only the flat Triton program id is supported",
                    code="unsupported_triton_launch",
                )
            self.values[results[0]] = _Bound("index", _Expr(1, 0, 0))
        elif code == "range":
            start = attributes.get("start")
            end = attributes.get("end")
            if type(start) is not int or type(end) is not int or start != 0 or end <= 0:
                raise UnsupportedFeatureError(
                    "only a zero-based static Triton range is supported",
                    code="unsupported_triton_launch",
                )
            self.values[results[0]] = _Bound(
                "index", _Expr(0, 1, start), tile=True
            )
        elif code == "splat":
            self._splat(operands[0], results[0])
        elif code == "addi" or code == "subi" or code == "muli":
            self._integer_arithmetic(code, operands, results[0])
        elif code == "cmpi":
            self._compare(operands, results[0], attributes)
        elif code == "addf" or code == "mulf":
            self._float_arithmetic(code, operands, results[0])
        elif code == "add_ptr":
            self._add_pointer(operands, results[0])
        elif code == "load":
            self._load(operation, operands, results[0])
        elif code == "store":
            self._store(operands)
        elif code == "return":
            self._returned = True
        else:
            raise UnsupportedFeatureError(
                f"the LLK Triton launcher does not accept {code!r}",
                code="unsupported_triton_launch",
            )

    def _constant_operation(self, result, attributes):
        value = attributes.get("value")
        spec = self.specs[result]
        if spec.dtype in _INDEX_DTYPES and type(value) is int:
            self.values[result] = _Bound("index", _Expr(0, 0, value))
            return
        if spec.dtype == "i1" and type(value) is bool:
            mask = self._fresh("i1", spec.shape)
            self._emit("constant", (), (mask,), {"value": value})
            self.values[result] = _Bound("mask", value=mask)
            return
        if spec.dtype in ("f32", "bf16", "f16") and type(value) in (int, float):
            self.values[result] = _Bound(
                "float", value=self._constant(float(value), spec.dtype)
            )
            return
        raise UnsupportedFeatureError(
            "only integer, boolean and floating constants are supported",
            code="unsupported_triton_launch",
        )

    def _splat(self, operand, result):
        if operand.kind == "index" and operand.expr is not None:
            self.values[result] = _Bound("index", operand.expr, tile=True)
            return
        if operand.kind == "ptr":
            self.values[result] = _Bound("ptr", port=operand.port)
            return
        if operand.kind == "mask":
            self.values[result] = _Bound("mask", value=operand.value)
            return
        raise UnsupportedFeatureError(
            "only index, pointer and mask splats are supported",
            code="unsupported_triton_launch",
        )

    def _integer_arithmetic(self, code, operands, result):
        left, right = self._index(operands[0]), self._index(operands[1])
        tile = operands[0].tile or operands[1].tile
        if code == "muli":
            if left.uniform():
                expr = _expr_scale(right, left.constant)
            elif right.uniform():
                expr = _expr_scale(left, right.constant)
            else:
                raise UnsupportedFeatureError(
                    "non-constant index multiplication is unsupported",
                    code="unsupported_triton_launch",
                )
        elif code == "addi":
            expr = _expr_add(left, right)
        else:
            expr = _expr_add(left, _expr_scale(right, -1))
        if expr.lane not in (0, 1):
            raise UnsupportedFeatureError(
                "the accepted lower only tracks a single flat lane",
                code="unsupported_triton_launch",
            )
        self.values[result] = _Bound("index", expr, tile=tile)

    def _compare(self, operands, result, attributes):
        predicate = {
            "eq": "eq",
            "ne": "ne",
            "slt": "lt",
            "ult": "lt",
            "sle": "le",
            "ule": "le",
            "sgt": "gt",
            "ugt": "gt",
            "sge": "ge",
            "uge": "ge",
        }.get(attributes.get("predicate"))
        if predicate is None:
            raise UnsupportedFeatureError(
                "unsupported integer comparison predicate",
                code="unsupported_triton_launch",
            )
        left = self._flat_value(self._index(operands[0]))
        right = self._flat_value(self._index(operands[1]))
        shape = broadcast_shape(left.shape, right.shape)
        mask = self._fresh("i1", shape)
        self._emit("compare", (left, right), (mask,), {"predicate": predicate})
        self.values[result] = _Bound("mask", value=mask)

    def _float_arithmetic(self, code, operands, result):
        if operands[0].kind != "float" or operands[1].kind != "float":
            raise UnsupportedFeatureError(
                "floating arithmetic requires two loaded value tiles",
                code="unsupported_triton_launch",
            )
        left, right = operands[0].value, operands[1].value
        shape = broadcast_shape(left.shape, right.shape)
        output = self._fresh("f32", shape)
        self._emit("mul" if code == "mulf" else "add", (left, right), (output,))
        self.values[result] = _Bound("float", value=output)

    def _add_pointer(self, operands, result):
        pointer, offset = operands
        if pointer.kind != "ptr" or pointer.port is None or pointer.expr is not None:
            raise UnsupportedFeatureError(
                "only a kernel parameter pointer plus an index is supported",
                code="unsupported_triton_launch",
            )
        self.values[result] = _Bound(
            "ptr", port=pointer.port, expr=self._index(offset), tile=True
        )

    def _load(self, operation, operands, result):
        pointer = operands[0]
        if pointer.kind != "ptr" or pointer.expr is None or pointer.port is None:
            raise UnsupportedFeatureError(
                "a load requires a bound kernel parameter address",
                code="unsupported_triton_launch",
            )
        if len(operands) > 2:
            raise UnsupportedFeatureError(
                "padded Triton loads are not part of the accepted launch contract",
                code="unsupported_triton_launch",
            )
        address = self._address(pointer.port, pointer.expr)
        if len(operands) == 2:
            if operands[1].kind != "mask" or operands[1].value is None:
                raise UnsupportedFeatureError(
                    "a load mask must be a boolean tile",
                    code="unsupported_triton_launch",
                )
            mask = operands[1].value
        else:
            mask = self._true_mask(address.shape)
        dtype = self.layouts[pointer.port].dtype
        value = self._fresh(dtype, address.shape)
        other = self._constant(0.0, dtype)
        self._emit("load", (address, mask, other), (value,))
        self._loads += 1
        self.values[result] = _Bound("float", value=value)

    def _store(self, operands):
        pointer, written = operands[0], operands[1]
        if pointer.kind != "ptr" or pointer.expr is None or pointer.port is None:
            raise UnsupportedFeatureError(
                "a store requires a bound kernel parameter address",
                code="unsupported_triton_launch",
            )
        if written.kind != "float" or written.value is None:
            raise UnsupportedFeatureError(
                "a store requires a loaded floating value tile",
                code="unsupported_triton_launch",
            )
        address = self._address(pointer.port, pointer.expr)
        if len(operands) == 3:
            if operands[2].kind != "mask" or operands[2].value is None:
                raise UnsupportedFeatureError(
                    "a store mask must be a boolean tile",
                    code="unsupported_triton_launch",
                )
            mask = operands[2].value
        else:
            mask = self._true_mask(address.shape)
        self._emit("store", (address, written.value, mask))
        self._stores += 1


# ---------------------------------------------------------------------------
# Binding: template + concrete arguments -> checked CompileRequest.
# ---------------------------------------------------------------------------


def _fail(message, *, code="invalid_frontend_buffer"):
    raise TensorContractError(message, code=code, stage="launch")


def _unsupported(message):
    raise UnsupportedFeatureError(message, code="unsupported_triton_launch")


def _provider_layout(parameter, argument):
    device = getattr(argument, "device", None)
    if device is not None and getattr(device, "type", "cpu") != "cpu":
        _fail(
            f"LLK CPU execution accepts CPU tensors only; got device {device}"
        )
    layout = _buffers.layout_of(argument)
    if layout.dtype != parameter.dtype:
        _fail(
            f"tensor {parameter.name} has element type {layout.dtype}; the "
            f"qualified artifact binds {parameter.dtype}"
        )
    if layout.address % parameter.alignment_bytes:
        _fail(
            f"tensor {parameter.name} at {layout.address:#x} violates the "
            f"artifact's {parameter.alignment_bytes}-byte alignment constraint"
        )
    if parameter.access == "write" and layout.read_only:
        _fail(f"output tensor {parameter.name} is read-only")
    return layout


def _scalar_value(parameter, argument):
    value = argument
    if getattr(type(value), "__name__", "") == "constexpr":
        value = getattr(value, "value", value)
    if type(value) is not int:
        _fail(f"scalar {parameter.name} must be an integer")
    bounds = _INT_RANGES.get(parameter.dtype)
    if bounds is not None and not bounds[0] <= value <= bounds[1]:
        _fail(
            f"scalar {parameter.name}={value} does not fit its {parameter.dtype} "
            "artifact type"
        )
    return value


def _constexpr_value(parameter, argument):
    value = argument
    if getattr(type(value), "__name__", "") == "constexpr":
        value = getattr(value, "value", value)
    if value != parameter.value:
        _fail(
            f"constexpr {parameter.name}={value!r} does not match the compiled "
            f"artifact's {parameter.value!r}",
            code="invalid_frontend_argument",
        )
    return value


def _axis_grid(grid):
    if not isinstance(grid, (tuple, list)) or not 1 <= len(grid) <= 3:
        _fail("grid must have one to three integer axes", code="invalid_launch_coverage")
    axes = tuple(int(axis) for axis in grid) + (1,) * (3 - len(grid))
    if any(type(axis) is not int or axis < 1 for axis in axes):
        _fail("grid axes must be positive integers", code="invalid_launch_coverage")
    return axes


def _tile_geometry(extent, shape, grid):
    rows, columns = shape
    if grid[1] != 1 or grid[2] != 1:
        _fail(
            "the flat Triton program id maps to a one-dimensional runtime grid",
            code="invalid_launch_coverage",
        )
    if extent % columns == 0:
        block_rows = extent // columns
        if rows % block_rows:
            _fail(
                f"block {extent} does not tile the declared {rows}x{columns} "
                "domain with whole rows",
                code="invalid_launch_coverage",
            )
        expected = rows // block_rows
        if grid[0] != expected:
            _fail(
                f"runtime grid {grid} does not cover a {rows}x{columns} domain "
                f"with {block_rows}x{columns} blocks",
                code="invalid_launch_coverage",
            )
        return "rows", (block_rows, columns, 1), (expected, 1, 1)
    if columns % extent == 0:
        block_columns = columns // extent
        expected = rows * block_columns
        if grid[0] != expected:
            _fail(
                f"runtime grid {grid} does not cover a {rows}x{columns} domain "
                f"with 1x{extent} blocks",
                code="invalid_launch_coverage",
            )
        return "cols", (1, extent, 1), (rows, block_columns, 1)
    _unsupported(
        f"block extent {extent} does not tile the {rows}x{columns} row-major "
        "storage into whole rows or row segments"
    )


def _find_range(template):
    ranges = []
    programs = []
    for operation in template.operations:
        if operation.opcode == "range":
            ranges.append(operation)
        elif operation.opcode == "program_id":
            programs.append(operation)
    if len(ranges) != 1 or len(programs) != 1:
        _unsupported(
            "the accepted launch corpus has exactly one static range and one "
            "flat program id"
        )
    if programs[0].attributes.get("axis") != 0:
        _unsupported("only the flat Triton program id is supported")
    result_id = ranges[0].results[0]
    spec = next(
        (value for value in template.values if value.id == result_id), None
    )
    if spec is None or len(spec.shape) != 1 or spec.shape[0] < 1:
        _unsupported("the static range must produce one positive lane extent")
    return spec.shape[0]


def _bind(template, args, grid, options, source):
    if not isinstance(template, UnboundFrontendTemplate):
        raise LLKError("bind_triton_template requires an exported template")
    if not isinstance(options, CompileOptions):
        raise LLKError("bind_triton_template requires CompileOptions")
    if options.numerical_contract != NumericalContract():
        raise UnsupportedFeatureError(
            "the accepted Triton launch corpus is the strict f32 contract; "
            "relaxed numerical policies are explicit IR semantics, not "
            "launcher flags",
            code="unsupported_triton_launch",
        )
    parameters = template.parameters
    if len(args) != len(parameters):
        _fail(
            f"launch passes {len(args)} arguments for {len(parameters)} "
            "artifact parameters",
            code="invalid_frontend_argument",
        )
    runtime_grid = _axis_grid(grid)
    layouts = {}
    scalars = {}
    for parameter, argument in zip(parameters, args):
        if parameter.kind == "constexpr":
            _constexpr_value(parameter, argument)
        elif parameter.kind == "scalar":
            scalars[parameter.name] = _scalar_value(parameter, argument)
        else:
            layouts[parameter.name] = _provider_layout(parameter, argument)
    if not layouts or not scalars:
        _unsupported(
            "the accepted launch corpus binds at least one tensor and one "
            "scalar"
        )
    shapes = {layout.shape for layout in layouts.values()}
    if len(shapes) != 1:
        _unsupported("every bound tensor must have the same rank-two shape")
    shape = shapes.pop()
    if any(layout.dtype != "f32" for layout in layouts.values()):
        _unsupported("the accepted launch corpus is strict f32")
    extent = _find_range(template)
    mode, block, native_grid = _tile_geometry(extent, shape, runtime_grid)

    binder = _Binder(template, layouts, scalars, block, mode, runtime_grid, source)
    operations = binder.run()

    tensors = []
    inputs, outputs = [], []
    tensor_reports = []
    for parameter in parameters:
        if parameter.kind != "pointer":
            continue
        layout = layouts[parameter.name]
        if parameter.access == "read":
            direction, effect = "input", "read"
            inputs.append(layout.owner)
        elif parameter.access == "write":
            direction, effect = "output", "write"
            outputs.append(layout.owner)
        else:
            _unsupported(
                f"pointer {parameter.name} must be a read input or a write output"
            )
        tensors.append(
            TensorSpec(
                parameter.name,
                layout.shape,
                layout.dtype,
                (layout.shape[1], 1),
                direction,
                effect,
            )
        )
        tensor_reports.append(
            {
                "name": parameter.name,
                "shape": list(layout.shape),
                "strides": list(layout.strides),
                "dtype": layout.dtype,
                "direction": direction,
                "effect": effect,
                "alignment_bytes": parameter.alignment_bytes,
            }
        )
    if len(inputs) != 2 or len(outputs) != 1:
        _unsupported("the accepted elementwise importer needs two inputs and one output")
    scalar_reports = tuple(
        {
            "name": name,
            "dtype": next(
                parameter.dtype
                for parameter in parameters
                if parameter.name == name
            ),
            "value": value,
        }
        for name, value in scalars.items()
    )
    program = FrontendProgram(
        template.entry,
        tuple(tensors),
        dict(template.constexprs),
        LaunchSpec(native_grid, block),
        operations,
        source=source,
        captured_constants={
            "triton_template_identity": template.identity(),
            "triton_source_hash": template.source_hash,
            "triton_ttir_hash": template.ttir_hash,
            "triton_entry": template.entry,
            "triton_runtime_grid": runtime_grid,
            "triton_backend_version": BACKEND_VERSION,
        },
    )
    request = CompileRequest(
        program, replace(options, numerical_contract=program.numerical_contract)
    )
    return _PreparedLaunch(
        request,
        tuple(inputs),
        tuple(outputs),
        runtime_grid,
        native_grid,
        block,
        tuple(tensor_reports),
        scalar_reports,
    )


def _compilation_cache():
    global _COMPILATION_CACHE
    with _ACTIVATION_LOCK:
        if _COMPILATION_CACHE is None:
            from llk.runtime._cache import CompilationCache

            _COMPILATION_CACHE = CompilationCache()
        return _COMPILATION_CACHE


def _compile_request(request):
    from llk.runtime import _ffi

    return _ffi.compile_request(request)


# ---------------------------------------------------------------------------
# Upstream launcher.
# ---------------------------------------------------------------------------


def _export_options():
    from llk.triton import toolchain_lock

    lock = toolchain_lock()
    target = lock.get("target")
    if not isinstance(target, str) or ":" not in target:
        raise UnsupportedFeatureError(
            "the toolchain lock does not record a qualified export target",
            code="upstream_toolchain_unqualified",
        )
    name, arch = target.split(":", 1)
    try:
        arch = int(arch)
    except ValueError as error:
        raise UnsupportedFeatureError(
            f"the toolchain lock target {target!r} has no integer architecture",
            code="upstream_toolchain_unqualified",
        ) from error
    return {"target": name, "arch": arch}


def _unwrap_constexpr(value):
    if getattr(type(value), "__name__", "") == "constexpr":
        return getattr(value, "value", value)
    return value


def _source_facts(src):
    from triton.compiler.compiler import ASTSource

    if not isinstance(src, ASTSource):
        _unsupported("the LLK CPU backend compiles upstream AST sources only")
    kernel = getattr(getattr(src, "fn", None), "fn", None)
    code = getattr(kernel, "__code__", None)
    if code is None:
        _unsupported("the upstream source does not expose its Python function")
    signature = getattr(src, "signature", None)
    if not isinstance(signature, Mapping) or not signature:
        _unsupported("the upstream source has no typed signature")
    ordered = {}
    for name, spec in signature.items():
        if not isinstance(name, str) or not isinstance(spec, str):
            _unsupported("only flat, string-typed upstream signatures are supported")
        ordered[name] = spec
    names = list(getattr(src.fn, "arg_names", ())) or list(ordered)
    constants = {}
    for path, value in dict(getattr(src, "constants", {}) or {}).items():
        if isinstance(path, str):
            path = (names.index(path),)
        if not isinstance(path, tuple) or len(path) != 1:
            _unsupported("only flat constexpr paths are supported")
        constants[names[path[0]]] = _unwrap_constexpr(value)
    specialization = {}
    for path, attrs in dict(getattr(src, "attrs", {}) or {}).items():
        if isinstance(path, str):
            path = (names.index(path),)
        if not isinstance(path, tuple) or len(path) != 1:
            _unsupported("only flat specialization paths are supported")
        name = names[path[0]]
        constraints = {}
        for item in attrs:
            key, value = item
            if key != "tt.divisibility":
                _unsupported(f"unsupported upstream attribute {key!r}")
            constraints["divisibility"] = int(value)
        specialization[name] = constraints
    return kernel, ordered, constants, specialization


_SCALAR_SPEC = {
    "f32": "fp32",
    "f16": "fp16",
    "bf16": "bf16",
    "i1": "i1",
    "i8": "i8",
    "i16": "i16",
    "i32": "i32",
    "i64": "i64",
    "index": "index",
}


def _validate_template(template, source_hash, signature, constants, specialization):
    if template.source_hash != source_hash:
        raise CompilationError(
            "the template was exported from different source bytes",
            code="triton_artifact_mismatch",
            stage="launch",
        )
    runtime = [p.name for p in template.parameters if p.kind != "constexpr"]
    expected_runtime = [name for name, spec in signature.items() if spec != "constexpr"]
    if runtime != expected_runtime:
        raise CompilationError(
            "the template runtime parameters disagree with the upstream signature",
            code="triton_artifact_mismatch",
            stage="launch",
        )
    constexpr_names = [
        p.name for p in template.parameters if p.kind == "constexpr"
    ]
    expected_constexprs = [
        name for name, spec in signature.items() if spec == "constexpr"
    ]
    if constexpr_names != expected_constexprs:
        raise CompilationError(
            "the template constexpr parameters disagree with the upstream signature",
            code="triton_artifact_mismatch",
            stage="launch",
        )
    for parameter in template.parameters:
        if parameter.kind == "constexpr":
            if constants.get(parameter.name, object()) != parameter.value:
                raise CompilationError(
                    f"constexpr {parameter.name} disagrees with the template",
                    code="triton_artifact_mismatch",
                    stage="launch",
                )
        elif parameter.kind == "pointer":
            if signature[parameter.name] != f"*{parameter.dtype}":
                raise CompilationError(
                    f"pointer {parameter.name} disagrees with the template",
                    code="triton_artifact_mismatch",
                    stage="launch",
                )
            expected = specialization.get(parameter.name, {}).get("divisibility", 1)
            if parameter.alignment_bytes != expected:
                raise CompilationError(
                    f"pointer {parameter.name} alignment disagrees with the template",
                    code="triton_artifact_mismatch",
                    stage="launch",
                )
        elif signature[parameter.name] != _SCALAR_SPEC.get(
            parameter.dtype, parameter.dtype
        ):
            raise CompilationError(
                f"scalar {parameter.name} disagrees with the template",
                code="triton_artifact_mismatch",
                stage="launch",
            )
    return template


def _template_for(loaded, kernel, signature, constants, specialization, source_hash):
    from llk.triton import capture_kernel, resolve_triton_python

    key = (
        loaded.artifact.identity(),
        source_hash,
        json.dumps(signature, sort_keys=False),
        json.dumps(constants, sort_keys=True, default=repr),
        json.dumps(specialization, sort_keys=True, default=repr),
        kernel.__name__,
    )
    with _ACTIVATION_LOCK:
        cached = _TEMPLATES.get(key)
        if cached is not None:
            _TEMPLATES.move_to_end(key)
            return cached
    path = Path(kernel.__code__.co_filename)
    if not path.is_file():
        raise UnsupportedFeatureError(
            f"the upstream kernel source {path} is unavailable",
            code="source_unavailable",
        )
    export = capture_kernel(
        path,
        kernel.__name__,
        signature,
        constexprs=constants,
        specialization=specialization,
        options=_export_options(),
        triton_python=resolve_triton_python(),
    )
    if export.template.entry != kernel.__name__:
        raise CompilationError(
            "the exported template names a different entry function",
            code="triton_artifact_mismatch",
            stage="launch",
        )
    template = _validate_template(
        export.template, source_hash, signature, constants, specialization
    )
    with _ACTIVATION_LOCK:
        _TEMPLATES[key] = template
        while len(_TEMPLATES) > TEMPLATE_CACHE_ENTRIES:
            _TEMPLATES.popitem(last=False)
    return template


def _validate_artifact(loaded, template):
    artifact = loaded.artifact
    if artifact.ttir_hash != template.ttir_hash:
        raise CompilationError(
            "the compiled artifact TTIR does not match the exported template",
            code="triton_artifact_mismatch",
            stage="launch",
        )
    if artifact.entry != template.entry:
        raise CompilationError(
            "the compiled artifact entry does not match the exported template",
            code="triton_artifact_mismatch",
            stage="launch",
        )
    if artifact.upstream_version != template.toolchain.triton_version:
        raise CompilationError(
            "the artifact was produced by a different upstream Triton version",
            code="triton_artifact_mismatch",
            stage="launch",
        )
    if artifact.schema_version != ARTIFACT_SCHEMA_VERSION:
        raise CompilationError(
            "the artifact schema version is unsupported",
            code="triton_artifact_mismatch",
            stage="launch",
        )


class CPULauncher:
    """Upstream launcher: bind the artifact at launch and invoke LLK natively."""

    def __init__(self, src, metadata):
        self._src = src
        self._metadata = metadata

    # -- upstream protocol -----------------------------------------------

    def __call__(
        self,
        grid0,
        grid1,
        grid2,
        stream,
        function,
        packed_metadata,
        launch_metadata,
        enter_hook,
        exit_hook,
        *args,
    ):
        if not isinstance(function, LoadedArtifact):
            raise CompilationError(
                "the loaded Triton artifact is not an LLK unbound template",
                code="triton_artifact_mismatch",
                stage="launch",
            )
        activation = _require_activation()
        compiled = self._launch(function, args, (grid0, grid1, grid2), activation)
        return compiled

    def _launch(self, loaded, args, grid, activation):
        kernel, signature, constants, specialization = _source_facts(self._src)
        path = Path(kernel.__code__.co_filename)
        try:
            source_bytes = path.read_bytes()
        except OSError as error:
            raise UnsupportedFeatureError(
                f"cannot read the upstream kernel source {path}: {error}",
                code="source_unavailable",
            ) from error
        source_hash = hashlib.sha256(source_bytes).hexdigest()
        template = _template_for(
            loaded, kernel, signature, constants, specialization, source_hash
        )
        _validate_artifact(loaded, template)
        by_name = dict(zip(getattr(self._src.fn, "arg_names", ()), args))
        if len(by_name) != len(args):
            _fail("upstream kernel argument names are not unique")
        missing = [
            parameter.name
            for parameter in template.parameters
            if parameter.name not in by_name
        ]
        if missing:
            _fail(
                f"launch did not supply template parameters {missing}",
                code="invalid_frontend_argument",
            )
        ordered = tuple(by_name[parameter.name] for parameter in template.parameters)
        prepared = _bind(
            template,
            ordered,
            grid,
            activation.options,
            source_bytes.decode("utf-8", "replace"),
        )
        cache = _compilation_cache()
        compiled = cache.get_or_compile(prepared.request, _compile_request)
        try:
            compiled.invoke(prepared.inputs, prepared.outputs)
        finally:
            compiled.close()
        report = compiled.report
        activation.record(
            TritonLaunchReport(
                entry=template.entry,
                template_identity=template.identity(),
                artifact_identity=loaded.artifact.identity(),
                source_hash=template.source_hash,
                ttir_hash=template.ttir_hash,
                native_request_identity=report.native_request_identity,
                frontend_identity=report.frontend_identity,
                compiler_identity=report.compiler_identity,
                plan_id=report.plan_id,
                abi_hash=report.abi_hash,
                backend=report.backend,
                target_hash=report.target_hash,
                triton_version=template.toolchain.triton_version,
                runtime_grid=prepared.runtime_grid,
                native_grid=prepared.native_grid,
                block=prepared.block,
                tensors=prepared.tensors,
                scalars=prepared.scalars,
                constexprs=dict(template.constexprs),
                native_invocations=activation.native_invocations,
            )
        )
        return compiled

    # -- binding contract -------------------------------------------------

    @staticmethod
    def bind_triton_template(template, args, grid, options, *, source=""):
        """Bind a PF17 template and launch arguments into a ``CompileRequest``.

        ``args`` are in the template's parameter order: runtime pointers and
        scalars with their concrete values, plus each constexpr value.  Every
        deferred tensor/scalar/alignment/coverage constraint is checked here,
        before any native request exists.
        """
        if not isinstance(options, CompileOptions):
            raise LLKError("bind_triton_template requires CompileOptions")
        return _bind(template, args, grid, options, source).request


# ---------------------------------------------------------------------------
# Driver and activation.
# ---------------------------------------------------------------------------


def _benchmark(kernel_call, *, quantiles=(0.5,), warmup=5, rep=10, **kwargs):
    for _ in range(max(1, int(warmup))):
        kernel_call()
    samples = []
    for _ in range(max(1, int(rep))):
        start = time.perf_counter()
        kernel_call()
        samples.append(time.perf_counter() - start)
    samples.sort()
    last = len(samples) - 1
    return [samples[min(last, max(0, int(q * len(samples))))] for q in quantiles]


class LLKCPUDriver(DriverBase):
    """Upstream driver for the ``llk_cpu`` target.

    ``is_active`` is False so a machine with a GPU keeps its own default
    driver; :func:`activate` installs this driver explicitly and scoped.
    """

    utils = CPUUtils()
    launcher_cls = CPULauncher

    @staticmethod
    def is_active():
        return False

    def map_python_to_cpp_type(self, ty):
        mapping = {
            "i1": "bool",
            "i8": "int8_t",
            "i16": "int16_t",
            "i32": "int32_t",
            "i64": "int64_t",
            "fp16": "uint16_t",
            "bf16": "uint16_t",
            "fp32": "float",
            "fp64": "double",
        }
        if ty in mapping:
            return mapping[ty]
        if isinstance(ty, str) and ty.startswith("*"):
            inner = mapping.get(ty[1:])
            if inner is not None:
                return inner + "*"
        raise UnsupportedFeatureError(
            f"the LLK CPU driver has no C type for {ty!r}",
            code="unsupported_triton_launch",
        )

    def get_current_target(self):
        return GPUTarget(BACKEND_NAME, "native", 1)

    def get_active_torch_device(self):
        try:
            import torch
        except ImportError:  # pragma: no cover - Torch lane requires it
            return "cpu"
        return torch.device("cpu")

    def get_benchmarker(self):
        return _benchmark

    def get_current_device(self):
        return 0

    def get_current_stream(self, device):
        return None


def register_backend():
    """Register the LLK CPU backend with upstream Triton's in-process registry."""
    ensure_triton_importable()
    from triton.backends import Backend, backends

    existing = backends.get(BACKEND_NAME)
    if existing is None:
        backends[BACKEND_NAME] = Backend(LLKCPUBackend, LLKCPUDriver)
    elif (
        existing.compiler is not LLKCPUBackend
        or existing.driver is not LLKCPUDriver
    ):
        raise UnsupportedFeatureError(
            f"another plugin already owns the {BACKEND_NAME!r} Triton backend",
            code="upstream_toolchain_unqualified",
        )
    return backends[BACKEND_NAME]


def _current_driver_config():
    from triton.runtime.driver import driver as driver_config

    return driver_config


_MISSING = object()


class Activation:
    """Scoped process-global activation of the LLK CPU Triton driver."""

    def __init__(self, options):
        self._options = options
        self._previous = _MISSING
        self._entered = False
        self._last_report = None
        self._invocations = 0

    @property
    def options(self):
        return self._options

    @property
    def native_invocations(self):
        return self._invocations + 1

    def last_launch_report(self):
        with _ACTIVATION_LOCK:
            return self._last_report

    def record(self, report):
        with _ACTIVATION_LOCK:
            self._invocations = report.native_invocations
            self._last_report = report
            global _LAST_REPORT
            _LAST_REPORT = report

    def __enter__(self):
        _ACTIVATION_LOCK.acquire()
        try:
            self._attach()
        except BaseException:
            _ACTIVATION_LOCK.release()
            raise
        return self

    def _attach(self):
        driver_config = _current_driver_config()
        self._previous = getattr(driver_config, "_active", _MISSING)
        driver_config.set_active(LLKCPUDriver())
        _ACTIVATIONS.append(self)
        self._entered = True

    def __exit__(self, exc_type, exc_value, traceback):
        try:
            self._detach()
        finally:
            _ACTIVATION_LOCK.release()
        return False

    def _detach(self):
        if not self._entered:
            return
        self._entered = False
        if _ACTIVATIONS and _ACTIVATIONS[-1] is self:
            _ACTIVATIONS.pop()
        driver_config = _current_driver_config()
        if self._previous is _MISSING:
            driver_config._active = None
        elif self._previous is None:
            driver_config._active = None
        else:
            driver_config.set_active(self._previous)


def _require_activation():
    with _ACTIVATION_LOCK:
        if not _ACTIVATIONS:
            raise UnsupportedFeatureError(
                "Triton kernels launch only inside llk.triton.activate()",
                code="native_backend_unavailable",
            )
        return _ACTIVATIONS[-1]


def last_launch_report():
    """Return the most recent launch evidence, or None before any launch."""
    with _ACTIVATION_LOCK:
        if _ACTIVATIONS and _ACTIVATIONS[-1]._last_report is not None:
            return _ACTIVATIONS[-1]._last_report
        return _LAST_REPORT


def _coexistence_key():
    return (
        os.environ.get("LLK_FRONTEND_LIBRARY"),
        os.environ.get("LLK_TRITON_TOOLCHAIN_LOCK"),
        str(Path(__file__).resolve().parent),
    )


def _probe_worker():
    return Path(__file__).resolve().parent / "_probe.py"


def _run_probe(order, triton_python):
    argv = [
        sys.executable,
        "-I",
        str(_probe_worker()),
        "--order",
        order,
    ]
    if triton_python:
        argv.extend(["--triton-python", str(triton_python)])
    argv.extend(["--llk-python", str(Path(__file__).resolve().parents[2])])
    try:
        process = subprocess.Popen(
            argv,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=dict(os.environ),
        )
    except OSError as error:
        return False, f"cannot start the coexistence probe: {error}"
    try:
        stdout, stderr = process.communicate(timeout=PROBE_TIMEOUT)
    except subprocess.TimeoutExpired:
        process.kill()
        process.communicate()
        return False, f"the coexistence probe ({order}) timed out"
    overflow = len(stdout) > MAX_PROBE_OUTPUT or len(stderr) > MAX_PROBE_OUTPUT
    tail = stderr.decode("utf-8", "replace").strip()
    if len(tail) > 600:
        tail = "..." + tail[-600:]
    lines = stdout.decode("utf-8", "replace").split("\n", 1)
    if overflow or len(lines) != 2 or lines[0] != MARKER:
        reason = f"the coexistence probe ({order}) produced no framed result"
        return False, f"{reason}: {tail}" if tail else reason
    try:
        frame = json.loads(lines[1])
    except json.JSONDecodeError:
        return False, f"the coexistence probe ({order}) returned malformed JSON"
    if frame.get("ok") is True:
        return True, ""
    message = str(frame.get("message", "unknown failure"))
    return False, f"{order}: {message}" + (f": {tail}" if tail else "")


def coexistence_status(force=False):
    """Qualify loading Torch/Triton/LLK in both orders, in subprocesses.

    A duplicate LLVM image is a process-fatal load error, so this can never be
    probed in the calling process.  Results are cached per process.
    """
    key = _coexistence_key()
    if not force:
        with _ACTIVATION_LOCK:
            cached = _COEXISTENCE.get(key)
            if cached is not None:
                return cached
    from llk.triton._toolchain import resolve_triton_python

    triton_python = resolve_triton_python()
    if triton_python is None:
        status = CoexistenceStatus(
            False, ("the qualified Triton package is not importable",)
        )
    else:
        reasons = []
        for order in ("triton-llk", "llk-triton"):
            ok, reason = _run_probe(order, triton_python)
            if not ok:
                reasons.append(reason)
        status = CoexistenceStatus(not reasons, tuple(reasons))
    with _ACTIVATION_LOCK:
        _COEXISTENCE[key] = status
    return status


def _plain(value):
    if isinstance(value, Mapping):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    return value


def _qualify_version():
    import triton

    from llk.triton import toolchain_lock

    lock = toolchain_lock()
    key = json.dumps(_plain(lock), sort_keys=True)
    with _ACTIVATION_LOCK:
        cached = _QUALIFIED.get(key)
    if cached is not None:
        identity = cached
    else:
        from llk.triton import qualify_toolchain

        identity = qualify_toolchain()
        with _ACTIVATION_LOCK:
            _QUALIFIED[key] = identity
    if triton.__version__ != identity.triton_version:
        raise UnsupportedFeatureError(
            f"the imported Triton {triton.__version__} is not the qualified "
            f"{identity.triton_version}",
            code="upstream_toolchain_unqualified",
        )
    return identity


def _native_library_available():
    from llk.runtime import _ffi

    if _ffi._library_file() is None:
        raise UnsupportedFeatureError(
            "LLKFrontendCAPI is unavailable; build with "
            "-DLLK_BUILD_PYTHON_CAPI=ON and set LLK_FRONTEND_LIBRARY",
            code="native_backend_unavailable",
        )


def activate(options=None):
    """Return a scoped activation context for the ``llk_cpu`` Triton target.

    The context serializes against Triton's process-global driver, installs the
    LLK CPU driver, and restores the previous driver on success or error.
    """
    if options is None:
        options = CompileOptions()
    elif isinstance(options, Mapping):
        options = CompileOptions(**dict(options))
    elif not isinstance(options, CompileOptions):
        raise TypeError("options must be CompileOptions or a mapping")
    ensure_triton_importable()
    _qualify_version()
    _native_library_available()
    status = coexistence_status()
    if not status.qualified:
        raise UnsupportedFeatureError(
            "upstream Triton and LLK native libraries do not coexist in one "
            "process: " + "; ".join(status.reasons),
            code="upstream_toolchain_unqualified",
        )
    register_backend()
    return Activation(options)
