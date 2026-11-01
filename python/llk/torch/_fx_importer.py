"""Strict conversion of normalized ATen FX graphs into frontend requests.

The boundary accepts exactly the computations the shared native frontend can
represent: two-operand elementwise add/multiply over one compact f32 rank-two
domain, one whole-tensor matmul, and the fused ``silu(x @ wg) * (x @ wu)``
pattern. Everything else identifies the offending FX node and fails; no path
executes ``gm.forward`` or mutates the graph.

The generated program is the same tile program the LLK source frontend emits,
so the native importer proves the access, mask and reduction contracts before
mapping and execution.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.fx import GraphModule, Node

from llk._errors import TensorContractError, UnsupportedFeatureError
from llk.frontend import (
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
)
from llk.runtime._ffi import CompiledKernel

FX_SPAN = SourceSpan("<torch.fx>", 1, 0)
FLOAT32 = "f32"
INDEX = "index"
PREDICATE = "i1"
POINTER_F32 = "ptr<f32>"
ELEMENTWISE_BLOCK = (8, 16)
FLOAT_BYTES = 4

SWIGLU_CONTRACT = NumericalContract(
    activation_expression="x/(1+exp(-x))",
    activation_approximation="exact",
)
DEFAULT_CONTRACT = NumericalContract()

ATEN = torch.ops.aten
_ADD = ATEN.add.Tensor
_MUL = ATEN.mul.Tensor
_MM = ATEN.mm.default
_SILU = ATEN.silu.default

_VIEW_METHODS = {
    "as_strided",
    "expand",
    "expand_as",
    "flatten",
    "narrow",
    "permute",
    "reshape",
    "reshape_as",
    "select",
    "slice",
    "squeeze",
    "t",
    "transpose",
    "unsqueeze",
    "view",
    "view_as",
}


class UnsupportedFXError(UnsupportedFeatureError):
    """An FX graph is outside the strict supported envelope."""

    def __init__(
        self,
        message: str,
        *,
        code: str,
        node_name: str,
        node_target: object = None,
    ) -> None:
        super().__init__(message, code=code, stage="compile")
        self.node_name = node_name
        self.node_target = node_target


@dataclass(frozen=True)
class ImportedFXOutput:
    """One output's request plus the graph-argument mapping it invokes with."""

    request: CompileRequest
    input_indices: tuple[int, ...]
    output_shape: tuple[int, int]
    device: str


@dataclass(frozen=True)
class _Pattern:
    kind: str
    input_nodes: tuple[Node, ...]
    input_shapes: tuple[tuple[int, int], ...]
    output_shape: tuple[int, int]
    contract: NumericalContract
    multiply: bool = False
    reduction: int | None = None
    nodes: frozenset = frozenset()


# --- graph validation -----------------------------------------------------


def _target_name(target) -> str:
    if isinstance(target, str):
        return target
    if isinstance(target, torch._ops.OpOverload):
        parts = str(target).split(".")
        return parts[1] if len(parts) > 1 else parts[0]
    return str(getattr(target, "__name__", target))


def _is_mutable(target) -> bool:
    if isinstance(target, torch._ops.OpOverload):
        return bool(target._schema.is_mutable)
    name = _target_name(target)
    return name.endswith("_") and not name.startswith("__") and len(name) > 1


def _is_view(target) -> bool:
    return _target_name(target) in _VIEW_METHODS


def reject_unsupported_graph(graph) -> None:
    """Reject mutation and layout dependence before normalization.

    Dynamo hands the backend a Python-target graph, while AOTAutograd hands the
    compiler an ATen-target graph. In the first form an in-place call is a
    method such as ``add_``; in the second it is a mutable ``OpOverload``.
    Both must fail with the node's identity, never be functionalized into a
    silently different graph.
    """
    if not isinstance(graph, GraphModule):
        raise TypeError("LLK requires a torch.fx.GraphModule")
    for node in graph.graph.nodes:
        # Dynamo captures tensor operations as call_method nodes and names the
        # method (for example ``add_``); AOTAutograd's normalized graph uses
        # call_function with an OpOverload. Both forms must be inspected.
        if node.op not in ("call_function", "call_method"):
            continue
        if _is_mutable(node.target):
            raise UnsupportedFXError(
                f"in-place FX node {node.name!r} is not supported by the "
                "strict LLK backend",
                code="unsupported_fx_mutation",
                node_name=node.name,
                node_target=node.target,
            )
        if _is_view(node.target):
            raise UnsupportedFXError(
                f"layout-dependent FX node {node.name!r} is not supported by "
                "the strict LLK backend",
                code="unsupported_fx_view",
                node_name=node.name,
                node_target=node.target,
            )


def _unsupported(node: Node, message: str, code: str) -> UnsupportedFXError:
    return UnsupportedFXError(
        message,
        code=code,
        node_name=node.name,
        node_target=getattr(node, "target", None),
    )


def _tensor_metadata(tensor, node: Node):
    if not isinstance(tensor, torch.Tensor):
        if type(tensor).__name__ == "SymInt":
            raise UnsupportedFXError(
                f"graph input {node.name!r} has a symbolic extent; the first "
                "LLK release specializes concrete shapes only",
                code="unsupported_symbolic_shape",
                node_name=node.name,
            )
        raise UnsupportedFXError(
            f"graph input {node.name!r} is not a tensor",
            code="unsupported_fx_operand",
            node_name=node.name,
        )
    from torch._subclasses.fake_tensor import FakeTensor

    if type(tensor) is not torch.Tensor and not isinstance(tensor, FakeTensor):
        raise UnsupportedFXError(
            f"graph input {node.name!r} uses an unsupported tensor subclass",
            code="unsupported_fx_subclass",
            node_name=node.name,
        )
    if tensor.device.type != "cpu":
        raise UnsupportedFXError(
            f"graph input {node.name!r} is on {tensor.device}; the strict "
            "backend executes CPU tensors only",
            code="unsupported_fx_device",
            node_name=node.name,
        )
    if tensor.dtype is not torch.float32:
        raise UnsupportedFXError(
            f"graph input {node.name!r} has dtype {tensor.dtype}; the strict "
            "backend accepts float32 only",
            code="unsupported_fx_dtype",
            node_name=node.name,
        )
    if tensor.dim() != 2:
        raise unsupported_shape(node, "graph inputs must be rank two")
    shape = tuple(int(size) for size in tensor.shape)
    if any(size < 1 for size in shape):
        raise unsupported_shape(node, "graph inputs require positive extents")
    if tensor.requires_grad:
        raise UnsupportedFXError(
            "inference only: requires_grad inputs need the registered "
            "training path (PF16)",
            code="unsupported_training_contract",
            node_name=node.name,
        )
    strides = tuple(int(stride) for stride in tensor.stride())
    if strides != (shape[1], 1) or int(tensor.storage_offset()) != 0:
        raise UnsupportedFXError(
            f"graph input {node.name!r} must be compact row-major with zero "
            "storage offset",
            code="unsupported_tensor_layout",
            node_name=node.name,
        )
    return shape, strides


def unsupported_shape(node: Node, message: str) -> UnsupportedFXError:
    return _unsupported(
        node, f"{message}: FX node {node.name!r}", "unsupported_fx_shape"
    )


def _binary_nodes(node: Node) -> tuple[Node, Node]:
    if node.kwargs:
        raise _unsupported(
            node,
            f"FX node {node.name!r} has unsupported keyword arguments",
            "unsupported_fx_operand",
        )
    if len(node.args) != 2 or any(not isinstance(arg, Node) for arg in node.args):
        raise _unsupported(
            node,
            f"FX node {node.name!r} requires two tensor operands",
            "unsupported_fx_operand",
        )
    return node.args[0], node.args[1]


def _require_placeholders(node: Node, operands) -> None:
    for operand in operands:
        if operand.op != "placeholder":
            raise _unsupported(
                operand,
                f"FX node {node.name!r} composes unsupported node "
                f"{operand.name!r}; the strict backend accepts only supported "
                "operations directly on graph inputs",
                "unsupported_fx_graph",
            )


def _matches(target, expected) -> bool:
    return target is expected or target == expected


# --- recognition ----------------------------------------------------------


def _recognize(root: Node, metadata) -> _Pattern:
    target = root.target
    if _matches(target, _MUL):
        fused = _match_swiglu(root, metadata)
        if fused is not None:
            return fused
        return _match_elementwise(root, metadata, multiply=True)
    if _matches(target, _ADD):
        return _match_elementwise(root, metadata, multiply=False)
    if _matches(target, _MM):
        return _match_matmul(root, metadata)
    if _matches(target, _SILU):
        raise _unsupported(
            root,
            "standalone SiLU is unsupported; only silu(x @ wg) * (x @ wu) "
            "lowers to the fused SwiGLU kernel",
            "unsupported_fx_graph",
        )
    raise _unsupported(
        root,
        f"unsupported FX node {root.name!r} ({_target_name(target)})",
        "unsupported_fx_operation",
    )


def _match_elementwise(root: Node, metadata, *, multiply: bool) -> _Pattern:
    left, right = _binary_nodes(root)
    _require_placeholders(root, (left, right))
    left_shape = metadata[left][0]
    right_shape = metadata[right][0]
    if left_shape != right_shape:
        raise unsupported_shape(root, "elementwise operands must have one shape")
    return _Pattern(
        kind="elementwise",
        input_nodes=(left, right),
        input_shapes=(left_shape, right_shape),
        output_shape=left_shape,
        contract=DEFAULT_CONTRACT,
        multiply=multiply,
        nodes=frozenset((root, left, right)),
    )


def _match_matmul(root: Node, metadata) -> _Pattern:
    left, right = _binary_nodes(root)
    _require_placeholders(root, (left, right))
    left_shape = metadata[left][0]
    right_shape = metadata[right][0]
    if left_shape[1] != right_shape[0]:
        raise unsupported_shape(
            root, "matmul contraction extents must agree"
        )
    return _Pattern(
        kind="matmul",
        input_nodes=(left, right),
        input_shapes=(left_shape, right_shape),
        output_shape=(left_shape[0], right_shape[1]),
        contract=DEFAULT_CONTRACT,
        reduction=left_shape[1],
        nodes=frozenset((root, left, right)),
    )


def _match_swiglu(root: Node, metadata):
    left, right = _binary_nodes(root)
    if not isinstance(left, Node) or not isinstance(right, Node):
        return None
    if _matches(left.target, _SILU):
        silu, up = left, right
    elif _matches(right.target, _SILU):
        silu, up = right, left
    else:
        return None
    if not _matches(up.target, _MM) or len(up.args) != 2:
        return None
    if any(not isinstance(arg, Node) for arg in up.args):
        return None
    if len(silu.args) != 1 or not isinstance(silu.args[0], Node):
        return None
    gate = silu.args[0]
    if not _matches(gate.target, _MM) or len(gate.args) != 2:
        return None
    if any(not isinstance(arg, Node) for arg in gate.args):
        return None
    _require_placeholders(
        root, (gate.args[0], gate.args[1], up.args[0], up.args[1])
    )
    if gate.args[0] is not up.args[0]:
        raise _unsupported(
            root,
            "fused SwiGLU requires both projections to read the same left "
            "operand",
            "unsupported_fx_graph",
        )
    if gate.args[1] is up.args[1]:
        raise _unsupported(
            root,
            "fused SwiGLU requires two distinct weight tensors",
            "unsupported_fx_graph",
        )
    lhs = metadata[gate.args[0]][0]
    gate_rhs = metadata[gate.args[1]][0]
    up_rhs = metadata[up.args[1]][0]
    if lhs[1] != gate_rhs[0] or lhs[1] != up_rhs[0]:
        raise unsupported_shape(root, "SwiGLU contraction extents must agree")
    if gate_rhs != up_rhs:
        raise unsupported_shape(
            root, "SwiGLU gate and up projections must have one weight shape"
        )
    return _Pattern(
        kind="swiglu",
        input_nodes=(gate.args[0], gate.args[1], up.args[1]),
        input_shapes=(lhs, gate_rhs, up_rhs),
        output_shape=(lhs[0], gate_rhs[1]),
        contract=SWIGLU_CONTRACT,
        reduction=lhs[1],
        nodes=frozenset(
            (root, silu, up, gate, gate.args[0], gate.args[1], up.args[1])
        ),
    )


# --- program construction -------------------------------------------------


class _Builder:
    """Deterministic SSA emission matching the source frontend's tile program."""

    def __init__(self) -> None:
        self.operations: list[Operation] = []
        self._suspended: list[list[Operation]] = []
        self._next = 0

    def value(self, dtype: str, shape=()) -> Value:
        self._next += 1
        return Value(f"_v{self._next}", dtype, tuple(shape))

    def emit(
        self,
        opcode: str,
        operands=(),
        *,
        dtype: str | None = None,
        shape=(),
        attributes=None,
    ):
        result = None if dtype is None else self.value(dtype, shape)
        self.operations.append(
            Operation(
                opcode=opcode,
                operands=tuple(
                    item.id if isinstance(item, Value) else str(item)
                    for item in operands
                ),
                results=() if result is None else (result,),
                attributes=attributes or {},
                span=FX_SPAN,
            )
        )
        return result

    def begin_region(self) -> None:
        self._suspended.append(self.operations)
        self.operations = []

    def end_region(self) -> tuple[Operation, ...]:
        body = tuple(self.operations)
        self.operations = self._suspended.pop()
        return body

    def append(self, operation: Operation) -> None:
        self.operations.append(operation)


def _constant(b: _Builder, dtype: str, value, shape=()):
    return b.emit("constant", dtype=dtype, shape=shape, attributes={"value": value})


def _index(b: _Builder, value: int):
    return _constant(b, INDEX, int(value))


def _expand(b: _Builder, value: Value, axis: int, shape):
    return b.emit(
        "expand_dims", (value,), dtype=INDEX, shape=shape, attributes={"axis": axis}
    )


def _tile_axes(b: _Builder, block0: int, block1: int):
    pid0 = b.emit("program_id", dtype=INDEX, attributes={"axis": 0})
    pid1 = b.emit("program_id", dtype=INDEX, attributes={"axis": 1})
    rows_base = b.emit(
        "mul", (pid0, _index(b, block0)), dtype=INDEX, shape=()
    )
    rows_range = b.emit(
        "range", dtype=INDEX, shape=(block0,), attributes={"start": 0, "end": block0}
    )
    rows = b.emit("add", (rows_base, rows_range), dtype=INDEX, shape=(block0,))
    cols_base = b.emit(
        "mul", (pid1, _index(b, block1)), dtype=INDEX, shape=()
    )
    cols_range = b.emit(
        "range", dtype=INDEX, shape=(block1,), attributes={"start": 0, "end": block1}
    )
    cols = b.emit("add", (cols_base, cols_range), dtype=INDEX, shape=(block1,))
    return rows, cols


def _row_major_offsets(b: _Builder, rows: Value, cols: Value, row_extent: int, block):
    block0, block1 = block
    rows_e = _expand(b, rows, 1, (block0, 1))
    row_off = b.emit(
        "mul", (rows_e, _index(b, row_extent)), dtype=INDEX, shape=(block0, 1)
    )
    row_base = b.emit("add", (_index(b, 0), row_off), dtype=INDEX, shape=(block0, 1))
    cols_e = _expand(b, cols, 0, (1, block1))
    offsets = b.emit(
        "add", (row_base, cols_e), dtype=INDEX, shape=(block0, block1)
    )
    return offsets, rows_e, cols_e


def _true_mask(b: _Builder, shape):
    true = _constant(b, PREDICATE, True)
    return b.emit(
        "broadcast", (true,), dtype=PREDICATE, shape=shape, attributes={"dimensions": ()}
    )


def _bounds_mask(b: _Builder, rows_e: Value, cols_e: Value, extents, block):
    block0, block1 = block
    row_ok = b.emit(
        "compare",
        (rows_e, _index(b, extents[0])),
        dtype=PREDICATE,
        shape=(block0, 1),
        attributes={"predicate": "lt"},
    )
    col_ok = b.emit(
        "compare",
        (cols_e, _index(b, extents[1])),
        dtype=PREDICATE,
        shape=(1, block1),
        attributes={"predicate": "lt"},
    )
    false = _constant(b, PREDICATE, False)
    return b.emit(
        "select", (row_ok, col_ok, false), dtype=PREDICATE, shape=(block0, block1)
    )


def _address(b: _Builder, tensor: str, offsets: Value, row_extent: int, shape):
    divisor = _index(b, row_extent)
    rows = b.emit("div", (offsets, divisor), dtype=INDEX, shape=shape)
    cols = b.emit("rem", (offsets, divisor), dtype=INDEX, shape=shape)
    return b.emit(
        "address", (tensor, rows.id, cols.id), dtype=POINTER_F32, shape=shape
    )


def _load(b: _Builder, tensor: str, offsets: Value, mask: Value, row_extent: int, shape):
    pointer = _address(b, tensor, offsets, row_extent, shape)
    padding = _constant(b, FLOAT32, 0.0)
    return b.emit("load", (pointer, mask, padding), dtype=FLOAT32, shape=shape)


def _store(
    b: _Builder,
    tensor: str,
    offsets: Value,
    value: Value,
    mask: Value,
    row_extent: int,
    shape,
) -> None:
    pointer = _address(b, tensor, offsets, row_extent, shape)
    b.emit("store", (pointer, value, mask))


def _spec(name: str, shape, direction: str, effect: str) -> TensorSpec:
    return TensorSpec(name, shape, FLOAT32, (shape[1], 1), direction, effect)


def _ceil_divide(value: int, block: int) -> int:
    return value // block + (value % block != 0)


def _elementwise_body(b: _Builder, pattern: _Pattern) -> None:
    extent0, extent1 = pattern.output_shape
    block = ELEMENTWISE_BLOCK
    rows, cols = _tile_axes(b, block[0], block[1])
    offsets, rows_e, cols_e = _row_major_offsets(
        b, rows, cols, extent1, block
    )
    mask = _bounds_mask(b, rows_e, cols_e, (extent0, extent1), block)
    left = _load(b, "a", offsets, mask, extent1, block)
    right = _load(b, "b", offsets, mask, extent1, block)
    result = b.emit(
        "mul" if pattern.multiply else "add",
        (left, right),
        dtype=FLOAT32,
        shape=block,
    )
    _store(b, "output", offsets, result, mask, extent1, block)


def _contraction_body(b: _Builder, pattern: _Pattern) -> None:
    extent0, extent1 = pattern.output_shape
    reduction = pattern.reduction
    arms = 2 if pattern.kind == "swiglu" else 1
    names = ("x", "wg", "wu") if arms == 2 else ("a", "b")
    block = (extent0, extent1)
    rows, cols = _tile_axes(b, extent0, extent1)
    reduction_range = b.emit(
        "range",
        dtype=INDEX,
        shape=(reduction,),
        attributes={"start": 0, "end": reduction},
    )
    zeros = tuple(_constant(b, FLOAT32, 0.0, block) for _ in range(arms))
    lower = _index(b, 0)
    upper = _index(b, reduction)
    induction = b.value(INDEX)
    carried = tuple(b.value(FLOAT32, block) for _ in range(arms))
    results = tuple(b.value(FLOAT32, block) for _ in range(arms))
    b.begin_region()
    rows_e = _expand(b, rows, 1, (extent0, 1))
    row_off = b.emit(
        "mul", (rows_e, _index(b, reduction)), dtype=INDEX, shape=(extent0, 1)
    )
    row_base = b.emit(
        "add", (_index(b, 0), row_off), dtype=INDEX, shape=(extent0, 1)
    )
    reduction_lanes = b.emit(
        "add", (induction, reduction_range), dtype=INDEX, shape=(reduction,)
    )
    lhs_cols = _expand(b, reduction_lanes, 0, (1, reduction))
    lhs_offsets = b.emit(
        "add", (row_base, lhs_cols), dtype=INDEX, shape=(extent0, reduction)
    )
    lhs = _load(
        b,
        names[0],
        lhs_offsets,
        _true_mask(b, (extent0, reduction)),
        reduction,
        (extent0, reduction),
    )
    rhs_tiles = []
    for index in range(1, arms + 1):
        rhs_rows = _expand(b, reduction_lanes, 1, (reduction, 1))
        rhs_off = b.emit(
            "mul", (rhs_rows, _index(b, extent1)), dtype=INDEX, shape=(reduction, 1)
        )
        rhs_base = b.emit(
            "add", (_index(b, 0), rhs_off), dtype=INDEX, shape=(reduction, 1)
        )
        cols_e = _expand(b, cols, 0, (1, extent1))
        offsets = b.emit(
            "add", (rhs_base, cols_e), dtype=INDEX, shape=(reduction, extent1)
        )
        rhs_tiles.append(
            _load(
                b,
                names[index],
                offsets,
                _true_mask(b, (reduction, extent1)),
                extent1,
                (reduction, extent1),
            )
        )
    dots = tuple(
        b.emit("dot", (lhs, tile, accumulator), dtype=FLOAT32, shape=block)
        for tile, accumulator in zip(rhs_tiles, carried)
    )
    b.emit("yield", dots)
    body = b.end_region()
    b.append(
        Operation(
            opcode="counted_loop",
            operands=tuple([lower.id, upper.id] + [zero.id for zero in zeros]),
            results=results,
            attributes={"step": reduction},
            regions=(Region((induction,) + carried, body),),
            span=FX_SPAN,
        )
    )
    output_mask = _true_mask(b, block)
    if arms == 2:
        activated = b.emit(
            "silu",
            (results[0],),
            dtype=FLOAT32,
            shape=block,
            attributes={"approximation": "exact"},
        )
        value = b.emit("mul", (activated, results[1]), dtype=FLOAT32, shape=block)
    else:
        value = results[0]
    output_offsets, _, _ = _row_major_offsets(b, rows, cols, extent1, block)
    _store(
        b,
        "output",
        output_offsets,
        value,
        output_mask,
        extent1,
        block,
    )


def _build_program(
    pattern: _Pattern,
    *,
    entry: str,
    source: str,
    options: CompileOptions,
) -> CompileRequest:
    """Build the shared tile program for one recognized operator shape.

    The decorator, Dynamo and custom-op routes all reach the native service
    through this builder so their tile programs cannot drift apart.
    """
    b = _Builder()
    if pattern.kind == "elementwise":
        _elementwise_body(b, pattern)
        specs = (
            _spec("a", pattern.input_shapes[0], "input", "read"),
            _spec("b", pattern.input_shapes[1], "input", "read"),
            _spec("output", pattern.output_shape, "output", "write"),
        )
        launch = LaunchSpec(
            (
                _ceil_divide(pattern.output_shape[0], ELEMENTWISE_BLOCK[0]),
                _ceil_divide(pattern.output_shape[1], ELEMENTWISE_BLOCK[1]),
                1,
            ),
            (ELEMENTWISE_BLOCK[0], ELEMENTWISE_BLOCK[1], 1),
        )
        constexprs = {
            "M": pattern.output_shape[0],
            "N": pattern.output_shape[1],
            "BLOCK_M": ELEMENTWISE_BLOCK[0],
            "BLOCK_N": ELEMENTWISE_BLOCK[1],
        }
    else:
        _contraction_body(b, pattern)
        if pattern.kind == "swiglu":
            input_names = ("x", "wg", "wu")
        else:
            input_names = ("a", "b")
        specs = tuple(
            _spec(name, shape, "input", "read")
            for name, shape in zip(input_names, pattern.input_shapes)
        ) + (_spec("output", pattern.output_shape, "output", "write"),)
        launch = LaunchSpec(
            (1, 1, 1),
            (pattern.output_shape[0], pattern.output_shape[1], 1),
        )
        constexprs = {
            "M": pattern.output_shape[0],
            "N": pattern.output_shape[1],
            "K": pattern.reduction,
            "BM": pattern.output_shape[0],
            "BN": pattern.output_shape[1],
            "BK": pattern.reduction,
        }
    b.emit("return")
    program = FrontendProgram(
        entry=entry,
        tensors=specs,
        constexprs=constexprs,
        launch=launch,
        operations=tuple(b.operations),
        numerical_contract=pattern.contract,
        source=source,
    )
    compile_options = dataclasses.replace(
        options, numerical_contract=pattern.contract
    )
    return CompileRequest(program, compile_options)


def _build_request(
    graph: GraphModule,
    pattern: _Pattern,
    index: int,
    placeholder_index,
    options: CompileOptions,
) -> ImportedFXOutput:
    return ImportedFXOutput(
        request=_build_program(
            pattern,
            entry=f"torch_output_{index}",
            source=graph.code or "",
            options=options,
        ),
        input_indices=tuple(placeholder_index(node) for node in pattern.input_nodes),
        output_shape=pattern.output_shape,
        device="cpu",
    )


def _op_pattern(kind: str, input_shapes) -> _Pattern:
    """Validate one functional operator's extents and build its pattern."""

    def fail(message):
        raise TensorContractError(message, code="invalid_frontend_argument")

    shapes = tuple(tuple(int(size) for size in shape) for shape in input_shapes)
    if any(len(shape) != 2 or any(size < 1 for size in shape) for shape in shapes):
        fail("LLK operators require positive rank-two input extents")
    if kind in ("add", "mul"):
        if len(shapes) != 2:
            fail(f"llk::{kind} requires two tensor operands")
        left, right = shapes
        if left != right:
            fail(f"llk::{kind} requires operands with one extent")
        return _Pattern(
            kind="elementwise",
            input_nodes=(),
            input_shapes=(left, right),
            output_shape=left,
            contract=DEFAULT_CONTRACT,
            multiply=kind == "mul",
        )
    if kind == "matmul":
        if len(shapes) != 2:
            fail("llk::matmul requires two tensor operands")
        left, right = shapes
        if left[1] != right[0]:
            fail("llk::matmul contraction extents must agree")
        return _Pattern(
            kind="matmul",
            input_nodes=(),
            input_shapes=(left, right),
            output_shape=(left[0], right[1]),
            contract=DEFAULT_CONTRACT,
            reduction=left[1],
        )
    if kind == "swiglu":
        if len(shapes) != 3:
            fail("llk::swiglu requires x, wg and wu tensors")
        x, gate, up = shapes
        if x[1] != gate[0] or x[1] != up[0]:
            fail("llk::swiglu contraction extents must agree")
        if gate != up:
            fail("llk::swiglu gate and up projections must have one extent")
        return _Pattern(
            kind="swiglu",
            input_nodes=(),
            input_shapes=(x, gate, up),
            output_shape=(x[0], gate[1]),
            contract=SWIGLU_CONTRACT,
            reduction=x[1],
        )
    fail(f"unsupported LLK operator kind {kind!r}")


def op_output_shape(kind: str, input_shapes) -> tuple[int, int]:
    """Return the declared output extent for one functional operator."""
    return _op_pattern(kind, input_shapes).output_shape


def op_request(
    kind: str, input_shapes, options: CompileOptions
) -> CompileRequest:
    """Build one supported functional operator's native request."""
    if not isinstance(options, CompileOptions):
        raise TypeError("op_request requires CompileOptions")
    return _build_program(
        _op_pattern(kind, input_shapes),
        entry=f"llk_{kind}",
        source=f"torch.ops.llk.{kind}",
        options=options,
    )


def _roots(output: Node) -> tuple[Node, ...]:
    value = output.args[0] if output.args else None
    if isinstance(value, Node):
        return (value,)
    if isinstance(value, (tuple, list)):
        if not value:
            raise UnsupportedFXError(
                "the FX graph returns no tensor",
                code="unsupported_fx_outputs",
                node_name=output.name,
            )
        roots = []
        for item in value:
            if not isinstance(item, Node):
                raise UnsupportedFXError(
                    "the FX graph returns a non-node value",
                    code="unsupported_fx_outputs",
                    node_name=output.name,
                )
            roots.append(item)
        return tuple(roots)
    raise UnsupportedFXError(
        "the FX graph returns an unsupported value",
        code="unsupported_fx_outputs",
        node_name=output.name,
    )


def import_fx_outputs(
    graph, example_inputs, options: CompileOptions
) -> tuple[ImportedFXOutput, ...]:
    """Recognize every output of one normalized ATen graph, in graph order."""
    if not isinstance(options, CompileOptions):
        raise TypeError("import_fx requires CompileOptions")
    reject_unsupported_graph(graph)
    placeholders = [node for node in graph.graph.nodes if node.op == "placeholder"]
    example_inputs = tuple(example_inputs)
    if len(example_inputs) != len(placeholders):
        raise UnsupportedFXError(
            "the FX graph and its example inputs declare different arities",
            code="unsupported_fx_operand",
            node_name="input",
        )
    for node in graph.graph.nodes:
        if node.op not in ("placeholder", "call_function", "output"):
            raise _unsupported(
                node,
                f"unsupported FX node kind {node.op!r}",
                "unsupported_fx_operation",
            )
    metadata = {
        node: _tensor_metadata(example_inputs[index], node)
        for index, node in enumerate(placeholders)
    }
    outputs = [node for node in graph.graph.nodes if node.op == "output"]
    if len(outputs) != 1:
        raise UnsupportedFXError(
            "the FX graph must have exactly one output node",
            code="unsupported_fx_outputs",
            node_name="output",
        )
    roots = _roots(outputs[0])
    position = {node: index for index, node in enumerate(placeholders)}
    imported = []
    consumed = set()
    for index, root in enumerate(roots):
        if root.op != "call_function":
            raise UnsupportedFXError(
                "aliasing a graph input as an output is unsupported",
                code="unsupported_fx_outputs",
                node_name=root.name,
            )
        pattern = _recognize(root, metadata)
        consumed |= pattern.nodes
        imported.append(
            _build_request(
                graph,
                pattern,
                index,
                lambda node: position[node],
                options,
            )
        )
    for node in graph.graph.nodes:
        if node.op in ("placeholder", "output") or node in consumed:
            continue
        raise _unsupported(
            node,
            f"FX node {node.name!r} is not part of a supported output",
            "unsupported_fx_operation",
        )
    return tuple(imported)


def import_fx_requests(graph, example_inputs, options: CompileOptions):
    """Return one request per output; the strict backend compiles each one."""
    return tuple(
        item.request for item in import_fx_outputs(graph, example_inputs, options)
    )


def import_fx(graph, example_inputs, options: CompileOptions) -> CompileRequest:
    """Import a single-output normalized ATen graph into one request."""
    requests = import_fx_requests(graph, example_inputs, options)
    if len(requests) != 1:
        raise UnsupportedFXError(
            "the single-request import interface requires one graph output",
            code="unsupported_fx_outputs",
            node_name="output",
        )
    return requests[0]


# --- compile-time options and invocation ----------------------------------


_OPTION_FIELDS = frozenset(field.name for field in dataclasses.fields(CompileOptions))


def default_config_root() -> str | None:
    """Find the repository configuration tree during source-tree development."""
    candidates = list(Path(__file__).resolve().parents) + [Path.cwd()]
    for candidate in candidates:
        if (candidate / "machines" / "x86-avx2-v2.yaml").is_file():
            return str(candidate)
    return None


def compile_options(changes=None) -> CompileOptions:
    """Build bounded compile options, discovering repository configs if needed."""
    changes = dict(changes or {})
    unknown = set(changes) - _OPTION_FIELDS
    if unknown:
        raise TypeError(f"unknown LLK compile options: {sorted(unknown)}")
    if "config_root" not in changes and "config_files" not in changes:
        root = default_config_root()
        if root is not None:
            changes["config_root"] = root
    return CompileOptions(**changes)


class TorchBuffer:
    """Explicit, checked CPU f32 tensor provider for the native ABI.

    FakeTensor has no storage and must never expose one. Real tensors are
    validated here, immediately before invocation, and the provider keeps the
    tensor strongly referenced for the duration of the ctypes call.
    """

    def __init__(self, tensor) -> None:
        if type(tensor) is not torch.Tensor:
            raise TensorContractError(
                "LLK native invocation requires a plain CPU torch.Tensor",
                code="invalid_frontend_buffer",
            )
        if tensor.device.type != "cpu" or tensor.layout is not torch.strided:
            raise TensorContractError(
                "LLK native invocation requires CPU strided tensors",
                code="invalid_frontend_buffer",
            )
        if tensor.dtype is not torch.float32:
            raise TensorContractError(
                "LLK native invocation requires float32 tensors",
                code="invalid_frontend_buffer",
            )
        if tensor.dim() != 2 or any(int(size) < 1 for size in tensor.shape):
            raise TensorContractError(
                "LLK native invocation requires positive rank-two tensors",
                code="invalid_frontend_buffer",
            )
        if not tensor.is_contiguous() or int(tensor.storage_offset()) != 0:
            raise TensorContractError(
                "LLK native invocation requires compact row-major storage "
                "with zero storage offset",
                code="invalid_frontend_buffer",
            )
        if tensor.requires_grad:
            raise TensorContractError(
                "LLK native invocation is inference only",
                code="unsupported_training_contract",
            )
        self._tensor = tensor

    @property
    def __array_interface__(self):
        tensor = self._tensor
        return {
            "shape": tuple(int(size) for size in tensor.shape),
            "typestr": "<f4",
            "data": (int(tensor.data_ptr()), False),
            "strides": tuple(
                int(stride) * FLOAT_BYTES for stride in tensor.stride()
            ),
            "version": 3,
        }


class _Invoker:
    """One imported output, its compiled handle and its argument mapping."""

    def __init__(self, imported: ImportedFXOutput, kernel: CompiledKernel):
        self._imported = imported
        self._kernel = kernel

    def __call__(self, arguments):
        inputs = []
        for index in self._imported.input_indices:
            if index >= len(arguments):
                raise TensorContractError(
                    "native invocation is missing a graph argument",
                    code="invalid_frontend_argument",
                )
            inputs.append(TorchBuffer(arguments[index]))
        output = torch.empty(
            self._imported.output_shape,
            dtype=torch.float32,
            device=self._imported.device,
        )
        self._kernel.invoke(tuple(inputs), (TorchBuffer(output),))
        return output


def compile_graph(
    graph,
    example_inputs,
    options: CompileOptions,
    cache,
) -> "object":
    """Compile every output of one normalized graph into a boxed callable."""
    from llk.runtime._ffi import compile_request

    imported = import_fx_outputs(graph, example_inputs, options)
    invokers = tuple(
        _Invoker(item, cache.get_or_compile(item.request, compile_request))
        for item in imported
    )
    from . import _compat

    def run(*arguments):
        return tuple(invoker(arguments) for invoker in invokers)

    return _compat.boxed_function(run)
