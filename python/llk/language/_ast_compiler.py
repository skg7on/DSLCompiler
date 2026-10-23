"""Capture a finite Python kernel subset into the checked frontend v1 contract."""

from __future__ import annotations

import ast
import operator
from dataclasses import dataclass, replace

import llk.language as language
from llk._errors import LLKError, UnsupportedFeatureError
from llk.frontend import (
    CastSite,
    FrontendProgram,
    KernelBinding,
    Operation,
    Region,
    SourceSpan,
    Value,
)
from llk.frontend._binding import source_of
from llk.frontend._ir import fail, freeze, integer, shape_tuple
from llk.frontend._validation import broadcast_shape

DTYPES = {
    "float32": "f32",
    "bfloat16": "bf16",
    "float16": "f16",
    "int32": "i32",
    "int8": "i8",
}
BINOPS = {
    ast.Add: ("add", operator.add),
    ast.Sub: ("sub", operator.sub),
    ast.Mult: ("mul", operator.mul),
    ast.Div: ("div", operator.truediv),
    ast.FloorDiv: ("div", operator.floordiv),
    ast.Mod: ("rem", operator.mod),
}
COMPARE = {
    ast.Eq: ("eq", operator.eq),
    ast.NotEq: ("ne", operator.ne),
    ast.Lt: ("lt", operator.lt),
    ast.LtE: ("le", operator.le),
    ast.Gt: ("gt", operator.gt),
    ast.GtE: ("ge", operator.ge),
}
MISSING = object()


def is_literal(value):
    return type(value) in (type(None), bool, int, float, str) or (
        type(value) is tuple and all(is_literal(item) for item in value)
    )


@dataclass(frozen=True)
class Pointer:
    tensor: object
    offset: object = 0


class Capture:
    def __init__(self, binding):
        self.binding = binding
        self.env = {
            **binding.captured_constants,
            **binding.constexprs,
            **{t.name: Pointer(t) for t in binding.tensors},
        }
        self.symbols = binding.symbols
        self.ops = []
        self.counter = 0
        self.used_ids = set(self.env)
        self.dependencies = {}
        self.casts = []
        self.dots = []
        self.numerical = binding.options.numerical_contract
        self.loop_depth = 0

    def span(self, node):
        return SourceSpan(
            self.binding.span.file,
            self.binding.span.line + node.lineno - 1,
            self.binding.span.column + node.col_offset,
        )

    def error(self, node, message, code="unsupported_frontend_call"):
        fail(message, self.span(node), code=code, error=UnsupportedFeatureError)

    def fresh(self, dtype, shape=()):
        while True:
            self.counter += 1
            name = f"_v{self.counter}"
            if name not in self.used_ids:
                self.used_ids.add(name)
                return Value(name, dtype, tuple(shape))

    def emit(self, node, opcode, operands=(), dtype=None, shape=(), attributes=None):
        result = self.fresh(dtype, shape) if dtype else None
        self.ops.append(
            Operation(
                opcode,
                tuple(v.id for v in operands),
                (result,) if result else (),
                attributes or {},
                span=self.span(node),
            )
        )
        if result:
            self.dependencies[result.id] = tuple(v.id for v in operands)
        return result

    def value(self, node, item, dtype=None, shape=()):
        if isinstance(item, Value):
            if dtype is not None and dtype != item.dtype:
                self.error(
                    node, "explicit conversion required for differing value types"
                )
            return item
        if type(item) not in (bool, int, float):
            self.error(node, "expected a numeric scalar or SSA value")
        inferred = (
            "i1" if type(item) is bool else "index" if type(item) is int else "f32"
        )
        dtype = dtype or inferred
        if dtype in ("f32", "bf16", "f16") and type(item) in (float, int):
            item = float(item)
        elif (
            dtype == "i1"
            and type(item) is not bool
            or dtype in ("index", "i32", "i8")
            and type(item) is not int
        ):
            self.error(node, "scalar does not match the required type")
        return self.emit(
            node, "constant", dtype=dtype, shape=shape, attributes={"value": item}
        )

    def pair(self, node, left, right):
        dtype = (
            left.dtype
            if isinstance(left, Value)
            else right.dtype
            if isinstance(right, Value)
            else None
        )
        return self.value(node, left, dtype), self.value(node, right, dtype)

    def binary(self, node, kind, left, right):
        if isinstance(left, Pointer) or isinstance(right, Pointer):
            if (
                isinstance(right, Pointer)
                and not isinstance(left, Pointer)
                and kind is ast.Add
            ):
                left, right = right, left
            if (
                not isinstance(left, Pointer)
                or isinstance(right, Pointer)
                or kind not in (ast.Add, ast.Sub)
            ):
                self.error(
                    node, "only tensor pointer plus/minus integer offsets is accepted"
                )
            offset = self.binary(node, kind, left.offset, right)
            if isinstance(offset, Value) and offset.dtype != "index":
                self.error(node, "pointer offsets must have index type")
            if not isinstance(offset, Value):
                integer(offset)
            return Pointer(left.tensor, offset)
        if kind in (ast.BitAnd, ast.BitOr):
            if type(left) is bool and type(right) is bool:
                return left & right if kind is ast.BitAnd else left | right
            a, b = self.pair(node, left, right)
            if a.dtype != "i1" or b.dtype != "i1":
                self.error(node, "bitwise mask operations require i1 values")
            literal = self.value(node, kind is ast.BitOr, "i1")
            return self.emit(
                node,
                "select",
                (a, literal, b) if kind is ast.BitOr else (a, b, literal),
                "i1",
                broadcast_shape(a.shape, b.shape),
            )
        if kind not in BINOPS:
            self.error(node, "unsupported binary operator")
        opcode, operation = BINOPS[kind]
        if not isinstance(left, Value) and not isinstance(right, Value):
            if type(left) not in (int, float) or type(right) not in (int, float):
                self.error(node, "constant arithmetic requires numeric literals")
            return freeze(operation(left, right))
        a, b = self.pair(node, left, right)
        if a.dtype != b.dtype or a.dtype == "i1" or a.dtype.startswith("ptr<"):
            self.error(node, "arithmetic requires matching numeric types")
        if kind is ast.Div and a.dtype in ("index", "i32", "i8"):
            self.error(node, "integer true division requires an explicit float cast")
        if kind in (ast.FloorDiv, ast.Mod) and a.dtype not in ("index", "i32", "i8"):
            self.error(node, "floor division/remainder requires integer operands")
        return self.emit(
            node, opcode, (a, b), a.dtype, broadcast_shape(a.shape, b.shape)
        )

    def symbol(self, node):
        if isinstance(node, ast.Name):
            return self.symbols.get(node.id, MISSING)
        if isinstance(node, ast.Attribute) and self.symbol(node.value) is language:
            return vars(language).get(node.attr, MISSING)
        return MISSING

    def call_args(self, node, names, defaults=None):
        if len(node.args) > len(names) or any(
            isinstance(a, ast.Starred) for a in node.args
        ):
            self.error(node, "incorrect call arity")
        supplied = dict(zip(names, node.args))
        for keyword in node.keywords:
            if keyword.arg not in names or keyword.arg in supplied:
                self.error(node, "duplicate, unpacked or unknown call keyword")
            supplied[keyword.arg] = keyword.value
        result = {}
        for name in names:
            if name in supplied:
                result[name] = self.expr(supplied[name])
            elif defaults is not None and name in defaults:
                result[name] = defaults[name]
            else:
                self.error(node, f"missing required {name} argument")
        return result

    def address(self, node, pointer):
        if not isinstance(pointer, Pointer):
            self.error(node, "memory operation requires a declared tensor pointer")
        offset = self.value(node, pointer.offset, "index")
        width = self.value(node, pointer.tensor.shape[1], "index")
        rows = self.emit(node, "div", (offset, width), "index", offset.shape)
        cols = self.emit(node, "rem", (offset, width), "index", offset.shape)
        base = Value(pointer.tensor.name, f"ptr<{pointer.tensor.dtype}>", ())
        return self.emit(node, "address", (base, rows, cols), base.dtype, offset.shape)

    def mask(self, node, mask, shape):
        mask = self.value(node, mask, "i1")
        if mask.shape != shape:
            if broadcast_shape(mask.shape, shape) != shape:
                self.error(node, "mask cannot broadcast to the memory tile")
            dimensions = tuple(range(len(shape) - len(mask.shape), len(shape)))
            mask = self.emit(
                node, "broadcast", (mask,), "i1", shape, {"dimensions": dimensions}
            )
        return mask

    def cast(self, node, item, dtype, rounding):
        item = self.value(node, item)
        if dtype not in DTYPES or rounding != "rtne":
            self.error(node, "unsupported explicit dtype/rounding")
        result = self.emit(
            node, "cast", (item,), DTYPES[dtype], item.shape, {"rounding": rounding}
        )
        self.casts.append(CastSite(result.id, item.dtype, result.dtype, rounding))
        return result

    def activation(self, node, item, approximation):
        item = self.value(node, item)
        policy = self.numerical
        if item.dtype != "f32" or approximation not in ("exact", "native"):
            self.error(
                node, "SiLU requires f32 and an explicit supported approximation"
            )
        if (
            approximation != "exact"
            and policy.activation_approximation != approximation
        ):
            self.error(node, "relaxed activation requires an explicit numerical policy")
        if policy.activation_expression not in (
            "none",
            "x/(1+exp(-x))",
        ) or policy.activation_approximation not in ("none", approximation):
            self.error(node, "source activation contradicts the numerical policy")
        self.numerical = replace(
            policy,
            activation_expression="x/(1+exp(-x))",
            activation_approximation=approximation,
        )
        return self.emit(
            node,
            "silu",
            (item,),
            item.dtype,
            item.shape,
            {"approximation": approximation},
        )

    def call(self, node):
        symbol = self.symbol(node.func)
        if symbol is language.program_id:
            args = self.call_args(node, ("axis",))
            integer(args["axis"], minimum=0)
            if args["axis"] > 2:
                self.error(node, "program axis must be 0..2")
            return self.emit(node, "program_id", dtype="index", attributes=args)
        if symbol is language.arange:
            args = self.call_args(node, ("start", "end"))
            start, end = integer(args["start"]), integer(args["end"])
            if end <= start:
                self.error(node, "arange requires a positive static extent")
            return self.emit(
                node, "range", dtype="index", shape=(end - start,), attributes=args
            )
        if symbol in (language.load, language.store):
            if symbol is language.load:
                args = self.call_args(
                    node, ("pointer", "mask", "other"), {"mask": True}
                )
            else:
                args = self.call_args(
                    node, ("pointer", "value", "mask"), {"mask": True}
                )
            pointer = self.address(node, args["pointer"])
            mask = self.mask(node, args["mask"], pointer.shape)
            dtype = pointer.dtype[4:-1]
            if symbol is language.load:
                other = self.value(node, args["other"], dtype)
                if other.shape not in ((), pointer.shape):
                    self.error(
                        node, "padding must be scalar or have the exact load tile shape"
                    )
                return self.emit(
                    node, "load", (pointer, mask, other), dtype, pointer.shape
                )
            item = self.value(node, args["value"], dtype)
            if item.shape != pointer.shape:
                self.error(node, "store value must have the exact memory tile shape")
            self.emit(node, "store", (pointer, item, mask))
            return None
        if symbol is language.zeros:
            args = self.call_args(node, ("shape", "dtype"), {"dtype": language.float32})
            shape = shape_tuple(args["shape"])
            if len(shape) > 2 or args["dtype"] not in DTYPES:
                self.error(
                    node, "zeros requires a supported dtype and rank at most two"
                )
            dtype = DTYPES[args["dtype"]]
            return self.value(
                node, 0.0 if dtype in ("f32", "bf16", "f16") else 0, dtype, shape
            )
        if symbol is language.dot:
            args = self.call_args(node, ("a", "b", "acc"), {"acc": None})
            a, b = self.value(node, args["a"]), self.value(node, args["b"])
            if (
                len(a.shape) != 2
                or len(b.shape) != 2
                or a.shape[1] != b.shape[0]
                or a.dtype != b.dtype
                or a.dtype not in ("f32", "bf16")
            ):
                self.error(node, "dot requires compatible rank-two f32/bf16 operands")
            shape = (a.shape[0], b.shape[1])
            acc = (
                self.value(node, 0.0, "f32", shape)
                if args["acc"] is None
                else self.value(node, args["acc"], "f32")
            )
            if acc.shape != shape:
                self.error(node, "dot accumulator shape mismatch")
            result = self.emit(node, "dot", (a, b, acc), "f32", shape)
            self.dots.append((result.id, acc.id))
            return result
        if symbol is language.cast:
            args = self.call_args(
                node, ("value", "dtype", "rounding"), {"rounding": "rtne"}
            )
            return self.cast(node, args["value"], args["dtype"], args["rounding"])
        if isinstance(node.func, ast.Attribute) and node.func.attr == "to":
            item = self.expr(node.func.value)
            if isinstance(item, Value):
                args = self.call_args(node, ("dtype", "rounding"), {"rounding": "rtne"})
                return self.cast(node, item, args["dtype"], args["rounding"])
        if symbol is language.where:
            args = self.call_args(node, ("condition", "x", "y"))
            a, b = self.pair(node, args["x"], args["y"])
            mask = self.value(node, args["condition"], "i1")
            return self.emit(
                node,
                "select",
                (mask, a, b),
                a.dtype,
                broadcast_shape(mask.shape, a.shape, b.shape),
            )
        if symbol is language.cdiv:
            args = self.call_args(node, ("a", "b"))
            a, b = args["a"], args["b"]
            if type(a) is int and type(b) is int:
                if b <= 0:
                    self.error(node, "cdiv divisor must be positive")
                return -(-a // b)
            a, b = self.pair(node, a, b)
            if a.dtype != "index":
                self.error(node, "cdiv requires index operands")
            return self.emit(
                node, "ceildiv", (a, b), "index", broadcast_shape(a.shape, b.shape)
            )
        if symbol is language.silu:
            args = self.call_args(
                node, ("value", "approximation"), {"approximation": "exact"}
            )
            return self.activation(node, args["value"], args["approximation"])
        self.error(node, "unsupported frontend call")

    def expr(self, node):
        try:
            if isinstance(node, ast.Constant):
                if type(node.value) in (type(None), bool, int, float, str):
                    return freeze(node.value)
            elif isinstance(node, ast.Name):
                if node.id in self.env:
                    return self.env[node.id]
                self.error(node, f"undefined value: {node.id}")
            elif isinstance(node, ast.Tuple):
                return tuple(self.expr(item) for item in node.elts)
            elif isinstance(node, ast.Attribute):
                symbol = self.symbol(node)
                if type(symbol) is str and symbol in DTYPES:
                    return symbol
                self.error(node, "unsupported attribute access")
            elif isinstance(node, ast.Call):
                return self.call(node)
            elif isinstance(node, ast.BinOp):
                # Exact recognized source expression, with no general exp opcode.
                denominator = node.right
                if (
                    isinstance(node.op, ast.Div)
                    and isinstance(denominator, ast.BinOp)
                    and isinstance(denominator.op, ast.Add)
                    and isinstance(denominator.left, ast.Constant)
                    and denominator.left.value == 1
                ):
                    exp = denominator.right
                    if (
                        isinstance(exp, ast.Call)
                        and self.symbol(exp.func) is language.exp
                        and len(exp.args) == 1
                        and not exp.keywords
                    ):
                        negative = exp.args[0]
                        if (
                            isinstance(negative, ast.UnaryOp)
                            and isinstance(negative.op, ast.USub)
                            and ast.dump(node.left) == ast.dump(negative.operand)
                        ):
                            return self.activation(node, self.expr(node.left), "exact")
                return self.binary(
                    node, type(node.op), self.expr(node.left), self.expr(node.right)
                )
            elif isinstance(node, ast.UnaryOp):
                if (
                    isinstance(node.op, ast.USub)
                    and isinstance(node.operand, ast.Constant)
                    and type(node.operand.value) in (int, float)
                ):
                    # Negate before range checking so the signed minimum is a
                    # valid literal, and retain the sign bit of floating zero.
                    return freeze(operator.neg(node.operand.value))
                item = self.expr(node.operand)
                if isinstance(node.op, ast.USub):
                    if type(item) in (int, float):
                        return freeze(operator.neg(item))
                    if isinstance(item, Value) and item.dtype in ("f32", "bf16", "f16"):
                        return self.emit(node, "neg", (item,), item.dtype, item.shape)
                    return self.binary(node, ast.Sub, 0, item)
                if isinstance(node.op, ast.UAdd) and type(item) in (int, float):
                    return +item
                if isinstance(node.op, (ast.Invert, ast.Not)):
                    if isinstance(node.op, ast.Not):
                        if not is_literal(item):
                            self.error(
                                node,
                                "not requires a constexpr condition",
                                "unsupported_frontend_control_flow",
                            )
                        return not item
                    if type(item) is int:
                        return integer(~item)
                    if not isinstance(item, Value):
                        self.error(node, "mask inversion requires an i1 SSA value")
                    item = self.value(node, item, "i1")
                    return self.emit(
                        node,
                        "select",
                        (item, self.value(node, False), self.value(node, True)),
                        "i1",
                        item.shape,
                    )
            elif (
                isinstance(node, ast.Compare)
                and len(node.ops) == 1
                and type(node.ops[0]) in COMPARE
            ):
                a, b = self.expr(node.left), self.expr(node.comparators[0])
                predicate, comparison = COMPARE[type(node.ops[0])]
                if is_literal(a) and is_literal(b):
                    return comparison(a, b)
                a, b = self.pair(node, a, b)
                return self.emit(
                    node,
                    "compare",
                    (a, b),
                    "i1",
                    broadcast_shape(a.shape, b.shape),
                    {"predicate": predicate},
                )
            elif isinstance(node, ast.Subscript):
                item = self.expr(node.value)
                if (
                    type(item) is tuple
                    and isinstance(node.slice, ast.Constant)
                    and type(node.slice.value) is int
                ):
                    return item[node.slice.value]
                item = self.value(node, item)
                indices = (
                    node.slice.elts
                    if isinstance(node.slice, ast.Tuple)
                    else [node.slice]
                )
                if len(indices) != len(item.shape) + sum(
                    isinstance(x, ast.Constant) and x.value is None for x in indices
                ):
                    self.error(node, "indexing only supports full slices and new axes")
                for axis, index in enumerate(indices):
                    if isinstance(index, ast.Constant) and index.value is None:
                        shape = item.shape[:axis] + (1,) + item.shape[axis:]
                        item = self.emit(
                            node,
                            "expand_dims",
                            (item,),
                            item.dtype,
                            shape,
                            {"axis": axis},
                        )
                    elif not isinstance(index, ast.Slice) or any(
                        v is not None for v in (index.lower, index.upper, index.step)
                    ):
                        self.error(
                            node, "indexing only supports full slices and new axes"
                        )
                return item
            self.error(node, f"unsupported expression: {type(node).__name__}")
        except LLKError as error:
            if error.file is None:
                span = self.span(node)
                error.file, error.line, error.column = span.file, span.line, span.column
            raise
        except (
            ValueError,
            TypeError,
            OverflowError,
            ZeroDivisionError,
            IndexError,
        ) as error:
            self.error(node, f"invalid expression: {error}")

    def depends(self, value_id, target, seen=None):
        if value_id == target:
            return True
        seen = set() if seen is None else seen
        if value_id in seen:
            return False
        seen.add(value_id)
        return any(
            self.depends(parent, target, seen)
            for parent in self.dependencies.get(value_id, ())
        )

    def loop(self, node):
        if (
            not isinstance(node.target, ast.Name)
            or not isinstance(node.iter, ast.Call)
            or self.symbol(node.iter.func) is not range
            or node.orelse
            or node.iter.keywords
        ):
            self.error(
                node,
                "only canonical counted range loops are supported",
                "unsupported_frontend_control_flow",
            )
        bounds = [self.expr(arg) for arg in node.iter.args]
        if len(bounds) == 1:
            lower, upper, step = 0, bounds[0], 1
        elif len(bounds) in (2, 3):
            lower, upper = bounds[:2]
            step = bounds[2] if len(bounds) == 3 else 1
        else:
            self.error(node, "range requires one to three bounds")
        integer(step, "loop step", minimum=1)
        lower, upper = (
            self.value(node, lower, "index"),
            self.value(node, upper, "index"),
        )
        if lower.shape or upper.shape:
            self.error(node, "loop bounds must be scalar indices")
        outer = dict(self.env)
        if node.target.id in outer:
            self.error(
                node,
                "loop induction variable must have a fresh binding",
                "unsupported_frontend_control_flow",
            )
        assigned = sorted(
            (
                target
                for statement in ast.walk(node)
                if isinstance(statement, (ast.Assign, ast.AugAssign))
                for target in (
                    statement.targets
                    if isinstance(statement, ast.Assign)
                    else [statement.target]
                )
                if isinstance(target, ast.Name)
            ),
            key=lambda x: (x.lineno, x.col_offset),
        )
        names = list(
            dict.fromkeys(target.id for target in assigned if target.id in outer)
        )
        if not names or any(
            not isinstance(outer[name], Value)
            or outer[name].dtype != "f32"
            or len(outer[name].shape) != 2
            for name in names
        ):
            self.error(
                node,
                "counted loops require typed dot accumulator state",
                "unsupported_frontend_control_flow",
            )
        initial = [outer[name] for name in names]
        iv = self.fresh("index")
        if node.target.id in names:
            self.error(node, "loop induction variable cannot overwrite carried state")
        arguments = [self.fresh(v.dtype, v.shape) for v in initial]
        for argument, init in zip(arguments, initial):
            self.dependencies[argument.id] = (init.id,)
        self.env[node.target.id] = iv
        self.env.update(zip(names, arguments))
        parent_ops = self.ops
        self.ops = []
        dot_start = len(self.dots)
        self.loop_depth += 1
        self.statements(node.body)
        self.loop_depth -= 1
        yielded = [self.value(node, self.env[name]) for name in names]
        for argument, result in zip(arguments, yielded):
            if (result.dtype, result.shape) != (
                argument.dtype,
                argument.shape,
            ) or not any(
                self.depends(acc, argument.id) and self.depends(result.id, dot)
                for dot, acc in self.dots[dot_start:]
            ):
                self.error(
                    node,
                    "loop yield must preserve a recognized dot accumulator",
                    "unsupported_frontend_control_flow",
                )
        self.emit(node, "yield", yielded)
        region = Region((iv, *arguments), tuple(self.ops))
        self.ops = parent_ops
        self.env = outer
        results = tuple(self.fresh(v.dtype, v.shape) for v in initial)
        self.ops.append(
            Operation(
                "counted_loop",
                (lower.id, upper.id, *(v.id for v in initial)),
                results,
                {"step": step},
                (region,),
                self.span(node),
            )
        )
        for name, result, value in zip(names, results, yielded):
            self.env[name] = result
            self.dependencies[result.id] = (value.id,)

    def statements(self, statements):
        for index, node in enumerate(statements):
            try:
                if (
                    isinstance(node, ast.Assign)
                    and len(node.targets) == 1
                    and isinstance(node.targets[0], ast.Name)
                ):
                    item = self.expr(node.value)
                    if item is None:
                        self.error(node, "void operation cannot be assigned")
                    self.env[node.targets[0].id] = item
                elif isinstance(node, ast.AugAssign) and isinstance(
                    node.target, ast.Name
                ):
                    if node.target.id not in self.env:
                        self.error(node, "assignment requires a defined value")
                    self.env[node.target.id] = self.binary(
                        node,
                        type(node.op),
                        self.env[node.target.id],
                        self.expr(node.value),
                    )
                elif isinstance(node, ast.Expr):
                    if (
                        isinstance(node.value, ast.Constant)
                        and isinstance(node.value.value, str)
                        and index == 0
                    ):
                        continue
                    if not isinstance(node.value, ast.Call):
                        self.error(
                            node, "only recognized calls can be expression statements"
                        )
                    self.expr(node.value)
                elif isinstance(node, ast.Pass):
                    continue
                elif isinstance(node, ast.Return):
                    if (
                        self.loop_depth
                        or node.value is not None
                        and not (
                            isinstance(node.value, ast.Constant)
                            and node.value.value is None
                        )
                    ):
                        self.error(
                            node,
                            "pointer-output kernels may only return None outside loops",
                            "unsupported_frontend_control_flow",
                        )
                    return True
                elif isinstance(node, ast.If):
                    condition = self.expr(node.test)
                    if not is_literal(condition):
                        self.error(
                            node.test,
                            "runtime branches are unsupported",
                            "unsupported_frontend_control_flow",
                        )
                    if self.statements(node.body if condition else node.orelse):
                        return True
                elif isinstance(node, ast.For):
                    self.loop(node)
                else:
                    self.error(
                        node,
                        f"unsupported statement: {type(node).__name__}",
                        "unsupported_frontend_control_flow",
                    )
            except LLKError as error:
                if error.file is None:
                    span = self.span(node)
                    error.file, error.line, error.column = (
                        span.file,
                        span.line,
                        span.column,
                    )
                raise
        return False


def compile_kernel(fn, binding: KernelBinding) -> FrontendProgram:
    """Capture with a frozen argument binding; never execute the function body."""
    source, span = source_of(fn)
    if (
        not isinstance(binding, KernelBinding)
        or source != binding.source
        or span != binding.span
    ):
        fail(
            "capture requires the binding for this exact function source",
            span,
            code="invalid_argument_binding",
        )
    try:
        tree = ast.parse(source)
        definitions = [
            n
            for n in tree.body
            if isinstance(n, ast.FunctionDef) and n.name == fn.__name__
        ]
        if len(definitions) != 1:
            fail(
                "expected one ordinary kernel function",
                span,
                code="unsupported_frontend_call",
            )
        definition = definitions[0]
        visitor = Capture(binding)
        visitor.statements(definition.body)
        visitor.emit(definition, "return")
        if (
            binding.options.numerical_contract.cast_sites
            and binding.options.numerical_contract.cast_sites != tuple(visitor.casts)
        ):
            fail("explicit cast policy contradicts captured source", span)
        numerical = replace(visitor.numerical, cast_sites=tuple(visitor.casts))
        return FrontendProgram(
            definition.name,
            binding.tensors,
            binding.constexprs,
            binding.launch,
            tuple(visitor.ops),
            numerical,
            source=source,
            captured_constants=binding.captured_constants,
        )
    except LLKError as error:
        if error.file is None:
            error.file, error.line, error.column = span.file, span.line, span.column
        raise
    except (SyntaxError, ValueError, TypeError, OverflowError) as error:
        fail(f"invalid kernel source: {error}", span, code="unsupported_frontend_call")
