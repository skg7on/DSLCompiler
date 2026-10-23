"""Schema-v1 SSA/type/effect/region checks, independent of native compilation."""

from ._ir import CastSite, Value, fail, integer, shape_tuple

# Exact attribute and operand/result schemas; no opcode silently passes through.
SCHEMAS = {
    "constant": (0, 1, ("value",)),
    "program_id": (0, 1, ("axis",)),
    "range": (0, 1, ("start", "end")),
    "address": (3, 1, ()),
    "load": (3, 1, ()),
    "store": (3, 0, ()),
    "add": (2, 1, ()),
    "sub": (2, 1, ()),
    "mul": (2, 1, ()),
    "div": (2, 1, ()),
    "rem": (2, 1, ()),
    "ceildiv": (2, 1, ()),
    "compare": (2, 1, ("predicate",)),
    "select": (3, 1, ()),
    "broadcast": (1, 1, ("dimensions",)),
    "expand_dims": (1, 1, ("axis",)),
    "cast": (1, 1, ("rounding",)),
    "dot": (3, 1, ()),
    "silu": (1, 1, ("approximation",)),
    "return": (0, 0, ()),
    "yield": (None, 0, ()),
    "counted_loop": (None, None, ("step",)),
}


def broadcast_shape(*shapes):
    rank = max((len(s) for s in shapes), default=0)
    result = [1] * rank
    for shape in shapes:
        padded = (1,) * (rank - len(shape)) + tuple(shape)
        for i, size in enumerate(padded):
            if result[i] != 1 and size != 1 and size != result[i]:
                fail("incompatible broadcast shapes")
            result[i] = max(result[i], size)
    return shape_tuple(result)


def validate_program(program):
    tensors = {t.name: t for t in program.tensors}
    if len(tensors) != len(program.tensors):
        fail("tensor names must be unique")
    for tensor in program.tensors:
        expected = (
            (program.numerical_contract.storage_dtype,)
            if tensor.direction == "input"
            else (
                program.numerical_contract.storage_dtype,
                program.numerical_contract.accumulator_dtype,
            )
        )
        if tensor.dtype not in expected:
            fail("tensor dtype disagrees with the numerical storage contract")
    visible = {t.name: Value(t.name, f"ptr<{t.dtype}>", ()) for t in program.tensors}
    defined = set(visible)
    origins = dict(tensors)
    casts = []

    def define(value, scope, span):
        if value.id in defined:
            fail(f"duplicate value id: {value.id}", span)
        defined.add(value.id)
        scope[value.id] = value

    def same(actual, dtype, shape, span):
        if actual.dtype != dtype or actual.shape != tuple(shape):
            fail("result type/shape does not match the operation", span)

    def walk(operations, scope, *, region=False):
        if not operations or operations[-1].opcode != ("yield" if region else "return"):
            fail("operation sequence requires an explicit terminal yield/void return")
        for i, operation in enumerate(operations):
            code, attrs, span = operation.opcode, operation.attributes, operation.span
            if code not in SCHEMAS:
                fail(
                    f"unsupported frontend opcode: {code}",
                    span,
                    code="unsupported_frontend_opcode",
                )
            operand_count, result_count, attr_names = SCHEMAS[code]
            if (
                (operand_count is not None and len(operation.operands) != operand_count)
                or (result_count is not None and len(operation.results) != result_count)
                or set(attrs) != set(attr_names)
            ):
                fail(f"incorrect {code} arity or attributes", span)
            if (code == "counted_loop" and len(operation.regions) != 1) or (
                code != "counted_loop" and operation.regions
            ):
                fail("only counted_loop may contain one typed region", span)
            if code in ("return", "yield") and (
                i != len(operations) - 1 or code != ("yield" if region else "return")
            ):
                fail("misplaced region/function terminator", span)
            operands = []
            for name in operation.operands:
                if name not in scope:
                    fail(f"undefined or escaping value: {name}", span)
                operands.append(scope[name])
            results = operation.results
            result = results[0] if results else None
            if code == "constant":
                value = attrs["value"]
                if result.dtype == "i1":
                    valid = type(value) is bool
                elif result.dtype in ("index", "i32", "i8"):
                    valid = type(value) is int
                    if valid:
                        integer(value)
                        width = {"index": 64, "i32": 32, "i8": 8}[result.dtype]
                        valid = -(2 ** (width - 1)) <= value < 2 ** (width - 1)
                else:
                    valid = result.dtype in ("f32", "bf16", "f16") and type(value) in (
                        float,
                        int,
                    )
                if not valid:
                    fail("constant value does not match its declared dtype", span)
            elif code == "program_id":
                axis = integer(attrs["axis"], "program axis", minimum=0)
                if axis > 2:
                    fail("program axis must be 0..2", span)
                same(result, "index", (), span)
            elif code == "range":
                start, end = integer(attrs["start"]), integer(attrs["end"])
                if end <= start or end - start > 2**63 - 1:
                    fail("range requires a positive signed-index extent", span)
                same(result, "index", (end - start,), span)
            elif code in ("add", "sub", "mul", "div", "rem", "ceildiv", "compare"):
                a, b = operands
                if a.dtype != b.dtype or a.dtype.startswith("ptr<") or a.dtype == "i1":
                    fail("arithmetic requires matching numeric operands", span)
                if code in ("rem", "ceildiv") and a.dtype not in ("index", "i32", "i8"):
                    fail("integer operation requires integer operands", span)
                dims = broadcast_shape(a.shape, b.shape)
                if code == "compare":
                    if attrs["predicate"] not in ("eq", "ne", "lt", "le", "gt", "ge"):
                        fail("unsupported comparison predicate", span)
                    same(result, "i1", dims, span)
                else:
                    same(result, a.dtype, dims, span)
            elif code == "select":
                mask, a, b = operands
                if (
                    mask.dtype != "i1"
                    or a.dtype != b.dtype
                    or a.dtype.startswith("ptr<")
                ):
                    fail(
                        "select requires an i1 condition and matching numeric arms",
                        span,
                    )
                same(
                    result, a.dtype, broadcast_shape(mask.shape, a.shape, b.shape), span
                )
            elif code == "expand_dims":
                a = operands[0]
                axis = integer(attrs["axis"], "expand axis", minimum=0)
                if axis > len(a.shape) or len(a.shape) >= 2:
                    fail("invalid expand_dims axis/rank", span)
                same(result, a.dtype, a.shape[:axis] + (1,) + a.shape[axis:], span)
            elif code == "broadcast":
                a = operands[0]
                dimensions = attrs["dimensions"]
                if (
                    not isinstance(dimensions, tuple)
                    or len(dimensions) != len(a.shape)
                    or any(type(d) is not int for d in dimensions)
                ):
                    fail("broadcast dimensions must map each input axis", span)
                if tuple(sorted(set(dimensions))) != dimensions or any(
                    d < 0 or d >= len(result.shape) for d in dimensions
                ):
                    fail("broadcast dimensions must be increasing valid axes", span)
                if any(
                    a.shape[i] not in (1, result.shape[d])
                    for i, d in enumerate(dimensions)
                ):
                    fail("incompatible broadcast dimension extents", span)
                same(result, a.dtype, result.shape, span)
            elif code == "address":
                base, rows, cols = operands
                if (
                    base.id not in tensors
                    or base.shape
                    or rows.dtype != "index"
                    or cols.dtype != "index"
                ):
                    fail(
                        "address requires a named base tensor and typed row/column indices",
                        span,
                    )
                same(result, base.dtype, broadcast_shape(rows.shape, cols.shape), span)
                origins[result.id] = tensors[base.id]
            elif code in ("load", "store"):
                pointer = operands[0]
                if pointer.id not in origins or not pointer.dtype.startswith("ptr<"):
                    fail("memory operations require a declared tensor address", span)
                tensor = origins[pointer.id]
                dtype = pointer.dtype[4:-1]
                if code == "load":
                    mask, other = operands[1:]
                    if (
                        tensor.effect != "read"
                        or other.dtype != dtype
                        or other.shape not in ((), pointer.shape)
                    ):
                        fail("load effect or padding type mismatch", span)
                    same(result, dtype, pointer.shape, span)
                else:
                    value, mask = operands[1:]
                    if (
                        tensor.effect != "write"
                        or value.dtype != dtype
                        or value.shape != pointer.shape
                    ):
                        fail("store effect or value type mismatch", span)
                if mask.dtype != "i1" or mask.shape != pointer.shape:
                    fail("mask must have exact i1 tile shape", span)
            elif code == "cast":
                a = operands[0]
                if (
                    a.dtype.startswith("ptr<")
                    or result.dtype.startswith("ptr<")
                    or a.shape != result.shape
                    or attrs["rounding"] != "rtne"
                ):
                    fail("invalid cast shape, dtype or rounding", span)
                casts.append(
                    CastSite(result.id, a.dtype, result.dtype, attrs["rounding"])
                )
            elif code == "dot":
                a, b, acc = operands
                if (
                    len(a.shape) != 2
                    or len(b.shape) != 2
                    or a.dtype != b.dtype
                    or a.dtype not in ("f32", "bf16")
                    or acc.dtype != "f32"
                    or a.shape[1] != b.shape[0]
                    or acc.shape != (a.shape[0], b.shape[1])
                ):
                    fail(
                        "dot requires compatible rank-two inputs and f32 accumulator",
                        span,
                    )
                same(result, "f32", acc.shape, span)
            elif code == "silu":
                a = operands[0]
                numerical = program.numerical_contract
                if (
                    a.dtype != "f32"
                    or numerical.activation_expression != "x/(1+exp(-x))"
                    or attrs["approximation"] != numerical.activation_approximation
                ):
                    fail("SiLU must match its explicit numerical contract", span)
                same(result, a.dtype, a.shape, span)
            elif code == "counted_loop":
                if (
                    len(operands) < 2
                    or len(results) != len(operands) - 2
                    or any(v.dtype != "index" or v.shape for v in operands[:2])
                ):
                    fail(
                        "counted_loop requires index bounds and one result per init value",
                        span,
                    )
                integer(attrs["step"], "loop step", minimum=1)
                body = operation.regions[0]
                if len(body.arguments) != len(results) + 1:
                    fail("loop body requires induction and typed iter arguments", span)
                same(body.arguments[0], "index", (), span)
                local = dict(scope)
                for arg in body.arguments:
                    define(arg, local, span)
                for initial, arg, res in zip(operands[2:], body.arguments[1:], results):
                    if initial.dtype.startswith("ptr<"):
                        fail("pointer loop-carried state is unsupported", span)
                    same(arg, initial.dtype, initial.shape, span)
                    same(res, initial.dtype, initial.shape, span)
                yielded = walk(body.operations, local, region=True)
                if len(yielded) != len(results):
                    fail("loop yield/result arity mismatch", span)
                for value, res in zip(yielded, results):
                    same(value, res.dtype, res.shape, span)
            for res in results:
                define(res, scope, span)
            if code == "yield":
                return operands
        return ()

    walk(program.operations, visible)
    if tuple(casts) != program.numerical_contract.cast_sites:
        fail("ordered explicit cast sites must exactly match the numerical contract")
