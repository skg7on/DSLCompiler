"""Typed source fixtures shared with later importer acceptance tasks."""

from llk.frontend import (
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
)

SPAN = SourceSpan("add_kernel.py", 1, 0)


def op(code, operands=(), results=(), attributes=None, regions=()):
    return Operation(code, operands, results, attributes or {}, regions, SPAN)


def make_add_program(shape=(32, 64), block=(8, 16)):
    bm, bn = block
    v = lambda name, dtype="index", dims=(): Value(name, dtype, dims)
    tile = (bm, bn)
    operations = (
        op("program_id", results=(v("pm"),), attributes={"axis": 0}),
        op("program_id", results=(v("pn"),), attributes={"axis": 1}),
        op("constant", results=(v("bm"),), attributes={"value": bm}),
        op("constant", results=(v("bn"),), attributes={"value": bn}),
        op("mul", ("pm", "bm"), (v("rm"),)),
        op("mul", ("pn", "bn"), (v("cn"),)),
        op("range", results=(v("ri", dims=(bm,)),), attributes={"start": 0, "end": bm}),
        op("range", results=(v("ci", dims=(bn,)),), attributes={"start": 0, "end": bn}),
        op("add", ("rm", "ri"), (v("r", dims=(bm,)),)),
        op("add", ("cn", "ci"), (v("c", dims=(bn,)),)),
        op("expand_dims", ("r",), (v("rows", dims=(bm, 1)),), {"axis": 1}),
        op("expand_dims", ("c",), (v("cols", dims=(1, bn)),), {"axis": 0}),
        op("constant", results=(v("mask", "i1", tile),), attributes={"value": True}),
        op("constant", results=(v("zero", "f32"),), attributes={"value": 0.0}),
        op("address", ("a", "rows", "cols"), (v("pa", "ptr<f32>", tile),)),
        op("load", ("pa", "mask", "zero"), (v("va", "f32", tile),)),
        op("address", ("b", "rows", "cols"), (v("pb", "ptr<f32>", tile),)),
        op("load", ("pb", "mask", "zero"), (v("vb", "f32", tile),)),
        op("add", ("va", "vb"), (v("sum", "f32", tile),)),
        op("address", ("out", "rows", "cols"), (v("po", "ptr<f32>", tile),)),
        op("store", ("po", "sum", "mask")),
        op("return"),
    )
    return FrontendProgram(
        entry="add",
        tensors=(
            TensorSpec("a", shape, "f32", (shape[1], 1), "input", "read"),
            TensorSpec("b", shape, "f32", (shape[1], 1), "input", "read"),
            TensorSpec("out", shape, "f32", (shape[1], 1), "output", "write"),
        ),
        constexprs={"BLOCK_M": bm, "BLOCK_N": bn},
        launch=LaunchSpec((shape[0] // bm, shape[1] // bn, 1), (bm, bn, 1)),
        operations=operations,
        source="def add(a, b, out): # typed add\n    pass\n",
    )


def make_loop_program():
    value = lambda name, dtype="f32": Value(name, dtype, ())
    cast_sites = (
        CastSite("rounded", "f32", "bf16", "rtne"),
        CastSite("restored", "bf16", "f32", "rtne"),
    )
    inner = op(
        "counted_loop",
        ("lo", "hi", "acc"),
        (value("inner_result"),),
        {"step": 1},
        (
            Region(
                (value("j", "index"), value("inner_acc")),
                (
                    op(
                        "cast",
                        ("inner_acc",),
                        (value("rounded", "bf16"),),
                        {"rounding": "rtne"},
                    ),
                    op(
                        "cast", ("rounded",), (value("restored"),), {"rounding": "rtne"}
                    ),
                    op("yield", ("restored",)),
                ),
            ),
        ),
    )
    outer = op(
        "counted_loop",
        ("lo", "hi", "zero"),
        (value("final"),),
        {"step": 1},
        (
            Region(
                (value("i", "index"), value("acc")),
                (inner, op("yield", ("inner_result",))),
            ),
        ),
    )
    return FrontendProgram(
        entry="nested",
        tensors=(),
        constexprs={},
        launch=LaunchSpec((1, 1, 1), (1, 1, 1)),
        operations=(
            op("constant", results=(value("lo", "index"),), attributes={"value": 0}),
            op("constant", results=(value("hi", "index"),), attributes={"value": 2}),
            op("constant", results=(value("zero"),), attributes={"value": 0.0}),
            outer,
            op("return"),
        ),
        numerical_contract=NumericalContract(cast_sites=cast_sites),
        source="nested counted K loops",
    )


def request(program=None, **options):
    program = program or make_add_program()
    options.setdefault("numerical_contract", program.numerical_contract)
    if "config_root" not in options:
        options.update(config_root=".", config_files={})
    return CompileRequest(program, CompileOptions(**options))
