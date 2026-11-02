"""PF13: strict import of normalized ATen FX graphs into frontend requests.

These are unit tests for the graph-to-``CompileRequest`` boundary. They never
reach native code: every graph is recognized from its ATen targets and example
tensor metadata, and every unsupported graph must identify the offending FX
node instead of falling back to graph execution.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode

from llk.frontend import CompileOptions, serialize_request
from llk.torch._fx_importer import (
    UnsupportedFXError,
    import_fx,
    import_fx_requests,
)

ATEN = torch.ops.aten


def fake_inputs(*shapes, requires_grad=False):
    with FakeTensorMode():
        return tuple(
            torch.empty(shape, dtype=torch.float32, requires_grad=requires_grad)
            for shape in shapes
        )


def _rename_like_dynamo(graph):
    """Use the clean node names Dynamo gives captured ATen calls."""
    used = set()
    for node in graph.nodes:
        if node.op != "call_function" or not isinstance(
            node.target, torch._ops.OpOverload
        ):
            continue
        base = str(node.target).split(".")[1]
        name, suffix = base, 1
        while name in used:
            name = f"{base}_{suffix}"
            suffix += 1
        node.name = name
        used.add(name)


def build_graph(inputs, build):
    """Build a GraphModule whose call_function targets are explicit ATen ops."""
    graph = torch.fx.Graph()
    placeholders = [graph.placeholder(name) for name in inputs]
    roots = build(graph, *placeholders)
    if not isinstance(roots, tuple):
        roots = (roots,)
    graph.output(tuple(roots))
    _rename_like_dynamo(graph)
    return torch.fx.GraphModule({}, graph)


def options():
    return CompileOptions(config_files={})


def test_imports_add_and_multiply_elementwise_graphs():
    for target, opcode in (
        (ATEN.add.Tensor, "add"),
        (ATEN.mul.Tensor, "mul"),
    ):
        gm = build_graph(("a", "b"), lambda g, a, b: g.call_function(target, (a, b)))
        request = import_fx(gm, fake_inputs((4, 5), (4, 5)), options())
        program = request.program
        assert [t.name for t in program.tensors] == ["a", "b", "output"]
        assert [t.direction for t in program.tensors] == ["input", "input", "output"]
        assert all(t.shape == (4, 5) and t.dtype == "f32" for t in program.tensors)
        assert program.launch.block == (8, 16, 1)
        assert program.launch.grid == (1, 1, 1)
        assert opcode in {op.opcode for op in program.operations}
        assert program.operations[-1].opcode == "return"
        assert request.options.numerical_contract == program.numerical_contract


def test_imports_matmul_as_a_counted_contraction():
    gm = build_graph(
        ("a", "b"), lambda g, a, b: g.call_function(ATEN.mm.default, (a, b))
    )
    request = import_fx(gm, fake_inputs((4, 6), (6, 5)), options())
    program = request.program
    assert [t.name for t in program.tensors] == ["a", "b", "output"]
    assert [t.shape for t in program.tensors] == [(4, 6), (6, 5), (4, 5)]
    assert program.launch.block == (4, 5, 1)
    assert program.launch.grid == (1, 1, 1)
    loop = next(op for op in program.operations if op.opcode == "counted_loop")
    assert "dot" in {op.opcode for op in loop.regions[0].operations}
    assert program.operations[-1].opcode == "return"
    assert program.numerical_contract.activation_expression == "none"


def test_imports_fused_swiglu_with_the_exact_activation_contract():
    gm = build_graph(
        ("x", "wg", "wu"),
        lambda g, x, wg, wu: g.call_function(
            ATEN.mul.Tensor,
            (
                g.call_function(
                    ATEN.silu.default,
                    (g.call_function(ATEN.mm.default, (x, wg)),),
                ),
                g.call_function(ATEN.mm.default, (x, wu)),
            ),
        ),
    )
    request = import_fx(gm, fake_inputs((3, 6), (6, 4), (6, 4)), options())
    program = request.program
    assert [t.name for t in program.tensors] == ["x", "wg", "wu", "output"]
    assert program.launch.block == (3, 4, 1)
    assert program.launch.grid == (1, 1, 1)
    assert "silu" in {op.opcode for op in program.operations}
    assert program.numerical_contract.activation_expression == "x/(1+exp(-x))"
    assert program.numerical_contract.activation_approximation == "exact"


def test_request_identity_is_deterministic():
    gm = build_graph(
        ("a", "b"), lambda g, a, b: g.call_function(ATEN.add.Tensor, (a, b))
    )
    inputs = fake_inputs((4, 5), (4, 5))
    first = import_fx(gm, inputs, options())
    second = import_fx(gm, inputs, options())
    assert serialize_request(first) == serialize_request(second)
    assert first.program.source == second.program.source
    assert first.program.entry == second.program.entry


def test_unsupported_node_identifies_sin():
    gm = build_graph(
        ("a", "b"),
        lambda g, a, b: g.call_function(
            ATEN.sin.default,
            (g.call_function(ATEN.add.Tensor, (a, b)),),
        ),
    )
    with pytest.raises(UnsupportedFXError) as caught:
        import_fx(gm, fake_inputs((4, 5), (4, 5)), options())
    assert caught.value.node_name == "sin"
    assert caught.value.code == "unsupported_fx_operation"


def test_view_identifies_the_layout_dependent_node():
    gm = build_graph(
        ("a", "b"),
        lambda g, a, b: g.call_function(
            ATEN.view.default,
            (g.call_function(ATEN.add.Tensor, (a, b)), [-1]),
        ),
    )
    with pytest.raises(UnsupportedFXError) as caught:
        import_fx(gm, fake_inputs((4, 5), (4, 5)), options())
    assert caught.value.node_name == "view"
    assert caught.value.code == "unsupported_fx_view"


def test_in_place_operations_are_rejected_by_node_identity():
    gm = build_graph(
        ("a", "b"), lambda g, a, b: g.call_function(ATEN.add_.Tensor, (a, b))
    )
    with pytest.raises(UnsupportedFXError) as caught:
        import_fx(gm, fake_inputs((4, 5), (4, 5)), options())
    assert caught.value.node_name == "add_"
    assert caught.value.code == "unsupported_fx_mutation"


def test_scalar_operand_identifies_the_node():
    gm = build_graph(
        ("a", "b"),
        lambda g, a, b: g.call_function(ATEN.add.Tensor, (a, 1.0)),
    )
    with pytest.raises(UnsupportedFXError) as caught:
        import_fx(gm, fake_inputs((4, 5), (4, 5)), options())
    assert caught.value.node_name == "add"
    assert caught.value.code == "unsupported_fx_operand"


def test_inference_import_rejects_requires_grad_inputs():
    gm = build_graph(
        ("a", "b"), lambda g, a, b: g.call_function(ATEN.add.Tensor, (a, b))
    )
    with pytest.raises(UnsupportedFXError) as caught:
        import_fx(
            gm,
            fake_inputs((4, 5), (4, 5), requires_grad=True),
            options(),
        )
    assert caught.value.code == "unsupported_training_contract"


def test_noncontiguous_example_metadata_is_rejected():
    gm = build_graph(
        ("a", "b"), lambda g, a, b: g.call_function(ATEN.add.Tensor, (a, b))
    )
    with FakeTensorMode():
        base = torch.empty((5, 4), dtype=torch.float32)
        transposed = base.t()
        inputs = (transposed, torch.empty((4, 5), dtype=torch.float32))
    with pytest.raises(UnsupportedFXError) as caught:
        import_fx(gm, inputs, options())
    assert caught.value.code == "unsupported_tensor_layout"
    assert caught.value.node_name == "a"


def test_import_never_reads_fake_storage():
    gm = build_graph(
        ("a", "b"), lambda g, a, b: g.call_function(ATEN.add.Tensor, (a, b))
    )
    inputs = fake_inputs((4, 5), (4, 5))

    def refuse(*args, **kwargs):
        raise AssertionError("import_fx must not read tensor storage")

    with patch.object(FakeTensor, "data_ptr", refuse), patch.object(
        torch.Tensor, "data_ptr", refuse
    ):
        request = import_fx(gm, inputs, options())
    assert request.program.tensors[2].direction == "output"


def test_multiple_outputs_are_imported_in_graph_order():
    gm = build_graph(
        ("a", "b"),
        lambda g, a, b: (
            g.call_function(ATEN.add.Tensor, (a, b)),
            g.call_function(ATEN.mul.Tensor, (a, b)),
        ),
    )
    requests = import_fx_requests(gm, fake_inputs((4, 5), (4, 5)), options())
    assert len(requests) == 2
    assert [r.program.entry for r in requests] == ["torch_output_0", "torch_output_1"]

    def data_opcode(program):
        producers = {
            result.id: op.opcode
            for op in program.operations
            for result in op.results
        }
        store = next(op for op in program.operations if op.opcode == "store")
        return producers[store.operands[1]]

    assert data_opcode(requests[0].program) == "add"
    assert data_opcode(requests[1].program) == "mul"
    with pytest.raises(UnsupportedFXError) as caught:
        import_fx(gm, fake_inputs((4, 5), (4, 5)), options())
    assert caught.value.code == "unsupported_fx_outputs"


def test_failed_import_executes_nothing_and_does_not_mutate_the_graph():
    gm = build_graph(
        ("a", "b"),
        lambda g, a, b: g.call_function(
            ATEN.sin.default,
            (g.call_function(ATEN.add.Tensor, (a, b)),),
        ),
    )
    before = gm.code
    with patch.object(gm, "forward", wraps=gm.forward) as forward:
        with pytest.raises(UnsupportedFXError) as caught:
            import_fx(gm, fake_inputs((4, 5), (4, 5)), options())
    assert caught.value.node_name == "sin"
    assert forward.call_count == 0
    assert gm.code == before


def test_dead_or_composed_nodes_are_not_silently_ignored():
    # The sin node is dead: the output is the add below it.
    graph = torch.fx.Graph()
    a = graph.placeholder("a")
    b = graph.placeholder("b")
    added = graph.call_function(ATEN.add.Tensor, (a, b))
    graph.call_function(ATEN.sin.default, (added,))
    graph.output((added,))
    _rename_like_dynamo(graph)
    gm = torch.fx.GraphModule({}, graph)
    with pytest.raises(UnsupportedFXError) as caught:
        import_fx(gm, fake_inputs((4, 5), (4, 5)), options())
    assert caught.value.node_name == "sin"
    assert caught.value.code == "unsupported_fx_operation"
