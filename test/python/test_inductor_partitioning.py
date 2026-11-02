"""PF15: scoped FX partitioning into registered LLK external kernels.

The rewrite is exercised directly so every report field is asserted: accepted
regions, declined nodes with reasons, a graph with no native region, and the
metadata-free fallback that makes ``example_inputs`` meaningful. Registration
is checked for idempotence and foreign-owner safety, mirroring the strict
``llk`` backend contract.
"""

from __future__ import annotations

import dataclasses
import importlib
import operator
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from torch.fx.experimental.proxy_tensor import make_fx

from llk import UnsupportedFeatureError
from llk.torch import _inductor_backend
from llk.torch._inductor_backend import (
    PartitionReport,
    inductor_backend,
    rewrite_supported_subgraphs,
)

PYTHON_ROOT = Path(__file__).resolve().parents[2] / "python"


@pytest.fixture(autouse=True)
def reset_dynamo():
    torch._dynamo.reset()
    yield
    torch._dynamo.reset()


def traced(fn, shapes, dtype=torch.float32):
    """Return an ATen FX graph carrying per-node fake metadata plus inputs."""
    inputs = tuple(torch.randn(shape, dtype=dtype) for shape in shapes)
    return make_fx(fn, tracing_mode="fake")(*inputs), inputs


def direct_module(targets, shapes):
    """Build an FX graph without any example-value metadata."""
    graph = torch.fx.Graph()
    arguments = [graph.placeholder(f"arg{index}") for index in range(len(shapes))]
    graph.output((targets(graph, *arguments),))
    inputs = tuple(torch.randn(shape) for shape in shapes)
    return torch.fx.GraphModule({}, graph), inputs


def targets(module):
    return [node.target for node in module.graph.nodes if node.op == "call_function"]


def test_partition_report_is_frozen():
    report = PartitionReport(
        llk_regions=1, matched_nodes=("add",), declined=(("sin", "unmatched_operation"),)
    )
    assert report.llk_regions == 1
    assert report.matched_nodes == ("add",)
    assert report.declined == (("sin", "unmatched_operation"),)
    with pytest.raises(dataclasses.FrozenInstanceError):
        report.llk_regions = 2


def test_add_region_is_replaced_and_unmatched_activation_is_declined():
    module, inputs = traced(lambda a, b: torch.sin(a + b), ((4, 5), (4, 5)))
    rewritten, report = rewrite_supported_subgraphs(module, inputs, None)

    assert report.llk_regions == 1
    assert report.matched_nodes == ("add",)
    assert report.declined == (("sin", "unmatched_operation"),)

    assert targets(rewritten).count(torch.ops.llk.add.default) == 1
    # The source graph is untouched: partitioning clones first.
    assert torch.ops.llk.add.default not in targets(module)

    llk_add = next(
        node for node in rewritten.graph.nodes if node.target is torch.ops.llk.add.default
    )
    sin = next(node for node in rewritten.graph.nodes if node.target is torch.ops.aten.sin.default)
    assert list(llk_add.users) == [sin]
    rewritten.graph.lint()


def test_swiglu_region_replaces_every_internal_node():
    module, inputs = traced(
        lambda x, wg, wu: torch.cos(F.silu(x @ wg) * (x @ wu)),
        ((3, 6), (6, 4), (6, 4)),
    )
    rewritten, report = rewrite_supported_subgraphs(module, inputs, None)

    assert report.llk_regions == 1
    assert report.matched_nodes == ("mul",)
    assert report.declined == (("cos", "unmatched_operation"),)

    remaining = targets(rewritten)
    assert remaining.count(torch.ops.llk.swiglu.default) == 1
    for internal in (torch.ops.aten.mm.default, torch.ops.aten.silu.default):
        assert internal not in remaining
    assert torch.ops.aten.cos.default in remaining
    rewritten.graph.lint()


def test_shared_users_of_a_native_region_are_preserved():
    def fn(a, b):
        summed = a + b
        return torch.sin(summed) + torch.cos(summed)

    module, inputs = traced(fn, ((4, 5), (4, 5)))
    rewritten, report = rewrite_supported_subgraphs(module, inputs, None)

    assert report.llk_regions == 1
    assert report.matched_nodes == ("add",)
    # The outer add reads two non-native activations and is declined.
    assert ("add_1", "operand_not_native") in report.declined

    llk_adds = [
        node for node in rewritten.graph.nodes if node.target is torch.ops.llk.add.default
    ]
    assert len(llk_adds) == 1
    assert len(llk_adds[0].users) == 2
    assert {node.target for node in llk_adds[0].users} == {
        torch.ops.aten.sin.default,
        torch.ops.aten.cos.default,
    }


def test_chained_regions_read_the_replacing_llk_output():
    def fn(a, b, c):
        return torch.sin((a + b) + c)

    module, inputs = traced(fn, ((4, 5), (4, 5), (4, 5)))
    rewritten, report = rewrite_supported_subgraphs(module, inputs, None)

    assert report.llk_regions == 2
    assert report.matched_nodes == ("add", "add_1")
    llk_adds = [
        node
        for node in rewritten.graph.nodes
        if node.target is torch.ops.llk.add.default
    ]
    assert len(llk_adds) == 2
    # The second native region consumes the first one's fresh output.
    assert llk_adds[0] in llk_adds[1].args
    rewritten.graph.lint()


def test_graph_without_a_native_region_reports_zero_partitions():
    module, inputs = traced(
        lambda a, b: torch.exp(a) + torch.exp(b), ((4, 5), (4, 5))
    )
    rewritten, report = rewrite_supported_subgraphs(module, inputs, None)

    assert report.llk_regions == 0
    assert report.matched_nodes == ()
    assert ("add", "operand_not_native") in report.declined
    assert not any("llk" in str(target) for target in targets(rewritten))


def test_bf16_swiglu_is_declined_with_a_dtype_reason():
    module, inputs = traced(
        lambda x, wg, wu: torch.cos(F.silu(x @ wg) * (x @ wu)),
        ((3, 6), (6, 4), (6, 4)),
        dtype=torch.bfloat16,
    )
    rewritten, report = rewrite_supported_subgraphs(module, inputs, None)

    assert report.llk_regions == 0
    assert ("mul", "unsupported_dtype") in report.declined
    assert not any("llk" in str(target) for target in targets(rewritten))


def test_unmatched_activation_and_reduction_policies_are_declined():
    module, inputs = traced(
        lambda x, wg, wu: torch.sigmoid(x @ wg) * (x @ wu),
        ((3, 6), (6, 4), (6, 4)),
    )
    rewritten, report = rewrite_supported_subgraphs(module, inputs, None)
    # The incompatible fused region declines, but both plain contractions are
    # still separate native regions with an Inductor-owned activation between.
    assert report.llk_regions == 2
    assert report.matched_nodes == ("mm", "mm_1")
    assert ("mul", "operand_not_native") in report.declined
    assert ("sigmoid", "unmatched_operation") in report.declined
    assert [node.target for node in rewritten.graph.nodes].count(
        torch.ops.llk.matmul.default
    ) == 2

    module, inputs = traced(lambda x: F.silu(x), ((4, 5),))
    _, report = rewrite_supported_subgraphs(module, inputs, None)
    assert report.llk_regions == 0
    assert ("silu", "unmatched_operation") in report.declined

    module, inputs = traced(lambda a: torch.sum(a, dim=1), ((4, 5),))
    _, report = rewrite_supported_subgraphs(module, inputs, None)
    assert report.llk_regions == 0
    assert ("sum_1", "unmatched_operation") in report.declined


def test_graph_with_a_mutation_declines_partitioning():
    graph = torch.fx.Graph()
    a = graph.placeholder("a")
    b = graph.placeholder("b")
    graph.call_method("add_", (a, b))
    summed = graph.call_function(torch.ops.aten.add.Tensor, (a, b))
    graph.output(graph.call_function(torch.ops.aten.sin.default, (summed,)))
    module = torch.fx.GraphModule({}, graph)
    inputs = (torch.randn(4, 5), torch.randn(4, 5))

    rewritten, report = rewrite_supported_subgraphs(module, inputs, None)
    assert report.llk_regions == 0
    assert [reason for _, reason in report.declined].count("unsupported_effect") == 1
    assert not any("llk" in str(target) for target in targets(rewritten))


def test_symbolic_extents_are_declined():
    inputs = (torch.randn(4, 5), torch.randn(4, 5))
    module = make_fx(lambda a, b: a + b, tracing_mode="symbolic")(*inputs)

    _, report = rewrite_supported_subgraphs(module, inputs, None)
    assert report.llk_regions == 0
    assert ("add", "unsupported_dynamic_shape") in report.declined


def test_non_compact_placeholder_is_declined_instead_of_copied():
    source = torch.randn(5, 4).t()
    other = torch.randn(4, 5)
    module = make_fx(lambda a, b: a + b, tracing_mode="fake")(source, other)
    _, report = rewrite_supported_subgraphs(module, (source, other), None)

    assert report.llk_regions == 0
    assert ("add", "unsupported_layout") in report.declined


def test_missing_metadata_falls_back_to_the_example_inputs():
    def direct(graph, a, b):
        return graph.call_function(operator.add, (a, b))

    module, inputs = direct_module(direct, ((4, 5), (4, 5)))
    assert all(
        node.meta.get("val") is None and node.meta.get("example_value") is None
        for node in module.graph.nodes
        if node.op == "call_function"
    )

    rewritten, report = rewrite_supported_subgraphs(module, inputs, None)
    assert report.llk_regions == 1
    assert targets(rewritten) == [torch.ops.llk.add.default]


def test_rewrite_accepts_only_mapping_or_none_options():
    module, inputs = traced(lambda a, b: a + b, ((4, 5), (4, 5)))
    rewritten, report = rewrite_supported_subgraphs(module, inputs, {"max_autotune": False})
    assert report.llk_regions == 1
    with pytest.raises(TypeError):
        rewrite_supported_subgraphs(module, inputs, object())


def test_registration_is_idempotent_across_module_reload():
    from torch._dynamo.backends.registry import lookup_backend

    assert lookup_backend("llk_inductor") is inductor_backend
    importlib.reload(_inductor_backend)
    assert _inductor_backend.register() is None
    assert lookup_backend("llk_inductor") is _inductor_backend.inductor_backend
    assert _inductor_backend.inductor_backend is inductor_backend


def test_foreign_backend_owner_is_not_replaced(tmp_path):
    code = """
        import torch
        from llk import UnsupportedFeatureError
        from torch._dynamo.backends.registry import register_backend

        register_backend(
            compiler_fn=lambda gm, inputs, **kwargs: gm.forward, name="llk_inductor"
        )
        try:
            import llk.torch
        except UnsupportedFeatureError as error:
            assert error.code == "backend_registration_conflict"
        else:
            raise AssertionError("Foreign llk_inductor was silently adopted")
    """
    env = dict(
        os.environ,
        PYTHONPATH=str(PYTHON_ROOT),
        PYTHONDONTWRITEBYTECODE="1",
    )
    result = subprocess.run(
        [sys.executable, "-u", "-c", textwrap.dedent(code)],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_inductor_backend_rejects_unknown_keyword_arguments():
    module, inputs = traced(lambda a, b: a + b, ((4, 5), (4, 5)))
    with pytest.raises(TypeError):
        inductor_backend(module, inputs, unexpected=True)
