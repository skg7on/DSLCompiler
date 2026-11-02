"""PF15: LLK external kernels composed with real Inductor compilation.

Each case requires both promises at once: a genuine ``torch._inductor``
``compile_fx`` call and a counted native LLK invocation. A correct eager result
alone, or an Inductor result computed without LLK, fails the assertions.
Inductor is a required dependency of this lane; there is no skip path.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest
import torch
import torch.nn.functional as F
from torch._inductor import compile_fx as inductor_module

from llk.runtime import _ffi
from llk.torch import _inductor_backend

MM_TOLERANCE = {"rtol": 1e-4, "atol": 1e-5}
SIN_TOLERANCE = {"rtol": 1e-6, "atol": 1e-6}


@pytest.fixture(autouse=True)
def reset_dynamo():
    torch._dynamo.reset()
    yield
    torch._dynamo.reset()


def compile_llk(fn):
    return torch.compile(
        fn,
        backend="llk_inductor",
        fullgraph=True,
        dynamic=False,
    )


def values(*shapes, seed=0):
    generator = torch.Generator().manual_seed(seed)
    return tuple(
        torch.randn(shape, generator=generator, dtype=torch.float32)
        for shape in shapes
    )


def run_composed(compiled, inputs):
    """Compile and run one call, returning its report and both spies."""
    reports = []
    real_rewrite = _inductor_backend.rewrite_supported_subgraphs

    def capture(module, examples, options):
        result = real_rewrite(module, examples, options)
        reports.append(result[1])
        return result

    with (
        patch.object(_inductor_backend, "rewrite_supported_subgraphs", side_effect=capture),
        patch.object(_ffi, "native_invoke", wraps=_ffi.native_invoke) as native,
        patch.object(inductor_module, "compile_fx", wraps=inductor_module.compile_fx) as inductor,
    ):
        output = compiled(*inputs)
    assert len(reports) == 1
    return output, reports[0], native, inductor


def test_llk_add_is_called_inside_a_real_inductor_compile():
    a, b = values((4, 5), (4, 5))
    output, partition_report, native, inductor = run_composed(
        compile_llk(lambda a, b: torch.sin(a + b)), (a, b)
    )

    assert partition_report.llk_regions == 1
    assert native.call_count == 1
    assert inductor.call_count == 1
    assert partition_report.matched_nodes == ("add",)
    assert ("sin", "unmatched_operation") in partition_report.declined
    torch.testing.assert_close(output, torch.sin(a + b), rtol=0, atol=0)

    # The graph handed to Inductor is the rewritten one: one extern LLK call
    # followed by the Inductor-owned activation.
    rewritten = inductor.call_args.args[0]
    llk_calls = [
        node for node in rewritten.graph.nodes if node.target is torch.ops.llk.add.default
    ]
    assert len(llk_calls) == 1


def test_swiglu_region_is_one_native_call_with_an_inductor_remainder():
    x, wg, wu = values((3, 6), (6, 4), (6, 4))
    output, partition_report, native, inductor = run_composed(
        compile_llk(lambda x, wg, wu: torch.tanh(F.silu(x @ wg) * (x @ wu))),
        (x, wg, wu),
    )

    assert partition_report.llk_regions == 1
    assert native.call_count == 1
    assert inductor.call_count == 1
    assert partition_report.matched_nodes == ("mul",)
    assert ("tanh", "unmatched_operation") in partition_report.declined
    expected = torch.tanh(F.silu(x @ wg) * (x @ wu))
    torch.testing.assert_close(output, expected, **MM_TOLERANCE)

    rewritten = inductor.call_args.args[0]
    assert [
        node.target
        for node in rewritten.graph.nodes
        if node.target is torch.ops.llk.swiglu.default
    ] == [torch.ops.llk.swiglu.default]
    assert torch.ops.aten.mm.default not in [
        node.target for node in rewritten.graph.nodes
    ]


def test_shared_users_reach_one_native_call_and_one_inductor_compile():
    def fn(a, b):
        summed = a + b
        return torch.sin(summed) + torch.cos(summed)

    a, b = values((4, 5), (4, 5))
    output, partition_report, native, inductor = run_composed(
        compile_llk(fn), (a, b)
    )

    assert partition_report.llk_regions == 1
    assert native.call_count == 1
    assert inductor.call_count == 1
    expected = torch.sin(a + b) + torch.cos(a + b)
    torch.testing.assert_close(output, expected, **SIN_TOLERANCE)

    rewritten = inductor.call_args.args[0]
    llk_calls = [
        node for node in rewritten.graph.nodes if node.target is torch.ops.llk.add.default
    ]
    assert len(llk_calls) == 1
    assert len(llk_calls[0].users) == 2


def test_graph_without_a_native_region_reports_zero_partitions():
    def fn(a, b):
        return torch.exp(a) + torch.exp(b)

    a, b = values((4, 5), (4, 5))
    output, partition_report, native, inductor = run_composed(
        compile_llk(fn), (a, b)
    )

    assert partition_report.llk_regions == 0
    assert partition_report.matched_nodes == ()
    assert ("add", "operand_not_native") in partition_report.declined
    assert native.call_count == 0
    assert inductor.call_count == 1
    torch.testing.assert_close(output, torch.exp(a) + torch.exp(b), **SIN_TOLERANCE)


def test_bf16_swiglu_declines_and_is_compiled_by_inductor():
    x, wg, wu = values((3, 6), (6, 4), (6, 4))
    x, wg, wu = x.bfloat16(), wg.bfloat16(), wu.bfloat16()
    output, partition_report, native, inductor = run_composed(
        compile_llk(lambda x, wg, wu: torch.tanh(F.silu(x @ wg) * (x @ wu))),
        (x, wg, wu),
    )

    assert partition_report.llk_regions == 0
    assert ("mul", "unsupported_dtype") in partition_report.declined
    assert native.call_count == 0
    assert inductor.call_count == 1
    expected = torch.tanh(F.silu(x @ wg) * (x @ wu))
    torch.testing.assert_close(output, expected, rtol=1e-2, atol=1e-2)


def test_backend_options_are_scoped_to_the_single_compile():
    before = torch._inductor.config.get_config_copy()
    compiled = torch.compile(
        lambda a, b: torch.sin(a + b),
        backend="llk_inductor",
        fullgraph=True,
        dynamic=False,
        options={"max_autotune": False},
    )
    a, b = values((4, 5), (4, 5))
    with patch.object(
        inductor_module, "compile_fx", wraps=inductor_module.compile_fx
    ) as inductor:
        compiled(a, b)

    patches = [
        call.kwargs.get("config_patches") for call in inductor.call_args_list
    ]
    assert {"max_autotune": False} in patches
    assert torch._inductor.config.get_config_copy() == before


def test_backend_configuration_does_not_leak_into_ordinary_inductor():
    def fn(a, b):
        return torch.sin(a + b)

    a, b = values((4, 5), (4, 5))
    run_composed(compile_llk(fn), (a, b))

    torch._dynamo.reset()
    ordinary = torch.compile(
        fn, backend="inductor", fullgraph=True, dynamic=False
    )
    with (
        patch.object(_ffi, "native_invoke", wraps=_ffi.native_invoke) as native,
        patch.object(inductor_module, "compile_fx", wraps=inductor_module.compile_fx) as inductor,
    ):
        output = ordinary(a, b)

    assert native.call_count == 0
    assert inductor.call_count == 1
    torch.testing.assert_close(output, torch.sin(a + b), **SIN_TOLERANCE)
