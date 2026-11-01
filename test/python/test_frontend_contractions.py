"""PF11: counted contractions and guarded f32 SwiGLU through the shared frontend.

A canonical counted-K loop with one or two dot accumulators is recovered as the
native matmul or fused SwiGLU operator. The kernel the compiler produces is
executed on caller-owned buffers and every element is checked. Patterns the
recoverer cannot preserve -- a second writer, a different left operand per
projection, a rounded intermediate, a reduction past K -- are rejected by name
instead of being lowered approximately.
"""

from __future__ import annotations

import math

import llk
import llk.language as tl
import native_cases as cases
import pytest
from llk import LLKError, UnsupportedFeatureError
from llk.frontend import LaunchSpec, NumericalContract, TensorArgument, TensorSpec

MATMUL_SHAPE = (32, 64, 16)  # M, N, K
MATMUL_TILE = (32, 64, 16)  # BM, BN, BK
SMALL_SHAPE = (7, 9, 5)
SMALL_TILE = (7, 9, 5)
SWIGLU_SHAPE = (16, 64, 16)
SWIGLU_TILE = (16, 64, 16)
F32_TOLERANCE = {"rel": 1e-4, "abs": 1e-5}


def _kernel(fn, backend=None, numerical_contract=None):
    options = {"config_root": ".", "config_files": cases.config_files()}
    if backend is not None:
        options["backend"] = backend
    if numerical_contract is not None:
        options["numerical_contract"] = numerical_contract
    return llk.jit(**options)(fn)


def matmul(
    a,
    b,
    out,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    pm = tl.program_id(0)
    pn = tl.program_id(1)
    rows = pm * BM + tl.arange(0, BM)
    cols = pn * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)
    acc = tl.zeros((BM, BN), tl.float32)
    for kk in range(0, K, BK):
        at = tl.load(a + rows[:, None] * K + (kk + rk)[None, :], True, 0.0)
        bt = tl.load(b + (kk + rk)[:, None] * N + cols[None, :], True, 0.0)
        acc = tl.dot(at, bt, acc)
    tl.store(out + rows[:, None] * N + cols[None, :], acc, True)


def matmul_accumulator(
    a,
    b,
    init,
    out,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    pm = tl.program_id(0)
    pn = tl.program_id(1)
    rows = pm * BM + tl.arange(0, BM)
    cols = pn * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)
    acc = tl.load(init + rows[:, None] * N + cols[None, :], True, 0.0)
    for kk in range(0, K, BK):
        at = tl.load(a + rows[:, None] * K + (kk + rk)[None, :], True, 0.0)
        bt = tl.load(b + (kk + rk)[:, None] * N + cols[None, :], True, 0.0)
        acc = tl.dot(at, bt, acc)
    tl.store(out + rows[:, None] * N + cols[None, :], acc, True)


def swiglu(
    x,
    wg,
    wu,
    out,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    pm = tl.program_id(0)
    pn = tl.program_id(1)
    rows = pm * BM + tl.arange(0, BM)
    cols = pn * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)
    gate = tl.zeros((BM, BN), tl.float32)
    up = tl.zeros((BM, BN), tl.float32)
    for kk in range(0, K, BK):
        xt = tl.load(x + rows[:, None] * K + (kk + rk)[None, :], True, 0.0)
        gt = tl.load(wg + (kk + rk)[:, None] * N + cols[None, :], True, 0.0)
        ut = tl.load(wu + (kk + rk)[:, None] * N + cols[None, :], True, 0.0)
        gate = tl.dot(xt, gt, gate)
        up = tl.dot(xt, ut, up)
    activated = tl.silu(gate)
    tl.store(out + rows[:, None] * N + cols[None, :], activated * up, True)


def swiglu_native(
    x,
    wg,
    wu,
    out,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    pm = tl.program_id(0)
    pn = tl.program_id(1)
    rows = pm * BM + tl.arange(0, BM)
    cols = pn * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)
    gate = tl.zeros((BM, BN), tl.float32)
    up = tl.zeros((BM, BN), tl.float32)
    for kk in range(0, K, BK):
        xt = tl.load(x + rows[:, None] * K + (kk + rk)[None, :], True, 0.0)
        gt = tl.load(wg + (kk + rk)[:, None] * N + cols[None, :], True, 0.0)
        ut = tl.load(wu + (kk + rk)[:, None] * N + cols[None, :], True, 0.0)
        gate = tl.dot(xt, gt, gate)
        up = tl.dot(xt, ut, up)
    activated = tl.silu(gate, approximation="native")
    tl.store(out + rows[:, None] * N + cols[None, :], activated * up, True)


def swiglu_distinct_inputs(
    x0,
    x1,
    wg,
    wu,
    out,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    pm = tl.program_id(0)
    pn = tl.program_id(1)
    rows = pm * BM + tl.arange(0, BM)
    cols = pn * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)
    gate = tl.zeros((BM, BN), tl.float32)
    up = tl.zeros((BM, BN), tl.float32)
    for kk in range(0, K, BK):
        at = tl.load(x0 + rows[:, None] * K + (kk + rk)[None, :], True, 0.0)
        bt = tl.load(x1 + rows[:, None] * K + (kk + rk)[None, :], True, 0.0)
        gt = tl.load(wg + (kk + rk)[:, None] * N + cols[None, :], True, 0.0)
        ut = tl.load(wu + (kk + rk)[:, None] * N + cols[None, :], True, 0.0)
        gate = tl.dot(at, gt, gate)
        up = tl.dot(bt, ut, up)
    activated = tl.silu(gate)
    tl.store(out + rows[:, None] * N + cols[None, :], activated * up, True)


def swiglu_rounded_activation(
    x,
    wg,
    wu,
    out,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    pm = tl.program_id(0)
    pn = tl.program_id(1)
    rows = pm * BM + tl.arange(0, BM)
    cols = pn * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)
    gate = tl.zeros((BM, BN), tl.float32)
    up = tl.zeros((BM, BN), tl.float32)
    for kk in range(0, K, BK):
        xt = tl.load(x + rows[:, None] * K + (kk + rk)[None, :], True, 0.0)
        gt = tl.load(wg + (kk + rk)[:, None] * N + cols[None, :], True, 0.0)
        ut = tl.load(wu + (kk + rk)[:, None] * N + cols[None, :], True, 0.0)
        gate = tl.dot(xt, gt, gate)
        up = tl.dot(xt, ut, up)
    activated = tl.silu(gate).to(tl.bfloat16).to(tl.float32)
    tl.store(out + rows[:, None] * N + cols[None, :], activated * up, True)


def matmul_extra_store(
    a,
    b,
    out,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    pm = tl.program_id(0)
    pn = tl.program_id(1)
    rows = pm * BM + tl.arange(0, BM)
    cols = pn * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)
    acc = tl.zeros((BM, BN), tl.float32)
    for kk in range(0, K, BK):
        at = tl.load(a + rows[:, None] * K + (kk + rk)[None, :], True, 0.0)
        bt = tl.load(b + (kk + rk)[:, None] * N + cols[None, :], True, 0.0)
        acc = tl.dot(at, bt, acc)
    tl.store(out + rows[:, None] * N + cols[None, :], acc, True)
    tl.store(out + rows[:, None] * N + cols[None, :], acc + acc, True)


def matmul_wrong_extent(
    a,
    b,
    out,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    pm = tl.program_id(0)
    pn = tl.program_id(1)
    rows = pm * BM + tl.arange(0, BM)
    cols = pn * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)
    acc = tl.zeros((BM, BN), tl.float32)
    for kk in range(0, K + BK, BK):
        at = tl.load(a + rows[:, None] * K + (kk + rk)[None, :], True, 0.0)
        bt = tl.load(b + (kk + rk)[:, None] * N + cols[None, :], True, 0.0)
        acc = tl.dot(at, bt, acc)
    tl.store(out + rows[:, None] * N + cols[None, :], acc, True)


def matmul_ignored_accumulator(
    a,
    b,
    out,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    pm = tl.program_id(0)
    pn = tl.program_id(1)
    rows = pm * BM + tl.arange(0, BM)
    cols = pn * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)
    for kk in range(0, K, BK):
        at = tl.load(a + rows[:, None] * K + (kk + rk)[None, :], True, 0.0)
        bt = tl.load(b + (kk + rk)[:, None] * N + cols[None, :], True, 0.0)
        acc = tl.dot(at, bt)
        tl.store(out + rows[:, None] * N + cols[None, :], acc, True)


# --- fixtures -------------------------------------------------------------


def _spec(name, shape, direction, effect):
    return TensorSpec(name, shape, "f32", (shape[1], 1), direction, effect)


def _values(count, modulus=7, offset=3.0):
    return [float(index % modulus) - offset for index in range(count)]


def _launch(tile):
    return LaunchSpec((1, 1, 1), (tile[0], tile[1], 1))


def _matmul_arguments(shape, tile):
    M, N, K = shape
    a = cases.Guarded((M, K), _values(M * K))
    b = cases.Guarded((K, N), _values(K * N, modulus=5, offset=2.0))
    out = cases.Guarded((M, N))
    specs = (
        _spec("a", (M, K), "input", "read"),
        _spec("b", (K, N), "input", "read"),
        _spec("out", (M, N), "output", "write"),
    )
    bound = (
        TensorArgument(specs[0], a.view),
        TensorArgument(specs[1], b.view),
        TensorArgument(specs[2], out.view),
    )
    return bound + (M, N, K, tile[0], tile[1], tile[2]), (a, b, out), specs


def _accumulator_arguments(shape, tile):
    M, N, K = shape
    args, (a, b, out), specs = _matmul_arguments(shape, tile)
    init = cases.Guarded((M, N), [float(index % 11) * 0.25 - 1.0 for index in range(M * N)])
    init_spec = _spec("init", (M, N), "input", "read")
    bound = (
        args[0],
        args[1],
        TensorArgument(init_spec, init.view),
        args[2],
    )
    return bound + args[3:], (a, b, init, out), (
        specs[0],
        specs[1],
        init_spec,
        specs[2],
    )


def _swiglu_arguments():
    M, N, K = SWIGLU_SHAPE
    x = cases.Guarded((M, K), _values(M * K))
    wg = cases.Guarded((K, N), _values(K * N, modulus=5, offset=2.0))
    wu = cases.Guarded((K, N), [float(index % 3) - 1.0 for index in range(K * N)])
    out = cases.Guarded((M, N))
    specs = (
        _spec("x", (M, K), "input", "read"),
        _spec("wg", (K, N), "input", "read"),
        _spec("wu", (K, N), "input", "read"),
        _spec("out", (M, N), "output", "write"),
    )
    bound = (
        TensorArgument(specs[0], x.view),
        TensorArgument(specs[1], wg.view),
        TensorArgument(specs[2], wu.view),
        TensorArgument(specs[3], out.view),
    )
    return bound + (*SWIGLU_SHAPE, *SWIGLU_TILE), (x, wg, wu, out), specs


def _matmul_reference(a, b, init=None):
    M, N, K = MATMUL_SHAPE
    if a.rows != M:
        M, N, K = a.rows, b.columns, a.columns
    out = []
    for row in range(M):
        for column in range(N):
            total = 0.0 if init is None else init.view[row, column]
            for k in range(K):
                total += a.view[row, k] * b.view[k, column]
            out.append(total)
    return out


def _swiglu_reference(x, wg, wu):
    M, N, K = SWIGLU_SHAPE
    out = []
    for row in range(M):
        for column in range(N):
            gate = 0.0
            up = 0.0
            for k in range(K):
                gate += x.view[row, k] * wg.view[k, column]
                up += x.view[row, k] * wu.view[k, column]
            out.append(gate / (1.0 + math.exp(-gate)) * up)
    return out


def _invoke(compiled, providers):
    compiled.invoke(tuple(providers[:-1]), (providers[-1],))


def _expect(buffers, expected):
    got = buffers[-1].values()
    assert len(got) == len(expected)
    for actual, want in zip(got, expected):
        assert abs(actual - want) <= F32_TOLERANCE["abs"] + F32_TOLERANCE["rel"] * abs(
            want
        ), (actual, want)


# --- execution ------------------------------------------------------------


@pytest.mark.parametrize(("shape", "tile"), [(MATMUL_SHAPE, MATMUL_TILE), (SMALL_SHAPE, SMALL_TILE)])
def test_matmul_executes_with_exact_callers_storage(shape, tile):
    args, buffers, _ = _matmul_arguments(shape, tile)
    a, b, out = buffers
    expected = _matmul_reference(a, b)
    compiled = _kernel(matmul).compile(*args, grid=_launch(tile))
    try:
        inputs_before = (a.bytes(), b.bytes())
        _invoke(compiled, (a.view, b.view, out.view))
        _expect(buffers, expected)
        assert (a.bytes(), b.bytes()) == inputs_before
        assert a.guards_intact() and b.guards_intact() and out.guards_intact()
        assert compiled.report.source_adapter == "llk-contraction-v1"
        assert compiled.report.backend == "reference"
        assert "micro.mma" in compiled.micro_ir
    finally:
        compiled.close()


def test_caller_accumulator_is_read_and_preserved():
    args, buffers, _ = _accumulator_arguments(MATMUL_SHAPE, MATMUL_TILE)
    a, b, init, out = buffers
    expected = _matmul_reference(a, b, init)
    before = init.bytes()
    compiled = _kernel(matmul_accumulator).compile(*args, grid=_launch(MATMUL_TILE))
    try:
        _invoke(compiled, (a.view, b.view, init.view, out.view))
        _expect(buffers, expected)
        assert init.bytes() == before
        assert out.guards_intact()
    finally:
        compiled.close()


def test_fused_swiglu_shares_one_input_and_keeps_numerical_policy():
    args, buffers, _ = _swiglu_arguments()
    x, wg, wu, out = buffers
    expected = _swiglu_reference(x, wg, wu)
    compiled = _kernel(swiglu).compile(*args, grid=_launch(SWIGLU_TILE))
    try:
        _invoke(compiled, (x.view, wg.view, wu.view, out.view))
        _expect(buffers, expected)
        assert x.guards_intact() and wg.guards_intact() and wu.guards_intact()
        assert out.guards_intact()
        assert "silu" in compiled.micro_ir
    finally:
        compiled.close()


def test_relaxed_activation_is_a_different_identity():
    exact_args, _, _ = _swiglu_arguments()
    relaxed_args, _, _ = _swiglu_arguments()
    exact = _kernel(swiglu).compile(*exact_args, grid=_launch(SWIGLU_TILE))
    relaxed = _kernel(
        swiglu_native,
        numerical_contract=NumericalContract(activation_approximation="native"),
    ).compile(*relaxed_args, grid=_launch(SWIGLU_TILE))
    try:
        assert exact.report.native_request_identity != relaxed.report.native_request_identity
        assert exact.report.workload_identity != relaxed.report.workload_identity
    finally:
        exact.close()
        relaxed.close()


def test_nan_propagates_instead_of_being_hidden():
    args, buffers, _ = _swiglu_arguments()
    x, wg, wu, out = buffers
    x.view[0, 0] = float("nan")
    compiled = _kernel(swiglu).compile(*args, grid=_launch(SWIGLU_TILE))
    try:
        _invoke(compiled, (x.view, wg.view, wu.view, out.view))
        assert math.isnan(out.view[0, 0])
        assert out.guards_intact()
    finally:
        compiled.close()


def test_selected_target_executes_or_reports_an_unavailable_host():
    args, buffers, _ = _matmul_arguments(MATMUL_SHAPE, MATMUL_TILE)
    a, b, out = buffers
    expected = _matmul_reference(a, b)
    jit = _kernel(matmul, backend="selected-target")
    try:
        compiled = jit.compile(*args, grid=_launch(MATMUL_TILE))
    except UnsupportedFeatureError as error:
        assert error.code == "native_backend_unavailable"
        return
    try:
        _invoke(compiled, (a.view, b.view, out.view))
        _expect(buffers, expected)
        assert compiled.report.backend == "selected-target"
        assert compiled.report.selected_group_count > 0
        assert compiled.report.reference_group_count == 0
    finally:
        compiled.close()


# --- rejected patterns ----------------------------------------------------


def _distinct_inputs_arguments():
    M, N, K = SWIGLU_SHAPE
    x0 = cases.Guarded((M, K), _values(M * K))
    x1 = cases.Guarded((M, K), _values(M * K, modulus=5, offset=2.0))
    wg = cases.Guarded((K, N), _values(K * N))
    wu = cases.Guarded((K, N), _values(K * N, modulus=3, offset=1.0))
    out = cases.Guarded((M, N))
    specs = (
        _spec("x0", (M, K), "input", "read"),
        _spec("x1", (M, K), "input", "read"),
        _spec("wg", (K, N), "input", "read"),
        _spec("wu", (K, N), "input", "read"),
        _spec("out", (M, N), "output", "write"),
    )
    bound = tuple(
        TensorArgument(spec, buffer.view)
        for spec, buffer in zip(specs, (x0, x1, wg, wu, out))
    )
    return bound + (*SWIGLU_SHAPE, *SWIGLU_TILE), (x0, x1, wg, wu, out)


def test_two_different_left_operands_are_not_fused_swiglu():
    args, _ = _distinct_inputs_arguments()
    with pytest.raises(LLKError) as caught:
        _kernel(swiglu_distinct_inputs).compile(*args, grid=_launch(SWIGLU_TILE))
    assert caught.value.code == "unsupported_frontend_program"


def test_a_rounded_intermediate_is_declined_not_approximated():
    args, _, _ = _swiglu_arguments()
    with pytest.raises(LLKError) as caught:
        _kernel(swiglu_rounded_activation).compile(*args, grid=_launch(SWIGLU_TILE))
    assert caught.value.code == "unsupported_numerical_contract"


def test_a_second_writer_is_outside_the_family():
    args, _ = _matmul_arguments(MATMUL_SHAPE, MATMUL_TILE)[:2]
    with pytest.raises(LLKError) as caught:
        _kernel(matmul_extra_store).compile(*args, grid=_launch(MATMUL_TILE))
    assert caught.value.code == "unsupported_frontend_program"


def test_the_counted_loop_must_traverse_exactly_k():
    args, _ = _matmul_arguments(MATMUL_SHAPE, MATMUL_TILE)[:2]
    with pytest.raises(LLKError) as caught:
        _kernel(matmul_wrong_extent).compile(*args, grid=_launch(MATMUL_TILE))
    assert caught.value.code == "unsupported_frontend_program"


def test_a_dot_without_its_accumulator_does_not_capture():
    args, _ = _matmul_arguments(MATMUL_SHAPE, MATMUL_TILE)[:2]
    with pytest.raises(UnsupportedFeatureError) as caught:
        _kernel(matmul_ignored_accumulator).compile(*args, grid=_launch(MATMUL_TILE))
    assert caught.value.code == "unsupported_frontend_control_flow"
