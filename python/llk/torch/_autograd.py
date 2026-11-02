"""Registered gradients for the checked LLK Torch operators.

``add``, ``mul`` and ``matmul`` get first-order autograd formulas. Every
backward computation that has a native realization calls the same ``llk::*``
custom operator as the forward path, so its compiled kernel is visible as a
distinct backward report. ``matmul`` backward first materializes both transposed
operands with ``contiguous()`` in ATen, because the compact row-major ABI cannot
read a transposed view; those copies are recorded explicitly as non-LLK work.
``swiglu`` deliberately has no formula: training it raises
``unsupported_training_contract`` instead of falling back to eager autograd.

The strict ``llk`` AOTAutograd backend also uses :func:`compile_backward_graph`.
That compiler accepts native gradient graphs (an identity pass-through or
``add``/``mul`` regions) and rejects a backward graph that still needs a view or
transpose with ``unsupported_training_contract``. Non-LLK backward regions are
only delegated to Inductor by the separate ``llk_inductor`` backend.
"""

from __future__ import annotations

import contextlib
import dataclasses
import threading

from .._errors import TensorContractError, UnsupportedFeatureError

try:
    import torch
except ModuleNotFoundError as error:
    # Missing dependencies inside an installed Torch indicate a broken install.
    if error.name != "torch":
        raise
    torch = None


SUPPORTED_TRAINING_OPERATORS = ("add", "mul", "matmul")


@dataclasses.dataclass(frozen=True)
class NativeReport:
    """One native kernel that ran, tagged by the training phase that needed it."""

    phase: str
    operator: str
    report: object


@dataclasses.dataclass(frozen=True)
class Materialization:
    """A layout copy performed outside LLK before the compact ABI boundary."""

    operand: str
    operation: str


@dataclasses.dataclass(frozen=True)
class TrainingReport:
    """Forward and backward native identities observed during one training step."""

    forward: tuple[NativeReport, ...]
    backward: tuple[NativeReport, ...]
    materializations: tuple[Materialization, ...]


class TrainingRecorder:
    """Collect native reports and explicit materializations for one step."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._forward = []
        self._backward = []
        self._materializations = []

    def _add(self, phase, operator, report) -> None:
        item = NativeReport(phase=phase, operator=operator, report=report)
        with self._lock:
            if phase == "backward":
                self._backward.append(item)
            else:
                self._forward.append(item)

    def _materialize(self, operand, operation) -> None:
        item = Materialization(operand=operand, operation=operation)
        with self._lock:
            if item not in self._materializations:
                self._materializations.append(item)

    @property
    def forward(self) -> tuple[NativeReport, ...]:
        with self._lock:
            return tuple(self._forward)

    @property
    def backward(self) -> tuple[NativeReport, ...]:
        with self._lock:
            return tuple(self._backward)

    @property
    def all(self) -> tuple[NativeReport, ...]:
        return self.forward + self.backward

    @property
    def materializations(self) -> tuple[Materialization, ...]:
        with self._lock:
            return tuple(self._materializations)

    @property
    def report(self) -> TrainingReport:
        return TrainingReport(
            forward=self.forward,
            backward=self.backward,
            materializations=self.materializations,
        )


_state = threading.local()
_recorders_lock = threading.Lock()
_recorders = []


def _active_recorders():
    with _recorders_lock:
        return tuple(_recorders)


@contextlib.contextmanager
def record_native_calls():
    """Record native calls and materializations made by one training step.

    AOTAutograd may trace formulas or run backward graphs on a worker thread,
    so active recorders are process-visible rather than thread-local. Every
    active recorder observes the same events; concurrent unrelated training
    steps should not share one recorder context.
    """
    recorder = TrainingRecorder()
    with _recorders_lock:
        _recorders.append(recorder)
    try:
        yield recorder
    finally:
        with _recorders_lock:
            try:
                _recorders.remove(recorder)
            except ValueError:  # pragma: no cover - defensive
                pass


@contextlib.contextmanager
def _phase(name):
    previous = getattr(_state, "phase", None)
    _state.phase = name
    try:
        yield
    finally:
        _state.phase = previous


def _in_autograd_backward() -> bool:
    """Whether the engine is currently running a backward graph task."""
    try:
        from . import _compat

        return _compat.in_autograd_backward()
    except Exception:  # pragma: no cover - defensive around an internal probe
        return False


def current_phase() -> str:
    """Return ``forward`` or ``backward`` for the native call being made.

    An explicit formula phase wins. Compiled backward graphs do not execute the
    Python formula, so they fall back to the autograd engine's graph-task state.
    """
    override = getattr(_state, "phase", None)
    if override is not None:
        return override
    return "backward" if _in_autograd_backward() else "forward"


def record_native(operator, report, phase=None) -> None:
    """Publish one compiled native kernel to every active recorder."""
    resolved = phase or current_phase()
    for recorder in _active_recorders():
        recorder._add(resolved, operator, report)


def record_materialization(operand, operation) -> None:
    """Publish one explicit ATen layout copy to every active recorder."""
    for recorder in _active_recorders():
        recorder._materialize(operand, operation)


# --- formulas -------------------------------------------------------------


def _compact(tensor, operand, operation="contiguous"):
    """Return a compact row-major tensor, recording an explicit ATen copy.

    An incoming gradient from ``sum().backward()`` is often an expanded view
    with zero strides, and a transposed operand is never compact row-major.
    Both must be materialized before the LLK ABI; the copy stays outside the
    native kernel and is reported rather than hidden.
    """
    if tensor.is_contiguous() and int(tensor.storage_offset()) == 0:
        return tensor
    record_materialization(operand, operation)
    compacted = tensor.contiguous()
    if int(compacted.storage_offset()) != 0:
        # ``contiguous()`` is a no-op for an offset view whose strides already
        # match; the compact ABI additionally requires a zero storage offset.
        compacted = compacted.clone()
    return compacted


def _add_backward(ctx, grad):
    return grad, grad


def _add_setup_context(ctx, inputs, output):
    return None


def _mul_backward(ctx, grad):
    a, b = ctx.saved_tensors
    with _phase("backward"):
        gradient = _compact(grad, "grad")
        return (
            torch.ops.llk.mul(gradient, _compact(b, "b")),
            torch.ops.llk.mul(_compact(a, "a"), gradient),
        )


def _mul_setup_context(ctx, inputs, output):
    a, b = inputs
    ctx.save_for_backward(a, b)


def _matmul_backward(ctx, grad):
    a, b = ctx.saved_tensors
    with _phase("backward"):
        # A transposed view is not compact row-major, so the copy is explicit
        # ATen work. It is recorded rather than hidden inside the native call.
        gradient = _compact(grad, "grad")
        right = _compact(b.transpose(0, 1), "b", "transpose_contiguous")
        left = _compact(a.transpose(0, 1), "a", "transpose_contiguous")
        return (
            torch.ops.llk.matmul(gradient, right),
            torch.ops.llk.matmul(left, gradient),
        )


def _matmul_setup_context(ctx, inputs, output):
    a, b = inputs
    ctx.save_for_backward(a, b)


_FORMULAS = {
    "add": (_add_backward, _add_setup_context),
    "mul": (_mul_backward, _mul_setup_context),
    "matmul": (_matmul_backward, _matmul_setup_context),
}

if "_registration_lock" not in globals():
    _registration_lock = threading.RLock()
if "_autograd_registered" not in globals():
    _autograd_registered = False


def register_autograd_formulas() -> None:
    """Register the supported first-order formulas once.

    A repeated call or module reload preserves the existing registration. A
    foreign owner of the operator namespace is reported by
    ``_ops.register_operators`` before any formula is attached.
    """
    global _autograd_registered

    if torch is None:
        return

    from . import _ops

    _ops.register_operators()
    with _registration_lock:
        if _autograd_registered:
            return
        for name, (backward, setup_context) in _FORMULAS.items():
            torch.library.register_autograd(
                f"llk::{name}",
                backward,
                setup_context=setup_context,
            )
        _autograd_registered = True


# --- AOTAutograd backward compiler ---------------------------------------


def compile_backward_graph(graph, example_inputs, options, cache):
    """Compile a strict-mode AOTAutograd backward graph through the native ABI.

    Identity outputs (for example the gradient of ``add``) pass through with no
    native call. ``mul`` gradients are native ``mul`` regions. A graph that still
    contains a transpose view cannot enter the compact row-major ABI, so strict
    mode rejects it with ``unsupported_training_contract`` instead of silently
    copying or handing the graph back to eager execution.
    """
    from . import _fx_importer

    try:
        return _fx_importer.compile_graph(
            graph,
            example_inputs,
            options,
            cache,
            allow_training=True,
            allow_passthrough=True,
            phase="backward",
        )
    except _fx_importer.UnsupportedFXError as error:
        if error.code == "unsupported_fx_view":
            guidance = (
                "a transpose or view needs explicit materialization before the "
                "compact LLK ABI; use backend='llk_inductor' for that delegated "
                "region."
            )
        else:
            guidance = (
                "strict mode does not delegate unsupported backward nodes; use "
                "backend='llk_inductor' to leave that region with Inductor."
            )
        raise UnsupportedFeatureError(
            f"the LLK strict backend cannot compile backward node "
            f"{error.node_name!r}: {error.message}. {guidance}",
            code="unsupported_training_contract",
            stage="compile",
        ) from error


def require_trainable(kind, tensors) -> tuple:
    """Return operands safe for native invocation under a registered formula."""
    if not any(tensor.requires_grad for tensor in tensors):
        return tuple(tensors)
    if kind not in SUPPORTED_TRAINING_OPERATORS:
        raise TensorContractError(
            f"LLK operator '{kind}' has no registered gradient formula; "
            "training requires add, multiply or matmul",
            code="unsupported_training_contract",
        )
    if torch is not None and torch.is_grad_enabled():
        raise TensorContractError(
            "higher-order gradients through LLK operators are not supported",
            code="unsupported_training_contract",
        )
    return tuple(tensor.detach() for tensor in tensors)
