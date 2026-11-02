"""Scoped LLK external-kernel partitioning inside Inductor compilation.

``backend="llk_inductor"`` clones the graph Dynamo captured, replaces
effect-safe, natively representable regions with the registered ``llk::*``
functional operators, and hands the rewritten graph to Inductor's own
``compile_fx``. Unmatched operations stay with Inductor; the returned
:class:`PartitionReport` states exactly which regions were accepted and why the
rest were declined.

The rewrite is local to one backend invocation. It does not patch Inductor,
register a device scheduler, or change any process-global Dynamo state, so an
ordinary ``backend="inductor"`` compilation is unaffected.
"""

from __future__ import annotations

import dataclasses
import functools
import operator
import threading
from collections.abc import Mapping

from .._errors import UnsupportedFeatureError

try:
    import torch
except ModuleNotFoundError as error:
    # Missing dependencies inside an installed Torch indicate a broken install.
    if error.name != "torch":
        raise
    torch = None


@dataclasses.dataclass(frozen=True)
class PartitionReport:
    """What one partitioning pass routed through LLK and what it declined.

    ``matched_nodes`` holds the root FX node names replaced by LLK calls, in
    graph order, one per accepted region. ``declined`` pairs every other
    candidate node name with a stable reason. The report is descriptive only:
    declined nodes remain in the graph for Inductor to compile.
    """

    llk_regions: int
    matched_nodes: tuple[str, ...]
    declined: tuple[tuple[str, str], ...]


# Stable decline reasons. They name the contract that failed, not a target.
UNMATCHED = "unmatched_operation"
OPERAND_NOT_NATIVE = "operand_not_native"
SHARED_USER = "shared_user"
UNSUPPORTED_DTYPE = "unsupported_dtype"
UNSUPPORTED_DEVICE = "unsupported_device"
UNSUPPORTED_SHAPE = "unsupported_shape"
UNSUPPORTED_DYNAMIC_SHAPE = "unsupported_dynamic_shape"
UNSUPPORTED_LAYOUT = "unsupported_layout"
UNSUPPORTED_TRAINING = "unsupported_training_contract"
UNSUPPORTED_EFFECT = "unsupported_effect"


@dataclasses.dataclass(frozen=True)
class _Region:
    """One accepted-shape candidate: its root, read operands and internals."""

    kind: str
    root: object
    operands: tuple
    internal: tuple = ()


@functools.lru_cache(maxsize=1)
def _targets():
    """Recognize both Dynamo-level callables and normalized ATen overloads."""
    aten = torch.ops.aten
    return {
        "add": (operator.add, torch.add, aten.add.Tensor),
        "mul": (operator.mul, torch.mul, aten.mul.Tensor),
        "matmul": (
            operator.matmul,
            torch.matmul,
            aten.mm.default,
            aten.matmul.default,
        ),
        "silu": (torch.nn.functional.silu, aten.silu.default),
    }


def _is_target(target, kind) -> bool:
    return any(target is candidate for candidate in _targets()[kind])


def _nodes(*values):
    return all(isinstance(value, torch.fx.Node) for value in values)


def _llk_op(kind: str):
    return getattr(torch.ops.llk, kind).default


# --- value metadata -------------------------------------------------------


def _example_value(node):
    value = node.meta.get("val")
    if value is None:
        value = node.meta.get("example_value")
    return value


def _interpreted_values(module, example_inputs):
    """Replay the graph on fake tensors when the capture carries no metadata."""
    from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode
    from torch.fx import Interpreter

    inputs = tuple(example_inputs)
    tensor_inputs = tuple(isinstance(value, torch.Tensor) for value in inputs)
    fake_mode = None
    if any(tensor_inputs) and not all(
        isinstance(value, FakeTensor)
        for value, is_tensor in zip(inputs, tensor_inputs)
        if is_tensor
    ):
        fake_mode = FakeTensorMode()

    collected = {}

    class _Propagate(Interpreter):
        def run_node(self, node):
            value = super().run_node(node)
            collected[node] = value
            return value

    if fake_mode is None:
        _Propagate(module).run(*inputs)
    else:
        with fake_mode:
            fakes = tuple(
                fake_mode.from_tensor(value) if is_tensor else value
                for value, is_tensor in zip(inputs, tensor_inputs)
            )
            _Propagate(module).run(*fakes)
    return collected


def _node_values(module, example_inputs):
    """Return one example value per metadata-bearing node."""
    values = {}
    complete = True
    for node in module.graph.nodes:
        if node.op not in ("placeholder", "call_function"):
            continue
        value = _example_value(node)
        if value is None:
            complete = False
            continue
        values[node] = value
    if complete:
        return values
    return _interpreted_values(module, example_inputs)


# --- matching -------------------------------------------------------------


def _shape(value):
    return tuple(int(size) for size in value.shape)


def _operand_reason(node, values, materialized):
    """Return why one operand cannot enter a native region, or ``None``."""
    value = values.get(node)
    if not isinstance(value, torch.Tensor):
        return OPERAND_NOT_NATIVE
    if value.dtype is not torch.float32:
        return UNSUPPORTED_DTYPE
    if value.device.type != "cpu":
        return UNSUPPORTED_DEVICE
    if value.dim() != 2:
        return UNSUPPORTED_SHAPE
    if any(isinstance(size, torch.SymInt) for size in value.shape):
        return UNSUPPORTED_DYNAMIC_SHAPE
    if any(int(size) < 1 for size in value.shape):
        return UNSUPPORTED_SHAPE
    if not value.is_contiguous() or int(value.storage_offset()) != 0:
        return UNSUPPORTED_LAYOUT
    if value.requires_grad:
        return UNSUPPORTED_TRAINING
    if node.op == "placeholder" or node in materialized:
        return None
    # A value computed by anything other than an accepted LLK region is not
    # guaranteed compact or copy-free at invocation time.
    return OPERAND_NOT_NATIVE


def _elementwise_region(node, kind):
    if len(node.args) != 2 or not _nodes(*node.args):
        return None
    if node.kwargs.get("alpha", 1) != 1:
        return None
    return _Region(kind=kind, root=node, operands=tuple(node.args))


def _matmul_region(node):
    if len(node.args) != 2 or not _nodes(*node.args):
        return None
    return _Region(kind="matmul", root=node, operands=tuple(node.args))


def _swiglu_region(node):
    """Recognize ``silu(x @ wg) * (x @ wu)`` with one shared left operand."""
    if not _is_target(node.target, "mul") or len(node.args) != 2:
        return None
    if not _nodes(*node.args):
        return None
    left, right = node.args
    if _is_target(left.target, "silu"):
        silu, up = left, right
    elif _is_target(right.target, "silu"):
        silu, up = right, left
    else:
        return None
    if not _is_target(up.target, "matmul") or len(up.args) != 2:
        return None
    if not _nodes(*up.args):
        return None
    if len(silu.args) != 1 or not isinstance(silu.args[0], torch.fx.Node):
        return None
    gate = silu.args[0]
    if not _is_target(gate.target, "matmul") or len(gate.args) != 2:
        return None
    if not _nodes(*gate.args):
        return None
    if gate.args[0] is not up.args[0] or gate.args[1] is up.args[1]:
        return None
    return _Region(
        kind="swiglu",
        root=node,
        operands=(gate.args[0], gate.args[1], up.args[1]),
        internal=(silu, gate, up),
    )


def _single_region(node):
    if _is_target(node.target, "add"):
        return _elementwise_region(node, "add")
    if _is_target(node.target, "mul"):
        return _elementwise_region(node, "mul")
    if _is_target(node.target, "matmul"):
        return _matmul_region(node)
    return None


def _region_reason(region, values, materialized):
    """Return why a structurally matched region cannot be native, or ``None``."""
    region_nodes = frozenset((region.root, *region.internal))
    for internal in region.internal:
        if any(user not in region_nodes for user in internal.users):
            return SHARED_USER

    reasons = [
        _operand_reason(operand, values, materialized)
        for operand in region.operands
    ]
    for reason in reasons:
        if reason is not None:
            return reason

    shapes = [_shape(values[operand]) for operand in region.operands]
    if region.kind in ("add", "mul") and shapes[0] != shapes[1]:
        return UNSUPPORTED_SHAPE
    if region.kind == "matmul" and shapes[0][1] != shapes[1][0]:
        return UNSUPPORTED_SHAPE
    if region.kind == "swiglu" and (
        shapes[0][1] != shapes[1][0] or shapes[1] != shapes[2]
    ):
        return UNSUPPORTED_SHAPE
    return None


_MUTATING_DUNDERS = frozenset({"__setitem__", "__delitem__"})


def _method_mutates(name: str) -> bool:
    if name.startswith("__") and name.endswith("__"):
        return name in _MUTATING_DUNDERS or name.startswith("__i")
    return name.endswith("_")


def _has_mutation(module):
    """Report whether the graph contains an in-place tensor effect.

    The rewrite has no alias analysis, so a mutating op anywhere in the graph
    makes the compact, copy-free operand contract unprovable and partitioning
    declines rather than guessing.
    """
    for node in module.graph.nodes:
        if node.op == "call_method":
            if _method_mutates(str(node.target)):
                return True
        elif node.op == "call_function":
            schema = getattr(node.target, "_schema", None)
            if schema is not None and getattr(schema, "is_mutable", False):
                return True
    return False


def _plan(module, values, mutated=False):
    """Return accepted regions and the report's declined entries."""
    nodes = list(module.graph.nodes)
    matched = {}
    internal = set()

    # SwiGLU roots first, in reverse topological order, so their matmul and
    # activation nodes are not claimed as separate regions.
    for node in reversed(nodes):
        if node.op != "call_function" or node in internal:
            continue
        region = _swiglu_region(node)
        if region is not None:
            matched[node] = region
            internal.update(region.internal)

    for node in nodes:
        if node.op != "call_function" or node in internal or node in matched:
            continue
        region = _single_region(node)
        if region is not None:
            matched[node] = region

    materialized = set()
    accepted = []
    declined_roots = {}
    for node in nodes:
        region = matched.get(node)
        if region is None:
            continue
        reason = UNSUPPORTED_EFFECT if mutated else _region_reason(
            region, values, materialized
        )
        if reason is None:
            materialized.add(node)
            accepted.append(region)
        else:
            declined_roots[node] = reason

    declined = []
    for node in nodes:
        if node.op != "call_function" or node in internal:
            continue
        if node in materialized:
            continue
        reason = declined_roots.get(node, UNMATCHED)
        declined.append((node.name, reason))
    return accepted, tuple(declined)


# --- rewriting ------------------------------------------------------------


def _clone(module):
    graph = torch.fx.Graph()
    output = graph.graph_copy(module.graph, {})
    if output is None:
        raise ValueError("cannot partition an FX graph without an output node")
    graph.output(output)
    return torch.fx.GraphModule(module, graph)


def _apply(module, region, replacements):
    graph = module.graph
    operands = tuple(replacements.get(operand, operand) for operand in region.operands)
    with graph.inserting_before(region.root):
        node = graph.call_function(_llk_op(region.kind), operands)
    replacements[region.root] = node
    node.meta.update(region.root.meta)
    region.root.replace_all_uses_with(node)
    graph.erase_node(region.root)
    # Consumers always follow producers, so erasing in reverse graph order
    # releases every internal node once its user is gone.
    positions = {node: index for index, node in enumerate(module.graph.nodes)}
    for internal in sorted(
        region.internal, key=lambda item: positions[item], reverse=True
    ):
        if not internal.users:
            graph.erase_node(internal)


def rewrite_supported_subgraphs(gm, example_inputs, options=None):
    """Clone ``gm`` and replace natively representable regions with LLK ops.

    ``options`` is the mapping ``torch.compile`` forwards to this backend; it
    is forwarded unchanged to Inductor as scoped config patches and is not
    interpreted by the partition policy. The source module is never mutated.
    """
    if torch is None:
        raise UnsupportedFeatureError(
            "Torch is not installed; LLK Inductor partitioning is unavailable.",
            code="native_backend_unavailable",
        )
    if options is not None and not isinstance(options, Mapping):
        raise TypeError("backend options must be a mapping")

    module = _clone(gm)
    values = _node_values(module, example_inputs)
    accepted, declined = _plan(module, values, mutated=_has_mutation(module))
    replacements = {}
    for region in accepted:
        _apply(module, region, replacements)
    module.graph.lint()
    return module, PartitionReport(
        llk_regions=len(accepted),
        matched_nodes=tuple(region.root.name for region in accepted),
        declined=declined,
    )


# --- backend --------------------------------------------------------------


class _InductorBackend:
    """Dynamo entry point that composes LLK extern calls with Inductor."""

    def __call__(self, gm, example_inputs, **kwargs):
        if torch is None:
            raise UnsupportedFeatureError(
                "Torch is not installed; the LLK Inductor backend is unavailable.",
                code="native_backend_unavailable",
            )
        from ..runtime import _ffi
        from . import _compat

        _compat.require_supported_version()

        options = kwargs.pop("options", None)
        if "mode" in kwargs:
            mode = kwargs.pop("mode")
            if mode not in (None, "default"):
                raise TypeError(
                    "llk_inductor does not accept a torch.compile mode; pass "
                    "Inductor config patches through options instead"
                )
        if kwargs:
            raise TypeError(f"unexpected llk_inductor options: {sorted(kwargs)}")
        if options is not None and not isinstance(options, Mapping):
            raise TypeError("backend options must be a mapping")

        # A registered backend whose native service cannot be loaded must say
        # so before Inductor spends time compiling around it.
        _ffi.library()

        rewritten, report = rewrite_supported_subgraphs(
            gm, example_inputs, options
        )
        compiled = _compat.inductor_compile_fx()(
            rewritten, example_inputs, config_patches=options
        )
        try:
            compiled._llk_partition_report = report
        except AttributeError:  # pragma: no cover - exotic callable result
            pass
        return compiled


# Reload retains the module dictionary. Preserve the lock and callable: a
# redefined class is not an identity check for an already registered backend.
if "_registration_lock" not in globals():
    _registration_lock = threading.RLock()
if "_inductor_backend_instance" not in globals():
    _inductor_backend_instance = _InductorBackend()
inductor_backend = _inductor_backend_instance


def register() -> None:
    """Register ``llk_inductor`` once without resetting Dynamo."""
    if torch is None:
        return

    from torch._dynamo.backends.registry import lookup_backend, register_backend
    from torch._dynamo.exc import InvalidBackend

    with _registration_lock:
        try:
            registered = lookup_backend("llk_inductor")
        except InvalidBackend:
            register_backend(compiler_fn=inductor_backend, name="llk_inductor")
            return
        if registered is not inductor_backend:
            raise UnsupportedFeatureError(
                "The Torch backend name 'llk_inductor' is already registered by "
                "another owner.",
                code="backend_registration_conflict",
            )
