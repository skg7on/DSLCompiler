"""Strict Dynamo backend that compiles supported ATen graphs through LLK.

``backend="llk"`` normalizes captured graphs with AOTAutograd's ATen
interface, imports one shared frontend request per output, and invokes the
compiled native handles over caller-owned buffers. Unsupported graphs identify
the offending FX node and fail; this backend never returns ``gm.forward`` and
never executes the graph eagerly.
"""

from __future__ import annotations

import threading

from .._errors import UnsupportedFeatureError

try:
    import torch
except ModuleNotFoundError as error:
    # Missing dependencies inside an installed Torch indicate a broken install.
    if error.name != "torch":
        raise
    torch = None


class LLKBackend:
    """Dynamo entry point backed by the shared native frontend service.

    Keyword arguments are default ``CompileOptions`` fields. Per-compilation
    overrides arrive through ``torch.compile(..., options={...})`` and are
    applied for that graph only.
    """

    def __init__(self, **options):
        self._options = dict(options)
        self._cache = None
        self._cache_lock = threading.Lock()

    def _compilation_cache(self):
        with self._cache_lock:
            if self._cache is None:
                from ..runtime._cache import CompilationCache

                self._cache = CompilationCache()
            return self._cache

    def cache_info(self):
        """Return this backend's cache counters, or ``None`` before use."""
        with self._cache_lock:
            cache = self._cache
        return None if cache is None else cache.info()

    def __call__(self, gm, example_inputs, **kwargs):
        if torch is None:
            raise UnsupportedFeatureError(
                "Torch is not installed; the LLK Dynamo backend is unavailable.",
                code="native_backend_unavailable",
            )
        # The native service must exist before any graph is accepted: a
        # registered backend that can never execute must say so, not report an
        # unrelated graph-shape diagnostic.
        from ..runtime import _ffi

        _ffi.library()

        from . import _compat, _fx_importer

        _compat.require_supported_version()
        overrides = kwargs.pop("options", None)
        if kwargs:
            raise TypeError(
                f"unexpected LLK backend options: {sorted(kwargs)}"
            )
        changes = dict(self._options)
        if overrides is not None:
            if not isinstance(overrides, dict):
                raise TypeError("backend options must be a mapping")
            changes.update(overrides)
        options = _fx_importer.compile_options(changes)
        # Mutation and layout dependence are properties of the captured graph.
        # Reject them before normalization can fold an in-place call into a
        # different functional graph.
        _fx_importer.reject_unsupported_graph(gm)
        cache = self._compilation_cache()

        def compiler(graph, inputs):
            return _fx_importer.compile_graph(graph, inputs, options, cache)

        def training_compiler(graph, inputs):
            # AOTAutograd selects the forward compiler only when autograd is
            # required. PF16 registers the trained formulas; until then this is
            # a concrete rejection instead of silently running an eager graph
            # or narrowing parameters to inference.
            raise UnsupportedFeatureError(
                "LLK compiles inference graphs only; requires_grad inputs and "
                "training graphs need the registered backward path (PF16)",
                code="unsupported_training_contract",
                stage="compile",
            )

        return _compat.aot_autograd_compiler(training_compiler, compiler)(
            gm, example_inputs
        )


# reload retains the module dictionary. Preserve the lock and callable: a
# redefined class is not an identity check for an already registered instance.
if "_registration_lock" not in globals():
    _registration_lock = threading.RLock()
if "_llk_backend" not in globals():
    _llk_backend = LLKBackend()


def get_backend() -> LLKBackend:
    """Return the exact singleton used for public reporting and registration."""
    return _llk_backend


def register() -> None:
    """Register once without resetting Dynamo or replacing another owner."""
    if torch is None:
        return

    from torch._dynamo.backends.registry import lookup_backend, register_backend
    from torch._dynamo.exc import InvalidBackend

    with _registration_lock:
        try:
            registered = lookup_backend("llk")
        except InvalidBackend:
            register_backend(compiler_fn=_llk_backend, name="llk")
            return
        if registered is not _llk_backend:
            raise UnsupportedFeatureError(
                "The Torch backend name 'llk' is already registered by another owner.",
                code="backend_registration_conflict",
            )
