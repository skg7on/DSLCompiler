"""Synchronous LLK decorator: bind, compile and invoke caller-owned buffers."""

from __future__ import annotations

from dataclasses import replace
from typing import Callable

from llk._errors import TensorContractError, UnsupportedFeatureError
from llk.frontend import CompileOptions, CompileRequest, bind_kernel
from llk.language._ast_compiler import compile_kernel

from ._ffi import CompiledKernel, compile_request


def _options(options, changes):
    if options is not None:
        if changes:
            raise TypeError("pass either CompileOptions or keyword options, not both")
        if not isinstance(options, CompileOptions):
            raise TypeError("options must be a CompileOptions instance")
        return options
    return CompileOptions(**changes)


def _check_launch(binding):
    """Require the launch to cover every public tensor domain exactly."""
    grid, block = binding.launch.grid, binding.launch.block
    if grid[2] != 1 or block[2] != 1:
        raise TensorContractError(
            "the frontend ABI launches a rank-two domain; grid z and block z "
            "must both be one",
            code="invalid_launch_coverage",
        )
    for spec in binding.tensors:
        if (
            grid[0] * block[0] != spec.shape[0]
            or grid[1] * block[1] != spec.shape[1]
        ):
            raise TensorContractError(
                f"launch grid {grid} with block {block} does not cover tensor "
                f"{spec.name} of shape {spec.shape}",
                code="invalid_launch_coverage",
            )


def _providers(binding):
    inputs, outputs = [], []
    for spec, provider in zip(binding.tensors, binding.providers):
        if provider is None:
            raise TensorContractError(
                f"tensor {spec.name} requires a caller buffer provider",
                code="invalid_tensor_binding",
            )
        (inputs if spec.direction == "input" else outputs).append(provider)
    return tuple(inputs), tuple(outputs)


class _GridLauncher:
    """``kernel[grid]`` block-programming form."""

    def __init__(self, kernel, grid):
        self._kernel = kernel
        self._grid = grid

    def __call__(self, *args, **kwargs):
        if "grid" in kwargs:
            raise TypeError("grid passed twice: kernel[grid](..., grid=...)")
        return self._kernel(*args, grid=self._grid, **kwargs)

    def compile(self, *args, **kwargs):
        if "grid" in kwargs:
            raise TypeError("grid passed twice: kernel[grid].compile(..., grid=...)")
        return self._kernel.compile(*args, grid=self._grid, **kwargs)

    def __repr__(self):
        return f"<GridLauncher grid={self._grid!r} kernel={self._kernel!r}>"


class JitFunction:
    """Kernel definition with typed capture and synchronous native invocation."""

    def __init__(self, fn: Callable, options=None, **changes):
        self._fn = fn
        self._options = _options(options, changes)

    def __getitem__(self, grid):
        return _GridLauncher(self, grid)

    def __call__(self, *args, grid=None, **kwargs):
        compiled, providers = self._build(args, kwargs, grid)
        try:
            compiled.invoke(*providers)
        finally:
            compiled.close()

    def compile(self, *args, grid=None, **kwargs) -> CompiledKernel:
        """Compile without invoking; the caller owns the returned handle.

        Tensor arguments may be metadata-only ``TensorSpec`` records here; the
        actual provider buffers are supplied to ``invoke``.
        """
        compiled, _ = self._build(args, kwargs, grid, require_providers=False)
        return compiled

    def _build(self, args, kwargs, grid, *, require_providers=True):
        if grid is None:
            raise TensorContractError(
                "a launch grid is required; use kernel[grid](...) or "
                "kernel(..., grid=grid)",
                code="invalid_launch_coverage",
            )
        binding = bind_kernel(self._fn, args, kwargs, grid, self._options)
        _check_launch(binding)
        program = compile_kernel(self._fn, binding)
        options = replace(binding.options, numerical_contract=program.numerical_contract)
        compiled = compile_request(CompileRequest(program, options))
        try:
            providers = _providers(binding) if require_providers else None
        except BaseException:
            compiled.close()
            raise
        return compiled, providers

    def frontend_ir(self, binding):
        """Return checked frontend metadata for an explicit frozen binding."""
        return compile_kernel(self._fn, binding)

    def ttir(self) -> str:
        raise UnsupportedFeatureError(
            "frontend metadata is not upstream TTIR; the upstream adapter is unqualified",
            code="upstream_toolchain_unqualified",
        )

    def __repr__(self) -> str:
        return f"<JitFunction {self._fn.__name__}>"


def jit(fn: Callable | None = None, options=None, **changes) -> JitFunction | Callable:
    """Mark a kernel for typed capture and native compilation.

    ``@llk.jit`` and ``@llk.jit()`` both work. Keyword options are bound into a
    frozen ``CompileOptions``; for example ``@llk.jit(backend="reference")``.
    """
    if fn is not None:
        return JitFunction(fn, options, **changes)

    def decorator(target: Callable) -> JitFunction:
        return JitFunction(target, options, **changes)

    return decorator
