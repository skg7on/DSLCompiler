"""Deterministic metadata binding; storage adaptation and invocation are PF08."""

from __future__ import annotations

import ast
import inspect
import textwrap
from dataclasses import dataclass
from types import FunctionType, MappingProxyType, ModuleType
from typing import Mapping

import llk.language as language
from llk._errors import (
    LLKError,
    TensorContractError,
    UnsupportedFeatureError,
)

from ._ir import (
    CompileOptions,
    LaunchSpec,
    SourceSpan,
    TensorSpec,
    constants,
    fail,
)


@dataclass(frozen=True, slots=True)
class TensorArgument:
    """Explicit metadata plus a strongly retained, never serialized owner.

    The provider is deliberately opaque here. The invocation adapter must check
    its real dtype/layout/offset/allocation/alias metadata before native use.
    """

    spec: TensorSpec
    provider: object

    def __post_init__(self):
        if not isinstance(self.spec, TensorSpec):
            fail("TensorArgument requires a TensorSpec", error=TensorContractError)


@dataclass(frozen=True, slots=True)
class KernelBinding:
    providers: tuple[object, ...]
    tensors: tuple[TensorSpec, ...]
    constexprs: Mapping
    captured_constants: Mapping
    launch: LaunchSpec
    options: CompileOptions
    source: str
    span: SourceSpan
    symbols: Mapping

    def __post_init__(self):
        object.__setattr__(self, "providers", tuple(self.providers))
        object.__setattr__(self, "tensors", tuple(self.tensors))
        object.__setattr__(self, "constexprs", constants(self.constexprs))
        object.__setattr__(
            self, "captured_constants", constants(self.captured_constants)
        )
        object.__setattr__(self, "symbols", MappingProxyType(dict(self.symbols)))


def annotation_value(annotation, namespace):
    """Resolve name/attribute references without eval or user attribute hooks."""
    if annotation is language.constexpr or annotation is inspect.Parameter.empty:
        return annotation
    if not isinstance(annotation, str):
        return annotation
    for _ in range(8):
        try:
            tree = ast.parse(annotation, mode="eval").body
        except (SyntaxError, ValueError):
            fail("invalid annotation expression", code="unsupported_annotation")
        if isinstance(tree, ast.Constant) and isinstance(tree.value, str):
            annotation = tree.value
            continue

        def resolve(node):
            if isinstance(node, ast.Name) and node.id in namespace:
                return namespace[node.id]
            if isinstance(node, ast.Attribute):
                owner = resolve(node.value)
                if isinstance(owner, ModuleType) and node.attr in vars(owner):
                    return vars(owner)[node.attr]
            fail(
                "annotations must resolve by symbol identity",
                code="unsupported_annotation",
            )

        return resolve(tree)
    fail("annotation nesting exceeds schema limits", code="unsupported_annotation")


def source_of(fn):
    code = getattr(fn, "__code__", None)
    span = SourceSpan(
        getattr(code, "co_filename", "<unknown>"),
        max(1, getattr(code, "co_firstlineno", 1)),
        0,
    )
    if not inspect.isfunction(fn):
        fail(
            "kernel must be a Python function",
            span,
            code="unsupported_frontend_call",
            error=UnsupportedFeatureError,
        )
    try:
        lines, line = inspect.getsourcelines(fn)
    except (OSError, TypeError):
        fail(
            "kernel source is unavailable",
            span,
            code="source_unavailable",
            error=UnsupportedFeatureError,
        )
    indentation = len(lines[0]) - len(lines[0].lstrip())
    return textwrap.dedent("".join(lines)), SourceSpan(span.file, line, indentation)


def kernel_signature(fn, source):
    # Even annotationlib's STRING format can execute deferred annotation calls
    # on Python 3.14. Inspect an annotation-free function with the same calling
    # convention; never invoke the original annotator or user __signature__.
    bare = FunctionType(
        fn.__code__, fn.__globals__, fn.__name__, fn.__defaults__, fn.__closure__
    )
    bare.__kwdefaults__ = fn.__kwdefaults__
    signature = inspect.signature(bare, follow_wrapped=False, eval_str=False)
    annotator = getattr(fn, "__annotate__", None)
    if annotator is None:
        annotations = fn.__annotations__
    else:
        tree = ast.parse(source)
        definition = next(
            (
                node
                for node in tree.body
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            ),
            None,
        )
        if definition is None:
            fail(
                "kernel annotation source is unavailable", code="unsupported_annotation"
            )
        parameters = (
            definition.args.posonlyargs
            + definition.args.args
            + definition.args.kwonlyargs
        )
        annotations = {
            parameter.arg: ast.get_source_segment(source, parameter.annotation)
            for parameter in parameters
            if parameter.annotation is not None
        }
    return signature.replace(
        parameters=[
            parameter.replace(annotation=annotations.get(name, inspect.Parameter.empty))
            for name, parameter in signature.parameters.items()
        ]
    ), annotator


def bind_kernel(
    fn, args: tuple, kwargs: dict, grid, options: CompileOptions
) -> KernelBinding:
    source, span = source_of(fn)
    try:
        if not isinstance(options, CompileOptions):
            fail("binding requires CompileOptions")
        if not isinstance(args, tuple) or not isinstance(kwargs, dict):
            fail("arguments must be an ordered tuple and keyword mapping")
        signature, annotator = kernel_signature(fn, source)
        if any(
            p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD)
            for p in signature.parameters.values()
        ):
            fail(
                "variadic kernel parameters are unsupported",
                code="invalid_argument_binding",
            )
        try:
            bound = signature.bind(*args, **kwargs)
        except TypeError as error:
            fail(str(error), code="invalid_argument_binding")
        bound.apply_defaults()
        closure = inspect.getclosurevars(fn)
        # Annotations may reference a renamed sentinel that is not loaded by
        # the function body and hence is absent from getclosurevars().
        namespace = dict(fn.__globals__)
        namespace.update(closure.nonlocals)
        if inspect.isfunction(annotator) and annotator.__closure__:
            namespace.update(
                zip(
                    annotator.__code__.co_freevars,
                    (cell.cell_contents for cell in annotator.__closure__),
                )
            )
        if fn.__closure__:
            namespace.update(
                zip(
                    fn.__code__.co_freevars,
                    (cell.cell_contents for cell in fn.__closure__),
                )
            )
        providers = []
        tensors = []
        constexprs = {}
        captured = {}
        symbols = {}
        for name, parameter in signature.parameters.items():
            annotation = annotation_value(parameter.annotation, namespace)
            if annotation is language.constexpr:
                constexprs[name] = constants({name: bound.arguments[name]})[name]
                continue
            if annotation is not inspect.Parameter.empty:
                fail(f"unsupported annotation on {name}", code="unsupported_annotation")
            value = bound.arguments[name]
            if isinstance(value, TensorArgument):
                spec, provider = value.spec, value.provider
            elif isinstance(value, TensorSpec):
                spec, provider = value, None
            else:
                fail(
                    f"{name} requires TensorSpec or TensorArgument metadata",
                    error=TensorContractError,
                    code="invalid_tensor_binding",
                )
            # The signature owns public port names; never guess effects from a
            # spelling such as 'out', or copy/convert provider data implicitly.
            spec = TensorSpec(
                name, spec.shape, spec.dtype, spec.strides, spec.direction, spec.effect
            )
            tensors.append(spec)
            providers.append(provider)
        language_symbols = tuple(
            value
            for name, value in vars(language).items()
            if not name.startswith("_")
            and inspect.isfunction(value)
            and value.__module__ == language.__name__
        )
        for name, value in {
            **closure.globals,
            **closure.nonlocals,
            **closure.builtins,
        }.items():
            if (
                value is language
                or value is language.constexpr
                or any(value is symbol for symbol in language_symbols)
                or value is range
            ):
                symbols[name] = value
            else:
                try:
                    captured.update(constants({name: value}))
                except LLKError:
                    tree = ast.parse(source)
                    references = [
                        node
                        for node in ast.walk(tree)
                        if isinstance(node, ast.Name)
                        and node.id == name
                        and isinstance(node.ctx, ast.Load)
                    ]
                    calls = []
                    for call in (
                        node for node in ast.walk(tree) if isinstance(node, ast.Call)
                    ):
                        root = call.func
                        while isinstance(root, ast.Attribute):
                            root = root.value
                        if isinstance(root, ast.Name) and root.id == name:
                            calls.append(call)
                    reference = min(
                        calls or references,
                        key=lambda n: (n.lineno, n.col_offset),
                        default=None,
                    )
                    location = (
                        span
                        if reference is None
                        else SourceSpan(
                            span.file,
                            span.line + reference.lineno - 1,
                            span.column + reference.col_offset,
                        )
                    )
                    fail(
                        f"unsupported mutable or dynamic capture: {name}",
                        location,
                        code="unsupported_frontend_call"
                        if calls
                        else "unsupported_frontend_constant",
                        error=UnsupportedFeatureError,
                    )
        if callable(grid):
            # Invoke exactly once, after all argument/constant validation. Only
            # immutable constexpr metadata is made available to the callable.
            grid = grid(MappingProxyType(dict(constexprs)))
        if isinstance(grid, LaunchSpec):
            launch = grid
        else:
            if not isinstance(grid, tuple) or not 1 <= len(grid) <= 3:
                fail(
                    "grid must have one to three integer axes",
                    code="invalid_launch_coverage",
                )
            launch = LaunchSpec(grid + (1,) * (3 - len(grid)), (1, 1, 1))
        return KernelBinding(
            tuple(providers),
            tuple(tensors),
            constexprs,
            captured,
            launch,
            options,
            source,
            span,
            symbols,
        )
    except LLKError as error:
        if error.file is None:
            error.file, error.line, error.column = span.file, span.line, span.column
        raise
    except (ValueError, TypeError, OverflowError) as error:
        fail(f"invalid kernel binding: {error}", span, code="invalid_argument_binding")
