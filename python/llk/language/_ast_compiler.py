"""Python AST visitor that lowers @llk.jit functions to Triton MLIR.

Walks the AST of a captured function and emits Triton IR ops via
TritonIRBuilder. Recognizes tl.* calls, Python control flow, and
tl.constexpr annotations.
"""

import ast
import inspect
import textwrap
from typing import Any

from ._triton_ir import TritonIRBuilder


class _TritonASTVisitor(ast.NodeVisitor):
    """Walks a Python function AST and emits Triton MLIR."""

    def __init__(self, builder: TritonIRBuilder):
        self.builder = builder
        self._constexprs: dict[str, int] = {}
        self._params: list[str] = []

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        # Extract parameters and constexpr annotations
        for arg in node.args.args:
            self._params.append(arg.arg)
            if arg.annotation:
                self._check_constexpr(arg.arg, arg.annotation)

        self.builder.start_function(node.name, self._params)
        # Visit body statements
        for stmt in node.body:
            self.visit(stmt)
        self.builder.end_function()

    def _check_constexpr(self, name: str, annotation: ast.expr) -> None:
        """Extract tl.constexpr-annotated parameters."""
        if (isinstance(annotation, ast.Attribute) and
            isinstance(annotation.value, ast.Name) and
            annotation.value.id == 'tl' and
            annotation.attr == 'constexpr'):
            self._constexprs[name] = None  # value known at compile time

    def visit_Assign(self, node: ast.Assign) -> None:
        for target in node.targets:
            if isinstance(target, ast.Name):
                self.visit(node.value)

    def visit_Call(self, node: ast.Call) -> Any:
        # Dispatch tl.* calls
        if isinstance(node.func, ast.Attribute):
            if (isinstance(node.func.value, ast.Name) and
                node.func.value.id == 'tl'):
                return self._handle_tl_call(node.func.attr, node.args, node.keywords)
        return None

    def _handle_tl_call(self, name: str, args: list[ast.expr],
                        keywords: list[ast.keyword]) -> Any:
        """Map tl.xxx() calls to Triton IR ops."""
        if name == 'program_id':
            axis = self._eval_literal(args[0])
            return self.builder.get_program_id(axis)
        elif name == 'arange':
            start = self._eval_literal(args[0])
            end = self._eval_literal(args[1])
            return self.builder.make_range(start, end)
        elif name == 'load':
            self.builder.load(str(args[0]), str(args[1]) if len(args) > 1 else None)
        elif name == 'store':
            self.builder.store(str(args[0]), str(args[1]),
                              str(args[2]) if len(args) > 2 else None)
        elif name == 'zeros':
            shape = self._eval_literal(args[0])
            dtype = self._eval_kwarg_str(keywords, 'dtype', 'float32')
            return self.builder.splat(shape, dtype)
        elif name == 'dot':
            return self.builder.dot(str(args[0]), str(args[1]),
                                    str(args[2]) if len(args) > 2 else None)
        return None

    @staticmethod
    def _eval_literal(node: ast.expr) -> Any:
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.Name):
            return node.id  # e.g. BLOCK_SIZE — pass name through
        if isinstance(node, ast.Tuple):
            return tuple(_TritonASTVisitor._eval_literal(el) for el in node.elts)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            return -_TritonASTVisitor._eval_literal(node.operand)
        return str(node)  # fallback for complex expressions

    @staticmethod
    def _eval_kwarg_str(keywords: list[ast.keyword], key: str,
                        default: str) -> str:
        for kw in keywords:
            if kw.arg == key:
                return str(_TritonASTVisitor._eval_literal(kw.value))
        return default


def compile_kernel(fn) -> str:
    """Capture a Python function via inspect.getsource and lower to Triton MLIR.

    Args:
        fn: The @llk.jit-decorated function.

    Returns:
        Triton MLIR module as a string.
    """
    source = inspect.getsource(fn)
    source = textwrap.dedent(source)
    tree = ast.parse(source)

    builder = TritonIRBuilder()
    visitor = _TritonASTVisitor(builder)

    # Find the function definition in the module
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == fn.__name__:
            visitor.visit(node)
            break

    return str(builder)
