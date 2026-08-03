"""Triton MLIR dialect IR builder.

Builds Triton MLIR operations as strings that will be parsed by the
C++ pipeline. In a future iteration, this will use MLIR Python bindings
to construct ops directly."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class IRValue:
    """Placeholder for a Triton IR SSA value."""
    name: str
    type_str: str = "!tt.ptr"


class TritonIRBuilder:
    """Builds a Triton MLIR module one op at a time."""

    def __init__(self) -> None:
        self._counter = 0
        self._ops: list[str] = []
        self._func_name: Optional[str] = None

    def _next_name(self, prefix: str = "v") -> str:
        self._counter += 1
        return f"%{prefix}{self._counter}"

    def start_function(self, name: str, args: list[str]) -> None:
        self._func_name = name
        arg_str = ", ".join(args)
        self._ops.append(f"  tt.func @{name}({arg_str}) {{")

    def end_function(self) -> None:
        self._ops.append("    tt.return")
        self._ops.append("  }")

    def make_range(self, start: int, end: int) -> IRValue:
        """Emit tt.make_range {{start = {start}, end = {end}}}."""
        result = self._next_name("range")
        if isinstance(end, str) or isinstance(start, str):
            size_str = f"({end} - {start})"
        else:
            size_str = str(end - start)
        self._ops.append(
            f"    {result} = tt.make_range {{start = {start}, end = {end}}}"
            f" : tensor<{size_str}xindex>"
        )
        return IRValue(name=result, type_str=f"tensor<{size_str}xindex>")

    def get_program_id(self, axis: int) -> IRValue:
        """Emit tt.get_program_id {{axis = {axis}}}."""
        result = self._next_name("pid")
        self._ops.append(
            f"    {result} = tt.get_program_id {{axis = {axis}}} : index"
        )
        return IRValue(name=result, type_str="index")

    def load(self, ptr: str, mask: Optional[str] = None) -> IRValue:
        result = self._next_name("loaded")
        mask_str = f" {{mask = {mask}}}" if mask else ""
        self._ops.append(f"    {result} = tt.load {ptr}{mask_str} : !tt.ptr -> f32")
        return IRValue(name=result, type_str="f32")

    def store(self, ptr: str, value: str, mask: Optional[str] = None) -> None:
        mask_str = f" {{mask = {mask}}}" if mask else ""
        self._ops.append(f"    tt.store {ptr}, {value}{mask_str} : !tt.ptr, f32")

    def splat(self, shape: tuple, dtype: str) -> IRValue:
        result = self._next_name("splat")
        shape_str = "x".join(str(s) for s in shape)
        self._ops.append(
            f"    {result} = tt.splat {{shape = [{shape_str}], dtype = {dtype}}}"
        )
        return IRValue(name=result, type_str=f"tensor<{shape_str}x{dtype}>")

    def dot(self, a: str, b: str, acc: Optional[str] = None) -> IRValue:
        result = self._next_name("dot")
        acc_str = f", {acc}" if acc else ""
        self._ops.append(f"    {result} = tt.dot {a}, {b}{acc_str} : f32")
        return IRValue(name=result, type_str="f32")

    @property
    def module(self) -> str:
        lines = ['"builtin.module"() ({']
        for op in self._ops:
            lines.append(op)
        lines.append("}) : () -> ()")
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.module
