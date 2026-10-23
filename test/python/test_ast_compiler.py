from dataclasses import replace

from ast_cases import binding, counted_dot, masked_add
from llk.frontend import (
    CompileRequest,
    FrontendProgram,
    deserialize_request,
    serialize_request,
)
from llk.language._ast_compiler import compile_kernel


def test_compile_elementwise_kernel():
    bound = binding(masked_add)
    program = compile_kernel(masked_add, bound)
    assert isinstance(program, FrontendProgram)
    assert [(t.name, t.effect) for t in program.tensors] == [
        ("a", "read"),
        ("b", "read"),
        ("out", "write"),
    ]
    assert sum(op.opcode == "load" for op in program.operations) == 2
    assert sum(op.opcode == "store" for op in program.operations) == 1
    request = CompileRequest(program, bound.options)
    assert deserialize_request(serialize_request(request)) == request


def test_compile_kernel_with_dot():
    bound = binding(counted_dot)
    program = compile_kernel(counted_dot, bound)
    loop = next(op for op in program.operations if op.opcode == "counted_loop")
    assert sum(op.opcode == "dot" for op in loop.regions[0].operations) == 1
    assert len(program.numerical_contract.cast_sites) == 2
    request = CompileRequest(
        program, replace(bound.options, numerical_contract=program.numerical_contract)
    )
    assert deserialize_request(serialize_request(request)) == request
