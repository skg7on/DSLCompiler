import pytest


def test_compile_elementwise_kernel():
    """AST compiler should produce valid Triton MLIR for a simple kernel."""
    from llk.language._ast_compiler import compile_kernel
    import llk.language as tl

    def add_kernel(x_ptr, y_ptr, N: int, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < N
        x = tl.load(x_ptr + offs, mask=mask)
        tl.store(y_ptr + offs, x, mask=mask)

    ttir = compile_kernel(add_kernel)

    assert "tt.func" in ttir
    assert "tt.get_program_id" in ttir
    assert "tt.make_range" in ttir
    assert "tt.load" in ttir
    assert "tt.store" in ttir


def test_compile_kernel_with_dot():
    """AST compiler should handle tl.dot and tl.zeros."""
    from llk.language._ast_compiler import compile_kernel
    import llk.language as tl

    def matmul_kernel(
        A_ptr, B_ptr, C_ptr,
        M: int, N: int, K: int,
        BLOCK: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offs_am = pid * BLOCK + tl.arange(0, BLOCK)
        offs_bn = pid * BLOCK + tl.arange(0, BLOCK)
        acc = tl.zeros((BLOCK, BLOCK), dtype=tl.float32)
        a = tl.load(A_ptr + offs_am[:, None] * K + offs_bn[None, :], mask=True)
        b = tl.load(B_ptr + offs_am[:, None] * K + offs_bn[None, :], mask=True)
        acc = tl.dot(a, b, acc)

    ttir = compile_kernel(matmul_kernel)
    assert "tt.dot" in ttir
    assert "tt.splat" in ttir
