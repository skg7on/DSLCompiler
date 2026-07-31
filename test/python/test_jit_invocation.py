import pytest


def test_jit_call_emits_triton_ir():
    """Calling a @llk.jit function should capture AST and emit Triton IR."""
    import llk
    import llk.language as tl

    @llk.jit
    def elementwise_add(x_ptr, y_ptr, N: int, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < N
        x = tl.load(x_ptr + offs, mask=mask)
        tl.store(y_ptr + offs, x, mask=mask)

    # On first call, should attempt compilation
    result = elementwise_add.ttir()

    assert "tt.func" in result
    assert "tt.get_program_id" in result
    assert "tt.load" in result
