# Milestone 7: Python Frontend — Design Spec (Stub)

**Status:** Planned, not yet scheduled

## Overview

Two complementary entry paths converge at the LLK dialect:

- **Explicit DSL:** Triton-inspired `@llk.jit` decorator with block-programming model
- **PyTorch Capture:** `torch.compile(backend="llk")` or `torch.export` → pattern recognition

All expensive optimization, code generation, caching, and runtime execution remains in C++.

## DSL Design (Sketch)

```python
import llk
import llk.language as tl

@llk.jit
def fused_swiglu(x: tl.Tensor[M, K], wg: tl.Tensor[K, N], wu: tl.Tensor[K, N]) -> tl.Tensor[M, N]:
    gate = tl.dot(x, wg)
    up = tl.dot(x, wu)
    gate = tl.silu(gate)
    return gate * up
```

## PyTorch Capture (Sketch)

```python
import torch
import llk.torch

model = MyModel().cuda()
opt_model = torch.compile(model, backend="llk")
output = opt_model(input_data)
```

## Implementation Plan

1. MLIR Python bindings for the LLK dialect (`python/llk/dialect/`)
2. AST capture via `inspect.getsource()` and Python AST → LLK IR lowering
3. PyTorch Dynamo backend registration (`python/llk/torch/`)
4. ATen → LLK op pattern matching (SwiGLU fusion pattern, RoPE pattern, attention pattern)
5. JIT invocation from Python via ctypes/cffi to the C++ runtime

## Dependencies

- All M1–M6 C++ infrastructure complete and stable
- MLIR Python bindings (from LLVM/MLIR build)
- PyTorch 2.0+ (for `torch.compile` integration)

## References

- [ARCHITECTURE.md](../../ARCHITECTURE.md) §2.2 Python Frontend
- Triton DSL: https://triton-lang.org/
- PyTorch Dynamo: https://pytorch.org/docs/stable/torch.compiler.html
