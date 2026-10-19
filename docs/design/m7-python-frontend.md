# Milestone 7: Python Frontend — Design Spec (Stub)

**Scope alignment (2026-10-12):** This is a CPU validation milestone or frontend-adapter proposal within the [generic MicroIR-based DNN compiler framework](../superpowers/specs/2026-10-12-generic-microir-dnn-compiler-framework-design.md). SwiGLU, matmul, RoPE and attention are example workloads; operator-specific algorithms, schedule fields, ABI and performance criteria below apply to these fixtures rather than the whole framework. Original milestone status and design intent are retained as history.

**Current status (2026-10-19):** This historical sketch is superseded by the
[frontend refinement design](../superpowers/specs/2026-10-18-python-frontend-refinement-design.md).
PF01 packaging, optional imports, idempotent registration and explicit unavailable
errors are implemented in [PR #233](https://github.com/skg7on/DSLCompiler/pull/233).
Use the [current frontend guide](../manual/source/python-frontend.rst) for the
project `.venv` and tested versions. Native Python compilation, Inductor and
upstream Triton execution remain later tasks under #203. The sketches below,
including the CUDA example and MLIR-binding dependency, describe original intent
and are not runnable PF01 examples or current package prerequisites.

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
