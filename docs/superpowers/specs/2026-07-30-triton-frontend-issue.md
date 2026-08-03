# M8: Triton-Compatible Python Frontend with Multi-Stage Lowering to LLK Dialect

**Type:** Feature / Epic
**Status:** Proposal
**Depends on:** M1-M6 (complete)

## Summary

Add a Triton-compatible Python frontend (`@llk.jit`) and a 5-stage lowering pipeline that compiles Triton MLIR IR through the existing LLK compiler backend, enabling CPU execution of Triton kernels for LLM inference.

## Motivation

The project currently requires hand-authoring LLK dialect IR or C++ builder code to express kernels. This is a high barrier for kernel authors accustomed to Triton's block-programming model. The `m7-python-frontend.md` is a 53-line stub with no implementation detail.

A proper Triton frontend:
- Lets kernel authors write CPU kernels in the Triton DSL they already know
- Enables reuse of existing open-source Triton kernels for LLM inference
- Provides a Python-accessible JIT compilation path
- Demonstrates the LLK compiler backend generalizes to a standard frontend

## Design Principles

1. **Triton-compatible.** The DSL mirrors `triton.language`. Real Triton kernels should run with minimal modification. GPU→CPU semantic gaps caught at compile time.

2. **Multi-stage lowering.** Five small, independently testable passes — one per GPU→CPU semantic gap — rather than one monolithic conversion. Each pass is ~500-800 lines and FileCheck-testable.

3. **Converge at LLK dialect.** The Triton lowering produces LLK + Linalg + Vector IR. Known fusion patterns (SwiGLU, RoPE, Attention) recognized and emitted as `llk.*` ops. No parallel lowering path — reuse the existing M1-M6 pipeline unchanged from Linalg downward.

4. **Hybrid block→thread execution.** Triton blocks map to CPU thread pool tasks. Shared memory maps to per-thread scratch. Vector lanes provide intra-block SIMD. Pattern detection identifies accumulation tiles and lowers to explicit register micro-tiles.

5. **Triton tile sizes respected.** `BLOCK_M`/`BLOCK_N`/`BLOCK_K` from `tl.constexpr` are the default schedule parameters. The schedule DB provides vector micro-tile (`VM`/`VN`) and parallel grain — Triton controls the outer tile, the compiler controls the inner micro-kernel.

6. **Triton IR as interchange format.** All kernels flow through Triton MLIR IR — inspectable, testable at each stage, and compatible with upstream Triton's frontend in the future.

## Proposed Architecture

```
Python DSL (triton-compatible)
    ↓ inspect.getsource() + Python AST
Triton MLIR IR
    ↓
Stage 1: TritonToStructured     — tt.dot → linalg.matmul / llk.fused_swiglu
Stage 2: BlockPointerToVector   — tt.make_block_ptr → vector.transfer_read/write
Stage 3: SharedMemToScratch     — tt.alloc → memref.alloc (per-thread scratch)
Stage 4: AtomicToLLRT           — tt.atomic_* → llrt.atomic_* runtime calls
Stage 5: GridToForall           — tt.get_program_id → scf.forall tile decomposition
    ↓
LLK + Linalg + Vector IR → Existing M1-M6 pipeline → ORC JIT
```

## Milestones

| Milestone | Scope | Exit Criterion |
|-----------|-------|----------------|
| **M8a: Python DSL + Triton IR** | `@llk.jit`, `tl.load/store/dot`, AST→Triton IR | Valid Triton MLIR emitted from Python |
| **M8b: SwiGLU from Python** | Stages 1+5, SwiGLU pattern, ABI bridge | SwiGLU kernel compiles + JIT-executes via Python |
| **M8c: Block pointer lowering** | Stage 2, vector transfers, mask gen | Elementwise Triton kernels work |
| **M8d: Shared memory + atomics** | Stages 3+4, scratch allocation, atomics | Matmul + flash-attention patterns work |
| **M8e: Full compatibility** | All stages, verifier, schedule DB, torch.compile | 10+ real Triton kernels compile; ≥80% pass rate |

## Scope: Full Block Programming Model

`tl.load/store/dot/arange/program_id`, `tl.make_block_ptr/advance`, `tl.alloc/async_copy`, `tl.atomic_*`, `tl.reduce/sum/max`, all elementwise math. No warp-level primitives (`tl.inline_asm`, MMA layout constraints) — these are rejected with clear errors.

## Key Design Decisions

See the full spec: `docs/superpowers/specs/2026-07-30-triton-frontend-design.md`

- **5 stages, not 1 monolithic pass** — each independently testable, composable, and ~500-800 lines
- **Pattern recognition order:** SwiGLU fusion pattern checked before generic `linalg.matmul` fallback
- **`llk.make_tensor` op** bridges Triton's pointer-based ABI and MLIR's tensor pipeline
- **Schedule DB extension:** `"inherit_from_triton"` for BM/BN/BK; DB provides VM/VN/grain
- **TritonCPUVerifier pass:** catches unsupported GPU ops at compile time with actionable errors
- **Synchronous shared memory:** `tt.async_copy` → `memref.copy` (no barrier needed on CPU)
- **Grid linearization:** 1D/2D/3D grids all map to linearized `scf.forall`

## Files Changed

- **New:** 6 C++ lowering passes + verifier (~5000 lines)
- **New:** Python package `python/llk/` (~2000 lines)
- **New:** ~15 FileCheck tests + 7 end-to-end tests
- **Modified:** `LLKOps.td` (2 new ops), `LLKDialect.td` (new math mode), `KernelContext.h` (Triton fields), `schedule_db.json` (Triton entries)
- **New dependency:** Triton MLIR dialect (upstream or vendored)
