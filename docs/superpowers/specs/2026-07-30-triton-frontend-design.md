# Triton Frontend + LLK Backend — Design Spec

**Status:** Design Approved | **Date:** 2026-07-30
**Parent:** [ARCHITECTURE.md](../../../ARCHITECTURE.md) — extends §2.2 Python Frontend
**Supersedes:** [m7-python-frontend.md](../../design/m7-python-frontend.md) (stub)

---

## 1. Overview

### 1.1 Goal

Add a Triton-compatible Python frontend to the LLK compiler with a staged lowering from Triton MLIR to the LLK dialect, enabling existing Triton kernels to compile and execute on CPU via the established M1-M6 pipeline.

### 1.2 Design Principles

1. **Triton-compatible, not just Triton-inspired.** The DSL surface mirrors `triton.language`. Real Triton kernels should run with minimal modification. Semantic gaps (GPU→CPU) are caught at compile time with actionable errors, not silent divergence.

2. **Staged lowering over monolithic passes.** Each GPU→CPU semantic gap (shared memory, block pointers, atomics, grid dispatch, compute ops) is its own pass — independently testable, progressively buildable, and composable. Passes that match no patterns are no-ops.

3. **Converge at the LLK dialect, not beside it.** The Triton lowering produces LLK + Linalg + Vector IR that flows through the existing pipeline. Known fusion patterns (SwiGLU, RoPE, Attention) are recognized and emitted as `llk.*` ops. Generic patterns become `linalg.*` ops. No parallel lowering path.

4. **Block→thread hybrid execution.** Triton blocks map to CPU thread pool tasks. Shared memory maps to per-thread scratch. Vector lanes provide the intra-block SIMD parallelism Triton expects from warps. Pattern detection identifies matmul accumulation tiles and lowers them to explicit `VM×VN` register tiles.

5. **Schedule DB extends, not replaces, Triton tile sizes.** Triton's `BLOCK_M`/`BLOCK_N`/`BLOCK_K` constants are the default tile sizes. The schedule DB provides the vector micro-tile (`VM`/`VN`) and parallel grain. Triton authors control the outer schedule; the compiler controls the inner micro-kernel.

6. **Triton IR as the interchange format.** All kernels flow through Triton MLIR IR. This makes the lowering inspectable, testable at each stage, and enables eventual reuse of upstream Triton's Python frontend.

### 1.3 Scope — Full Block Programming Model

| Feature | Included? |
|---------|-----------|
| `tl.load`, `tl.store` (with masking) | ✅ |
| `tl.dot` (with accumulator) | ✅ |
| `tl.arange`, `tl.program_id` | ✅ |
| `tl.make_block_ptr`, `tl.advance` | ✅ |
| `tl.alloc` (shared memory), `tl.async_copy` | ✅ |
| `tl.atomic_*` | ✅ |
| `tl.reduce`, `tl.sum`, `tl.max` | ✅ |
| Elementwise math (`tl.exp`, `tl.silu`, `tl.sigmoid`, `tl.where`, etc.) | ✅ |
| Warp-level primitives (`tl.inline_asm`, MMA layout constraints) | ❌ Rejected with error |
| Modulo constraints on block pointers | ⚠️ Warned, not enforced |

---

## 2. Pipeline Architecture

### 2.1 Overall Flow

```
┌──────────────────────────────────────────────────────┐
│                  NEW: Triton Frontend (M8)            │
│                                                      │
│  Python DSL (triton-compatible)                      │
│      ↓ inspect.getsource() + Python AST              │
│  Triton MLIR IR (tt.load, tt.dot, tt.make_block_ptr) │
│      ↓ 5-stage lowering                              │
│  LLK Dialect + Linalg (existing pipeline entry)      │
└──────────────────────────────────────────────────────┘
     │
     ▼
┌──────────────────────────────────────────────────────┐
│              EXISTING: M1-M6 Pipeline                 │
│                                                      │
│  LLKToLinalg → shape-specialize → select-schedule     │
│  → fuse → tile-and-vectorize → parallel-decompose     │
│  → One-Shot Bufferize → scratch-analysis              │
│  → convert-to-llvm → ORC JIT                         │
└──────────────────────────────────────────────────────┘
```

### 2.2 Five-Stage Lowering Detail

```
Triton MLIR IR
    │
    ▼
┌──────────────────────────────────────────┐
│ Stage 1: TritonToStructured              │
│ tt.dot → linalg.matmul                   │
│ tt.dot × 2 + silu → llk.fused_swiglu     │
│ tt.dot + softmax + tt.dot → llk.attention│
│ tt.reduce → linalg.reduce                │
│ Elementwise → linalg.generic             │
└──────────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────────┐
│ Stage 2: BlockPointerToVector            │
│ tt.make_block_ptr → vector.transfer_read │
│ tt.advance → index arithmetic (folded)   │
│ tt.load → vector.transfer_read + mask    │
│ tt.store → vector.transfer_write + mask  │
└──────────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────────┐
│ Stage 3: SharedMemToScratch              │
│ tt.alloc (shared memory) → memref.alloc  │
│ tt.async_copy → memref.copy (sync)       │
└──────────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────────┐
│ Stage 4: AtomicToLLRT                    │
│ tt.atomic_add → llrt.atomic_add_f32      │
│ tt.atomic_max → llrt.atomic_max_f32      │
│ tt.atomic_cas → llrt.atomic_cas          │
└──────────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────────┐
│ Stage 5: GridToForall                    │
│ tt.get_program_id → scf.forall tile idx  │
│ 1D/2D/3D grid → linearized forall       │
└──────────────────────────────────────────┘
    │
    ▼
LLK + Linalg + Vector IR → Existing M1-M6 pipeline
```

Stages are ordered but independently optional — a simple elementwise kernel only exercises stages 2+5. Stage 1 runs `canonicalize` before pattern matching so dead-code elimination removes individual `tt.dot` ops after fusion recognition.

### 2.3 TritonCPUVerifier Pass

Runs after Stage 1 and before Stage 2. Checks every Triton IR op against a CPU-support whitelist. Emits actionable errors for unsupported patterns at compilation time rather than JIT execution.

| Op/Feature | Status |
|------------|--------|
| `tt.dot` (without MMA layout) | ✅ Supported |
| `tt.dot` with MMA layout | ❌ Rejected: "use tl.dot without mma layout for CPU" |
| `tt.load`, `tt.store` | ✅ Supported |
| `tt.make_block_ptr`, `tt.advance` | ✅ Supported |
| `tt.alloc`, `tt.async_copy` | ✅ Supported |
| `tt.atomic_add`, `tt.atomic_max`, `tt.atomic_cas` | ✅ Supported |
| `tt.inline_asm` | ❌ Rejected |
| Warp-group ops, `tl.inline_asm` | ❌ Rejected |

---

## 3. Python Frontend

### 3.1 DSL Surface

The frontend mirrors Triton's `triton.language` API:

```python
import llk
import llk.language as tl

@llk.jit
def fused_swiglu(x_ptr, wg_ptr, wu_ptr, y_ptr,
                 M: int, N: int, K: int,
                 BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                 BLOCK_K: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    x_ptrs = tl.make_block_ptr(
        base=x_ptr, shape=(M, K), strides=(K, 1),
        offsets=(offs_m[:, None], offs_k[None, :]),
        block_shape=(BLOCK_M, BLOCK_K), order=(0, 1))

    gate_acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    up_acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_block in range(0, K, BLOCK_K):
        x_tile = tl.load(tl.advance(x_ptrs, (0, k_block)))
        wg_tile = tl.load(tl.advance(wg_ptrs, (k_block, 0)))
        wu_tile = tl.load(tl.advance(wu_ptrs, (k_block, 0)))

        gate_acc = tl.dot(x_tile, wg_tile, acc=gate_acc)
        up_acc = tl.dot(x_tile, wu_tile, acc=up_acc)

    gate = tl.silu(gate_acc)
    result = gate * up_acc

    y_ptrs = tl.make_block_ptr(
        base=y_ptr, shape=(M, N), strides=(N, 1),
        offsets=(offs_m[:, None], offs_n[None, :]),
        block_shape=(BLOCK_M, BLOCK_N), order=(0, 1))
    mask_m = offs_m[:, None] < M
    mask_n = offs_n[None, :] < N
    tl.store(y_ptrs, result, mask=mask_m & mask_n)
```

### 3.2 AST Capture → Triton IR

The `@llk.jit` decorator captures the Python AST via `inspect.getsource()` and lowers to Triton MLIR through an AST visitor:

```
Python function AST
    │  AST visitor walks function body
    │  Each tl.* call recognized as a Triton IR op
    ▼
Triton MLIR module (tt dialect)
    │  One tt.func per @llk.jit function
    │  tt.load, tt.store, tt.dot, tt.make_block_ptr, tt.advance, tt.reduce, etc.
    ▼
Into the 5-stage lowering pipeline
```

**AST → Triton IR mapping:**

| Python API | Triton MLIR Op |
|------------|---------------|
| `tl.program_id(axis)` | `tt.get_program_id {axis = 0\|1\|2}` |
| `tl.arange(start, end)` | `tt.make_range {start, end}` |
| `tl.load(ptr, mask, other)` | `tt.load %ptr {mask, other}` |
| `tl.store(ptr, value, mask)` | `tt.store %ptr, %value {mask}` |
| `tl.dot(a, b, acc)` | `tt.dot %a, %b, %acc` |
| `tl.make_block_ptr(...)` | `tt.make_block_ptr` |
| `tl.advance(ptr, offsets)` | `tt.advance %ptr, %offsets` |
| `tl.zeros(shape, dtype)` | `tt.splat %c0` |
| `tl.silu(x)` | `arith` math decomposition (recognized as SiLU in Stage 1) |
| `tl.exp(x)` | `math.exp` |
| `tl.where(cond, a, b)` | `arith.select` |
| `tl.sum(x, axis)` | `tt.reduce {kind = "add"}` |

### 3.3 JIT Invocation

```python
import llk

kernel = llk.jit(fused_swiglu)

# Compiles on first call; cached on subsequent calls with same shapes
y = kernel(x, wg, wu, M=128, N=4096, K=4096,
           BLOCK_M=32, BLOCK_N=64, BLOCK_K=64)

# Introspection
print(kernel.ttir)   # Triton IR
print(kernel.llkir)  # Post-lowering LLK IR
print(kernel.asm)    # Disassembly
```

### 3.4 Python Package Structure

```
python/llk/
├── __init__.py              # llk.jit decorator
├── language/
│   ├── __init__.py          # tl.load, tl.store, tl.dot, tl.arange, etc.
│   ├── _ast_compiler.py     # AST capture + visitor → Triton IR builder
│   ├── _triton_ir.py        # Triton MLIR dialect Python bindings
│   └── _block_pointer.py    # Block pointer shape/strides tracking
├── runtime/
│   ├── __init__.py          # JitFunction, Tensor wrapper
│   ├── _jit.py              # Compilation orchestration, cache keying
│   └── _ffi.py              # ctypes bridge to C++ runtime
└── torch/
    ├── __init__.py          # torch.compile(backend="llk")
    └── _dynamo_backend.py   # Dynamo FX graph → Triton pattern matching
```

---

## 4. The Five Lowering Stages

### 4.1 Stage 1: TritonToStructured

**Purpose:** Convert Triton compute ops to MLIR structured ops. Pattern-recognize known fusion patterns and emit `llk.*` ops directly.

**Pattern recognition (priority order):**

| Priority | Triton Pattern | LLK Op Emitted |
|----------|---------------|----------------|
| 1 | `tt.dot(x,wg)` + `tt.dot(x,wu)` + `silu(gate)*up` sharing same `x` | `llk.fused_swiglu` |
| 2 | `tt.dot(q,k^T)` + softmax + `tt.dot(p,v)` with causal mask | `llk.attention` |
| 3 | Even/odd pair rotation with cos/sin tables | `llk.rope` |
| 4 | Generic `tt.dot(a, b)` | `linalg.matmul` |
| 5 | Elementwise ops | `linalg.generic` |

**IR transform (SwiGLU fusion):**
```mlir
// Input: Triton IR
%gate = tt.dot %x_tile, %wg_tile, %gate_acc
%up   = tt.dot %x_tile, %wu_tile, %up_acc
// ... plus SiLU epilogue in arith

// Output: LLK IR (single op)
%y = llk.fused_swiglu ins(%x, %wg, %wu : tensor<BMxBKxbf16>, ...)
    outs(%init : tensor<BMxBNxbf16>)
    {accumulator_type = f32, activation = #llk.activation<silu>,
     math_mode = #llk.math_mode<bounded_fast>}
    → tensor<BMxBNxbf16>
```

**Precondition checks for SwiGLU fusion:**
1. Both `tt.dot` ops share the same `%x` operand (same SSA value)
2. Both `tt.dot` ops share the same iteration space
3. Consumer elementwise chain matches SiLU decomposition: `negf → exp → addf → divf → mulf` or `arith.select`-based
4. No other uses of the intermediate dot results
5. Accumulators are `f32` typed

### 4.2 Stage 2: BlockPointerToVector

**Purpose:** Lower Triton block pointers and load/store ops to explicit `vector.transfer_read`/`transfer_write` with boundary masking.

```mlir
// Input: Triton block pointer + load
%x_ptr = tt.make_block_ptr base(%x_base) shape(%M, %K)
    strides(%K, %c1) offsets(%off_m, %off_k)
    block_shape(%BM, %BK) order(0, 1)
%x_tile = tt.load %x_ptr {mask = %mask}

// Output: Explicit vector transfer
%x_view = memref.subview %x_base[%off_m, %off_k] [%BM, %BK] [1, 1]
%x_tile = vector.transfer_read %x_view[%c0, %c0], %c0_bf16
    {in_bounds = [false, false]}
    : memref<BMxBKxbf16> → vector<BMxBKxbf16>
```

**Boundary masking:** When block shape doesn't evenly divide the global shape:
```mlir
%mask = vector.create_mask %remaining_m, %remaining_n : vector<BMxBNxi1>
vector.transfer_write %result, %y_view[%c0, %c0], %mask
    : vector<BMxBNxbf16> → memref<BMxBNxbf16>
```

`tt.advance` folds to index arithmetic — no IR instruction remains after folding.

### 4.3 Stage 3: SharedMemToScratch

**Purpose:** Lower Triton shared memory allocations to per-worker CPU scratch buffers.

```mlir
// Input
%smem = tt.alloc : memref<BMxBKxbf16, 3>  // 3 = shared memory space
tt.async_copy %global_ptr, %smem
tt.await

// Output
%scratch = memref.alloc() : memref<BMxBKxbf16>
memref.copy %global_ptr, %scratch
// No await — copy is synchronous on CPU
```

Allocations hoisted outside the K-loop by `allocation-hoisting` (existing M3 pass). Static-sized scratch placed in `KernelContext.scratch` buffer; dynamic sizes fall back to individual `memref.alloc`.

### 4.4 Stage 4: AtomicToLLRT

**Purpose:** Lower Triton atomic ops to `llrt` runtime calls wrapping `std::atomic` or CPU CAS.

```mlir
// Input
tt.atomic_add %ptr, %value : f32

// Output
llrt.atomic_add_f32 %ptr, %value : !llvm.ptr, f32
```

Supported: `atomic_add`, `atomic_max`, `atomic_min`, `atomic_cas`. Unsupported atomics rejected by `TritonCPUVerifier`.

### 4.5 Stage 5: GridToForall

**Purpose:** Map Triton's grid of blocks to the existing `scf.forall` parallel decomposition.

```mlir
// Input: per-block function with get_program_id
%pid_m = tt.get_program_id 0 : index
%pid_n = tt.get_program_id 1 : index

// Output: scf.forall over linearized tile space
%num_m_tiles = arith.ceildivsi %M, %BM
%num_n_tiles = arith.ceildivsi %N, %BN
%total = arith.muli %num_m_tiles, %num_n_tiles
scf.forall (%tid) in (%total) {
    %m = arith.divsi %tid, %num_n_tiles
    %n = arith.remsi %tid, %num_n_tiles
    // body uses %m, %n instead of tt.get_program_id
}
```

1D grids map to a single `forall`; 2D grids linearize as above; 3D grids linearize all three dimensions. This feeds directly into the existing M4 `ParallelDecompose` → `ForallToLLRT` path.

---

## 5. LLK Dialect Additions

### 5.1 New Ops

```tablegen
// Generic matmul — fallback when no fusion pattern matches
def LLK_MatmulOp : LLK_Op<"matmul", [
    DeclareOpInterfaceMethods<DestinationStyleOpInterface>,
    DeclareOpInterfaceMethods<TilingInterface>
]> {
    let summary = "Generic batched matmul fallback";
    let arguments = (ins
        TensorOf<[BF16, F16, F32]>:$a,
        TensorOf<[BF16, F16, F32]>:$b,
        TensorOf<[F32]>:$init,
        AnyAttr:$accumulator_type,
        LLK_MathModeAttr:$math_mode
    );
    let results = (outs TensorOf<[BF16, F16, F32]>:$result);
}

// Raw pointer → tensor ABI bridge
def LLK_MakeTensorOp : LLK_Op<"make_tensor"> {
    let summary = "Wrap raw pointer + shape as tensor for pipeline entry";
    let arguments = (ins
        AnyType:$base_ptr,     // !llvm.ptr
        I64Attr:$dim0,
        I64Attr:$dim1,
        TypeAttr:$dtype
    );
    let results = (outs AnyTensor:$result);
}
```

`llk.make_tensor` bridges Triton's pointer-based world and MLIR's tensor-based pipeline. It has no lowering — eliminated during bufferization when tensors become memrefs backed by the original raw pointer.

### 5.2 Extended Math Mode

Add `#llk.math_mode<triton_fast>` — Triton's default numerical behavior: finite-range exp/silu/cos/sin approximations, allow reassociation, flush subnormals. Semantically equivalent to `bounded_fast` but documented as the Triton compatibility mode.

---

## 6. Schedule Database Integration

### 6.1 Triton Entry Schema

Triton kernels carry their own `BLOCK_M`/`BLOCK_N`/`BLOCK_K` from Python `tl.constexpr` annotations. The schedule DB accepts these via `"inherit_from_triton"`:

```json
{
  "version": 2,
  "entries": [
    {
      "source": "triton",
      "operation": "fused_swiglu",
      "target": "x86-avx2",
      "shape": { "M_bucket": 4, "N": 4096, "K": 4096 },
      "dtype": "bf16",
      "schedule": {
        "BM": "inherit_from_triton",
        "BN": "inherit_from_triton",
        "BK": "inherit_from_triton",
        "VM": 4, "VN": 8,
        "vector_width": 8,
        "num_threads": 8,
        "grain_size": 4
      }
    }
  ]
}
```

Triton provides the outer tile; the schedule DB provides the vector micro-tile (`VM`/`VN`) and parallel grain. This is the hybrid approach — Triton authors control the outer schedule, the compiler controls the inner micro-kernel.

### 6.2 Fallback Behavior

When no schedule DB entry exists for a (`M_bucket`, `N`, `K`) combination:
1. Inherit `BM`/`BN`/`BK` from Triton `tl.constexpr`
2. Default `VM=4, VN=8` for AVX2 BF16
3. Default `num_threads = min(hardware_concurrency, total_tiles)`
4. Default `grain_size = max(1, total_tiles / num_threads)`
5. Emit warning: "No schedule entry for (bucket, N, K). Using Triton tile sizes with conservative defaults."

---

## 7. CPU→GPU Semantic Gap Handling

| Gap | Triton Assumption | CPU Reality | Strategy |
|-----|-------------------|-------------|----------|
| **Shared memory sync** | `tt.async_copy` + barrier | CPU threads are independent; no intra-block sync | `tl.alloc` allowed; `async_copy` → synchronous `memref.copy`. No barrier needed. Programs relying on warp-level sync rejected |
| **Block pointer modulo** | `block_shape` must divide pointer strides for coalescing | Strided access cheap on CPU | Ignore modulo constraints; emit subview with computed strides. Warn if stride % vector_width ≠ 0 |
| **MMA/TC layout** | `tt.dot` forces TF32/TF16 tensor core layout | CPU uses FMA via `vector.contract` | `tt.dot` lowered to `linalg.matmul` in f32; input_precision ignored. Verify result within 1e-2 of FP64 reference |
| **Warp-level primitives** | `tl.dot` with mma layout, `tl.inline_asm` | No CPU equivalent | Rejected by TritonCPUVerifier with clear error message |
| **Grid dimension** | 1D/2D/3D CUDA grid | 1D linearized tile space | All grid dims linearized in Stage 5; 3D → `m*grid_n*grid_k + n*grid_k + k` |

---

## 8. ABI — Python ↔ C++ Boundary

### 8.1 Tensor2D (Unchanged from M1)

```cpp
struct Tensor2D {
    void* data;
    int64_t dim0, dim1;
    int64_t stride0, stride1;  // Triton kernels express non-contiguous strides naturally
};
```

Non-contiguous strides from `tl.make_block_ptr` propagate into `memref.subview` strides — no copy required.

### 8.2 KernelContext Extension

```cpp
struct KernelContext {
    void* scratch;             // Base scratch buffer
    size_t scratch_size;
    ThreadPool* thread_pool;
    int64_t M, N, K;          // Runtime dimensions
    int64_t total_tiles;
    int64_t grain_size;
    // New for Triton:
    int64_t grid_dim_x;       // Triton grid dimensions
    int64_t grid_dim_y;
    int64_t BLOCK_M;          // Triton tile sizes (from tl.constexpr)
    int64_t BLOCK_N;
    int64_t BLOCK_K;
};
```

---

## 9. Test Strategy

### 9.1 Per-Stage Tests (FileCheck)

| Stage | Test File | Key Assertions |
|-------|-----------|----------------|
| TritonToStructured | `triton_to_structured.mlir` | `tt.dot`→`linalg.matmul`; `tt.dot×2+silu`→`llk.fused_swiglu`; `dot+softmax+dot`→`llk.attention` |
| TritonToStructured | `triton_to_structured_reject.mlir` | Non-fusible patterns remain as individual `linalg.matmul` |
| BlockPointerToVector | `block_ptr_to_vector.mlir` | 2D block pointer→`transfer_read`; masked boundary→`create_mask`; stride-based subview |
| BlockPointerToVector | `block_ptr_1d.mlir` | 1D pointer→`transfer_read` |
| SharedMemToScratch | `shared_mem_to_scratch.mlir` | `tt.alloc`→`memref.alloc`; `async_copy`→`memref.copy` |
| AtomicToLLRT | `atomic_to_llrt.mlir` | `atomic_add`→`llrt.atomic_add_f32`; unsupported atomic→verifier error |
| GridToForall | `grid_to_forall.mlir` | 1D grid→`forall`; 2D grid→linearized `forall`; 3D grid→linearized |
| GridToForall | `grid_empty.mlir` | Zero-element grid→verifier error |
| TritonCPUVerifier | `cpu_verifier_pass.mlir` | Supported ops pass; warp-level ops rejected |
| TritonCPUVerifier | `cpu_verifier_reject.mlir` | MMA-layout dot, inline_asm rejected with descriptive messages |
| Full Pipeline | `triton_full_pipeline.mlir` | Triton IR→LLK IR through all 5 stages |

### 9.2 Python Tests (pytest)

| Test | What It Verifies |
|------|-----------------|
| `test/python/test_ast_compiler.py` | `@llk.jit` function produces expected Triton IR |
| `test/python/test_triton_ir_builder.py` | API surface completeness: all `tl.*` ops emit valid Triton MLIR |
| `test/python/test_jit_invocation.py` | Compile + invoke round-trip for simple elementwise kernel |
| `test/python/test_cache.py` | Repeated calls with same shapes return cached function pointer |

### 9.3 End-to-End Tests (pytest + GTest)

| Test | What It Verifies | Tolerance |
|------|-----------------|-----------|
| `test/Execution/triton_swiglu.py` | `@llk.jit` SwiGLU vs PyTorch reference | Max abs error ≤ 1e-2 (BF16) |
| `test/Execution/triton_rope.py` | `@llk.jit` RoPE vs PyTorch | Max abs error ≤ 1e-4 (F32) |
| `test/Execution/triton_attention.py` | `@llk.jit` attention with online softmax vs PyTorch | Max abs error ≤ 1e-3 (BF16) |
| `test/Execution/triton_elementwise.cpp` | Elementwise Triton kernels (SiLU, add, mul) across shapes | Bit-exact vs scalar reference |
| `test/Execution/triton_shared_mem.cpp` | Matmul with shared memory tiles | Max abs error ≤ 1e-2 |
| `test/Execution/triton_atomics.cpp` | Atomic add on histogram kernel | Correct final counts |
| `test/Execution/triton_compat.cpp` | 10+ real Triton kernels from open-source LLM repos | Pass rate ≥ 80% |

---

## 10. Build Order & Milestones

| Milestone | What | Components | Exit Criterion |
|-----------|------|-----------|----------------|
| **M8a** | Python DSL + Triton IR | `@llk.jit`, `tl.load/store/dot/arange`, AST→Triton IR, Python bindings for tt dialect | `@llk.jit` function parses and emits valid Triton MLIR; pytest suite passes |
| **M8b** | TritonToStructured + GridToForall | Stages 1 + 5, SwiGLU pattern, `llk.make_tensor` | SwiGLU kernel from `@llk.jit` compiles and JIT-executes correctly for all M1 shapes |
| **M8c** | BlockPointerToVector | Stage 2, `vector.transfer_read/write`, mask gen, stride handling | Elementwise Triton kernels execute correctly |
| **M8d** | SharedMemToScratch + AtomicToLLRT | Stages 3 + 4, scratch lowering, atomic runtime | Triton matmul with shared memory tiles; flash-attention pattern works |
| **M8e** | Full Triton Compatibility | All stages, TritonCPUVerifier, schedule DB entries, torch.compile backend | 10+ real Triton kernels compile and run; pass rate ≥ 80% |

M8b is the **key early milestone** — SwiGLU working from Python in ~2-3 weeks of effort, requiring only stages 1+5 (~1500 lines of C++ plus the Python AST compiler).

---

## 11. New Files

```
New C++ files:
  include/LLK/Conversion/TritonToLLK/
    TritonToStructured.h           # Stage 1 pass header
    BlockPointerToVector.h         # Stage 2 pass header
    SharedMemToScratch.h           # Stage 3 pass header
    AtomicToLLRT.h                 # Stage 4 pass header
    GridToForall.h                 # Stage 5 pass header
    TritonCPUVerifier.h            # Verification pass header
  lib/Conversion/TritonToLLK/
    TritonToStructured.cpp         # Stage 1: compute ops → structured
    BlockPointerToVector.cpp       # Stage 2: block pointers → vector transfers
    SharedMemToScratch.cpp         # Stage 3: shared memory → scratch
    AtomicToLLRT.cpp               # Stage 4: atomics → runtime calls
    GridToForall.cpp               # Stage 5: grid → scf.forall
    TritonCPUVerifier.cpp          # Verification pass
    TritonToLLKPasses.cpp          # Pass registration

New Python files:
  python/llk/__init__.py
  python/llk/language/__init__.py
  python/llk/language/_ast_compiler.py
  python/llk/language/_triton_ir.py
  python/llk/language/_block_pointer.py
  python/llk/runtime/__init__.py
  python/llk/runtime/_jit.py
  python/llk/runtime/_ffi.py
  python/llk/torch/__init__.py
  python/llk/torch/_dynamo_backend.py

New test files:
  test/Conversion/TritonToLLK/
    triton_to_structured.mlir
    triton_to_structured_reject.mlir
    block_ptr_to_vector.mlir
    block_ptr_1d.mlir
    shared_mem_to_scratch.mlir
    atomic_to_llrt.mlir
    grid_to_forall.mlir
    grid_empty.mlir
    cpu_verifier_pass.mlir
    cpu_verifier_reject.mlir
    triton_full_pipeline.mlir
  test/python/
    test_ast_compiler.py
    test_triton_ir_builder.py
    test_jit_invocation.py
    test_cache.py
  test/Execution/
    triton_swiglu.py
    triton_rope.py
    triton_attention.py
    triton_elementwise.cpp
    triton_shared_mem.cpp
    triton_atomics.cpp
    triton_compat.cpp

New runtime files:
  lib/Conversion/TritonToLLK/CMakeLists.txt
  include/LLK/Conversion/TritonToLLK/CMakeLists.txt

Modified files:
  include/LLK/Dialect/LLKOps.td        # Add llk.matmul, llk.make_tensor
  include/LLK/Dialect/LLKDialect.td    # Add triton_fast math mode
  lib/Dialect/LLK/LLKOps.cpp           # Verification for new ops
  schedules/schedule_db.json           # Triton entries
  CMakeLists.txt                       # TritonToLLK library target
  include/LLK/Runtime/KernelContext.h  # Triton grid + tile fields
```

---

## 12. Dependencies

- M1-M6 C++ infrastructure (stable)
- MLIR Python bindings (from LLVM/MLIR build)
- Triton MLIR dialect (from upstream LLVM/MLIR or vendored)
- PyTorch 2.0+ (for `torch.compile` integration, M8e)

---

## 13. Related

- [ARCHITECTURE.md](../../../ARCHITECTURE.md) — §2.2 Python Frontend, full pipeline
- [m7-python-frontend.md](../../design/m7-python-frontend.md) — superseded stub
- [m1-m6 design docs](../../design/) — existing pipeline this builds upon
- Triton DSL: https://triton-lang.org/
- Triton MLIR dialect: https://github.com/triton-lang/triton/tree/main/lib/Dialect/Triton/IR
