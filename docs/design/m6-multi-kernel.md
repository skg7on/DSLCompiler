# Milestone 6: Second and Third Kernels (RoPE + Attention)

**Parent:** [ARCHITECTURE.md](../../ARCHITECTURE.md) — See §4 Implementation Sequence
**Previous:** [Milestone 5: Specialization & Tuning](m5-specialization-tuning.md)
**Status:** Design Approved | **Level:** Algorithm & Data Structure

---

## 1. Objectives & Exit Criterion

Add `llk.rope` and `llk.attention` operations to prove the compiler infrastructure generalizes beyond SwiGLU. Each kernel exercises distinct compiler capabilities not covered by the fused matmul path. Refactor shared infrastructure to remove SwiGLU-specific assumptions.

**Exit criteria:**
- Both kernels execute correctly through JIT for all supported shapes; verified against PyTorch reference
- Both compile via the same pipeline infrastructure (LLKToLinalg → schedule → vectorize → bufferize → JIT)
- No SwiGLU-specific assumptions remain in shared passes (verified by `grep` test)
- Online softmax: numerically stable for L ≤ 2048 (max error < 1e-3 vs FP64, no NaN)

---

## 2. Kernel A: RoPE (Rotary Position Embedding)

### 2.1 Computation

```
For input X [B, H, L, D]:
    For each position p in L:
        For each pair (2i, 2i+1) in D:
            theta_i = theta_base^(-2i/D)
            cos_val = cos(p * theta_i)
            sin_val = sin(p * theta_i)
            X[p, 2i]   = X[p, 2i] * cos_val - X[p, 2i+1] * sin_val
            X[p, 2i+1] = X[p, 2i+1] * cos_val + X[p, 2i] * sin_val
```

### 2.2 What RoPE Exercises

- Non-contiguous indexing (even/odd pair extraction)
- `vector.shuffle` for pairwise data rearrangement
- Elementwise math approximation (cos, sin — first use beyond exp)
- Broadcast of cos/sin tables across B×H
- No reductions: purely parallel elementwise + permutation

### 2.3 MLIR Definition

```mlir
%y = llk.rope ins(%x : tensor<BxHxLxDxf32>)
               outs(%init : tensor<BxHxLxDxf32>)
    {
      position_ids = %pos : tensor<Lxi64>,
      theta = 10000.0 : f64,
      math_mode = #llk.math_mode<bounded_fast>
    } → tensor<BxHxLxDxf32>
```

### 2.4 Lowering Algorithm

```
LLKRoPEToLinalg:
    1. Precompute freq table: for i in 0..D/2:
         freqs[i] = theta ^ (-2i/D)    → tensor<D/2 x f64>

    2. Compute angles: angles[p, i] = position_ids[p] * freqs[i]
       → tensor<L x D/2 x f64>

    3. Trig tables:
         cos_table = cos(angles)       → tensor<L x D/2 x f32>
         sin_table = sin(angles)

    4. Broadcast to [B, H, L, D/2]:
         cos_bcast = broadcast(cos_table, [B, H, 1, 1])
         sin_bcast = broadcast(sin_table, [B, H, 1, 1])

    5. Split X into even/odd:
         x_even = X[..., 0::2]         → [B, H, L, D/2]
         x_odd  = X[..., 1::2]

    6. Rotate:
         x_rot = x_even * cos_bcast - x_odd * sin_bcast
         y_rot = x_odd * cos_bcast + x_even * sin_bcast

    7. Interleave back: Y[..., 0::2] = x_rot, Y[..., 1::2] = y_rot
```

### 2.5 Vector Shuffle for Even/Odd Interleave

For AVX2 with 8×f32 vectors:

```mlir
// Interleave even from v_even with even from v_odd
%interleaved_lo = vector.shuffle %even, %odd
    [0, 8, 1, 9, 2, 10, 3, 11] : vector<8xf32>, vector<8xf32> → vector<8xf32>

// Interleave odd from v_even with odd from v_odd
%interleaved_hi = vector.shuffle %even, %odd
    [4, 12, 5, 13, 6, 14, 7, 15] : vector<8xf32>, vector<8xf32> → vector<8xf32>
```

### 2.6 RoPE Schedule Parameters

| Parameter | Typical Values | Note |
|-----------|---------------|------|
| B×H parallel | Full parallel at batch×head level | Independent across B and H |
| L_BLOCK | 32–128 | Sequence tile for cos/sin cache reuse |
| D vectorized | 4 or 8 | D typically 64/96/128 — 1-2 vector registers per pair |

---

## 3. Kernel B: Simplified Attention

### 3.1 Computation (per head)

```
S = Q × K^T                    // [Lq, D] × [D, Lk] → [Lq, Lk]
S = S / sqrt(D)                 // scale
S = S + causal_mask             // lower-triangular 0, upper-triangular -inf
P = softmax(S, dim=-1)          // online algorithm
O = P × V                       // [Lq, Lk] × [Lk, D] → [Lq, D]
```

### 3.2 Scope Limitation

This is *simplified* attention — no flash-attention tiling over Lq/Lk blocks. For long sequences, the Lq×Lk attention matrix may be materialized. The goal is: correct online softmax, multi-head parallelism, and the 3-operand matmul fusion pattern. IO-optimal attention is future work.

### 3.3 MLIR Definition

```mlir
%o = llk.attention ins(%q, %k, %v : tensor<BxHxLqxDxf32>,
                                  tensor<BxHxLkxDxf32>,
                                  tensor<BxHxLkxDxf32>)
                    outs(%init : tensor<BxHxLqxDxf32>)
    {
      scale = 0.08838834764,              // 1/sqrt(128) for D=128
      causal_mask = true,
      softmax_mode = #llk.softmax_mode<online>,
      math_mode = #llk.math_mode<bounded_fast>
    } → tensor<BxHxLqxDxf32>
```

### 3.4 Online Softmax Algorithm

```
For each query position q in 0..Lq:
    m = -infinity       // running max (scalar)
    d = 0.0             // running denominator (scalar)
    o_acc = zeros(D)    // running output accumulator

    for k_block in 0..Lk step BK:
        S_block = Q[q, :] × K[k_block:k_block+BK, :]^T   // [1, BK]
        S_block = S_block / sqrt(D)
        S_block = causal_mask(S_block, q, k_block, BK)

        m_new = max(m, max(S_block))
        scale_factor = exp(m - m_new)

        // Rescale old state
        d = d * scale_factor
        o_acc = o_acc * scale_factor

        // Accumulate new block
        exp_block = exp(S_block - m_new)
        d = d + sum(exp_block)
        o_acc = o_acc + exp_block × V[k_block:k_block+BK, :]

        m = m_new

    O[q, :] = o_acc / d
```

### 3.5 IR Structure at Linalg Level

```mlir
%q_loop = scf.for %q = %c0 to %Lq step %c1 {
    %final_o, %final_m, %final_d = scf.for %kb = %c0 to %Lk step %BK
        iter_args(%o_acc, %m = %neg_inf, %d = %c0_f32) {

        // S_block = Q[q] × K[kb:kb+BK]^T
        %s = linalg.matmul ins(%q_slice, %k_slice) outs(%s_init)
            : tensor<1xDxf32>, tensor<DxBKxf32> → tensor<1xBKxf32>

        // Scale + causal mask
        %s_scaled = linalg.generic {scale + causal_mask}
            ins(%s) outs(%s_masked_init) { ... }

        // Online softmax update
        %m_new = arith.maxf %m, (vector.reduction <max>, %s_scaled)
        %scale = math.exp (arith.subf %m, %m_new)
        %d_new = arith.addf (arith.mulf %d, %scale),
                             (vector.reduction <add>, math.exp(s_scaled - m_new))
        %o_new = vector.fma %o_acc, %scale_bcast,
                            (vector.contract %exp_block, %v_slice, %o_acc)

        scf.yield %o_new, %m_new, %d_new
    }

    // Normalize
    %o_normalized = arith.divf %final_o, %final_d
    linalg.yield %o_normalized
}
```

---

## 4. Shared Infrastructure Refactoring

Extract generic utilities that all three kernels use. Verify nothing remains SwiGLU-specific:

```
New shared modules:
  lib/Transforms/Common/
    TilingUtils.cpp              # Generic tile-size computation, loop nest construction
    VectorizationUtils.cpp       # Generic vectorization pattern: transfer_read → contract → transfer_write
    ScheduleLoader.cpp           # Common schedule_db.json reader (all kernel types)
    MaskGeneration.cpp           # Masked load/store for vector tail handling
    MathApproximation.cpp        # Polynomial approximations: exp, sigmoid, cos, sin

Key refactoring rules:
  - TileAndVectorize: reads op type → dispatches to generic tiling + op-specific vectorization
  - ScheduleSelection: accepts OperationKind enum
  - Bufferization pipeline: identical for all op types
  - JIT cache: KernelKey.operation distinguishes cache entries
```

### Generalized Pass Pipeline (Post-Refactor)

```
IR with any llk.* op
  → shape-specialize                    (generic)
  → LLKToLinalg                         (dispatches per op type)
  → canonicalize
  → select-schedule                     (generic: uses OperationKind)
  → op-specific fusion:
      SwiGLU:    fuse-double-contraction → pack-weights
      RoPE:      fuse-cos-sin-tables     → vectorize-shuffles
      Attention: online-softmax          → causal-mask-generate
  → tile-and-vectorize                  (generic)
  → parallel-decompose                  (generic)
  → serial-parallel-dispatch            (generic)
  → canonicalize
  → One-Shot Bufferize                  (generic)
  → scratch-analysis                    (generic)
  → allocation-hoisting                 (generic)
  → buffer-deallocation                 (generic)
  → forall-to-openmp / forall-to-llrt-runtime  (generic)
  → convert-vector-to-llvm              (generic)
  → scf/cf/arith/math/func/memref-to-llvm  (generic)
  → LLVM IR → JIT cache → execute      (generic)
```

---

## 5. Test Specifications

| Test | What it verifies | Tool |
|------|-----------------|------|
| **RoPE** | | |
| `test/Conversion/rope_lowering.mlir` | llk.rope → even/odd split, cos/sin table, rotation, interleave | `llk-opt` + FileCheck |
| `test/Transforms/rope_vector_shuffle.mlir` | `vector.shuffle` with correct masks for AVX2 pairwise interleave | `llk-opt` + FileCheck |
| `test/Execution/rope_correctness.cpp` | vs PyTorch: {(1,1,128,64), (1,32,2048,128), (4,16,512,96)}. Max abs error < 1e-4 | GTest |
| `test/Numerical/rope_precision.cpp` | cos/sin approximation at theta=10000, L=[1,2048]: max error < 1e-4 vs FP64 | GTest |
| `test/Transforms/rope_broadcast.mlir` | B×H broadcast of cos/sin tables across batch dims | `llk-opt` + FileCheck |
| **Attention** | | |
| `test/Conversion/attention_lowering.mlir` | llk.attention → matmul×2 + online softmax loop + causal mask | `llk-opt` + FileCheck |
| `test/Transforms/online_softmax.mlir` | Max tracking, rescaling, running denominator, block-wise exp | `llk-opt` + FileCheck |
| `test/Transforms/causal_mask.mlir` | Upper-triangular = -inf, lower-triangular = 0 | `llk-opt` + FileCheck |
| `test/Execution/attention_correctness.cpp` | vs `torch.nn.functional.scaled_dot_product_attention`: {(1,1,128,128,128), (1,4,512,512,128), (4,8,256,256,64)}. Max abs error < 1e-3 (BF16) | GTest |
| `test/Numerical/attention_stability.cpp` | L=2048: online softmax max-error < 1e-3 vs FP64; no NaN/Inf for extremes (±100) | GTest |
| **Shared infra** | | |
| `test/Execution/shared_pipeline.cpp` | All 3 ops compile + execute via same `llk-compile` binary; no op-specific paths in JIT/cache/bufferization | GTest |
| `test/Execution/no_swiglu_leak.cpp` | `grep -r "swiglu" lib/Transforms/Common/ lib/Conversion/LinalgToCPU/ runtime/` returns zero matches | Shell test |

---

## 6. Dependencies

- [Milestone 5](m5-specialization-tuning.md) — specialization, schedule DB, JIT cache, autotuning
- [Milestone 4](m4-parallel-execution.md) — parallel execution
- [Milestone 3](m3-fused-memory.md) — fused contraction pattern informs online softmax accumulator design
- [Milestone 2](m2-explicit-vector.md) — vector shuffles for RoPE
- [Milestone 1](m1-scalar-pipeline.md) — end-to-end pipeline
- PyTorch (reference implementations for tests only)

---

## 7. Related

- [ARCHITECTURE.md](../../ARCHITECTURE.md) — §2.1 Custom Dialect, §2.5 Math Approximation
- [m3-fused-memory.md](m3-fused-memory.md) — double-contraction pattern parallels online softmax accumulator
- [m5-specialization-tuning.md](m5-specialization-tuning.md) — schedule DB extended with RoPE + Attention entries
