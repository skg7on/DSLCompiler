# Milestone 3: Fused Memory Lowering

**Parent:** [ARCHITECTURE.md](../../ARCHITECTURE.md) — See §4 Implementation Sequence
**Previous:** [Milestone 2: Explicit Vector Path](m2-explicit-vector.md)
**Next:** [Milestone 4: Parallel Execution](m4-parallel-execution.md)
**Status:** Design Approved | **Level:** Algorithm & Data Structure

---

## 1. Objectives & Exit Criterion

Eliminate the full-size `gate` and `up` intermediate tensors. Fuse the double-contraction so each loaded X tile feeds both weight contractions before being discarded. The SiLU epilogue executes on register-resident accumulators only after the full K reduction.

**Exit criterion:** Memory analysis confirms **zero full-size intermediate buffers** for gate and up. For (M=128, N=4096, K=4096) BF16, full intermediates would be ~2GB — peak RSS must be << 500MB. Only scratch: weight packing buffers, per-thread tile scratch, alignment padding.

---

## 2. Key Optimization: Double-Contraction Fusion

### 2.1 Desired Tile-Level Structure

```
parallel_for (mTile in 0..M step BM, nTile in 0..N step BN):
    gate_acc[VM][VN] = 0.0   // register tile, never materialized to memory
    up_acc[VM][VN]   = 0.0
    for (kTile in 0..K step BK):
        x_tile  = load X[mTile:mTile+BM, kTile:kTile+BK]       // loaded ONCE
        wg_tile = load Wg_packed[kTile:kTile+BK, nTile:nTile+BN]
        wu_tile = load Wu_packed[kTile:kTile+BK, nTile:nTile+BN]
        gate_acc += contract(x_tile, wg_tile)  // same x_tile for both
        up_acc   += contract(x_tile, wu_tile)
    result = silu(gate_acc) * up_acc           // epilogue on registers
    store result to Y[mTile:mTile+BM, nTile:nTile+BN]
```

---

## 3. Components to Build

```
New/modified files:
  include/LLK/Transforms/FuseDoubleContraction.h  # Fusion pass header
  lib/Transforms/FuseDoubleContraction.cpp         # Rewrite: matmul×2 + generic → fused
  lib/Transforms/TileAndVectorize.cpp              # UPDATE: handle double-accumulator pattern
  include/LLK/Transforms/ScratchAnalysis.h         # Allocation audit pass
  lib/Transforms/ScratchAnalysis.cpp               # Verify no full-sized allocs post-bufferization
  include/LLK/Transforms/PackWeights.h             # Weight packing transform
  lib/Transforms/PackWeights.cpp                   # Repack K×N weights for cache-friendly access
  runtime/PackedWeights.cpp                        # Runtime packing utilities
  include/LLK/Runtime/PackedWeights.h
  test/Transforms/fuse_double_contraction.mlir     # FileCheck: fusion pattern
  test/Transforms/fuse_reject.mlir                  # FileCheck: non-fusible patterns rejected
  test/Transforms/pack_weights.mlir                 # FileCheck: weight packing layout
  test/Transforms/scratch_analysis.mlir             # FileCheck: no full allocs
  test/Execution/swiglu_fused_memory.cpp            # Profiler-based memory verification
  test/Execution/packed_vs_unpacked.cpp             # Numerical: packed == unpacked
```

---

## 4. FuseDoubleContraction Pass

### 4.1 Pattern to Match

```
  %gate = linalg.matmul ins(%x, %wg) outs(%gate_init)
  %up   = linalg.matmul ins(%x, %wu) outs(%up_init)
  %y    = linalg.generic {silu(g) * u} ins(%gate, %up) outs(%out)
```

### 4.2 Precondition Checks

1. Both matmuls share the same `%x` operand (same SSA value)
2. Both matmuls share the same iteration space: `[M, K] × [K, N]`
3. Consumer generic has identity indexing maps (elementwise)
4. Consumer body matches SiLU activation: `silu(gate) * up → trunc → yield`
5. No other uses of `%gate` or `%up` exist (single consumer)

### 4.3 Fusion Algorithm

```
fuse(op):
    M, N, K = dims from matmul contracts

    // Create fused linalg.generic: 3 inputs, 2 accumulators, 2 reduction outputs
    fused = linalg.generic {
        indexing_maps = [
            affine_map<(m,n,k) -> (m,k)>,     // X
            affine_map<(m,n,k) -> (k,n)>,     // Wg
            affine_map<(m,n,k) -> (k,n)>,     // Wu
            affine_map<(m,n,k) -> (m,n)>,     // gate_acc
            affine_map<(m,n,k) -> (m,n)>      // up_acc
        ],
        iterator_types = ["parallel", "parallel", "reduction"]
    } ins(x, wg, wu, gate_init, up_init)
      outs(gate_acc, up_acc)
    {
    ^bb0(%x_el: bf16, %wg_el: bf16, %wu_el: bf16,
         %g_acc: f32, %u_acc: f32):
        %x_f32 = arith.extf %x_el : bf16 → f32
        %wg_f32 = arith.extf %wg_el : bf16 → f32
        %wu_f32 = arith.extf %wu_el : bf16 → f32
        %g_new = arith.addf %g_acc, (arith.mulf %x_f32, %wg_f32)
        %u_new = arith.addf %u_acc, (arith.mulf %x_f32, %wu_f32)
        linalg.yield %g_new, %u_new : f32, f32
    }

    // Epilogue: silu(gate_acc) * up_acc → bf16 output
    epilogue = linalg.generic {
        indexing_maps = [id, id, id],
        iterator_types = ["parallel", "parallel"]
    } ins(fused#0, fused#1) outs(%out) {
    ^bb0(%g: f32, %u: f32, %old: bf16):
        %silu_val = compute_silu(%g)
        %result = arith.mulf %silu_val, %u
        %cast = arith.truncf %result : f32 → bf16
        linalg.yield %cast : bf16
    }

    rewriter.replaceOp(original_consumer, epilogue)
```

---

## 5. Weight Packing

### 5.1 Packed Layout

Original row-major `W[K][N]` repacked as `W_packed[BK][K/BK][N]` where each BK×N block is stored contiguously:

```
Original:  W[k][n] at offset k * N + n
Packed:    W_packed[kb * BK * N + k_inner * N + n]
           where kb = k / BK, k_inner = k % BK
```

Block-major order enables contiguous vector loads for each K-block.

### 5.2 Runtime Packing API

```cpp
struct PackedWeights {
    void* data;               // Packed buffer (owned, allocated once)
    int64_t K_blocks;         // K / BK
    int64_t N;                // Original N
    int64_t block_size;       // BK * N * element_size
    DType dtype;
};

// Pack a single weight tensor
PackedWeights packWeightMatrix(const Tensor2D& W, int64_t BK);

// Repack when weights change
void repack(PackedWeights& pw, const Tensor2D& W);
```

---

## 6. Scratch Analysis Pass

Runs post-bufferization to audit all heap allocations:

```
analyzeAllocations(func):
    for each memref.alloc in func:
        size = product(static_dims) × element_size
        dims = extract_static_dims(alloc)

        if dims ≈ [M, N] and M > BM and N > BN:
            report(ERROR, "Full-size gate/up materialized at ", loc)

        if size > BM * BN * sizeof(f32) * 2:
            report(WARN, "Large scratch alloc: ", size, " bytes at ", loc)

        classify as: TILE_SCRATCH | WEIGHT_PACKED | PADDING | UNKNOWN

    emit summary:
        total_scratch_bytes, num_allocations, max_single_allocation
        classification_counts
```

---

## 7. Updated Pass Pipeline

Pipeline additions from M2 shown in **bold**:

```
llk.fused_swiglu
  → LLKToLinalg
  → canonicalize
  → shape-specialize
  → fuse-double-contraction        (NEW)
  → pack-weights                   (NEW: annotate weights with packed_layout)
  → tile-and-vectorize             (UPDATED: handle fused 5-input generic)
  → canonicalize
  → One-Shot Bufferize
  → scratch-analysis               (NEW: verify no full-size gate/up allocs)
  → allocation-hoisting            (NEW: hoist tile-scratch outside parallel loops)
  → buffer-deallocation
  → convert-vector-to-llvm
  → scf/cf/arith/math/func/memref-to-llvm
  → LLVM IR
```

---

## 8. Test Specifications

| Test | What it verifies | Tool |
|------|-----------------|------|
| `test/Transforms/fuse_double_contraction.mlir` | 2× matmul + SiLU → single fused generic with 2 reduction outputs, shared X | `llk-opt` + FileCheck |
| `test/Transforms/fuse_reject.mlir` | Non-fusible patterns rejected: different X, mismatched shapes, non-SiLU consumer, multiple consumers | `llk-opt` + FileCheck |
| `test/Transforms/pack_weights.mlir` | Weights receive `llk.packed_layout` attributes; block-major access generated | `llk-opt` + FileCheck |
| `test/Transforms/scratch_analysis.mlir` | Post-bufferization: zero full-size [M,N] allocs. Only tile-scratch + packed-weight allocs | `llk-opt` + FileCheck |
| `test/Execution/swiglu_fused_memory.cpp` | (M=128,N=4096,K=4096): peak RSS << 500MB. No malloc for gate/up-sized buffers | GTest + `malloc_stats` |
| `test/Execution/packed_vs_unpacked.cpp` | Packed-weight kernel == unpacked scalar reference across all M1 shapes | GTest |

---

## 9. Dependencies

- [Milestone 2](m2-explicit-vector.md) — tiling + vectorization infrastructure
- [Milestone 1](m1-scalar-pipeline.md) — end-to-end test harness

---

## 10. Related

- [ARCHITECTURE.md](../../ARCHITECTURE.md) — §2.5 Math Approximation, §2.6 Memory & Bufferization
- [m2-explicit-vector.md](m2-explicit-vector.md) — vectorization this fuses
- [m4-parallel-execution.md](m4-parallel-execution.md) — adds parallelism on top of fused tiles
- [m5-specialization-tuning.md](m5-specialization-tuning.md) — L4 packed-weight cache
