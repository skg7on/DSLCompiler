# Milestone 2: Explicit Vector Path

**Parent:** [ARCHITECTURE.md](../../ARCHITECTURE.md) — See §4 Implementation Sequence
**Previous:** [Milestone 1: Scalar End-to-End Pipeline](m1-scalar-pipeline.md)
**Next:** [Milestone 3: Fused Memory Lowering](m3-fused-memory.md)
**Status:** Design Approved | **Level:** Algorithm & Data Structure

---

## 1. Objectives & Exit Criterion

Introduce tiling and explicit SIMD via the MLIR Vector dialect. Replace the scalar inner-K loop with `vector.contract`, `vector.transfer_read`/`transfer_write`, and masked N-tail handling. Target: AVX2 with BF16→FP32 accumulation.

**Exit criterion:** Disassembly of the compiled function shows **no scalar inner-K loop**. Vectorized contraction uses FMA instructions (`vfmadd231ps`). Masked tail handling covers non-multiple-of-8 N. All M1 correctness tests pass with the vector pipeline.

---

## 2. Components to Build

```
New files:
  include/LLK/Transforms/Passes.td                 # Transform pass registry (TableGen)
  include/LLK/Transforms/TileAndVectorize.h        # Tiling + vectorization pass header
  include/LLK/Target/X86/TargetAVX2.h              # AVX2 target feature detection
  lib/Transforms/TileAndVectorize.cpp              # Apply Transform dialect schedule
  lib/Target/X86/TargetAVX2.cpp                    # AVX2 ISA capability query
  schedules/x86_avx2/schedule_bf16.mlir            # Transform dialect schedule file
  test/Transforms/tile_and_vectorize.mlir           # FileCheck: tiled + vectorized IR
  test/Transforms/mask_gen.mlir                     # FileCheck: non-multiple-of-8 N tails
  test/Transforms/schedule_bf16_check.mlir         # FileCheck: schedule parses correctly
  test/Execution/swiglu_vector_avx2.cpp             # Assembly inspection + numerical
  test/Numerical/vector_vs_scalar.cpp               # Cross-validation: vector == scalar
```

---

## 3. Tiling + Vectorization Schedule

Stored as a Transform dialect IR file — not hardcoded in passes. Parameterized by target.

### 3.1 AVX2 BF16 Schedule Parameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| BM (outer M tile) | 32 | Fits L2 cache working set |
| BN (outer N tile) | 64 | Vector-friendly N |
| BK (K reduction tile) | 64 | Fits L1 cache: X tile + 2× weight tiles |
| VM (register M micro-tile) | 4 | 4 rows in YMM registers |
| VN (register N micro-tile) | 8 | 8 × f32 = 256-bit YMM |
| vector_width | 8 | AVX2: 256-bit / 32-bit float |

### 3.2 Transform Sequence (schedule_bf16.mlir)

```mlir
transform.named_sequence @schedule_avx2_bf16(
    %root: !transform.any_op {transform.readonly}) {

  %swiglu = transform.structured.match ops{["llk.fused_swiglu"]} in %root

  // 1. Tile outer M×N parallel loops
  %loops, %tiled = transform.structured.tile_using_forall %swiglu
      tile_sizes [32, 64]

  // 2. Fuse both matmul producers into consumer tile
  %fused = transform.structured.fuse_into_containing_op
      %tiled into %loops

  // 3. Tile K reduction dimension
  %k_loops, %k_tiled = transform.structured.tile_reduction_using_for
      %fused tile_sizes [64]

  // 4. Vectorize inner tiles
  transform.structured.vectorize %k_tiled
      vector_sizes [4, 8]

  transform.yield
}
```

---

## 4. Desired Post-Vectorization IR

After the schedule executes, the inner K-loop body at the Vector dialect level:

```mlir
// Outer: scf.forall over M tiles × N tiles
// Inner: scf.for over K tiles

%x_tile   = vector.transfer_read %x[%m, %k0]
    {in_bounds = [true, true]} : tensor<32x64xbf16> → vector<4x64xbf16>

%wg_tile  = vector.transfer_read %wg_packed[%k0, %n]
    {in_bounds = [true, true]} : tensor<64x64xbf16> → vector<64x8xbf16>

%wu_tile  = vector.transfer_read %wu_packed[%k0, %n]
    {in_bounds = [true, true]} : tensor<64x64xbf16> → vector<64x8xbf16>

// Double-accumulator contraction
%gate_acc = vector.contract {
    indexing_maps = [
        affine_map<(m,k) -> (m,k)>,    // X tile: VM × BK
        affine_map<(n,k) -> (k,n)>,    // Wg tile: BK × VN
        affine_map<(m,n) -> (m,n)>     // Accumulator: VM × VN
    ],
    iterator_types = ["parallel", "parallel", "reduction"],
    kind = #vector.kind<add>
} %x_tile, %wg_tile, %gate_acc
    : vector<4x64xbf16>, vector<64x8xbf16> → vector<4x8xf32>

%up_acc = vector.contract {
    // Same indexing maps, different weight operand
} %x_tile, %wu_tile, %up_acc
    : vector<4x64xbf16>, vector<64x8xbf16> → vector<4x8xf32>
```

**After K-loop exit — epilogue on register accumulators:**

```mlir
// gate_acc, up_acc: vector<4x8xf32> in registers
%silu_vec = <SiLU polynomial on vector<4x8xf32>>   // detailed in M3
%result   = vector.mulf %silu_vec, %up_acc : vector<4x8xf32>
%trunc    = arith.truncf %result : vector<4x8xf32> → vector<4x8xbf16>

// Masked store for partial N
%mask = vector.create_mask %remaining_n : vector<8xi1>
vector.transfer_write %trunc, %y[%m, %n] {mask = %mask}
    : vector<4x8xbf16> → tensor<?x?xbf16>
```

---

## 5. Mask Generation for N-Tail Handling

When N is not a multiple of VN (8):

```mlir
%n_dim     = memref.dim %y, %c1 : index
%n_offset  = arith.addi %n, %c64 : index           // End of full tile
%remaining = arith.subi %n_dim, %n_offset : index   // Elements past tile end, or 0

%mask = vector.create_mask %remaining : vector<8xi1>
```

---

## 6. AVX2 Target Profile

| Property | Value |
|----------|-------|
| ISA features | AVX2 + FMA |
| Vector width | 256-bit = 8 × f32 = 16 × bf16 |
| BF16 handling | Upcast to f32 on load (no native AVX2 BF16); downcast via trunc on store |
| FMA pattern | `vector.contract` → `vector.fma` → `@llvm.fmuladd.*` → `vfmadd231ps` |
| Mask lowering | `vector.create_mask` → compare + blend sequences |

```cpp
struct CpuFeatures {
    bool avx2;
    bool fma;
    bool avx512f;
    bool avx512bf16;
    bool amx_bf16;

    static CpuFeatures detect();  // CPUID-based
    CpuIsa bestIsa() const;
};
```

---

## 7. Updated Pass Pipeline

Pipeline additions from M1 shown in **bold**:

```
llk.fused_swiglu
  → LLKToLinalg
  → canonicalize
  → shape-specialize        (NEW: propagate static shapes for bucketing)
  → tile-and-vectorize      (NEW: apply Transform dialect schedule)
  → canonicalize             (clean up after tiling + vectorization)
  → One-Shot Bufferize
  → convert-vector-to-llvm   (NEW: explicit vector → LLVM dialect)
  → convert-scf-to-cf
  → convert-arith-to-llvm
  → convert-math-to-llvm
  → convert-func-to-llvm
  → convert-memref-to-llvm
  → reconcile-unrealized-casts
  → LLVM IR
```

---

## 8. Test Specifications

| Test | What it verifies | Tool |
|------|-----------------|------|
| `test/Transforms/tile_and_vectorize.mlir` | linalg.matmul×2 + generic → tiled + fused + vectorized IR; `vector.contract` present; NO scalar inner-K loop | `llk-opt` + FileCheck |
| `test/Transforms/mask_gen.mlir` | N=127 → `vector.create_mask` and masked `transfer_write` | `llk-opt` + FileCheck |
| `test/Transforms/schedule_bf16_check.mlir` | Schedule file parses without error | `llk-opt --transform-schedule=...` + FileCheck |
| `test/Execution/swiglu_vector_avx2.cpp` | JIT + disassemble: `vfmadd231ps` present. Shapes: {(32,128,256), (16,512,512), (32,256,127) ← tail} | GTest + disasm |
| `test/Numerical/vector_vs_scalar.cpp` | All M1 shapes: vector results == scalar results bit-exact | GTest |

---

## 9. Dependencies

- [Milestone 1](m1-scalar-pipeline.md) — scalar pipeline, JIT cache, basic tooling
- AVX2-capable x86-64 build host for execution tests

---

## 10. Related

- [ARCHITECTURE.md](../../ARCHITECTURE.md) — §2.3 Scheduling System, §2.4 Explicit SIMD
- [m1-scalar-pipeline.md](m1-scalar-pipeline.md) — pipeline this builds upon
- [m3-fused-memory.md](m3-fused-memory.md) — fuses the double-contraction after vectorization works
