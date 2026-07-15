# Milestone 1: Scalar End-to-End Pipeline

**Parent:** [ARCHITECTURE.md](../../ARCHITECTURE.md) — See §4 Implementation Sequence
**Next:** [Milestone 2: Explicit Vector Path](m2-explicit-vector.md)
**Status:** Design Approved | **Level:** Algorithm & Data Structure

---

## 1. Objectives & Exit Criterion

Build the minimal end-to-end compiler path: take a single `llk.fused_swiglu` op through every MLIR lowering stage and execute the result via ORC JIT. Everything is scalar — no tiling, no vectorization, no parallelism.

**Exit criterion:** A GTest program constructs `llk.fused_swiglu` IR for arbitrary supported shapes (M×K, K×N), runs the full pipeline, JIT-executes, and produces results within max absolute error ≤ 1e-2 (BF16 vs FP64 reference).

---

## 2. Components to Build

```
New files:
  include/LLK/Dialect/LLKDialect.td               # Dialect declaration (TableGen)
  include/LLK/Dialect/LLKOps.td                   # llk.fused_swiglu op definition
  include/LLK/Dialect/LLKAttributes.td            # math_mode, activation enum attributes
  include/LLK/Conversion/LLKToLinalg.h            # Lowering pass header
  lib/Dialect/LLK/LLKDialect.cpp                  # Dialect registration + print/parse hooks
  lib/Dialect/LLK/LLKOps.cpp                      # Op verification, shape inference, interfaces
  lib/Conversion/LLKToLinalg/LLKToLinalg.cpp      # Semantic lowering: llk → linalg
  tools/llk-opt/llk-opt.cpp                       # MLIR opt tool (register passes, parse/dump IR)
  tools/llk-compile/llk-compile.cpp                # End-to-end compilation driver
  runtime/JitCache.cpp                             # Minimal single-level JIT cache
  include/LLK/Runtime/JitCache.h
  test/Dialect/ops.mlir                            # FileCheck: op round-trip + verifier
  test/Dialect/ops_invalid.mlir                    # FileCheck: verifier rejects bad shapes
  test/Conversion/llk_to_linalg.mlir               # FileCheck: LLK→Linalg lowering
  test/Execution/swiglu_scalar.cpp                 # JIT execution + numerical correctness
  test/Execution/swiglu_reference.cpp              # C++ reference validated against PyTorch
  test/Execution/edge_cases.cpp                    # Zero-dim, unit-dim, K=1
  CMakeLists.txt                                   # Top-level build (out-of-tree MLIR project)
```

---

## 3. MLIR Definitions

### 3.1 Dialect

`LLK` dialect — single op initially: `llk.fused_swiglu`. Implements `DestinationStyleOpInterface` and `TilingInterface`.

### 3.2 llk.fused_swiglu

| Field | Type | Description |
|-------|------|-------------|
| Operand `x` | `tensor<?x?xbf16>` | Input activation X [M, K] |
| Operand `wg` | `tensor<?x?xbf16>` | Gate weight Wg [K, N] |
| Operand `wu` | `tensor<?x?xbf16>` | Up weight Wu [K, N] |
| Init | `tensor<?x?xbf16>` | Output initializer [M, N] |
| `accumulator_type` | `TypeAttr` | `f32` |
| `activation` | `#llk.activation<silu>` | Activation identity |
| `math_mode` | `#llk.math_mode<bounded_fast>` | Numerical policy |
| Result | `tensor<?x?xbf16>` | Output Y [M, N] |

**Verifier invariants:**
- `dim(X, 0) == dim(output, 0)`  (M matches)
- `dim(Wg, 1) == dim(output, 1)` (N matches)
- `dim(X, 1) == dim(Wg, 0)`      (K matches for gate)
- `dim(X, 1) == dim(Wu, 0)`      (K matches for up)
- `dim(Wg, 1) == dim(Wu, 1)`     (N matches across weights)

**Shape inference:** `output_shape = [dim(X, 0), dim(Wg, 1)]`

### 3.3 Attributes

```tablegen
// Activation enum
def LLK_SiLU : I32EnumAttrCase<"silu", 0, "silu">;
def LLK_ActivationAttr : EnumAttr<LLK_Dialect, [LLK_SiLU], "activation">;

// Math mode enum
def LLK_Strict      : I32EnumAttrCase<"strict", 0, "strict">;
def LLK_BoundedFast : I32EnumAttrCase<"bounded_fast", 1, "bounded_fast">;
def LLK_UnsafeFast  : I32EnumAttrCase<"unsafe_fast", 2, "unsafe_fast">;
def LLK_MathModeAttr : EnumAttr<LLK_Dialect, [LLK_Strict, LLK_BoundedFast, LLK_UnsafeFast], "math_mode">;
```

---

## 4. Pass Pipeline & Lowering

The `llk-compile` tool runs this fixed pipeline:

```
llk.fused_swiglu
    ↓  --llk-to-linalg
linalg.matmul (gate) + linalg.matmul (up) + linalg.generic (silu * up)
    ↓  --canonicalize
    ↓  --one-shot-bufferize {bufferize-function-boundaries}
memref.alloc + memref.dealloc + scf loops + arith
    ↓  --convert-scf-to-cf
    ↓  --convert-arith-to-llvm
    ↓  --convert-math-to-llvm
    ↓  --convert-func-to-llvm
    ↓  --convert-memref-to-llvm
    ↓  --reconcile-unrealized-casts
LLVM dialect
    ↓  mlir-translate --mlir-to-llvmir
LLVM IR → ORC JIT → executable function pointer
```

### 4.1 LLKToLinalg Pass

**Algorithm:**
```
RewritePattern matchAndRewrite(llk::FusedSwiGLUOp op):
    loc = op.getLoc()
    M = dim(x, 0); N = dim(wg, 1); K = dim(x, 1)

    // Create f32 accumulators
    gate_init = linalg.fill(value=0.0f32, shape=[M, N])
    up_init   = linalg.fill(value=0.0f32, shape=[M, N])

    // Gate projection: X × Wg → [M, K] × [K, N] → [M, N]
    gate = linalg.matmul ins(x, wg) outs(gate_init)
        : tensor<MxKxbf16>, tensor<KxNxbf16> → tensor<MxNxf32>

    // Up projection: X × Wu → [M, K] × [K, N] → [M, N]
    up = linalg.matmul ins(x, wu) outs(up_init)
        : tensor<MxKxbf16>, tensor<KxNxbf16> → tensor<MxNxf32>

    // Elementwise: SiLU(gate) * up → bf16
    y = linalg.generic {
        indexing_maps = [affine_map<(m,n)->(m,n)>,
                         affine_map<(m,n)->(m,n)>,
                         affine_map<(m,n)->(m,n)>],
        iterator_types = ["parallel", "parallel"]
    } ins(gate, up) outs(out) {
    ^bb0(%g: f32, %u: f32, %old: bf16):
        %neg     = arith.negf %g
        %exp     = math.exp %neg
        %one     = arith.constant 1.0
        %den     = arith.addf %one, %exp
        %sigmoid = arith.divf %one, %den
        %silu    = arith.mulf %g, %sigmoid
        %r       = arith.mulf %silu, %u
        %cast    = arith.truncf %r : f32 → bf16
        linalg.yield %cast : bf16
    } → tensor<MxNxbf16>

    rewriter.replaceOp(op, y)
```

---

## 5. Runtime Components

### 5.1 JIT Cache (Minimal)

Single-level object-code cache keyed by a hash string.

```cpp
class JitCache {
public:
    using KernelFn = void(*)(const Tensor2D*, const Tensor2D*, const Tensor2D*,
                              Tensor2D*, KernelContext*);

    llvm::Expected<KernelFn> lookupOrCompile(const std::string& cache_key,
                                              mlir::ModuleOp module);
    void clear();
    size_t size() const;

private:
    std::unique_ptr<llvm::orc::LLJIT> jit_;
    std::unordered_map<std::string, KernelFn> cache_;
    std::shared_mutex mutex_;
};
```

### 5.2 Stable Public ABI

```cpp
struct Tensor2D {
    void* data;
    int64_t dim0, dim1;
    int64_t stride0, stride1;
};

struct KernelContext {
    void* scratch;
    size_t scratch_size;
    // ThreadPool* thread_pool;  // added in Milestone 4
};

extern "C" void llk_swiglu(
    const Tensor2D* x, const Tensor2D* wg, const Tensor2D* wu,
    Tensor2D* y, KernelContext* context);
```

The function receives raw C structs. A thin ABI wrapper generated during lowering converts to/from MLIR memref descriptors.

---

## 6. Test Specifications

| Test | What it verifies | Tool |
|------|-----------------|------|
| `test/Dialect/ops.mlir` | Round-trip parse → print → parse of `llk.fused_swiglu`; verifier accepts valid shapes | `llk-opt` + FileCheck |
| `test/Dialect/ops_invalid.mlir` | Verifier rejects mismatched M, N, K dimensions | `llk-opt --verify-diagnostics` |
| `test/Conversion/llk_to_linalg.mlir` | LLKToLinalg produces 2× `linalg.matmul` + 1× `linalg.generic` with SiLU body; verify `math.exp`, `arith.truncf` present | `llk-opt --llk-to-linalg` + FileCheck |
| `test/Execution/swiglu_scalar.cpp` | End-to-end for {(1,128,256), (16,512,512), (64,256,128)}: build → compile → JIT → execute vs naive C++ reference. Max abs error ≤ 1e-2 | GTest |
| `test/Execution/swiglu_reference.cpp` | The naive C++ reference itself matches PyTorch for same shapes | GTest |
| `test/Execution/edge_cases.cpp` | M=0 (empty), (M=1,N=1,K=1) unit, K=1 (no reduction) | GTest |

---

## 7. Dependencies

None — bootstrap milestone. Includes: CMake project, LLVM/MLIR dependency, GTest integration, CI skeleton.

---

## 8. Related

- [ARCHITECTURE.md](../../ARCHITECTURE.md) — overall architecture and component map
- [m2-explicit-vector.md](m2-explicit-vector.md) — adds tiling + explicit SIMD on top of this pipeline
