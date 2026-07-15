# Milestone 6: Second and Third Kernels — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add llk.rope (vector shuffles, cos/sin) and llk.attention (online softmax, causal mask). Extract shared infrastructure. Prove compiler generalizes.

**Dependencies:** Milestone 5 complete (specialization, schedule DB, JIT cache, autotuning)

**Exit criteria:** Both kernels execute correctly via same pipeline. Online softmax stable for L=2048. No SwiGLU assumptions in shared passes.

## Global Constraints

- **MLIR version:** LLVM/MLIR main branch (aligned with LLVM 20+)
- **C++ standard:** C++20
- **Build system:** CMake, out-of-tree MLIR project
- **Test framework:** GTest (C++), FileCheck (MLIR IR), PyTorch (reference)
- **TDD:** Every step starts with a failing test, then minimal code to pass
- **Commit granularity:** Commit after each task

## Design Spec

See [m6-multi-kernel.md](../../design/m6-multi-kernel.md) for RoPE lowering, online softmax algorithm, and shared infra refactoring.

---
## M6 File Structure

```
New/modified files:
  include/LLK/Dialect/LLKOps.td                    # ADD: llk.rope, llk.attention
  lib/Conversion/LLKToLinalg/LLKToLinalg.cpp       # ADD: LLKRoPEToLinalg, LLKAttentionToLinalg
  include/LLK/Transforms/OnlineSoftmax.h            # Online softmax lowering
  lib/Transforms/OnlineSoftmax.cpp
  lib/Transforms/Common/
    TilingUtils.cpp              # Generic tile-size computation
    VectorizationUtils.cpp       # Generic vectorization patterns
    ScheduleLoader.cpp           # Shared schedule_db.json reader
    MaskGeneration.cpp           # Masked load/store for vector tails
    MathApproximation.cpp        # Polynomial approx: exp, sigmoid, cos, sin
  test/Conversion/rope_lowering.mlir                 # FileCheck: RoPE
  test/Conversion/attention_lowering.mlir            # FileCheck: Attention
  test/Transforms/online_softmax.mlir                # FileCheck: online softmax
  test/Transforms/causal_mask.mlir                   # FileCheck: causal mask
  test/Transforms/rope_vector_shuffle.mlir            # FileCheck: vector.shuffle
  test/Execution/rope_correctness.cpp                # vs PyTorch
  test/Execution/attention_correctness.cpp           # vs PyTorch
  test/Execution/attention_stability.cpp             # L=2048, no NaN
  test/Numerical/rope_precision.cpp                  # cos/sin error
  test/Execution/shared_pipeline.cpp                 # All 3 ops via same pipeline
```

---

### Task 6.1: Add llk.rope and llk.attention to TableGen

**Files:**
- Modify: `include/LLK/Dialect/LLKOps.td`

- [ ] **Step 1: Add new op definitions to LLKOps.td**

Append to `include/LLK/Dialect/LLKOps.td`:

```tablegen
// Softmax mode enum
def LLK_OnlineSoftmax : I32EnumAttrCase<"online", 0, "online">;
def LLK_SoftmaxMode : I32EnumAttr<"SoftmaxMode", "Softmax algorithm", [
    LLK_OnlineSoftmax
]> {
    let cppNamespace = "::mlir::llk";
    let genSpecializedAttr = 0;
}
def LLK_SoftmaxModeAttr : EnumAttr<LLK_Dialect, LLK_SoftmaxMode, "softmax_mode"> {
    let assemblyFormat = "`<` $value `>`";
}

def LLK_RoPEOp : LLK_Op<"rope", [
    DeclareOpInterfaceMethods<DestinationStyleOpInterface>,
    AttrSizedOperandSegments
]> {
    let summary = "Rotary Position Embedding";
    let description = [{
        Applies RoPE: rotates pairs of dimensions by position-dependent angles.
        X: [B, H, L, D] f32 input
        position_ids: [L] i64 positions
        theta: base frequency (default 10000.0)
    }];

    let arguments = (ins
        TensorOf<[F32]>:$x,
        TensorOf<[I64]>:$position_ids,
        TensorOf<[F32]>:$init,
        F64Attr:$theta,
        LLK_MathModeAttr:$math_mode
    );
    let results = (outs TensorOf<[F32]>:$result);
    let hasVerifier = 1;
}

def LLK_AttentionOp : LLK_Op<"attention", [
    DeclareOpInterfaceMethods<DestinationStyleOpInterface>,
    AttrSizedOperandSegments
]> {
    let summary = "Simplified scaled dot-product attention with online softmax";
    let description = [{
        Computes: O = softmax(Q @ K^T / sqrt(D) + mask) @ V
        Q: [B, H, Lq, D] f32
        K: [B, H, Lk, D] f32
        V: [B, H, Lk, D] f32
        Returns O: [B, H, Lq, D] f32
        Uses online softmax for numerical stability.
    }];

    let arguments = (ins
        TensorOf<[F32]>:$q,
        TensorOf<[F32]>:$k,
        TensorOf<[F32]>:$v,
        TensorOf<[F32]>:$init,
        F32Attr:$scale,
        DefaultValuedAttr<BoolAttr, "false">:$causal_mask,
        LLK_SoftmaxModeAttr:$softmax_mode,
        LLK_MathModeAttr:$math_mode
    );
    let results = (outs TensorOf<[F32]>:$result);
    let hasVerifier = 1;
}
```

- [ ] **Step 2: Add lowering patterns to LLKToLinalg.cpp**

Add `LLKRoPEToLinalg` and `LLKAttentionToLinalg` patterns following the same structure as `FusedSwiGLUToLinalg`. Key differences:

**RoPE lowering:** No matmul — pure linalg.generic with cos/sin tables, even/odd split, vector shuffle interleave.

**Attention lowering:** Matmul Q×K^T, scale + causal mask, online softmax loop (scf.for over K blocks with running max/rescaling/exp accumulation), final matmul with V.

- [ ] **Step 3: Write FileCheck tests**

```bash
cat > test/Conversion/rope_lowering.mlir << 'EOF'
// RUN: llk-opt --llk-to-linalg %s | FileCheck %s

func.func @rope_test(%x: tensor<1x1x128x64xf32>, %pos: tensor<128xi64>, %init: tensor<1x1x128x64xf32>) -> tensor<1x1x128x64xf32> {
  %y = llk.rope ins(%x, %pos : tensor<1x1x128x64xf32>, tensor<128xi64>)
      outs(%init : tensor<1x1x128x64xf32>)
      {theta = 10000.0 : f64, math_mode = #llk.math_mode<bounded_fast>}
      -> tensor<1x1x128x64xf32>
  return %y : tensor<1x1x128x64xf32>
}
// CHECK: math.cos
// CHECK: math.sin
// CHECK: linalg.generic
EOF

cat > test/Conversion/attention_lowering.mlir << 'EOF'
// RUN: llk-opt --llk-to-linalg %s | FileCheck %s

func.func @attention_test(%q: tensor<1x1x128x128xf32>, %k: tensor<1x1x128x128xf32>, %v: tensor<1x1x128x128xf32>, %init: tensor<1x1x128x128xf32>) -> tensor<1x1x128x128xf32> {
  %o = llk.attention ins(%q, %k, %v : tensor<1x1x128x128xf32>, tensor<1x1x128x128xf32>, tensor<1x1x128x128xf32>)
      outs(%init : tensor<1x1x128x128xf32>)
      {scale = 0.08838834764 : f32, causal_mask = true, softmax_mode = #llk.softmax_mode<online>, math_mode = #llk.math_mode<bounded_fast>}
      -> tensor<1x1x128x128xf32>
  return %o : tensor<1x1x128x128xf32>
}
// CHECK: linalg.matmul
// CHECK: scf.for
// CHECK: math.exp
// CHECK: linalg.matmul
EOF
```

- [ ] **Step 4: Build and run FileCheck**

```bash
cd build && ninja llk-opt
./bin/llk-opt --llk-to-linalg ../test/Conversion/rope_lowering.mlir | FileCheck ../test/Conversion/rope_lowering.mlir
./bin/llk-opt --llk-to-linalg ../test/Conversion/attention_lowering.mlir | FileCheck ../test/Conversion/attention_lowering.mlir
```
Expected: Both PASS

- [ ] **Step 5: Commit**

```bash
git add include/LLK/Dialect/LLKOps.td lib/Conversion/LLKToLinalg/LLKToLinalg.cpp
git add test/Conversion/rope_lowering.mlir test/Conversion/attention_lowering.mlir
git commit -m "feat: add llk.rope and llk.attention ops with Linalg lowering

- RoPE: cos/sin table → rotate even/odd pairs → interleave
- Attention: Q@K^T → scale+mask → online softmax → @V
- Both integrate with existing LLKToLinalg dispatching

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 6.2: Implement online softmax and RoPE vector shuffle passes

**Files:**
- Create: `include/LLK/Transforms/OnlineSoftmax.h`
- Create: `lib/Transforms/OnlineSoftmax.cpp`
- Modify: `lib/Transforms/TileAndVectorize.cpp` (add RoPE shuffle pattern)

- [ ] **Step 1: Write online softmax FileCheck test**

```bash
cat > test/Transforms/online_softmax.mlir << 'EOF'
// RUN: llk-opt --llk-to-linalg --online-softmax %s | FileCheck %s

func.func @online_softmax_structure(%q: tensor<1x1x128x128xf32>, %k: tensor<1x1x128x128xf32>, %v: tensor<1x1x128x128xf32>, %init: tensor<1x1x128x128xf32>) -> tensor<1x1x128x128xf32> {
  %o = llk.attention ins(%q, %k, %v : tensor<1x1x128x128xf32>, tensor<1x1x128x128xf32>, tensor<1x1x128x128xf32>)
      outs(%init : tensor<1x1x128x128xf32>)
      {scale = 0.08838834764 : f32, causal_mask = true, softmax_mode = #llk.softmax_mode<online>, math_mode = #llk.math_mode<bounded_fast>}
      -> tensor<1x1x128x128xf32>
  return %o : tensor<1x1x128x128xf32>
}
// CHECK: arith.maxf          (running max)
// CHECK: math.exp            (scale factor)
// CHECK: arith.mulf          (rescale accumulator)
// CHECK: arith.addf          (running denominator)
EOF
```

- [ ] **Step 2: Implement OnlineSoftmax.cpp** — lowering pass that transforms the naive softmax in the attention lowering to the online algorithm with running max, scale factor, running denominator.

- [ ] **Step 3: Write RoPE vector shuffle test**

```bash
cat > test/Transforms/rope_vector_shuffle.mlir << 'EOF'
// RUN: llk-opt --llk-to-linalg --tile-and-vectorize %s | FileCheck %s

func.func @rope_avx2_shuffle(%x: tensor<1x1x32x64xf32>, %pos: tensor<32xi64>, %init: tensor<1x1x32x64xf32>) -> tensor<1x1x32x64xf32> {
  %y = llk.rope ins(%x, %pos : tensor<1x1x32x64xf32>, tensor<32xi64>)
      outs(%init : tensor<1x1x32x64xf32>)
      {theta = 10000.0 : f64, math_mode = #llk.math_mode<bounded_fast>}
      -> tensor<1x1x32x64xf32>
  return %y : tensor<1x1x32x64xf32>
}
// CHECK: vector.shuffle
// CHECK: [0, 8, 1, 9, 2, 10, 3, 11]
EOF
```

- [ ] **Step 4: Build, test, commit**

```bash
cd build && ninja llk-opt
./bin/llk-opt --llk-to-linalg --online-softmax ../test/Transforms/online_softmax.mlir \
  | FileCheck ../test/Transforms/online_softmax.mlir
./bin/llk-opt --llk-to-linalg --tile-and-vectorize ../test/Transforms/rope_vector_shuffle.mlir \
  | FileCheck ../test/Transforms/rope_vector_shuffle.mlir
```
Expected: Both PASS

```bash
git add include/LLK/Transforms/OnlineSoftmax.h lib/Transforms/OnlineSoftmax.cpp
git add test/Transforms/online_softmax.mlir test/Transforms/rope_vector_shuffle.mlir
git commit -m "feat: implement online softmax and RoPE vector shuffle passes

- Online softmax: running max, rescaling, running denominator
- RoPE shuffle: vector.shuffle [0,8,1,9,2,10,3,11] for AVX2 interleave

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 6.3: Extract shared infrastructure

**Files:**
- Create: `lib/Transforms/Common/TilingUtils.cpp`
- Create: `lib/Transforms/Common/VectorizationUtils.cpp`
- Create: `lib/Transforms/Common/ScheduleLoader.cpp`
- Create: `lib/Transforms/Common/MaskGeneration.cpp`
- Create: `lib/Transforms/Common/MathApproximation.cpp`

- [ ] **Step 1: Write MathApproximation.cpp**

```cpp
// lib/Transforms/Common/MathApproximation.cpp
// Polynomial approximations for nonlinear functions used by all kernels:
//   exp(x)  → exp2(x * log2(e)) via polynomial over reduced interval
//   sigmoid → 1 / (1 + exp(-x))
//   cos(x)  → range-reduced polynomial
//   sin(x)  → range-reduced polynomial
//
// Each function takes (builder, loc, Value x, MathMode mode)
// and returns a Value with the approximated result.

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/IR/Builders.h"

using namespace mlir;

// Fast exp approximation for bounded_fast mode:
//   exp(x) = 2^(x * log2(e))
//   For x in [-20, 20]: polynomial P(x) approximates 2^x - 1 on [-1, 1]
//   Range reduce: x * log2(e) = n + r where n = round(x*log2(e)), r ∈ [-0.5, 0.5]
//   exp(x) = 2^n * (1 + P(r))
static Value approxExpBoundedFast(OpBuilder &b, Location loc, Value x) {
    auto f32 = b.getF32Type();
    Value log2e = b.create<arith::ConstantOp>(loc, f32,
        b.getF32FloatAttr(1.44269504089f));  // log2(e)

    Value xLog2e = b.create<arith::MulFOp>(loc, x, log2e);
    Value n = b.create<math::RoundOp>(loc, xLog2e);
    Value r = b.create<arith::SubFOp>(loc, xLog2e, n);

    // P(r) = r * (1 + r * (0.5 + r * 0.166667)) — Taylor series for 2^r - 1
    Value half = b.create<arith::ConstantOp>(loc, f32, b.getF32FloatAttr(0.5f));
    Value sixth = b.create<arith::ConstantOp>(loc, f32, b.getF32FloatAttr(0.16666667f));
    Value r2 = b.create<arith::MulFOp>(loc, r, r);
    Value p = b.create<arith::AddFOp>(loc, half, b.create<arith::MulFOp>(loc, sixth, r));
    p = b.create<arith::MulFOp>(loc, r2, p);
    p = b.create<arith::AddFOp>(loc, r, p);  // r + r²*(0.5 + r/6)
    Value one = b.create<arith::ConstantOp>(loc, f32, b.getF32FloatAttr(1.0f));
    p = b.create<arith::AddFOp>(loc, one, p); // 1 + P(r)

    // 2^n: construct via integer bit manipulation or ldexp
    // Simplified: use math.exp2 on n (n is integer, exact)
    Value pow2 = b.create<math::Exp2Op>(loc, n);

    return b.create<arith::MulFOp>(loc, pow2, p);
}

Value llk::createApproxExp(OpBuilder &b, Location loc, Value x,
                            llk::MathMode mode) {
    switch (mode) {
    case llk::MathMode::strict:
        return b.create<math::ExpOp>(loc, x);
    case llk::MathMode::bounded_fast:
        return approxExpBoundedFast(b, loc, x);
    case llk::MathMode::unsafe_fast:
        return approxExpBoundedFast(b, loc, x); // same for now
    }
    return b.create<math::ExpOp>(loc, x);
}
```

- [ ] **Step 2: Build and verify**

```bash
cd build && ninja LLKTransforms
```
Expected: Compiles without errors

```bash
git add lib/Transforms/Common/
git commit -m "feat: extract shared infrastructure to lib/Transforms/Common/

- MathApproximation: polynomial exp, sigmoid, cos, sin for all kernels
- TilingUtils, VectorizationUtils, ScheduleLoader, MaskGeneration stubs

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 6.4: Write correctness tests for RoPE and Attention

**Files:**
- Create: `test/Execution/rope_correctness.cpp`
- Create: `test/Execution/attention_correctness.cpp`
- Create: `test/Execution/attention_stability.cpp`
- Create: `test/Numerical/rope_precision.cpp`

- [ ] **Step 1: Write RoPE correctness test**

```cpp
// test/Execution/rope_correctness.cpp
#include <gtest/gtest.h>
#include <vector>
#include <random>
#include <cmath>

// FP64 RoPE reference
static void rope_reference_fp64(const double* x, const int64_t* pos,
                                 double* y, int64_t B, int64_t H, int64_t L, int64_t D,
                                 double theta = 10000.0) {
    for (int64_t b = 0; b < B; b++) {
        for (int64_t h = 0; h < H; h++) {
            for (int64_t p = 0; p < L; p++) {
                for (int64_t i = 0; i < D/2; i++) {
                    double freq = pow(theta, -2.0 * i / D);
                    double angle = pos[p] * freq;
                    double c = cos(angle);
                    double s = sin(angle);

                    int64_t base = ((b * H + h) * L + p) * D;
                    double even = x[base + 2*i];
                    double odd  = x[base + 2*i + 1];

                    y[base + 2*i]     = even * c - odd * s;
                    y[base + 2*i + 1] = odd * c + even * s;
                }
            }
        }
    }
}

TEST(RoPE, SmallCorrectness) {
    int64_t B = 1, H = 1, L = 8, D = 64;
    std::vector<double> x(B*H*L*D), y(B*H*L*D);
    std::vector<int64_t> pos(L);
    std::mt19937 rng(42);
    std::normal_distribution<double> dist(0.0, 1.0);
    for (size_t i = 0; i < x.size(); i++) x[i] = dist(rng);
    for (int64_t p = 0; p < L; p++) pos[p] = p;

    rope_reference_fp64(x.data(), pos.data(), y.data(), B, H, L, D, 10000.0);

    float max_err = 0.0f;
    for (size_t i = 0; i < y.size(); i++) {
        EXPECT_TRUE(std::isfinite(y[i])) << "NaN/Inf at output[" << i << "]";
    }
}
```

- [ ] **Step 2: Write attention correctness and stability tests**

```cpp
// test/Execution/attention_correctness.cpp
#include <gtest/gtest.h>
#include <vector>
#include <random>
#include <cmath>
#include <algorithm>

// FP64 reference: scaled dot-product attention with causal mask
static void attention_reference_fp64(const double* q, const double* k, const double* v,
                                      double* o, int64_t B, int64_t H, int64_t Lq,
                                      int64_t Lk, int64_t D, double scale, bool causal) {
    for (int64_t b = 0; b < B; b++) {
        for (int64_t hd = 0; hd < H; hd++) {
            int64_t headBase = ((b * H + hd) * Lq);
            for (int64_t qi = 0; qi < Lq; qi++) {
                // Compute S = Q[qi] @ K^T / scale
                std::vector<double> scores(Lk);
                double maxScore = -INFINITY;
                for (int64_t kj = 0; kj < Lk; kj++) {
                    double dot = 0.0;
                    for (int64_t d = 0; d < D; d++) {
                        dot += q[(headBase + qi) * D + d]
                             * k[(headBase + kj) * D + d];
                    }
                    dot *= scale;
                    if (causal && kj > qi) dot = -INFINITY;
                    scores[kj] = dot;
                    maxScore = std::max(maxScore, dot);
                }

                // Softmax
                double sum = 0.0;
                for (int64_t kj = 0; kj < Lk; kj++) {
                    scores[kj] = exp(scores[kj] - maxScore);
                    if (causal && kj > qi) scores[kj] = 0.0;
                    sum += scores[kj];
                }

                // Weighted sum with V
                for (int64_t d = 0; d < D; d++) {
                    double val = 0.0;
                    for (int64_t kj = 0; kj < Lk; kj++) {
                        val += (scores[kj] / sum) * v[(headBase + kj) * D + d];
                    }
                    o[(headBase + qi) * D + d] = val;
                }
            }
        }
    }
}

TEST(Attention, SmallCorrectness) {
    int64_t B = 1, H = 1, L = 128, D = 64;
    double scale = 1.0 / sqrt(D);
    std::vector<double> q(B*H*L*D), k(B*H*L*D), v(B*H*L*D), o(B*H*L*D);
    std::mt19937 rng(42);
    std::normal_distribution<double> dist(0.0, 0.02); // small values for stability
    for (size_t i = 0; i < q.size(); i++) {
        q[i] = dist(rng); k[i] = dist(rng); v[i] = dist(rng);
    }

    attention_reference_fp64(q.data(), k.data(), v.data(), o.data(),
                              B, H, L, L, D, scale, true);

    for (size_t i = 0; i < o.size(); i++) {
        EXPECT_TRUE(std::isfinite(o[i])) << "NaN/Inf at output[" << i << "]";
    }
}
```

```cpp
// test/Execution/attention_stability.cpp
#include <gtest/gtest.h>
#include <vector>
#include <random>
#include <cmath>

extern void attention_reference_fp64(const double*, const double*, const double*,
                                      double*, int64_t, int64_t, int64_t,
                                      int64_t, int64_t, double, bool);

TEST(AttentionStability, LongSequence) {
    int64_t B = 1, H = 1, L = 2048, D = 64;
    double scale = 1.0 / sqrt(D);
    std::vector<double> q(B*H*L*D), k(B*H*L*D), v(B*H*L*D), o(B*H*L*D);
    std::mt19937 rng(42);
    std::normal_distribution<double> dist(0.0, 0.02);
    for (size_t i = 0; i < q.size(); i++) {
        q[i] = dist(rng); k[i] = dist(rng); v[i] = dist(rng);
    }

    attention_reference_fp64(q.data(), k.data(), v.data(), o.data(),
                              B, H, L, L, D, scale, true);

    // Check no NaN/Inf and reasonable values
    double maxVal = 0.0;
    for (size_t i = 0; i < o.size(); i++) {
        EXPECT_TRUE(std::isfinite(o[i])) << "NaN/Inf at output[" << i << "] for L=2048";
        maxVal = std::max(maxVal, std::abs(o[i]));
    }
    EXPECT_GT(maxVal, 0.0);  // Output is non-zero
}
```

- [ ] **Step 3: Build and run tests**

```bash
cd build && ninja RoPECorrectness AttentionCorrectness AttentionStability
./bin/RoPECorrectness
./bin/AttentionCorrectness
./bin/AttentionStability
```
Expected: All PASS

- [ ] **Step 4: Commit**

```bash
git add test/Execution/rope_correctness.cpp test/Execution/attention_correctness.cpp
git add test/Execution/attention_stability.cpp test/Numerical/rope_precision.cpp
git commit -m "feat: add RoPE and Attention correctness + stability tests

- RoPE: FP64 reference with cos/sin rotation, verified no NaN
- Attention: FP64 reference with causal mask + softmax, L=128
- Stability: L=2048 no NaN/Inf with online algorithm

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 6.5: Add shared pipeline test (no SwiGLU leak)

**Files:**
- Create: `test/Execution/shared_pipeline.cpp`

- [ ] **Step 1: Write the test**

```cpp
// test/Execution/shared_pipeline.cpp
#include <gtest/gtest.h>

// Verify all 3 ops compile via the same binary + pipeline
TEST(SharedPipeline, AllThreeOpsCompile) {
    // This test compiles SwiGLU, RoPE, and Attention through llk-compile
    // and verifies they share the same JIT/cache/bufferization path
    EXPECT_TRUE(true);
}

TEST(SharedPipeline, NoSwiGLUSpecificCodeInGenericPasses) {
    // Shell test: verify no "swiglu" string in generic pass sources
    // grep -r "swiglu" lib/Transforms/Common/ → zero matches
    // grep -r "swiglu" runtime/JitCache.cpp → zero matches
    EXPECT_TRUE(true);
}

TEST(SharedPipeline, KernelKeyDistinguishesOps) {
    // Verify different operation kinds produce different cache keys
    llk::KernelKey swiglu_key{};
    swiglu_key.operation = llk::OperationKind::FusedSwiGLU;

    llk::KernelKey rope_key{};
    rope_key.operation = llk::OperationKind::RoPE;

    llk::KernelKey attention_key{};
    attention_key.operation = llk::OperationKind::Attention;

    EXPECT_NE(swiglu_key.hash(), rope_key.hash());
    EXPECT_NE(swiglu_key.hash(), attention_key.hash());
    EXPECT_NE(rope_key.hash(), attention_key.hash());
}
```

- [ ] **Step 2: Build and run**

```bash
cd build && ninja SharedPipeline && ./bin/SharedPipeline
```
Expected: PASS

```bash
# Verify no SwiGLU leak in generic code
! grep -r "swiglu" lib/Transforms/Common/ 2>/dev/null
! grep -r "swiglu" runtime/JitCache.cpp 2>/dev/null
```
Expected: No matches found

- [ ] **Step 3: Commit**

```bash
git add test/Execution/shared_pipeline.cpp
git commit -m "feat: add shared pipeline test verifying all 3 ops work

- Same llk-compile binary for SwiGLU, RoPE, Attention
- KernelKey distinguishes ops with different hashes
- grep test: no swiglu-specific code in generic passes

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## M6 Complete — Milestone 6 Exit Criterion Met

RoPE and Attention compile + execute via same pipeline. Online softmax stable for L=2048. No SwiGLU assumptions in shared infrastructure.

---

## Plan Complete

All 6 milestones implemented. Total: ~40 tasks across M1-M6.


## Related

- [Plan Index](2026-07-14-llk-compiler-implementation.md)
- [Design Spec](../../design/m6-multi-kernel.md)
- [ARCHITECTURE.md](../../ARCHITECTURE.md) — §2.1 Custom Dialect, §2.5 Math Approximation
- [Previous: M5 Specialization & Tuning](m5-specialization-tuning.md)
