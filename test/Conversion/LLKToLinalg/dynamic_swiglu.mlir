// RUN: llk-opt --llk-to-linalg %s | FileCheck %s

// Dynamic SwiGLU lowered to Linalg. The tensor-level lowering is allowed to
// accept dynamic extents: every temporary it creates is sized from the
// source's own runtime dimensions, and the arithmetic and math-mode semantics
// are unchanged.

module {
  func.func @swiglu_dynamic(%x: tensor<?x?xbf16>, %wg: tensor<?x?xbf16>,
                            %wu: tensor<?x?xbf16>, %init: tensor<?x?xbf16>)
      -> tensor<?x?xbf16> {
    %y = llk.fused_swiglu ins(%x, %wg, %wu : tensor<?x?xbf16>,
                              tensor<?x?xbf16>, tensor<?x?xbf16>)
        outs(%init : tensor<?x?xbf16>)
        {accumulator_type = f32, activation = #llk.activation<silu>,
         math_mode = #llk.math_mode<bounded_fast>}
        -> tensor<?x?xbf16>
    return %y : tensor<?x?xbf16>
  }
}

// CHECK-LABEL: func.func @swiglu_dynamic

// Each temporary is sized from a runtime dimension, never from a fabricated
// static extent.
// CHECK: %[[M:.*]] = tensor.dim %{{.*}}, %{{.*}} : tensor<?x?xbf16>
// CHECK: %[[N:.*]] = tensor.dim %{{.*}}, %{{.*}} : tensor<?x?xbf16>
// CHECK: tensor.empty(%[[M]], %[[N]]) : tensor<?x?xbf16>

// Both projections lower to Linalg matmuls and the epilogue keeps the source
// arithmetic: SiLU in f32, then a bf16 truncation.
// CHECK: linalg.fill
// CHECK: linalg.matmul
// CHECK: linalg.fill
// CHECK: linalg.matmul
// CHECK: linalg.generic
// CHECK: arith.extf
// CHECK: math.exp
// CHECK: arith.truncf

// The lowering never invents a static destination shape.
// CHECK-NOT: tensor<{{[0-9]+}}x{{[0-9]+}}xbf16>
