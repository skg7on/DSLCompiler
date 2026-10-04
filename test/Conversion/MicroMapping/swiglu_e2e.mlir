// RUN: llk-opt --llk-to-micro="schedule-db=missing.json" %s \
// RUN:   | llk-opt "--micro-map=target=x86-avx2 \
// RUN:       machine=%p/../../../machines/x86-avx2-v2.yaml \
// RUN:       layouts=%p/../../../mapping/x86-avx2/layouts.llkmap \
// RUN:       rules=%p/../../../mapping/x86-avx2/rules.llkmap \
// RUN:       emitters=avx2_vector_add,avx2_vector_convert,avx2_vector_silu,avx2_vector_mul,avx2_mma,avx2_reduce,avx2_copy,avx2_tile_copy,avx2_tile_store \
// RUN:       mode=exact" \
// RUN:   | FileCheck %s
//
// The RUN line above is documentation only: this project has no lit runner, and
// `//` lines are comments. The real invocation is the MicroMappingSwiGLUE2E
// CTest entry in CMakeLists.txt, which runs the same two-pass pipeline with
// absolute paths.
//
// End-to-end acceptance for gap #3: the shipped AVX2 target must map a
// compiler-generated SwiGLU. The input is exactly the LLK source of
// test/Conversion/LLKToMicro/swiglu_to_micro.mlir, so the kernel under test is
// produced live by `--llk-to-micro` (never hand-authored), and the second pass
// is the shipped `--micro-map` with the shipped `x86-avx2` rule file. The
// schedule database is pointed at a missing path so the built-in conservative
// schedule (BM=8, BN=32, BK=32) fixes the emitted tile program.

func.func @swiglu(%x: tensor<16x64xbf16>, %wg: tensor<64x64xbf16>,
                  %wu: tensor<64x64xbf16>, %init: tensor<16x64xbf16>)
    -> tensor<16x64xbf16> {
  %y = llk.fused_swiglu ins(%x, %wg, %wu : tensor<16x64xbf16>,
                            tensor<64x64xbf16>, tensor<64x64xbf16>)
      outs(%init : tensor<16x64xbf16>)
      {accumulator_type = f32, activation = #llk.activation<silu>,
       math_mode = #llk.math_mode<bounded_fast>}
      -> tensor<16x64xbf16>
  return %y : tensor<16x64xbf16>
}

// The export is non-destructive: the source function keeps its LLK op.
// CHECK-LABEL: func.func @swiglu
// CHECK: llk.fused_swiglu

// The mapper binds a complete plan onto the lowering-produced kernel.
// CHECK-LABEL: micro.kernel @fused_swiglu_M16_N64_K64 attributes {
// CHECK-SAME: micro.plan

// Staged copies carry their placement.
// CHECK: micro.tile_async_copy
// CHECK-SAME: micro.mapping

// Both projections bind to the matrix engine.
// CHECK: micro.mma
// CHECK-SAME: micro.mapping
// CHECK: micro.mma
// CHECK-SAME: micro.mapping

// The gating epilogue is three vector variants -- SiLU, the multiply, and the
// narrowing conversion -- none of which had a rule before this task.
// CHECK: micro.vector "silu"
// CHECK-SAME: micro.mapping
// CHECK: micro.vector "mul"
// CHECK-SAME: micro.mapping
// CHECK: micro.vector "convert"
// CHECK-SAME: micro.mapping

// The write back carries a placement too.
// CHECK: micro.tile_store
// CHECK-SAME: micro.mapping
