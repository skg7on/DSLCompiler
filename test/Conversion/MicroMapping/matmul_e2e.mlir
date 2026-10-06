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
// `//` lines are comments. The real invocation is the MicroMappingMatmulE2E
// CTest entry in CMakeLists.txt, which runs the same two-pass pipeline with
// absolute paths.
//
// End-to-end acceptance for gap #3: the shipped AVX2 target must map a
// compiler-generated matmul. The input is exactly the LLK source of
// test/Conversion/LLKToMicro/matmul_to_micro.mlir, so the kernel under test is
// produced live by `--llk-to-micro` (never hand-authored), and the second pass
// is the shipped `--micro-map` with the shipped `x86-avx2` rule file. The
// schedule database is pointed at a missing path so the built-in conservative
// schedule (BM=8, BN=32, BK=32) fixes the emitted tile program.

func.func @matmul(%a: tensor<16x64xbf16>, %b: tensor<64x64xbf16>,
                  %init: tensor<16x64xbf16>) -> tensor<16x64xbf16> {
  %y = llk.matmul ins(%a, %b : tensor<16x64xbf16>, tensor<64x64xbf16>)
      outs(%init : tensor<16x64xbf16>)
      {accumulator_type = f32, math_mode = #llk.math_mode<bounded_fast>}
      -> tensor<16x64xbf16>
  return %y : tensor<16x64xbf16>
}

// The export is non-destructive: the source function keeps its LLK op.
// CHECK-LABEL: func.func @matmul
// CHECK: llk.matmul

// The mapper binds a complete plan onto the lowering-produced kernel: the
// `micro.plan` marker appears, and no node is left with `no_matching_rule`.
// CHECK: micro.kernel @matmul_M16_N64_K64(%{{.*}}: tensor<16x64xbf16>, %{{.*}}: tensor<64x64xbf16>) -> tensor<16x64xbf16> attributes {
// CHECK-SAME: micro.plan

// Staged copies carry their placement, bound by their own rule.
// CHECK: micro.tile_async_copy
// CHECK-SAME: rule = "avx2.tile_async_copy"

// The GEMM core binds to the matrix engine.
// CHECK: micro.mma
// CHECK-SAME: rule = "avx2.mma_bf16"

// The narrowing epilogue -- `micro.vector "convert"` -- is the op that had no
// rule before this task.
// CHECK: micro.vector "convert"
// CHECK-SAME: rule = "avx2.vector_convert"

// The write back carries a placement too.
// CHECK: micro.tile_store
// CHECK-SAME: rule = "avx2.tile_store"

//===----------------------------------------------------------------------===//
// Stage C3: the whole conversion traverses
//===----------------------------------------------------------------------===//
//
// This is the pipeline a compiler takes, and it has to finish: the spatial
// loops become `scf.for` carrying the output, the contraction becomes a
// `linalg.matmul` whose accumulator is the value the K loop threads, and the
// write-back becomes an `tensor.insert_slice` into the carried output. Nothing
// micro is left for a backend to trip over.
// TRAVERSE-LABEL: func.func @matmul_M16_N64_K64
// TRAVERSE: scf.for
// TRAVERSE: tensor.extract_slice
// TRAVERSE: linalg.matmul
// TRAVERSE: tensor.insert_slice
// TRAVERSE: return
// TRAVERSE-NOT: micro.

// The numeric half of this chain's acceptance is
// test/Execution/mapped_acceptance.cpp: it exports the same LLK source, maps
// it against the shipped AVX2 target and *invokes* the result, checking every
// element. This file asserts the IR; that one asserts the answer.
