// RUN: llk-opt --verify-diagnostics --split-input-file \
// RUN:   --llk-to-micro="schedule-db=missing.json" %s

// Rejections the Micro-IR export has to make explicit.
//
// These are the failures the LLK dialect's own verifiers cannot catch: an LLK
// operation knows whether its dimensions agree, but only the schedule says
// whether the problem can be covered by tiles. Cases the dialect already
// rejects -- operand rank, K agreement between the operands -- are left to its
// verifiers and are not repeated here.
//
// The schedule database does not exist, so every case that gets as far as the
// schedule lookup also reports the fallback it used. Those warnings are
// expected here because --verify-diagnostics accounts for every diagnostic the
// run produces.

// Dynamic shapes cannot become static !micro.tile values, and the kernel has
// no tile mask to fall back on.
func.func @dynamic_shape(%x: tensor<?x64xbf16>, %wg: tensor<64x64xbf16>,
                         %wu: tensor<64x64xbf16>, %init: tensor<?x64xbf16>)
    -> tensor<?x64xbf16> {
  // expected-error @below {{left-hand side must have a static shape; dynamic shapes are not supported by the Micro-IR export}}
  %y = llk.fused_swiglu ins(%x, %wg, %wu : tensor<?x64xbf16>,
                            tensor<64x64xbf16>, tensor<64x64xbf16>)
      outs(%init : tensor<?x64xbf16>)
      {accumulator_type = f32, activation = #llk.activation<silu>,
       math_mode = #llk.math_mode<bounded_fast>}
      -> tensor<?x64xbf16>
  return %y : tensor<?x64xbf16>
}

// -----

// The schedule's BM does not divide M, so 8-row tiles cannot cover 12 rows.
func.func @m_not_divisible(%x: tensor<12x64xbf16>, %wg: tensor<64x64xbf16>,
                           %wu: tensor<64x64xbf16>, %init: tensor<12x64xbf16>)
    -> tensor<12x64xbf16> {
  // expected-warning @below {{no schedule entry for fused_swiglu}}
  // expected-error @below {{BM tile 8 does not divide BM = 12; tail_policy is 'none'}}
  %y = llk.fused_swiglu ins(%x, %wg, %wu : tensor<12x64xbf16>,
                            tensor<64x64xbf16>, tensor<64x64xbf16>)
      outs(%init : tensor<12x64xbf16>)
      {accumulator_type = f32, activation = #llk.activation<silu>,
       math_mode = #llk.math_mode<bounded_fast>}
      -> tensor<12x64xbf16>
  return %y : tensor<12x64xbf16>
}

// -----

// Same on the reduction axis: a K loop stepping 32 cannot cover K = 48.
func.func @k_not_divisible(%x: tensor<16x48xbf16>, %wg: tensor<48x64xbf16>,
                           %wu: tensor<48x64xbf16>, %init: tensor<16x64xbf16>)
    -> tensor<16x64xbf16> {
  // expected-warning @below {{no schedule entry for fused_swiglu}}
  // expected-error @below {{BK tile 32 does not divide BK = 48; tail_policy is 'none'}}
  %y = llk.fused_swiglu ins(%x, %wg, %wu : tensor<16x48xbf16>,
                            tensor<48x64xbf16>, tensor<48x64xbf16>)
      outs(%init : tensor<16x64xbf16>)
      {accumulator_type = f32, activation = #llk.activation<silu>,
       math_mode = #llk.math_mode<bounded_fast>}
      -> tensor<16x64xbf16>
  return %y : tensor<16x64xbf16>
}

// -----

// The result is not the contraction's output shape. llk.matmul's verifier only
// relates the two operands, so this one is the export's to reject.
func.func @result_shape_mismatch(%a: tensor<16x64xbf16>, %b: tensor<64x64xbf16>,
                                 %init: tensor<16x32xbf16>)
    -> tensor<16x32xbf16> {
  // expected-error @below {{result shape does not match the contraction: expected M = 16 and N = 64}}
  %y = llk.matmul ins(%a, %b : tensor<16x64xbf16>, tensor<64x64xbf16>)
      outs(%init : tensor<16x32xbf16>)
      {accumulator_type = f32, math_mode = #llk.math_mode<bounded_fast>}
      -> tensor<16x32xbf16>
  return %y : tensor<16x32xbf16>
}
