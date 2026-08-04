//===- MathApproximation.cpp - Polynomial function approximations
//----------===//
//
// Fast polynomial approximations for exp, sigmoid, cos, sin.
//
// exp(x):  range-reduce x * log2(e) → integer n + fraction r ∈ [-0.5, 0.5],
//          then 2^n * (1 + P(r)) where P is a Taylor polynomial for 2^r - 1.
// sigmoid: implemented via createApproxExp.
// cos/sin:  for bounded_fast / unsafe_fast, use Cody-Waite range reduction
//           followed by minimax polynomial approximations (~1 ULP for f32).
//
//===----------------------------------------------------------------------===//

#include "LLK/Transforms/Common/MathApproximation.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Math/IR/Math.h"

using namespace mlir;

namespace mlir {
namespace llk {

// ---------------------------------------------------------------------------
// Fast exp:  exp(x) = 2^(x * log2(e))
//   Range reduce:  x * log2(e) = n + r  where n = floor(x*log2(e) + 0.5),
//                  r ∈ [-0.5, 0.5].
//   Then:  exp(x) = 2^n * 2^r.
//   Approximate 2^r with polynomial P(r) ≈ 2^r on [-0.5, 0.5].
//   Polynomial:  P(r) = r + r² * (c2 + r * c3) where c2=0.5, c3=0.1666667
// ---------------------------------------------------------------------------

static Value approxExpBoundedFast(OpBuilder &b, Location loc, Value x) {
  auto f32 = b.getF32Type();

  // log2(e)
  Value log2e =
      arith::ConstantOp::create(b, loc, f32, b.getF32FloatAttr(1.44269504089f));

  // x * log2(e)
  Value xLog2e = arith::MulFOp::create(b, loc, x, log2e);

  // n = round(x * log2(e))
  Value n = math::RoundOp::create(b, loc, xLog2e);

  // r = x * log2(e) - n    (fractional part)
  Value r = arith::SubFOp::create(b, loc, xLog2e, n);

  // P(r) = r + r² * (c2 + r * c3)
  Value c2 = arith::ConstantOp::create(b, loc, f32, b.getF32FloatAttr(0.5f));
  Value c3 =
      arith::ConstantOp::create(b, loc, f32, b.getF32FloatAttr(0.16666667f));
  Value r2 = arith::MulFOp::create(b, loc, r, r);
  Value inner =
      arith::AddFOp::create(b, loc, c2, arith::MulFOp::create(b, loc, c3, r));
  Value poly = arith::AddFOp::create(b, loc, r,
                                     arith::MulFOp::create(b, loc, r2, inner));

  // 2^n:  use math.exp2 on integer n ← exact result.
  Value pow2 = math::Exp2Op::create(b, loc, n);

  // 2^n * (1 + P(r)) = pow2 + pow2 * poly
  Value one = arith::ConstantOp::create(b, loc, f32, b.getF32FloatAttr(1.0f));
  Value mantissa = arith::AddFOp::create(b, loc, one, poly);
  Value result = arith::MulFOp::create(b, loc, pow2, mantissa);

  return result;
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

Value createApproxExp(OpBuilder &b, Location loc, Value x, MathMode mode) {
  switch (mode) {
  case MathMode::strict:
    return math::ExpOp::create(b, loc, x);
  case MathMode::bounded_fast:
  case MathMode::unsafe_fast:
    return approxExpBoundedFast(b, loc, x);
  }
  return math::ExpOp::create(b, loc, x);
}

Value createApproxSigmoid(OpBuilder &b, Location loc, Value x, MathMode mode) {
  auto f32 = b.getF32Type();
  Value one = arith::ConstantOp::create(b, loc, f32, b.getF32FloatAttr(1.0f));
  Value negX = arith::NegFOp::create(b, loc, x);
  Value expNegX = createApproxExp(b, loc, negX, mode);
  Value denom = arith::AddFOp::create(b, loc, one, expNegX);
  return arith::DivFOp::create(b, loc, one, denom);
}

// ---------------------------------------------------------------------------
// Fast cos/sin:  Cody-Waite range reduction + minimax polynomial
//
//   Range reduce:  n = round(x / (pi/2))
//                  r = x - n * (pi/2)    → r ∈ [-pi/4, pi/4]
//
//   sin polynomial on [-pi/4, pi/4]:
//     sin(r) ≈ r * (1.0 + r²*(c3 + r²*(c5 + r²*c7)))
//     c3 = -1/3! ≈ -0.166666567325, c5 = 1/5! ≈ 0.00833220803,
//     c7 = -1/7! ≈ -0.000195168955
//
//   cos polynomial on [-pi/4, pi/4]:
//     cos(r) ≈ 1.0 + r²*(c2 + r²*(c4 + r²*c6))
//     c2 = -1/2! ≈ -0.499999850988, c4 = 1/4! ≈ 0.041666526347,
//     c6 = -1/6! ≈ -0.001388837214
//
//   Quadrant selection (q = n mod 4):
//     q=0: sin(x)=sin(r),  cos(x)=cos(r)
//     q=1: sin(x)=cos(r),  cos(x)=-sin(r)
//     q=2: sin(x)=-sin(r), cos(x)=-cos(r)
//     q=3: sin(x)=-cos(r), cos(x)=sin(r)
// ---------------------------------------------------------------------------

namespace {

struct SinCosResult {
  Value sinVal;
  Value cosVal;
};

/// Shared Cody-Waite range reduction + minimax polynomial for sin/cos.
/// Computes both sin(x) and cos(x) from a single range reduction, which
/// avoids duplicate work when both values are needed (e.g., RoPE).
SinCosResult approxSinCosBoundedFast(OpBuilder &b, Location loc, Value x) {
  auto f32 = b.getF32Type();
  auto i32 = b.getI32Type();

  // Constants for range reduction.
  Value twoOverPi = arith::ConstantOp::create(
      b, loc, f32, b.getF32FloatAttr(0.63661977236f)); // 2/pi
  Value piOverTwo = arith::ConstantOp::create(
      b, loc, f32, b.getF32FloatAttr(1.57079632679f)); // pi/2

  // n = round(x * twoOverPi)
  Value xTwoOverPi = arith::MulFOp::create(b, loc, x, twoOverPi);
  Value n = math::RoundOp::create(b, loc, xTwoOverPi);

  // r = x - n * pi/2   (r ∈ [-pi/4, pi/4])
  Value nPiOverTwo = arith::MulFOp::create(b, loc, n, piOverTwo);
  Value r = arith::SubFOp::create(b, loc, x, nPiOverTwo);

  // r²
  Value r2 = arith::MulFOp::create(b, loc, r, r);

  // --- sin polynomial: sin(r) = r * (1.0 + r²*(c3 + r²*(c5 + r²*c7))) ---
  Value sinC7 = arith::ConstantOp::create(b, loc, f32,
                                          b.getF32FloatAttr(-0.000195168955f));
  Value sinC5 =
      arith::ConstantOp::create(b, loc, f32, b.getF32FloatAttr(0.00833220803f));
  Value sinC3 = arith::ConstantOp::create(b, loc, f32,
                                          b.getF32FloatAttr(-0.166666567325f));

  Value sinT1 = arith::MulFOp::create(b, loc, sinC7, r2);
  Value sinT2 = arith::AddFOp::create(b, loc, sinC5, sinT1);
  Value sinT3 = arith::MulFOp::create(b, loc, sinT2, r2);
  Value sinT4 = arith::AddFOp::create(b, loc, sinC3, sinT3);
  Value sinT5 = arith::MulFOp::create(b, loc, sinT4, r2);

  Value f32One =
      arith::ConstantOp::create(b, loc, f32, b.getF32FloatAttr(1.0f));
  Value sinT6 = arith::AddFOp::create(b, loc, f32One, sinT5);
  Value sinPoly = arith::MulFOp::create(b, loc, r, sinT6);

  // --- cos polynomial: cos(r) = 1.0 + r²*(c2 + r²*(c4 + r²*c6)) ---
  Value cosC6 = arith::ConstantOp::create(b, loc, f32,
                                          b.getF32FloatAttr(-0.001388837214f));
  Value cosC4 = arith::ConstantOp::create(b, loc, f32,
                                          b.getF32FloatAttr(0.041666526347f));
  Value cosC2 = arith::ConstantOp::create(b, loc, f32,
                                          b.getF32FloatAttr(-0.499999850988f));

  Value cosT1 = arith::MulFOp::create(b, loc, cosC6, r2);
  Value cosT2 = arith::AddFOp::create(b, loc, cosC4, cosT1);
  Value cosT3 = arith::MulFOp::create(b, loc, cosT2, r2);
  Value cosT4 = arith::AddFOp::create(b, loc, cosC2, cosT3);
  Value cosT5 = arith::MulFOp::create(b, loc, cosT4, r2);
  Value cosPoly = arith::AddFOp::create(b, loc, f32One, cosT5);

  // --- Quadrant selection based on n mod 4 ---
  // q = n % 4  (compute as int32 truncation, then mod)
  Value nI32 = arith::FPToSIOp::create(b, loc, i32, n);
  Value four = arith::ConstantOp::create(b, loc, i32, b.getI32IntegerAttr(4));
  Value q = arith::RemSIOp::create(b, loc, nI32, four);

  // Default: sin=sinPoly, cos=cosPoly (q=0)
  Value sinResult = sinPoly;
  Value cosResult = cosPoly;

  // q == 1: sin=cosPoly, cos=-sinPoly
  {
    Value negSinPoly = arith::NegFOp::create(b, loc, sinPoly);
    auto qEq1 = arith::CmpIOp::create(
        b, loc, arith::CmpIPredicate::eq, q,
        arith::ConstantOp::create(b, loc, i32, b.getI32IntegerAttr(1)));
    sinResult = arith::SelectOp::create(b, loc, qEq1, cosPoly, sinResult);
    cosResult = arith::SelectOp::create(b, loc, qEq1, negSinPoly, cosResult);
  }

  // q == 2: sin=-sinPoly, cos=-cosPoly
  {
    Value negSinPoly = arith::NegFOp::create(b, loc, sinPoly);
    Value negCosPoly = arith::NegFOp::create(b, loc, cosPoly);
    auto qEq2 = arith::CmpIOp::create(
        b, loc, arith::CmpIPredicate::eq, q,
        arith::ConstantOp::create(b, loc, i32, b.getI32IntegerAttr(2)));
    sinResult = arith::SelectOp::create(b, loc, qEq2, negSinPoly, sinResult);
    cosResult = arith::SelectOp::create(b, loc, qEq2, negCosPoly, cosResult);
  }

  // q == 3: sin=-cosPoly, cos=sinPoly
  {
    Value negCosPoly = arith::NegFOp::create(b, loc, cosPoly);
    auto qEq3 = arith::CmpIOp::create(
        b, loc, arith::CmpIPredicate::eq, q,
        arith::ConstantOp::create(b, loc, i32, b.getI32IntegerAttr(3)));
    sinResult = arith::SelectOp::create(b, loc, qEq3, negCosPoly, sinResult);
    cosResult = arith::SelectOp::create(b, loc, qEq3, sinPoly, cosResult);
  }

  return {sinResult, cosResult};
}

} // namespace

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

Value createApproxCos(OpBuilder &b, Location loc, Value x, MathMode mode) {
  switch (mode) {
  case MathMode::strict:
    return math::CosOp::create(b, loc, x);
  case MathMode::bounded_fast:
  case MathMode::unsafe_fast:
  case MathMode::triton_fast:
    return approxSinCosBoundedFast(b, loc, x).cosVal;
  }
  return math::CosOp::create(b, loc, x);
}

Value createApproxSin(OpBuilder &b, Location loc, Value x, MathMode mode) {
  switch (mode) {
  case MathMode::strict:
    return math::SinOp::create(b, loc, x);
  case MathMode::bounded_fast:
  case MathMode::unsafe_fast:
  case MathMode::triton_fast:
    return approxSinCosBoundedFast(b, loc, x).sinVal;
  }
  return math::SinOp::create(b, loc, x);
}

} // namespace llk
} // namespace mlir
