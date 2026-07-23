//===- MathApproximation.cpp - Polynomial function approximations
//----------===//
//
// Fast polynomial approximations for exp, sigmoid, cos, sin.
//
// exp(x):  range-reduce x * log2(e) → integer n + fraction r ∈ [-0.5, 0.5],
//          then 2^n * (1 + P(r)) where P is a Taylor polynomial for 2^r - 1.
// sigmoid: implemented via createApproxExp.
// cos/sin:  for bounded_fast mode, defer to math::CosOp / math::SinOp with
//           an optional range-reduction wrapper (placeholder for future
//           polynomial implementations).
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

Value createApproxCos(OpBuilder &b, Location loc, Value x, MathMode mode) {
  switch (mode) {
  case MathMode::strict:
    return math::CosOp::create(b, loc, x);
  case MathMode::bounded_fast:
  case MathMode::unsafe_fast:
    // Use exact math for now; polynomial cos/sin approximation can be
    // added here in a future enhancement (range reduction + minimax poly).
    return math::CosOp::create(b, loc, x);
  }
  return math::CosOp::create(b, loc, x);
}

Value createApproxSin(OpBuilder &b, Location loc, Value x, MathMode mode) {
  switch (mode) {
  case MathMode::strict:
    return math::SinOp::create(b, loc, x);
  case MathMode::bounded_fast:
  case MathMode::unsafe_fast:
    return math::SinOp::create(b, loc, x);
  }
  return math::SinOp::create(b, loc, x);
}

} // namespace llk
} // namespace mlir
