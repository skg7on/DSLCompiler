//===- MathApproximation.h - Polynomial function approximations --*- C++
//-*-===//
//
// Polynomial approximations for nonlinear functions used by all kernels:
//   exp, sigmoid, cos, sin
// Each function takes (builder, loc, Value x, MathMode mode) and returns
// a Value with the approximated result.
//
//===----------------------------------------------------------------------===//

#ifndef LLK_TRANSFORMS_COMMON_MATHAPPROXIMATION_H
#define LLK_TRANSFORMS_COMMON_MATHAPPROXIMATION_H

#include "LLK/Dialect/LLKEnums.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Value.h"

namespace mlir {
namespace llk {

/// Create an approximation of exp(x) according to the math mode.
/// - strict:      use math::ExpOp (IEEE-compliant)
/// - bounded_fast: range-reduced polynomial with documented ULP bound
/// - unsafe_fast:  same as bounded_fast for now
Value createApproxExp(OpBuilder &b, Location loc, Value x, MathMode mode);

/// Create an approximation of sigmoid(x) = 1 / (1 + exp(-x)).
Value createApproxSigmoid(OpBuilder &b, Location loc, Value x, MathMode mode);

/// Create an approximation of cos(x) via range-reduced polynomial.
Value createApproxCos(OpBuilder &b, Location loc, Value x, MathMode mode);

/// Create an approximation of sin(x) via range-reduced polynomial.
Value createApproxSin(OpBuilder &b, Location loc, Value x, MathMode mode);

} // namespace llk
} // namespace mlir

#endif // LLK_TRANSFORMS_COMMON_MATHAPPROXIMATION_H
