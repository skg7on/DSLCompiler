//===- ShapeEvaluation.cpp - Evaluate checked shape index arithmetic ------===//
//
// Part of the LLK Compiler Project, under the Apache License v2.0 with
// LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "LLK/Mapping/ShapeEvaluation.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Operation.h"

#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/Support/Error.h"

#include <algorithm>
#include <optional>

namespace mlir::llk::mapping {

namespace {

llvm::Error shapeError(const llvm::Twine &message) {
  return llvm::make_error<llvm::StringError>(
      "shape_evaluation: " + message.str(), llvm::inconvertibleErrorCode());
}

llvm::Expected<int64_t> evaluate(mlir::Value value,
                                 const ShapeEvaluationContext &context,
                                 unsigned depth);

llvm::Expected<int64_t> known(int64_t value, llvm::StringRef what) {
  if (value < 0)
    return shapeError(what + " evaluated to the negative index " +
                      llvm::Twine(value) + "; the shape sample is invalid");
  return value;
}

template <typename Op>
llvm::Expected<int64_t>
evaluateBinary(Op op, llvm::StringRef what,
               const ShapeEvaluationContext &context, unsigned depth,
               llvm::function_ref<int64_t(int64_t, int64_t)> operation) {
  llvm::Expected<int64_t> lhs = evaluate(op.getLhs(), context, depth + 1);
  if (!lhs)
    return lhs.takeError();
  llvm::Expected<int64_t> rhs = evaluate(op.getRhs(), context, depth + 1);
  if (!rhs)
    return rhs.takeError();
  return known(operation(*lhs, *rhs), what);
}

llvm::Expected<int64_t> evaluate(mlir::Value value,
                                 const ShapeEvaluationContext &context,
                                 unsigned depth) {
  if (depth > 64)
    return shapeError("the shape expression nests too deeply");
  if (!value)
    return shapeError("a null index value cannot be evaluated");

  if (std::optional<int64_t> bound = context.lookup(value))
    return known(*bound, "a bound index value");
  if (std::optional<int64_t> constant = mlir::getConstantIntValue(value))
    return known(*constant, "an index constant");

  mlir::Operation *defining = value.getDefiningOp();
  if (!defining)
    return shapeError("an unbound block argument is not a shape fact");

  if (auto add = mlir::dyn_cast<arith::AddIOp>(defining))
    return evaluateBinary(add, "arith.addi", context, depth,
                          [](int64_t a, int64_t b) {
                            int64_t result = 0;
                            if (__builtin_add_overflow(a, b, &result))
                              return int64_t{-1};
                            return result;
                          });
  if (auto sub = mlir::dyn_cast<arith::SubIOp>(defining))
    return evaluateBinary(sub, "arith.subi", context, depth,
                          [](int64_t a, int64_t b) { return a - b; });
  if (auto mul = mlir::dyn_cast<arith::MulIOp>(defining))
    return evaluateBinary(mul, "arith.muli", context, depth,
                          [](int64_t a, int64_t b) {
                            int64_t result = 0;
                            if (__builtin_mul_overflow(a, b, &result))
                              return int64_t{-1};
                            return result;
                          });
  if (auto div = mlir::dyn_cast<arith::DivUIOp>(defining)) {
    if (std::optional<int64_t> rhs = context.lookup(div.getRhs());
        rhs && *rhs == 0)
      return shapeError("arith.divui divides by zero");
    return evaluateBinary(
        div, "arith.divui", context, depth,
        [](int64_t a, int64_t b) { return b == 0 ? int64_t{-1} : a / b; });
  }
  if (auto rem = mlir::dyn_cast<arith::RemUIOp>(defining))
    return evaluateBinary(
        rem, "arith.remui", context, depth,
        [](int64_t a, int64_t b) { return b == 0 ? int64_t{-1} : a % b; });
  if (auto ceil = mlir::dyn_cast<arith::CeilDivUIOp>(defining))
    return evaluateBinary(ceil, "arith.ceildivui", context, depth,
                          [](int64_t a, int64_t b) {
                            if (b == 0)
                              return int64_t{-1};
                            if (a == 0)
                              return int64_t{0};
                            int64_t sum = 0;
                            if (__builtin_add_overflow(a, b - 1, &sum))
                              return int64_t{-1};
                            return sum / b;
                          });
  if (auto min = mlir::dyn_cast<arith::MinSIOp>(defining))
    return evaluateBinary(min, "arith.minsi", context, depth,
                          [](int64_t a, int64_t b) { return std::min(a, b); });
  if (auto max = mlir::dyn_cast<arith::MaxSIOp>(defining))
    return evaluateBinary(max, "arith.maxsi", context, depth,
                          [](int64_t a, int64_t b) { return std::max(a, b); });
  if (auto min = mlir::dyn_cast<arith::MinUIOp>(defining))
    return evaluateBinary(min, "arith.minui", context, depth,
                          [](int64_t a, int64_t b) { return std::min(a, b); });
  if (auto max = mlir::dyn_cast<arith::MaxUIOp>(defining))
    return evaluateBinary(max, "arith.maxui", context, depth,
                          [](int64_t a, int64_t b) { return std::max(a, b); });

  if (auto compare = mlir::dyn_cast<arith::CmpIOp>(defining)) {
    llvm::Expected<int64_t> lhs =
        evaluate(compare.getLhs(), context, depth + 1);
    if (!lhs)
      return lhs.takeError();
    llvm::Expected<int64_t> rhs =
        evaluate(compare.getRhs(), context, depth + 1);
    if (!rhs)
      return rhs.takeError();
    bool result = false;
    switch (compare.getPredicate()) {
    case arith::CmpIPredicate::eq:
      result = *lhs == *rhs;
      break;
    case arith::CmpIPredicate::ne:
      result = *lhs != *rhs;
      break;
    case arith::CmpIPredicate::slt:
    case arith::CmpIPredicate::ult:
      result = *lhs < *rhs;
      break;
    case arith::CmpIPredicate::sle:
    case arith::CmpIPredicate::ule:
      result = *lhs <= *rhs;
      break;
    case arith::CmpIPredicate::sgt:
    case arith::CmpIPredicate::ugt:
      result = *lhs > *rhs;
      break;
    case arith::CmpIPredicate::sge:
    case arith::CmpIPredicate::uge:
      result = *lhs >= *rhs;
      break;
    }
    return int64_t{result ? 1 : 0};
  }

  if (auto select = mlir::dyn_cast<arith::SelectOp>(defining)) {
    llvm::Expected<int64_t> condition =
        evaluate(select.getCondition(), context, depth + 1);
    if (!condition)
      return condition.takeError();
    if (*condition != 0 && *condition != 1)
      return shapeError("arith.select condition is not a boolean fact");
    return evaluate(*condition ? select.getTrueValue() : select.getFalseValue(),
                    context, depth + 1);
  }

  if (auto cast = mlir::dyn_cast<arith::IndexCastOp>(defining))
    return evaluate(cast.getIn(), context, depth + 1);
  if (auto cast = mlir::dyn_cast<arith::IndexCastUIOp>(defining))
    return evaluate(cast.getIn(), context, depth + 1);
  if (auto extend = mlir::dyn_cast<arith::ExtSIOp>(defining))
    return evaluate(extend.getIn(), context, depth + 1);
  if (auto extend = mlir::dyn_cast<arith::ExtUIOp>(defining))
    return evaluate(extend.getIn(), context, depth + 1);
  if (auto truncate = mlir::dyn_cast<arith::TruncIOp>(defining))
    return evaluate(truncate.getIn(), context, depth + 1);

  if (mlir::isa<arith::DivSIOp>(defining))
    return shapeError("arith.divsi is not part of the supported shape "
                      "arithmetic; shape lowering emits unsigned division");
  if (mlir::isa<tensor::DimOp>(defining) || mlir::isa<memref::DimOp>(defining))
    return shapeError("a dimension query is not bound to a shape sample");

  return shapeError("unsupported shape expression '" +
                    defining->getName().getStringRef() +
                    "'; bind its result or use checked shape arithmetic");
}

} // namespace

llvm::Expected<int64_t>
evaluateShapeIndex(mlir::Value value, const ShapeEvaluationContext &context) {
  return evaluate(value, context, 0);
}

} // namespace mlir::llk::mapping
