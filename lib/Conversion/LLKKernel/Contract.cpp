//===- Contract.cpp - Contraction contract inference and checks -----------===//
//
// Part of the LLK Compiler Project, under the Apache License v2.0 with
// LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "LLK/Conversion/LLKKernel/LLKKernelLowering.h"
#include "LLK/Dialect/Micro/MicroHelpers.h"
#include "LLK/Transforms/Common/ScheduleLoader.h"
#include "mlir/IR/BuiltinTypes.h"

#include <optional>

namespace mlir::llk {
namespace {

/// Rank and element-type checks shared by the static and symbolic paths. MLIR's
/// dynamic sentinel is not a dtype and never reaches shape arithmetic.
LogicalResult checkTensorKind(Operation *root, Value value, const char *what) {
  auto shaped = dyn_cast<ShapedType>(value.getType());
  if (!shaped || !shaped.hasRank() || shaped.getRank() != 2)
    return root->emitError() << what << " must be a rank-2 tensor";
  if (!micro::dtypeOfElementType(shaped.getElementType()))
    return root->emitError()
           << what << " element type is not a Micro tile dtype "
           << "(expected f32, f16, bf16, i32, or i8)";
  return success();
}

LogicalResult checkAccumulatorType(Operation *root, Attribute accumulator,
                                   TypeAttr &accumulatorTypeAttr) {
  if (accumulator)
    accumulatorTypeAttr = dyn_cast<TypeAttr>(accumulator);
  if (!accumulatorTypeAttr ||
      !micro::dtypeOfElementType(accumulatorTypeAttr.getValue()))
    return root->emitError()
           << "accumulator_type must be a type attribute naming a Micro tile "
              "dtype";
  return success();
}

} // namespace

LogicalResult checkMicroTensor(Operation *root, Value value, const char *what) {
  if (failed(checkTensorKind(root, value, what)))
    return failure();
  auto shaped = dyn_cast<ShapedType>(value.getType());
  if (!shaped.hasStaticShape())
    return root->emitError()
           << what
           << " must have a static shape; dynamic shapes are not supported "
              "by the Micro-IR export";
  return success();
}

FailureOr<shape::Dimension>
resolveContractionDimension(Operation *root, llvm::StringRef axis,
                            llvm::ArrayRef<int64_t> extents) {
  std::optional<int64_t> resolved;
  for (int64_t extent : extents) {
    if (ShapedType::isDynamic(extent))
      continue;
    if (resolved && *resolved != extent)
      return root->emitError()
             << "contraction dimension mismatch: " << axis << " is "
             << *resolved << " in one operand and " << extent << " in another";
    resolved = extent;
  }
  if (resolved)
    return shape::Dimension(*resolved);
  // No operand pins the axis, so the relationship itself becomes a symbol.
  return shape::Dimension::symbol(axis.str());
}

FailureOr<ContractionDimensions>
inferContractionDimensions(Operation *root, Value lhs, Value rhs, Value out) {
  auto l = dyn_cast<RankedTensorType>(lhs.getType());
  auto r = dyn_cast<RankedTensorType>(rhs.getType());
  auto o = dyn_cast<RankedTensorType>(out.getType());
  if (!l || !r || !o || l.getRank() != 2 || r.getRank() != 2 ||
      o.getRank() != 2)
    return root->emitError("contraction requires rank-2 tensors");

  FailureOr<shape::Dimension> M = resolveContractionDimension(
      root, "M", {l.getDimSize(0), o.getDimSize(0)});
  if (failed(M))
    return failure();
  FailureOr<shape::Dimension> N = resolveContractionDimension(
      root, "N", {r.getDimSize(1), o.getDimSize(1)});
  if (failed(N))
    return failure();
  FailureOr<shape::Dimension> K = resolveContractionDimension(
      root, "K", {l.getDimSize(1), r.getDimSize(0)});
  if (failed(K))
    return failure();
  return ContractionDimensions{*M, *N, *K};
}

LogicalResult inferContractionContract(Operation *root, Value lhs, Value rhs,
                                       Value out,
                                       Attribute accumulatorAttribute,
                                       KernelContract &info) {
  auto l = dyn_cast<RankedTensorType>(lhs.getType());
  auto r = dyn_cast<RankedTensorType>(rhs.getType());
  auto o = dyn_cast<RankedTensorType>(out.getType());
  if (!l || !r || !o || l.getRank() != 2 || r.getRank() != 2 ||
      o.getRank() != 2)
    return root->emitError("contraction requires rank-2 tensors");
  info.M = l.getDimSize(0);
  info.K = l.getDimSize(1);
  info.N = r.getDimSize(1);
  info.inputElemType = l.getElementType();
  info.outputElemType = o.getElementType();
  auto acc = dyn_cast_or_null<TypeAttr>(accumulatorAttribute);
  info.accumulatorElemType = acc ? acc.getValue() : Type();
  info.mBucket = l.hasStaticShape() ? classifyM(info.M) : -1;
  return success();
}

LogicalResult validateMicroContractionShape(Operation *root, Value lhs,
                                            Value rhs, Value out,
                                            Attribute accumulatorAttribute,
                                            ContractionDimensions &dims,
                                            KernelContract &info) {
  if (failed(checkTensorKind(root, lhs, "left-hand side")) ||
      failed(checkTensorKind(root, rhs, "right-hand side")) ||
      failed(checkTensorKind(root, out, "result")))
    return failure();

  auto dimsOr = inferContractionDimensions(root, lhs, rhs, out);
  if (failed(dimsOr))
    return failure();

  auto lhsShaped = cast<RankedTensorType>(lhs.getType());
  auto rhsShaped = cast<RankedTensorType>(rhs.getType());
  auto outShaped = cast<RankedTensorType>(out.getType());
  if (rhsShaped.getElementType() != lhsShaped.getElementType())
    return root->emitError()
           << "the concrete Micro-IR exporter requires matching input and "
              "weight element types";

  TypeAttr accumulatorTypeAttr;
  if (failed(checkAccumulatorType(root, accumulatorAttribute,
                                  accumulatorTypeAttr)))
    return failure();

  dims = *dimsOr;
  // The legacy integer fields stay unset until every axis resolves statically;
  // no consumer may perform shape arithmetic on a symbol.
  info.M = dims.M.isConstant() ? dims.M.getConstant() : ShapedType::kDynamic;
  info.N = dims.N.isConstant() ? dims.N.getConstant() : ShapedType::kDynamic;
  info.K = dims.K.isConstant() ? dims.K.getConstant() : ShapedType::kDynamic;
  info.inputElemType = lhsShaped.getElementType();
  info.accumulatorElemType = accumulatorTypeAttr.getValue();
  info.outputElemType = outShaped.getElementType();
  info.mBucket = dims.M.isConstant() ? classifyM(info.M) : -1;
  return success();
}

LogicalResult validateMicroContraction(Operation *root, Value lhs, Value rhs,
                                       Value out,
                                       Attribute accumulatorAttribute,
                                       KernelContract &info) {
  if (failed(checkMicroTensor(root, lhs, "left-hand side")) ||
      failed(checkMicroTensor(root, rhs, "right-hand side")) ||
      failed(checkMicroTensor(root, out, "result")))
    return failure();
  auto lhsShaped = cast<ShapedType>(lhs.getType());
  auto rhsShaped = cast<ShapedType>(rhs.getType());
  auto outShaped = cast<ShapedType>(out.getType());
  info.M = lhsShaped.getDimSize(0);
  info.K = lhsShaped.getDimSize(1);
  info.N = rhsShaped.getDimSize(1);

  if (rhsShaped.getDimSize(0) != info.K)
    return root->emitError()
           << "contraction dimension mismatch: the left-hand side contracts "
              "over "
           << info.K << " but the right-hand side has "
           << rhsShaped.getDimSize(0) << " rows";

  if (rhsShaped.getElementType() != lhsShaped.getElementType())
    return root->emitError()
           << "the concrete Micro-IR exporter requires matching input and "
              "weight element types";

  if (outShaped.getDimSize(0) != info.M || outShaped.getDimSize(1) != info.N)
    return root->emitError()
           << "result shape does not match the contraction: expected M = "
           << info.M << " and N = " << info.N;

  TypeAttr accumulatorTypeAttr;
  if (failed(checkAccumulatorType(root, accumulatorAttribute,
                                  accumulatorTypeAttr)))
    return failure();

  info.inputElemType = lhsShaped.getElementType();
  info.accumulatorElemType = accumulatorTypeAttr.getValue();
  info.outputElemType = outShaped.getElementType();
  info.mBucket = llk::classifyM(info.M);
  return success();
}

} // namespace mlir::llk
