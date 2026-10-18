//===- Matmul.cpp - LLK matmul operator lowering --------------------------===//
//
// Part of the LLK Compiler Project, under the Apache License v2.0 with
// LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "LLK/Dialect/LLKDialect.h"                  // IWYU pragma: keep
#include "LLK/Dialect/LLKEnums.h"                    // IWYU pragma: keep
#include "LLK/Transforms/Common/MathApproximation.h" // IWYU pragma: keep

#include "mlir/Bytecode/BytecodeOpInterface.h" // IWYU pragma: keep
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h" // IWYU pragma: keep
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h" // IWYU pragma: keep
#include "mlir/Interfaces/TilingInterface.h"             // IWYU pragma: keep
#include "mlir/Pass/Pass.h"                              // IWYU pragma: keep
#include "mlir/Support/LLVM.h"

#include <cmath>  // IWYU pragma: keep
#include <limits> // IWYU pragma: keep

// Generated attribute and op class declarations.
#define GET_ATTRDEF_CLASSES
#include "LLK/Dialect/LLKAttributes.h.inc"
#define GET_OP_CLASSES
#include "LLK/Dialect/LLKOps.h.inc"

using namespace mlir;

#include "LLK/Conversion/LLKKernel/LLKKernelLowering.h"
#include "LLK/Dialect/Micro/MicroDialect.h" // IWYU pragma: keep
#include "LLK/Dialect/Micro/MicroEnums.h"   // IWYU pragma: keep
#include "mlir/IR/BuiltinTypes.h"
#define GET_ATTRDEF_CLASSES
#include "LLK/Dialect/Micro/MicroAttributes.h.inc"
#define GET_TYPEDEF_CLASSES
#include "LLK/Dialect/Micro/MicroTypes.h.inc"
#define GET_OP_CLASSES
#include "LLK/Dialect/Micro/MicroOps.h.inc"

namespace {
class MatmulLowering final : public llk::LLKKernelLowering {
public:
  StringRef getOpName() const override { return "llk.matmul"; }
  LogicalResult inferContract(Operation *root,
                              llk::KernelContract &c) const override {
    c.workload = "matmul";
    return llk::inferContractionContract(
        root, root->getOperand(0), root->getOperand(1), root->getResult(0),
        root->getAttr("accumulator_type"), c);
  }
  bool hasLinalgLowering() const override { return true; }
  LogicalResult lowerToLinalg(Operation *root,
                              OpBuilder &builder) const override {
    auto op = cast<llk::MatmulOp>(root);
    auto accAttr = dyn_cast<TypeAttr>(op.getAccumulatorType());
    auto accType =
        accAttr ? dyn_cast<FloatType>(accAttr.getValue()) : FloatType();
    if (!accType)
      return op.emitError("accumulator_type must name a floating-point type "
                          "for Linalg lowering");
    auto initType = dyn_cast<RankedTensorType>(op.getInit().getType());
    auto outType = dyn_cast<RankedTensorType>(op.getResult().getType());
    if (!initType || !outType || initType.getRank() != 2 ||
        outType.getRank() != 2 || initType.getShape() != outType.getShape())
      return op.emitError(
          "initialized destination shape must match the rank-2 result");
    auto convertTensor = [&](Value source, FloatType target) -> Value {
      auto sourceType = cast<RankedTensorType>(source.getType());
      auto sourceElem = cast<FloatType>(sourceType.getElementType());
      if (sourceElem == target)
        return source;
      auto type = RankedTensorType::get(sourceType.getShape(), target,
                                        sourceType.getEncoding());
      SmallVector<Value> dynamicSizes;
      for (int64_t dim = 0; dim < sourceType.getRank(); ++dim)
        if (sourceType.isDynamicDim(dim))
          dynamicSizes.push_back(
              tensor::DimOp::create(builder, op.getLoc(), source, dim));
      Value empty =
          tensor::EmptyOp::create(builder, op.getLoc(), type, dynamicSizes);
      auto id = AffineMap::getMultiDimIdentityMap(2, builder.getContext());
      return linalg::GenericOp::create(
                 builder, op.getLoc(), TypeRange{type}, ValueRange{source},
                 ValueRange{empty}, ArrayRef<AffineMap>{id, id},
                 ArrayRef<utils::IteratorType>{utils::IteratorType::parallel,
                                               utils::IteratorType::parallel},
                 [target, sourceWidth = sourceElem.getWidth(),
                  targetWidth = target.getWidth()](OpBuilder &b, Location loc,
                                                   ValueRange args) {
                   Value value = args[0];
                   // f16 and bf16 have equal widths but different formats.
                   // Convert through f32 so neither extf nor truncf receives
                   // equal widths.
                   if (sourceWidth == targetWidth) {
                     value =
                         arith::ExtFOp::create(b, loc, b.getF32Type(), value);
                     value = arith::TruncFOp::create(b, loc, target, value);
                   } else if (sourceWidth < targetWidth)
                     value = arith::ExtFOp::create(b, loc, target, value);
                   else
                     value = arith::TruncFOp::create(b, loc, target, value);
                   linalg::YieldOp::create(b, loc, value);
                 })
          .getResult(0);
    };
    Value init = convertTensor(op.getInit(), accType);
    Value lhs = op.getA(), rhs = op.getB();
    // Named matmul handles ordinary extension/truncation. Equal-width
    // format changes need the explicit conversion used for the init tensor.
    for (Value *input : {&lhs, &rhs}) {
      auto elem =
          cast<FloatType>(cast<ShapedType>(input->getType()).getElementType());
      if (elem != accType && elem.getWidth() == accType.getWidth())
        *input = convertTensor(*input, accType);
    }
    Value result =
        linalg::MatmulOp::create(builder, op.getLoc(), ValueRange{lhs, rhs},
                                 ValueRange{init})
            .getResult(0);
    result = convertTensor(result, cast<FloatType>(outType.getElementType()));
    op.replaceAllUsesWith(result);
    op.erase();
    return success();
  }
  bool hasMicroLowering() const override { return true; }
  LogicalResult
  describeMicroKernel(Operation *root, const llk::KernelContract &c,
                      llk::MicroKernelFragment &f) const override {
    auto validated = c;
    if (failed(llk::validateMicroContraction(
            root, root->getOperand(0), root->getOperand(1), root->getResult(0),
            root->getAttr("accumulator_type"), validated)))
      return failure();
    f.armCount = 1;
    f.hasOutputInit = true;
    f.rhsOperandTypes = {root->getOperand(1).getType()};
    f.buildEpilogue = [](OpBuilder &, Location, ArrayRef<Value> acc) {
      return acc[0];
    };
    return success();
  }
};
} // namespace
namespace mlir::llk {
void registerMatmulLowering(LLKKernelRegistry &r) { r.add<MatmulLowering>(); }
} // namespace mlir::llk
