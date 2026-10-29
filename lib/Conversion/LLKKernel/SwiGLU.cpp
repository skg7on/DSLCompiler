//===- SwiGLU.cpp - LLK fused SwiGLU operator lowering --------------------===//
//
// Part of the LLK Compiler Project, under the Apache License v2.0 with
// LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "LLK/Dialect/LLKDialect.h" // IWYU pragma: keep
#include "LLK/Dialect/LLKEnums.h"
#include "LLK/Transforms/Common/MathApproximation.h"

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
LogicalResult lowerSwiGLU(OpBuilder &builder, mlir::llk::FusedSwiGLUOp op) {
  Location loc = op.getLoc();

  Value x = op.getX();
  Value wg = op.getWg();
  Value wu = op.getWu();
  Value init = op.getInit();

  auto xType = mlir::cast<ShapedType>(x.getType());
  auto wgType = mlir::cast<ShapedType>(wg.getType());
  int64_t M = xType.getDimSize(0);
  int64_t N = wgType.getDimSize(1);

  llk::MathMode mode = op.getMathMode();

  auto bf16Type = builder.getBF16Type();
  auto f32Type = builder.getF32Type();
  auto gateUpType = RankedTensorType::get({M, N}, bf16Type);

  // A dynamic extent is never fabricated: the conversion tensors are sized
  // from the source's own runtime dimensions, so the arithmetic and
  // math-mode semantics below are the same for a known and an unknown M or N.
  SmallVector<Value> dynamicSizes;
  if (ShapedType::isDynamic(M))
    dynamicSizes.push_back(tensor::DimOp::create(builder, loc, x, /*index=*/0));
  if (ShapedType::isDynamic(N))
    dynamicSizes.push_back(
        tensor::DimOp::create(builder, loc, wg, /*index=*/1));

  Value bf16Zero = arith::ConstantOp::create(
      builder, loc, bf16Type, builder.getFloatAttr(bf16Type, 0.0));

  // Gate projection: X @ Wg
  Value gateInit =
      tensor::EmptyOp::create(builder, loc, gateUpType, dynamicSizes);
  Value gateFilled = linalg::FillOp::create(builder, loc, ValueRange{bf16Zero},
                                            ValueRange{gateInit})
                         .getResult(0);
  Value gateResult = linalg::MatmulOp::create(builder, loc, ValueRange{x, wg},
                                              ValueRange{gateFilled})
                         .getResult(0);

  // Up projection: X @ Wu
  Value upInit =
      tensor::EmptyOp::create(builder, loc, gateUpType, dynamicSizes);
  Value upFilled = linalg::FillOp::create(builder, loc, ValueRange{bf16Zero},
                                          ValueRange{upInit})
                       .getResult(0);
  Value upResult = linalg::MatmulOp::create(builder, loc, ValueRange{x, wu},
                                            ValueRange{upFilled})
                       .getResult(0);

  // Elementwise: SiLU(gate) * up, f32 internals, truncate to bf16
  AffineMap idMap = AffineMap::getMultiDimIdentityMap(2, builder.getContext());

  Value siluResult =
      linalg::GenericOp::create(
          builder, loc,
          /*resultTensorTypes=*/TypeRange{gateUpType},
          /*inputs=*/ValueRange{gateResult, upResult},
          /*outputs=*/ValueRange{init},
          /*indexingMaps=*/ArrayRef<AffineMap>{idMap, idMap, idMap},
          /*iteratorTypes=*/
          ArrayRef<utils::IteratorType>{utils::IteratorType::parallel,
                                        utils::IteratorType::parallel},
          /*bodyBuilder=*/
          [&](OpBuilder &b, Location l, ValueRange args) {
            Value g = args[0]; // bf16
            Value u = args[1]; // bf16

            Value gF32 = arith::ExtFOp::create(b, l, f32Type, g);
            Value uF32 = arith::ExtFOp::create(b, l, f32Type, u);

            // SiLU(g) = g * sigmoid(g) = g / (1 + exp(-g))
            Value neg = arith::NegFOp::create(b, l, gF32);
            Value exp = llk::createApproxExp(b, l, neg, mode);
            Value f32One = arith::ConstantOp::create(b, l, f32Type,
                                                     b.getF32FloatAttr(1.0));
            Value den = arith::AddFOp::create(b, l, f32One, exp);
            Value invDen = arith::DivFOp::create(b, l, f32One, den);
            Value silu = arith::MulFOp::create(b, l, gF32, invDen);
            Value r = arith::MulFOp::create(b, l, silu, uF32);

            Value trunc = arith::TruncFOp::create(b, l, bf16Type, r);
            linalg::YieldOp::create(b, l, trunc);
          })
          .getResult(0);

  op.replaceAllUsesWith(siluResult);
  op.erase();
  return success();
}

class SwiGLULowering final : public llk::LLKKernelLowering {
public:
  StringRef getOpName() const override { return "llk.fused_swiglu"; }
  LogicalResult inferContract(Operation *root,
                              llk::KernelContract &c) const override {
    auto op = cast<llk::FusedSwiGLUOp>(root);
    c.workload = "fused_swiglu";
    return llk::inferContractionContract(root, op.getX(), op.getWg(),
                                         op.getResult(),
                                         op.getAccumulatorType(), c);
  }
  FailureOr<llk::ContractionDimensions>
  inferDimensions(Operation *root) const override {
    auto op = cast<llk::FusedSwiGLUOp>(root);
    auto dims = llk::inferContractionDimensions(root, op.getX(), op.getWg(),
                                                op.getResult());
    if (failed(dims))
      return failure();
    // The up-projection weight and the initialized destination each pin the
    // same axes; unify them so a mixed static/dynamic root resolves to the
    // constants the source already guarantees.
    auto resolve = [&](llvm::StringRef axis,
                       const llk::shape::Dimension &current,
                       llvm::ArrayRef<int64_t> extents)
        -> FailureOr<llk::shape::Dimension> {
      llvm::SmallVector<int64_t, 2> combined = {
          current.isConstant() ? current.getConstant() : ShapedType::kDynamic};
      combined.append(extents.begin(), extents.end());
      return llk::resolveContractionDimension(root, axis, combined);
    };
    auto init = cast<ShapedType>(op.getInit().getType());
    auto wu = cast<ShapedType>(op.getWu().getType());
    FailureOr<llk::shape::Dimension> M =
        resolve("M", dims->M, {init.getDimSize(0)});
    if (failed(M))
      return failure();
    FailureOr<llk::shape::Dimension> N =
        resolve("N", dims->N, {wu.getDimSize(1), init.getDimSize(1)});
    if (failed(N))
      return failure();
    FailureOr<llk::shape::Dimension> K =
        resolve("K", dims->K, {wu.getDimSize(0)});
    if (failed(K))
      return failure();
    return llk::ContractionDimensions{*M, *N, *K};
  }
  bool hasMicroLowering() const override { return true; }
  LogicalResult describeMicroKernel(Operation *root,
                                    const llk::KernelContract &contract,
                                    llk::MicroKernelFragment &f,
                                    bool requireStatic = true) const override {
    auto op = cast<llk::FusedSwiGLUOp>(root);
    auto validated = contract;
    if (requireStatic) {
      if (failed(llk::validateMicroContraction(
              root, op.getX(), op.getWg(), op.getResult(),
              op.getAccumulatorType(), validated)))
        return failure();
      if (failed(
              llk::checkMicroTensor(root, op.getWu(), "up-projection weight")))
        return failure();
      auto wu = cast<ShapedType>(op.getWu().getType());
      if (wu.getDimSize(0) != contract.K || wu.getDimSize(1) != contract.N)
        return root->emitError()
               << "the up-projection weight must be shaped [K, N] = ["
               << contract.K << ", " << contract.N << "]";
    } else {
      // `validateMicroContractionShape` covers X/Wg/result; `inferDimensions`
      // additionally unifies Wu, which the shared validation cannot see.
      llk::ContractionDimensions dimensions;
      if (failed(llk::validateMicroContractionShape(
              root, op.getX(), op.getWg(), op.getResult(),
              op.getAccumulatorType(), dimensions, validated)))
        return failure();
      if (failed(inferDimensions(root)))
        return failure();
    }
    auto wu = cast<ShapedType>(op.getWu().getType());
    if (wu.getElementType() != validated.inputElemType)
      return root->emitError("the concrete Micro-IR exporter requires matching "
                             "input and up-projection weight element types");
    if (op.getActivation() != llk::Activation::silu)
      return root->emitError(
          "unsupported SwiGLU activation; the Micro-IR export emits silu only");
    f.armCount = 2;
    f.hasOutputInit = false;
    f.lhsOperandType = op.getX().getType();
    f.rhsOperandTypes = {op.getWg().getType(), op.getWu().getType()};
    // The SwiGLU destination is the output port, not a read input: its
    // incoming values never feed the computation.
    f.destinationOperandType = op.getInit().getType();
    auto mode = op.getMathMode();
    f.buildEpilogue = [mode](OpBuilder &b, Location loc,
                             ArrayRef<Value> acc) -> Value {
      Value gate = micro::VectorOp::create(
          b, loc, acc[0].getType(), "silu", ValueRange{acc[0]},
          b.getStringAttr(llk::stringifyMathMode(mode)));
      return micro::VectorOp::create(b, loc, acc[0].getType(), "mul",
                                     ValueRange{gate, acc[1]}, StringAttr());
    };
    return success();
  }
  bool hasLinalgLowering() const override { return true; }
  LogicalResult lowerToLinalg(Operation *root, OpBuilder &b) const override {
    return lowerSwiGLU(b, cast<llk::FusedSwiGLUOp>(root));
  }
};
} // namespace
namespace mlir::llk {
void registerSwiGLULowering(LLKKernelRegistry &r) { r.add<SwiGLULowering>(); }
} // namespace mlir::llk
