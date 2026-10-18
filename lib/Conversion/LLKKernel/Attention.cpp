//===- Attention.cpp - LLK attention operator lowering --------------------===//
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

#include <cmath> // IWYU pragma: keep
#include <limits>

// Generated attribute and op class declarations.
#define GET_ATTRDEF_CLASSES
#include "LLK/Dialect/LLKAttributes.h.inc"
#define GET_OP_CLASSES
#include "LLK/Dialect/LLKOps.h.inc"

using namespace mlir;

#include "LLK/Conversion/LLKKernel/LLKKernelLowering.h"
#include "LLK/Transforms/Common/ScheduleLoader.h"
namespace {
LogicalResult lowerAttention(OpBuilder &builder, mlir::llk::AttentionOp op) {
  Location loc = op.getLoc();
  Value q = op.getQ();
  Value k = op.getK();
  Value v = op.getV();
  Value init = op.getInit();
  float scale = static_cast<float>(op.getScale().convertToDouble());
  bool causal = op.getCausalMask();

  auto qType = mlir::cast<RankedTensorType>(q.getType());
  int64_t B = qType.getDimSize(0);
  int64_t H = qType.getDimSize(1);
  int64_t Lq = qType.getDimSize(2);
  int64_t D = qType.getDimSize(3);
  int64_t Lk = mlir::cast<RankedTensorType>(k.getType()).getDimSize(2);
  int64_t BH = B * H;

  auto f32Type = builder.getF32Type();

  llk::MathMode mode = op.getMathMode();

  // Collapse B,H into a single batch dimension: [B, H, ...] → [B*H, ...].
  SmallVector<ReassociationIndices> collapse3D = {{0, 1}, {2}, {3}};
  auto q3DType = RankedTensorType::get({BH, Lq, D}, f32Type);
  auto k3DType = RankedTensorType::get({BH, Lk, D}, f32Type);
  auto v3DType = RankedTensorType::get({BH, Lk, D}, f32Type);
  auto o3DType = RankedTensorType::get({BH, Lq, D}, f32Type);

  Value q3D =
      tensor::CollapseShapeOp::create(builder, loc, q3DType, q, collapse3D);
  Value k3D =
      tensor::CollapseShapeOp::create(builder, loc, k3DType, k, collapse3D);
  Value v3D =
      tensor::CollapseShapeOp::create(builder, loc, v3DType, v, collapse3D);
  Value init3D =
      tensor::CollapseShapeOp::create(builder, loc, o3DType, init, collapse3D);

  // ---- Transpose K: [BH, Lk, D] → [BH, D, Lk] using linalg.generic ----
  auto kTransposedType = RankedTensorType::get({BH, D, Lk}, f32Type);
  Value kTransposedInit =
      tensor::EmptyOp::create(builder, loc, kTransposedType, ValueRange{});
  AffineMap kMapIn = AffineMap::getMultiDimIdentityMap(3, builder.getContext());
  // (b, d, lk) reads from k3D at (b, lk, d)
  AffineMap kMapOut =
      AffineMap::get(3, 0,
                     {getAffineDimExpr(0, builder.getContext()),
                      getAffineDimExpr(2, builder.getContext()),
                      getAffineDimExpr(1, builder.getContext())},
                     builder.getContext());
  Value kTP =
      linalg::GenericOp::create(
          builder, loc, TypeRange{kTransposedType}, ValueRange{k3D},
          ValueRange{kTransposedInit}, ArrayRef<AffineMap>{kMapOut, kMapIn},
          ArrayRef<utils::IteratorType>{utils::IteratorType::parallel,
                                        utils::IteratorType::parallel,
                                        utils::IteratorType::parallel},
          [&](OpBuilder &b, Location l, ValueRange args) {
            linalg::YieldOp::create(b, l, args[0]);
          })
          .getResult(0);

  // ---- S = Q @ K^T: contraction [BH, Lq, D] × [BH, D, Lk] → [BH, Lq, Lk] ----
  auto sType = RankedTensorType::get({BH, Lq, Lk}, f32Type);
  Value sInit = tensor::EmptyOp::create(builder, loc, sType, ValueRange{});
  Value zeroVal = arith::ConstantOp::create(builder, loc, f32Type,
                                            builder.getF32FloatAttr(0.0f));
  Value sFilled = linalg::FillOp::create(builder, loc, ValueRange{zeroVal},
                                         ValueRange{sInit})
                      .getResult(0);

  // Batched matmul via linalg.generic: S[bh,q,k] += Q[bh,q,d] * K[bh,d,k]
  // 4D iteration space: (bh, q, d, k)
  AffineMap qMapBM = AffineMap::get(4, 0,
                                    {getAffineDimExpr(0, builder.getContext()),
                                     getAffineDimExpr(1, builder.getContext()),
                                     getAffineDimExpr(2, builder.getContext())},
                                    builder.getContext());
  AffineMap kMapBM = AffineMap::get(4, 0,
                                    {getAffineDimExpr(0, builder.getContext()),
                                     getAffineDimExpr(2, builder.getContext()),
                                     getAffineDimExpr(3, builder.getContext())},
                                    builder.getContext());
  AffineMap sMapBM = AffineMap::get(4, 0,
                                    {getAffineDimExpr(0, builder.getContext()),
                                     getAffineDimExpr(1, builder.getContext()),
                                     getAffineDimExpr(3, builder.getContext())},
                                    builder.getContext());

  Value scores =
      linalg::GenericOp::create(
          builder, loc, TypeRange{sType}, ValueRange{q3D, kTP},
          ValueRange{sFilled}, ArrayRef<AffineMap>{qMapBM, kMapBM, sMapBM},
          ArrayRef<utils::IteratorType>{
              utils::IteratorType::parallel, utils::IteratorType::parallel,
              utils::IteratorType::reduction, utils::IteratorType::parallel},
          [&](OpBuilder &b, Location l, ValueRange args) {
            Value mul = arith::MulFOp::create(b, l, args[0], args[1]);
            Value add = arith::AddFOp::create(b, l, mul, args[2]);
            linalg::YieldOp::create(b, l, ValueRange{add});
          })
          .getResult(0);

  // ---- Scale + causal mask over 3D tensor [BH, Lq, Lk] ----
  Value scaleVal = arith::ConstantOp::create(builder, loc, f32Type,
                                             builder.getF32FloatAttr(scale));
  Value negInfVal = arith::ConstantOp::create(
      builder, loc, f32Type,
      builder.getF32FloatAttr(-std::numeric_limits<float>::infinity()));

  Value scaledInit = tensor::EmptyOp::create(builder, loc, sType, ValueRange{});
  AffineMap idMap3D =
      AffineMap::getMultiDimIdentityMap(3, builder.getContext());
  Value scaled =
      linalg::GenericOp::create(
          builder, loc, TypeRange{sType}, ValueRange{scores},
          ValueRange{scaledInit}, ArrayRef<AffineMap>{idMap3D, idMap3D},
          ArrayRef<utils::IteratorType>{utils::IteratorType::parallel,
                                        utils::IteratorType::parallel,
                                        utils::IteratorType::parallel},
          [&](OpBuilder &b, Location l, ValueRange args) {
            Value s = args[0];
            Value scaledS = arith::MulFOp::create(b, l, s, scaleVal);
            if (causal) {
              Value qIdx = linalg::IndexOp::create(b, l, 1);
              Value kIdx = linalg::IndexOp::create(b, l, 2);
              Value kGtQ = arith::CmpIOp::create(
                  b, l, arith::CmpIPredicate::sgt, kIdx, qIdx);
              Value result =
                  arith::SelectOp::create(b, l, kGtQ, negInfVal, scaledS);
              linalg::YieldOp::create(b, l, ValueRange{result});
            } else {
              linalg::YieldOp::create(b, l, ValueRange{scaledS});
            }
          })
          .getResult(0);

  // ---- Stable softmax ----
  // Step a: row-wise max over Lk.  scaled: [BH, Lq, Lk] → maxVals: [BH, Lq]
  auto maxPerRowType = RankedTensorType::get({BH, Lq}, f32Type);
  Value maxInitRow =
      tensor::EmptyOp::create(builder, loc, maxPerRowType, ValueRange{});
  Value maxFilled = linalg::FillOp::create(builder, loc, ValueRange{negInfVal},
                                           ValueRange{maxInitRow})
                        .getResult(0);

  // 3D iteration space: (bh, q, k) with k reduction.
  // Input scaled: 3D → (bh, q, k); Output max: 2D → (bh, q)
  AffineMap reduce3DIn =
      AffineMap::get(3, 0,
                     {getAffineDimExpr(0, builder.getContext()),
                      getAffineDimExpr(1, builder.getContext()),
                      getAffineDimExpr(2, builder.getContext())},
                     builder.getContext());
  AffineMap reduceOut2D =
      AffineMap::get(3, 0,
                     {getAffineDimExpr(0, builder.getContext()),
                      getAffineDimExpr(1, builder.getContext())},
                     builder.getContext());

  Value maxVals =
      linalg::GenericOp::create(
          builder, loc, TypeRange{maxPerRowType}, ValueRange{scaled},
          ValueRange{maxFilled}, ArrayRef<AffineMap>{reduce3DIn, reduceOut2D},
          ArrayRef<utils::IteratorType>{utils::IteratorType::parallel,
                                        utils::IteratorType::parallel,
                                        utils::IteratorType::reduction},
          [&](OpBuilder &b, Location l, ValueRange args) {
            Value cur = args[0];
            Value acc = args[1];
            Value m = arith::MaximumFOp::create(b, l, cur, acc);
            linalg::YieldOp::create(b, l, ValueRange{m});
          })
          .getResult(0);

  // Step b: sum of exp(s - max) over Lk for denominator.
  Value sumInit =
      tensor::EmptyOp::create(builder, loc, maxPerRowType, ValueRange{});
  Value sumFilled = linalg::FillOp::create(builder, loc, ValueRange{zeroVal},
                                           ValueRange{sumInit})
                        .getResult(0);

  Value denominator =
      linalg::GenericOp::create(
          builder, loc, TypeRange{maxPerRowType}, ValueRange{scaled, maxVals},
          ValueRange{sumFilled},
          ArrayRef<AffineMap>{reduce3DIn, reduceOut2D, reduceOut2D},
          ArrayRef<utils::IteratorType>{utils::IteratorType::parallel,
                                        utils::IteratorType::parallel,
                                        utils::IteratorType::reduction},
          [&](OpBuilder &b, Location l, ValueRange args) {
            Value s = args[0];
            Value m = args[1];
            Value acc = args[2];
            Value shifted = arith::SubFOp::create(b, l, s, m);
            Value expVal = llk::createApproxExp(b, l, shifted, mode);
            Value sum = arith::AddFOp::create(b, l, expVal, acc);
            linalg::YieldOp::create(b, l, ValueRange{sum});
          })
          .getResult(0);

  // Step c: softmax weights = exp(s - max) / denominator.  3D elementwise.
  auto wType = RankedTensorType::get({BH, Lq, Lk}, f32Type);
  Value wInit = tensor::EmptyOp::create(builder, loc, wType, ValueRange{});
  // scaled: 3D (bh,q,k), maxVals: 2D (bh,q), denom: 2D (bh,q), w: 3D (bh,q,k)
  Value weights =
      linalg::GenericOp::create(
          builder, loc, TypeRange{wType},
          ValueRange{scaled, maxVals, denominator}, ValueRange{wInit},
          ArrayRef<AffineMap>{reduce3DIn, reduceOut2D, reduceOut2D, reduce3DIn},
          ArrayRef<utils::IteratorType>{utils::IteratorType::parallel,
                                        utils::IteratorType::parallel,
                                        utils::IteratorType::parallel},
          [&](OpBuilder &b, Location l, ValueRange args) {
            Value s = args[0];
            Value m = args[1];
            Value d = args[2];
            Value shifted = arith::SubFOp::create(b, l, s, m);
            Value expVal = llk::createApproxExp(b, l, shifted, mode);
            Value softmaxVal = arith::DivFOp::create(b, l, expVal, d);
            linalg::YieldOp::create(b, l, ValueRange{softmaxVal});
          })
          .getResult(0);

  // ---- O = softmax @ V: contraction [BH, Lq, Lk] × [BH, Lk, D] → [BH, Lq, D]
  // ---- 4D iteration space: (bh, q, k, d) O[bh,q,d] += W[bh,q,k] * V[bh,k,d]
  AffineMap wMapBM2 =
      AffineMap::get(4, 0,
                     {getAffineDimExpr(0, builder.getContext()),
                      getAffineDimExpr(1, builder.getContext()),
                      getAffineDimExpr(2, builder.getContext())},
                     builder.getContext());
  AffineMap vMapBM2 =
      AffineMap::get(4, 0,
                     {getAffineDimExpr(0, builder.getContext()),
                      getAffineDimExpr(2, builder.getContext()),
                      getAffineDimExpr(3, builder.getContext())},
                     builder.getContext());
  AffineMap oMapBM2 =
      AffineMap::get(4, 0,
                     {getAffineDimExpr(0, builder.getContext()),
                      getAffineDimExpr(1, builder.getContext()),
                      getAffineDimExpr(3, builder.getContext())},
                     builder.getContext());

  Value o3D =
      linalg::GenericOp::create(
          builder, loc, TypeRange{o3DType}, ValueRange{weights, v3D},
          ValueRange{init3D}, ArrayRef<AffineMap>{wMapBM2, vMapBM2, oMapBM2},
          ArrayRef<utils::IteratorType>{
              utils::IteratorType::parallel, utils::IteratorType::parallel,
              utils::IteratorType::reduction, utils::IteratorType::parallel},
          [&](OpBuilder &b, Location l, ValueRange args) {
            Value mul = arith::MulFOp::create(b, l, args[0], args[1]);
            Value add = arith::AddFOp::create(b, l, mul, args[2]);
            linalg::YieldOp::create(b, l, ValueRange{add});
          })
          .getResult(0);

  // Expand back to [B, H, Lq, D].
  auto oType = RankedTensorType::get({B, H, Lq, D}, f32Type);
  SmallVector<ReassociationIndices> expandAssoc = {{0, 1}, {2}, {3}};
  Value result =
      tensor::ExpandShapeOp::create(builder, loc, oType, o3D, expandAssoc);

  op.replaceAllUsesWith(result);
  op.erase();
  return success();
}

class AttentionLowering final : public llk::LLKKernelLowering {
public:
  StringRef getOpName() const override { return "llk.attention"; }
  LogicalResult inferContract(Operation *root,
                              llk::KernelContract &c) const override {
    auto op = cast<llk::AttentionOp>(root);
    auto type = dyn_cast<RankedTensorType>(op.getQ().getType());
    if (!type || type.getRank() != 4)
      return failure();
    c.workload = "attention";
    c.M = type.getDimSize(0);
    c.N = type.getDimSize(3);
    c.K = type.getDimSize(3);
    c.inputElemType = type.getElementType();
    c.outputElemType = type.getElementType();
    c.accumulatorElemType = type.getElementType();
    c.mBucket = type.hasStaticShape() ? llk::classifyM(c.M) : -1;
    return success();
  }
  bool hasLinalgLowering() const override { return true; }
  LogicalResult lowerToLinalg(Operation *root, OpBuilder &b) const override {
    return lowerAttention(b, cast<llk::AttentionOp>(root));
  }
};
} // namespace
namespace mlir::llk {
void registerAttentionLowering(LLKKernelRegistry &r) {
  r.add<AttentionLowering>();
}
} // namespace mlir::llk
