//===- FuseDoubleContraction.cpp - Fuse 2 matmuls sharing X + SiLU -------===//
//
// Part of the LLK Compiler Project, under the Apache License v2.0.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Fuses 2 linalg.matmul ops (sharing the same X operand) + their linalg.generic
// SiLU consumer into a single linalg.generic with 2 reduction outputs, plus
// a separate elementwise epilogue for the SiLU(gate) * up computation.
//
// This eliminates full-size intermediate tensors (gate, up) that would
// otherwise require O(M*N) storage and bandwidth between the matmuls and the
// SiLU.  The fused operation computes both projections in one pass over K,
// writing only the final result.
//
// Matching criteria:
//   - Consumer is linalg.generic with 2 inputs
//   - Both inputs are results of linalg::MatmulOp
//   - Both matmuls share the same X (first input) operand
//   - Both matmul results have one use (the consumer)
//
//===----------------------------------------------------------------------===//

#include "LLK/Transforms/FuseDoubleContraction.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Config/llvm-config.h"

using namespace mlir;

namespace {

//===----------------------------------------------------------------------===//
// FuseDoubleMatmulWithSiLU pattern
//===----------------------------------------------------------------------===//

struct FuseDoubleMatmulWithSiLU : public OpRewritePattern<linalg::GenericOp> {
  using OpRewritePattern<linalg::GenericOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::GenericOp consumer,
                                PatternRewriter &rewriter) const override {
    // Match: consumer takes two inputs (gate, up) and does silu(gate) * up
    if (consumer.getNumDpsInputs() != 2)
      return failure();

    Value gateVal = consumer.getDpsInputOperand(0)->get();
    Value upVal = consumer.getDpsInputOperand(1)->get();

    auto gateMatmul = gateVal.getDefiningOp<linalg::MatmulOp>();
    auto upMatmul = upVal.getDefiningOp<linalg::MatmulOp>();
    if (!gateMatmul || !upMatmul)
      return failure();

    // Both matmuls must share the same X operand
    Value xGate = gateMatmul.getDpsInputOperand(0)->get();
    Value xUp = upMatmul.getDpsInputOperand(0)->get();
    if (xGate != xUp)
      return failure();

    // Both must have only one use (the consumer)
    if (!gateVal.hasOneUse() || !upVal.hasOneUse())
      return failure();

    //=== Verify consumer body is actually a SiLU pattern ===

    // Check identity indexing maps (elementwise)
    for (AffineMap map : consumer.getIndexingMapsArray()) {
      if (!map.isIdentity())
        return failure();
    }

    // Check output type is bf16 (SiLU epilogue truncates to bf16)
    for (Value result : consumer.getResults()) {
      auto resultTy = mlir::dyn_cast<RankedTensorType>(result.getType());
      if (!resultTy || !resultTy.getElementType().isBF16())
        return failure();
    }

    // Verify body contains SiLU operations.
    // Accept both math::ExpOp (strict mode) and math::Exp2Op (bounded_fast
    // mode).
    Block &body = consumer->getRegion(0).front();
    bool hasNegf = false, hasExp = false, hasAddf = false;
    bool hasDivf = false, hasMulf = false, hasTruncf = false;
    for (Operation &op : body) {
      if (isa<arith::NegFOp>(op))
        hasNegf = true;
      if (isa<math::ExpOp>(op) || isa<math::Exp2Op>(op))
        hasExp = true;
      if (isa<arith::AddFOp>(op))
        hasAddf = true;
      if (isa<arith::DivFOp>(op))
        hasDivf = true;
      if (isa<arith::MulFOp>(op))
        hasMulf = true;
      if (isa<arith::TruncFOp>(op))
        hasTruncf = true;
    }
    if (!hasNegf || !hasExp || !hasAddf || !hasDivf || !hasMulf || !hasTruncf)
      return failure();

    // Build fused operation
    Location loc = consumer.getLoc();
    Value x = xGate;
    Value wg = gateMatmul.getDpsInputOperand(1)->get();
    Value wu = upMatmul.getDpsInputOperand(1)->get();
    Value gateInit = gateMatmul.getDpsInitOperand(0)->get();
    Value upInit = upMatmul.getDpsInitOperand(0)->get();

    auto f32Type = rewriter.getF32Type();

    // Create f32 init tensors for f32 accumulation.
    auto gateTy = mlir::cast<RankedTensorType>(gateInit.getType());
    auto upTy = mlir::cast<RankedTensorType>(upInit.getType());
    auto f32GateTy = RankedTensorType::get(gateTy.getShape(), f32Type);
    auto f32UpTy = RankedTensorType::get(upTy.getShape(), f32Type);
    Value f32Zero = arith::ConstantOp::create(rewriter, loc, f32Type,
                                              rewriter.getF32FloatAttr(0.0));
    Value f32GateInit =
        tensor::EmptyOp::create(rewriter, loc, f32GateTy, ValueRange{});
    f32GateInit = linalg::FillOp::create(rewriter, loc, ValueRange{f32Zero},
                                         ValueRange{f32GateInit})
                      .getResult(0);
    Value f32UpInit =
        tensor::EmptyOp::create(rewriter, loc, f32UpTy, ValueRange{});
    f32UpInit = linalg::FillOp::create(rewriter, loc, ValueRange{f32Zero},
                                       ValueRange{f32UpInit})
                    .getResult(0);

    SmallVector<AffineMap> indexingMaps = {
        AffineMap::get(
            3, 0, {rewriter.getAffineDimExpr(0), rewriter.getAffineDimExpr(2)},
            rewriter.getContext()), // X: (m,k)
        AffineMap::get(
            3, 0, {rewriter.getAffineDimExpr(2), rewriter.getAffineDimExpr(1)},
            rewriter.getContext()), // Wg: (k,n)
        AffineMap::get(
            3, 0, {rewriter.getAffineDimExpr(2), rewriter.getAffineDimExpr(1)},
            rewriter.getContext()), // Wu: (k,n)
        AffineMap::get(
            3, 0, {rewriter.getAffineDimExpr(0), rewriter.getAffineDimExpr(1)},
            rewriter.getContext()), // gate_acc: (m,n)
        AffineMap::get(
            3, 0, {rewriter.getAffineDimExpr(0), rewriter.getAffineDimExpr(1)},
            rewriter.getContext()), // up_acc: (m,n)
    };

    auto fused = linalg::GenericOp::create(
        rewriter, loc, TypeRange{f32GateTy, f32UpTy}, // 2 f32 results
        ValueRange{x, wg, wu},                        // 3 inputs
        ValueRange{f32GateInit, f32UpInit},           // 2 f32 inits
        indexingMaps,
        SmallVector<utils::IteratorType>{utils::IteratorType::parallel,
                                         utils::IteratorType::parallel,
                                         utils::IteratorType::reduction},
        [&](OpBuilder &b, Location loc, ValueRange args) {
          Value xEl = arith::ExtFOp::create(b, loc, f32Type, args[0]);
          Value wgEl = arith::ExtFOp::create(b, loc, f32Type, args[1]);
          Value wuEl = arith::ExtFOp::create(b, loc, f32Type, args[2]);
          // args[3] and args[4] are already f32 (init tensors)
          Value gNew = arith::AddFOp::create(
              b, loc, args[3], arith::MulFOp::create(b, loc, xEl, wgEl));
          Value uNew = arith::AddFOp::create(
              b, loc, args[4], arith::MulFOp::create(b, loc, xEl, wuEl));
          linalg::YieldOp::create(b, loc, ValueRange{gNew, uNew});
        });

    // Build SiLU epilogue on the two accumulators
    auto bf16Type = rewriter.getBF16Type();
    Value cstOne = arith::ConstantOp::create(rewriter, loc, f32Type,
                                             rewriter.getF32FloatAttr(1.0f));
    auto epilogue = linalg::GenericOp::create(
        rewriter, loc, consumer.getResultTypes(),
        ValueRange{fused.getResult(0), fused.getResult(1)},
        consumer.getDpsInitOperand(0)->get(),
        ArrayRef<AffineMap>{rewriter.getMultiDimIdentityMap(2),
                            rewriter.getMultiDimIdentityMap(2),
                            rewriter.getMultiDimIdentityMap(2)},
        SmallVector<utils::IteratorType>{utils::IteratorType::parallel,
                                         utils::IteratorType::parallel},
        [&](OpBuilder &b, Location loc, ValueRange args) {
          Value g = args[0];
          Value u = args[1];
          Value neg = arith::NegFOp::create(b, loc, g);
          Value exp = math::ExpOp::create(b, loc, neg);
          Value den = arith::AddFOp::create(b, loc, cstOne, exp);
          Value sigmoid = arith::DivFOp::create(b, loc, cstOne, den);
          Value silu = arith::MulFOp::create(b, loc, g, sigmoid);
          Value r = arith::MulFOp::create(b, loc, silu, u);
          Value cast = arith::TruncFOp::create(b, loc, bf16Type, r);
          linalg::YieldOp::create(b, loc, cast);
        });

    rewriter.replaceOp(consumer, epilogue.getResult(0));
    rewriter.eraseOp(gateMatmul);
    rewriter.eraseOp(upMatmul);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// FuseDoubleContractionPass
//===----------------------------------------------------------------------===//

struct FuseDoubleContractionPass
    : public PassWrapper<FuseDoubleContractionPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(FuseDoubleContractionPass)

  StringRef getArgument() const override { return "fuse-double-contraction"; }
  StringRef getDescription() const override {
    return "Fuse two matmuls sharing X operand with SiLU consumer";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect>();
    registry.insert<linalg::LinalgDialect>();
    registry.insert<math::MathDialect>();
    registry.insert<tensor::TensorDialect>();
  }

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<FuseDoubleMatmulWithSiLU>(&getContext());
    if (failed(
#if LLVM_VERSION_MAJOR >= 21
            applyPatternsGreedily(getOperation(), std::move(patterns))
#else
            applyPatternsAndFoldGreedily(getOperation(), std::move(patterns))
#endif
                ))
      signalPassFailure();
  }
};

} // namespace

namespace mlir {
namespace llk {
std::unique_ptr<Pass> createFuseDoubleContractionPass() {
  return std::make_unique<FuseDoubleContractionPass>();
}
} // namespace llk
} // namespace mlir
