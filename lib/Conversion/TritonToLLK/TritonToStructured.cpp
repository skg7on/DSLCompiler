//===- TritonToStructured.cpp - tt.dot → linalg.matmul -------------------===//
//
// Lowers Triton compute ops to MLIR structured ops. The simple (generic)
// fallback rewrites every tt.dot into a linalg.matmul, keeping the Triton
// accumulator operand as the linalg output (destination-passing style).
//
//===----------------------------------------------------------------------===//

#include "LLK/Conversion/TritonToLLK/TritonToStructured.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Config/llvm-config.h"

using namespace mlir;

namespace {

//===----------------------------------------------------------------------===//
// Generic dot → linalg.matmul pattern
//===----------------------------------------------------------------------===//

struct DotToMatmulPattern : public RewritePattern {
  DotToMatmulPattern(MLIRContext *ctx)
      : RewritePattern(MatchAnyOpTypeTag(), /*benefit=*/1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    if (op->getName().getStringRef() != "tt.dot")
      return failure();

    Value a = op->getOperand(0);
    Value b = op->getOperand(1);
    Value acc = op->getOperand(2);
    Location loc = op->getLoc();

    auto resultType = mlir::cast<ShapedType>(acc.getType());
    auto matmul = linalg::MatmulOp::create(rewriter, loc, resultType,
                                           ValueRange{a, b}, ValueRange{acc});
    rewriter.replaceOp(op, matmul.getResult(0));
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Pass Definition
//===----------------------------------------------------------------------===//

struct TritonToStructuredPass
    : public PassWrapper<TritonToStructuredPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TritonToStructuredPass)

  StringRef getArgument() const override { return "triton-to-structured"; }
  StringRef getDescription() const override {
    return "Lower Triton compute ops to Linalg structured ops";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect>();
    registry.insert<linalg::LinalgDialect>();
    registry.insert<tensor::TensorDialect>();
  }

  void runOnOperation() override {
    MLIRContext *ctx = &getContext();
    RewritePatternSet patterns(ctx);
    patterns.add<DotToMatmulPattern>(ctx);
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

std::unique_ptr<Pass> mlir::llk::createTritonToStructuredPass() {
  return std::make_unique<TritonToStructuredPass>();
}
