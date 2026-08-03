//===- SharedMemToScratch.cpp - shared mem → scratch allocs ---------------===//
//
// Lowers Triton tt.alloc (shared memory space 3) to per-worker memref.alloc
// scratch buffers. tt.async_copy becomes synchronous memref.copy since CPU
// threads are independent workers — no barrier is needed.
//
//===----------------------------------------------------------------------===//

#include "LLK/Conversion/TritonToLLK/SharedMemToScratch.h"

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Config/llvm-config.h"

using namespace mlir;

namespace {

/// Check if an op matches a given Triton dialect name.
static bool isTritonOp(Operation *op, StringRef name) {
  return op->getName().getStringRef() == name;
}

//===----------------------------------------------------------------------===//
// Pattern: tt.alloc → memref.alloc (shared memory space 3 → 0)
//===----------------------------------------------------------------------===//

struct SharedAllocPattern : public RewritePattern {
  SharedAllocPattern(MLIRContext *ctx)
      : RewritePattern(MatchAnyOpTypeTag(), /*benefit=*/1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    if (!isTritonOp(op, "tt.alloc"))
      return failure();

    auto allocType = mlir::cast<MemRefType>(op->getResult(0).getType());
    // Drop the shared-memory space (3 → 0) so the buffer becomes a plain
    // per-worker scratch allocation.
    auto scratchType =
        MemRefType::get(allocType.getShape(), allocType.getElementType(),
                        allocType.getLayout(), Attribute());

    auto scratch = memref::AllocOp::create(rewriter, op->getLoc(), scratchType);
    rewriter.replaceOp(op, scratch.getResult());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Pattern: tt.async_copy → memref.copy (synchronous)
//===----------------------------------------------------------------------===//

struct AsyncCopyPattern : public RewritePattern {
  AsyncCopyPattern(MLIRContext *ctx)
      : RewritePattern(MatchAnyOpTypeTag(), /*benefit=*/1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    if (!isTritonOp(op, "tt.async_copy"))
      return failure();
    if (op->getNumOperands() < 2)
      return failure();

    Value src = op->getOperand(0);
    Value dst = op->getOperand(1);
    memref::CopyOp::create(rewriter, op->getLoc(), src, dst);
    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Pass Definition
//===----------------------------------------------------------------------===//

struct SharedMemToScratchPass
    : public PassWrapper<SharedMemToScratchPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SharedMemToScratchPass)

  StringRef getArgument() const override {
    return "triton-shared-mem-to-scratch";
  }
  StringRef getDescription() const override {
    return "Lower Triton shared memory allocs to per-worker scratch buffers";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<memref::MemRefDialect>();
  }

  void runOnOperation() override {
    MLIRContext *ctx = &getContext();
    RewritePatternSet patterns(ctx);
    patterns.add<SharedAllocPattern>(ctx);
    patterns.add<AsyncCopyPattern>(ctx);

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

std::unique_ptr<Pass> mlir::llk::createSharedMemToScratchPass() {
  return std::make_unique<SharedMemToScratchPass>();
}
