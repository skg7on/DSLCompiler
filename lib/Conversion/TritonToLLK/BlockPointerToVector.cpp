//===- BlockPointerToVector.cpp - legacy contracts ------------------------===//
//
// Part of the LLK Compiler Project, under the Apache License v2.0 with
// LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Experimental generic block pointers use physical element strides. Subviews
// use unit logical steps, clipped sizes, and checked nonnegative offsets.
#include "LLK/Conversion/TritonToLLK/BlockPointerToVector.h"
#include "LegacyValidation.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Config/llvm-config.h"
using namespace mlir;
namespace {
struct BlockAccessPattern : RewritePattern {
  BlockAccessPattern(MLIRContext *ctx, StringRef name)
      : RewritePattern(name, 1, ctx) {}
  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    bool load = op->getName().getStringRef() == "tt.load";
    // The module preflight checks the schema. Keep the pattern's defining-op
    // requirement local as well, since block arguments have no defining op.
    auto *ptr = op->getOperand(0).getDefiningOp();
    if (!ptr || ptr->getName().getStringRef() != "tt.make_block_ptr")
      return rewriter.notifyMatchFailure(op,
                                         "expected a block pointer definition");
    Value base = ptr->getOperand(0);
    auto type = cast<MemRefType>(base.getType());
    auto block =
        ptr->getAttrOfType<DenseI64ArrayAttr>("block_shape").asArrayRef();
    auto shape = type.getShape();
    unsigned rank = type.getRank();
    Location loc = op->getLoc();
    SmallVector<OpFoldResult> offsets, sizes, steps;
    SmallVector<Value> remaining, zeros;
    Value zero = arith::ConstantIndexOp::create(rewriter, loc, 0);
    for (unsigned i = 0; i < rank; ++i) {
      Value off = ptr->getOperand(1 + i);
      Value dim = arith::ConstantIndexOp::create(rewriter, loc, shape[i]);
      Value nonnegative = arith::CmpIOp::create(
          rewriter, loc, arith::CmpIPredicate::sge, off, zero);
      Value belowEnd = arith::CmpIOp::create(
          rewriter, loc, arith::CmpIPredicate::slt, off, dim);
      Value valid = arith::AndIOp::create(rewriter, loc, nonnegative, belowEnd);
      cf::AssertOp::create(rewriter, loc, valid,
                           "triton_invalid_schema: block offset outside base");
      Value left = arith::SubIOp::create(rewriter, loc, dim, off);
      Value blockSize = arith::ConstantIndexOp::create(rewriter, loc, block[i]);
      Value clipped = arith::MinSIOp::create(rewriter, loc, left, blockSize);
      remaining.push_back(left);
      offsets.push_back(off);
      sizes.push_back(clipped);
      // Physical strides already belong to the base; never multiply again.
      steps.push_back(rewriter.getIndexAttr(1));
      zeros.push_back(zero);
    }
    auto view =
        memref::SubViewOp::create(rewriter, loc, base, offsets, sizes, steps);
    auto maskType = VectorType::get(block, rewriter.getI1Type());
    Value mask =
        vector::CreateMaskOp::create(rewriter, loc, maskType, remaining);
    unsigned maskIndex = load ? 1 : 2;
    if (op->getNumOperands() > maskIndex)
      mask =
          arith::AndIOp::create(rewriter, loc, mask, op->getOperand(maskIndex));
    auto permutation = AffineMapAttr::get(
        AffineMap::getMultiDimIdentityMap(rank, rewriter.getContext()));
    auto inBounds = rewriter.getBoolArrayAttr(SmallVector<bool>(rank, false));
    if (load) {
      auto vectorType = VectorType::get(block, type.getElementType());
      Value pad = arith::ConstantOp::create(
          rewriter, loc, type.getElementType(),
          rewriter.getZeroAttr(type.getElementType()));
      auto read = vector::TransferReadOp::create(
          rewriter, loc, vectorType, view.getResult(), ValueRange(zeros),
          permutation, pad, mask, inBounds);
      rewriter.replaceOp(op, read.getResult());
    } else {
      vector::TransferWriteOp::create(rewriter, loc, op->getOperand(1),
                                      view.getResult(), ValueRange(zeros),
                                      permutation, mask, inBounds);
      rewriter.eraseOp(op);
    }
    if (ptr->use_empty())
      rewriter.eraseOp(ptr);
    return success();
  }
};
struct BlockPointerToVectorPass
    : PassWrapper<BlockPointerToVectorPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(BlockPointerToVectorPass)
  StringRef getArgument() const override {
    return "triton-block-ptr-to-vector";
  }
  StringRef getDescription() const override {
    return "Lower checked experimental block pointers to vector transfers";
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, cf::ControlFlowDialect,
                    memref::MemRefDialect, vector::VectorDialect>();
  }
  void runOnOperation() override {
    if (failed(mlir::llk::validateLegacyTriton(getOperation()))) {
      signalPassFailure();
      return;
    }
    RewritePatternSet patterns(&getContext());
    patterns.add<BlockAccessPattern>(&getContext(), "tt.load");
    patterns.add<BlockAccessPattern>(&getContext(), "tt.store");
    if (failed(
#if LLVM_VERSION_MAJOR >= 21
            applyPatternsGreedily(getOperation(), std::move(patterns))
#else
            applyPatternsAndFoldGreedily(getOperation(), std::move(patterns))
#endif
                ))
      signalPassFailure();
    SmallVector<Operation *> unused;
    getOperation().walk([&](Operation *op) {
      if (op->getName().getStringRef() == "tt.make_block_ptr" &&
          op->use_empty())
        unused.push_back(op);
    });
    for (auto *op : unused)
      op->erase();
  }
};
} // namespace
std::unique_ptr<Pass> mlir::llk::createBlockPointerToVectorPass() {
  return std::make_unique<BlockPointerToVectorPass>();
}
