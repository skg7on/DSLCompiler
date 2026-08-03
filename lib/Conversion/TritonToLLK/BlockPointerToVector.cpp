//===- BlockPointerToVector.cpp - block ptr → vector.transfer -------------===//
//
// Lowers Triton block pointers and load/store to explicit vector.transfer
// operations with boundary masking. tt.make_block_ptr becomes a memref.subview;
// tt.advance folds to index arithmetic.
//
//===----------------------------------------------------------------------===//

#include "LLK/Conversion/TritonToLLK/BlockPointerToVector.h"
#include "LLK/Transforms/Common/VectorizationUtils.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Config/llvm-config.h"

using namespace mlir;

namespace {

//===----------------------------------------------------------------------===//
// Helpers
//===----------------------------------------------------------------------===//

/// Build an AffineMapAttr for the identity permutation map of a given rank.
static AffineMapAttr getIdentityPermutationMapAttr(unsigned rank,
                                                   MLIRContext *ctx) {
  return AffineMapAttr::get(AffineMap::getMultiDimIdentityMap(rank, ctx));
}

/// Check if an op matches a given Triton dialect name.
static bool isTritonOp(Operation *op, StringRef name) {
  return op->getName().getStringRef() == name;
}

//===----------------------------------------------------------------------===//
// Pattern: tt.make_block_ptr + tt.load → vector.transfer_read
//===----------------------------------------------------------------------===//

struct BlockPtrLoadPattern : public RewritePattern {
  BlockPtrLoadPattern(MLIRContext *ctx)
      : RewritePattern(MatchAnyOpTypeTag(), /*benefit=*/1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    if (!isTritonOp(op, "tt.load"))
      return failure();

    // tt.load operands: ptr, [mask]
    if (op->getNumOperands() < 1)
      return failure();

    Value ptrVal = op->getOperand(0);
    auto ptrOp = ptrVal.getDefiningOp();
    if (!ptrOp || !isTritonOp(ptrOp, "tt.make_block_ptr"))
      return failure();

    Value base = ptrOp->getOperand(0);
    Location loc = op->getLoc();

    // Extract block pointer attributes.
    auto shapeAttr = ptrOp->getAttrOfType<DenseI64ArrayAttr>("shape");
    auto blockShapeAttr =
        ptrOp->getAttrOfType<DenseI64ArrayAttr>("block_shape");
    auto stridesAttr = ptrOp->getAttrOfType<DenseI64ArrayAttr>("strides");

    if (!shapeAttr || !blockShapeAttr || !stridesAttr)
      return failure();

    auto shape = shapeAttr.asArrayRef();
    auto blockShape = blockShapeAttr.asArrayRef();
    auto strides = stridesAttr.asArrayRef();
    unsigned rank = static_cast<unsigned>(blockShape.size());

    if (rank == 0 || rank > 2)
      return failure();

    // Build offsets from make_block_ptr operands (operands 1..rank).
    SmallVector<OpFoldResult> offsets;
    for (unsigned i = 0; i < rank; ++i)
      offsets.push_back(ptrOp->getOperand(1 + i));

    // Build sizes and subview strides from attributes.
    SmallVector<OpFoldResult> sizes;
    SmallVector<OpFoldResult> subStrides;
    for (unsigned i = 0; i < rank; ++i) {
      sizes.push_back(rewriter.getIndexAttr(blockShape[i]));
      subStrides.push_back(rewriter.getIndexAttr(strides[i]));
    }

    // Create memref.subview.
    auto subview = memref::SubViewOp::create(rewriter, loc, base, offsets,
                                             sizes, subStrides);

    // Determine vector type for the transfer read.
    auto memrefType = mlir::cast<MemRefType>(base.getType());
    auto elementType = memrefType.getElementType();
    SmallVector<int64_t> vectorShape(blockShape.begin(), blockShape.end());
    auto vectorType = VectorType::get(vectorShape, elementType);

    // Zero indices for in-bounds read starting at subview origin.
    SmallVector<Value> zeroIndices = buildZeroIndices(rank, loc, rewriter);

    // Build identity permutation map.
    auto permMapAttr =
        getIdentityPermutationMapAttr(rank, rewriter.getContext());

    // Padding value for out-of-bounds elements.
    Value pad = arith::ConstantOp::create(rewriter, loc, elementType,
                                          rewriter.getZeroAttr(elementType));

    // Optional mask from operand 1 of tt.load.
    Value mask;
    if (op->getNumOperands() >= 2)
      mask = op->getOperand(1);

    // Generate a boundary mask when the block may extend past the global
    // shape. vector.create_mask yields lane-wise `index < remaining`, so a
    // full block (remaining >= blockShape) produces an all-true mask and a
    // partial edge tile is masked to the in-range elements.
    if (!mask) {
      SmallVector<Value> remaining;
      for (unsigned i = 0; i < rank; ++i) {
        Value dim = arith::ConstantIndexOp::create(rewriter, loc, shape[i]);
        Value off = ptrOp->getOperand(1 + i);
        remaining.push_back(arith::SubIOp::create(rewriter, loc, dim, off));
      }
      mask = vector::CreateMaskOp::create(
          rewriter, loc, VectorType::get(vectorShape, rewriter.getI1Type()),
          remaining);
    }

    // All dimensions are in-bounds within the subview.
    SmallVector<bool> inBoundsVec(rank, true);
    auto inBoundsAttr = rewriter.getBoolArrayAttr(inBoundsVec);

    auto transferRead = vector::TransferReadOp::create(
        rewriter, loc, vectorType, subview.getResult(), ValueRange{zeroIndices},
        permMapAttr, pad, mask ? mask : Value(), inBoundsAttr);

    rewriter.replaceOp(op, transferRead.getResult());

    // Erase make_block_ptr if it has no remaining uses.
    if (ptrOp->use_empty())
      rewriter.eraseOp(ptrOp);

    return success();
  }
};

//===----------------------------------------------------------------------===//
// Pattern: tt.make_block_ptr + tt.store → vector.transfer_write
//===----------------------------------------------------------------------===//

struct BlockPtrStorePattern : public RewritePattern {
  BlockPtrStorePattern(MLIRContext *ctx)
      : RewritePattern(MatchAnyOpTypeTag(), /*benefit=*/1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    if (!isTritonOp(op, "tt.store"))
      return failure();

    // tt.store operands: ptr, value, [mask]
    if (op->getNumOperands() < 2)
      return failure();

    Value ptrVal = op->getOperand(0);
    Value storeVal = op->getOperand(1);
    auto ptrOp = ptrVal.getDefiningOp();
    if (!ptrOp || !isTritonOp(ptrOp, "tt.make_block_ptr"))
      return failure();

    Value base = ptrOp->getOperand(0);
    Location loc = op->getLoc();

    auto shapeAttr = ptrOp->getAttrOfType<DenseI64ArrayAttr>("shape");
    auto blockShapeAttr =
        ptrOp->getAttrOfType<DenseI64ArrayAttr>("block_shape");
    auto stridesAttr = ptrOp->getAttrOfType<DenseI64ArrayAttr>("strides");
    if (!shapeAttr || !blockShapeAttr || !stridesAttr)
      return failure();

    auto shape = shapeAttr.asArrayRef();
    auto blockShape = blockShapeAttr.asArrayRef();
    auto strides = stridesAttr.asArrayRef();
    unsigned rank = static_cast<unsigned>(blockShape.size());

    if (rank == 0 || rank > 2)
      return failure();

    // Build offsets from make_block_ptr operands.
    SmallVector<OpFoldResult> offsets;
    for (unsigned i = 0; i < rank; ++i)
      offsets.push_back(ptrOp->getOperand(1 + i));

    SmallVector<OpFoldResult> sizes;
    SmallVector<OpFoldResult> subStrides;
    for (unsigned i = 0; i < rank; ++i) {
      sizes.push_back(rewriter.getIndexAttr(blockShape[i]));
      subStrides.push_back(rewriter.getIndexAttr(strides[i]));
    }

    auto subview = memref::SubViewOp::create(rewriter, loc, base, offsets,
                                             sizes, subStrides);

    // Zero indices at subview origin.
    SmallVector<Value> zeroIndices = buildZeroIndices(rank, loc, rewriter);

    auto permMapAttr =
        getIdentityPermutationMapAttr(rank, rewriter.getContext());

    // Optional mask from operand 2 of tt.store.
    Value mask;
    if (op->getNumOperands() >= 3)
      mask = op->getOperand(2);

    // Vector shape derived from the block shape for mask generation.
    SmallVector<int64_t> vectorShape(blockShape.begin(), blockShape.end());

    // Generate a boundary mask when the block may extend past the global
    // shape, so a partial edge tile never writes out-of-bounds elements.
    if (!mask) {
      SmallVector<Value> remaining;
      for (unsigned i = 0; i < rank; ++i) {
        Value dim = arith::ConstantIndexOp::create(rewriter, loc, shape[i]);
        Value off = ptrOp->getOperand(1 + i);
        remaining.push_back(arith::SubIOp::create(rewriter, loc, dim, off));
      }
      mask = vector::CreateMaskOp::create(
          rewriter, loc, VectorType::get(vectorShape, rewriter.getI1Type()),
          remaining);
    }

    // All dimensions are in-bounds within the subview.
    SmallVector<bool> inBoundsVec(rank, true);
    auto inBoundsAttr = rewriter.getBoolArrayAttr(inBoundsVec);

    vector::TransferWriteOp::create(
        rewriter, loc, storeVal, subview.getResult(), ValueRange{zeroIndices},
        permMapAttr, mask ? mask : Value(), inBoundsAttr);

    rewriter.eraseOp(op);

    // Erase make_block_ptr if it has no remaining uses.
    if (ptrOp->use_empty())
      rewriter.eraseOp(ptrOp);

    return success();
  }
};

//===----------------------------------------------------------------------===//
// Pass Definition
//===----------------------------------------------------------------------===//

struct BlockPointerToVectorPass
    : public PassWrapper<BlockPointerToVectorPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(BlockPointerToVectorPass)

  StringRef getArgument() const override {
    return "triton-block-ptr-to-vector";
  }
  StringRef getDescription() const override {
    return "Lower Triton block pointers to vector.transfer_read/write";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect>();
    registry.insert<memref::MemRefDialect>();
    registry.insert<vector::VectorDialect>();
  }

  void runOnOperation() override {
    MLIRContext *ctx = &getContext();
    RewritePatternSet patterns(ctx);
    patterns.add<BlockPtrLoadPattern>(ctx);
    patterns.add<BlockPtrStorePattern>(ctx);

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

std::unique_ptr<Pass> mlir::llk::createBlockPointerToVectorPass() {
  return std::make_unique<BlockPointerToVectorPass>();
}
