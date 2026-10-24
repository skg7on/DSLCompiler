//===- SVEBackendLowering.cpp - Selected SVE scalable-vector lowering -----===//
//
// Part of the LLK Compiler Project, under the Apache License v2.0 with
// LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// The selected `arm-sve` backend pass. It runs after the generic Micro to
// Linalg conversion and realizes the f32 vector add/multiply groups the
// `sve_vector_add`/`sve_vector_mul` emitters handed off as scalable SVE
// Vector IR.
//
// It reads only opaque mapping provenance (`micro.mapping` with the selected
// emitter key), never a target's bundle fields: the generic conversion copies
// that metadata for every target, while a target-specific handoff attribute is
// not part of the bridge's contract. This mirrors the AVX2 selected pass.
//
// The lowering is a scalable tiling followed by vectorization:
//   * the row dimension stays whole, so an 8x8 tile is one Vector IR value;
//   * the column loop advances by `vscale * 4` f32 lanes, the SVE quantum.
// Vectorization then emits `vector<rows x [4]xf32>` with a `vector.create_mask`
// tail, so the same IR is correct at any SVE vector length. The modeled
// 512-bit profile selects 16 f32 lanes (vscale = 4); this pass requires that
// profile width but deliberately emits the length-agnostic form.

#include "LLK/Target/AArch64/Mapping/SVEMappingTarget.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/TilingInterfaceImpl.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/UB/IR/UBOps.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Transforms/LoweringPatterns.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <memory>
#include <optional>

namespace mlir::llk::target::sve {
namespace {

/// The SVE architectural quantum expressed in f32 lanes: every supported SVE
/// length is a multiple of 128 bits, which is four f32 elements. A scalable
/// vector `<[4]xf32>` therefore has `vscale * 4` lanes.
constexpr int64_t kSVEF32ScalableElements = 4;

/// The modeled 512-bit profile: 16 f32 lanes = vscale 4 of the quantum.
constexpr int64_t kModeledF32VectorWidth = 16;

/// The emitter key a selected operation carries, or null when it carries none.
StringAttr selectedEmitter(Operation *op) {
  auto mapping = op->getAttrOfType<DictionaryAttr>("micro.mapping");
  return mapping ? mapping.getAs<StringAttr>("emitter") : StringAttr();
}

/// True when `op` carries the selected provenance of one of the two arithmetic
/// emitters this target lowers. The emitter key is opaque metadata here; the
/// pass never infers a target decision from the operation's own spelling.
bool isSelectedSVEGroup(Operation *op) {
  StringAttr emitter = selectedEmitter(op);
  if (!emitter)
    return false;
  return emitter.getValue() == "sve_vector_add" ||
         emitter.getValue() == "sve_vector_mul";
}

/// True when `op` is the signature this target's emitters implement and this
/// pass can realize: the emitter's own `micro.vector` add/multiply on a static
/// rank-two f32 tile.
///
/// This mirrors `SVEEmitter::hasLoweringFor`. Keeping the two views of the
/// executable subset identical matters: a group the emitter declines is
/// counted by the shared layer as reference-lowered, so the pass must not
/// silently realize it, and a group the emitter accepts must not be skipped
/// here or the compilation would report success with unlowered selected work.
bool isExecutableSignature(Operation *op) {
  auto linalgOp = dyn_cast<linalg::LinalgOp>(op);
  if (!linalgOp)
    return false;
  StringAttr emitter = selectedEmitter(op);
  auto name = op->getAttrOfType<StringAttr>("micro.op");
  if (!emitter || !name)
    return false;
  llvm::StringRef expected =
      emitter.getValue() == "sve_vector_add" ? "add" : "mul";
  if (name.getValue() != expected)
    return false;
  // Exactly one result is part of the executable signature: the emitters
  // implement a single tile result, and indexing result 0 without this check
  // would assert on a zero-result (memory-out) Linalg form.
  if (linalgOp->getNumResults() != 1)
    return false;
  auto output = dyn_cast<ShapedType>(linalgOp->getResult(0).getType());
  return output && output.getRank() == 2 && output.hasStaticShape() &&
         output.getElementType().isF32();
}

/// True when the transfer's leading vector dimension maps to the source's
/// leading index, which is what lets the row rewrite advance that index.
bool leadingVectorDimensionIsRow(AffineMap map) {
  if (!map.isMinorIdentity() || map.getNumResults() != 2)
    return false;
  auto row = dyn_cast<AffineDimExpr>(map.getResult(0));
  return row && row.getPosition() == 0;
}

/// The rank-two scalable vector type the vectorizer emits for a whole tile, or
/// null when `type` is not `vector<rows x [k]xT>` with a static row count and a
/// scalable trailing dimension.
std::optional<VectorType> rowScalableVector(Type type) {
  auto vector = dyn_cast<VectorType>(type);
  if (!vector || vector.getRank() != 2)
    return std::nullopt;
  ArrayRef<bool> scalable = vector.getScalableDims();
  if (scalable[0] || !scalable[1])
    return std::nullopt;
  return vector;
}

/// Rewrites a rank-two masked transfer into one rank-one transfer per row.
///
/// The project's AOT LLVM pipeline lowers a rank-two scalable transfer through
/// VectorToSCF, which stores the rank-two mask vector into an `alloca` before
/// rank-reducing it. AArch64 cannot select a store of a scalable predicate
/// vector, so that form does not compile. A rank-one masked transfer lowers
/// directly to `llvm.masked.load`/`llvm.masked.store`, which the SVE backend
/// selects as a predicated load/store.
///
/// The rank-two vector value is preserved through `vector.extract` and
/// `vector.insert`, so the selected artifact keeps its whole-tile scalable
/// arithmetic; only the transfers become per-row.
LogicalResult lowerRowScalableTransfers(Operation *root, IRRewriter &rewriter) {
  SmallVector<vector::TransferReadOp> reads;
  SmallVector<vector::TransferWriteOp> writes;
  root->walk([&](Operation *op) {
    if (auto read = dyn_cast<vector::TransferReadOp>(op))
      reads.push_back(read);
    else if (auto write = dyn_cast<vector::TransferWriteOp>(op))
      writes.push_back(write);
  });

  for (vector::TransferReadOp read : reads) {
    VectorType vector = cast<VectorType>(read.getVectorType());
    std::optional<VectorType> rowVector = rowScalableVector(vector);
    AffineMap map = read.getPermutationMap();
    if (!rowVector || !leadingVectorDimensionIsRow(map))
      continue;
    Location loc = read.getLoc();
    VectorType rowType =
        VectorType::get({vector.getShape()[1]}, vector.getElementType(),
                        /*scalableDims=*/true);
    AffineMap rowMap =
        AffineMap::get(map.getNumDims(), map.getNumSymbols(),
                       {map.getResult(1)}, rewriter.getContext());
    ArrayAttr inBounds = read.getInBounds();
    ArrayAttr rowBounds = ArrayAttr::get(rewriter.getContext(), {inBounds[1]});
    Value mask = read.getMask();
    SmallVector<Value> indices(read.getIndices().begin(),
                               read.getIndices().end());
    rewriter.setInsertionPoint(read);
    Value result = ub::PoisonOp::create(rewriter, loc, vector);
    for (int64_t row = 0; row != vector.getShape()[0]; ++row) {
      Value rowMask;
      if (mask)
        rowMask = vector::ExtractOp::create(rewriter, loc, mask, ValueRange{},
                                            ArrayRef<int64_t>{row});
      SmallVector<Value> rowIndices(indices);
      rowIndices[0] = arith::AddIOp::create(
                          rewriter, loc, indices[0],
                          arith::ConstantIndexOp::create(rewriter, loc, row))
                          .getResult();
      auto rowRead = vector::TransferReadOp::create(
          rewriter, loc, rowType, read.getBase(), rowIndices, rowMap,
          read.getPadding(), rowMask, rowBounds);
      result = vector::InsertOp::create(rewriter, loc, rowRead, result,
                                        ValueRange{}, ArrayRef<int64_t>{row});
    }
    rewriter.replaceOp(read, result);
  }

  for (vector::TransferWriteOp write : writes) {
    auto vector = dyn_cast<VectorType>(write.getValueToStore().getType());
    if (!vector)
      continue;
    std::optional<VectorType> rowVector = rowScalableVector(vector);
    AffineMap map = write.getPermutationMap();
    if (!rowVector || !leadingVectorDimensionIsRow(map))
      continue;
    Location loc = write.getLoc();
    AffineMap rowMap =
        AffineMap::get(map.getNumDims(), map.getNumSymbols(),
                       {map.getResult(1)}, rewriter.getContext());
    ArrayAttr inBounds = write.getInBounds();
    ArrayAttr rowBounds = ArrayAttr::get(rewriter.getContext(), {inBounds[1]});
    Value mask = write.getMask();
    SmallVector<Value> indices(write.getIndices().begin(),
                               write.getIndices().end());
    Type resultType = write.getResult() ? write.getResult().getType() : Type();
    Value destination = write.getBase();
    rewriter.setInsertionPoint(write);
    for (int64_t row = 0; row != vector.getShape()[0]; ++row) {
      Value rowValue =
          vector::ExtractOp::create(rewriter, loc, write.getValueToStore(),
                                    ValueRange{}, ArrayRef<int64_t>{row});
      Value rowMask;
      if (mask)
        rowMask = vector::ExtractOp::create(rewriter, loc, mask, ValueRange{},
                                            ArrayRef<int64_t>{row});
      SmallVector<Value> rowIndices(indices);
      rowIndices[0] = arith::AddIOp::create(
                          rewriter, loc, indices[0],
                          arith::ConstantIndexOp::create(rewriter, loc, row))
                          .getResult();
      auto rowWrite = vector::TransferWriteOp::create(
          rewriter, loc, resultType, rowValue, destination, rowIndices, rowMap,
          rowMask, rowBounds);
      if (write.getResult())
        destination = rowWrite.getResult();
    }
    if (write.getResult())
      rewriter.replaceOp(write, destination);
    else
      rewriter.eraseOp(write);
  }
  return success();
}

class SVEBackendLoweringPass
    : public PassWrapper<SVEBackendLoweringPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SVEBackendLoweringPass)

  SVEBackendLoweringPass() = default;
  SVEBackendLoweringPass(const SVEBackendLoweringPass &other)
      : PassWrapper(other), f32Width_(other.f32Width_) {}
  explicit SVEBackendLoweringPass(int64_t f32Width) : f32Width_(f32Width) {}

  StringRef getArgument() const final { return "llk-sve-selected-lowering"; }
  StringRef getDescription() const final {
    return "Realize selected SVE f32 elementwise groups as scalable Vector IR";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry
        .insert<affine::AffineDialect, arith::ArithDialect,
                linalg::LinalgDialect, scf::SCFDialect, tensor::TensorDialect,
                ub::UBDialect, vector::VectorDialect>();
    // Tiling a Linalg operation goes through the TilingInterface, which Linalg
    // implementations supply as an external model rather than a dialect-level
    // interface. A context that only loaded the dialect would fail the cast.
    linalg::registerTilingInterfaceExternalModels(registry);
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    realized_ = 0;

    SmallVector<Operation *> covered;
    module.walk([&](linalg::LinalgOp linalgOp) {
      if (isSelectedSVEGroup(linalgOp))
        covered.push_back(linalgOp);
    });

    // An empty or entirely non-executable selected set leaves the realized
    // count at zero. That is not this pass's failure to report: the shared
    // compilation contract turns it into "realized no selected compute
    // groups", which is the diagnostic a caller needs, instead of a generic
    // pipeline failure.
    IRRewriter rewriter(&getContext());
    for (Operation *op : covered) {
      // A group the emitters decline (another dtype or shape) is not this
      // target's selected code; the shared layer already counted it as
      // reference-lowered, and the compilation contract reports it. Skipping
      // it here keeps the reference-bridge accounting honest instead of
      // realizing work the emitter said it does not implement.
      if (!isExecutableSignature(op))
        continue;
      if (failed(lowerSelectedGroup(op, rewriter))) {
        signalPassFailure();
        return;
      }
      ++realized_;
    }

    // Dissolve `vector.mask` into the mask operands of the side-effecting
    // vector operations. Preparing a mapped kernel for invocation runs
    // ownership-based buffer deallocation, which inserts an ownership
    // constant at the start of every region it traverses; a `vector.mask`
    // region holds exactly one operation, so that insertion fails the
    // verifier. Masked transfers model the same predication directly and are
    // the form the downstream bufferization and LLVM lowering consume.
    if (realized_ > 0) {
      RewritePatternSet patterns(&getContext());
      vector::populateVectorMaskLoweringPatternsForSideEffectingOps(patterns);
      vector::MaskOp::getCanonicalizationPatterns(patterns, &getContext());
      if (failed(applyPatternsGreedily(module, std::move(patterns)))) {
        module.emitError(
            "SVE backend could not lower the masked vector operations");
        signalPassFailure();
        return;
      }
      if (failed(lowerRowScalableTransfers(module, rewriter))) {
        module.emitError(
            "SVE backend could not rank-reduce the scalable transfers");
        signalPassFailure();
        return;
      }
    }

    module->setAttr(
        "llk.backend_groups_realized",
        IntegerAttr::get(IntegerType::get(&getContext(), 64), realized_));
  }

private:
  /// Tiles one selected elementwise group by whole rows and scalable f32
  /// columns, then vectorizes each tile into masked scalable Vector IR.
  LogicalResult lowerSelectedGroup(Operation *op, IRRewriter &rewriter) {
    auto linalgOp = dyn_cast<linalg::LinalgOp>(op);
    auto name = op->getAttrOfType<StringAttr>("micro.op");
    if (!linalgOp || !name ||
        (name.getValue() != "add" && name.getValue() != "mul")) {
      op->emitError("SVE backend only lowers add/multiply vector groups");
      return failure();
    }
    // The signature predicate already requires one result, but this function
    // is the one that indexes it. Refuse with a diagnostic instead of
    // asserting if a caller reaches it with a zero-result group.
    if (linalgOp->getNumResults() != 1) {
      op->emitError("SVE backend needs exactly one result");
      return failure();
    }
    auto output = dyn_cast<ShapedType>(linalgOp->getResult(0).getType());
    if (!output || output.getRank() != 2 || !output.hasStaticShape() ||
        !output.getElementType().isF32()) {
      op->emitError("SVE backend needs a static rank-two f32 result");
      return failure();
    }
    if (f32Width_ != kModeledF32VectorWidth) {
      op->emitError("SVE backend supports the modeled 16-lane f32 width");
      return failure();
    }

    // Advance the column loop by one SVE quantum (vscale * 4 f32 lanes). The
    // tile is clamped to the remaining columns, which is what makes the
    // vectorizer emit a mask for a tile narrower than the vector.
    Location loc = op->getLoc();
    scf::SCFTilingOptions tiling;
    tiling.setTileSizeComputationFunction([loc](OpBuilder &builder,
                                                Operation *) {
      SmallVector<OpFoldResult> sizes;
      sizes.push_back(builder.getIndexAttr(0));
      Value unit =
          arith::ConstantIndexOp::create(builder, loc, kSVEF32ScalableElements);
      Value vscale =
          vector::VectorScaleOp::create(builder, loc, builder.getIndexType());
      sizes.push_back(
          arith::MulIOp::create(builder, loc, unit, vscale).getResult());
      return sizes;
    });
    auto tilingInterface = cast<TilingInterface>(linalgOp.getOperation());
    FailureOr<scf::SCFTilingResult> tiled =
        scf::tileUsingSCF(rewriter, tilingInterface, tiling);
    if (failed(tiled)) {
      op->emitError("SVE scalable tiling failed");
      return failure();
    }
    rewriter.replaceOp(op, tiled->replacements);

    SmallVector<int64_t> vectorSizes{output.getShape()[0],
                                     kSVEF32ScalableElements};
    SmallVector<bool> scalableVecDims{false, true};
    for (Operation *tiledOp : tiled->tiledOps) {
      if (!isa<linalg::LinalgOp>(tiledOp)) {
        tiledOp->emitError("SVE scalable tiling produced a non-Linalg tile");
        return failure();
      }
      FailureOr<linalg::VectorizationResult> vectorized =
          linalg::vectorize(rewriter, tiledOp, vectorSizes, scalableVecDims);
      if (failed(vectorized)) {
        tiledOp->emitError("SVE scalable vectorization failed");
        return failure();
      }
      rewriter.replaceOp(tiledOp, vectorized->replacements);
    }
    return success();
  }

  int64_t f32Width_ = kModeledF32VectorWidth;
  unsigned realized_ = 0;
};

} // namespace

void buildSVESelectedBackendPipeline(mlir::OpPassManager &pm,
                                     int64_t f32VectorWidth) {
  pm.addPass(std::make_unique<SVEBackendLoweringPass>(f32VectorWidth));
}

} // namespace mlir::llk::target::sve
