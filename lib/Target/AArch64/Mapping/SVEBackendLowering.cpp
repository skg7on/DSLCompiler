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
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <memory>

namespace mlir::llk::target::sve {
namespace {

/// The SVE architectural quantum expressed in f32 lanes: every supported SVE
/// length is a multiple of 128 bits, which is four f32 elements. A scalable
/// vector `<[4]xf32>` therefore has `vscale * 4` lanes.
constexpr int64_t kSVEF32ScalableElements = 4;

/// The modeled 512-bit profile: 16 f32 lanes = vscale 4 of the quantum.
constexpr int64_t kModeledF32VectorWidth = 16;

/// True when `op` carries the selected provenance of one of the two arithmetic
/// emitters this target lowers. The emitter key is opaque metadata here; the
/// pass never infers a target decision from the operation's own spelling.
bool isSelectedSVEGroup(Operation *op) {
  auto mapping = op->getAttrOfType<DictionaryAttr>("micro.mapping");
  if (!mapping)
    return false;
  auto emitter = mapping.getAs<StringAttr>("emitter");
  if (!emitter)
    return false;
  return emitter.getValue() == "sve_vector_add" ||
         emitter.getValue() == "sve_vector_mul";
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

    // No selected SVE group is not this pass's failure to report. A selected
    // plan that this target cannot lower is accounted for by the shared
    // compilation contract -- reference groups and a zero realized count are
    // what it diagnoses -- and failing here would replace that stable message
    // with a pipeline failure.
    IRRewriter rewriter(&getContext());
    for (Operation *op : covered) {
      if (failed(lowerSelectedGroup(op, rewriter))) {
        signalPassFailure();
        return;
      }
      ++realized_;
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
