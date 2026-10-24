//===- SMEVectorLowering.cpp - Selected SME vector lowering --------------===//
//
// Part of the LLK Compiler Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// The pre-bufferization half of the selected Arm SME backend. It reads only
// opaque mapping provenance (`micro.mapping` with the selected emitter key),
// tiles the selected matmul with scalable [4]x[4] sizes (16x16 at SVL 512) and
// vectorizes it to `vector.contract`. The post-bufferization pass applies the
// Vector lowering patterns and converts the result to ArmSME: the reference
// ArmSME pipeline does the same after bufferization because the unit-dim
// transfer patterns operate on memrefs.

#include "LLK/Target/AArch64/Mapping/SMEMappingTarget.h"

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
#include "mlir/Transforms/Passes.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <memory>

namespace mlir::llk::target::sme {
namespace {

constexpr int64_t kSMEQuantumF32 = 4;       // vscale * 4 lanes
constexpr int64_t kModeledContraction = 16; // M = N = K at SVL 512

bool isSelectedContraction(Operation *op) {
  // `MicroToLinalg` casts narrower inputs to the accumulator type, so a bf16
  // contraction reaches this pass with f32 operands and an f32 result. The
  // selected bundle name is the surviving provenance that distinguishes it
  // from the f32 signature the emitter accepted.
  auto mapping = op->getAttrOfType<DictionaryAttr>("micro.mapping");
  if (!mapping)
    return false;
  auto emitter = mapping.getAs<StringAttr>("emitter");
  auto bundle = mapping.getAs<StringAttr>("bundle");
  return emitter && emitter.getValue() == "sme_mma_za" && bundle &&
         bundle.getValue() == "sme.mma.f32";
}

bool isExecutableSignature(Operation *op) {
  auto matmul = dyn_cast<linalg::MatmulOp>(op);
  if (!matmul || matmul->getNumResults() != 1)
    return false;
  auto microOp = op->getAttrOfType<StringAttr>("micro.op");
  if (!microOp || microOp.getValue() != "mma")
    return false;
  auto result = dyn_cast<ShapedType>(matmul->getResult(0).getType());
  return result && result.getRank() == 2 && result.hasStaticShape() &&
         result.getShape() ==
             ArrayRef<int64_t>{kModeledContraction, kModeledContraction} &&
         result.getElementType().isF32();
}

class SMEVectorLoweringPass
    : public PassWrapper<SMEVectorLoweringPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SMEVectorLoweringPass)
  StringRef getArgument() const final { return "llk-sme-vector-lowering"; }
  StringRef getDescription() const final {
    return "Tile and vectorize the selected SME contraction";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry
        .insert<affine::AffineDialect, arith::ArithDialect,
                linalg::LinalgDialect, scf::SCFDialect, tensor::TensorDialect,
                ub::UBDialect, vector::VectorDialect>();
    linalg::registerTilingInterfaceExternalModels(registry);
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    SmallVector<Operation *> selected;
    module.walk([&](linalg::LinalgOp op) {
      if (isSelectedContraction(op))
        selected.push_back(op);
    });
    if (selected.empty()) {
      module->setAttr("llk.sme.selected_groups",
                      IntegerAttr::get(IntegerType::get(&getContext(), 64), 0));
      return;
    }

    IRRewriter rewriter(&getContext());
    unsigned realized = 0;
    for (Operation *op : selected) {
      if (!isExecutableSignature(op)) {
        op->emitError(
            "SME selected lowering needs a static 16x16x16 f32 contraction");
        signalPassFailure();
        return;
      }
      if (failed(lowerSelectedMatmul(cast<linalg::MatmulOp>(op), rewriter))) {
        signalPassFailure();
        return;
      }
      ++realized;
    }
    module->setAttr(
        "llk.sme.selected_groups",
        IntegerAttr::get(IntegerType::get(&getContext(), 64), realized));
  }

private:
  LogicalResult lowerSelectedMatmul(linalg::MatmulOp matmul,
                                    IRRewriter &rewriter) {
    Location loc = matmul.getLoc();
    scf::SCFTilingOptions tiling;
    tiling.setTileSizeComputationFunction([loc](OpBuilder &builder,
                                                Operation *) {
      Value vscale =
          vector::VectorScaleOp::create(builder, loc, builder.getIndexType());
      Value quantum =
          arith::ConstantIndexOp::create(builder, loc, kSMEQuantumF32);
      Value tile =
          arith::MulIOp::create(builder, loc, quantum, vscale).getResult();
      return SmallVector<OpFoldResult>{tile, tile, builder.getIndexAttr(1)};
    });
    FailureOr<scf::SCFTilingResult> tiled = scf::tileUsingSCF(
        rewriter, cast<TilingInterface>(matmul.getOperation()), tiling);
    if (failed(tiled)) {
      matmul.emitError("SME scalable tiling failed");
      return failure();
    }
    rewriter.replaceOp(matmul, tiled->replacements);

    SmallVector<int64_t> vectorSizes{kSMEQuantumF32, kSMEQuantumF32, 1};
    SmallVector<bool> scalableVecDims{true, true, false};
    for (Operation *tiledOp : tiled->tiledOps) {
      FailureOr<linalg::VectorizationResult> vectorized = linalg::vectorize(
          rewriter, tiledOp, vectorSizes, scalableVecDims,
          /*vectorizeNDExtract=*/false, /*flatten1DDepthwiseConv=*/false,
          /*assumeDynamicDimsMatchVecSizes=*/false,
          /*createNamedContraction=*/true);
      if (failed(vectorized)) {
        tiledOp->emitError("SME contraction vectorization failed");
        return failure();
      }
      rewriter.replaceOp(tiledOp, vectorized->replacements);
    }
    return success();
  }
};

} // namespace

void buildSMESelectedBackendPipeline(mlir::OpPassManager &pm) {
  pm.addPass(std::make_unique<SMEVectorLoweringPass>());
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createCSEPass());
}

} // namespace mlir::llk::target::sme
