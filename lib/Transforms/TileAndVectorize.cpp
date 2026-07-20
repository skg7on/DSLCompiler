//===- TileAndVectorize.cpp - Tiling + Vectorization Pass -----------------===//
//
// Part of the LLK Compiler Project, under the Apache License v2.0.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Applies tiling and vectorization to Linalg operations produced by lowering
// LLK ops.  Uses scf::tileUsingSCF for two-pass tiling (reduction K with
// scf.for, then parallel M/N with scf.forall), followed by linalg::vectorize.
//
// Pipeline:
//   1. Lower LLK to Linalg (if llk.fused_swiglu ops still present)
//   2. Tile matmul K-reduction with scf.for  (BK = 64)
//   3. Tile matmul + generic M/N with scf.forall (BM = 32, BN = 64)
//   4. Vectorize innermost tiles (VM = 4, VN = 8)
//
// Produces: scf.forall + scf.for + vector.contract + vector.transfer_read/write
//
//===----------------------------------------------------------------------===//

#include "LLK/Transforms/TileAndVectorize.h"
#include "LLK/Conversion/LLKToLinalg.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Transforms/VectorTransforms.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LLVM.h"
#include "llvm/Config/llvm-config.h"

using namespace mlir;

// Include the generated pass base class inside the mlir::llk namespace so
// that the generated impl::TileAndVectorizeBase can resolve
// TileAndVectorizeOptions (defined in the same namespace by the public header).
namespace mlir {
namespace llk {
#define GEN_PASS_DEF_TILEANDVECTORIZE
#include "LLK/Transforms/Passes.h.inc"
} // namespace llk
} // namespace mlir

// Tile dimensions for AVX2 BF16.
static constexpr int64_t kBM = 32;
static constexpr int64_t kBN = 64;
static constexpr int64_t kBK = 64;
static constexpr int64_t kVM = 4;
static constexpr int64_t kVN = 8;

//===----------------------------------------------------------------------===//
// TileAndVectorizePass
//===----------------------------------------------------------------------===//

struct TileAndVectorizePass
    : public llk::impl::TileAndVectorizeBase<TileAndVectorizePass> {
  using TileAndVectorizeBase =
      llk::impl::TileAndVectorizeBase<TileAndVectorizePass>;
  using TileAndVectorizeBase::TileAndVectorizeBase;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect>();
    registry.insert<scf::SCFDialect>();
    registry.insert<vector::VectorDialect>();
    registry.insert<func::FuncDialect>();
    registry.insert<tensor::TensorDialect>();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    MLIRContext *ctx = &getContext();

    // Step 1: Lower LLK to Linalg (idempotent -- no-op if already done).
    {
      PassManager pm(ctx);
      pm.addPass(llk::createLLKToLinalgPass());
      if (failed(pm.run(module))) {
        signalPassFailure();
        return;
      }
    }

    IRRewriter rewriter(ctx);

    // Step 2: Collect matmul and generic ops before modifying the IR.
    SmallVector<linalg::MatmulOp> matmuls;
    SmallVector<linalg::GenericOp> generics;
    module.walk([&](linalg::MatmulOp op) { matmuls.push_back(op); });
    module.walk([&](linalg::GenericOp op) { generics.push_back(op); });

    // Step 3: Tile matmuls: K-reduction with scf.for, then M/N with
    //         scf.forall, then vectorize.
    for (auto matmul : matmuls) {
      // 3a. Tile K reduction with scf.for (FullReduction strategy).
      //     LoopType::ForOp + FullReduction avoids the partial-reduction check
      //     that blocks linalg.matmul.
      scf::SCFTilingOptions kOpts;
      kOpts.setLoopType(scf::SCFTilingOptions::LoopType::ForOp);
      kOpts.setTileSizes(
          {rewriter.getI64IntegerAttr(0), rewriter.getI64IntegerAttr(0),
           rewriter.getI64IntegerAttr(kBK)});
      auto kTiled = scf::tileUsingSCF(rewriter,
                                       cast<TilingInterface>(matmul.getOperation()),
                                       kOpts);
      if (failed(kTiled))
        continue;

      // 3b. Tile M/N parallel of the inner op with scf.forall.
      //     After K-tiling, the inner op is in kTiled->tiledOps[0].
      if (kTiled->tiledOps.empty())
        continue;
      scf::SCFTilingOptions mnOpts;
      mnOpts.setLoopType(scf::SCFTilingOptions::LoopType::ForallOp);
      mnOpts.setTileSizes(
          {rewriter.getI64IntegerAttr(kBM), rewriter.getI64IntegerAttr(kBN),
           rewriter.getI64IntegerAttr(0)});
      auto tiledOp = cast<TilingInterface>(kTiled->tiledOps.front());
      auto mnTiled = scf::tileUsingSCF(rewriter, tiledOp, mnOpts);
      if (succeeded(mnTiled) && !mnTiled->tiledOps.empty()) {
        // 3c. Vectorize the innermost matmul tile.
        auto *tiledOp = mnTiled->tiledOps.front();
        auto linalgOp = cast<linalg::LinalgOp>(tiledOp);

        // Check if any operand has dynamic shapes (from affine.min tail).
        bool hasDynamic = false;
        for (auto operand : linalgOp->getOperands()) {
          auto shapedTy = dyn_cast<ShapedType>(operand.getType());
          if (shapedTy && !shapedTy.hasStaticShape()) {
            hasDynamic = true;
            break;
          }
        }

        if (hasDynamic) {
          // Dynamic shapes: inner-tile M and N to VM/VN first, then vectorize
          // with explicit sizes so masking can be generated for the tail.
          auto innerOp = cast<TilingInterface>(tiledOp);

          // Inner M tile: scf.for step VM=4.
          scf::SCFTilingOptions mInnerOpts;
          mInnerOpts.setLoopType(scf::SCFTilingOptions::LoopType::ForOp);
          mInnerOpts.setTileSizes(
              {rewriter.getI64IntegerAttr(kVM), rewriter.getI64IntegerAttr(0),
               rewriter.getI64IntegerAttr(0)});
          auto mInnerTiled = scf::tileUsingSCF(rewriter, innerOp, mInnerOpts);
          if (failed(mInnerTiled) || mInnerTiled->tiledOps.empty())
            continue;

          // Inner N tile: scf.for step VN=8.
          auto innerMOp =
              cast<TilingInterface>(mInnerTiled->tiledOps.front());
          scf::SCFTilingOptions nInnerOpts;
          nInnerOpts.setLoopType(scf::SCFTilingOptions::LoopType::ForOp);
          nInnerOpts.setTileSizes(
              {rewriter.getI64IntegerAttr(0), rewriter.getI64IntegerAttr(kVN),
               rewriter.getI64IntegerAttr(0)});
          auto nInnerTiled =
              scf::tileUsingSCF(rewriter, innerMOp, nInnerOpts);
          if (succeeded(nInnerTiled) && !nInnerTiled->tiledOps.empty()) {
            SmallVector<bool> scalableDims(3, false);
#if LLVM_VERSION_MAJOR >= 21
            auto vecResult = linalg::vectorize(rewriter,
                                               nInnerTiled->tiledOps.front(),
                                               {kVM, kVN, kBK},
                                               scalableDims);
            if (succeeded(vecResult)) {
              rewriter.replaceOp(nInnerTiled->tiledOps.front(),
                                 vecResult->replacements);
            }
#else
            (void)linalg::vectorize(rewriter,
                                    nInnerTiled->tiledOps.front(),
                                    {kVM, kVN, kBK},
                                    scalableDims);
#endif
          }
        } else {
          // Static shapes: vectorize directly (original code path).
#if LLVM_VERSION_MAJOR >= 21
          auto vecResult = linalg::vectorize(rewriter, tiledOp);
          if (succeeded(vecResult)) {
            rewriter.replaceOp(tiledOp, vecResult->replacements);
          }
#else
          (void)linalg::vectorize(rewriter, tiledOp);
#endif
        }
      }
    }

    // Step 4: Tile generic M/N with scf.forall, then inner-tile to VM/VN,
    //         then vectorize.
    for (auto generic : generics) {
      scf::SCFTilingOptions opts;
      opts.setLoopType(scf::SCFTilingOptions::LoopType::ForallOp);
      opts.setTileSizes(
          {rewriter.getI64IntegerAttr(kBM), rewriter.getI64IntegerAttr(kBN)});
      auto tiled = scf::tileUsingSCF(rewriter,
                                      cast<TilingInterface>(generic.getOperation()),
                                      opts);
      if (succeeded(tiled) && !tiled->tiledOps.empty()) {
        auto *tiledOp = tiled->tiledOps.front();
        auto linalgOp = cast<linalg::LinalgOp>(tiledOp);

        // Check if any operand has dynamic shapes.
        bool hasDynamic = false;
        for (auto operand : linalgOp->getOperands()) {
          auto shapedTy = dyn_cast<ShapedType>(operand.getType());
          if (shapedTy && !shapedTy.hasStaticShape()) {
            hasDynamic = true;
            break;
          }
        }

        if (hasDynamic) {
          // Dynamic shapes: inner-tile M and N to VM/VN first.
          auto innerOp = cast<TilingInterface>(tiledOp);

          scf::SCFTilingOptions mInnerOpts;
          mInnerOpts.setLoopType(scf::SCFTilingOptions::LoopType::ForOp);
          mInnerOpts.setTileSizes(
              {rewriter.getI64IntegerAttr(kVM), rewriter.getI64IntegerAttr(0)});
          auto mInnerTiled = scf::tileUsingSCF(rewriter, innerOp, mInnerOpts);
          if (failed(mInnerTiled) || mInnerTiled->tiledOps.empty())
            continue;

          auto innerMOp =
              cast<TilingInterface>(mInnerTiled->tiledOps.front());
          scf::SCFTilingOptions nInnerOpts;
          nInnerOpts.setLoopType(scf::SCFTilingOptions::LoopType::ForOp);
          nInnerOpts.setTileSizes(
              {rewriter.getI64IntegerAttr(0), rewriter.getI64IntegerAttr(kVN)});
          auto nInnerTiled =
              scf::tileUsingSCF(rewriter, innerMOp, nInnerOpts);
          if (succeeded(nInnerTiled) && !nInnerTiled->tiledOps.empty()) {
            SmallVector<bool> scalableDims(2, false);
#if LLVM_VERSION_MAJOR >= 21
            auto vecResult = linalg::vectorize(rewriter,
                                               nInnerTiled->tiledOps.front(),
                                               {kVM, kVN},
                                               scalableDims);
            if (succeeded(vecResult)) {
              rewriter.replaceOp(nInnerTiled->tiledOps.front(),
                                 vecResult->replacements);
            }
#else
            (void)linalg::vectorize(rewriter,
                                    nInnerTiled->tiledOps.front(),
                                    {kVM, kVN},
                                    scalableDims);
#endif
          }
        } else {
          // Static shapes: vectorize directly (original code path).
#if LLVM_VERSION_MAJOR >= 21
          auto vecResult = linalg::vectorize(rewriter, tiledOp);
          if (succeeded(vecResult)) {
            rewriter.replaceOp(tiledOp, vecResult->replacements);
          }
#else
          (void)linalg::vectorize(rewriter, tiledOp);
#endif
        }
      }
    }
  }
};

//===----------------------------------------------------------------------===//
// Pass factory functions
//===----------------------------------------------------------------------===//

namespace mlir {
namespace llk {

std::unique_ptr<Pass> createTileAndVectorizePass() {
  return std::make_unique<TileAndVectorizePass>();
}

std::unique_ptr<Pass> createTileAndVectorizePass(
    const TileAndVectorizeOptions &opts) {
  return std::make_unique<TileAndVectorizePass>(opts);
}

} // namespace llk
} // namespace mlir
