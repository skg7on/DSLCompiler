//===- ScratchAnalysis.cpp - Audit memref allocs for intermediates --------===//
//
// Part of the LLK Compiler Project, under the Apache License v2.0.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Post-bufferization pass that walks all memref.alloc ops and warns on
// allocations larger than 1 MB, flagging potential full-size intermediates
// that could be fused away.
//
//===----------------------------------------------------------------------===//

#include "LLK/Transforms/ScratchAnalysis.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinOps.h"

using namespace mlir;

namespace {

struct ScratchAnalysisPass
    : public PassWrapper<ScratchAnalysisPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ScratchAnalysisPass)

  StringRef getArgument() const override { return "scratch-analysis"; }
  StringRef getDescription() const override {
    return "Audit memref allocations for full-size intermediates";
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    module.walk([&](memref::AllocOp alloc) {
      auto memrefType = alloc.getType();
      if (!memrefType.hasStaticShape())
        return;

      int64_t numElements = 1;
      for (int64_t dim : memrefType.getShape())
        numElements *= dim;

      int64_t elementSize = memrefType.getElementTypeBitWidth() / 8;
      int64_t totalBytes = numElements * elementSize;

      // Flag allocations that look like full [M,N] intermediates
      // (threshold: > 1 MB for a single allocation)
      if (totalBytes > 1 * 1024 * 1024) {
        alloc.emitWarning()
            << "Large allocation: " << totalBytes << " bytes (" << numElements
            << " elements of " << memrefType.getElementType()
            << "). Verify this is not a full-size intermediate.";
      }
    });
  }
};

} // namespace

namespace mlir {
namespace llk {
std::unique_ptr<Pass> createScratchAnalysisPass() {
  return std::make_unique<ScratchAnalysisPass>();
}
} // namespace llk
} // namespace mlir
