//===- LinearizeForall.cpp - Multi-D scf.forall to 1D -------------------===//
//
// Part of the LLK Compiler Project, under the Apache License v2.0.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "LLK/Transforms/LinearizeForall.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;

namespace {

struct LinearizeForallPass
    : public PassWrapper<LinearizeForallPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LinearizeForallPass)

  StringRef getArgument() const override { return "linearize-forall"; }
  StringRef getDescription() const override {
    return "Linearize multi-dimensional scf.forall to 1D for static chunking";
  }

  void runOnOperation() override {
    SmallVector<scf::ForallOp> toProcess;
    getOperation().walk(
        [&](scf::ForallOp forall) { toProcess.push_back(forall); });

    for (auto forall : toProcess) {
      if (forall.getRank() <= 1)
        continue;

      linearize(forall);
    }
  }

private:
  void linearize(scf::ForallOp forall) {
    OpBuilder builder(forall);
    Location loc = forall.getLoc();

    SmallVector<OpFoldResult> ub = forall.getMixedUpperBound();
    int64_t numDims = forall.getRank();

    // This pass requires static upper bounds. The tiling pipeline always
    // produces constant tile counts, so dynamic bounds indicate a pipeline
    // ordering issue.
    for (int64_t i = 0; i < numDims; i++) {
      assert(ub[i].is<Attribute>() &&
             isa<IntegerAttr>(ub[i].get<Attribute>()) &&
             "linearize-forall requires static upper bounds; "
             "run after tiling with constant tile sizes");
    }

    // Compute total_tiles = product of all upper bounds
    // NOTE: Overflow possible for extreme tile counts (e.g., >2^31 each).
    // In practice tile sizes are small (e.g., BM=32, BN=64) so this is safe.
    Value totalTiles = getValueOrCreateConstantIndexOp(builder, loc, ub[0]);
    for (int64_t i = 1; i < numDims; i++) {
      Value ubVal = getValueOrCreateConstantIndexOp(builder, loc, ub[i]);
      totalTiles = arith::MulIOp::create(builder, loc, totalTiles, ubVal);
    }

    // Compute strides: stride[i] = product of ub[j] for j > i
    SmallVector<int64_t> strides(numDims, 1);
    for (int64_t i = numDims - 2; i >= 0; i--) {
      auto intAttr = cast<IntegerAttr>(ub[i + 1].get<Attribute>());
      strides[i] = strides[i + 1] * intAttr.getInt();
    }

    // Create 1D forall using builder
    auto newForall = scf::ForallOp::create(
        builder, loc, SmallVector<OpFoldResult>{totalTiles},
        forall.getOutputs(), forall.getMapping());

    // Populate the new body
    Block *newBody = newForall.getBody();
    builder.setInsertionPointToStart(newBody);
    Value tid = newBody->getArgument(0);

    // Build IRMapping: old block args → computed IVs
    Block *oldBody = forall.getBody();
    IRMapping mapper;

    for (int64_t i = 0; i < numDims; i++) {
      Value iv;
      Value dimSize = getValueOrCreateConstantIndexOp(builder, loc, ub[i]);

      if (i == numDims - 1) {
        iv = arith::RemSIOp::create(builder, loc, tid, dimSize);
      } else {
        Value strideVal =
            arith::ConstantIndexOp::create(builder, loc, strides[i]);
        Value divided = arith::DivSIOp::create(builder, loc, tid, strideVal);
        iv = arith::RemSIOp::create(builder, loc, divided, dimSize);
      }
      mapper.map(oldBody->getArgument(i), iv);
    }

    // Map output block arguments (shared_outs)
    for (unsigned i = 0; i < forall.getOutputs().size(); i++) {
      mapper.map(oldBody->getArgument(numDims + i),
                 newBody->getArgument(1 + i));
    }

    // Clone operations from old body (except the InParallelOp terminator)
    // into the new body. The new forall already has its own terminator.
    for (auto &op : oldBody->without_terminator()) {
      builder.clone(op, mapper);
    }

    // Replace results
    for (unsigned i = 0; i < forall.getNumResults(); i++) {
      forall.getResult(i).replaceAllUsesWith(newForall.getResult(i));
    }

    forall.erase();
  }
};

} // namespace

namespace mlir {
namespace llk {

std::unique_ptr<Pass> createLinearizeForallPass() {
  return std::make_unique<LinearizeForallPass>();
}

} // namespace llk
} // namespace mlir
