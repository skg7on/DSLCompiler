//===- LinearizeForall.cpp - Multi-D scf.forall to 1D -------------------===//
//
// Part of the LLK Compiler Project, under the Apache License v2.0.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "LLK/Transforms/LinearizeForall.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
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

    // Compute total_tiles = product of all upper bounds
    Value totalTiles = getValueOrCreateConstantIndex(builder, loc, ub[0]);
    for (int64_t i = 1; i < numDims; i++) {
      Value ubVal = getValueOrCreateConstantIndex(builder, loc, ub[i]);
      totalTiles = builder.create<arith::MulIOp>(loc, totalTiles, ubVal);
    }

    // Compute strides for each dimension (product of extents of dims to the
    // right)
    SmallVector<int64_t> strides(numDims, 1);
    for (int64_t i = numDims - 2; i >= 0; i--) {
      OpFoldResult nextUb = ub[i + 1];
      if (nextUb.is<Attribute>()) {
        auto intAttr = dyn_cast<IntegerAttr>(nextUb.get<Attribute>());
        if (intAttr) {
          strides[i] = strides[i + 1] * intAttr.getInt();
          continue;
        }
      }
      strides[i] = -1;
    }

    // Create 1D forall using builder
    auto newForall = builder.create<scf::ForallOp>(
        loc, SmallVector<OpFoldResult>{totalTiles}, forall.getOutputs(),
        forall.getMapping());

    // Populate the new body
    Block *newBody = newForall.getBody();
    builder.setInsertionPointToStart(newBody);
    Value tid = newBody->getArgument(0);

    // Build IRMapping: old block args → computed IVs
    Block *oldBody = forall.getBody();
    IRMapping mapper;

    Value remaining = tid;
    for (int64_t i = 0; i < numDims; i++) {
      Value iv;
      Value dimSize = getValueOrCreateConstantIndex(builder, loc, ub[i]);

      if (i == numDims - 1) {
        iv = builder.create<arith::RemSIOp>(loc, remaining, dimSize);
      } else if (strides[i] > 1) {
        Value strideVal =
            builder.create<arith::ConstantIndexOp>(loc, strides[i]);
        Value divided =
            builder.create<arith::DivSIOp>(loc, remaining, strideVal);
        iv = builder.create<arith::RemSIOp>(loc, divided, dimSize);
      } else {
        iv = builder.create<arith::RemSIOp>(loc, remaining, dimSize);
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

  Value getValueOrCreateConstantIndex(OpBuilder &builder, Location loc,
                                      OpFoldResult ofr) {
    if (ofr.is<Value>())
      return ofr.get<Value>();
    int64_t constVal = cast<IntegerAttr>(ofr.get<Attribute>()).getInt();
    return builder.create<arith::ConstantIndexOp>(loc, constVal);
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
