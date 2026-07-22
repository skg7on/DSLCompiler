//===- SerialParallelDispatch.cpp - Serial/parallel dispatch --------------===//
//
// Part of the LLK Compiler Project, under the Apache License v2.0.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Replaces scf.forall with serial scf.for when total_tiles < num_threads.
// Otherwise leaves the scf.forall in place for parallel execution.
//
//===----------------------------------------------------------------------===//

#include "LLK/Transforms/SerialParallelDispatch.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include <thread>

using namespace mlir;

namespace {

struct SerialParallelDispatchPass
    : public PassWrapper<SerialParallelDispatchPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SerialParallelDispatchPass)

  StringRef getArgument() const override { return "serial-parallel-dispatch"; }
  StringRef getDescription() const override {
    return "Convert scf.forall to serial scf.for when tile count is below "
           "threshold";
  }

  void runOnOperation() override {
    int threshold = std::thread::hardware_concurrency();

    SmallVector<scf::ForallOp> toProcess;
    getOperation().walk(
        [&](scf::ForallOp forall) { toProcess.push_back(forall); });

    for (auto forall : toProcess) {
      // Compute total tiles from the upper bound
      int64_t totalTiles = 1;
      bool allStatic = true;
      for (auto ub : forall.getMixedUpperBound()) {
        if (ub.is<Attribute>()) {
          auto intAttr = dyn_cast<IntegerAttr>(ub.get<Attribute>());
          if (intAttr) {
            totalTiles *= intAttr.getInt();
            continue;
          }
        }
        allStatic = false;
        break;
      }

      // Only convert when we can statically determine tiles < threshold
      if (!allStatic || totalTiles >= threshold)
        continue;

      convertToSerialFor(forall, totalTiles);
    }
  }

private:
  void convertToSerialFor(scf::ForallOp forall, int64_t totalTiles) {
    OpBuilder builder(forall);
    Location loc = forall.getLoc();

    Value lb = builder.create<arith::ConstantIndexOp>(loc, 0);
    Value ub = builder.create<arith::ConstantIndexOp>(loc, totalTiles);
    Value step = builder.create<arith::ConstantIndexOp>(loc, 1);

    auto forOp = builder.create<scf::ForOp>(loc, lb, ub, step);

    // Clone body from forall into for
    Block *oldBody = forall.getBody();
    Block *newBody = forOp.getBody();

    IRMapping mapper;
    // Map the forall's induction variable (there is only 1 since we run
    // after linearize-forall) to the for's induction variable
    mapper.map(oldBody->getArgument(0), newBody->getArgument(0));

    builder.setInsertionPointToStart(newBody);

    // Clone all ops except the InParallelOp terminator
    for (auto &op : oldBody->without_terminator()) {
      builder.clone(op, mapper);
    }

    forall.erase();
  }
};

} // namespace

namespace mlir {
namespace llk {

std::unique_ptr<Pass> createSerialParallelDispatchPass() {
  return std::make_unique<SerialParallelDispatchPass>();
}

} // namespace llk
} // namespace mlir
