//===- GridToForall.cpp - tt.get_program_id → scf.forall -----------------===//
//
// Lowers Triton grid dispatch to scf.forall parallel tile decomposition.
// 1D/2D/3D grids linearized into a single forall matching the existing
// M4 parallel infrastructure.
//
//===----------------------------------------------------------------------===//

#include "LLK/Conversion/TritonToLLK/GridToForall.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/Debug.h"

using namespace mlir;

namespace {

//===----------------------------------------------------------------------===//
// Pass: replace tt.get_program_id with scf.forall induction variable
//===----------------------------------------------------------------------===//

struct TritonGridToForallPass
    : public PassWrapper<TritonGridToForallPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TritonGridToForallPass)

  StringRef getArgument() const override { return "triton-grid-to-forall"; }
  StringRef getDescription() const override {
    return "Lower Triton tt.get_program_id grid dispatch to scf.forall";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect>();
    registry.insert<scf::SCFDialect>();
  }

  void runOnOperation() override;
};

void TritonGridToForallPass::runOnOperation() {
  ModuleOp module = getOperation();

  module.walk([&](func::FuncOp funcOp) {
    // Collect all tt.get_program_id ops in this function.
    SmallVector<Operation *> pidOps;
    funcOp.walk([&](Operation *op) {
      if (op->getName().getStringRef() == "tt.get_program_id")
        pidOps.push_back(op);
    });

    if (pidOps.empty())
      return WalkResult::advance();

    // Determine grid dimensionality from the max axis attribute.
    int maxAxis = -1;
    for (auto *op : pidOps) {
      if (auto axisAttr = op->getAttrOfType<IntegerAttr>("axis"))
        maxAxis = std::max(maxAxis, (int)axisAttr.getInt());
    }
    int gridDims = maxAxis + 1;

    // Build tile count and forall at the beginning of the function body.
    OpBuilder builder(funcOp);
    builder.setInsertionPointToStart(&funcOp.getBody().front());
    Location loc = funcOp.getLoc();

    // Convention: function arguments encode grid dimensions.
    // args[0 .. gridDims-1] = dimension sizes (M, N, ...)
    // args[gridDims .. 2*gridDims-1] = block sizes (BM, BN, ...)
    auto args = funcOp.getArguments();
    if ((int)args.size() < 2 * gridDims) {
      funcOp.emitWarning(
          "triton-grid-to-forall: not enough function arguments for "
          "a ")
          << gridDims << "D grid";
      return WalkResult::advance();
    }

    // Compute num_tiles for each axis: ceilDiv(dim, block).
    SmallVector<Value> numTilesPerAxis;
    for (int i = 0; i < gridDims; ++i) {
      Value dim = args[i];
      Value block = args[gridDims + i];
      Value numTiles = arith::CeilDivSIOp::create(builder, loc, dim, block);
      numTilesPerAxis.push_back(numTiles);
    }

    // Compute total tiles = product of all per-axis tile counts.
    Value totalTiles = numTilesPerAxis[0];
    for (int i = 1; i < gridDims; ++i) {
      totalTiles =
          arith::MulIOp::create(builder, loc, totalTiles, numTilesPerAxis[i]);
    }

    // Create 1D scf.forall over total tiles.
    auto forallOp = scf::ForallOp::create(
        builder, loc, SmallVector<OpFoldResult>{totalTiles},
        /*outputs=*/ValueRange{}, /*mapping=*/std::nullopt);

    // Inside the forall body, compute per-axis tile indices from the linear
    // thread ID and replace the tt.get_program_id results.
    builder.setInsertionPointToStart(forallOp.getBody());
    Value tid = forallOp.getBody()->getArgument(0);

    // Decompose linear tid into per-axis tile indices.
    // For 2+D: idx[i] = (tid / stride[i]) % numTiles[i]
    //   where stride[i] = product(numTiles[j]) for j > i
    SmallVector<Value> tileIndices;
    if (gridDims == 1) {
      tileIndices.push_back(tid);
    } else {
      // Compute strides from right to left.
      SmallVector<Value> strides(gridDims);
      strides[gridDims - 1] = arith::ConstantIndexOp::create(builder, loc, 1);
      for (int i = gridDims - 2; i >= 0; --i) {
        strides[i] = arith::MulIOp::create(builder, loc, strides[i + 1],
                                           numTilesPerAxis[i + 1]);
      }
      for (int i = 0; i < gridDims; ++i) {
        Value divided = arith::DivSIOp::create(builder, loc, tid, strides[i]);
        Value idx =
            arith::RemSIOp::create(builder, loc, divided, numTilesPerAxis[i]);
        tileIndices.push_back(idx);
      }
    }

    // Replace tt.get_program_id results with the computed tile indices.
    for (auto *op : pidOps) {
      int axis = (int)op->getAttrOfType<IntegerAttr>("axis").getInt();
      op->getResult(0).replaceAllUsesWith(tileIndices[axis]);
      op->erase();
    }

    return WalkResult::advance();
  });
}

} // namespace

std::unique_ptr<Pass> mlir::llk::createTritonGridToForallPass() {
  return std::make_unique<TritonGridToForallPass>();
}
