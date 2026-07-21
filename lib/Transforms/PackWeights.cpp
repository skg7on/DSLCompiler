//===- PackWeights.cpp - Weight packing annotation pass -------------------===//
//
// Part of the LLK Compiler Project, under the Apache License v2.0.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Annotates weight operands in Linalg matmul/generic ops with packed layout
// attributes so the downstream TileAndVectorize pass can read them.
//
// Annotation strategy:
//   - For function block arguments: uses func::FuncOp::setArgAttr to annotate
//     the specific argument with "llk.packed_layout" = block_size (BK = 64).
//   - For tensor::EmptyOp producers: annotates the defining op directly with
//     "llk.packed_layout" = block_size.
//   - For linalg::MatmulOp: annotates the RHS (second dps input) operand.
//   - For linalg::GenericOp: annotates operands whose indexing maps contain
//     a reduction iterator.
//
//===----------------------------------------------------------------------===//

#include "LLK/Transforms/PackWeights.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Value.h"

/// Declare isa/cast/dyn_cast for AffineExpr.
using namespace mlir;

// Default block size for weight packing.
static constexpr int64_t kDefaultBK = 64;

namespace {

//===----------------------------------------------------------------------===//
// Helper: annotate a weight value with the packed layout attribute
//===----------------------------------------------------------------------===//

static void annotateWeight(Value weight, MLIRContext *ctx) {
  // If already annotated, skip.
  if (auto blockArg = dyn_cast<BlockArgument>(weight)) {
    auto funcOp = dyn_cast<func::FuncOp>(blockArg.getOwner()->getParentOp());
    if (!funcOp)
      return;
    unsigned argNo = blockArg.getArgNumber();
    if (funcOp.getArgAttr(argNo, "llk.packed_layout"))
      return;
    funcOp.setArgAttr(argNo, "llk.packed_layout",
                      IntegerAttr::get(IntegerType::get(ctx, 64), kDefaultBK));
    return;
  }

  Operation *defOp = weight.getDefiningOp();
  if (!defOp)
    return;

  // Only annotate tensors produced by tensor::EmptyOp (weight initializers).
  if (!isa<tensor::EmptyOp>(defOp))
    return;

  if (defOp->hasAttr("llk.packed_layout"))
    return;
  defOp->setAttr("llk.packed_layout",
                 IntegerAttr::get(IntegerType::get(ctx, 64), kDefaultBK));
}

//===----------------------------------------------------------------------===//
// Helper: check if a linalg::GenericOp operand has a reduction dimension
//===----------------------------------------------------------------------===//

static bool hasReductionDim(linalg::GenericOp genericOp, unsigned operandIdx) {
  auto maps = genericOp.getIndexingMapsArray();
  if (operandIdx >= maps.size())
    return false;
  AffineMap map = maps[operandIdx];
  auto iterators = genericOp.getIteratorTypesArray();
  for (unsigned d = 0; d < map.getNumResults(); ++d) {
    auto dimExpr = dyn_cast<AffineDimExpr>(map.getResult(d));
    if (!dimExpr)
      continue;
    unsigned dimPos = dimExpr.getPosition();
    if (dimPos < iterators.size() &&
        iterators[dimPos] == utils::IteratorType::reduction) {
      return true;
    }
  }
  return false;
}

//===----------------------------------------------------------------------===//
// PackWeightsPass
//===----------------------------------------------------------------------===//

struct PackWeightsPass
    : public PassWrapper<PackWeightsPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PackWeightsPass)

  StringRef getArgument() const override { return "pack-weights"; }
  StringRef getDescription() const override {
    return "Annotate weight operands with packed layout attributes";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect>();
    registry.insert<tensor::TensorDialect>();
    registry.insert<func::FuncDialect>();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();

    // Walk all linalg::MatmulOp: annotate the RHS (weight) operand.
    module.walk([&](linalg::MatmulOp matmul) {
      // Operand 0 = X (activation), Operand 1 = W (weight)
      if (matmul.getNumDpsInputs() >= 2) {
        Value weight = matmul.getDpsInputOperand(1)->get();
        annotateWeight(weight, &getContext());
      }
    });

    // Walk all linalg::GenericOp: annotate operands with reduction dims.
    module.walk([&](linalg::GenericOp genericOp) {
      for (unsigned i = 0; i < genericOp.getNumDpsInputs(); ++i) {
        if (hasReductionDim(genericOp, i)) {
          Value operand = genericOp.getDpsInputOperand(i)->get();
          annotateWeight(operand, &getContext());
        }
      }
    });
  }
};

} // namespace

namespace mlir {
namespace llk {
std::unique_ptr<Pass> createPackWeightsPass() {
  return std::make_unique<PackWeightsPass>();
}
} // namespace llk
} // namespace mlir
