//===- ShapeSpecialization.cpp - M dimension bucketing pass ---------------===//
//
// Classifies the M dimension of llk.fused_swiglu operations into one of
// 5 buckets used by the schedule selection system.
//
// M buckets:
//   Bucket 0: M = 1           → GEMV-like
//   Bucket 1: M ∈ [2, 4]      → Micro-GEMM
//   Bucket 2: M ∈ [5, 16]     → Small tile
//   Bucket 3: M ∈ [17, 64]    → Medium tile
//   Bucket 4: M ≥ 65          → Large tile
//
//===----------------------------------------------------------------------===//

#include "LLK/Transforms/ShapeSpecialization.h"

#include "LLK/Conversion/LLKKernel/LLKKernelLowering.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"
#include "mlir/Interfaces/TilingInterface.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;

// ---------------------------------------------------------------------------
// M bucket classifier
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// ShapeSpecializationPass
// ---------------------------------------------------------------------------

namespace {

struct ShapeSpecializationPass
    : public PassWrapper<ShapeSpecializationPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ShapeSpecializationPass)

  StringRef getArgument() const override { return "shape-specialize"; }
  StringRef getDescription() const override {
    return "Classify M dimension into buckets for schedule selection";
  }

  void runOnOperation() override {
    llk::LLKKernelRegistry registry;
    llk::populateLLKKernelLowerings(registry);
    getOperation().walk([&](Operation *op) {
      if (op->getName().getDialectNamespace() != "llk")
        return;
      auto *entry = registry.lookup(op->getName().getStringRef());
      if (!entry)
        return;
      llk::KernelContract contract;
      if (failed(entry->inferContract(op, contract)))
        return;
      if (contract.mBucket < 0) {
        op->emitRemark() << "dynamic M → bucket -1 (generic)";
        return;
      }
      op->emitRemark() << "M=" << contract.M << " -> bucket "
                       << contract.mBucket;
    });
  }
};

} // namespace

namespace mlir {
namespace llk {

std::unique_ptr<Pass> createShapeSpecializationPass() {
  return std::make_unique<ShapeSpecializationPass>();
}

} // namespace llk
} // namespace mlir
