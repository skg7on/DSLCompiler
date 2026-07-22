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

#include "LLK/Dialect/LLKEnums.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"
#include "mlir/Interfaces/TilingInterface.h"
#include "mlir/Pass/Pass.h"

// Attribute class declarations (needed before LLKOps.h.inc).
#define GET_ATTRDEF_CLASSES
#include "LLK/Dialect/LLKAttributes.h.inc"

// Access the generated op interfaces.
#define GET_OP_CLASSES
#include "LLK/Dialect/LLKOps.h.inc"

using namespace mlir;

// ---------------------------------------------------------------------------
// M bucket classifier
// ---------------------------------------------------------------------------

static int classifyM(int64_t M) {
  if (M == 1)
    return 0;
  if (M <= 4)
    return 1;
  if (M <= 16)
    return 2;
  if (M <= 64)
    return 3;
  return 4;
}

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
    getOperation().walk([&](llk::FusedSwiGLUOp op) {
      auto xType = op.getX().getType();
      if (!xType.hasStaticShape()) {
        op.emitRemark() << "dynamic M → bucket -1 (generic)";
        return;
      }
      int64_t M = xType.getDimSize(0);
      int bucket = classifyM(M);
      op.emitRemark() << "M=" << M << " -> bucket " << bucket;
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
