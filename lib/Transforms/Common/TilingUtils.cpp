//===- TilingUtils.cpp - Generic tile-size computation --------------------===//
//
// Common utilities for computing tile sizes and constructing loop nests
// from schedule parameters.  Provides hasDynamicOperand for tiling-pass
// dispatch decisions and resolveTileParams for mapping schedule entries
// to concrete tile dimensions.
//
//===----------------------------------------------------------------------===//

#include "LLK/Transforms/Common/TilingUtils.h"
#include "LLK/Transforms/Common/ScheduleLoader.h"

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"

using namespace mlir;

namespace mlir {
namespace llk {

bool hasDynamicOperand(Operation *op) {
  for (auto operand : op->getOperands()) {
    auto shapedTy = dyn_cast<ShapedType>(operand.getType());
    if (shapedTy && !shapedTy.hasStaticShape())
      return true;
  }
  return false;
}

TileParams resolveTileParams(const ScheduleEntry &entry, int vectorWidth) {
  TileParams params;
  params.BM = entry.BM;
  params.BN = entry.BN;
  params.BK = entry.BK;

  // If the schedule entry doesn't specify tile sizes, compute defaults
  // from the vector width.  This mirrors the ISA-aware defaults used by
  // TileAndVectorize.
  if (params.BM == 0 && vectorWidth > 0)
    params.BM = vectorWidth * 8; // ~8 rows per vector tile
  if (params.BN == 0 && vectorWidth > 0)
    params.BN = vectorWidth * 16; // ~16 columns per vector tile
  if (params.BK == 0 && vectorWidth > 0)
    params.BK = vectorWidth * 16; // K-tile proportional to vector width

  return params;
}

} // namespace llk
} // namespace mlir
