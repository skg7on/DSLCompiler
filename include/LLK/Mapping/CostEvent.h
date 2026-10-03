//===- CostEvent.h - Shared cost-event vocabulary -------------------------===//
//
// Part of the target-independent mapping core (epic #67).
//
// The mapping search and the performance evaluator account for the same kinds
// of machine work, so they have to name them the same way (design §17.2). A
// route the mapper chose and a hop the simulator charges are only comparable
// if both call it a transfer hop.
//
// These five categories are the whole vocabulary. They are deliberately coarse
// -- what the work *is*, not how much of it -- because cost stays
// multi-dimensional until an objective ranks it.
//
//===----------------------------------------------------------------------===//

#ifndef LLK_MAPPING_COSTEVENT_H
#define LLK_MAPPING_COSTEVENT_H

#include "LLK/Mapping/CostModel.h"

#include "llvm/ADT/StringRef.h"

#include <optional>
#include <string>

namespace mlir::llk::mapping {

/// The normalized categories of machine work both layers account for.
enum class CostEventKind {
  Compute,         ///< rule-local work issued on a compute capability
  TransferHop,     ///< one memory-to-memory hop over a link
  Transform,       ///< a layout conversion
  Synchronization, ///< a wait or a barrier
  Capacity         ///< a resource-occupancy charge
};

llvm::StringRef stringifyCostEventKind(CostEventKind kind);
std::optional<CostEventKind> symbolizeCostEventKind(llvm::StringRef text);

/// One normalized cost event: what kind of work, on which machine resource,
/// for how much. The mapping search emits these from a selected plan; the
/// evaluator emits them from a scheduled kernel.
struct CostEvent {
  CostEventKind kind = CostEventKind::Compute;
  /// The machine resource the work occupies, named the way the machine model
  /// names it (`mxu`, `dma.0`, an executor id).
  std::string resource;
  Cost cost;
};

} // namespace mlir::llk::mapping

#endif // LLK_MAPPING_COSTEVENT_H
