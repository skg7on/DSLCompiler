//===- Placement.h - Placement and connection synthesis (D5) -------------===//
//
// Part of the target-independent mapping core (epic #67, workstream D5).
//
// Placement turns an unplaced `MappingCandidate` into legal `CandidateInstance`
// objects on concrete machine resources: which executor runs it, which compute
// capability and memory it attaches to, and which layouts its ports satisfy
// (design §15.1). A candidate that cannot be placed yields no instances -- it
// never fails the whole search, because another candidate for the same node
// may still be legal.
//
// Everything is target-independent. Executor requirements are matched by
// abstract owner kind (design §11.5); a rule never names a concrete executor.
//
//===----------------------------------------------------------------------===//

#ifndef LLK_MAPPING_PLACEMENT_H
#define LLK_MAPPING_PLACEMENT_H

#include "LLK/Machine/MachineModel.h"
#include "LLK/Mapping/LlkMap.h"
#include "LLK/Mapping/MappingPlan.h"
#include "LLK/Mapping/MappingTarget.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <memory>
#include <vector>

namespace mlir {
class MLIRContext;
} // namespace mlir

namespace mlir::llk::mapping {

struct PlacementOptions {
  /// Collapse interchangeable executors to one representative (design §15.1).
  /// Two executors are interchangeable only when swapping them cannot change a
  /// binding: same kind, same parent, same attached compute and memory nodes.
  bool reduceSymmetry = true;
  /// Upper bound on the instances one candidate may produce.
  unsigned maxInstances = 64;
  /// Upper bound on the routes one connection may consider.
  unsigned maxRoutesPerConnection = 4;
};

/// Enumerates legal placements of `candidate` on `target`, in machine
/// declaration order. Returns an empty vector when the candidate cannot be
/// placed; fails only on a malformed input, such as a layout requirement that
/// names a layout the target does not declare.
llvm::Expected<std::vector<CandidateInstance>>
enumeratePlacements(const MappingCandidate &candidate,
                    const MappingTarget &target, mlir::MLIRContext &context,
                    const LayoutContext &layoutContext,
                    const PlacementOptions &options = {});

} // namespace mlir::llk::mapping

#endif // LLK_MAPPING_PLACEMENT_H
