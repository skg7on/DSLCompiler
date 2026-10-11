//===- MappingLowering.h - The target lowering contract (issue #67, C2) ---===//
//
// Part of the target-independent mapping core (epic #67).
//
// A `MappingTarget` verifies that a selected bundle is complete; this header is
// where the plugin also *lowers* it. Generic mapping code selects a bundle and
// hands the covered operations, the selected plan facts, and a rewriter to the
// target, which owns what that bundle means for its own instruction set.
//
// The separation is the same one §14.3 draws for verification: the generic core
// passes a bundle through as an opaque value and never reads a field as target
// semantics. Only the plugin interprets `name`, `parameters` and `emitterKey`.
//
//===----------------------------------------------------------------------===//

#ifndef LLK_MAPPING_MAPPINGLOWERING_H
#define LLK_MAPPING_MAPPINGLOWERING_H

#include "LLK/Machine/MachineModel.h"
#include "LLK/Mapping/MappingPlan.h"

#include "llvm/ADT/ArrayRef.h"

namespace mlir::llk::mapping {

/// The selected plan facts a target needs while lowering: the machine it is
/// lowering onto, plus the placements and connections the search already
/// committed to.
///
/// A lowerer reads these so it *realizes* the selected physical decisions --
/// which executor, which layout, which route -- rather than re-deriving them.
/// A lowerer that re-decided would be free to emit code for a plan the search
/// never ranked.
struct TargetLoweringContext {
  const machine::MachineModel &machine;
  llvm::ArrayRef<PlanPlacement> placements;
  llvm::ArrayRef<PlanConnection> connections;
  InstanceId instance = 0;
};

} // namespace mlir::llk::mapping

#endif // LLK_MAPPING_MAPPINGLOWERING_H
