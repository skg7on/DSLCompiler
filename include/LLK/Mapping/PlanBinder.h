//===- PlanBinder.h - Materialize a selected plan (D7/#50) ---------------===//
//
// Part of the target-independent mapping core (epic #67, #50 revision).
//
// The binder takes a selected `CoveringPlan` and writes it onto a private
// clone of the source kernel, so the search-space operation and the original
// module are never mutated (design §18).
//
// What it writes is metadata, not new operations: generic, target-neutral
// containers (design §18.1) naming the selected rule, bundle, emitter,
// executor, memories, layouts, and routes. The dialect verifier still knows
// nothing about what any of those ids mean -- resolving them is the
// machine-aware phase below, and interpreting a bundle is the target
// emitter's job.
//
//===----------------------------------------------------------------------===//

#ifndef LLK_MAPPING_PLANBINDER_H
#define LLK_MAPPING_PLANBINDER_H

#include "LLK/Mapping/MappingPlan.h"
#include "LLK/Mapping/MappingTarget.h"

#include "mlir/IR/BuiltinOps.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <string>
#include <vector>

namespace mlir::llk::mapping {

/// A selected plan materialized onto a private clone of the source module.
struct BoundPlan {
  mlir::OwningOpRef<mlir::ModuleOp> module;
  mlir::Operation *kernel = nullptr;
  PlanId planId = 0;
  /// Connections the binder could not materialize, with the reason. Empty on a
  /// fully materialized plan.
  std::vector<std::string> unmaterialized;
};

/// Clones `source`, extracts its workload graph, and writes the plan's
/// selections onto the clone: a `micro.plan` dictionary on the kernel (plan
/// id, binding hash, truncation) and `micro.mapping` on each covered
/// operation (rule, bundle, emitter, executor, memories, layouts), plus
/// `micro.routes` for the selected connections.
///
/// Fails when the module has no `micro.kernel`, when a covered node is not in
/// the kernel, or when the plan names a rule the target does not declare.
///
/// Materialization (design §18.2): a selected connection that moves a value
/// between two memories is emitted as `micro.async_copy` followed by
/// `micro.wait` right after the producing operation, and every other use of
/// the original value is rewired to the copy -- so the value the consumer sees
/// is the one that lives in its memory.
///
/// A connection the binder cannot materialize is *reported*, not silently
/// dropped: `BoundPlan::unmaterialized` names it and why. The current limits
/// are deliberate and documented in the implementation: a value whose type is
/// not a shaped (tensor) type has no generic copy form (`micro.tile_async_copy`
/// needs a destination-memory-typed tile, which cannot be built without the
/// dialect's type class), and a target layout id has no Micro operation form at
/// all (design §13.4 keeps target layout ids out of `#micro.layout`). Every
/// reported entry names the connection's value id; when the binder cannot
/// materialize a whole connection kind, the entry also carries a stable reason
/// token (a `LayoutTransform` connection is reported as
/// `layout_transform_requires_dialect_op`), while a movement that fails reports
/// its own specific cause.
llvm::Expected<BoundPlan> bindPlan(mlir::ModuleOp source,
                                   const CoveringPlan &plan,
                                   const MappingTarget &target);

/// Layered verification of mapped Micro-IR (design §18.3), in order:
///
///   1. structural -- the module passes the dialect verifier;
///   2. machine-aware -- every rule, executor, memory, layout, route node, and
///      link named by the metadata resolves in the target, and each memory is
///      visible from the executor it is bound to;
///   3. target -- every selected rule's emitter is one the target declares.
///
/// Returns the first violation, walking operations in order so diagnostics are
/// deterministic. A violation that a §22.3 code covers is reported as
/// `<code>: <detail>` (for example `no_matching_rule: mapped op ...: unknown
/// rule '...'`), so the code is the stable interface and the detail is prose;
/// the structural failure has no §22.3 code of its own and is reported plainly.
llvm::Error verifyMappedMicroIR(mlir::ModuleOp module,
                                const MappingTarget &target);

} // namespace mlir::llk::mapping

#endif // LLK_MAPPING_PLANBINDER_H
