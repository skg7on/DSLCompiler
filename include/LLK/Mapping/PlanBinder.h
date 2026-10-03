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

namespace mlir::llk::mapping {

/// A selected plan materialized onto a private clone of the source module.
struct BoundPlan {
  mlir::OwningOpRef<mlir::ModuleOp> module;
  mlir::Operation *kernel = nullptr;
  PlanId planId = 0;
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
/// Emitting tile copies, allocations, transforms, and waits (design §18.2) is
/// deliberately **not** done here. A connection's endpoints are inferred, not
/// chosen by the plan -- D5 falls back to the executor's first visible memory
/// -- so materializing them would bake an assumption into IR that looks
/// authoritative. The metadata makes the same selection inspectable and
/// verifiable until a plan carries chosen memories.
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
/// deterministic.
llvm::Error verifyMappedMicroIR(mlir::ModuleOp module,
                                const MappingTarget &target);

} // namespace mlir::llk::mapping

#endif // LLK_MAPPING_PLANBINDER_H
