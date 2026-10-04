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

/// How strict binding is about execution-affecting decisions the binder cannot
/// yet materialize (design §18.2). The distinction is the difference between a
/// useful *analysis* artifact and something a backend may execute: a partial
/// plan is honest about what it left out, but must not be mistaken for
/// executable code.
enum class BindContract {
  /// Bind every decision that can be materialized and report the rest in
  /// `BoundPlan::unmaterialized`. The result is a partial plan, for analysis,
  /// reporting, and the report-only workflow. This is the default.
  Partial,
  /// Refuse a plan that is not fully executable: any unresolved
  /// execution-affecting decision is an error naming the decisions, so a caller
  /// that will hand the result to a backend cannot silently receive IR that
  /// omits part of the selected plan.
  Executable,
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
/// between two memories is emitted as one `micro.async_copy` + `micro.wait` per
/// route hop, right after the producing operation. One chain is emitted per
/// (value, route), and each chain rewires only the consumers its connection
/// names -- so a value carried two ways gets two chains, and neither redirects
/// the other's readers. A `Replicate` (a fan-out copy) is materialized the same
/// way; a connection whose consumers are not recorded rewires nobody.
///
/// A layout conversion becomes one target-neutral `micro.transform`, which
/// names the two layouts by their solved affine maps rather than by target ids
/// (design §13.4 keeps target layout ids out of `#micro.layout`): a
/// `LayoutTransform` connection emits it in place, and a
/// `TransferAndTransform` connection emits its copies followed by it. Only
/// `Reduce` (a gather) still has no Micro operation form and is reported.
///
/// A connection the binder cannot materialize is *reported*, not silently
/// dropped: `BoundPlan::unmaterialized` names it and why. The remaining limit
/// is deliberate: a `Reduce` (a gather) has no Micro operation form yet, so it
/// is reported as `reduce_not_materialized`; every other connection kind is
/// emitted. (A value whose type is not shaped has no generic copy form either
/// -- `micro.tile_async_copy` needs a destination-memory-typed tile the binder
/// cannot construct -- and such an entry names its own specific cause.)
///
/// `contract` decides what an unresolved decision means. Under
/// `BindContract::Partial` (the default) the plan is bound with those
/// connections reported, which is the analysis/reporting contract; under
/// `BindContract::Executable` the same situation is an error naming every
/// unresolved decision, so a caller that will hand the result to a backend
/// cannot receive IR that omits part of the selected plan.
llvm::Expected<BoundPlan>
bindPlan(mlir::ModuleOp source, const CoveringPlan &plan,
         const MappingTarget &target,
         BindContract contract = BindContract::Partial);

/// Layered verification of mapped Micro-IR (design §18.3), in order:
///
///   1. structural -- the module passes the dialect verifier, and every generic
///      metadata container (`micro.plan`, `micro.mapping`, `micro.routes`) has
///      the shape it claims: a container of the wrong kind, or an entry of the
///      wrong type, is a stable diagnostic rather than an unchecked cast that
///      aborts the process (design §25.1);
///   2. machine-aware -- every kernel is *mapped* (carries `micro.plan` and
///      annotates each workload operation that needs a rule), and every rule,
///      executor, memory, layout, route node, transfer engine, and link named
///      by the metadata resolves in the target, with each memory visible from
///      the executor it is bound to, the selected rule implementing the
///      operation it is recorded on, and every role the rule requires bound;
///   3. target -- the recorded bundle and emitter match the selected rule, the
///      emitter is one the target declares, and the target's own emitter
///      accepts the bundle before lowering.
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
