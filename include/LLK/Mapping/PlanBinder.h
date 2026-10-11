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

/// Owns construction of the concrete operations a selected plan implies (design
/// §18.2, "Materialization versus target readiness").
///
/// The mapping core is target-neutral and links no dialect; a plan's connection
/// is a generic container (a rule id, a route of machine node ids, endpoint
/// occurrences), and only a dialect-aware component can turn it into IR. This
/// interface is that seam: `bindPlan` always persists the selected state as
/// metadata, then -- when a materializer is supplied -- hands the cloned module
/// to it to emit the movements, transforms, and endpoint rewiring the plan
/// selects. The canonical implementation lives outside `lib/Mapping` (in
/// `lib/Conversion/MicroMapping/PlanMaterialization.cpp`), so the mapping
/// library never depends on the Micro dialect.
///
/// A materializer appends one stable reason per connection it cannot
/// materialize to `BoundPlan::unmaterialized` (never silently dropping a
/// selected decision) and rewires exactly the consumer endpoints a connection
/// records -- not every use of a value. Callers that need code a backend may
/// run pass `BindContract::Executable`, which turns the first such reason into
/// an error.
class PlanMaterializer {
public:
  virtual ~PlanMaterializer() = default;

  /// Emits `plan`'s movements and transforms onto `module` (a private clone of
  /// the source kernel already carrying the selected metadata) and rewires the
  /// recorded consumer endpoints. Appends to `bound.unmaterialized` for any
  /// connection it cannot materialize. `module` and `target` are the same
  /// module and target `bindPlan` was called with.
  virtual llvm::Error materialize(mlir::ModuleOp module,
                                  const CoveringPlan &plan,
                                  const MappingTarget &target,
                                  BoundPlan &bound) = 0;
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
/// The concrete operations a selected connection implies -- the copy chains
/// that move a value between memories, the `micro.transform` that re-represents
/// a layout, and the endpoint rewiring -- are constructed by a
/// `PlanMaterializer`, not by this target-neutral function. The canonical
/// implementation lives in the Micro mapping library; `lib/Mapping` never names
/// a dialect operation.
///
/// A connection a materializer cannot materialize is *reported*, not silently
/// dropped: `BoundPlan::unmaterialized` names it and its stable reason. The
/// remaining limit is deliberate: a `Reduce` (a gather) has no Micro operation
/// form yet, so it is reported as `reduce_not_materialized`; every other kind
/// is emitted when a materializer is supplied.
///
/// `contract` decides what an unresolved decision means. Under
/// `BindContract::Partial` (the default) the plan is bound with those
/// connections reported, which is the analysis/reporting contract; under
/// `BindContract::Executable` the same situation is an error naming every
/// unresolved decision, so a caller that will hand the result to a backend
/// cannot receive IR that omits part of the selected plan.
///
/// `materializer` (optional, and null by default) is the canonical
/// construction path for the selected movements and transforms (see
/// `PlanMaterializer`). When it is null the binding is *metadata-only*: the
/// selected state is still persisted, but every connection that would need a
/// new operation is reported in `BoundPlan::unmaterialized` with the reason
/// `no_plan_materializer`, so a standalone mapping caller gets an honest
/// partial plan rather than IR that silently omits the movement. Under
/// `BindContract::Executable` a null materializer is therefore always an error
/// -- a backend-facing caller must supply one.
llvm::Expected<BoundPlan>
bindPlan(mlir::ModuleOp source, const CoveringPlan &plan,
         const MappingTarget &target,
         BindContract contract = BindContract::Partial,
         PlanMaterializer *materializer = nullptr);

/// Verifies the *physical memory facts* of `plan` against `graph` and
/// `machine` (issue #129, task R3): every placement must cover a node of the
/// graph, and the value each output occurrence produces must resolve to exactly
/// one memory by endpoint -- a named rule requirement, the occurrence's own
/// explicit tile memory kind, or the rule's single bare requirement. An
/// occurrence whose kind resolves to no node the executor can address, or to
/// several, is a failure; the message names the missing or ambiguous memory and
/// asks for a named port.
///
/// `BindContract::Executable` calls this, so a backend-facing binding cannot
/// succeed while any materialized value's memory is unstated or ambiguous --
/// the guarantee the storage-skip escape used to defeat. It is deliberately
/// independent of whether a `PlanMaterializer` exists: the verdict is about
/// physical facts, not about a target emitter.
llvm::Error
verifyPlanPhysicalCompleteness(const WorkloadGraph &graph,
                               const CoveringPlan &plan,
                               const machine::MachineModel &machine);

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
