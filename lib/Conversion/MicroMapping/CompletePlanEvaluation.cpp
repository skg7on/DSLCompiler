//===- CompletePlanEvaluation.cpp - Evaluate a complete proposal (R7) -----===//
//
// See CompletePlanEvaluation.h. The evaluation is the one place a search
// proposal becomes a physical, schedulable plan, and it is deliberately built
// only from functions the planner and the performance model already share
// (`finalizeStoragePlan`, `bindPlan`, `analyzeSelectedKernel`) so a plan's
// feasibility and its predicted cost are one analysis of one kernel.
//
//===----------------------------------------------------------------------===//

#include "CompletePlanEvaluation.h"

#include "MicroMappingCommon.h"

#include "LLK/Mapping/CostEvent.h"
#include "LLK/Mapping/MappingMetadata.h"
#include "LLK/Mapping/StoragePlan.h"

#include "LLK/Perf/SelectedKernelAnalysis.h"

#include "llvm/Support/Error.h"

#include <memory>
#include <string>
#include <utility>

namespace mlir::llk::mapping {

namespace {

/// The bounded number of finalize passes one evaluation runs before it declares
/// the choice function a cycle. In practice the first pass decides every
/// durable choice and the second only *reproduces* it (the R4 idempotence
/// property, preserved by the reuse ordering R5 merges into the step DAG), so
/// the bound is never reached on a well-formed plan.
constexpr unsigned kMaxEvaluationPasses = 4;

CompletePlanEvaluation rejected(DiagnosticCode code, std::string message) {
  CompletePlanEvaluation evaluation;
  evaluation.rejection = Diagnostic{code, std::move(message)};
  return evaluation;
}

/// Records the plan's provenance: the content hashes a reader requires before a
/// plan may be replayed or bound, and the search binding it came from. None of
/// these enters the decision-only v3 id (`canonicalPlanString` folds only the
/// decisions), so setting them cannot move an id.
void recordProvenance(CoveringPlan &plan, const WorkloadGraph &graph,
                      const MappingTarget &target) {
  plan.graphHash = computeSourceGraphHash(graph);
  plan.targetHash = computeTargetContentHash(target);
  plan.machineHash = machine::computeContentHash(target.machine());
  plan.layoutHash = target.layouts().computeContentHash();
  plan.ruleHash = target.rules().computeContentHash();
}

/// Compares the analysis's per-memory live peak against each memory's capacity.
/// `peakBytes` is keyed by memory *node id* when the plan's R5 liveness ran; a
/// key the machine does not model (the extraction's kind-keyed fallback) is
/// skipped here, because a capacity verdict is only meaningful for a modelled
/// node. Returns the first overflow, or an empty string.
std::string occupancyOverflow(const perf::SelectedKernelAnalysis &analysis,
                              const machine::MachineModel &machine) {
  for (const auto &entry : analysis.peakBytes) {
    const machine::MemoryNode *memory = machine.findMemory(entry.first);
    if (!memory)
      continue;
    if (entry.second > memory->capacityBytes)
      return "memory '" + entry.first + "' requires " +
             std::to_string(entry.second) + " bytes live but machine has " +
             std::to_string(memory->capacityBytes) + " bytes";
  }
  return {};
}

/// The fallback score used when the selected static analysis cannot run: the
/// plan's own synthesized events, scheduled by the shared scheduler. This is
/// the score every plan had before the analysis existed, so a modelling gap
/// leaves the plan scored rather than unscored, and the reason is reported.
void scoreFromOwnEvents(CoveringPlan &plan,
                        const machine::MachineModel &machine) {
  llvm::Expected<PlanEventDAG> events = buildPlanEvents(plan, machine);
  if (!events) {
    plan.diagnostics.warnings.push_back(
        "plan score: accumulation reported, not scheduled (the plan's events "
        "could not be built: " +
        llvm::toString(events.takeError()) + ")");
    return;
  }
  llvm::Expected<Cost> scheduled = schedulePlanEvents(*events, machine);
  if (!scheduled) {
    plan.diagnostics.warnings.push_back(
        "plan score: accumulation reported, not scheduled (the shared schedule "
        "failed: " +
        llvm::toString(scheduled.takeError()) + ")");
    return;
  }
  plan.totalCost = *scheduled;
  plan.scoreSource = PlanScoreSource::Schedule;
}

} // namespace

llvm::Expected<CompletePlanEvaluation>
evaluateCompletePlan(mlir::ModuleOp source, const WorkloadGraph &graph,
                     const MappingTarget &target, const CoveringPlan &proposal,
                     BindContract contract) {
  const machine::MachineModel &machine = target.machine();
  const bool strict = contract == BindContract::Executable;

  // --- 1. clone, provenance, contract -------------------------------------
  CoveringPlan plan = proposal;
  recordProvenance(plan, graph, target);

  // --- 2. finalize physical storage ---------------------------------------
  //
  // This resolves the maps/compute/ports the placement decided, decides the
  // movement hops and their storage slots and steps, validates occupancy
  // against each memory's capacity, and assigns the decision-only v3 id (the
  // hops are identity-bearing decisions). The strict attempt runs first; when
  // *only* a physical fact is incomplete it is re-run as an analysis artifact,
  // which succeeds and records `physicalComplete=false` plus its ordered
  // reasons. A genuine rejection -- a capacity overflow, a cyclic graph, a
  // coverage gap -- fails both passes and is dropped.
  plan.materialized = true;
  if (llvm::Error error = finalizeStoragePlan(graph, plan, machine)) {
    std::string reason = llvm::toString(std::move(error));
    CoveringPlan analysis = proposal;
    analysis.materialized = false;
    recordProvenance(analysis, graph, target);
    if (llvm::Error second = finalizeStoragePlan(graph, analysis, machine)) {
      std::string secondText = llvm::toString(std::move(second));
      if (secondText != reason)
        reason += "; analysis: " + secondText;
      return rejected(DiagnosticCode::MemoryCapacityExceeded,
                      std::move(reason));
    }
    plan = std::move(analysis);
  }

  // --- 3. bind a preview onto a private clone -----------------------------
  //
  // The binder clones `source`, so the caller's module is never mutated. Under
  // the executable contract the preview *is* the executable binding: a decision
  // the binder cannot materialize, or a value whose physical memory does not
  // resolve, is refused here exactly as `bindPlan` refuses it for the caller.
  //
  // That refusal is a *candidate* rejection (issue #129, task R7): it is a
  // property of this proposal, so the search records it with its stable code
  // and keeps enumerating -- some other covering may be executable, and a
  // cheap non-executable one must not end the search. The message is the
  // binder's own, so a caller that ends up with no plan reads exactly the
  // refusal a direct `bindPlan` under the executable contract would give. The
  // direct (outside-search) `bindPlan` keeps failing hard; only the search
  // context turns it into a rejection.
  std::unique_ptr<PlanMaterializer> materializer =
      micro_mapping_detail::createCanonicalPlanMaterializer();
  llvm::Expected<BoundPlan> bound =
      bindPlan(source, plan, target, contract, materializer.get());
  if (!bound) {
    std::string reason = llvm::toString(bound.takeError());
    if (strict)
      return rejected(DiagnosticCode::UnsupportedMaterialization,
                      std::move(reason));
    return llvm::createStringError(llvm::inconvertibleErrorCode(), reason);
  }

  // --- 4. selected static analysis ----------------------------------------
  //
  // The plan-carrying overload attaches R5 storage liveness, so `peakBytes` is
  // the plan's own capacity-verdict peak and not a second, weaker relation. The
  // analysis is taken leniently: a stream the machine does not fully model is
  // *reported* (its reasons travel with the plan), never a rejection -- a
  // `micro-perf` run on the same kernel reports the same incompleteness, so the
  // planner and the simulator stay one analysis rather than two verdicts.
  llvm::Expected<perf::SelectedKernelAnalysis> analysis =
      perf::analyzeSelectedKernel(bound->kernel, machine, /*requireComplete=*/
                                  false, plan, graph);
  if (!analysis) {
    // The analysis could not run at all -- the machine does not model a memory
    // the bound kernel names, for instance. That is a modelling gap, not a
    // property of this candidate: the plan is kept and scored from its own
    // synthesized events, exactly as it was before this stage existed, and the
    // reason is reported. (`micro-perf` on the same kernel reports the same
    // gap, so the two never disagree about a kernel neither can model.)
    plan.diagnostics.warnings.push_back(
        "plan evaluation: the selected static analysis could not run: " +
        llvm::toString(analysis.takeError()));
    scoreFromOwnEvents(plan, machine);
    CompletePlanEvaluation evaluation;
    evaluation.plan = std::move(plan);
    return evaluation;
  }

  // --- 5. validate occupancy ----------------------------------------------
  //
  // `finalizeStoragePlan` already rejected a capacity overflow in both modes,
  // so this is the explicit cross-check that the analysis's own live peak --
  // the number `micro-perf` reports -- is within capacity. A disagreement here
  // is a candidate rejection, never a silently kept plan.
  if (std::string overflow = occupancyOverflow(*analysis, machine);
      !overflow.empty())
    return rejected(DiagnosticCode::MemoryCapacityExceeded,
                    std::move(overflow));

  // --- 6. attach the derived snapshot and final cost ----------------------
  //
  // The snapshot is the materialized work's own event stream: once attached,
  // `buildPlanEvents` returns it, so the plan's final score and `micro-perf`'s
  // prediction are one schedule of one kernel. A stream the machine cannot
  // model -- an abstract engine an unrouted boundary movement names -- cannot
  // be attached; the reason is recorded and the plan keeps the (already
  // derived) analysis cost, which is the column a report must show either way.
  plan.totalCost = analysis->cost;
  plan.scoreSource = PlanScoreSource::Schedule;
  if (llvm::Error error =
          attachPlanAnalysisEvents(plan, analysis->events, machine)) {
    std::string reason = llvm::toString(std::move(error));
    plan.diagnostics.warnings.push_back(
        "plan evaluation: the derived event snapshot could not be attached: " +
        reason);
  }

  // --- 7. durable choices must be stable ----------------------------------
  //
  // Re-finalizing the decided plan must reproduce its id: the hops and storage
  // slots the evaluation chose are decisions, and a second pass over the same
  // proposal rebuilds them identically. A different id means the choice
  // function depends on the derived ordering it is supposed to determine -- a
  // cycle. The loop is a bounded deterministic fixpoint: it reproduces the
  // choices the first pass retained and never spins.
  {
    bool stable = false;
    for (unsigned pass = 0; pass < kMaxEvaluationPasses; ++pass) {
      CoveringPlan probe = plan;
      if (llvm::Error error = finalizeStoragePlan(graph, probe, machine))
        break;
      // Only the *decisions* are compared: the probe carries no derived
      // snapshot or cost, so its id is the pure function of the choices.
      if (probe.id == plan.id) {
        stable = true;
        break;
      }
    }
    if (!stable) {
      if (strict)
        return rejected(
            DiagnosticCode::UnsupportedMaterialization,
            "plan evaluation: the finalized plan's decisions do not "
            "reproduce a stable identity");
      plan.diagnostics.warnings.push_back(
          "plan evaluation: the plan's decisions did not reach a stable "
          "identity within the evaluation bound");
    }
  }

  CompletePlanEvaluation evaluation;
  evaluation.plan = std::move(plan);
  return evaluation;
}

} // namespace mlir::llk::mapping
