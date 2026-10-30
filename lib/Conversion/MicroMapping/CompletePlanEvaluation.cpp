//===- CompletePlanEvaluation.cpp - Evaluate a complete proposal (R7) -----===//
//
// Part of the LLK Compiler Project, under the Apache License v2.0 with
// LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// See CompletePlanEvaluation.h. The evaluation is the one place a search
// proposal becomes a physical, schedulable plan, and it is deliberately built
// only from functions the planner and the performance model already share
// (`finalizeStoragePlan`, `bindPlan`, `analyzeSelectedKernel`) so a plan's
// feasibility and its predicted cost are one analysis of one kernel.

#include "CompletePlanEvaluation.h"

#include "TargetCoverageValidation.h"

#include "MicroMappingCommon.h"

#include "LLK/Conversion/MicroMapping/ShapeSpecialization.h"

#include "LLK/Mapping/CostEvent.h"
#include "LLK/Mapping/CostModel.h"
#include "LLK/Mapping/MappingMetadata.h"
#include "LLK/Mapping/StoragePlan.h"

#include "LLK/Perf/SelectedKernelAnalysis.h"

#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/Error.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

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

/// Sums the analysis's per-memory live peaks into the whole-plan resident
/// footprint the global byte budget bounds (design §9.3), or an empty string
/// when the total is within `budget`. The sum is checked, so an overflow is a
/// reported condition rather than a wrapped total that silently passes.
std::string budgetOverflow(const perf::SelectedKernelAnalysis &analysis,
                           uint64_t budget) {
  uint64_t total = 0;
  for (const auto &entry : analysis.peakBytes) {
    if (__builtin_add_overflow(total, entry.second, &total))
      return "the plan's live bytes overflow a 64-bit total";
  }
  if (total > budget)
    return "the plan requires " + std::to_string(total) +
           " bytes live, over the whole-plan budget of " +
           std::to_string(budget) + " bytes";
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

void recordAnalysisCompleteness(const perf::SelectedKernelAnalysis &analysis,
                                CoveringPlan &plan) {
  if (analysis.complete)
    return;
  for (const std::string &reason : analysis.incompleteReasons)
    plan.diagnostics.warnings.push_back(
        "plan evaluation: the selected static analysis is incomplete: " +
        reason);
}

namespace {

std::string sampleLabel(const shape::ShapeValues &values) {
  std::string label;
  for (const auto &entry : values) {
    if (!label.empty())
      label += ",";
    label += entry.first + "=" + std::to_string(entry.second);
  }
  return label;
}

/// Scores one dynamic plan on its explicit samples.
///
/// Capacity and backend legality are established over the full symbolic domain
/// first: the plan's storage liveness is independent of loop trip counts, so a
/// sample can never make an illegal plan legal. Each sample then materializes
/// an analysis-only clone, binds the *same* plan to it and runs the same strict
/// analysis a static plan uses. The final cost is the profile's weighted mean;
/// the per-sample costs stay on the plan so a report and a materialized-IR run
/// can be compared at each observation.
llvm::Expected<CompletePlanEvaluation>
evaluateSampledPlan(mlir::ModuleOp source, const WorkloadGraph &graph,
                    const MappingTarget &target, CoveringPlan plan,
                    const BoundPlan &bound, const ShapeSampleOptions &shapes,
                    std::optional<uint64_t> memoryBudgetBytes) {
  const machine::MachineModel &machine = target.machine();
  if (!shapes.contract)
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "sampled plan evaluation requires a shape contract");
  if (llvm::Error error =
          perf::validateShapeProfile(*shapes.contract, *shapes.profile))
    return std::move(error);

  // Full-domain capacity/backend legality. The bound dynamic kernel's loop
  // bounds are runtime values, so only the plan's own storage liveness is
  // required here -- never a one-iteration event stream.
  llvm::Expected<perf::SelectedKernelAnalysis> domain =
      perf::analyzeSelectedKernel(bound.kernel, machine,
                                  /*requireComplete=*/false, plan, graph);
  if (!domain)
    return rejected(DiagnosticCode::UnsupportedMaterialization,
                    "analysis_incomplete: " +
                        llvm::toString(domain.takeError()));
  if (!domain->storageLivenessFromPlan)
    return rejected(DiagnosticCode::UnsupportedMaterialization,
                    "analysis_incomplete: the full symbolic domain's capacity "
                    "could not be certified from the plan's liveness");
  if (std::string overflow = occupancyOverflow(*domain, machine);
      !overflow.empty())
    return rejected(DiagnosticCode::MemoryCapacityExceeded,
                    std::move(overflow));
  if (memoryBudgetBytes) {
    if (std::string overflow = budgetOverflow(*domain, *memoryBudgetBytes);
        !overflow.empty())
      return rejected(DiagnosticCode::MemoryCapacityExceeded,
                      std::move(overflow));
  }

  std::vector<WeightedCost> costs;
  costs.reserve(shapes.profile->samples.size());
  plan.shapeSamples.clear();
  plan.shapeSamples.reserve(shapes.profile->samples.size());
  for (const perf::ShapeSample &sample : shapes.profile->samples) {
    const std::string label = sampleLabel(sample.values);
    llvm::Expected<mlir::OwningOpRef<mlir::ModuleOp>> clone =
        materializeShapeSample(source, *shapes.contract, sample.values);
    if (!clone)
      return rejected(DiagnosticCode::UnsupportedMaterialization,
                      "shape_sample(" + label +
                          "): " + llvm::toString(clone.takeError()));

    mlir::Operation *sampleKernel = nullptr;
    unsigned kernels = 0;
    (*clone)->walk([&](mlir::Operation *op) {
      if (op->getName().getStringRef() == "micro.kernel") {
        ++kernels;
        if (!sampleKernel)
          sampleKernel = op;
      }
    });
    if (!sampleKernel || kernels != 1)
      return rejected(
          DiagnosticCode::UnsupportedMaterialization,
          "shape_sample(" + label +
              "): the source must contain exactly one micro.kernel");
    llvm::Expected<WorkloadGraph> sampleGraph =
        extractWorkloadGraph(sampleKernel);
    if (!sampleGraph)
      return rejected(DiagnosticCode::UnsupportedMaterialization,
                      "shape_sample(" + label +
                          "): " + llvm::toString(sampleGraph.takeError()));

    std::unique_ptr<PlanMaterializer> materializer =
        micro_mapping_detail::createCanonicalPlanMaterializer();
    llvm::Expected<BoundPlan> sampleBound =
        bindPlan(clone->get(), plan, target, BindContract::Executable,
                 materializer.get());
    if (!sampleBound)
      return rejected(DiagnosticCode::UnsupportedMaterialization,
                      "shape_sample(" + label +
                          "): " + llvm::toString(sampleBound.takeError()));
    if (!sampleBound->unmaterialized.empty())
      return rejected(DiagnosticCode::UnsupportedMaterialization,
                      "shape_sample(" + label +
                          "): selected decisions were not materialized: " +
                          llvm::join(sampleBound->unmaterialized, "; "));
    if (llvm::Error error =
            verifyTargetOperationCoverage(*sampleBound->module, plan, target))
      return rejected(DiagnosticCode::UnsupportedMaterialization,
                      "shape_sample(" + label + "): target_coverage: " +
                          llvm::toString(std::move(error)));

    llvm::Expected<perf::SelectedKernelAnalysis> analysis =
        perf::analyzeSelectedKernel(sampleBound->kernel, machine,
                                    /*requireComplete=*/true, plan,
                                    *sampleGraph);
    if (!analysis)
      return rejected(DiagnosticCode::UnsupportedMaterialization,
                      "shape_sample(" + label +
                          "): " + llvm::toString(analysis.takeError()));
    if (std::string overflow = occupancyOverflow(*analysis, machine);
        !overflow.empty())
      return rejected(DiagnosticCode::MemoryCapacityExceeded,
                      "shape_sample(" + label + "): " + std::move(overflow));
    if (memoryBudgetBytes) {
      if (std::string overflow = budgetOverflow(*analysis, *memoryBudgetBytes);
          !overflow.empty())
        return rejected(DiagnosticCode::MemoryCapacityExceeded,
                        "shape_sample(" + label + "): " + std::move(overflow));
    }

    PlanShapeSampleCost record;
    record.values = sample.values;
    record.weight = sample.weight;
    record.cost = analysis->cost;
    record.peakBytes = analysis->peakBytes;
    record.fillElements = analysis->fillElements;
    plan.shapeSamples.push_back(std::move(record));
    costs.push_back({analysis->cost, sample.weight});

    // A single observation is also a usable event snapshot. A multi-sample plan
    // deliberately carries none: no one sample's stream describes the weighted
    // mean, and presenting one as if it did would make the plan's score and its
    // snapshot disagree.
    if (shapes.profile->samples.size() == 1) {
      if (llvm::Error error =
              attachPlanAnalysisEvents(plan, analysis->events, machine))
        plan.diagnostics.warnings.push_back(
            "sampled plan evaluation: the sample's event snapshot could not be "
            "attached: " +
            llvm::toString(std::move(error)));
    }
  }

  llvm::Expected<Cost> total = weightedMeanCost(costs);
  if (!total)
    return total.takeError();
  plan.totalCost = *total;
  plan.scoreSource = PlanScoreSource::Schedule;
  plan.shapeAggregation = shapes.profile->aggregation;
  plan.shapeMaxVariants = shapes.profile->maxVariants;
  plan.shapeProfileCanonical = perf::canonicalShapeProfile(*shapes.profile);
  if (shapes.profile->samples.size() > 1)
    plan.diagnostics.warnings.push_back(
        "sampled plan evaluation: cost is the " + plan.shapeAggregation +
        " over " + std::to_string(shapes.profile->samples.size()) +
        " shape samples; the event snapshot is absent by design");

  // The durable choices must still reproduce their id. This is the same check
  // the static path applies; sampling changes the score, never the decisions.
  bool stable = false;
  for (unsigned pass = 0; pass < kMaxEvaluationPasses; ++pass) {
    CoveringPlan probe = plan;
    if (llvm::Error error = finalizeStoragePlan(graph, probe, machine))
      break;
    if (probe.id == plan.id) {
      stable = true;
      break;
    }
  }
  if (!stable)
    return rejected(DiagnosticCode::UnsupportedMaterialization,
                    "plan evaluation: the finalized plan's decisions do not "
                    "reproduce a stable identity");

  CompletePlanEvaluation evaluation;
  evaluation.plan = std::move(plan);
  return evaluation;
}

} // namespace

llvm::Expected<CompletePlanEvaluation> evaluateCompletePlan(
    mlir::ModuleOp source, const WorkloadGraph &graph,
    const MappingTarget &target, const CoveringPlan &proposal,
    BindContract contract, std::optional<uint64_t> memoryBudgetBytes,
    AnalysisRequirement requirement, const ShapeSampleOptions &shapes) {
  const machine::MachineModel &machine = target.machine();
  const bool strict = contract == BindContract::Executable;
  const bool requireComplete =
      strict || requirement == AnalysisRequirement::Complete;

  // --- 1. clone, provenance, contract -------------------------------------
  CoveringPlan plan = proposal;
  recordProvenance(plan, graph, target);

  // --- 2. finalize physical storage ---------------------------------------
  //
  // This resolves the maps/compute/ports the placement decided, decides the
  // movement hops and their storage slots and steps, validates occupancy
  // against each memory's capacity, and assigns the decision-only v3 id (the
  // hops are identity-bearing decisions). Complete scoring requires this pass
  // to succeed, including on an analysis-only target. Only PartialReport with
  // partial binding may retry unresolved facts as an explicitly partial
  // artifact. Capacity/structural failures are rejected in either mode.
  plan.materialized = true;
  bool strictCapacity = false;
  if (llvm::Error error =
          finalizeStoragePlan(graph, plan, machine, &strictCapacity)) {
    std::string reason = llvm::toString(std::move(error));
    if (requireComplete)
      return rejected(strictCapacity
                          ? DiagnosticCode::MemoryCapacityExceeded
                          : DiagnosticCode::UnsupportedMaterialization,
                      std::move(reason));
    CoveringPlan analysis = proposal;
    analysis.materialized = false;
    recordProvenance(analysis, graph, target);
    bool analysisCapacity = false;
    if (llvm::Error second =
            finalizeStoragePlan(graph, analysis, machine, &analysisCapacity)) {
      std::string secondText = llvm::toString(std::move(second));
      if (secondText != reason)
        reason += "; analysis: " + secondText;
      // Both passes failed, so this is a *hard* rejection and not merely an
      // incomplete physical fact. Label it by its actual cause (issue #129,
      // task R7 review): only a real capacity overflow is a capacity rejection;
      // a cyclic alias chain, an unknown memory or value, an uncovered node or
      // a footprint overflow is a plan that cannot be materialized. Conflating
      // them inflated `plansRejectedByCapacity` and made the stable code lie.
      return rejected(strictCapacity || analysisCapacity
                          ? DiagnosticCode::MemoryCapacityExceeded
                          : DiagnosticCode::UnsupportedMaterialization,
                      std::move(reason));
    }
    plan = std::move(analysis);
  }

  if (requireComplete && !plan.diagnostics.physicalComplete)
    return rejected(DiagnosticCode::UnsupportedMaterialization,
                    "analysis_incomplete: " +
                        llvm::join(plan.diagnostics.physicalReasons, "; "));

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

  // Partial binding may leave selected decisions unmaterialized. Successful
  // analysis of the surviving IR cannot prove those omitted decisions happened.
  if (!bound->unmaterialized.empty()) {
    if (requireComplete)
      return rejected(DiagnosticCode::UnsupportedMaterialization,
                      "analysis_incomplete: selected decisions were not "
                      "materialized: " +
                          llvm::join(bound->unmaterialized, "; "));
    plan.materialized = false;
    for (const std::string &reason : bound->unmaterialized)
      plan.diagnostics.warnings.push_back(
          "plan evaluation: selected decision was not materialized: " + reason);
  }

  // --- 3b. target operation coverage -------------------------------------
  //
  // A well-formed bundle does not prove the target implements the operation it
  // covers. Validate the covered variants/types now, before analysis or
  // scoring, so an unsupported case is a candidate rejection rather than a
  // late lowering failure. The check is read-only.
  if (llvm::Error error =
          verifyTargetOperationCoverage(*bound->module, plan, target))
    return rejected(DiagnosticCode::UnsupportedMaterialization,
                    "target_coverage: " + llvm::toString(std::move(error)));

  // --- 3c. sampled dynamic analysis ---------------------------------------
  //
  // A plan whose executable kernel has runtime loop bounds cannot be ranked
  // from one charged iteration. When the caller supplies explicit samples, the
  // full symbolic domain is certified first and every sample is then scored on
  // its own analysis-only clone. Without a profile the pre-sample behavior is
  // preserved exactly: the strict analysis below refuses the plan.
  if (shapes.profile) {
    if (requirement != AnalysisRequirement::Complete)
      return rejected(DiagnosticCode::UnsupportedMaterialization,
                      "sampled analysis requires complete evaluation");
    return evaluateSampledPlan(source, graph, target, plan, *bound, shapes,
                               memoryBudgetBytes);
  }

  // --- 4. selected static analysis ----------------------------------------
  //
  // The plan-carrying overload attaches R5 storage liveness, so `peakBytes` is
  // the plan's own capacity-verdict peak and not a second, weaker relation.
  // Complete scoring and executable binding both require complete audited
  // analysis. Only explicit PartialReport may preserve unknown work or use
  // synthesized-event fallback; backend availability is a separate contract.
  llvm::Expected<perf::SelectedKernelAnalysis> analysis =
      perf::analyzeSelectedKernel(bound->kernel, machine, /*requireComplete=*/
                                  requireComplete, plan, graph);
  if (!analysis) {
    std::string reason = llvm::toString(analysis.takeError());
    if (requireComplete)
      return rejected(
          DiagnosticCode::UnsupportedMaterialization,
          "analysis_incomplete: selected analysis refused the plan: " + reason);
    plan.materialized = false;
    // Only an explicitly partial artifact may retain a modeling failure and
    // fall back to the plan's own synthesized events.
    plan.diagnostics.warnings.push_back(
        "plan evaluation: the selected static analysis could not run: " +
        reason);
    if (memoryBudgetBytes) {
      perf::SelectedKernelAnalysis storage;
      if (llvm::Error error = perf::attachPlanStorageLiveness(
              storage, plan, graph, machine, /*requireComplete=*/false))
        return std::move(error);
      if (!storage.storageLivenessFromPlan) {
        // Match finalization's explicit plan-step fallback when scheduled
        // liveness is unavailable. Weight slots by simultaneous residency;
        // temporal instances still share the same slot and aliases are counted
        // once by computePeakStorage. Never bypass a budget on a modeling gap.
        std::vector<StorageAllocation> weighted = plan.allocations;
        for (StorageAllocation &allocation : weighted)
          if (__builtin_mul_overflow(allocation.bytes,
                                     allocation.simultaneousOccurrences,
                                     &allocation.bytes))
            return rejected(DiagnosticCode::MemoryCapacityExceeded,
                            "the plan's live bytes overflow the byte budget");
        auto peak = computePeakStorage(weighted);
        if (!peak)
          return rejected(DiagnosticCode::UnsupportedMaterialization,
                          llvm::toString(peak.takeError()));
        storage.peakBytes = std::move(*peak);
      }
      if (std::string overflow = budgetOverflow(storage, *memoryBudgetBytes);
          !overflow.empty())
        return rejected(DiagnosticCode::MemoryCapacityExceeded,
                        std::move(overflow));
    }
    scoreFromOwnEvents(plan, machine);
    CompletePlanEvaluation evaluation;
    evaluation.plan = std::move(plan);
    return evaluation;
  }

  // The analysis ran. If it declared itself incomplete, record its ordered
  // reasons on the plan rather than scoring it as if it were exact (issue #129,
  // task R7 review). Strict analysis has already refused incompleteness;
  // partial analysis records the same reasons `micro-perf` reports.
  if (requireComplete && !analysis->complete)
    return rejected(DiagnosticCode::UnsupportedMaterialization,
                    "analysis_incomplete: " +
                        llvm::join(analysis->incompleteReasons, "; "));
  recordAnalysisCompleteness(*analysis, plan);

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

  // The whole-plan byte budget, on the same honest peak (issue #129, task R7
  // review). The search's own partial-state budget check is disabled once an
  // evaluator is installed -- its sum is an over-estimate it must not reject on
  // -- so without this the option would be silently unenforced on every
  // evaluator-backed run while the report still printed it.
  if (memoryBudgetBytes) {
    if (std::string overflow = budgetOverflow(*analysis, *memoryBudgetBytes);
        !overflow.empty())
      return rejected(DiagnosticCode::MemoryCapacityExceeded,
                      std::move(overflow));
  }

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
      if (requireComplete)
        return rejected(
            DiagnosticCode::UnsupportedMaterialization,
            "plan evaluation: the finalized plan's decisions do not "
            "reproduce a stable identity");
      plan.diagnostics.warnings.push_back(
          "plan evaluation: the plan's decisions did not reach a stable "
          "identity within the evaluation bound");
    }
  }

  if (!requireComplete) {
    plan.materialized = false;
    plan.diagnostics.warnings.push_back(
        "plan evaluation: partial report artifact; not eligible for ranking");
  }
  CompletePlanEvaluation evaluation;
  evaluation.plan = std::move(plan);
  return evaluation;
}

} // namespace mlir::llk::mapping
