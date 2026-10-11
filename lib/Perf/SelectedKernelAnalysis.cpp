//===- SelectedKernelAnalysis.cpp - One static analysis of a selected kernel
//==//
//
// Implements analyzeSelectedKernel() / analyzeSelectedDag(): the single
// extraction both the mapping planner and the performance report read (issue
// #129, task R6). See the header for the contract and the library boundary.

#include "LLK/Perf/SelectedKernelAnalysis.h"

#include "LLK/Mapping/EventSchedule.h"
#include "LLK/Mapping/StorageLiveness.h"
#include "LLK/Mapping/WorkloadGraph.h"

#include "llvm/ADT/Twine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/MathExtras.h"

#include <limits>

#include <cmath>
#include <optional>
#include <string>
#include <utility>

namespace mlir::llk::perf {
namespace {

llvm::Error analysisError(const llvm::Twine &message) {
  return llvm::make_error<llvm::StringError>(message.str(),
                                             llvm::inconvertibleErrorCode());
}

/// A checked accumulate: it adds `amount` to `value` and reports overflow. The
/// guard and the accumulation are one operation, so a totalled count can never
/// be left at zero by an overflow check that forgot to add.
bool addChecked(uint64_t &value, uint64_t amount) {
  if (amount > std::numeric_limits<uint64_t>::max() - value)
    return true;
  value += amount;
  return false;
}

/// The cycles one normalized event occupies its resource, rounded up exactly as
/// the shared scheduler charges it, so the reported busy totals and the
/// schedule's own `sequentialCycles` cannot disagree.
uint64_t eventCycles(const mapping::PlanCostEvent &event) {
  return static_cast<uint64_t>(std::ceil(event.event.cost.latencyCycles));
}

} // namespace

//===----------------------------------------------------------------------===//
// Shared analysis over one extracted DAG
//===----------------------------------------------------------------------===//

std::map<std::string, uint64_t> dagTrafficByMemory(const MicroDAG &dag) {
  std::map<std::string, uint64_t> traffic;
  for (const MicroEvent &event : dag.events) {
    if (event.bytes == 0)
      continue;
    // A movement crosses two levels; both see the traffic.
    if (!event.srcMemory.empty())
      traffic[event.srcMemory] += event.bytes;
    if (!event.tileMemory.empty())
      traffic[event.tileMemory] += event.bytes;
  }
  return traffic;
}

llvm::Expected<SelectedKernelAnalysis>
analyzeSelectedDag(const MicroDAG &dag, const machine::MachineModel &machine,
                   bool requireComplete) {
  SelectedKernelAnalysis analysis;

  // --- normalize every event through the one shared construction point ------
  //
  // `normalizedPlanEvent` is the same function the scheduler's normalized view
  // uses, and it already fills the execution facts both paths share: the
  // owner-occupancy pool (the mapped executor's id, or the abstract owner
  // symbol for unmapped analysis) and the two memories a movement crosses.
  // The stream this entry point produces *is* the materialized kernel's own:
  // it was extracted from the concrete Micro-IR, not accumulated from a plan's
  // rule estimates, so it carries the `Snapshot` label the plan path reserves
  // for a stream that describes the work the kernel actually does.
  analysis.events.source = mapping::PlanEventSource::Snapshot;
  std::vector<std::string> owners;
  analysis.events.events.reserve(dag.events.size());
  owners.reserve(dag.events.size());
  for (const MicroEvent &event : dag.events) {
    mapping::PlanCostEvent normalized = normalizedPlanEvent(event);
    owners.push_back(normalized.owner);
    analysis.events.events.push_back(std::move(normalized));
  }

  // --- traffic, under the one documented convention -------------------------
  //
  // The same named rule `computeL0StaticBound` reads its byte totals from, so
  // the L0 block and this summary cannot drift.
  analysis.trafficBytes = dagTrafficByMemory(dag);

  analysis.peakBytes = dag.liveTileBytesByMemory;

  // --- completeness ---------------------------------------------------------
  //
  // The extraction's own incompleteness facts first (an unknown trip count, an
  // uncomputable footprint, an unmodelled owner scope), then the strict
  // resource/dependency verdict: a stream that names a machine resource the
  // model does not have, or whose dependency edges are self, out-of-range,
  // duplicated or cyclic, cannot be scheduled as if it were complete.
  for (const std::string &reason : dag.incompleteReasons)
    analysis.incompleteReasons.push_back(reason);
  if (llvm::Error error =
          mapping::validateEventResources(analysis.events, machine))
    analysis.incompleteReasons.push_back(llvm::toString(std::move(error)));
  analysis.complete = analysis.incompleteReasons.empty();

  if (requireComplete && !analysis.complete) {
    std::string message =
        "selected-kernel analysis: the event stream is incomplete";
    for (const std::string &reason : analysis.incompleteReasons)
      message += "\n  " + reason;
    return analysisError(message);
  }

  // --- schedule -------------------------------------------------------------
  //
  // The same shared scheduler and the same owner constraints `scheduleL1` and
  // `schedulePlanEvents` apply, so all three are one schedule of one stream. It
  // runs even for an incomplete stream: a partial analysis still reports the
  // numbers an artifact is expected to carry, alongside its explicit reasons.
  analysis.schedule = mapping::scheduleNormalizedEvents(analysis.events.events,
                                                        machine, owners);

  // --- cost, derived from that one schedule ---------------------------------
  //
  // The identical formula `schedulePlanEvents` uses: the overlapped critical
  // path, the stream's byte total, and the aggregate compute/transfer load over
  // one shared machine window.
  mapping::Cost cost;
  cost.latencyCycles = static_cast<double>(analysis.schedule.predictedCycles);
  uint64_t computeBusy = 0;
  uint64_t transferBusy = 0;
  uint64_t totalBytes = 0;
  for (const mapping::PlanCostEvent &event : analysis.events.events) {
    if (addChecked(totalBytes, event.bytes))
      return analysisError("selected-kernel analysis: the event stream's byte "
                           "total overflows a 64-bit count");
    const uint64_t cycles = eventCycles(event);
    if (event.event.kind == mapping::CostEventKind::Compute) {
      if (addChecked(computeBusy, cycles))
        return analysisError("selected-kernel analysis: compute busy cycles "
                             "overflow a 64-bit count");
    } else if (event.event.kind == mapping::CostEventKind::TransferHop) {
      if (addChecked(transferBusy, cycles))
        return analysisError("selected-kernel analysis: transfer busy cycles "
                             "overflow a 64-bit count");
    }
  }
  cost.localBytes = totalBytes;
  // The DRAM dimension of the same run, through the shared helper, so this cost
  // and `schedulePlanEvents`'s cost of the same stream are one Cost.
  cost.dramBytes = mapping::dramTrafficBytes(analysis.events.events, machine);
  if (std::optional<double> utilization = mapping::utilizationEstimate(
          static_cast<double>(computeBusy), machine, machine.workerThreads))
    cost.computeUtilization = *utilization;
  if (std::optional<double> utilization =
          mapping::utilizationEstimate(static_cast<double>(transferBusy),
                                       machine, machine.transferEngineCount()))
    cost.transferUtilization = *utilization;
  analysis.cost = cost;
  return analysis;
}

//===----------------------------------------------------------------------===//
// Entry point
//===----------------------------------------------------------------------===//

llvm::Expected<SelectedKernelAnalysis>
analyzeSelectedKernel(mlir::Operation *kernel,
                      const machine::MachineModel &machine,
                      bool requireComplete) {
  llvm::Expected<MicroDAG> dag = buildMicroDAG(kernel, machine);
  if (!dag)
    return dag.takeError();
  return analyzeSelectedDag(*dag, machine, requireComplete);
}

llvm::Error
attachPlanStorageLiveness(SelectedKernelAnalysis &analysis,
                          const mapping::CoveringPlan &finalizedPlan,
                          const mapping::WorkloadGraph &graph,
                          const machine::MachineModel &machine,
                          bool requireComplete) {
  // The occupancy probe is built the way `finalizeStoragePlan` builds its own:
  // the plan's storage facts over its workload graph, on the plan's *own*
  // schedule. That is deliberate -- this peak has to be the same number the
  // plan's capacity verdict used, not a second relation that happens to be
  // close. The graph-carrying `buildPlanEvents` call is what derives each
  // allocation's uses from the plan, so it is the right one here.
  auto unavailable = [&](std::string reason) -> llvm::Error {
    analysis.incompleteReasons.push_back(reason);
    analysis.complete = false;
    if (!requireComplete)
      return llvm::Error::success();
    return analysisError("selected-kernel analysis: the finalized plan's "
                         "storage liveness could not be established: " +
                         reason);
  };

  llvm::Expected<mapping::PlanEventDAG> events =
      mapping::buildPlanEvents(finalizedPlan, machine, &graph);
  if (!events)
    return unavailable("the plan's storage-fact event stream could not be "
                       "built: " +
                       llvm::toString(events.takeError()));

  const mapping::EventScheduleResult schedule =
      mapping::scheduleNormalizedEvents(events->events, machine);
  llvm::Expected<mapping::StorageLivenessResult> live =
      mapping::analyzeStorageLiveness(finalizedPlan, *events, schedule);
  if (!live)
    return unavailable("the plan's live ranges do not resolve: " +
                       llvm::toString(live.takeError()));

  analysis.peakBytes = live->peakBytes;
  analysis.requiredReuseEdges = live->requiredReuseEdges;
  analysis.storageLivenessFromPlan = true;
  return llvm::Error::success();
}

llvm::Expected<SelectedKernelAnalysis> analyzeSelectedKernel(
    mlir::Operation *kernel, const machine::MachineModel &machine,
    bool requireComplete, const mapping::CoveringPlan &finalizedPlan,
    const mapping::WorkloadGraph &graph) {
  // The plan's measured durations travel to the extraction, so the analysis
  // charges the calibrated work the search ranked on instead of the static
  // machine formula the search had already overridden (issue #129 review
  // finding 6). Absent a provider hit the overrides are empty and the
  // extraction is unchanged.
  MeasuredOverrides measured;
  for (const mapping::PlanPlacement &placement : finalizedPlan.placements)
    if (placement.measuredCycles)
      measured.byInstance[placement.instance] = *placement.measuredCycles;
  for (const mapping::PlanConnection &connection :
       finalizedPlan.connectionPlans)
    if (connection.measuredCycles)
      measured.byConnection[connection.id] = *connection.measuredCycles;
  const bool haveMeasured =
      !measured.byInstance.empty() || !measured.byConnection.empty();

  llvm::Expected<MicroDAG> dag =
      buildMicroDAG(kernel, machine, haveMeasured ? &measured : nullptr);
  if (!dag)
    return dag.takeError();
  llvm::Expected<SelectedKernelAnalysis> analysis =
      analyzeSelectedDag(*dag, machine, requireComplete);
  if (!analysis)
    return analysis.takeError();
  // R5 liveness for the strict analysis: the plan's own occupancy replaces the
  // extraction's allocation accounting, so a strict caller reads the same peak
  // the plan's capacity verdict did.
  if (llvm::Error error = attachPlanStorageLiveness(
          *analysis, finalizedPlan, graph, machine, requireComplete))
    return std::move(error);
  return analysis;
}

} // namespace mlir::llk::perf
