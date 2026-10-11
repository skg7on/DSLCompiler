//===- SelectedKernelAnalysis.h - One static analysis of a selected kernel =//
//
// Part of the M10 performance simulator (issue #46); the shared extraction the
// mapping planner and `micro-perf` both read (issue #129, task R6).
//
// The planner and the simulator used to describe the same selected plan two
// different ways: the planner charged each compute step its rule-local estimate
// while the simulator charged the machine's elementwise or MMA formula, and
// neither recorded the owner-occupancy pool, the movement memories or the
// scheduled cost the other did. `analyzeSelectedKernel` is the single
// extraction that ends that: it walks the canonical mapped Micro-IR *once*,
// normalizes every event through the one shared construction point
// (`normalizedPlanEvent`), validates the stream against the machine (task R2),
// schedules it with the one shared scheduler (`scheduleNormalizedEvents`) under
// the same owner constraints, and summarizes its traffic and its live storage.
//
// The result is deliberately value-shaped and deterministic: the event stream,
// the schedule of that stream, the overlapped `Cost`, per-memory read/write
// traffic and per-memory live peak bytes. Nothing here is calibrated. Measured
// estimates stay separate (design §17.2): a `LatencyProvider` may override an
// event's *duration* only after this static analysis has established legality,
// and a provider hit must never erase the event's work, traffic or capacity.
// The default `micro-perf` run and every planner acceptance comparison are
// static, and this is the static baseline they share.
//
//===- Library boundary --------------------------------------------------===//
//
// This lives in LLKPerf and consumes LLKMapping types (`PlanEventDAG`,
// `EventScheduleResult`, `scheduleNormalizedEvents`, `validateEventResources`).
// The dependency is one-way -- LLKPerf -> LLKMapping -- and LLKMapping never
// links this library. Nothing here is reachable from the mapping core.
//
//===----------------------------------------------------------------------===//

#ifndef LLK_PERF_SELECTEDKERNELANALYSIS_H
#define LLK_PERF_SELECTEDKERNELANALYSIS_H

#include "LLK/Machine/MachineModel.h"
#include "LLK/Mapping/CostEvent.h"
#include "LLK/Mapping/CostModel.h"
#include "LLK/Mapping/EventSchedule.h"
#include "LLK/Mapping/MappingPlan.h"
#include "LLK/Perf/MicroDAG.h"

#include "llvm/Support/Error.h"

#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace mlir {
class Operation;
} // namespace mlir

namespace mlir::llk::perf {

/// The one static analysis of a selected kernel: its normalized event stream,
/// the schedule of that stream, the overlapped cost, the traffic each memory
/// carries and the bytes each memory must hold live at once, plus an explicit
/// completeness verdict.
///
/// `complete` is false when the extraction could not describe the kernel
/// exactly -- a loop whose trip count is unknown (charged one iteration), a
/// value whose bytes cannot be computed (charged zero), an event whose machine
/// resource is not modelled. `incompleteReasons` names each such fact in the
/// order the walk met it. A strict caller (`requireComplete`) refuses a partial
/// analysis; a lenient one receives it *with* its reasons and its cost, never
/// as a silent approximation.
struct SelectedKernelAnalysis {
  /// The normalized event stream, in extraction (dependency) order. Each event
  /// carries the extended execution facts the plan path records too -- owner,
  /// source/destination memory -- so a planner event and its materialized
  /// counterpart are comparable field for field (issue #129, task R5/R6).
  mapping::PlanEventDAG events;
  /// The shared resource schedule of `events`. Always present, even when
  /// `complete` is false: a partial stream still schedules to the numbers an
  /// analysis artifact reports.
  mapping::EventScheduleResult schedule;
  /// The overlapped cost of that schedule (latency, byte total, aggregate
  /// compute and transfer load). The same numbers `schedulePlanEvents` derives
  /// from the same stream.
  mapping::Cost cost;
  /// Per-memory bytes the event stream reads and writes, under the one
  /// documented convention `computeL0StaticBound` uses: an event with traffic
  /// charges its bytes to the memory it reads *and* the memory it writes, so a
  /// `dram -> sram` copy counts against both levels. A compute event carries no
  /// bytes and charges nothing.
  std::map<std::string, uint64_t> trafficBytes;
  /// Peak simultaneous live tile bytes per memory space. When the caller
  /// supplied the finalized plan this kernel was bound to, this is the plan's
  /// own *R5 storage-liveness* peak; otherwise it is the extraction's
  /// allocation accounting (`MicroDAG::liveTileBytesByMemory`), the kernel-side
  /// analogue. `storageLivenessFromPlan` says which.
  std::map<std::string, uint64_t> peakBytes;
  /// The reuse ordering edges the plan's liveness analysis requires: one per
  /// accepted alias, from the step that last reads the reused buffer to the
  /// step that first writes the alias. Empty unless `storageLivenessFromPlan`.
  std::vector<mapping::PlanStepEdge> requiredReuseEdges;
  /// True when `peakBytes` and `requiredReuseEdges` came from the finalized
  /// plan's R5 storage liveness rather than the extraction's own accounting.
  /// A caller that needs the liveness-derived occupancy -- the strict analysis
  /// the brief requires -- passes the finalized plan and graph to
  /// `analyzeSelectedKernel`; a kernel-only caller gets the extraction's
  /// accounting and can read this flag to see that no plan liveness was run.
  bool storageLivenessFromPlan = false;
  /// True only when every fact the stream needed was resolved.
  bool complete = false;
  /// The ordered reasons `complete` is false. Empty exactly when `complete`.
  std::vector<std::string> incompleteReasons;
};

/// The per-memory traffic of an extracted DAG under the one documented
/// convention: an event with traffic charges its bytes to the memory it reads
/// (its source kind) and the memory it writes (its destination kind), so a
/// `dram -> sram` copy counts against both levels and a same-kind movement
/// counts against that level twice -- exactly the rule `computeL0StaticBound`
/// has always applied, now named once so the L0 totals and the analysis's
/// `trafficBytes` cannot drift.
std::map<std::string, uint64_t> dagTrafficByMemory(const MicroDAG &dag);

/// Runs the shared static analysis over `kernel`, a `micro.kernel`.
///
/// The kernel is walked once (`buildMicroDAG`), its events normalized, the
/// stream validated (`validateEventResources`) and scheduled with the shared
/// scheduler under the owner-occupancy constraints its placement recorded. When
/// `requireComplete` is set, an incomplete stream is an error naming every
/// reason rather than a partial analysis; when it is clear, the partial
/// analysis is returned with `complete == false` and its reasons, which is what
/// an explicit analysis artifact reports.
///
/// The stream is the materialized kernel's own, so `events.source` is
/// `PlanEventSource::Snapshot`. This form is given no plan, so it runs no
/// storage liveness: `peakBytes` is the extraction's own allocation accounting
/// and `storageLivenessFromPlan` is false. Use the plan-carrying overload below
/// when the liveness-derived occupancy is required.
llvm::Expected<SelectedKernelAnalysis>
analyzeSelectedKernel(mlir::Operation *kernel,
                      const machine::MachineModel &machine,
                      bool requireComplete);

/// The strict form, with the finalized plan the kernel was bound to: the
/// analysis additionally carries the plan's R5 *storage-liveness* occupancy.
///
/// `finalizedPlan` must be the plan whose kernel is `kernel` (already through
/// `finalizeStoragePlan`, so it carries its allocations, step DAG and movement
/// hops) and `graph` the workload graph it was planned from. The liveness is
/// run over the plan's own storage facts -- `buildPlanEvents(plan, machine,
/// &graph)` scheduled exactly as `finalizeStoragePlan` schedules its occupancy
/// probe, then `analyzeStorageLiveness` -- so `peakBytes` is the plan's
/// capacity-verdict peak and `requiredReuseEdges` is the ordering that peak
/// relies on, and neither is a second, weaker relation invented here.
///
/// A plan whose liveness cannot be established is an error under
/// `requireComplete` and an explicit incomplete reason otherwise: the analysis
/// never reports the extraction's allocation accounting as if it were the
/// verified occupancy.
///
/// R5's signature is deliberately untouched: `analyzeStorageLiveness` keeps
/// taking a `CoveringPlan`, and this overload is where a plan's liveness meets
/// the kernel's analysis.
llvm::Expected<SelectedKernelAnalysis> analyzeSelectedKernel(
    mlir::Operation *kernel, const machine::MachineModel &machine,
    bool requireComplete, const mapping::CoveringPlan &finalizedPlan,
    const mapping::WorkloadGraph &graph);

/// Overlays the finalized plan's R5 storage-liveness occupancy onto `analysis`
/// in place: the peak becomes `live.peakBytes`, `requiredReuseEdges` its
/// ordering, and `storageLivenessFromPlan` is set. On failure the reason is
/// recorded and, when `requireComplete` is set, an error is returned. Exposed
/// so a caller that already has the analysis can attach the plan's liveness
/// without re-extracting the kernel.
llvm::Error
attachPlanStorageLiveness(SelectedKernelAnalysis &analysis,
                          const mapping::CoveringPlan &finalizedPlan,
                          const mapping::WorkloadGraph &graph,
                          const machine::MachineModel &machine,
                          bool requireComplete);

/// The shared core of `analyzeSelectedKernel` over an already-extracted DAG.
/// Exposed so a caller that needs the raw `MicroDAG` too -- the report builder,
/// which reads the extraction's live-byte and diagnostic facts -- extracts it
/// once and hands it here, instead of walking the kernel twice.
llvm::Expected<SelectedKernelAnalysis>
analyzeSelectedDag(const MicroDAG &dag, const machine::MachineModel &machine,
                   bool requireComplete);

} // namespace mlir::llk::perf

#endif // LLK_PERF_SELECTEDKERNELANALYSIS_H
