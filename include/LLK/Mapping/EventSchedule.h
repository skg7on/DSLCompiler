//===- EventSchedule.h - Shared resource scheduler (task B8) -------------===//
//
// Part of the target-independent mapping core (epic #67).
//
// The selected-plan score and the performance prediction must be the *same*
// schedule of the same normalized events, or the search would rank candidates
// by a model no downstream stage shares. This header owns that one schedule:
// a deterministic list scheduler that places every event on the resource pool
// it names, respecting its dependencies and resource multiplicity.
//
// It lives in the mapping core (not the perf library) because the search needs
// it to compute a candidate's final score; `lib/Perf` wraps it rather than
// re-implementing it. Resource identities are machine node ids and kinds, never
// target-name branches.
//
//===----------------------------------------------------------------------===//

#ifndef LLK_MAPPING_EVENTSCHEDULE_H
#define LLK_MAPPING_EVENTSCHEDULE_H

#include "LLK/Mapping/CostEvent.h"

#include "llvm/ADT/ArrayRef.h"

#include <cstdint>
#include <string>
#include <vector>

namespace mlir::llk::machine {
struct MachineModel;
} // namespace mlir::llk::machine

namespace mlir::llk::mapping {

/// One event's placement on the resource timeline.
struct ScheduledEvent {
  uint32_t id = 0;
  uint64_t start = 0;
  uint64_t finish = 0;
};

/// The result of scheduling one normalized event stream: enough for the mapping
/// search to score a candidate and for the performance report to assemble its
/// utilization figures without re-running the loop.
struct EventScheduleResult {
  uint64_t predictedCycles = 0;
  uint32_t criticalId = 0;
  /// The sum of every event's cycle charge, before overlap. The scheduler's
  /// prediction is never larger; the ratio is overlap efficiency.
  uint64_t sequentialCycles = 0;
  /// One entry per event, indexed by event id.
  std::vector<ScheduledEvent> entries;
};

/// Schedules `events` on `machine`. Among the events whose dependencies have
/// finished, the lowest event id goes first; it starts when both its
/// dependencies and its resource slots allow. Events that name the same
/// resource pool each other's slots -- but a pool is only ever as wide as the
/// one node it names: a transfer engine's own `count` (and equally a compute
/// node's own `concurrency`) is its concurrency, so an engine id is never
/// widened by another engine the machine happens to declare. Two engine ids of
/// one class are two pools. `owners`, when it is parallel to `events` and
/// non-empty, adds an owner-occupancy pool per owner, bounded by the machine's
/// owner count -- the constraint a `micro.spatial_for` puts on concurrent
/// tiles.
///
/// A resource the machine does not model is scheduled on a single slot rather
/// than rejected: scheduling produces a cost, and cost never decides legality.
/// That single slot is a deliberately partial answer -- it never claims the
/// whole machine's multiplicity for an unnamed pool. A caller that needs the
/// stream to be fully modelled calls `validateEventResources` first.
EventScheduleResult
scheduleNormalizedEvents(llvm::ArrayRef<PlanCostEvent> events,
                         const machine::MachineModel &machine,
                         llvm::ArrayRef<std::string> owners = {});

/// Bytes a normalized stream moves through the DRAM level, under the one
/// traffic convention (issue #129, task R6): an event with traffic charges its
/// bytes to the memory it reads and the memory it writes, so a `dram -> sram`
/// hop counts against DRAM once on each movement that touches it. A movement
/// names its memories by machine node id or by abstract kind; both resolve to
/// the node's declared kind, so a plan-derived event and a kernel-derived one
/// are counted the same way.
uint64_t dramTrafficBytes(llvm::ArrayRef<PlanCostEvent> events,
                          const machine::MachineModel &machine);

/// The strict counterpart to `scheduleNormalizedEvents`: every event must name
/// a resource the machine models, with a positive slot count, and its
/// dependency edges must form a real acyclic order over the stream. Returns an
/// error naming the first event that fails.
///
/// This is the check a caller uses when a zero-cost or single-slot fallback
/// would hide a modelling gap -- plan evaluation (R6) and any strict analysis
/// path call it before scheduling. `scheduleNormalizedEvents` itself stays
/// lenient, because a cost is not a legality verdict.
llvm::Error validateEventResources(const PlanEventDAG &dag,
                                   const machine::MachineModel &machine);

} // namespace mlir::llk::mapping

#endif // LLK_MAPPING_EVENTSCHEDULE_H
