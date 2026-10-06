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
/// resource pool each other's slots (a machine with N DMA engines runs N copies
/// at once); two engine ids of one class are two pools. `owners`, when it is
/// parallel to `events` and non-empty, adds an owner-occupancy pool per owner,
/// bounded by the machine's owner count -- the constraint a `micro.spatial_for`
/// puts on concurrent tiles.
///
/// A resource the machine does not model is scheduled on a single slot rather
/// than rejected: scheduling produces a cost, and cost never decides legality.
EventScheduleResult
scheduleNormalizedEvents(llvm::ArrayRef<PlanCostEvent> events,
                         const machine::MachineModel &machine,
                         llvm::ArrayRef<std::string> owners = {});

} // namespace mlir::llk::mapping

#endif // LLK_MAPPING_EVENTSCHEDULE_H
