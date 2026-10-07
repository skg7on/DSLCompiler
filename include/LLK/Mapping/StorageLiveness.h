//===- StorageLiveness.h - Live storage from execution events -------------===//
//
// Task R5 (issue #129). Occupancy used to be summarized from a *serial
// topological index*: every allocation was charged its producer's whole
// execution multiplicity, so sequential loop iterations that reuse one buffer
// were charged every iteration's worth, and a pipeline's overlap was invisible.
//
// This header owns the stronger relation the plan actually needs: a storage
// allocation is live from the first instant a scheduled event touches it -- the
// start of the operation that writes it, or the first read of a borrowed
// descriptor -- through the completion of the event that last reads it, and the
// bytes it reserves while live are its per-occurrence footprint times the
// number of *simultaneously resident* occurrences. Sequential occurrences reuse
// storage; simultaneous ones cannot. The live windows come from the scheduled
// event stream, so overlap is decided by execution order rather than by a
// topological index that lets parallel work look disjoint.
//
// Reuse is conservative: an allocation only aliases an earlier one when the
// dependency order *proves* they cannot overlap -- the earlier buffer is dead
// before the new writer begins, or the new writer itself takes the earlier
// buffer's last read (an in-place update) -- and the edge that orders them is
// reported as a required reuse edge for the caller to materialize. Nothing here
// invents overlap.
//
//===----------------------------------------------------------------------===//

#ifndef LLK_MAPPING_STORAGELIVENESS_H
#define LLK_MAPPING_STORAGELIVENESS_H

#include "LLK/Mapping/CostEvent.h"
#include "LLK/Mapping/EventSchedule.h"
#include "LLK/Mapping/MappingPlan.h"

#include "llvm/Support/Error.h"

#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace mlir::llk::mapping {

/// The live footprint of a plan's storage, derived from its scheduled events.
struct StorageLivenessResult {
  /// Peak simultaneous residency of each memory, by memory node id, over the
  /// live windows the schedule produces. This is a plan's *own* reservation:
  /// borrowed boundary descriptors are the caller's buffers and are counted
  /// (they occupy the node), while storage the plan proved dead and reused is
  /// counted once.
  std::map<std::string, uint64_t> peakBytes;
  /// The dependency edges the reuse decision needs to be sound: for each
  /// accepted alias, the edge from the point where the reused allocation's last
  /// reader completes to the point where its new writer begins. Empty when no
  /// reuse was chosen. The caller adds them to the plan's step DAG and
  /// materializes the ordering; they are derived timing, so they stay out of
  /// the plan's content id.
  std::vector<PlanStepEdge> requiredReuseEdges;
};

/// Analyzes storage liveness over a plan's scheduled event stream.
///
/// `events` must be the normalized stream of `plan` (built with the workload
/// graph, so each event records the storage its occurrences touch) and
/// `schedule` the resource schedule of that stream (the same shared schedule
/// the plan's score comes from). Every allocation of `plan` whose storage a
/// scheduled event touches gets a live window:
///
///   * a written allocation is live from the start of its writer through the
///     completion of its last reader -- the buffer is reserved for the write;
///   * a borrowed allocation (no writer) is live from the start of its first
///     read through the completion of its last.
///
/// An allocation no event touches is an error: an allocation the schedule
/// cannot place is a modelling gap, never a zero-byte reservation. Peak bytes
/// are summed per memory over the windows that cover a common instant, with
/// each allocation contributing `bytes * simultaneousOccurrences`; an aliased
/// allocation contributes to the same root as the storage it reuses.
///
/// The result is deterministic: windows and peaks are computed from sorted
/// event times, never from iteration order.
llvm::Expected<StorageLivenessResult>
analyzeStorageLiveness(const CoveringPlan &plan, const PlanEventDAG &events,
                       const EventScheduleResult &schedule);

} // namespace mlir::llk::mapping

#endif // LLK_MAPPING_STORAGELIVENESS_H
