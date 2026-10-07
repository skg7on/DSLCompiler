//===- StorageLiveness.cpp - Live storage from execution events -----------===//
//
// Task R5 (issue #129). See StorageLiveness.h for the model.
//
//===----------------------------------------------------------------------===//

#include "LLK/Mapping/StorageLiveness.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/Support/Error.h"

#include <algorithm>
#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <vector>

namespace mlir::llk::mapping {

namespace {

llvm::Error livenessError(const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(), message);
}

/// Checked addition over unsigned bytes; an overflow is a rejection rather than
/// a wrapped peak.
bool checkedAdd(uint64_t lhs, uint64_t rhs, uint64_t *result) {
  return __builtin_add_overflow(lhs, rhs, result);
}

/// The live window one allocation's scheduled uses produce. Both ends are
/// event-time instants; the window is half-open (`start` inclusive, `end`
/// exclusive) so a value that is written exactly when the value it replaces is
/// last read does not overlap it.
struct LiveWindow {
  uint64_t start = 0;
  uint64_t end = 0;
  /// The most simultaneous occurrences any one event accounts for this slot.
  /// It comes from the *stream*: each resident occurrence is one `StorageUse`,
  /// so the distinct occurrence ids an event names are how many versions of the
  /// slot it runs over. Zero until the slot is touched.
  uint64_t residency = 0;
  /// The start of the first read, and of the first write. Which one opens the
  /// window is decided by `written`: a buffer is reserved for its write, while
  /// a borrowed descriptor is live from its first read.
  uint64_t readStart = 0;
  uint64_t writeStart = 0;
  bool written = false;
  bool touched = false;
};

} // namespace

std::vector<PlanStepEdge>
requiredReuseEdgesFor(llvm::ArrayRef<StorageAllocation> allocations) {
  llvm::DenseMap<uint64_t, size_t> indexById;
  for (size_t index = 0; index < allocations.size(); ++index)
    indexById[allocations[index].id] = index;
  // Resolve each alias to the root it shares.
  auto rootOf = [&](size_t index) {
    std::vector<size_t> seen;
    while (allocations[index].aliasOf) {
      auto target = indexById.find(*allocations[index].aliasOf);
      if (target == indexById.end())
        break;
      if (llvm::is_contained(seen, index))
        break;
      seen.push_back(index);
      index = target->second;
    }
    return index;
  };
  std::vector<PlanStepEdge> edges;
  for (size_t index = 0; index < allocations.size(); ++index) {
    if (!allocations[index].aliasOf)
      continue;
    const StorageAllocation &reused = allocations[rootOf(index)];
    const StorageAllocation &alias = allocations[index];
    // The edge the reuse relies on: the reused buffer's last read precedes the
    // new writer's step. An in-place update reads and writes at one step, so
    // its edge is a self-step edge -- reported all the same, because it is the
    // statement that the two allocations are one buffer (issue #129, task R5).
    edges.push_back(PlanStepEdge{reused.endStep, alias.beginStep});
  }
  llvm::sort(edges, [](const PlanStepEdge &lhs, const PlanStepEdge &rhs) {
    return lhs.from != rhs.from ? lhs.from < rhs.from : lhs.to < rhs.to;
  });
  edges.erase(std::unique(edges.begin(), edges.end()), edges.end());
  return edges;
}

llvm::Expected<StorageLivenessResult>
analyzeStorageLiveness(const CoveringPlan &plan, const PlanEventDAG &events,
                       const EventScheduleResult &schedule) {
  if (schedule.entries.size() != events.events.size())
    return livenessError(
        "storage liveness: the schedule does not cover the event stream (" +
        llvm::Twine(schedule.entries.size()) + " entries for " +
        llvm::Twine(events.events.size()) + " events)");

  // --- every allocation's live window, from its scheduled uses --------------
  llvm::DenseMap<uint64_t, size_t> indexById;
  for (size_t index = 0; index < plan.allocations.size(); ++index)
    if (!indexById.insert({plan.allocations[index].id, index}).second)
      return livenessError("storage liveness: duplicate allocation id " +
                           llvm::Twine(plan.allocations[index].id));

  std::vector<LiveWindow> windows(plan.allocations.size());
  for (size_t event = 0; event < events.events.size(); ++event) {
    const ScheduledEvent &placed = schedule.entries[event];
    if (placed.id != event)
      return livenessError("storage liveness: schedule entry " +
                           llvm::Twine(event) + " names event " +
                           llvm::Twine(placed.id));
    // The distinct occurrence ids this event names for each allocation, so the
    // residency is read off the stream rather than trusted from a field.
    std::map<size_t, llvm::SmallSet<uint64_t, 8>> occurrences;
    for (const StorageUse &use : events.events[event].storageUses) {
      auto found = indexById.find(use.allocationId);
      if (found == indexById.end())
        return livenessError("storage liveness: event " + llvm::Twine(event) +
                             " touches allocation " +
                             llvm::Twine(use.allocationId) +
                             ", which the plan does not reserve");
      const size_t index = found->second;
      occurrences[index].insert(use.occurrence);
      LiveWindow &window = windows[index];
      if (!window.touched) {
        window.touched = true;
        window.start = placed.start;
        window.end = placed.finish;
        window.readStart = placed.start;
        window.writeStart = placed.start;
      }
      window.end = std::max(window.end, placed.finish);
      if (use.access == StorageAccess::Write) {
        window.written = true;
        window.writeStart = std::min(window.writeStart, placed.start);
      } else {
        window.readStart = std::min(window.readStart, placed.start);
      }
    }
    for (const auto &entry : occurrences) {
      LiveWindow &window = windows[entry.first];
      window.residency =
          std::max<uint64_t>(window.residency, entry.second.size());
    }
  }
  for (size_t index = 0; index < plan.allocations.size(); ++index) {
    LiveWindow &window = windows[index];
    if (!window.touched)
      return livenessError(
          "storage liveness: allocation " +
          llvm::Twine(plan.allocations[index].id) + " in memory '" +
          plan.allocations[index].memory +
          "' is touched by no scheduled event, so its live range is unknown");
    // A written buffer is reserved for its write; a borrowed descriptor is live
    // from its first read. `end` is the completion of the last touch either
    // way.
    window.start = window.written ? window.writeStart : window.readStart;
    if (window.residency == 0)
      window.residency = 1;
  }

  // --- reuse: the plan's own alias relation --------------------------------
  //
  // An allocation that reuses another's storage contributes nothing while its
  // root is live; the decision (and the ordering it needs) was made when the
  // plan was built, and this pass reports the edges that make it sound. A
  // broken or cyclic chain is a defect, not a footprint to guess at.
  std::vector<size_t> root(plan.allocations.size());
  for (size_t index = 0; index < plan.allocations.size(); ++index) {
    size_t current = index;
    llvm::SmallVector<size_t, 8> seen;
    while (plan.allocations[current].aliasOf) {
      if (llvm::is_contained(seen, current))
        return livenessError("storage liveness: cyclic alias chain at "
                             "allocation " +
                             llvm::Twine(plan.allocations[current].id));
      seen.push_back(current);
      auto target = indexById.find(*plan.allocations[current].aliasOf);
      if (target == indexById.end())
        return livenessError("storage liveness: allocation " +
                             llvm::Twine(plan.allocations[current].id) +
                             " reuses unknown allocation " +
                             llvm::Twine(*plan.allocations[current].aliasOf));
      current = target->second;
    }
    root[index] = current;
  }

  std::vector<PlanStepEdge> reuseEdges =
      requiredReuseEdgesFor(plan.allocations);

  // --- peak simultaneous residency ------------------------------------------
  //
  // A storage root -- the allocation a reuse chain bottoms out at -- reserves
  // its bytes while *any* of its aliases is live, so the group's window is the
  // union of its members'. The sum over roots changes only at a group window's
  // start or end, and using half-open windows the maximum is attained at some
  // start, so evaluating every distinct start is exact.
  std::map<size_t, LiveWindow> groups;
  for (size_t index = 0; index < windows.size(); ++index) {
    const size_t base = root[index];
    auto inserted = groups.emplace(base, windows[index]);
    if (inserted.second)
      continue;
    LiveWindow &group = inserted.first->second;
    group.start = std::min(group.start, windows[index].start);
    group.end = std::max(group.end, windows[index].end);
    // The group is as resident as its most-resident member: the root's buffer
    // has to hold every version any alias needs of it.
    group.residency = std::max(group.residency, windows[index].residency);
  }

  std::vector<uint64_t> candidateTimes;
  candidateTimes.reserve(groups.size());
  for (const auto &group : groups)
    candidateTimes.push_back(group.second.start);
  llvm::sort(candidateTimes);
  candidateTimes.erase(
      std::unique(candidateTimes.begin(), candidateTimes.end()),
      candidateTimes.end());

  StorageLivenessResult result;
  for (uint64_t time : candidateTimes) {
    std::map<std::string, uint64_t> live;
    for (const auto &group : groups) {
      const LiveWindow &window = group.second;
      if (time < window.start || time >= window.end)
        continue;
      const StorageAllocation &allocation = plan.allocations[group.first];
      // The residency comes from the stream -- the occurrence ids the events
      // named -- so the peak is a function of what is scheduled, not of a field
      // the caller could disagree with the events about.
      uint64_t weight = 0;
      if (__builtin_mul_overflow(allocation.bytes, window.residency, &weight))
        return livenessError("storage liveness: the footprint of allocation " +
                             llvm::Twine(allocation.id) + " overflows");
      uint64_t &held = live[allocation.memory];
      if (checkedAdd(held, weight, &held))
        return livenessError(
            "storage liveness: live bytes overflow for memory '" +
            allocation.memory + "'");
    }
    for (const auto &entry : live) {
      uint64_t &peak = result.peakBytes[entry.first];
      peak = std::max(peak, entry.second);
    }
  }
  result.requiredReuseEdges = std::move(reuseEdges);
  return result;
}

} // namespace mlir::llk::mapping
