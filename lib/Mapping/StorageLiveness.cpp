//===- StorageLiveness.cpp - Live storage from execution events -----------===//
//
// Task R5 (issue #129). See StorageLiveness.h for the model.
//
//===----------------------------------------------------------------------===//

#include "LLK/Mapping/StorageLiveness.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
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
  bool touched = false;
};

} // namespace

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
    for (const StorageUse &use : events.events[event].storageUses) {
      auto found = indexById.find(use.allocationId);
      if (found == indexById.end())
        return livenessError("storage liveness: event " + llvm::Twine(event) +
                             " touches allocation " +
                             llvm::Twine(use.allocationId) +
                             ", which the plan does not reserve");
      LiveWindow &window = windows[found->second];
      if (!window.touched) {
        // A written value occupies its buffer from the moment its writer
        // starts; a borrowed descriptor from the moment its first reader does.
        window.touched = true;
        window.start = placed.start;
        window.end = placed.finish;
      }
      window.start = std::min(window.start, placed.start);
      window.end = std::max(window.end, placed.finish);
    }
  }
  for (size_t index = 0; index < plan.allocations.size(); ++index)
    if (!windows[index].touched)
      return livenessError(
          "storage liveness: allocation " +
          llvm::Twine(plan.allocations[index].id) + " in memory '" +
          plan.allocations[index].memory +
          "' is touched by no scheduled event, so its live range is unknown");

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

  std::vector<PlanStepEdge> reuseEdges;
  for (size_t index = 0; index < plan.allocations.size(); ++index) {
    if (!plan.allocations[index].aliasOf)
      continue;
    const StorageAllocation &reused = plan.allocations[root[index]];
    const StorageAllocation &alias = plan.allocations[index];
    // The edge that makes the reuse sound: the reused buffer's last real use
    // completes before the new writer begins.
    if (reused.endStep != alias.beginStep)
      reuseEdges.push_back(PlanStepEdge{reused.endStep, alias.beginStep});
  }
  llvm::sort(reuseEdges, [](const PlanStepEdge &lhs, const PlanStepEdge &rhs) {
    return lhs.from != rhs.from ? lhs.from < rhs.from : lhs.to < rhs.to;
  });
  reuseEdges.erase(std::unique(reuseEdges.begin(), reuseEdges.end()),
                   reuseEdges.end());

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
      uint64_t weight = 0;
      if (__builtin_mul_overflow(allocation.bytes,
                                 allocation.simultaneousOccurrences, &weight))
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
