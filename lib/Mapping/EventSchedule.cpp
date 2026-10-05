//===- EventSchedule.cpp - Shared resource scheduler (task B8) ------------===//

#include "LLK/Mapping/EventSchedule.h"

#include "LLK/Machine/MachineModel.h"
#include "LLK/Mapping/CostModel.h"

#include "llvm/ADT/STLExtras.h"

#include <algorithm>
#include <cmath>
#include <map>
#include <string>
#include <vector>

namespace mlir::llk::mapping {

namespace {

/// The cycles one normalized event occupies its resource.
uint64_t eventCycles(const PlanCostEvent &event) {
  // Round up so a fractional estimate is never under-charged.
  return static_cast<uint64_t>(std::ceil(event.event.cost.latencyCycles));
}

/// The resource pool an event occupies and how many slots that pool offers.
/// Each engine id is a separate pool, so two events that name the same engine
/// share its slots rather than each getting the whole machine; a conversion
/// shares the vector pool of the engine it runs on, exactly as the performance
/// DAG's transform events did before both paths normalized.
struct PoolInfo {
  std::string key;
  uint32_t slots = 1;
};

PoolInfo poolFor(const PlanCostEvent &event,
                 const machine::MachineModel &machine) {
  switch (event.event.kind) {
  case CostEventKind::Compute:
  case CostEventKind::Transform: {
    const machine::ComputeNode *engine =
        machine.findCompute(event.event.resource);
    const bool matrix = engine && engine->kind == "matrix_engine";
    PoolInfo info;
    info.key = (matrix ? "matrix/" : "vector/") + event.event.resource;
    info.slots = std::max<uint32_t>(1, engine ? engine->concurrency : 1);
    return info;
  }
  case CostEventKind::TransferHop:
    return {"dma/" + event.event.resource,
            std::max<uint32_t>(1, machine.transferEngineCount())};
  case CostEventKind::Synchronization:
  case CostEventKind::Capacity:
    return {"sync/" + event.event.resource, 1};
  }
  return {"unknown/" + event.event.resource, 1};
}

/// A pool of identical resource slots, each tracking when it frees up.
/// Acquiring takes the slot that frees first, which is what a list scheduler
/// would do.
class ResourcePool {
public:
  explicit ResourcePool(uint32_t slots) : available_(std::max(1u, slots), 0) {}

  uint64_t earliestFree() const {
    return *std::min_element(available_.begin(), available_.end());
  }

  void acquire(uint64_t start, uint64_t duration) {
    auto slot = std::min_element(available_.begin(), available_.end());
    *slot = start + duration;
  }

private:
  std::vector<uint64_t> available_;
};

} // namespace

EventScheduleResult
scheduleNormalizedEvents(llvm::ArrayRef<PlanCostEvent> events,
                         const machine::MachineModel &machine,
                         llvm::ArrayRef<std::string> owners) {
  EventScheduleResult result;
  const size_t count = events.size();
  result.entries.resize(count);
  if (count == 0)
    return result;

  std::vector<uint64_t> start(count, 0);
  std::vector<uint64_t> finish(count, 0);
  std::vector<uint32_t> remaining(count, 0);
  std::vector<std::vector<uint32_t>> dependents(count);

  for (size_t id = 0; id < count; ++id) {
    for (uint32_t dep : events[id].deps) {
      if (dep >= id)
        continue;
      ++remaining[id];
      dependents[dep].push_back(static_cast<uint32_t>(id));
    }
  }

  // Lowest id among the ready set: deterministic and stable across runs.
  std::vector<uint32_t> ready;
  for (size_t id = 0; id < count; ++id)
    if (remaining[id] == 0)
      ready.push_back(static_cast<uint32_t>(id));
  std::make_heap(ready.begin(), ready.end(), std::greater<uint32_t>());

  std::map<std::string, ResourcePool> pools;
  auto poolForKey = [&](const std::string &key,
                        uint32_t slots) -> ResourcePool & {
    auto [it, inserted] = pools.try_emplace(key, slots);
    (void)inserted;
    return it->second;
  };

  const bool ownersParallel = owners.size() == count;
  while (!ready.empty()) {
    std::pop_heap(ready.begin(), ready.end(), std::greater<uint32_t>());
    uint32_t id = ready.back();
    ready.pop_back();
    const PlanCostEvent &event = events[id];

    PoolInfo info = poolFor(event, machine);
    ResourcePool &pool = poolForKey(info.key, info.slots);

    std::string ownerKey;
    ResourcePool *ownerPool = nullptr;
    if (ownersParallel && !owners[id].empty()) {
      ownerKey = "owner/" + owners[id];
      uint32_t slots = std::max<uint32_t>(1, machine.ownerCount(owners[id]));
      ownerPool = &poolForKey(ownerKey, slots);
    }

    uint64_t earliest = 0;
    for (uint32_t dep : event.deps)
      if (dep < id)
        earliest = std::max(earliest, finish[dep]);

    // Both the resource slot and the owner slot must be free at the same time.
    uint64_t begin = earliest;
    for (;;) {
      uint64_t candidate = std::max(begin, pool.earliestFree());
      if (ownerPool)
        candidate = std::max(candidate, ownerPool->earliestFree());
      if (candidate == begin)
        break;
      begin = candidate;
    }

    const uint64_t cycles = eventCycles(event);
    start[id] = begin;
    finish[id] = begin + cycles;
    pool.acquire(begin, cycles);
    if (ownerPool)
      ownerPool->acquire(begin, cycles);

    for (uint32_t dependent : dependents[id])
      if (--remaining[dependent] == 0) {
        ready.push_back(dependent);
        std::push_heap(ready.begin(), ready.end(), std::greater<uint32_t>());
      }
  }

  uint64_t predicted = 0;
  uint32_t criticalId = 0;
  uint64_t sequential = 0;
  for (size_t id = 0; id < count; ++id) {
    if (finish[id] > predicted) {
      predicted = finish[id];
      criticalId = static_cast<uint32_t>(id);
    }
    sequential += eventCycles(events[id]);
    result.entries[id] =
        ScheduledEvent{static_cast<uint32_t>(id), start[id], finish[id]};
  }
  result.predictedCycles = predicted;
  result.criticalId = predicted == 0 ? 0 : criticalId;
  result.sequentialCycles = sequential;
  return result;
}

llvm::Expected<Cost> schedulePlanEvents(const PlanEventDAG &dag,
                                        const machine::MachineModel &machine) {
  EventScheduleResult schedule = scheduleNormalizedEvents(dag.events, machine);

  Cost cost;
  cost.latencyCycles = static_cast<double>(schedule.predictedCycles);
  uint64_t computeBusy = 0;
  uint64_t transferBusy = 0;
  uint64_t totalBytes = 0;
  for (const PlanCostEvent &event : dag.events) {
    totalBytes += event.bytes;
    const uint64_t cycles = eventCycles(event);
    if (event.event.kind == CostEventKind::Compute)
      computeBusy += cycles;
    else if (event.event.kind == CostEventKind::TransferHop)
      transferBusy += cycles;
  }
  cost.localBytes = totalBytes;
  // Aggregate load factors, the same convention `Cost` documents: each busy
  // total over one shared machine window.
  if (std::optional<double> utilization = utilizationEstimate(
          static_cast<double>(computeBusy), machine, machine.workerThreads))
    cost.computeUtilization = *utilization;
  if (std::optional<double> utilization =
          utilizationEstimate(static_cast<double>(transferBusy), machine,
                              machine.transferEngineCount()))
    cost.transferUtilization = *utilization;
  return cost;
}

} // namespace mlir::llk::mapping
