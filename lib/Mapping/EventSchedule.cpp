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
/// Each engine id is a separate pool, sized by *that* node's own multiplicity:
/// two events that name the same engine share its slots rather than each
/// getting the whole machine. A machine-wide count would let one named engine
/// claim slots it does not own -- an unused second engine would double the
/// first one's, hiding a serialization the machine really has. A conversion
/// shares the vector pool of the engine it runs on, exactly as the performance
/// DAG's transform events did before both paths normalized.
///
/// An event whose resource the machine does not model gets a single slot: a
/// partial, conservative claim, never the whole machine's count. Strict callers
/// reject such an event through `validateEventResources` instead.
struct PoolInfo {
  std::string key;
  uint32_t slots = 1;
};

PoolInfo poolFor(const PlanCostEvent &event,
                 const machine::MachineModel &machine) {
  switch (event.event.kind) {
  case CostEventKind::Compute:
  case CostEventKind::Transform: {
    // The recorded compute node supplies the concurrency; an unmodelled
    // resource keeps one slot.
    const machine::ComputeNode *engine =
        machine.findCompute(event.event.resource);
    const bool matrix = engine && engine->kind == "matrix_engine";
    PoolInfo info;
    info.key = (matrix ? "matrix/" : "vector/") + event.event.resource;
    info.slots = std::max<uint32_t>(1, engine ? engine->concurrency : 1);
    return info;
  }
  case CostEventKind::TransferHop: {
    // The named engine node's own count governs its pool. Every engine id is
    // its own pool, so `dma.a` is never widened by the machine declaring
    // `dma.b` (or by `dma.a`'s own count being summed across nodes).
    const machine::TransferEngineNode *engine =
        machine.findTransferEngine(event.event.resource);
    return {"dma/" + event.event.resource,
            std::max<uint32_t>(1, engine ? engine->count : 1)};
  }
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

/// How many simultaneous executions an owner-occupancy pool offers. A plan
/// event's owner is the concrete *executor* a placement selected, so the
/// executor's own `concurrency` is its width (issue #129, task R6). An unmapped
/// analysis names an abstract owner *kind* instead (`worker`, `lane`), which
/// resolves through the machine's alias table to the executors refining it.
uint32_t ownerSlots(const machine::MachineModel &machine,
                    llvm::StringRef owner) {
  if (const machine::ExecutorNode *executor = machine.findExecutor(owner))
    return std::max<uint32_t>(1, executor->concurrency);
  return std::max<uint32_t>(1, machine.ownerCount(owner));
}

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

  // A dependency is honored whatever its id order. A plan's normalized events
  // can list a producer after its consumer (a synthesized step DAG is ordered
  // by covered node, not by emission), so the scheduler must not assume an edge
  // points backward -- silently dropping a forward edge would schedule the
  // dependent at zero and corrupt the score.
  for (size_t id = 0; id < count; ++id) {
    for (uint32_t dep : events[id].deps) {
      if (dep == id || dep >= count)
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
      ownerPool = &poolForKey(ownerKey, ownerSlots(machine, owners[id]));
    }

    uint64_t earliest = 0;
    for (uint32_t dep : event.deps)
      if (dep != id && dep < count)
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

uint64_t dramTrafficBytes(llvm::ArrayRef<PlanCostEvent> events,
                          const machine::MachineModel &machine) {
  // A memory is named either by a concrete node id (`dram.0`) or by an abstract
  // kind (`dram`, what the kernel extraction records); both resolve to the
  // node's declared kind, so the two event shapes count identically.
  auto kindOf = [&](llvm::StringRef name) -> llvm::StringRef {
    if (const machine::MemoryNode *node = machine.findMemory(name))
      return node->kind.empty() ? name : llvm::StringRef(node->kind);
    return name;
  };
  uint64_t total = 0;
  for (const PlanCostEvent &event : events) {
    if (event.bytes == 0)
      continue;
    if (!event.srcMemory.empty() && kindOf(event.srcMemory) == "dram")
      total += event.bytes;
    if (!event.dstMemory.empty() && kindOf(event.dstMemory) == "dram")
      total += event.bytes;
  }
  return total;
}

llvm::Expected<Cost> schedulePlanEvents(const PlanEventDAG &dag,
                                        const machine::MachineModel &machine) {
  // The same owner constraints `scheduleL1` applies (issue #129, task R6): an
  // event that names an owner-occupancy pool occupies it, so the plan's score
  // and the kernel's prediction cannot be two different schedules of one
  // stream.
  std::vector<std::string> owners;
  owners.reserve(dag.events.size());
  for (const PlanCostEvent &event : dag.events)
    owners.push_back(event.owner);
  EventScheduleResult schedule =
      scheduleNormalizedEvents(dag.events, machine, owners);

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
  // The DRAM level of the same traffic summary, so a plan's cost and the
  // performance report charge DRAM from one relation (issue #129, task R6).
  cost.dramBytes = dramTrafficBytes(dag.events, machine);
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

namespace {

llvm::Error resourceError(const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(), message);
}

} // namespace

llvm::Error validateEventResources(const PlanEventDAG &dag,
                                   const machine::MachineModel &machine) {
  const size_t count = dag.events.size();

  for (size_t id = 0; id < count; ++id) {
    const PlanCostEvent &event = dag.events[id];

    // The named pool must exist and be usable. The scheduler clamps a missing
    // or zero engine to one slot so a cost can still be produced; a strict
    // caller wants the fact reported instead, because a silently clamped pool
    // would understate the schedule.
    switch (event.event.kind) {
    case CostEventKind::Compute:
    case CostEventKind::Transform: {
      const machine::ComputeNode *node =
          machine.findCompute(event.event.resource);
      if (!node)
        return resourceError("event schedule: event " + llvm::Twine(id) +
                             " names compute resource '" +
                             event.event.resource + "', which machine '" +
                             machine.target + "' does not model");
      if (node->concurrency == 0)
        return resourceError("event schedule: event " + llvm::Twine(id) +
                             " names compute resource '" +
                             event.event.resource +
                             "', which declares concurrency 0; a pool needs at "
                             "least one slot");
      break;
    }
    case CostEventKind::TransferHop: {
      const machine::TransferEngineNode *node =
          machine.findTransferEngine(event.event.resource);
      if (!node)
        return resourceError("event schedule: event " + llvm::Twine(id) +
                             " names transfer engine '" + event.event.resource +
                             "', which machine '" + machine.target +
                             "' does not model");
      if (node->count == 0)
        return resourceError("event schedule: event " + llvm::Twine(id) +
                             " names transfer engine '" + event.event.resource +
                             "', which declares count 0; a pool needs at least "
                             "one slot");
      break;
    }
    case CostEventKind::Synchronization:
    case CostEventKind::Capacity:
      break;
    }

    // Every dependency must be a real, distinct, other event. The scheduler
    // drops a self or out-of-range edge rather than let it corrupt the score;
    // strict input must supply edges that mean something.
    for (size_t index = 0; index < event.deps.size(); ++index) {
      const uint32_t dep = event.deps[index];
      if (dep == id)
        return resourceError("event schedule: event " + llvm::Twine(id) +
                             " depends on itself");
      if (dep >= count)
        return resourceError("event schedule: event " + llvm::Twine(id) +
                             " depends on event " + llvm::Twine(dep) +
                             ", but the stream has " + llvm::Twine(count) +
                             " events");
      for (size_t prior = 0; prior < index; ++prior)
        if (event.deps[prior] == dep)
          return resourceError("event schedule: event " + llvm::Twine(id) +
                               " lists dependency " + llvm::Twine(dep) +
                               " more than once");
    }
  }

  // A cycle leaves its events unready forever; the scheduler would silently
  // start them at zero, so a strict caller rejects the stream instead.
  std::vector<size_t> indegree(count, 0);
  std::vector<std::vector<size_t>> dependents(count);
  for (size_t id = 0; id < count; ++id)
    for (uint32_t dep : dag.events[id].deps) {
      ++indegree[id];
      dependents[dep].push_back(id);
    }

  std::vector<size_t> ready;
  for (size_t id = 0; id < count; ++id)
    if (indegree[id] == 0)
      ready.push_back(id);
  size_t visited = 0;
  while (!ready.empty()) {
    const size_t id = ready.back();
    ready.pop_back();
    ++visited;
    for (size_t dependent : dependents[id])
      if (--indegree[dependent] == 0)
        ready.push_back(dependent);
  }
  if (visited != count)
    return resourceError("event schedule: dependency cycle among the " +
                         llvm::Twine(count) + " events");

  return llvm::Error::success();
}

} // namespace mlir::llk::mapping
