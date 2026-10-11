//===- MicroCostModel.cpp - L0 static bound and L1 resource scheduler
//------===//
//
// L0 answers "how fast could this kernel possibly run on this machine", by
// dividing total work by peak throughput and total bytes by memory bandwidth.
// It deliberately ignores dependencies, so it is a bound and not a prediction.
//
// L1 answers "how fast does this schedule actually run", by placing every event
// on a resource timeline. The scheduler is a deterministic list scheduler:
// among the events whose dependencies have finished, the lowest event id goes
// first, and it starts when both its dependencies and its resource slots allow.
// Resource multiplicity is modeled with a pool of slots per resource, so a
// machine with four DMA engines really can run four copies at once.
//
// Bandwidth is reported but never scheduled. An event occupies a compute or
// DMA slot; DRAM does not get a slot, so a utilization figure above 1 means the
// schedule outruns memory and the prediction is optimistic.
//
//===----------------------------------------------------------------------===//

#include "LLK/Perf/MicroCostModel.h"

#include "LLK/Mapping/EventSchedule.h"
#include "LLK/Perf/SelectedKernelAnalysis.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/MathExtras.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <map>
#include <string>
#include <utility>
#include <vector>

namespace mlir::llk::perf {
namespace {

/// Bottleneck labels, matching the vocabulary the tuning stage ranks on.
constexpr llvm::StringLiteral kBottleneckUnknown = "unknown";
constexpr llvm::StringLiteral kBottleneckMatrix = "matrix_engine";
constexpr llvm::StringLiteral kBottleneckVector = "vector_engine";
constexpr llvm::StringLiteral kBottleneckDma = "dma";
constexpr llvm::StringLiteral kBottleneckSync = "sync";
constexpr llvm::StringLiteral kBottleneckDependency = "dependency";
constexpr llvm::StringLiteral kBottleneckOwner = "owner_occupancy";
constexpr llvm::StringLiteral kBottleneckLayout = "layout_transform";

llvm::StringRef computeEngineLabel(const MicroEvent &event) {
  if (event.kind == EventKind::TileView ||
      event.kind == EventKind::TilePartition)
    return kBottleneckLayout;
  switch (event.resource) {
  case ResourceKind::MatrixEngine:
    return kBottleneckMatrix;
  case ResourceKind::VectorEngine:
    return kBottleneckVector;
  case ResourceKind::Dma:
    return kBottleneckDma;
  case ResourceKind::Sync:
    return kBottleneckSync;
  case ResourceKind::MemoryRead:
  case ResourceKind::MemoryWrite:
    return kBottleneckDependency;
  }
  return kBottleneckUnknown;
}

/// Total slots a machine offers in one engine class.
template <typename EngineRange>
uint32_t totalSlots(const EngineRange &engines) {
  uint32_t total = 0;
  for (const auto &engine : engines)
    total += engine->concurrency;
  return std::max<uint32_t>(1, total);
}

} // namespace

//===----------------------------------------------------------------------===//
// L0 static bound
//===----------------------------------------------------------------------===//

L0Report computeL0StaticBound(const MicroDAG &dag,
                              const machine::MachineModel &machine) {
  L0Report report;
  report.liveTileBytesByMemory = dag.liveTileBytesByMemory;
  // The one documented traffic convention, shared with the selected-kernel
  // analysis (issue #129, task R6) so the L0 byte block and the analysis's
  // `trafficBytes` are the same numbers.
  report.bytesByMemory = dagTrafficByMemory(dag);

  uint64_t matrixWork = 0;
  uint64_t vectorWork = 0;
  uint64_t syncWork = 0;

  for (const MicroEvent &event : dag.events) {
    switch (event.kind) {
    case EventKind::Mma:
      report.totalFlops += 2 * event.workItems;
      matrixWork += event.minCycles;
      break;
    case EventKind::Vector:
    case EventKind::Reduce:
    case EventKind::Transform:
    case EventKind::TileView:
    case EventKind::TilePartition:
      vectorWork += event.minCycles;
      break;
    case EventKind::Wait:
    case EventKind::Barrier:
      syncWork += event.minCycles;
      break;
    case EventKind::AsyncCopy:
    case EventKind::Load:
    case EventKind::Store:
      break;
    }
  }

  auto bytesAt = [&](llvm::StringRef space) {
    auto it = report.bytesByMemory.find(space.str());
    return it == report.bytesByMemory.end() ? uint64_t{0} : it->second;
  };
  report.totalBytesDram = bytesAt("dram");
  report.totalBytesSram = bytesAt("sram");

  // Work divided by the slots that can run it. The engine pools -- not
  // worker_threads -- are what L1 schedules against, so dividing by anything
  // else could let L0 come out slower than L1 and stop being a bound at all.
  // On the shipped AVX2 model the two agree: eight workers, eight engines.
  const uint64_t matrixSlots =
      totalSlots(machine.computesOfKind("matrix_engine"));
  const uint64_t vectorSlots =
      totalSlots(machine.computesOfKind("vector_engine"));

  // Matrix and vector work run on the pools the machine declares, which may be
  // disjoint, so a bound may assume they overlap. Synchronization does not
  // overlap with anything.
  report.computeCyclesLowerBound =
      std::max(llvm::divideCeil(matrixWork, matrixSlots),
               llvm::divideCeil(vectorWork, vectorSlots)) +
      syncWork;

  // The dominant memory path is the level that needs the longest; a lower bound
  // may assume the levels overlap.
  std::string dominantMemory;
  for (const auto &[space, bytes] : report.bytesByMemory) {
    const machine::MemoryNode *level = machine.findMemoryOfKind(space);
    if (!level || level->bandwidthBytesPerCycle <= 0)
      continue;
    uint64_t cycles =
        level->latencyCycles +
        static_cast<uint64_t>(std::ceil(static_cast<double>(bytes) /
                                        level->bandwidthBytesPerCycle));
    if (cycles > report.memoryCyclesLowerBound) {
      report.memoryCyclesLowerBound = cycles;
      dominantMemory = space;
    }
  }

  report.predictedCycles =
      std::max(report.computeCyclesLowerBound, report.memoryCyclesLowerBound);

  if (dag.events.empty()) {
    report.bottleneck = kBottleneckUnknown.str();
  } else if (report.computeCyclesLowerBound >= report.memoryCyclesLowerBound) {
    report.bottleneck =
        (matrixWork >= vectorWork ? kBottleneckMatrix : kBottleneckVector)
            .str();
  } else {
    report.bottleneck = dominantMemory.empty() ? kBottleneckUnknown.str()
                                               : dominantMemory + "_bandwidth";
  }
  return report;
}

//===----------------------------------------------------------------------===//
// L1 resource schedule
//===----------------------------------------------------------------------===//

L1Report scheduleL1(const MicroDAG &dag, const machine::MachineModel &machine) {
  // Normalize through the shared core (task B8) and schedule the stream, so the
  // performance prediction and the mapping search's candidate score are
  // literally one schedule of one event stream. The owner-occupancy constraint
  // is the event's mapped executor when it has one (issue #129, task R6), and
  // the abstract owner symbol otherwise -- the same fallback the shared
  // selected-kernel analysis uses.
  std::vector<mapping::PlanCostEvent> normalized;
  std::vector<std::string> owners;
  normalized.reserve(dag.events.size());
  owners.reserve(dag.events.size());
  for (const MicroEvent &event : dag.events) {
    mapping::PlanCostEvent entry = normalizedPlanEvent(event);
    owners.push_back(entry.owner);
    normalized.push_back(std::move(entry));
  }
  mapping::EventScheduleResult schedule =
      mapping::scheduleNormalizedEvents(normalized, machine, owners);
  return reportL1FromSchedule(dag, machine, schedule);
}

L1Report reportL1FromSchedule(const MicroDAG &dag,
                              const machine::MachineModel &machine,
                              const mapping::EventScheduleResult &schedule) {
  L1Report report;
  const size_t count = dag.events.size();
  if (count == 0) {
    report.bottleneck = kBottleneckUnknown.str();
    return report;
  }

  std::vector<uint64_t> start(count, 0);
  std::vector<uint64_t> finish(count, 0);
  for (const mapping::ScheduledEvent &entry : schedule.entries) {
    start[entry.id] = entry.start;
    finish[entry.id] = entry.finish;
  }

  const uint64_t predicted = schedule.predictedCycles;
  const uint32_t criticalId = schedule.criticalId;
  const uint64_t sequential = schedule.sequentialCycles;

  uint64_t busyByKind[6] = {0, 0, 0, 0, 0, 0};
  std::map<std::string, uint64_t> busy;
  std::map<std::string, uint32_t> ownerSlots;
  for (const MicroEvent &event : dag.events) {
    busyByKind[static_cast<size_t>(event.resource)] += event.minCycles;
    if (!event.tileOwner.empty()) {
      // Owner occupancy is utilization like any other, and a schedule that runs
      // out of owners is a different bottleneck than one that runs out of
      // engines. It only shows up if the owner's busy cycles are recorded too.
      std::string ownerKey = "owner/" + event.tileOwner;
      busy[ownerKey] += event.minCycles;
      ownerSlots.emplace(
          ownerKey, std::max<uint32_t>(1, machine.ownerCount(event.tileOwner)));
    }
  }

  report.predictedCycles = predicted;
  report.predictedNs = machine.clockHz
                           ? static_cast<double>(predicted) * 1e9 /
                                 static_cast<double>(*machine.clockHz)
                           : 0.0;

  report.schedule.reserve(count);
  for (const MicroEvent &event : dag.events)
    report.schedule.push_back({event.id, start[event.id], finish[event.id]});

  auto utilization = [&](uint64_t busyCycles, uint64_t slots) {
    if (predicted == 0 || slots == 0)
      return 0.0;
    return static_cast<double>(busyCycles) /
           (static_cast<double>(predicted) * static_cast<double>(slots));
  };

  uint64_t matrixSlots = 0;
  for (const machine::ComputeNode *engine :
       machine.computesOfKind("matrix_engine"))
    matrixSlots += engine->concurrency;
  uint64_t vectorSlots = 0;
  for (const machine::ComputeNode *engine :
       machine.computesOfKind("vector_engine"))
    vectorSlots += engine->concurrency;

  report.matrixUtilization = utilization(
      busyByKind[static_cast<size_t>(ResourceKind::MatrixEngine)], matrixSlots);
  report.vectorUtilization = utilization(
      busyByKind[static_cast<size_t>(ResourceKind::VectorEngine)], vectorSlots);
  report.dmaUtilization =
      utilization(busyByKind[static_cast<size_t>(ResourceKind::Dma)],
                  machine.transferEngineCount());

  auto bandwidthUtilization = [&](llvm::StringRef space, uint64_t bytes) {
    const machine::MemoryNode *level = machine.findMemoryOfKind(space);
    if (!level || predicted == 0 || level->bandwidthBytesPerCycle <= 0)
      return 0.0;
    return static_cast<double>(bytes) /
           (static_cast<double>(predicted) * level->bandwidthBytesPerCycle);
  };
  L0Report traffic = computeL0StaticBound(dag, machine);
  report.dramBandwidthUtilization =
      bandwidthUtilization("dram", traffic.totalBytesDram);
  report.sramBandwidthUtilization =
      bandwidthUtilization("sram", traffic.totalBytesSram);

  report.overlapEfficiency =
      sequential == 0 ? 0.0
                      : std::clamp(1.0 - static_cast<double>(predicted) /
                                             static_cast<double>(sequential),
                                   0.0, 1.0);

  // Bottleneck: the busiest resource if any is saturated, otherwise whatever
  // the critical path ends on.
  double worst = 0.0;
  std::string worstLabel;
  auto consider = [&](double value, llvm::StringRef label) {
    if (value > worst) {
      worst = value;
      worstLabel = label.str();
    }
  };
  consider(report.matrixUtilization, kBottleneckMatrix);
  consider(report.vectorUtilization, kBottleneckVector);
  consider(report.dmaUtilization, kBottleneckDma);
  consider(report.dramBandwidthUtilization, "dram_bandwidth");
  consider(report.sramBandwidthUtilization, "sram_bandwidth");
  for (const auto &[key, slots] : ownerSlots) {
    auto it = busy.find(key);
    consider(utilization(it == busy.end() ? 0 : it->second, slots),
             kBottleneckOwner);
  }

  report.bottleneck = worst > 0.75
                          ? worstLabel
                          : computeEngineLabel(dag.events[criticalId]).str();
  return report;
}

//===----------------------------------------------------------------------===//
// Capacity
//===----------------------------------------------------------------------===//

std::vector<std::string> checkCapacity(const MicroDAG &dag,
                                       const machine::MachineModel &machine) {
  std::vector<std::string> violations;
  for (const auto &[space, bytes] : dag.liveTileBytesByMemory) {
    const machine::MemoryNode *level = machine.findMemoryOfKind(space);
    if (!level || bytes <= level->capacityBytes)
      continue;
    violations.push_back(("memory " + space + " requires " +
                          llvm::Twine(bytes) + " bytes but machine has " +
                          llvm::Twine(level->capacityBytes) + " bytes")
                             .str());
  }
  return violations;
}

} // namespace mlir::llk::perf
