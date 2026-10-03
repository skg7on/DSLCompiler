//===- MicroCostModel.h - L0 static bound and L1 resource scheduling ------===//
//
// Part of the M10 AVX2 performance simulator (issue #46).
//
// Both levels consume the same MicroDAG, so they can never disagree about what
// the kernel does -- only about how good a schedule can be.
//
// L0 is a roofline-style static bound: total work divided by peak throughput,
// bytes divided by memory bandwidth, and the larger of the two. It ignores
// dependencies entirely and is therefore optimistic by construction.
//
// L1 schedules the actual event graph on the machine's resources with a
// deterministic list scheduler: lowest event id among the ready set, starting
// as soon as both its dependencies and its resource slots allow. Its
// prediction is never better than L0's, because L0 assumes perfect overlap.
//
// Diagnostics (capacity, layout, owner) are returned alongside rather than
// inside L1Report, so that `micro-perf --level=0` can report them too.
//
//===----------------------------------------------------------------------===//

#ifndef LLK_PERF_MICROCOSTMODEL_H
#define LLK_PERF_MICROCOSTMODEL_H

#include "LLK/Machine/MachineModel.h"
#include "LLK/Perf/MicroDAG.h"

#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace mlir::llk::perf {

/// Roofline-style static bound over the whole kernel.
struct L0Report {
  /// Two flops per `micro.mma` MAC; vector and reduce work is not counted here
  /// because it is not a multiply-accumulate.
  uint64_t totalFlops = 0;

  /// Bytes crossing each level of the hierarchy. A copy is charged to both of
  /// its endpoints, so a dram -> sram staged GEMM counts A and B against dram
  /// once each, not twice.
  uint64_t totalBytesDram = 0;
  uint64_t totalBytesSram = 0;
  std::map<std::string, uint64_t> bytesByMemory;

  /// Materialized tile bytes that must fit at once, from MicroDAG.
  std::map<std::string, uint64_t> liveTileBytesByMemory;

  /// Compute work divided by the engine slots that can run it, and the bytes of
  /// the busiest memory level. Both assume perfect overlap, and predictedCycles
  /// is the worse of the two.
  uint64_t computeCyclesLowerBound = 0;
  uint64_t memoryCyclesLowerBound = 0;
  uint64_t predictedCycles = 0;

  /// One of the bottleneck strings listed in MicroCostModel.cpp, or the
  /// dominant memory space suffixed with `_bandwidth`.
  std::string bottleneck;
};

/// Where one event landed on the resource timeline.
struct EventSlot {
  uint32_t id = 0;
  uint64_t start = 0;
  uint64_t finish = 0;
};

/// Result of scheduling the event graph on the machine's resources.
struct L1Report {
  uint64_t predictedCycles = 0;
  double predictedNs = 0;

  /// The timeline itself, indexed by event id, so a caller can ask why an event
  /// started when it did instead of only how long the kernel took.
  std::vector<EventSlot> schedule;

  /// Busy cycles divided by the cycles the resource pool was available. The
  /// bandwidth figures are bytes moved divided by what the level could carry in
  /// the predicted time, so they can exceed 1 when the schedule outruns memory.
  double matrixUtilization = 0;
  double vectorUtilization = 0;
  double dmaUtilization = 0;
  double dramBandwidthUtilization = 0;
  double sramBandwidthUtilization = 0;

  /// 1 - predicted / (sum of every event's own cycles), clamped to [0, 1].
  /// Zero means the schedule is fully serial; higher means resources overlap.
  double overlapEfficiency = 0;

  std::string bottleneck;
};

/// Static bound: work and bytes only, no dependencies.
L0Report computeL0StaticBound(const MicroDAG &dag,
                              const machine::MachineModel &machine);

/// Deterministic list schedule over the event graph.
L1Report scheduleL1(const MicroDAG &dag, const machine::MachineModel &machine);

/// Compares peak live tile bytes against modeled capacity. Violations make a
/// candidate illegal, but they do not fail the run.
std::vector<std::string> checkCapacity(const MicroDAG &dag,
                                       const machine::MachineModel &machine);

} // namespace mlir::llk::perf

#endif // LLK_PERF_MICROCOSTMODEL_H
