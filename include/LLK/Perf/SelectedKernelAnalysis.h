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
  /// Peak simultaneous live tile bytes per memory space, the kernel's own
  /// reservation (the same relation the storage planner's liveness summary
  /// reports for a plan), from `micro.alloc`/`micro.tile_alloc`, copy results
  /// and conversion results.
  std::map<std::string, uint64_t> peakBytes;
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
llvm::Expected<SelectedKernelAnalysis>
analyzeSelectedKernel(mlir::Operation *kernel,
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
