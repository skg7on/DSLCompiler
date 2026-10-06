//===- CostEvent.h - Shared cost-event vocabulary -------------------------===//
//
// Part of the target-independent mapping core (epic #67).
//
// The mapping search and the performance evaluator account for the same kinds
// of machine work, so they have to name them the same way (design §17.2). A
// route the mapper chose and a hop the simulator charges are only comparable
// if both call it a transfer hop.
//
// These five categories are the whole vocabulary. They are deliberately coarse
// -- what the work *is*, not how much of it -- because cost stays
// multi-dimensional until an objective ranks it.
//
//===----------------------------------------------------------------------===//

#ifndef LLK_MAPPING_COSTEVENT_H
#define LLK_MAPPING_COSTEVENT_H

#include "LLK/Mapping/CostModel.h"

#include "llvm/ADT/StringRef.h"

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace mlir::llk::mapping {

/// The normalized categories of machine work both layers account for.
enum class CostEventKind {
  Compute,         ///< rule-local work issued on a compute capability
  TransferHop,     ///< one memory-to-memory hop over a link
  Transform,       ///< a layout conversion
  Synchronization, ///< a wait or a barrier
  Capacity         ///< a resource-occupancy charge
};

llvm::StringRef stringifyCostEventKind(CostEventKind kind);
std::optional<CostEventKind> symbolizeCostEventKind(llvm::StringRef text);

/// One normalized cost event: what kind of work, on which machine resource,
/// for how much. The mapping search emits these from a selected plan; the
/// evaluator emits them from a scheduled kernel.
struct CostEvent {
  CostEventKind kind = CostEventKind::Compute;
  /// The machine resource the work occupies, named the way the machine model
  /// names it (`mxu`, `dma.0`, an executor id).
  std::string resource;
  Cost cost;
};

/// One normalized event with the structural facts a scheduler needs: the
/// category/resource/cycle triple, how many work items and bytes it accounts
/// for, and the events it depends on. The mapping search emits these from a
/// selected plan; the performance DAG emits them from a scheduled kernel
/// (task B8), so both sides are directly comparable.
struct PlanCostEvent {
  CostEvent event;
  /// MACs for a matrix event, elements otherwise -- the same convention
  /// `MicroEvent::workItems` uses.
  uint64_t workItems = 0;
  uint64_t bytes = 0;
  std::vector<uint32_t> deps;
};

/// The normalized event stream of one selected plan, in dependency order.
struct PlanEventDAG {
  std::vector<PlanCostEvent> events;
};

/// The single construction point both event paths use, so a plan event and its
/// materialized counterpart cannot drift in category, resource, cycles, work,
/// or bytes (task B8).
PlanCostEvent makePlanCostEvent(CostEventKind kind, std::string resource,
                                double latencyCycles, uint64_t workItems,
                                uint64_t bytes,
                                std::vector<uint32_t> deps = {});

struct CoveringPlan;

/// Builds the normalized event stream of a selected plan (task B8).
///
/// The plan must carry its execution structure -- the storage plan's step DAG
/// and per-event byte/work facts -- so the events describe the same work the
/// materialized kernel does: one compute event per placement, one transfer
/// event per route hop (plus the wait the movement implies), one transform
/// event per layout conversion, and one synchronization event per barrier. An
/// unknown strict fact (a movement with no route, a transform with no maps, a
/// gather with no declared semantics, a missing step DAG) is an error rather
/// than a silently charged single iteration.
llvm::Expected<PlanEventDAG>
buildPlanEvents(const CoveringPlan &plan, const machine::MachineModel &machine);

/// Schedules a normalized plan-event stream with the *same* resource scheduler
/// the performance evaluator uses (task B8): events share a resource pool by
/// resource name, deps are respected, and the returned `Cost` carries the
/// overlapped critical-path latency plus the schedule's byte total. Defined in
/// `lib/Mapping/EventSchedule.cpp`, the translation unit that owns the shared
/// scheduler, so the plan score and the perf prediction cannot be two different
/// schedules.
///
/// A plan event whose resource the machine does not model is scheduled on the
/// machine's default pool for its kind rather than rejected: the schedule is a
/// cost, and cost never decides legality.
llvm::Expected<Cost> schedulePlanEvents(const PlanEventDAG &dag,
                                        const machine::MachineModel &machine);

} // namespace mlir::llk::mapping

#endif // LLK_MAPPING_COSTEVENT_H
