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

/// Which way one event touches one storage allocation (issue #129, task R5).
enum class StorageAccess { Read, Write };

/// The largest number of simultaneous occurrences one event enumerates. The
/// event stream spells out one `StorageUse` per resident occurrence so storage
/// liveness can count them, and a residency beyond this bound would make the
/// stream's size a function of a loop bound. Exceeding it is an explicit
/// *incomplete* fact -- an error naming the allocation -- never a silently
/// truncated expansion.
inline constexpr uint64_t kMaxEnumeratedOccurrences = 1u << 12;

llvm::StringRef stringifyStorageAccess(StorageAccess access);

/// One storage slot one event touches: the allocation, the logical *occurrence*
/// of it the access belongs to, and whether the event reads or writes it. An
/// occurrence distinguishes loop iteration, owner and pipeline stage
/// deterministically, which is what lets liveness tell sequential reuse (one
/// buffer reused across occurrences) from simultaneous residency (several
/// buffers live at once). Every field defaults, so an event that predates the
/// storage relation is still a valid value.
struct StorageUse {
  uint64_t allocationId = 0;
  uint64_t occurrence = 0;
  StorageAccess access = StorageAccess::Read;
};

/// One normalized event with the structural facts a scheduler needs: the
/// category/resource/cycle triple, how many work items and bytes it accounts
/// for, and the events it depends on. The mapping search emits these from a
/// selected plan; the performance DAG emits them from a scheduled kernel
/// (task B8), so both sides are directly comparable.
///
/// The fields below the dependency list are the execution facts storage
/// liveness reads (issue #129, task R5): who runs the work, which storage it
/// touches, and which occurrence of the plan it belongs to. They are derived
/// from the plan, so they stay out of every content id; a consumer that ignores
/// them sees exactly the earlier event shape.
struct PlanCostEvent {
  CostEvent event;
  /// MACs for a matrix event, elements otherwise -- the same convention
  /// `MicroEvent::workItems` uses.
  uint64_t workItems = 0;
  uint64_t bytes = 0;
  std::vector<uint32_t> deps;
  /// The executor the work runs on, when the plan records one.
  std::string owner{};
  /// The memory a movement reads from and writes to. Empty for an event that
  /// moves nothing.
  std::string srcMemory{};
  std::string dstMemory{};
  /// The workload node the event accounts for, rendered as its id. Empty for an
  /// event that belongs to no node.
  std::string sourceNode{};
  /// A deterministic id of the logical execution occurrence this event's work
  /// belongs to: loop iteration, owner and pipeline stage folded together.
  /// `0` for an event that names no occurrence.
  uint64_t occurrence = 0;
  /// The plan step that produced this event, when it came from one.
  std::optional<uint64_t> planStep{};
  /// The connection a movement or wait event materializes, and which hop of it
  /// the event is (`hopIndex` is `0` for a single-hop movement and for every
  /// event that is not a hop).
  std::optional<uint64_t> connectionId{};
  uint64_t hopIndex = 0;
  /// The storage slots this event touches, one entry per simultaneously-live
  /// occurrence the event accounts for.
  std::vector<StorageUse> storageUses{};
};

/// Where a plan's normalized event stream came from (issue #129, task R6). A
/// plan whose kernel was bound and analyzed carries the *derived snapshot* the
/// shared selected-kernel analysis produced from that kernel, so its events are
/// the materialized work's own; a plan scored before it was bound -- the
/// search's partial-cost candidate, or a hand-built analysis fixture -- has no
/// snapshot, and its stream is the accumulation fallback. The distinction is
/// reported, never silent: a reader must not treat a fallback stream as the
/// materialized kernel's.
enum class PlanEventSource { Accumulation, Snapshot };

llvm::StringRef stringifyPlanEventSource(PlanEventSource source);

/// The normalized event stream of one selected plan, in dependency order.
struct PlanEventDAG {
  std::vector<PlanCostEvent> events;
  /// Which of the two construction paths produced `events`. Defaults to the
  /// accumulation fallback, the shape every pre-R6 caller assumed.
  PlanEventSource source = PlanEventSource::Accumulation;
};

/// The single construction point both event paths use, so a plan event and its
/// materialized counterpart cannot drift in category, resource, cycles, work,
/// or bytes (task B8).
PlanCostEvent makePlanCostEvent(CostEventKind kind, std::string resource,
                                double latencyCycles, uint64_t workItems,
                                uint64_t bytes,
                                std::vector<uint32_t> deps = {});

struct CoveringPlan;
class WorkloadGraph;

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
///
/// When `graph` is given, each event also records the execution facts storage
/// liveness reads (issue #129, task R5): the owner it runs under, the workload
/// node it accounts for, the storage allocation each operand/result occurrence
/// touches, and the connection and hop a movement belongs to. A carried value's
/// access is attributed to the value the loop carries it from, so the
/// accumulator's live interval reaches the epilogue that reads the loop's
/// result. Without a graph those fields stay empty (the search's scoring path
/// needs only the schedule); every derived field is outside any content id, so
/// the two shapes score identically.
llvm::Expected<PlanEventDAG>
buildPlanEvents(const CoveringPlan &plan, const machine::MachineModel &machine,
                const WorkloadGraph *graph = nullptr);

/// Verifies and attaches `events` to `plan` as its derived analysis snapshot
/// (issue #129, task R6): the normalized stream the shared selected-kernel
/// analysis (`analyzeSelectedKernel`) produced from the kernel this plan was
/// bound to. The stage that binds a plan's kernel -- the search evaluation
/// (task R7) -- calls this; `buildPlanEvents` then reads the snapshot instead
/// of accumulating rule-local estimates, so the plan's final score is the
/// schedule of the work the materialized kernel actually does.
///
/// The snapshot is a *derived* execution fact: it is excluded from
/// `canonicalPlanString`, so attaching it never changes a plan id, and a plan
/// decoded from metadata never carries one -- replay re-derives it or refuses
/// to rank, exactly as it re-derives every other derived fact.
///
/// The attachment is checked, not trusted. The stream must be non-empty, every
/// resource it names must be one the machine models, its dependency edges must
/// be a real acyclic order, and every `planStep`/`connectionId` it records must
/// name a step/connection the plan itself records. A stream that fails any of
/// these is rejected here rather than silently scored as if the kernel did
/// different work.
llvm::Error attachPlanAnalysisEvents(CoveringPlan &plan, PlanEventDAG events,
                                     const machine::MachineModel &machine);

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
