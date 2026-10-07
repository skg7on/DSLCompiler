//===- MicroDAG.h - Event DAG extracted from concrete Micro-IR ------------===//
//
// Part of the M10 AVX2 performance simulator (issue #46).
//
// buildMicroDAG() turns one concrete tile-centric `micro.kernel` into the flat
// event list the L1 resource scheduler consumes, plus the program properties
// (live tile bytes, layout/owner warnings) that do not depend on scheduling.
// Every event carries the tile metadata it was derived from -- shape, dtype,
// layout, memory space, owner -- so the cost model can ask "which tile, in
// which memory, owned by whom" instead of "which opcode".
//
// Dependency rules (MVP, analytical rather than cycle-accurate):
//
//   * Data: an op depends on the event that produced each operand, and a
//     `micro.wait` depends on the event that produced each token it waits on.
//   * Program order: an event depends on the immediately preceding event in
//     its region, unless both use the same resource *kind*. Same-class work is
//     separated by its shared resource pool instead, and that is what lets two
//     independent copies overlap when a machine has more than one DMA engine.
//   * Loops are unrolled so trip counts become real iteration costs. A
//     non-pipelined iteration depends on every event of the previous one; a
//     loop whose body is a `micro.pipeline stages >= 2` carries no such edge,
//     so the copy for iteration i+1 may issue while iteration i still computes.
//     A `micro.spatial_for` maps independent work onto resources and carries
//     no edge either; its owner pool bounds how many iterations run at once.
//   * Logical ops: `micro.tile_view` and `micro.tile_partition` are pure
//     metadata and normally produce no event. One is produced only when the
//     result names a layout the operand does not already have, i.e. when the op
//     describes a physical layout transform, which is not free.
//
// Loops with non-static bounds are executed once and reported as a warning,
// because a run of unknown length cannot be charged. A kernel that unrolls past
// kMaxEvents is rejected rather than silently approximated.
//
//===----------------------------------------------------------------------===//

#ifndef LLK_PERF_MICRODAG_H
#define LLK_PERF_MICRODAG_H

#include "LLK/Machine/MachineModel.h"
#include "LLK/Mapping/CostEvent.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <vector>

namespace mlir {
class Operation;
} // namespace mlir

namespace mlir::llk::perf {

/// Upper bound on unrolled events. Reached only by kernels far outside the MVP
/// scope; exceeding it is reported as an error so a cycle estimate is never
/// quietly produced from a truncated graph.
inline constexpr uint64_t kMaxEvents = 100000;

/// One machine-visible action. `Load` and `Barrier` are named for the event
/// families the model is meant to cover but no micro op produces them yet: the
/// dialect currently expresses every read as a copy and every sync as a wait.
/// `Transform` *is* produced -- `micro.transform` performs a real layout
/// conversion, so it is charged instead of falling through the zero-cost path.
enum class EventKind {
  TileView,
  TilePartition,
  Transform,
  AsyncCopy,
  Load,
  Store,
  Mma,
  Vector,
  Reduce,
  Wait,
  Barrier
};

/// The machine resource an event occupies while it executes. Events sharing a
/// kind are ordered by that resource's pool rather than by program order.
///
/// `MemoryRead` and `MemoryWrite` are reserved for memory work that bypasses
/// the DMA path; every movement op the dialect has (async copy, store) goes
/// through the machine's load/store engine and is modeled as `Dma`.
enum class ResourceKind {
  Dma,
  MatrixEngine,
  VectorEngine,
  MemoryRead,
  MemoryWrite,
  Sync
};

llvm::StringRef stringifyEventKind(EventKind kind);
llvm::StringRef stringifyResourceKind(ResourceKind kind);

/// The shared cost category an event belongs to, so the mapping search and the
/// simulator account for the same kinds of work (design §17.2).
mapping::CostEventKind costEventKindOf(EventKind kind);

struct MicroEvent {
  uint32_t id = 0;
  EventKind kind = EventKind::Vector;
  ResourceKind resource = ResourceKind::VectorEngine;

  /// Which instance of the resource: the engine that will run this event --
  /// the one `micro.mma` named, or the machine's first engine of that class --
  /// and `dma` or `sync` for the shared serial resources. Two events with the
  /// same name share a pool; two events naming different engines of the same
  /// class do not, because those are different hardware.
  std::string resourceName;

  /// MACs for `micro.mma`, elements for everything else. `micro.mma` work is
  /// two flops per unit, which is how the L0 bound recovers total flops.
  uint64_t workItems = 0;
  uint64_t bytes = 0;
  uint64_t minCycles = 0;
  std::vector<uint32_t> deps;
  std::string sourceOpName;
  /// The normalized cost category this event belongs to.
  mapping::CostEventKind costKind = mapping::CostEventKind::Compute;

  std::string tileShape;
  std::string tileLayout;
  /// Memory space the movement writes into, or where the compute tile lives.
  std::string tileMemory;
  /// Memory space a movement reads from; empty for compute.
  std::string srcMemory;
  std::string tileOwner;

  /// The owner-occupancy pool this event runs under (issue #129, task R6): the
  /// concrete *executor* a mapped op selected, so the simulator's owner
  /// constraint is the plan's own placement rather than an abstract owner
  /// class. Empty for an unmapped op -- a hand-written kernel, or a movement --
  /// where the tile's owner symbol (`tileOwner`) is the abstract-kind fallback.
  std::string ownerPool;
};

struct MicroDAG {
  std::vector<MicroEvent> events;

  /// Peak materialized tile bytes per memory space, from `micro.alloc` and
  /// `micro.tile_alloc`, async copy results, and layout-transform results.
  /// Pipeline stage count multiplies the tiles allocated inside the body.
  std::map<std::string, uint64_t> liveTileBytesByMemory;

  /// Extraction diagnostics. They describe the kernel/machine pair, not the
  /// schedule, so the report carries them at every level.
  std::vector<std::string> warnings;
  std::vector<std::string> layoutWarnings;
  std::vector<std::string> ownerWarnings;

  /// The extraction facts that make the event stream *incomplete*: a loop whose
  /// trip count is unknown (charged one iteration), an allocation or movement
  /// whose byte count cannot be computed (charged zero). A strict analysis
  /// (`analyzeSelectedKernel(..., requireComplete = true)`) refuses a stream
  /// with any of these rather than present a partial cost as a complete one
  /// (issue #129, task R6). Ordered the way the walk meets them, so two runs of
  /// equal inputs report equal reasons.
  std::vector<std::string> incompleteReasons;
};

/// A route the selected plan chose: the concrete endpoint *nodes* it connects
/// and the links it crosses, in order. The identity a movement is matched by is
/// the one the binder stamps on its copy -- the connection value and the hop's
/// destination node (`micro.value` / `micro.dst_node`); a hand-written kernel
/// that stamps neither falls back to the node pair its endpoint spaces resolve
/// to, and only when a space names several nodes to the plan's route order. The
/// spaces are carried alongside to narrow the candidates.
struct PlannedRoute {
  /// The connection this route carries, when the binder recorded it. Two
  /// movements can share an endpoint-kind pair (`l2 -> sram`), so the node pair
  /// and this value are what tell their routes apart.
  std::optional<uint64_t> value;
  /// False for a route that is not materialized as a movement (a reduce, a
  /// pure layout transform), so it is never charged to a copy op.
  bool moves = true;
  std::string srcNode;
  std::string dstNode;
  std::string srcSpace;
  std::string dstSpace;
  std::vector<const machine::LinkEdge *> hops;
};

/// Measured durations a selected plan recorded for its own work, keyed by the
/// identity the binder stamped on the mapped op (issue #129 review finding 6):
/// `byInstance` on `micro.mapping.instance` for a placement's compute work,
/// `byConnection` on the connection a routed copy records for its movement.
///
/// A hit replaces the static formula's duration for exactly that op's event --
/// work, traffic and capacity are untouched -- so the analysis the planner and
/// `micro-perf` share charges the same calibrated number the search ranked on,
/// instead of overwriting it with the static estimate.
struct MeasuredOverrides {
  llvm::DenseMap<uint64_t, double> byInstance;
  llvm::DenseMap<uint64_t, double> byConnection;
};

/// Builds the event DAG for `kernel` against `machine`. `kernel` must be a
/// `micro.kernel`. Fails when the kernel uses a memory space the machine does
/// not model, or expands past kMaxEvents.
///
/// When `measured` is given, an event whose op the plan recorded a measured
/// duration for charges that duration instead of the machine formula.
llvm::Expected<MicroDAG>
buildMicroDAG(mlir::Operation *kernel, const machine::MachineModel &machine,
              const MeasuredOverrides *measured = nullptr);

/// The normalized view of one DAG event (task B8): the same
/// `mapping::PlanCostEvent` shape `buildPlanEvents` produces, so a plan's
/// selected-plan event stream and its materialized kernel's event stream are
/// directly comparable kind for kind, resource for resource, work and byte for
/// work and byte.
mapping::PlanCostEvent normalizedPlanEvent(const MicroEvent &event);

/// Locates the `micro.kernel` to simulate anywhere under `root`: the one named
/// `symbol`, or the only one present when `symbol` is empty. Fails when the
/// name matches nothing or when the choice is ambiguous, so a multi-kernel file
/// is never simulated by accident.
llvm::Expected<mlir::Operation *> findMicroKernel(mlir::Operation *root,
                                                  llvm::StringRef symbol);

} // namespace mlir::llk::perf

#endif // LLK_PERF_MICRODAG_H
