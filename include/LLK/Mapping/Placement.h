//===- Placement.h - Placement and connection synthesis (D5) -------------===//
//
// Part of the target-independent mapping core (epic #67, workstream D5).
//
// Placement turns an unplaced `MappingCandidate` into legal `CandidateInstance`
// objects on concrete machine resources: which executor runs it, which compute
// capability and memory it attaches to, and which layouts its ports satisfy
// (design §15.1). A candidate that cannot be placed yields no instances -- it
// never fails the whole search, because another candidate for the same node
// may still be legal.
//
// Everything is target-independent. Executor requirements are matched by
// abstract owner kind (design §11.5); a rule never names a concrete executor.
//
//===----------------------------------------------------------------------===//

#ifndef LLK_MAPPING_PLACEMENT_H
#define LLK_MAPPING_PLACEMENT_H

#include "LLK/Machine/MachineModel.h"
#include "LLK/Mapping/LlkMap.h"
#include "LLK/Mapping/MappingPlan.h"
#include "LLK/Mapping/MappingTarget.h"
#include "LLK/Mapping/Routing.h"

#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Types.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

namespace mlir {
class MLIRContext;
} // namespace mlir

namespace mlir::llk::mapping {

struct PlacementOptions {
  /// Collapse interchangeable executors to one representative (design §15.1).
  /// A target-declared `equivalent_to` settles interchangeability outright --
  /// the target asserts the two can be swapped without changing a binding, so
  /// a declared group collapses even when its members differ in concurrency.
  /// Otherwise two executors are interchangeable only when swapping them cannot
  /// change a binding or a placement-relevant fact: same kind, parent, logical
  /// coordinates, concurrency, and scheduling class, and identical attached
  /// compute and visible memory nodes. Kept as an explicit switch so a
  /// diagnostic run can enumerate every representative.
  bool reduceSymmetry = true;
  /// Upper bound on the instances one candidate may produce.
  unsigned maxInstances = 64;
  /// Upper bound on the routes one connection may consider.
  unsigned maxRoutesPerConnection = 4;
};

/// Why a candidate produced no legal placement. Reported only when the result
/// is empty, so a caller can attach the stable §22.3 diagnostic code that says
/// *which* requirement could not be met rather than a single "unplaceable".
enum class PlacementFailure {
  /// At least one instance was placed.
  None,
  /// No executor satisfies the candidate's executor requirements.
  NoLegalExecutor,
  /// A required layout has no solution on the machine.
  NoLegalLayout,
  /// An executor matched but attaches no compute node of a required kind.
  UnsupportedComputeFragment,
  /// An executor matched but no visible memory of a required kind exists.
  NoLegalMemory,
};

/// Enumerates legal placements of `candidate` on `target`, in machine
/// declaration order. Returns an empty vector when the candidate cannot be
/// placed; fails only on a malformed input, such as a layout requirement that
/// names a layout the target does not declare.
///
/// When `truncated` is non-null it is set to true if `maxInstances` was
/// reached. The check is conservative: it also fires when the candidate has
/// exactly `maxInstances` legal placements, because a caller may not claim
/// optimality once the cap was touched (design §16.2).
///
/// When `failure` is non-null it is set to the reason the result is empty (or
/// `PlacementFailure::None` when instances were produced). Its value is
/// deterministic: when several executors fail for different reasons, the first
/// failing executor in machine order decides.
llvm::Expected<std::vector<CandidateInstance>> enumeratePlacements(
    const MappingCandidate &candidate, const MappingTarget &target,
    mlir::MLIRContext &context, const LayoutContext &layoutContext,
    const PlacementOptions &options = {}, bool *truncated = nullptr,
    PlacementFailure *failure = nullptr);

/// One dataflow edge to connect: where the value is produced, where the
/// consumer expects it, and whether the two ends want different layouts.
///
/// The optional fields carry what §10.2 direct-compatibility compares. They are
/// deliberately nullable: a caller that does not know a fact leaves it unset
/// and the corresponding check is skipped, never guessed.
struct ConnectionRequest {
  InstanceId producer = 0;
  InstanceId consumer = 0;
  WorkloadValueId value = 0;
  MemoryNodeId producerMemory;
  MemoryNodeId consumerMemory;
  std::optional<LayoutId> producerLayout;
  std::optional<LayoutId> consumerLayout;
  /// The concrete parameterization each endpoint solved for this value's
  /// layout, when it solved one. Two endpoints whose layout *classes* agree may
  /// still hold genuinely different representations (`VW = 4` versus `VW = 8`),
  /// so a transform is required when the classes differ *or* these parameters
  /// do. An unset map (the default, and what a caller that does not know the
  /// parameters leaves) compares equal to any other unset map, so a caller that
  /// only knows class ids keeps the class-only behaviour.
  llvm::StringMap<SearchValue> producerLayoutParameters;
  llvm::StringMap<SearchValue> consumerLayoutParameters;
  /// The moved value's type, as far as it is known. A modelled shaped type
  /// (`tensor`, `memref`, `vector`) states both its element type and its
  /// logical shape; a `!micro.tile` is read through `TileFacts`, which unwraps
  /// its printed head to the tensor it spells, so a tile here contributes the
  /// same element type and static shape an equivalent tensor would.
  mlir::Type elementType;
  /// The consumer port's expected type. When both it and `elementType` expose
  /// an element type or a static shape, they must agree.
  mlir::Type consumerType;
  /// Executor that runs each side, when known; the visibility check needs the
  /// consumer's so it can ask whether the producer's memory is addressable.
  std::optional<ExecutorId> producerExecutor;
  std::optional<ExecutorId> consumerExecutor;
  /// Affine relationship between each port's index space and the value.
  std::optional<mlir::AffineMap> producerMap;
  std::optional<mlir::AffineMap> consumerMap;
  uint64_t bytes = 0;
  uint64_t alignmentBytes = 1;
  /// Live bytes already charged to each memory in the caller's partial plan,
  /// keyed by memory node. A transfer route stages through an intermediate only
  /// when that memory can hold the value on top of this figure (§12.2's
  /// *intermediate capacity and liveness*). Empty means the caller models no
  /// live data, and every intermediate is treated as empty -- the same default
  /// `RouteRequest::liveBytesOnIntermediate` takes.
  llvm::StringMap<uint64_t> intermediateOccupancy;
};

/// True when the two ports can connect directly -- no transfer, no layout
/// transform -- under design §10.2: their element type, logical tile shape,
/// memory visibility, and affine index relation must agree. A property is
/// checked only when both ends state it, so an absent fact can never reject a
/// pair. Affine maps are compared through MLIR's own equality after
/// `simplifyAffineMap`, never by rendering them.
bool portsDirectCompatible(const ConnectionRequest &request,
                           const machine::MachineModel &machine);

/// Synthesizes every legal way to move the value (design §15.2), cheapest
/// shape first: a direct connection, an in-place layout transform, a transfer,
/// a transfer plus transform, and one plan per route D2 found -- so a
/// multi-hop route appears as a transfer whose route has intermediate nodes.
/// An empty result means the pair is incompatible; it does not fail.
///
/// Compatibility follows §10.2: an element type or affine index relation the
/// two ends disagree on rejects the pair; a layout difference becomes a
/// transform; and a consumer that cannot address the producer's memory cannot
/// take a direct connection.
///
/// When `truncated` is non-null it is set to true if a route enumeration hit
/// its cap, so the caller can report truncated search rather than optimality.
llvm::Expected<std::vector<ConnectionPlan>> synthesizeConnections(
    const ConnectionRequest &request, const machine::MachineModel &machine,
    const TopologyService &topology, const PlacementOptions &options = {},
    bool *truncated = nullptr);

/// Fan-out (design §15.3). Every consumer is given by its own fully-specified
/// `ConnectionRequest`, so §10.2 (element type, logical tile shape, visibility,
/// affine index relation) is checked against *each* consumer rather than one
/// representative.
///
/// When every consumer can legally read the producer's placement (a visibility
/// fact, not memory equality) one shared-read `Direct` plan carries them all.
/// Otherwise the consumers replicate: they are grouped by destination memory,
/// and one copy serves each group, labelled `ConnectionKind::Replicate` so its
/// cost and capacity are visible. A group whose every member can still read the
/// producer's memory in place keeps a `Direct` read instead of copying.
///
/// When several copies are legal for a group, the one the declared `objective`
/// prefers is chosen (design §17.1): the comparator is `costLess` under that
/// order, never a hard-coded dimension, so a non-latency objective (for
/// example `dram_bytes`) ranks the copies as it does everywhere else. An exact
/// tie keeps the first alternative, the same stable tie-break `pickBest` uses.
///
/// When `truncated` is non-null it is set to true when a replication route
/// enumeration hit its cap, so the caller can report truncated search rather
/// than optimality.
llvm::Expected<std::vector<ConnectionPlan>> synthesizeFanOut(
    const ConnectionRequest &base, llvm::ArrayRef<ConnectionRequest> consumers,
    const machine::MachineModel &machine, const TopologyService &topology,
    const PlacementOptions &options = {}, bool *truncated = nullptr,
    const ObjectiveOrder &objective = {},
    bool *choseAmongAlternatives = nullptr);

/// Fan-in (design §15.3): one gather plan collecting several producers into one
/// or more consumers that share a destination memory. `feedCost` is the summed
/// cost of moving each producer's value to that memory -- a gather sums its
/// feeds. `bytes` is the gathered tile's size; the search charges its capacity.
ConnectionPlan synthesizeFanIn(llvm::ArrayRef<InstanceId> producers,
                               llvm::ArrayRef<InstanceId> consumers,
                               WorkloadValueId value,
                               MemoryNodeId consumerMemory, uint64_t bytes,
                               const Cost &feedCost = {});

} // namespace mlir::llk::mapping

#endif // LLK_MAPPING_PLACEMENT_H
