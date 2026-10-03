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

#include "llvm/ADT/ArrayRef.h"
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
  /// Two executors are interchangeable only when swapping them cannot change a
  /// binding: same kind, same parent, same attached compute and memory nodes.
  bool reduceSymmetry = true;
  /// Upper bound on the instances one candidate may produce.
  unsigned maxInstances = 64;
  /// Upper bound on the routes one connection may consider.
  unsigned maxRoutesPerConnection = 4;
};

/// Enumerates legal placements of `candidate` on `target`, in machine
/// declaration order. Returns an empty vector when the candidate cannot be
/// placed; fails only on a malformed input, such as a layout requirement that
/// names a layout the target does not declare.
llvm::Expected<std::vector<CandidateInstance>>
enumeratePlacements(const MappingCandidate &candidate,
                    const MappingTarget &target, mlir::MLIRContext &context,
                    const LayoutContext &layoutContext,
                    const PlacementOptions &options = {});

/// One dataflow edge to connect: where the value is produced, where the
/// consumer expects it, and whether the two ends want different layouts.
struct ConnectionRequest {
  InstanceId producer = 0;
  InstanceId consumer = 0;
  WorkloadValueId value = 0;
  MemoryNodeId producerMemory;
  MemoryNodeId consumerMemory;
  std::optional<LayoutId> producerLayout;
  std::optional<LayoutId> consumerLayout;
  uint64_t bytes = 0;
  uint64_t alignmentBytes = 1;
};

/// Synthesizes every legal way to move the value (design §15.2), cheapest
/// shape first: a direct connection, an in-place layout transform, a transfer,
/// a transfer plus transform, and one plan per route D2 found -- so a
/// multi-hop route appears as a transfer whose route has intermediate nodes.
/// An empty result means the pair is incompatible; it does not fail.
llvm::Expected<std::vector<ConnectionPlan>> synthesizeConnections(
    const ConnectionRequest &request, const machine::MachineModel &machine,
    const TopologyService &topology, const PlacementOptions &options = {});

/// Fan-out (design §15.3): when every consumer reads the producer's memory a
/// single shared-read plan carries them all; otherwise each consumer gets its
/// own plan, which is replication.
llvm::Expected<std::vector<ConnectionPlan>> synthesizeFanOut(
    const ConnectionRequest &base, llvm::ArrayRef<InstanceId> consumers,
    llvm::ArrayRef<MemoryNodeId> consumerMemories,
    const machine::MachineModel &machine, const TopologyService &topology,
    const PlacementOptions &options = {});

/// Fan-in (design §15.3): one gather plan collecting several producers into
/// one consumer.
ConnectionPlan synthesizeFanIn(llvm::ArrayRef<InstanceId> producers,
                               InstanceId consumer, WorkloadValueId value,
                               MemoryNodeId consumerMemory, uint64_t bytes);

} // namespace mlir::llk::mapping

#endif // LLK_MAPPING_PLACEMENT_H
