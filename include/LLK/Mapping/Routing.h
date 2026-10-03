//===- Routing.h - Deterministic memory-to-memory routing (D2) ------------===//
//
// Part of the target-independent mapping core (epic #67, workstream D2).
//
// Direct source-to-destination copies are not enough for a memory hierarchy:
// data may stage through one or more intermediate memories, over links that
// each need a transfer engine able to reach them. `TopologyService` answers
// "how can this value get from here to there" with a bounded, cycle-free,
// deterministically ordered set of routes.
//
// A route is a sequence of memory nodes joined by directed links. Legality is
// a topology question -- does the link exist, is there an engine that can see
// its source, does each memory support the requested alignment, does an
// intermediate have room -- and never a comparison of target ids against the
// Micro vocabulary. Cost is the summed per-hop transfer cost; routes are
// ranked by cost, then hop count, then the lexicographic link-id sequence, so
// two runs on the same machine agree exactly.
//
// Materializing a chosen route as copies/waits is a later step (design §12.4,
// D5/plan binding); this layer only enumerates and ranks.
//
//===----------------------------------------------------------------------===//

#ifndef LLK_MAPPING_ROUTING_H
#define LLK_MAPPING_ROUTING_H

#include "LLK/Machine/MachineModel.h"
#include "LLK/Mapping/CostModel.h"
#include "LLK/Mapping/MappingPlan.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <optional>

namespace mlir::llk::mapping {

/// What needs to move, and any endpoint constraints. `alignmentBytes` is the
/// alignment the value requires; every memory on a chosen route must support
/// it (its own alignment must be a multiple of the requirement).
///
/// Design §12.1 also lists an element type. v2 memory and link nodes carry no
/// element-type facts, so checking it here would be vacuous; type support is a
/// rule-level concern (compute capabilities declare supported types).
struct RouteRequest {
  MemoryNodeId source;
  MemoryNodeId destination;
  uint64_t bytes = 0;
  uint64_t alignmentBytes = 1;
  std::optional<ExecutorId> producerExecutor;
  std::optional<ExecutorId> consumerExecutor;
};

/// One legal way to move a value, cheapest-first within an enumeration.
struct MemoryRoute {
  llvm::SmallVector<MemoryNodeId> nodes;
  llvm::SmallVector<LinkId> links;
  llvm::SmallVector<ExecutorId> transferEngines;
  Cost cost;

  uint64_t hopCount() const { return links.size(); }
};

/// Bounds on an enumeration. `maxHops` keeps a large topology from being
/// explored deeply; `maxRoutes` is the default cap a caller may lower.
struct RouteOptions {
  unsigned maxRoutes = 8;
  unsigned maxHops = 4;
};

/// Enumerates routes over one `MachineModel`.
class TopologyService {
public:
  explicit TopologyService(const machine::MachineModel &model,
                           RouteOptions options = {});

  /// Rejected: the service keeps a reference to the model, so binding it to a
  /// temporary would leave it dangling. Name the model instead.
  TopologyService(machine::MachineModel &&model,
                  RouteOptions options = {}) = delete;

  /// Returns up to `limit` routes from `request.source` to
  /// `request.destination`, best first. Fails when the request is malformed,
  /// an endpoint is unknown or unreachable by its executor, or no legal route
  /// exists.
  ///
  /// When `truncated` is non-null it is set to true if the enumeration reached
  /// its effective cap (the smaller of `limit` and `maxRoutes`). This is
  /// deliberately conservative: it may report truncation when exactly that many
  /// routes exist, because the caller is never allowed to claim optimality
  /// after a cap was reached (design §16.2).
  llvm::Expected<llvm::SmallVector<MemoryRoute>>
  enumerateRoutes(const RouteRequest &request, unsigned limit,
                  bool *truncated = nullptr) const;

private:
  const machine::MachineModel &model_;
  RouteOptions options_;
};

} // namespace mlir::llk::mapping

#endif // LLK_MAPPING_ROUTING_H
