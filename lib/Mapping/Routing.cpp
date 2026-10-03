//===- Routing.cpp - Deterministic memory-to-memory routing (D2) ----------===//

#include "LLK/Mapping/Routing.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Error.h"

#include <algorithm>
#include <queue>
#include <string>
#include <vector>

namespace mlir::llk::mapping {

namespace {

using machine::LinkEdge;
using machine::MachineModel;
using machine::MemoryNode;
using machine::TransferEngineNode;

llvm::Error routeError(llvm::StringRef message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(), message);
}

/// The DRAM class of memory, spelled as the Micro `MemorySpace` vocabulary
/// spells it. A hop's DRAM cost is keyed on the memory node's declared `kind`,
/// never on a hard-coded node id, so a profile may name its DRAM node anything.
constexpr llvm::StringLiteral kDramMemoryKind = "dram";

bool isDramClass(llvm::StringRef kind) { return kind == kDramMemoryKind; }

/// A memory supports an alignment request when its own alignment is a multiple
/// of the requirement: a 128-byte-aligned memory satisfies a 64-byte request,
/// not the other way round.
bool supportsAlignment(const MemoryNode &memory, uint64_t alignmentBytes) {
  return memory.alignmentBytes % alignmentBytes == 0;
}

/// A hop is legal when its link declares an engine that can reach the link's
/// source memory. The engine's executor must see the memory the data comes
/// from, which is what makes a transfer physically possible.
std::optional<ExecutorId> legalEngine(const MachineModel &model,
                                      const LinkEdge &link) {
  for (const std::string &engineId : link.transferEngines) {
    const TransferEngineNode *engine = model.findTransferEngine(engineId);
    if (engine && model.isVisible(link.source, engine->attachedTo))
      return engineId;
  }
  return std::nullopt;
}

/// One partial route under exploration. `linkIds` duplicates the link ids as
/// strings so the priority queue can tie-break lexicographically without
/// re-rendering them.
struct PartialRoute {
  llvm::SmallVector<MemoryNodeId> nodes;
  llvm::SmallVector<LinkId> links;
  llvm::SmallVector<ExecutorId> engines;
  std::vector<std::string> linkIds;
  double cost = 0.0;
};

/// Min-heap ordering: cheapest first, then fewest hops, then lexicographic
/// link-id sequence. `std::priority_queue` pops the element for which this is
/// "greater", so the comparison is inverted relative to a route ordering.
struct WorstFirst {
  bool operator()(const PartialRoute &lhs, const PartialRoute &rhs) const {
    if (lhs.cost != rhs.cost)
      return lhs.cost > rhs.cost;
    if (lhs.links.size() != rhs.links.size())
      return lhs.links.size() > rhs.links.size();
    return lhs.linkIds > rhs.linkIds;
  }
};

/// Links leaving `memoryId`, in link-id order so expansion is deterministic.
std::vector<const LinkEdge *> outgoing(const MachineModel &model,
                                       llvm::StringRef memoryId) {
  std::vector<const LinkEdge *> result;
  for (const LinkEdge &link : model.links)
    if (link.source == memoryId)
      result.push_back(&link);
  llvm::sort(result, [](const LinkEdge *lhs, const LinkEdge *rhs) {
    return lhs->id < rhs->id;
  });
  return result;
}

} // namespace

TopologyService::TopologyService(const MachineModel &model,
                                 RouteOptions options)
    : model_(model), options_(options) {}

llvm::Expected<llvm::SmallVector<MemoryRoute>>
TopologyService::enumerateRoutes(const RouteRequest &request, unsigned limit,
                                 bool *truncated) const {
  if (request.bytes == 0)
    return routeError("route request: bytes must be positive");
  if (request.alignmentBytes == 0)
    return routeError("route request: alignment_bytes must be positive");

  const MemoryNode *source = model_.findMemory(request.source);
  if (!source)
    return routeError("route request: unknown source memory '" +
                      request.source + "'");
  const MemoryNode *destination = model_.findMemory(request.destination);
  if (!destination)
    return routeError("route request: unknown destination memory '" +
                      request.destination + "'");

  if (request.producerExecutor &&
      !model_.isVisible(request.source, *request.producerExecutor))
    return routeError("executor '" + *request.producerExecutor +
                      "' cannot see memory '" + request.source + "'");
  if (request.consumerExecutor &&
      !model_.isVisible(request.destination, *request.consumerExecutor))
    return routeError("executor '" + *request.consumerExecutor +
                      "' cannot see memory '" + request.destination + "'");

  llvm::SmallVector<MemoryRoute> routes;
  if (source->id == destination->id) {
    // Nothing to move; a route with no hops is the honest answer.
    MemoryRoute trivial;
    trivial.nodes.push_back(source->id);
    routes.push_back(std::move(trivial));
    return routes;
  }

  if (!supportsAlignment(*source, request.alignmentBytes))
    return routeError("memory '" + source->id + "' does not support " +
                      std::to_string(request.alignmentBytes) +
                      "-byte alignment");
  if (!supportsAlignment(*destination, request.alignmentBytes))
    return routeError("memory '" + destination->id + "' does not support " +
                      std::to_string(request.alignmentBytes) +
                      "-byte alignment");

  unsigned effectiveLimit = std::min(limit, options_.maxRoutes);
  if (effectiveLimit == 0)
    return routeError("route request: limit must be positive");

  std::priority_queue<PartialRoute, std::vector<PartialRoute>, WorstFirst> work;
  PartialRoute start;
  start.nodes.push_back(source->id);
  work.push(std::move(start));

  while (!work.empty() && routes.size() < effectiveLimit) {
    PartialRoute current = work.top();
    work.pop();
    const MemoryNodeId currentId = current.nodes.back();

    if (currentId == destination->id) {
      MemoryRoute route;
      route.nodes = current.nodes;
      route.links = current.links;
      route.transferEngines = current.engines;
      route.cost.latencyCycles = current.cost;
      route.cost.localBytes = request.bytes;
      // DRAM bytes: charge the value once per hop whose source or destination
      // is the DRAM-class memory. A route that enters and later leaves DRAM is
      // charged on both hops, so the count is per-hop, not per-route.
      for (size_t hop = 0; hop + 1 < current.nodes.size(); ++hop) {
        const MemoryNode *from = model_.findMemory(current.nodes[hop]);
        const MemoryNode *to = model_.findMemory(current.nodes[hop + 1]);
        if ((from && isDramClass(from->kind)) || (to && isDramClass(to->kind)))
          route.cost.dramBytes += request.bytes;
      }
      // Transfer utilization over the machine's sync window. Left 0 when the
      // machine models no sync period or offers no transfer engine (design
      // §17.2): a missing denominator is not a fabricated one.
      if (std::optional<double> utilization = utilizationEstimate(
              current.cost, model_, model_.transferEngineCount()))
        route.cost.transferUtilization = *utilization;
      routes.push_back(std::move(route));
      continue;
    }

    if (current.links.size() >= options_.maxHops)
      continue;

    for (const LinkEdge *link : outgoing(model_, currentId)) {
      const MemoryNode *next = model_.findMemory(link->destination);
      if (!next)
        continue;
      // Cycle freedom: never revisit a memory already on this path.
      if (llvm::is_contained(current.nodes, next->id))
        continue;
      // Every hop needs a transfer engine that can reach the link's source.
      std::optional<ExecutorId> engine = legalEngine(model_, *link);
      if (!engine)
        continue;
      if (!supportsAlignment(*next, request.alignmentBytes))
        continue;
      // Intermediate storage must hold the value; the destination already
      // exists.
      if (next->id != destination->id && next->capacityBytes < request.bytes)
        continue;

      PartialRoute advanced = current;
      advanced.nodes.push_back(next->id);
      advanced.links.push_back(link->id);
      advanced.linkIds.push_back(link->id);
      if (!llvm::is_contained(advanced.engines, *engine))
        advanced.engines.push_back(*engine);
      advanced.cost +=
          static_cast<double>(link->latencyCycles) +
          static_cast<double>(request.bytes) / link->bandwidthBytesPerCycle;
      work.push(std::move(advanced));
    }
  }

  // Reaching the effective cap means the space was not exhausted. This is
  // deliberately conservative -- it also fires when exactly `effectiveLimit`
  // routes exist -- because §16.2 forbids implying optimality once a cap was
  // touched.
  if (truncated && routes.size() >= effectiveLimit)
    *truncated = true;

  if (routes.empty())
    return routeError("no route from '" + request.source + "' to '" +
                      request.destination + "'");
  return routes;
}

} // namespace mlir::llk::mapping
