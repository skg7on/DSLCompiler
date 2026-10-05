//===- MicroDAG.cpp - Event DAG extraction from concrete Micro-IR ---------===//
//
// Implements buildMicroDAG(): the single traversal that turns a concrete
// `micro.kernel` into schedulable events plus the program properties the
// capacity check needs. See MicroDAG.h for the dependency rules.
//
// Costs come from the machine model, never from constants in this file: an
// event's `minCycles` is the machine's own answer for the work the tile
// metadata describes. That is what keeps a second target from needing a second
// extractor.
//
//===----------------------------------------------------------------------===//

#include "LLK/Perf/MicroDAG.h"

#include "MicroTileInfo.h"

#include "LLK/Dialect/Micro/MicroEnums.h"
#include "LLK/Mapping/CostModel.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Error.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <vector>

namespace mlir::llk::perf {
namespace {

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::make_error<llvm::StringError>(message.str(),
                                             llvm::inconvertibleErrorCode());
}

/// The plain shaped type the shared transform-cost model reads. A `!micro.tile`
/// carries its shape and element type the same way a ranked tensor does, but
/// the cost model sits below the dialect, so the tile is presented to it as the
/// shaped type it describes.
mlir::Type shapedCostType(mlir::Type type) {
  if (llvm::isa<mlir::ShapedType>(type))
    return type;
  if (auto tile = llvm::dyn_cast<micro::TileType>(type))
    return mlir::RankedTensorType::get(tile.getShape(), tile.getElementType());
  return type;
}

/// Trip count of a `micro.for`/`micro.spatial_for` when both bounds and the
/// step are constants, which is what a concrete schedule is expected to have.
std::optional<uint64_t> staticTripCount(mlir::Value lower, mlir::Value upper,
                                        mlir::Value step) {
  auto constantOf = [](mlir::Value value) -> std::optional<int64_t> {
    auto constant = value.getDefiningOp<mlir::arith::ConstantOp>();
    if (!constant)
      return std::nullopt;
    if (auto integer = mlir::dyn_cast<mlir::IntegerAttr>(constant.getValue()))
      return integer.getInt();
    return std::nullopt;
  };

  std::optional<int64_t> begin = constantOf(lower);
  std::optional<int64_t> end = constantOf(upper);
  std::optional<int64_t> stride = constantOf(step);
  if (!begin || !end || !stride || *stride <= 0)
    return std::nullopt;
  if (*end <= *begin)
    return uint64_t{0};
  return static_cast<uint64_t>((*end - *begin + *stride - 1) / *stride);
}

} // namespace

//===----------------------------------------------------------------------===//
// Event DAG builder
//===----------------------------------------------------------------------===//

namespace {

/// The executor a mapped op selected, or empty when it carries no
/// `micro.mapping`. Reading it is what lets the cost model reflect the plan's
/// placement rather than the op's intrinsic engine default (design §17.2).
llvm::StringRef mappedExecutor(mlir::Operation &op) {
  auto mapping = op.getAttrOfType<mlir::DictionaryAttr>("micro.mapping");
  if (!mapping)
    return {};
  if (auto executor = mapping.getAs<mlir::StringAttr>("executor"))
    return executor.getValue();
  return {};
}

class DAGBuilder {
public:
  DAGBuilder(const machine::MachineModel &machine, llvm::StringRef kernelName)
      : machine(machine), kernelName(kernelName.str()) {}

  llvm::Expected<MicroDAG> run(mlir::Operation *kernel);

private:
  /// Walk state for one region. `pending` holds dependencies the next event
  /// must take whatever kind it is -- loop-carried ordering that the
  /// same-resource-kind shortcut must not drop.
  struct State {
    bool hasPrevious = false;
    uint32_t previous = 0;
    ResourceKind previousKind = ResourceKind::Dma;
    std::vector<uint32_t> pending;
    std::string owner;
    /// Buffers a body must hold at once: pipeline stages, times the owners a
    /// spatial loop spreads its iterations over.
    uint64_t storageFactor = 1;
  };

  llvm::Error walkBlock(mlir::Block &block, State state,
                        std::vector<uint32_t> &created);
  llvm::Error walkLoop(mlir::Block &body, mlir::Value lower, mlir::Value upper,
                       mlir::Value step, State &outer,
                       std::vector<uint32_t> &created, bool concurrent,
                       llvm::StringRef mapTarget);
  llvm::Error buildOp(mlir::Operation &op, State &state);

  llvm::Error buildLogicalTileOp(mlir::Operation &op, mlir::Value source,
                                 mlir::Value result,
                                 const std::optional<std::string> &owner,
                                 EventKind kind, State &state);
  llvm::Error buildCopyOp(mlir::Operation &op, llvm::StringRef srcMemory,
                          llvm::StringRef dstMemory, const TileInfo &sourceInfo,
                          const TileInfo &resultInfo, EventKind kind,
                          State &state, llvm::ArrayRef<mlir::Value> results);
  llvm::Error noteAllocation(mlir::Operation &op, const TileInfo &info,
                             llvm::StringRef space, const State &state);

  uint32_t addEvent(MicroEvent event, State &state,
                    llvm::ArrayRef<uint32_t> extraDeps = {});
  void noteDiagnostics(const MicroEvent &event);
  void noteWarning(std::string message);
  void noteLayoutUsage(llvm::StringRef engineName,
                       const std::vector<std::string> &supported,
                       llvm::StringRef layout);
  /// Counts a buffer once per allocation site: a loop body holds one copy of
  /// its tiles however many times the loop runs.
  void noteStorage(mlir::Operation &op, llvm::StringRef space, uint64_t bytes,
                   uint64_t storageFactor);
  /// Complains when a value's byte count cannot be computed, so the zero it is
  /// charged is never mistaken for a real measurement.
  void noteUnsizable(const TileInfo &info, mlir::Operation &op);

  /// Cost helpers, all answered by the machine model.
  uint64_t copyCycles(llvm::StringRef src, llvm::StringRef dst,
                      uint64_t bytes) const;
  uint64_t vectorCycles(const machine::ComputeNode *engine,
                        llvm::StringRef dtype, uint64_t elements) const;
  uint64_t mmaCycles(const machine::ComputeNode &engine,
                     llvm::ArrayRef<int64_t> shape, uint64_t flops) const;

  /// `executor` is the executor the plan mapped this op to, or empty when the
  /// op carries no `micro.mapping`. When set, the engine attached to it is
  /// preferred over the machine's first engine of the kind, so the cost model
  /// reflects the placement the plan chose rather than an intrinsic default.
  const machine::ComputeNode *
  pickMatrixEngine(llvm::StringRef requested, std::string &reason,
                   llvm::StringRef executor = "") const;
  const machine::ComputeNode *
  pickVectorEngine(std::string &reason, llvm::StringRef executor = "") const;

  llvm::SmallVector<uint32_t, 4> producerDeps(mlir::ValueRange values) const;
  void inheritProducer(mlir::Value result, mlir::Value source);
  std::string memoryOf(mlir::Value value) const;

  /// Resolves a movement endpoint, rejecting a space the machine does not model
  /// because no bandwidth or latency would be available for it.
  llvm::Error requireMemory(mlir::Operation &op, llvm::StringRef space) const;
  llvm::Error requireReachable(mlir::Operation &op, llvm::StringRef src,
                               llvm::StringRef dst);

  /// Reads `micro.routes` from a mapped kernel, so a routed movement is
  /// charged hop by hop instead of endpoint to endpoint.
  void loadRoutes(mlir::Operation *kernel);
  /// The links a movement is charged, matched by the most specific identity
  /// available: the destination node and connection value the binder stamps on
  /// a per-hop copy, then the connection value alone, then the node pair the
  /// endpoint spaces resolve to, and only when a space names several nodes the
  /// plan's route order. Empty when the movement is unrouted; a whole-route
  /// match claims its route so a second op cannot take it. The decision is
  /// memoized per op, so a copy a loop unrolls keeps its route on every
  /// iteration.
  llvm::SmallVector<const machine::LinkEdge *, 4>
  matchRoute(mlir::Operation &op, llvm::StringRef srcSpace,
             llvm::StringRef dstSpace);
  /// The single machine node of `kind`, or nullptr when the machine declares
  /// none or more than one -- a kind that names several nodes cannot pin a
  /// movement's endpoint on its own.
  const machine::MemoryNode *soleMemoryOfKind(llvm::StringRef kind) const;

  const machine::MachineModel &machine;
  std::string kernelName;
  std::vector<PlannedRoute> routes;
  /// Each declared route, by the connection id it was declared under. A stamped
  /// copy records its connection id and hop index, so the perf model can charge
  /// exactly that hop rather than a destination kind or the first link of the
  /// endpoint-kind pair -- which is what keeps two movements between distinct
  /// memories of one abstract kind apart.
  llvm::DenseMap<uint64_t, size_t> routeForConnection;
  std::vector<bool> routeClaimed;
  /// The links each movement op resolved to, so unrolled iterations of the
  /// same op do not each consume a fresh route (an empty result is a
  /// legitimate "this op is unrouted" answer and is cached too).
  llvm::DenseMap<mlir::Operation *,
                 llvm::SmallVector<const machine::LinkEdge *, 4>>
      routeOfOp;
  MicroDAG dag;
  llvm::DenseMap<mlir::Value, uint32_t> producers;
  llvm::SmallPtrSet<mlir::Operation *, 32> countedStorage;
  llvm::StringSet<> seenWarnings;
  std::map<std::string, uint64_t> liveBytes;
};

//===----------------------------------------------------------------------===//
// Diagnostics
//===----------------------------------------------------------------------===//

void DAGBuilder::noteWarning(std::string message) {
  if (seenWarnings.insert(message).second)
    dag.warnings.push_back(std::move(message));
}

void DAGBuilder::noteDiagnostics(const MicroEvent &event) {
  if (!event.tileOwner.empty() && !machine.hasOwnerKind(event.tileOwner)) {
    std::string message = "owner '" + event.tileOwner +
                          "' is not modeled by machine '" + machine.target +
                          "'";
    if (seenWarnings.insert(message).second)
      dag.ownerWarnings.push_back(std::move(message));
  }

  if (event.tileLayout.empty() || event.tileMemory.empty())
    return;
  const machine::MemoryNode *level = machine.findMemoryOfKind(event.tileMemory);
  if (!level || llvm::is_contained(level->supportedLayouts, event.tileLayout))
    return;
  std::string message =
      "memory '" + event.tileMemory + "' does not support layout '" +
      event.tileLayout +
      "' (supported: " + llvm::join(level->supportedLayouts, ", ") + ")";
  if (seenWarnings.insert(message).second)
    dag.layoutWarnings.push_back(std::move(message));
}

void DAGBuilder::noteLayoutUsage(llvm::StringRef engineName,
                                 const std::vector<std::string> &supported,
                                 llvm::StringRef layout) {
  if (layout.empty() || llvm::is_contained(supported, layout))
    return;
  std::string message = "engine '" + engineName.str() +
                        "' does not support layout '" + layout.str() +
                        "' (supported: " + llvm::join(supported, ", ") + ")";
  if (seenWarnings.insert(message).second)
    dag.layoutWarnings.push_back(std::move(message));
}

void DAGBuilder::noteUnsizable(const TileInfo &info, mlir::Operation &op) {
  // A value whose byte count this model cannot compute is charged as zero
  // rather than given an invented size -- but never quietly. Two ways to get
  // there: an extent that is dynamic, and an element type outside the micro
  // dtype vocabulary, which an f64 load would otherwise move for free.
  llvm::StringRef opName = op.getName().getStringRef();
  if (info.hasDynamicShape())
    noteWarning(opName.str() +
                " moves or allocates a value with a dynamic extent; its bytes "
                "are charged as zero");
  else if (info.elements() > 0 && info.bytes() == 0)
    noteWarning(opName.str() +
                " moves or allocates a value whose element type is not a micro "
                "dtype; its bytes are charged as zero");
}

void DAGBuilder::noteStorage(mlir::Operation &op, llvm::StringRef space,
                             uint64_t bytes, uint64_t storageFactor) {
  // DRAM holds the buffers the kernel reads from and writes back to, which the
  // caller owns -- `micro.alloc` refuses to target it for the same reason. Only
  // storage the kernel itself materializes can overflow a machine.
  if (space.empty() || space == "dram" || !countedStorage.insert(&op).second)
    return;
  liveBytes[space.str()] += bytes * storageFactor;
}

//===----------------------------------------------------------------------===//
// Costs
//===----------------------------------------------------------------------===//

void DAGBuilder::loadRoutes(mlir::Operation *kernel) {
  auto declared = kernel->getAttrOfType<mlir::ArrayAttr>("micro.routes");
  if (!declared)
    return;
  for (mlir::Attribute entry : declared) {
    auto route = mlir::dyn_cast<mlir::DictionaryAttr>(entry);
    if (!route)
      continue;
    auto nodes = route.getAs<mlir::ArrayAttr>("route");
    if (!nodes || nodes.size() < 2)
      continue;

    PlannedRoute built;
    if (auto value = route.getAs<mlir::IntegerAttr>("value"))
      built.value = static_cast<uint64_t>(value.getInt());
    // Only transfer connections become copies (design §18.2); a reduce or a
    // pure layout transform is not a movement and must not be matched to one.
    // A hand-written entry with no `kind` is taken as a movement route.
    if (auto kind = route.getAs<mlir::StringAttr>("kind"))
      built.moves = kind.getValue() == "transfer" ||
                    kind.getValue() == "transfer_and_transform";
    bool complete = true;
    for (size_t i = 1; i < nodes.size(); ++i) {
      std::string from =
          mlir::cast<mlir::StringAttr>(nodes[i - 1]).getValue().str();
      std::string to = mlir::cast<mlir::StringAttr>(nodes[i]).getValue().str();
      const machine::LinkEdge *link = nullptr;
      for (const machine::LinkEdge &candidate : machine.links)
        if (candidate.source == from && candidate.destination == to) {
          link = &candidate;
          break;
        }
      if (!link) {
        complete = false;
        break;
      }
      built.hops.push_back(link);
    }
    if (!complete || built.hops.empty())
      continue;

    // The endpoints are the route's first and last nodes, whatever the hops
    // say: the route is identified by the pair it connects, not by the kinds a
    // movement is spelled with.
    built.srcNode = mlir::cast<mlir::StringAttr>(nodes[0]).getValue().str();
    built.dstNode =
        mlir::cast<mlir::StringAttr>(nodes[nodes.size() - 1]).getValue().str();
    const machine::MemoryNode *source = machine.findMemory(built.srcNode);
    const machine::MemoryNode *destination = machine.findMemory(built.dstNode);
    if (!source || !destination)
      continue;
    built.srcSpace = source->kind;
    built.dstSpace = destination->kind;
    auto connectionId = route.getAs<mlir::IntegerAttr>("id");
    routes.push_back(std::move(built));
    if (connectionId)
      routeForConnection[static_cast<uint64_t>(
          connectionId.getValue().getZExtValue())] = routes.size() - 1;
  }
  routeClaimed.assign(routes.size(), false);
}

const machine::MemoryNode *
DAGBuilder::soleMemoryOfKind(llvm::StringRef kind) const {
  const machine::MemoryNode *found = nullptr;
  for (const machine::MemoryNode &node : machine.memories)
    if (node.kind == kind) {
      if (found)
        return nullptr; // several nodes of this kind: not an identity
      found = &node;
    }
  return found;
}

llvm::SmallVector<const machine::LinkEdge *, 4>
DAGBuilder::matchRoute(mlir::Operation &op, llvm::StringRef srcSpace,
                       llvm::StringRef dstSpace) {
  auto memo = routeOfOp.find(&op);
  if (memo != routeOfOp.end())
    return memo->second;

  llvm::SmallVector<const machine::LinkEdge *, 4> charged;

  // Most specific: the connection id and hop index the binder stamps on a
  // per-hop copy. Together they name exactly one hop of exactly one route, so a
  // movement between two distinct memories of one abstract kind -- which no
  // destination *kind* and no first-link-of-the-kind-pair heuristic can tell
  // apart -- is charged its own link.
  if (auto connection = op.getAttrOfType<mlir::IntegerAttr>("micro.connection"))
    if (auto hop = op.getAttrOfType<mlir::IntegerAttr>("micro.hop")) {
      auto found =
          routeForConnection.find(connection.getValue().getZExtValue());
      if (found != routeForConnection.end()) {
        const PlannedRoute &route = routes[found->second];
        uint64_t index = hop.getValue().getZExtValue();
        if (route.moves && index >= 1 && index <= route.hops.size()) {
          charged.push_back(route.hops[index - 1]);
          routeOfOp[&op] = charged;
          return charged;
        }
      }
    }

  auto stampNode = op.getAttrOfType<mlir::StringAttr>("micro.dst_node");
  auto stampValue = op.getAttrOfType<mlir::IntegerAttr>("micro.value");
  auto valueOf = [&](const PlannedRoute &route) {
    return stampValue && route.value &&
           *route.value == static_cast<uint64_t>(stampValue.getInt());
  };

  // Most specific: a per-hop copy the binder stamped. It names the memory node
  // the hop lands in, so charge exactly that link -- the one hop of the route
  // whose destination is the stamped node and whose source kind is this
  // movement's, disambiguated by the connection value. This is what lets a
  // multi-hop chain, materialized as one copy per hop, be charged hop by hop
  // rather than falling through to the first link of the endpoint kind pair.
  if (stampNode) {
    for (const PlannedRoute &route : routes) {
      if (!route.moves || (stampValue && !valueOf(route)))
        continue;
      for (const machine::LinkEdge *link : route.hops) {
        if (link->destination != stampNode.getValue())
          continue;
        const machine::MemoryNode *from = machine.findMemory(link->source);
        if (from && from->kind != srcSpace)
          continue; // a different leg into the same node
        charged.push_back(link);
        break;
      }
      if (!charged.empty())
        break;
    }
    if (!charged.empty()) {
      routeOfOp[&op] = charged;
      return charged;
    }
  }

  llvm::SmallVector<size_t, 2> candidates;
  for (size_t index = 0; index < routes.size(); ++index)
    if (!routeClaimed[index] && routes[index].moves &&
        routes[index].srcSpace == srcSpace &&
        routes[index].dstSpace == dstSpace)
      candidates.push_back(index);

  if (!candidates.empty()) {
    std::optional<size_t> chosen;

    // The connection value on the movement, for a copy that carries the route
    // as a whole rather than one stamped hop.
    if (stampValue)
      for (size_t index : candidates)
        if (valueOf(routes[index])) {
          chosen = index;
          break;
        }

    // Endpoint spaces that each name exactly one machine node pin the route's
    // node pair without ambiguity.
    if (!chosen) {
      const machine::MemoryNode *source = soleMemoryOfKind(srcSpace);
      const machine::MemoryNode *destination = soleMemoryOfKind(dstSpace);
      if (source && destination)
        for (size_t index : candidates)
          if (routes[index].srcNode == source->id &&
              routes[index].dstNode == destination->id) {
            chosen = index;
            break;
          }
    }

    // A space with several nodes (a machine that declares `sram.0`/`sram.1`)
    // leaves an unstamped movement without a node identity of its own. The
    // binder emits the plan's routes and their copies in the same
    // `connectionPlans` order, so the movements consume the matching routes in
    // that order.
    if (!chosen)
      chosen = candidates.front();

    routeClaimed[*chosen] = true;
    charged.append(routes[*chosen].hops.begin(), routes[*chosen].hops.end());
  }

  routeOfOp[&op] = charged;
  return charged;
}

uint64_t DAGBuilder::copyCycles(llvm::StringRef src, llvm::StringRef dst,
                                uint64_t bytes) const {
  const machine::MemoryNode *srcLevel = machine.findMemoryOfKind(src);
  const machine::MemoryNode *dstLevel = machine.findMemoryOfKind(dst);
  const machine::LinkEdge *path = machine.findLinkByKinds(src, dst);

  // A copy waits on the slower endpoint's latency; the destination's port sets
  // the rate unless the machine models the path separately.
  uint64_t latency = 0;
  if (srcLevel)
    latency = std::max(latency, srcLevel->latencyCycles);
  if (dstLevel)
    latency = std::max(latency, dstLevel->latencyCycles);
  double bandwidth = dstLevel ? dstLevel->bandwidthBytesPerCycle : 0.0;

  if (path) {
    if (path->latencyCycles)
      latency = path->latencyCycles;
    if (path->bandwidthBytesPerCycle > 0)
      bandwidth = path->bandwidthBytesPerCycle;
  }

  const machine::TransferEngineNode *engine = machine.primaryTransferEngine();
  uint64_t cycles = latency + (engine ? engine->setupCycles : 0);
  if (bandwidth > 0)
    cycles += static_cast<uint64_t>(
        std::ceil(static_cast<double>(bytes) / bandwidth));
  return cycles;
}

uint64_t DAGBuilder::vectorCycles(const machine::ComputeNode *engine,
                                  llvm::StringRef dtype,
                                  uint64_t elements) const {
  if (!engine)
    return elements;
  // A dtype the engine does not declare is issued one element at a time. That
  // is slower than the hardware, never faster, so it cannot hide a bottleneck.
  int64_t lanes = 1;
  auto it = engine->lanes.find(dtype.str());
  if (it != engine->lanes.end() && it->second > 0)
    lanes = it->second;
  uint64_t issues = llvm::divideCeil(elements, static_cast<uint64_t>(lanes));
  return issues * std::max<uint64_t>(1, engine->issueCycles);
}

uint64_t DAGBuilder::mmaCycles(const machine::ComputeNode &engine,
                               llvm::ArrayRef<int64_t> shape,
                               uint64_t flops) const {
  if (engine.throughputPerCycle && *engine.throughputPerCycle > 0)
    return static_cast<uint64_t>(
        std::ceil(static_cast<double>(flops) / *engine.throughputPerCycle));

  // No throughput figure: count how many declared fragments the shape needs.
  // A v2 capability may declare fragments of any rank; only the ones matching
  // the MMA's rank can be counted against it.
  auto volume = [](llvm::ArrayRef<int64_t> fragment) {
    int64_t product = 1;
    for (int64_t extent : fragment)
      product *= extent;
    return product;
  };
  const std::vector<int64_t> *fragment = nullptr;
  for (const std::vector<int64_t> &candidate : engine.shapes)
    if (candidate.size() == shape.size() &&
        (!fragment || volume(candidate) > volume(*fragment)))
      fragment = &candidate;
  if (!fragment)
    return std::max<uint64_t>(1, engine.issueCycles);

  uint64_t issues = 1;
  for (size_t i = 0; i < shape.size(); ++i)
    issues *= llvm::divideCeil(static_cast<uint64_t>(shape[i]),
                               static_cast<uint64_t>((*fragment)[i]));
  return issues * std::max<uint64_t>(1, engine.issueCycles);
}

const machine::ComputeNode *
DAGBuilder::pickMatrixEngine(llvm::StringRef requested, std::string &reason,
                             llvm::StringRef executor) const {
  if (!requested.empty()) {
    if (const machine::ComputeNode *engine = machine.findCompute(requested))
      return engine;
    reason = "engine '" + requested.str() + "' is not declared by machine '" +
             machine.target + "'";
    return nullptr;
  }
  // A mapped op runs on the executor the plan selected, so prefer the matrix
  // engine attached to it over the machine's declaration-order first: the
  // placement is a modelled decision, and a second engine on another executor
  // must not be treated as interchangeable with it.
  if (!executor.empty())
    for (const machine::ComputeNode *node : machine.computesFor(executor))
      if (node->kind == "matrix_engine")
        return node;
  std::vector<const machine::ComputeNode *> engines =
      machine.computesOfKind("matrix_engine");
  if (engines.empty()) {
    reason = "machine '" + machine.target + "' declares no matrix engine";
    return nullptr;
  }
  return engines.front();
}

const machine::ComputeNode *
DAGBuilder::pickVectorEngine(std::string &reason,
                             llvm::StringRef executor) const {
  // As above: the mapped executor's own vector engine wins over the machine's
  // first, so two engines on two executors are not conflated.
  if (!executor.empty())
    for (const machine::ComputeNode *node : machine.computesFor(executor))
      if (node->kind == "vector_engine")
        return node;
  std::vector<const machine::ComputeNode *> engines =
      machine.computesOfKind("vector_engine");
  if (engines.empty()) {
    reason = "machine '" + machine.target + "' declares no vector engine";
    return nullptr;
  }
  return engines.front();
}

//===----------------------------------------------------------------------===//
// Producers
//===----------------------------------------------------------------------===//

llvm::SmallVector<uint32_t, 4>
DAGBuilder::producerDeps(mlir::ValueRange values) const {
  llvm::SmallVector<uint32_t, 4> deps;
  for (mlir::Value value : values) {
    auto it = producers.find(value);
    if (it != producers.end())
      deps.push_back(it->second);
  }
  return deps;
}

void DAGBuilder::inheritProducer(mlir::Value result, mlir::Value source) {
  auto it = producers.find(source);
  if (it != producers.end())
    producers[result] = it->second;
}

std::string DAGBuilder::memoryOf(mlir::Value value) const {
  TileInfo info = describeType(value.getType());
  if (!info.memory.empty())
    return info.memory;
  // A logical view carries no memory of its own; the event that produced or
  // last moved the underlying data does.
  auto it = producers.find(value);
  if (it != producers.end())
    return dag.events[it->second].tileMemory;
  return {};
}

//===----------------------------------------------------------------------===//
// Event insertion
//===----------------------------------------------------------------------===//

uint32_t DAGBuilder::addEvent(MicroEvent event, State &state,
                              llvm::ArrayRef<uint32_t> extraDeps) {
  if (event.tileOwner.empty())
    event.tileOwner = state.owner;

  // Every event carries its shared cost category, so a report and a plan can
  // be compared category by category rather than event by event.
  event.costKind = costEventKindOf(event.kind);

  event.id = static_cast<uint32_t>(dag.events.size());
  ResourceKind kind = event.resource;

  // Program order, except between two events that share a resource kind: those
  // are ordered by their pool, and dropping the edge is what lets independent
  // same-class work overlap.
  if (state.hasPrevious && state.previousKind != kind)
    event.deps.push_back(state.previous);
  llvm::append_range(event.deps, state.pending);
  llvm::append_range(event.deps, extraDeps);

  llvm::sort(event.deps);
  event.deps.erase(std::unique(event.deps.begin(), event.deps.end()),
                   event.deps.end());
  event.deps.erase(std::remove(event.deps.begin(), event.deps.end(), event.id),
                   event.deps.end());

  noteDiagnostics(event);
  dag.events.push_back(std::move(event));

  state.previous = dag.events.back().id;
  state.previousKind = kind;
  state.hasPrevious = true;
  state.pending.clear();
  return state.previous;
}

//===----------------------------------------------------------------------===//
// Region walk
//===----------------------------------------------------------------------===//

llvm::Error DAGBuilder::walkBlock(mlir::Block &block, State state,
                                  std::vector<uint32_t> &created) {
  for (mlir::Operation &op : block) {
    if (llvm::isa<micro::YieldOp>(op))
      continue;

    if (auto forOp = llvm::dyn_cast<micro::ForOp>(op)) {
      State outer = state;
      if (llvm::Error err =
              walkLoop(forOp.getBody().front(), forOp.getLowerBound(),
                       forOp.getUpperBound(), forOp.getStep(), outer, created,
                       /*concurrent=*/false, llvm::StringRef()))
        return err;
      state = std::move(outer);
      continue;
    }

    if (auto spatialOp = llvm::dyn_cast<micro::SpatialForOp>(op)) {
      std::string target =
          micro::stringifyMappingTarget(spatialOp.getMap()).str();
      State outer = state;
      if (llvm::Error err =
              walkLoop(spatialOp.getBody().front(), spatialOp.getLowerBound(),
                       spatialOp.getUpperBound(), spatialOp.getStep(), outer,
                       created, /*concurrent=*/true, target))
        return err;
      state = std::move(outer);
      continue;
    }

    if (auto pipelineOp = llvm::dyn_cast<micro::PipelineOp>(op)) {
      State inner = state;
      inner.storageFactor =
          state.storageFactor * std::max<uint64_t>(1, pipelineOp.getStages());
      std::vector<uint32_t> pipelineEvents;
      if (llvm::Error err =
              walkBlock(pipelineOp.getBody().front(), inner, pipelineEvents))
        return err;
      created.insert(created.end(), pipelineEvents.begin(),
                     pipelineEvents.end());
      // Everything after the pipeline waits for all of it.
      state.pending = std::move(pipelineEvents);
      state.hasPrevious = false;
      continue;
    }

    size_t before = dag.events.size();
    if (llvm::Error err = buildOp(op, state))
      return err;
    for (size_t id = before; id < dag.events.size(); ++id)
      created.push_back(static_cast<uint32_t>(id));
  }
  return llvm::Error::success();
}

llvm::Error DAGBuilder::walkLoop(mlir::Block &body, mlir::Value lower,
                                 mlir::Value upper, mlir::Value step,
                                 State &outer, std::vector<uint32_t> &created,
                                 bool concurrent, llvm::StringRef mapTarget) {
  uint64_t trips = 1;
  if (std::optional<uint64_t> count = staticTripCount(lower, upper, step))
    trips = *count;
  else
    noteWarning("a loop has non-static bounds; the simulator runs a single "
                "iteration for it");

  if (trips == 0)
    return llvm::Error::success();

  // A `micro.pipeline stages >= 2` directly inside the loop is what lets a copy
  // for the next iteration issue while this one still computes.
  bool pipelined = false;
  for (mlir::Operation &op : body) {
    auto pipelineOp = llvm::dyn_cast<micro::PipelineOp>(op);
    if (pipelineOp && pipelineOp.getStages() >= 2) {
      pipelined = true;
      break;
    }
  }
  bool independent = concurrent || pipelined;

  std::string owner = outer.owner;
  uint64_t storageFactor = outer.storageFactor;
  if (concurrent && !mapTarget.empty()) {
    if (micro::symbolizeOwner(mapTarget)) {
      owner = mapTarget.str();
      // Each owner needs its own copy of the tiles the body materializes.
      storageFactor *= std::max<uint64_t>(1, machine.ownerCount(owner));
    } else {
      noteWarning("micro.spatial_for maps to '" + mapTarget.str() +
                  "', which is not an owner scope; owner occupancy is not "
                  "modeled for it");
    }
  }

  std::vector<uint32_t> previousIteration;
  for (uint64_t iteration = 0; iteration < trips; ++iteration) {
    State inner;
    inner.owner = owner;
    inner.storageFactor = storageFactor;
    if (iteration == 0) {
      inner.hasPrevious = outer.hasPrevious;
      inner.previous = outer.previous;
      inner.previousKind = outer.previousKind;
      inner.pending = outer.pending;
    } else if (!independent) {
      // Serial iterations: the next one starts only after this one is done.
      inner.pending = previousIteration;
    }

    std::vector<uint32_t> iterationEvents;
    if (llvm::Error err = walkBlock(body, inner, iterationEvents))
      return err;
    previousIteration = std::move(iterationEvents);
    created.insert(created.end(), previousIteration.begin(),
                   previousIteration.end());

    if (dag.events.size() > kMaxEvents)
      return invalid("kernel '" + kernelName + "' unrolls to more than " +
                     llvm::Twine(kMaxEvents) +
                     " events; it is too large for the MVP simulator");
  }

  outer.pending = std::move(previousIteration);
  outer.hasPrevious = false;
  return llvm::Error::success();
}

//===----------------------------------------------------------------------===//
// Per-op extraction
//===----------------------------------------------------------------------===//

llvm::Error DAGBuilder::buildOp(mlir::Operation &op, State &state) {
  if (auto view = llvm::dyn_cast<micro::TileViewOp>(op))
    return buildLogicalTileOp(op, view.getSource(), view.getResult(),
                              std::nullopt, EventKind::TileView, state);

  if (auto partition = llvm::dyn_cast<micro::TilePartitionOp>(op)) {
    std::optional<std::string> owner;
    if (std::optional<micro::Owner> attr = partition.getOwner())
      owner = micro::stringifyOwner(*attr).str();
    return buildLogicalTileOp(op, partition.getSource(), partition.getResult(),
                              owner, EventKind::TilePartition, state);
  }

  if (auto alloc = llvm::dyn_cast<micro::TileAllocOp>(op)) {
    TileInfo info = describeType(alloc.getResult().getType());
    return noteAllocation(op, info, info.memory, state);
  }

  if (auto alloc = llvm::dyn_cast<micro::AllocOp>(op)) {
    TileInfo info = describeType(alloc.getResult().getType());
    return noteAllocation(
        op, info, micro::stringifyMemorySpace(alloc.getMemory()), state);
  }

  if (auto copy = llvm::dyn_cast<micro::TileAsyncCopyOp>(op)) {
    TileInfo resultInfo = describeType(copy.getResult().getType());
    TileInfo sourceInfo = describeType(copy.getSource().getType());
    std::string src = memoryOf(copy.getSource());
    if (src.empty())
      src = "dram"; // an untagged operand is an external buffer
    return buildCopyOp(op, src,
                       micro::stringifyMemorySpace(copy.getDstMemory()),
                       sourceInfo, resultInfo, EventKind::AsyncCopy, state,
                       {copy.getResult(), copy.getToken()});
  }

  if (auto copy = llvm::dyn_cast<micro::AsyncCopyOp>(op)) {
    TileInfo resultInfo = describeType(copy.getResult().getType());
    TileInfo sourceInfo = describeType(copy.getSource().getType());
    return buildCopyOp(op, micro::stringifyMemorySpace(copy.getSrcMemory()),
                       micro::stringifyMemorySpace(copy.getDstMemory()),
                       sourceInfo, resultInfo, EventKind::AsyncCopy, state,
                       {copy.getResult(), copy.getToken()});
  }

  if (auto store = llvm::dyn_cast<micro::TileStoreOp>(op)) {
    TileInfo sourceInfo = describeType(store.getSource().getType());
    std::string src = memoryOf(store.getSource());
    if (src.empty())
      src = "sram"; // a tile without a tag came from on-chip storage
    return buildCopyOp(op, src,
                       micro::stringifyMemorySpace(store.getDstMemory()),
                       sourceInfo, TileInfo(), EventKind::Store, state, {});
  }

  if (auto store = llvm::dyn_cast<micro::StoreOp>(op)) {
    TileInfo sourceInfo = describeType(store.getSource().getType());
    return buildCopyOp(op, micro::stringifyMemorySpace(store.getSrcMemory()),
                       micro::stringifyMemorySpace(store.getDstMemory()),
                       sourceInfo, TileInfo(), EventKind::Store, state, {});
  }

  if (auto mma = llvm::dyn_cast<micro::MmaOp>(op)) {
    llvm::StringRef requested;
    if (auto engineAttr = mma.getEngine())
      requested = *engineAttr;
    std::string reason;
    const machine::ComputeNode *engine =
        pickMatrixEngine(requested, reason, mappedExecutor(op));
    if (!engine)
      return invalid("kernel '" + kernelName + "' uses micro.mma: " + reason);

    llvm::ArrayRef<int64_t> shape = mma.getShape();
    if (shape.size() != 3)
      return invalid("micro.mma shape must have three dimensions");

    uint64_t macs = static_cast<uint64_t>(shape[0]) *
                    static_cast<uint64_t>(shape[1]) *
                    static_cast<uint64_t>(shape[2]);
    std::string inputDtype = micro::stringifyDType(mma.getInput()).str();
    std::string accumulatorDtype =
        micro::stringifyDType(mma.getAccumulator()).str();
    if (!llvm::is_contained(engine->elementTypes, inputDtype))
      noteWarning("micro.mma input dtype '" + inputDtype +
                  "' is not supported by engine '" + engine->id +
                  "' (supported: " + llvm::join(engine->elementTypes, ", ") +
                  ")");
    if (!llvm::is_contained(engine->accumulatorDTypes, accumulatorDtype))
      noteWarning(
          "micro.mma accumulator dtype '" + accumulatorDtype +
          "' is not supported by engine '" + engine->id +
          "' (supported: " + llvm::join(engine->accumulatorDTypes, ", ") + ")");

    TileInfo lhsInfo = describeType(mma.getLhs().getType());
    TileInfo accInfo = describeType(mma.getAcc().getType());

    MicroEvent event;
    event.kind = EventKind::Mma;
    event.resource = ResourceKind::MatrixEngine;
    // An op that names no engine runs on the first engine of its class; one
    // that names an engine runs there. Either way the event names a single
    // pool, so two events naming the same engine share its slots instead of
    // each being handed the whole machine.
    event.resourceName = engine->id;
    event.workItems = macs;
    event.minCycles = mmaCycles(*engine, shape, 2 * macs);
    event.sourceOpName = op.getName().getStringRef().str();
    event.tileShape = shapeString(shape);
    event.tileLayout = lhsInfo.layout.empty() ? accInfo.layout : lhsInfo.layout;
    event.tileMemory = accInfo.memory;
    event.tileOwner = accInfo.owner;
    noteLayoutUsage(engine->id, engine->supportedLayouts, event.tileLayout);

    uint32_t id =
        addEvent(std::move(event), state,
                 producerDeps({mma.getLhs(), mma.getRhs(), mma.getAcc()}));
    producers[mma.getResult()] = id;
    return llvm::Error::success();
  }

  if (auto vector = llvm::dyn_cast<micro::VectorOp>(op)) {
    std::string reason;
    const machine::ComputeNode *engine =
        pickVectorEngine(reason, mappedExecutor(op));
    if (!engine)
      return invalid("kernel '" + kernelName +
                     "' uses micro.vector: " + reason);

    TileInfo resultInfo = describeType(vector.getResult().getType());
    MicroEvent event;
    event.kind = EventKind::Vector;
    event.resource = ResourceKind::VectorEngine;
    event.resourceName = engine->id;
    event.workItems = resultInfo.elements();
    event.minCycles =
        vectorCycles(engine, resultInfo.dtype, resultInfo.elements());
    event.sourceOpName = op.getName().getStringRef().str();
    event.tileShape = shapeString(resultInfo.shape);
    event.tileLayout = resultInfo.layout;
    event.tileMemory = resultInfo.memory;
    event.tileOwner = resultInfo.owner;
    noteLayoutUsage(engine->id, engine->supportedLayouts, event.tileLayout);

    uint32_t id =
        addEvent(std::move(event), state, producerDeps(vector.getInputs()));
    producers[vector.getResult()] = id;
    return llvm::Error::success();
  }

  if (auto reduce = llvm::dyn_cast<micro::ReduceOp>(op)) {
    std::string reason;
    const machine::ComputeNode *engine =
        pickVectorEngine(reason, mappedExecutor(op));
    if (!engine)
      return invalid("kernel '" + kernelName +
                     "' uses micro.reduce: " + reason);

    TileInfo inputInfo = describeType(reduce.getInput().getType());
    MicroEvent event;
    event.kind = EventKind::Reduce;
    event.resource = ResourceKind::VectorEngine;
    event.resourceName = engine->id;
    event.workItems = inputInfo.elements();
    event.minCycles =
        vectorCycles(engine, inputInfo.dtype, inputInfo.elements());
    event.sourceOpName = op.getName().getStringRef().str();
    event.tileShape = shapeString(inputInfo.shape);
    event.tileLayout = inputInfo.layout;
    event.tileMemory = inputInfo.memory;
    event.tileOwner = inputInfo.owner;
    noteLayoutUsage(engine->id, engine->supportedLayouts, event.tileLayout);

    uint32_t id =
        addEvent(std::move(event), state, producerDeps({reduce.getInput()}));
    producers[reduce.getResult()] = id;
    return llvm::Error::success();
  }

  if (auto transform = llvm::dyn_cast<micro::TransformOp>(op)) {
    // A conversion is real work: it runs on a compute capability like the other
    // mapped elementwise work, and its selected resource is the executor the
    // plan stamped when the op carries one.
    std::string reason;
    const machine::ComputeNode *engine =
        pickVectorEngine(reason, mappedExecutor(op));
    if (!engine)
      return invalid("kernel '" + kernelName +
                     "' uses micro.transform: " + reason);

    TileInfo sourceInfo = describeType(transform.getSource().getType());
    TileInfo resultInfo = describeType(transform.getResult().getType());
    // The conversion reads the memory its source landed in and writes its
    // output there: a logical view carries no memory of its own, so the
    // producing event is asked (as `memoryOf` does for a copy source).
    std::string sourceMemory = memoryOf(transform.getSource());
    std::string destination =
        resultInfo.memory.empty() ? sourceMemory : resultInfo.memory;

    // The one shared estimate the planner and the simulator both charge. An
    // unknown resource or footprint comes back as an error, never a silent
    // zero, so a mis-modeled conversion cannot vanish from the report.
    mapping::TransformCostInput costInput;
    costInput.inputType = shapedCostType(transform.getSource().getType());
    costInput.outputType = shapedCostType(transform.getResult().getType());
    if (std::optional<mlir::AffineMap> srcMap = transform.getSrcMap())
      costInput.srcMap = *srcMap;
    if (std::optional<mlir::AffineMap> dstMap = transform.getDstMap())
      costInput.dstMap = *dstMap;
    costInput.memoryNode = destination;
    costInput.computeResource = engine->id;

    llvm::Expected<mapping::Cost> cost =
        mapping::estimateTransformCost(costInput, machine);
    if (!cost)
      return invalid("kernel '" + kernelName + "' uses micro.transform: " +
                     llvm::toString(cost.takeError()));

    MicroEvent event;
    event.kind = EventKind::Transform;
    event.resource = ResourceKind::VectorEngine;
    event.resourceName = engine->id;
    event.workItems =
        resultInfo.elements() ? resultInfo.elements() : sourceInfo.elements();
    event.bytes = cost->localBytes;
    event.minCycles = static_cast<uint64_t>(std::ceil(cost->latencyCycles));
    event.sourceOpName = op.getName().getStringRef().str();
    event.tileShape = shapeString(resultInfo.shape.empty() ? sourceInfo.shape
                                                           : resultInfo.shape);
    event.tileLayout =
        resultInfo.layout.empty() ? sourceInfo.layout : resultInfo.layout;
    event.tileMemory = destination;
    event.srcMemory = sourceMemory;
    event.tileOwner =
        resultInfo.owner.empty() ? sourceInfo.owner : resultInfo.owner;
    noteLayoutUsage(engine->id, engine->supportedLayouts, event.tileLayout);

    const uint64_t bytes = event.bytes;
    uint32_t id = addEvent(std::move(event), state,
                           producerDeps({transform.getSource()}));
    // The result is produced by this event, so every consumer depends on the
    // conversion rather than on whatever produced its input.
    producers[transform.getResult()] = id;
    // The conversion materializes a fresh output buffer, which the capacity
    // check must see.
    noteStorage(op, destination, bytes, state.storageFactor);
    return llvm::Error::success();
  }

  if (auto wait = llvm::dyn_cast<micro::WaitOp>(op)) {
    MicroEvent event;
    event.kind = EventKind::Wait;
    event.resource = ResourceKind::Sync;
    event.resourceName = "sync";
    event.minCycles = machine.sync.waitCycles;
    event.sourceOpName = op.getName().getStringRef().str();
    addEvent(std::move(event), state, producerDeps(wait.getTokens()));
    return llvm::Error::success();
  }

  // Everything else -- arithmetic feeding loop bounds, micro.yield -- has no
  // execution cost in the MVP model.
  return llvm::Error::success();
}

llvm::Error DAGBuilder::buildLogicalTileOp(
    mlir::Operation &op, mlir::Value source, mlir::Value result,
    const std::optional<std::string> &owner, EventKind kind, State &state) {
  TileInfo sourceInfo = describeType(source.getType());
  TileInfo resultInfo = describeType(result.getType());
  if (owner) {
    resultInfo.owner = *owner;
    resultInfo.isTile = true;
  }

  // Value flow is preserved across a logical op, so a consumer still depends on
  // whatever produced the underlying data.
  inheritProducer(result, source);

  // A logical view is zero-cost metadata: it changes how the same bytes are
  // read, not how many there are. Only two known layouts that differ name real
  // work -- a view of an untagged tensor says nothing about the memory order it
  // has, so reading one as row-major is not a transform.
  if (!resultInfo.isTile || resultInfo.layout.empty() ||
      sourceInfo.layout.empty() || resultInfo.layout == sourceInfo.layout)
    return llvm::Error::success();

  std::string reason;
  const machine::ComputeNode *engine =
      pickVectorEngine(reason, mappedExecutor(op));
  if (!engine)
    return invalid("kernel '" + kernelName +
                   "' performs a layout transform but " + reason);

  MicroEvent event;
  event.kind = kind;
  event.resource = ResourceKind::VectorEngine;
  event.resourceName = engine->id;
  event.workItems = resultInfo.elements();
  event.bytes = resultInfo.bytes();
  event.minCycles =
      vectorCycles(engine, resultInfo.dtype, resultInfo.elements());
  event.sourceOpName = op.getName().getStringRef().str();
  event.tileShape = shapeString(resultInfo.shape);
  event.tileLayout = resultInfo.layout;
  event.tileMemory =
      resultInfo.memory.empty() ? sourceInfo.memory : resultInfo.memory;
  event.srcMemory = sourceInfo.memory;
  event.tileOwner = resultInfo.owner;
  noteLayoutUsage(engine->id, engine->supportedLayouts, event.tileLayout);

  std::string destination = event.tileMemory;
  uint32_t id = addEvent(std::move(event), state);
  producers[result] = id;
  noteStorage(op, destination, resultInfo.bytes(), state.storageFactor);
  return llvm::Error::success();
}

llvm::Error
DAGBuilder::buildCopyOp(mlir::Operation &op, llvm::StringRef srcMemory,
                        llvm::StringRef dstMemory, const TileInfo &sourceInfo,
                        const TileInfo &resultInfo, EventKind kind,
                        State &state, llvm::ArrayRef<mlir::Value> results) {
  if (llvm::Error err = requireMemory(op, dstMemory))
    return err;
  if (llvm::Error err = requireMemory(op, srcMemory))
    return err;
  if (llvm::Error err = requireReachable(op, srcMemory, dstMemory))
    return err;

  const TileInfo &shapeInfo = resultInfo.elements() ? resultInfo : sourceInfo;
  uint64_t bytes = shapeInfo.bytes();
  noteUnsizable(sourceInfo, op);
  noteUnsizable(resultInfo, op);

  // A movement the selected plan routed is charged per hop: the mapper chose a
  // path through the hierarchy, and the simulator has to see every link on it
  // rather than a single endpoint-to-endpoint transfer. The route and, for a
  // stamped per-hop copy, the single link are matched by node identity, so two
  // movements between same-kind endpoints still get their own links.
  llvm::SmallVector<const machine::LinkEdge *, 4> hops =
      matchRoute(op, srcMemory, dstMemory);

  auto describe = [&](const machine::LinkEdge &link, uint64_t cycles) {
    MicroEvent event;
    event.kind = kind;
    event.resource = ResourceKind::Dma;
    event.resourceName =
        link.transferEngines.empty() ? "dma" : link.transferEngines.front();
    event.workItems = shapeInfo.elements();
    event.bytes = bytes;
    event.minCycles = cycles;
    event.sourceOpName = op.getName().getStringRef().str();
    event.tileShape = shapeString(shapeInfo.shape);
    event.tileLayout =
        resultInfo.layout.empty() ? sourceInfo.layout : resultInfo.layout;
    const machine::MemoryNode *destination =
        machine.findMemory(link.destination);
    const machine::MemoryNode *source = machine.findMemory(link.source);
    event.tileMemory = destination ? destination->kind : dstMemory.str();
    event.srcMemory = source ? source->kind : srcMemory.str();
    event.tileOwner = resultInfo.owner;
    return event;
  };

  uint32_t id = 0;
  if (hops.empty()) {
    MicroEvent event;
    event.kind = kind;
    event.resource = ResourceKind::Dma;
    event.resourceName = "dma";
    event.workItems = shapeInfo.elements();
    event.bytes = bytes;
    event.minCycles = copyCycles(srcMemory, dstMemory, bytes);
    event.sourceOpName = op.getName().getStringRef().str();
    event.tileShape = shapeString(shapeInfo.shape);
    event.tileLayout =
        resultInfo.layout.empty() ? sourceInfo.layout : resultInfo.layout;
    event.tileMemory = dstMemory.str();
    event.srcMemory = srcMemory.str();
    event.tileOwner = resultInfo.owner;
    id = addEvent(std::move(event), state);
  } else {
    for (size_t hop = 0; hop < hops.size(); ++hop) {
      const machine::LinkEdge &link = *hops[hop];
      uint64_t cycles = link.latencyCycles;
      if (link.bandwidthBytesPerCycle > 0)
        cycles += static_cast<uint64_t>(std::ceil(static_cast<double>(bytes) /
                                                  link.bandwidthBytesPerCycle));
      llvm::SmallVector<uint32_t, 1> chain;
      if (hop > 0)
        chain.push_back(id);
      id = addEvent(describe(link, cycles), state, chain);
    }
  }
  for (mlir::Value result : results)
    producers[result] = id;

  // The moved tile is live in its destination. A pipelined or spatially mapped
  // body needs one buffer per concurrent iteration.
  noteStorage(op, dstMemory, bytes, state.storageFactor);
  return llvm::Error::success();
}

llvm::Error DAGBuilder::noteAllocation(mlir::Operation &op,
                                       const TileInfo &info,
                                       llvm::StringRef space,
                                       const State &state) {
  if (info.hasDynamicShape() || info.bytes() == 0) {
    noteUnsizable(info, op);
    return llvm::Error::success();
  }
  if (space.empty()) {
    noteWarning(op.getName().getStringRef().str() +
                " allocates a tile that names no memory space; it is not "
                "counted against capacity");
    return llvm::Error::success();
  }
  if (!machine.findMemoryOfKind(space)) {
    // An allocation needs no bandwidth or latency, so an unmodeled space only
    // costs us the capacity check.
    noteWarning("allocations target memory space '" + space.str() +
                "', which machine '" + machine.target +
                "' does not model; capacity is not checked for it");
    return llvm::Error::success();
  }
  noteStorage(op, space, info.bytes(), state.storageFactor);
  return llvm::Error::success();
}

llvm::Error DAGBuilder::requireMemory(mlir::Operation &op,
                                      llvm::StringRef space) const {
  if (space.empty())
    return invalid("kernel '" + kernelName +
                   "': " + op.getName().getStringRef() +
                   " moves data through a memory space that is neither named "
                   "on the value nor modeled by machine '" +
                   machine.target + "'");
  if (!machine.findMemoryOfKind(space))
    return invalid("kernel '" + kernelName +
                   "': " + op.getName().getStringRef() +
                   " moves data through memory space '" + space +
                   "', which machine '" + machine.target + "' does not model");
  return llvm::Error::success();
}

llvm::Error DAGBuilder::requireReachable(mlir::Operation &op,
                                         llvm::StringRef src,
                                         llvm::StringRef dst) {
  if (src == dst) {
    // Equal abstract kinds are not the same concrete memory: the movement is
    // real work only when the copy records two distinct concrete node ids.
    // Without that node identity a same-space copy is a no-op and is rejected.
    // (The machine verifier resolves the nodes, link and engine against the
    // model; this is the perf model's own guard.)
    auto srcNode = op.getAttrOfType<mlir::StringAttr>("micro.src_node");
    auto dstNode = op.getAttrOfType<mlir::StringAttr>("micro.dst_node");
    const bool distinctNodes =
        srcNode && dstNode && !srcNode.getValue().empty() &&
        !dstNode.getValue().empty() && srcNode.getValue() != dstNode.getValue();
    if (!distinctNodes)
      return invalid("kernel '" + kernelName +
                     "': " + op.getName().getStringRef() +
                     " copies from and to '" + src.str() + "'");
    return llvm::Error::success();
  }

  // A machine declares the copy paths it can perform. A pair it does not list
  // is still charged through both endpoints' latencies, but the gap is worth
  // reporting: that cost comes from a composition the machine never described.
  if (!machine.findLinkByKinds(src, dst))
    noteWarning("machine '" + machine.target + "' declares no copy path '" +
                src.str() + " -> " + dst.str() +
                "'; its cost is composed from both endpoints");
  return llvm::Error::success();
}

//===----------------------------------------------------------------------===//
// Entry point
//===----------------------------------------------------------------------===//

llvm::Expected<MicroDAG> DAGBuilder::run(mlir::Operation *kernel) {
  // A mapped kernel carries the routes its plan chose; read them before the
  // walk so a routed movement is charged per hop.
  loadRoutes(kernel);
  State state;
  std::vector<uint32_t> created;
  if (llvm::Error err = walkBlock(kernel->getRegion(0).front(), state, created))
    return std::move(err);

  if (dag.events.empty())
    noteWarning("kernel '" + kernelName +
                "' has no schedulable events; every figure below is zero");

  dag.liveTileBytesByMemory = std::move(liveBytes);
  return std::move(dag);
}

} // namespace

//===----------------------------------------------------------------------===//
// Public API
//===----------------------------------------------------------------------===//

llvm::Expected<MicroDAG> buildMicroDAG(mlir::Operation *kernel,
                                       const machine::MachineModel &machine) {
  auto kernelOp = llvm::dyn_cast<micro::KernelOp>(kernel);
  if (!kernelOp)
    return invalid("expected a micro.kernel, got '" +
                   kernel->getName().getStringRef() + "'");

  DAGBuilder builder(machine, kernelOp.getSymName());
  return builder.run(kernel);
}

llvm::Expected<mlir::Operation *> findMicroKernel(mlir::Operation *root,
                                                  llvm::StringRef symbol) {
  llvm::SmallVector<micro::KernelOp, 2> kernels;
  root->walk([&](micro::KernelOp kernel) { kernels.push_back(kernel); });

  if (!symbol.empty()) {
    for (micro::KernelOp kernel : kernels)
      if (kernel.getSymName() == symbol)
        return kernel.getOperation();
    return invalid("no micro.kernel named '" + symbol + "' in the input");
  }

  if (kernels.empty())
    return invalid("the input contains no micro.kernel");
  if (kernels.size() > 1)
    return invalid("the input contains " + llvm::Twine(kernels.size()) +
                   " micro.kernel ops; select one with --kernel=<name>");
  return kernels.front().getOperation();
}

llvm::StringRef stringifyEventKind(EventKind kind) {
  switch (kind) {
  case EventKind::TileView:
    return "tile_view";
  case EventKind::TilePartition:
    return "tile_partition";
  case EventKind::Transform:
    return "transform";
  case EventKind::AsyncCopy:
    return "async_copy";
  case EventKind::Load:
    return "load";
  case EventKind::Store:
    return "store";
  case EventKind::Mma:
    return "mma";
  case EventKind::Vector:
    return "vector";
  case EventKind::Reduce:
    return "reduce";
  case EventKind::Wait:
    return "wait";
  case EventKind::Barrier:
    return "barrier";
  }
  return "unknown";
}

mapping::CostEventKind costEventKindOf(EventKind kind) {
  switch (kind) {
  case EventKind::Mma:
  case EventKind::Vector:
  case EventKind::Reduce:
    return mapping::CostEventKind::Compute;
  case EventKind::AsyncCopy:
  case EventKind::Load:
  case EventKind::Store:
    return mapping::CostEventKind::TransferHop;
  case EventKind::TileView:
  case EventKind::TilePartition:
  case EventKind::Transform:
    return mapping::CostEventKind::Transform;
  case EventKind::Wait:
  case EventKind::Barrier:
    return mapping::CostEventKind::Synchronization;
  }
  return mapping::CostEventKind::Compute;
}

llvm::StringRef stringifyResourceKind(ResourceKind kind) {
  switch (kind) {
  case ResourceKind::Dma:
    return "dma";
  case ResourceKind::MatrixEngine:
    return "matrix_engine";
  case ResourceKind::VectorEngine:
    return "vector_engine";
  case ResourceKind::MemoryRead:
    return "memory_read";
  case ResourceKind::MemoryWrite:
    return "memory_write";
  case ResourceKind::Sync:
    return "sync";
  }
  return "unknown";
}

} // namespace mlir::llk::perf
