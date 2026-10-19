//===- MicroDAG.cpp - Event DAG extraction from concrete Micro-IR ---------===//
//
// Part of the LLK Compiler Project, under the Apache License v2.0 with
// LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Implements buildMicroDAG(): the single traversal that turns a concrete
// `micro.kernel` into schedulable events plus the program properties the
// capacity check needs. See MicroDAG.h for the dependency rules.
//
// Costs come from the machine model, never from constants in this file: an
// event's `minCycles` is the machine's own answer for the work the tile
// metadata describes. That is what keeps a second target from needing a second
// extractor.

#include "LLK/Perf/MicroDAG.h"
#include "LLK/Perf/OpDAGBuilder.h"

#include "MicroTileInfo.h"

#include "LLK/Dialect/Micro/MicroEnums.h" // IWYU pragma: keep
#include "LLK/Mapping/CostModel.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinOps.h" // IWYU pragma: keep
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/CheckedArithmetic.h"
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

/// The concrete capability node a mapped op *selected* for the compute
/// requirement of `kind` (`vector_engine`, `matrix_engine`), read from its
/// `micro.mapping` `compute_bindings` map, or empty when the op carries none.
/// R1 made this selection identity-bearing; the simulator must honour it rather
/// than resolving the executor's first attached engine of the kind, or two
/// plans that selected different engines of one kind would be charged as one.
llvm::StringRef mappedComputeEngine(mlir::Operation &op, llvm::StringRef kind) {
  auto mapping = op.getAttrOfType<mlir::DictionaryAttr>("micro.mapping");
  if (!mapping)
    return {};
  auto bindings = mapping.getAs<mlir::DictionaryAttr>("compute_bindings");
  if (!bindings)
    return {};
  if (auto engine = bindings.getAs<mlir::StringAttr>(kind))
    return engine.getValue();
  return {};
}

/// The instance a mapped op recorded, or nullopt when it carries none. `0` is a
/// legal instance id, so this is an optional rather than a sentinel.
std::optional<uint64_t> mappedInstance(mlir::Operation &op) {
  auto mapping = op.getAttrOfType<mlir::DictionaryAttr>("micro.mapping");
  if (!mapping)
    return std::nullopt;
  if (auto instance = mapping.getAs<mlir::IntegerAttr>("instance"))
    return instance.getValue().getZExtValue();
  return std::nullopt;
}

/// The connection a materialized copy records (`micro.connection`), or nullopt.
std::optional<uint64_t> mappedConnection(mlir::Operation &op) {
  if (auto connection = op.getAttrOfType<mlir::IntegerAttr>("micro.connection"))
    return connection.getValue().getZExtValue();
  return std::nullopt;
}

/// The executor a materialized `micro.transform` selected: stamped as
/// `micro.engine` by the binder (task B8), because a conversion is not a
/// covered workload node and carries no `micro.mapping`. A hand-written or
/// older kernel without the stamp falls back to `micro.mapping` exactly as a
/// compute op does.
llvm::StringRef transformExecutor(mlir::Operation &op) {
  if (auto engine = op.getAttrOfType<mlir::StringAttr>("micro.engine"))
    return engine.getValue();
  return mappedExecutor(op);
}

class DAGBuilder {
public:
  DAGBuilder(const machine::MachineModel &machine, llvm::StringRef kernelName,
             const OpDAGBuilderRegistry &builders,
             const MeasuredOverrides *measured = nullptr,
             DAGBuildOptions options = {CoverageMode::LegacyPermissive})
      : machine(machine), options(options), builders(builders),
        kernelName(kernelName.str()), measured(measured) {}

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
    /// The concrete executor a mapped op selected, inherited by the events a
    /// region builds so a nested compute event names the same owner pool its
    /// enclosing scope did (issue #129, task R6). Empty for unmapped analysis.
    std::string ownerPool;
    /// Buffers a body must hold at once: pipeline stages, times the owners a
    /// spatial loop spreads its iterations over.
    uint64_t storageFactor = 1;
  };

  class Context final : public OpDAGBuildContext {
  public:
    Context(DAGBuilder &builder, mlir::Operation &op, State &state)
        : builder(builder), op(op), state(state) {}
    void noteIncomplete(std::string reason) override {
      builder.noteIncomplete(std::move(reason));
    }
    mlir::Operation &operation() const override { return op; }
    const machine::MachineModel &machineModel() const override {
      return builder.machine;
    }
    llvm::StringRef kernelName() const override { return builder.kernelName; }
    uint32_t emitEvent(MicroEvent event, mlir::ValueRange inputs,
                       mlir::ValueRange outputs,
                       llvm::ArrayRef<uint32_t> extraDeps = {}) override {
      auto deps = builder.producerDeps(inputs);
      llvm::append_range(deps, extraDeps);
      if (event.kind != EventKind::Wait && event.kind != EventKind::Barrier)
        builder.applyMeasuredDuration(op, event);
      uint32_t id = builder.addEvent(std::move(event), state, deps);
      for (mlir::Value output : outputs)
        builder.producers[output] = id;
      return id;
    }
    void aliasValue(mlir::Value result, mlir::Value source) override {
      builder.inheritProducer(result, source);
    }
    std::string memoryOf(mlir::Value value) const override {
      return builder.memoryOf(value);
    }
    void accountStorage(llvm::StringRef memory, uint64_t bytes) override {
      builder.noteStorage(op, memory, bytes, state.storageFactor);
    }
    void noteUnsizable(const TileInfo &info) override {
      builder.noteUnsizable(info, op);
    }
    void noteWarning(std::string message) override {
      builder.noteWarning(std::move(message));
    }
    void noteLayoutUsage(llvm::StringRef engine,
                         const std::vector<std::string> &supported,
                         llvm::StringRef layout) override {
      builder.noteLayoutUsage(engine, supported, layout);
    }
    llvm::Error buildLogicalTile(mlir::Value source, mlir::Value result,
                                 const std::optional<std::string> &owner,
                                 EventKind kind) override {
      return builder.buildLogicalTileOp(op, source, result, owner, kind, state);
    }
    llvm::Error buildMovement(llvm::StringRef src, llvm::StringRef dst,
                              const TileInfo &source, const TileInfo &result,
                              EventKind kind,
                              llvm::ArrayRef<mlir::Value> outputs) override {
      return builder.buildCopyOp(op, src, dst, source, result, kind, state,
                                 outputs);
    }
    llvm::Error noteAllocation(const TileInfo &info,
                               llvm::StringRef memory) override {
      return builder.noteAllocation(op, info, memory, state);
    }
    const machine::ComputeNode *
    pickMatrixEngine(llvm::StringRef requested,
                     std::string &reason) const override {
      return builder.pickMatrixEngine(
          requested.empty() ? mappedComputeEngine(op, "matrix_engine")
                            : requested,
          reason, mappedExecutor(op));
    }
    const machine::ComputeNode *
    pickVectorEngine(std::string &reason) const override {
      return builder.pickVectorEngine(mappedComputeEngine(op, "vector_engine"),
                                      reason, mappedExecutor(op));
    }
    const machine::ComputeNode *
    pickContractionEngine(llvm::StringRef requested,
                          std::string &reason) const override {
      return builder.pickContractionEngine(op, requested, reason);
    }
    const machine::ComputeNode *
    pickTransformEngine(std::string &reason) const override {
      return builder.pickVectorEngine("", reason, transformExecutor(op));
    }
    uint64_t vectorCycles(const machine::ComputeNode *engine,
                          llvm::StringRef dtype,
                          uint64_t elements) const override {
      return builder.vectorCycles(engine, dtype, elements);
    }
    uint64_t mmaCycles(const machine::ComputeNode &engine,
                       llvm::ArrayRef<int64_t> shape,
                       uint64_t flops) const override {
      return builder.mmaCycles(engine, shape, flops);
    }
    llvm::Expected<uint64_t>
    contractionCycles(const machine::ComputeNode &engine,
                      llvm::ArrayRef<int64_t> shape, uint64_t flops,
                      llvm::StringRef accumulatorDtype) const override {
      return builder.contractionCycles(engine, shape, flops, accumulatorDtype);
    }

  private:
    DAGBuilder &builder;
    mlir::Operation &op;
    State &state;
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
  /// Records a fact that makes the event stream incomplete: it is a warning
  /// *and* an incompleteness reason, so a strict analysis can refuse the stream
  /// without string-matching diagnostics (issue #129, task R6).
  void noteIncomplete(std::string message);
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
  pickVectorEngine(llvm::StringRef requested, std::string &reason,
                   llvm::StringRef executor = "") const;
  /// The selected concrete engine a mapped contraction runs on. An unmapped
  /// contraction keeps the matrix default; a mapped one must name exactly one
  /// vector or matrix selection that resolves on its selected executor.
  const machine::ComputeNode *pickContractionEngine(mlir::Operation &op,
                                                    llvm::StringRef requested,
                                                    std::string &reason) const;
  /// Checked contraction cost: positive extents, checked MAC/FLOP products,
  /// and no one-cycle fallback for a missing lane or throughput fact.
  llvm::Expected<uint64_t>
  contractionCycles(const machine::ComputeNode &engine,
                    llvm::ArrayRef<int64_t> shape, uint64_t flops,
                    llvm::StringRef accumulatorDtype) const;

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
  DAGBuildOptions options;
  const OpDAGBuilderRegistry &builders;
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
  llvm::StringSet<> seenIncompleteReasons;
  std::map<std::string, uint64_t> liveBytes;
  /// Measured durations the selected plan recorded, or null when none (issue
  /// #129 review finding 6).
  const MeasuredOverrides *measured = nullptr;

  // Each stamped operation's nth execution belongs to the connection's nth
  // occurrence. This joins distinct hop/transform/consumer operations while
  // keeping repeated loop executions separate.
  llvm::DenseMap<mlir::Operation *, uint64_t> opOccurrences;
  std::map<std::pair<uint64_t, uint64_t>, std::vector<uint32_t>>
      measuredConnectionEvents;

  void applyMeasuredDuration(mlir::Operation &op, MicroEvent &event) {
    if (!measured)
      return;
    if (std::optional<uint64_t> instance = mappedInstance(op))
      if (auto found = measured->byInstance.find(*instance);
          found != measured->byInstance.end()) {
        event.minCycles = static_cast<uint64_t>(std::ceil(found->second));
        return;
      }
    if (std::optional<uint64_t> connection = mappedConnection(op))
      if (measured->byConnection.count(*connection))
        measuredConnectionEvents[{*connection, opOccurrences.lookup(&op)}]
            .push_back(static_cast<uint32_t>(dag.events.size()));
  }

  /// A provider measures the whole connection, so its hops and transforms
  /// share one duration. Retain their static proportions to distribute that
  /// aggregate across the same resource events. Cumulative rounding keeps the
  /// sum exactly ceil(measured), even for fractional or sub-hop durations.
  void applyMeasuredConnections() {
    for (const auto &[key, events] : measuredConnectionEvents) {
      const uint64_t cycles = static_cast<uint64_t>(
          std::ceil(measured->byConnection.lookup(key.first)));
      long double totalWeight = 0;
      for (uint32_t id : events)
        totalWeight += dag.events[id].minCycles;
      long double prefixWeight = 0;
      uint64_t previousCycles = 0;
      for (size_t index = 0; index < events.size(); ++index) {
        MicroEvent &event = dag.events[events[index]];
        prefixWeight += event.minCycles;
        const long double fraction =
            totalWeight > 0
                ? prefixWeight / totalWeight
                : static_cast<long double>(index + 1) / events.size();
        const uint64_t prefixCycles =
            index + 1 == events.size()
                ? cycles
                : static_cast<uint64_t>(std::floor(cycles * fraction));
        event.minCycles = prefixCycles - previousCycles;
        previousCycles = prefixCycles;
      }
    }
  }
};

//===----------------------------------------------------------------------===//
// Diagnostics
//===----------------------------------------------------------------------===//

void DAGBuilder::noteWarning(std::string message) {
  if (seenWarnings.insert(message).second)
    dag.warnings.push_back(std::move(message));
}

void DAGBuilder::noteIncomplete(std::string message) {
  if (seenIncompleteReasons.insert(message).second)
    dag.incompleteReasons.push_back(message);
  noteWarning(std::move(message));
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
    noteIncomplete(opName.str() +
                   " moves or allocates a value with a dynamic extent; its "
                   "bytes are charged as zero");
  else if (info.elements() > 0 && info.bytes() == 0)
    noteIncomplete(opName.str() +
                   " moves or allocates a value whose element type is not a "
                   "micro dtype; its bytes are charged as zero");
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
  // The one shared elementwise-issue formula (task B8), so a normalized plan
  // event and this DAG's event charge an elementwise pass identically.
  return mapping::elementwiseCycles(*engine, dtype, elements);
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
DAGBuilder::pickVectorEngine(llvm::StringRef requested, std::string &reason,
                             llvm::StringRef executor) const {
  // A recorded selection wins outright (issue #129, task R1): the mapped op
  // says which attached engine ran, so two plans of one rule that selected two
  // engines of a kind are charged two engines rather than both collapsing onto
  // the executor's first. A selection the machine does not declare is an error,
  // never silently replaced.
  if (!requested.empty()) {
    if (const machine::ComputeNode *engine = machine.findCompute(requested))
      return engine;
    reason = "engine '" + requested.str() + "' is not declared by machine '" +
             machine.target + "'";
    return nullptr;
  }
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

const machine::ComputeNode *DAGBuilder::pickContractionEngine(
    mlir::Operation &op, llvm::StringRef requested, std::string &reason) const {
  // An unmapped contraction is the reference/hand-written case: there is no
  // selected identity, so the documented matrix default still applies.
  auto mapping = op.getAttrOfType<mlir::DictionaryAttr>("micro.mapping");
  if (!mapping)
    return pickMatrixEngine(requested, reason);

  llvm::StringRef selectedKind;
  llvm::StringRef selectedId;
  if (auto bindings = mapping.getAs<mlir::DictionaryAttr>("compute_bindings")) {
    auto vector = bindings.getAs<mlir::StringAttr>("vector_engine");
    auto matrix = bindings.getAs<mlir::StringAttr>("matrix_engine");
    if (vector && matrix) {
      reason = "micro.mapping selects both vector_engine '" +
               vector.getValue().str() + "' and matrix_engine '" +
               matrix.getValue().str() + "' as its compute engine";
      return nullptr;
    }
    if (vector) {
      selectedId = vector.getValue();
      selectedKind = "vector_engine";
    } else if (matrix) {
      selectedId = matrix.getValue();
      selectedKind = "matrix_engine";
    }
  }
  if (selectedId.empty()) {
    reason = "micro.mapping records no selected compute engine "
             "(compute_bindings names no vector_engine or matrix_engine)";
    return nullptr;
  }

  const machine::ComputeNode *engine = machine.findCompute(selectedId);
  if (!engine) {
    reason = "selected engine '" + selectedId.str() +
             "' is not declared by machine '" + machine.target + "'";
    return nullptr;
  }
  if (llvm::StringRef(engine->kind) != selectedKind) {
    reason = "selected node '" + selectedId.str() + "' has kind '" +
             engine->kind + "', not " + selectedKind.str();
    return nullptr;
  }

  // The selected executor is part of the identity: an engine attached to a
  // different executor describes work the plan did not place here.
  llvm::StringRef executor = mappedExecutor(op);
  if (executor.empty()) {
    reason = "micro.mapping records no selected executor for the contraction";
    return nullptr;
  }
  if (llvm::StringRef(engine->attachedTo) != executor) {
    reason = "selected engine '" + selectedId.str() + "' is attached to '" +
             engine->attachedTo + "', not selected executor '" +
             executor.str() + "'";
    return nullptr;
  }

  // An op-level engine is a redundant spelling of the same decision; it may
  // agree, but it may never override or contradict the recorded selection.
  if (!requested.empty() && requested != selectedId) {
    reason = "explicit engine '" + requested.str() +
             "' disagrees with selected engine '" + selectedId.str() + "'";
    return nullptr;
  }
  return engine;
}

llvm::Expected<uint64_t>
DAGBuilder::contractionCycles(const machine::ComputeNode &engine,
                              llvm::ArrayRef<int64_t> shape, uint64_t flops,
                              llvm::StringRef accumulatorDtype) const {
  if (shape.size() != 3)
    return invalid("micro.mma shape must have three dimensions");
  for (int64_t extent : shape)
    if (extent <= 0)
      return invalid("micro.mma shape extents must be positive");

  // Validate the MAC and FLOP products before either cost formula runs, so an
  // overflowing shape cannot become a wrapped or one-cycle estimate.
  uint64_t macs = 1;
  for (int64_t extent : shape) {
    std::optional<uint64_t> product =
        llvm::checkedMulUnsigned<uint64_t>(macs, static_cast<uint64_t>(extent));
    if (!product)
      return invalid("micro.mma MAC count overflows 64 bits");
    macs = *product;
  }
  std::optional<uint64_t> checkedFlops =
      llvm::checkedMulUnsigned<uint64_t>(macs, 2);
  if (!checkedFlops)
    return invalid("micro.mma FLOP count overflows 64 bits");
  if (flops != *checkedFlops)
    return invalid("micro.mma cycle request does not match its shape");

  if (engine.throughputPerCycle && *engine.throughputPerCycle > 0)
    return static_cast<uint64_t>(
        std::ceil(static_cast<double>(flops) / *engine.throughputPerCycle));

  // Matrix capabilities keep the established fragment-based cost.
  if (engine.kind == "matrix_engine")
    return mmaCycles(engine, shape, flops);

  if (engine.kind != "vector_engine")
    return invalid("engine '" + engine.id + "' of kind '" + engine.kind +
                   "' cannot cost a contraction");

  auto lanes = engine.lanes.find(accumulatorDtype.str());
  if (lanes == engine.lanes.end() || lanes->second <= 0)
    return invalid("engine '" + engine.id +
                   "' declares no accumulator lane width for dtype '" +
                   accumulatorDtype.str() + "'");

  // M * K issues of ceil(N / lanes) vector MACs each. Missing facts are an
  // incomplete analysis, never a one-cycle event.
  uint64_t issues = llvm::divideCeil(static_cast<uint64_t>(shape[1]),
                                     static_cast<uint64_t>(lanes->second));
  std::optional<uint64_t> product = llvm::checkedMulUnsigned<uint64_t>(
      static_cast<uint64_t>(shape[0]), static_cast<uint64_t>(shape[2]));
  if (!product)
    return invalid("micro.mma vector issue count overflows 64 bits");
  product = llvm::checkedMulUnsigned<uint64_t>(*product, issues);
  if (!product)
    return invalid("micro.mma vector issue count overflows 64 bits");
  product = llvm::checkedMulUnsigned<uint64_t>(
      *product, std::max<uint64_t>(1, engine.issueCycles));
  if (!product)
    return invalid("micro.mma vector issue count overflows 64 bits");
  return *product;
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
  // The owner-occupancy pool the mapped placement selected is inherited from
  // the enclosing region when the op itself named none (issue #129, task R6).
  if (event.ownerPool.empty())
    event.ownerPool = state.ownerPool;

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
      // The spatial axis is an open symbol now, so the label is whatever the
      // loop recorded -- a target's spelling, not a fixed enumerator.
      std::string target = spatialOp.getMap().getSymbol().str();
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
    noteIncomplete("a loop has non-static bounds; the simulator runs a single "
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
    // The spatial axis is an open symbol; whether it names an execution scope
    // is the machine's answer, resolved through its alias data, not a fixed
    // list of spellings the dialect happens to know.
    if (machine.ownerClass(mapTarget)) {
      owner = mapTarget.str();
      // Each owner needs its own copy of the tiles the body materializes.
      storageFactor *= std::max<uint64_t>(1, machine.ownerCount(owner));
    } else {
      noteIncomplete("micro.spatial_for maps to '" + mapTarget.str() +
                     "', which is not an owner scope; owner occupancy is not "
                     "modeled for it");
    }
  }

  std::vector<uint32_t> previousIteration;
  for (uint64_t iteration = 0; iteration < trips; ++iteration) {
    State inner;
    inner.owner = owner;
    // A mapped op's executor pool is inherited through the loop nest, so a
    // compute event the body builds names the placement that selected it
    // (issue #129, task R6).
    inner.ownerPool = outer.ownerPool;
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
  ++opOccurrences[&op];
  const auto *handler = builders.lookup(op.getName().getStringRef());
  if (!handler) {
    if (options.coverage != CoverageMode::LegacyPermissive)
      noteIncomplete("unmodeled_leaf: " + op.getName().getStringRef().str());
    return llvm::Error::success();
  }
  Context context(*this, op, state);
  return (*handler)(op, context);
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
  const machine::ComputeNode *engine = pickVectorEngine(
      mappedComputeEngine(op, "vector_engine"), reason, mappedExecutor(op));
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
  applyMeasuredDuration(op, event);
  event.sourceOpName = op.getName().getStringRef().str();
  event.tileShape = shapeString(resultInfo.shape);
  event.tileLayout = resultInfo.layout;
  event.tileMemory =
      resultInfo.memory.empty() ? sourceInfo.memory : resultInfo.memory;
  event.srcMemory = sourceInfo.memory;
  event.tileOwner = resultInfo.owner;
  noteLayoutUsage(engine->id, engine->supportedLayouts, event.tileLayout);

  std::string destination = event.tileMemory;
  uint32_t id = addEvent(std::move(event), state, producerDeps({source}));
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
    applyMeasuredDuration(op, event);
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

  auto sourceDeps = producerDeps(op.getOperands());
  uint32_t id = 0;
  if (hops.empty()) {
    // A *mapped*, unrouted movement still crosses a concrete link when the
    // machine joins the two endpoint kinds: a boundary copy a rule placed (the
    // `dram -> sram` stage that feeds a compute node) carries no routed hop
    // identity, but the placement is a decision, and the machine declares the
    // link its endpoints are carried over. Charging that link names its
    // transfer engine, so the event's resource is one the machine -- and the
    // strict stream validator -- models, instead of the abstract `dma` pool no
    // node declares. The cost is unchanged either way: `copyCycles` already
    // reads this very link for its latency and bandwidth.
    //
    // A hand-written kernel with no `micro.mapping` keeps the abstract pool: an
    // unmapped analysis has not chosen an engine, and naming one it was never
    // told would be a claim the plan does not support.
    const machine::LinkEdge *link =
        op.getAttrOfType<mlir::DictionaryAttr>("micro.mapping") &&
                srcMemory != dstMemory
            ? machine.findLinkByKinds(srcMemory, dstMemory)
            : nullptr;
    if (link) {
      id = addEvent(describe(*link, copyCycles(srcMemory, dstMemory, bytes)),
                    state, sourceDeps);
    } else {
      MicroEvent event;
      event.kind = kind;
      event.resource = ResourceKind::Dma;
      event.resourceName = "dma";
      event.workItems = shapeInfo.elements();
      event.bytes = bytes;
      event.minCycles = copyCycles(srcMemory, dstMemory, bytes);
      applyMeasuredDuration(op, event);
      event.sourceOpName = op.getName().getStringRef().str();
      event.tileShape = shapeString(shapeInfo.shape);
      event.tileLayout =
          resultInfo.layout.empty() ? sourceInfo.layout : resultInfo.layout;
      event.tileMemory = dstMemory.str();
      event.srcMemory = srcMemory.str();
      event.tileOwner = resultInfo.owner;
      id = addEvent(std::move(event), state, sourceDeps);
    }
  } else {
    for (size_t hop = 0; hop < hops.size(); ++hop) {
      const machine::LinkEdge &link = *hops[hop];
      uint64_t cycles = link.latencyCycles;
      if (link.bandwidthBytesPerCycle > 0)
        cycles += static_cast<uint64_t>(std::ceil(static_cast<double>(bytes) /
                                                  link.bandwidthBytesPerCycle));
      llvm::SmallVector<uint32_t, 4> chain;
      if (hop > 0)
        chain.push_back(id);
      else
        chain = sourceDeps;
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

  applyMeasuredConnections();

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
                                       const machine::MachineModel &machine,
                                       const MeasuredOverrides *measured) {
  auto builders = makeDefaultOpDAGBuilders();
  return buildMicroDAG(kernel, machine, builders, measured);
}

llvm::Expected<MicroDAG> buildMicroDAG(mlir::Operation *kernel,
                                       const machine::MachineModel &machine,
                                       const OpDAGBuilderRegistry &builders,
                                       const MeasuredOverrides *measured) {
  auto kernelOp = llvm::dyn_cast<micro::KernelOp>(kernel);
  if (!kernelOp)
    return invalid("expected a micro.kernel, got '" +
                   kernel->getName().getStringRef() + "'");

  DAGBuilder builder(machine, kernelOp.getSymName(), builders, measured);
  return builder.run(kernel);
}

llvm::Expected<MicroDAG> buildMicroDAG(mlir::Operation *kernel,
                                       const machine::MachineModel &machine,
                                       const OpDAGBuilderRegistry &builders,
                                       const DAGBuildOptions &options,
                                       const MeasuredOverrides *measured) {
  auto kernelOp = llvm::dyn_cast<micro::KernelOp>(kernel);
  if (!kernelOp)
    return invalid("expected a micro.kernel");
  DAGBuilder builder(machine, kernelOp.getSymName(), builders, measured,
                     options);
  auto dag = builder.run(kernel);
  if (!dag)
    return dag.takeError();
  if (options.coverage == CoverageMode::Strict &&
      !dag->incompleteReasons.empty()) {
    std::string reasons;
    for (const auto &reason : dag->incompleteReasons) {
      if (!reasons.empty())
        reasons += "; ";
      reasons += reason;
    }
    return invalid(reasons);
  }
  return std::move(*dag);
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

mapping::PlanCostEvent normalizedPlanEvent(const MicroEvent &event) {
  // The single shared construction point, so this view and a plan-derived event
  // cannot differ in their normalized fields (task B8). The event's own
  // `resourceName` is the machine resource both paths name.
  std::vector<uint32_t> deps(event.deps.begin(), event.deps.end());
  mapping::PlanCostEvent normalized = mapping::makePlanCostEvent(
      costEventKindOf(event.kind), event.resourceName,
      static_cast<double>(event.minCycles), event.workItems, event.bytes,
      std::move(deps));
  // The execution facts the plan path also records (issue #129, task R6): the
  // owner-occupancy pool the event runs under -- the mapped executor's id when
  // the op was placed, otherwise the tile's abstract owner symbol -- and the
  // two memories a movement crosses. Only an event with traffic charges them
  // (a compute event's `bytes` is zero), but the owner joins every event's
  // normalized shape so a plan event and its materialized counterpart occupy
  // the same pool.
  normalized.owner =
      event.ownerPool.empty() ? event.tileOwner : event.ownerPool;
  normalized.srcMemory = event.srcMemory;
  normalized.dstMemory = event.tileMemory;
  return normalized;
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
