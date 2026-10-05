//===- StoragePlan.cpp - Physical footprints and value live ranges --------===//
//
// Task B3 (issue #67, stage B).
//
//===----------------------------------------------------------------------===//

#include "LLK/Mapping/StoragePlan.h"

#include "LLK/Mapping/TileFacts.h"

#include "LLK/Machine/MachineModel.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <limits>
#include <map>
#include <set>
#include <string>
#include <vector>

namespace mlir::llk::mapping {

namespace {

/// The conservative byte size a *reported* analysis fallback charges for a
/// value whose physical footprint cannot be derived. Matches the search's
/// assumed size, so two phases of one pipeline never disagree about the
/// fallback. It is only ever used for an explicit, non-materialized analysis
/// plan.
constexpr uint64_t kAssumedValueBytes = 4096;

/// The highest index-space rank whose affine image is bounded by evaluating the
/// corners (2^rank points). Higher ranks fall through to enumeration or are
/// refused.
constexpr unsigned kMaxCornerRank = 16;

llvm::Error storageError(const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(), message);
}

/// Checked addition/multiplication over any integer type. LLVM's `MathExtras`
/// `AddOverflow`/`MulOverflow` are signed-only, so the compiler builtins are
/// used directly; both clang and the CI compiler provide them.
template <typename T> bool checkedAdd(T lhs, T rhs, T &result) {
  return __builtin_add_overflow(lhs, rhs, &result);
}

template <typename T> bool checkedMul(T lhs, T rhs, T &result) {
  return __builtin_mul_overflow(lhs, rhs, &result);
}

std::string printType(mlir::Type type) {
  std::string text;
  llvm::raw_string_ostream stream(text);
  type.print(stream);
  return stream.str();
}

/// True when `expr` contains a `floordiv`, `ceildiv`, or `mod`, which makes it
/// semi-affine: its extreme value over a box is not necessarily at a corner.
bool isSemiAffine(mlir::AffineExpr expr) {
  using Kind = mlir::AffineExprKind;
  switch (expr.getKind()) {
  case Kind::FloorDiv:
  case Kind::CeilDiv:
  case Kind::Mod:
    return true;
  case Kind::Add:
  case Kind::Mul: {
    auto binary = mlir::cast<mlir::AffineBinaryOpExpr>(expr);
    return isSemiAffine(binary.getLHS()) || isSemiAffine(binary.getRHS());
  }
  default:
    return false;
  }
}

/// Evaluates an affine expression at a concrete index point, or nullopt on a
/// symbol (rejected earlier) or arithmetic overflow.
std::optional<int64_t> evalAt(mlir::AffineExpr expr,
                              llvm::ArrayRef<int64_t> point) {
  using Kind = mlir::AffineExprKind;
  switch (expr.getKind()) {
  case Kind::Constant:
    return mlir::cast<mlir::AffineConstantExpr>(expr).getValue();
  case Kind::DimId: {
    unsigned position = mlir::cast<mlir::AffineDimExpr>(expr).getPosition();
    if (position >= point.size())
      return std::nullopt;
    return point[position];
  }
  case Kind::SymbolId:
    return std::nullopt;
  case Kind::Add:
  case Kind::Mul: {
    auto binary = mlir::cast<mlir::AffineBinaryOpExpr>(expr);
    std::optional<int64_t> lhs = evalAt(binary.getLHS(), point);
    std::optional<int64_t> rhs = evalAt(binary.getRHS(), point);
    if (!lhs || !rhs)
      return std::nullopt;
    int64_t result = 0;
    bool overflow = expr.getKind() == Kind::Add
                        ? checkedAdd(*lhs, *rhs, result)
                        : checkedMul(*lhs, *rhs, result);
    if (overflow)
      return std::nullopt;
    return result;
  }
  case Kind::FloorDiv:
  case Kind::CeilDiv:
  case Kind::Mod: {
    auto binary = mlir::cast<mlir::AffineBinaryOpExpr>(expr);
    std::optional<int64_t> lhs = evalAt(binary.getLHS(), point);
    std::optional<int64_t> rhs = evalAt(binary.getRHS(), point);
    if (!lhs || !rhs || *rhs == 0)
      return std::nullopt;
    if (expr.getKind() == Kind::FloorDiv)
      return llvm::divideFloorSigned(*lhs, *rhs);
    if (expr.getKind() == Kind::CeilDiv)
      return llvm::divideCeilSigned(*lhs, *rhs);
    int64_t remainder = *lhs % *rhs;
    if (remainder != 0 && ((remainder < 0) != (*rhs < 0)))
      remainder += *rhs;
    return remainder;
  }
  }
  return std::nullopt;
}

/// The per-dimension extents of a layout map's image over the logical box
/// `shape`. For an affine map the extreme value of each result is at a corner
/// of the box, so the corners bound it exactly; a semi-affine map is enumerated
/// over its (bounded) logical points instead.
llvm::Expected<llvm::SmallVector<int64_t, 4>>
imageExtents(mlir::AffineMap map, llvm::ArrayRef<int64_t> shape) {
  const unsigned results = map.getNumResults();
  const unsigned rank = shape.size();

  uint64_t points = 1;
  for (int64_t extent : shape) {
    if (extent < 0)
      return storageError(
          "unsupported footprint: a dynamic extent has no static image");
    if (checkedMul(points, static_cast<uint64_t>(extent), points))
      return storageError(
          "unsupported footprint: the logical index space overflows");
  }
  if (points == 0) {
    // An empty index space occupies nothing.
    return llvm::SmallVector<int64_t, 4>(results, 0);
  }

  bool semiAffine = false;
  for (unsigned j = 0; j < results; ++j)
    semiAffine |= isSemiAffine(map.getResult(j));

  llvm::SmallVector<int64_t, 4> minima(results,
                                       std::numeric_limits<int64_t>::max());
  llvm::SmallVector<int64_t, 4> maxima(results,
                                       std::numeric_limits<int64_t>::min());
  auto visit = [&](llvm::ArrayRef<int64_t> point) -> llvm::Error {
    for (unsigned j = 0; j < results; ++j) {
      std::optional<int64_t> value = evalAt(map.getResult(j), point);
      if (!value)
        return storageError("unsupported footprint: the layout map cannot be "
                            "bounded at a static index");
      minima[j] = std::min(minima[j], *value);
      maxima[j] = std::max(maxima[j], *value);
    }
    return llvm::Error::success();
  };

  if (!semiAffine && rank <= kMaxCornerRank) {
    const uint64_t corners = uint64_t{1} << rank;
    llvm::SmallVector<int64_t, 4> point(rank);
    for (uint64_t mask = 0; mask < corners; ++mask) {
      for (unsigned d = 0; d < rank; ++d)
        point[d] = ((mask >> d) & 1) ? shape[d] - 1 : 0;
      if (llvm::Error error = visit(point))
        return error;
    }
  } else {
    if (points > kFootprintEnumerationLimit)
      return storageError(
          "unsupported footprint: the layout map is not affine and its image "
          "is too large to enumerate");
    llvm::SmallVector<int64_t, 4> point(rank, 0);
    for (uint64_t index = 0; index < points; ++index) {
      uint64_t remainder = index;
      for (unsigned d = 0; d < rank; ++d) {
        point[d] = static_cast<int64_t>(remainder % shape[d]);
        remainder /= static_cast<uint64_t>(shape[d]);
      }
      if (llvm::Error error = visit(point))
        return error;
    }
  }

  llvm::SmallVector<int64_t, 4> extents(results);
  for (unsigned j = 0; j < results; ++j) {
    if (checkedAdd(maxima[j], int64_t{1} - minima[j], extents[j]))
      return storageError("unsupported footprint: a physical extent overflows");
  }
  return extents;
}

} // namespace

//===----------------------------------------------------------------------===//
// Physical footprint
//===----------------------------------------------------------------------===//

llvm::Expected<PhysicalFootprint>
physicalFootprintFor(mlir::Type valueType, mlir::AffineMap map,
                     llvm::ArrayRef<int64_t> padding) {
  if (!valueType)
    return storageError("unsupported footprint: no value type");

  // The logical size (elements and bytes) comes from the shared tile-facts
  // derivation, so this path and the rest of the core never disagree about a
  // value's logical footprint. `facts.bytes = elements * width`, which lets the
  // element width be recovered exactly instead of duplicated here.
  TileFacts facts = tileFactsFor(valueType);
  if (!facts.known)
    return storageError("unsupported footprint: value type '" +
                        printType(valueType) +
                        "' does not state a static, sized tile");
  std::optional<llvm::SmallVector<int64_t, 4>> shape = staticShapeOf(valueType);
  if (!shape)
    return storageError("unsupported footprint: value type '" +
                        printType(valueType) +
                        "' does not state a static shape");

  uint64_t logicalElements = 1;
  for (int64_t extent : *shape) {
    if (extent < 0)
      return storageError(
          "unsupported footprint: a dynamic extent has no static image");
    if (checkedMul(logicalElements, static_cast<uint64_t>(extent),
                   logicalElements))
      return storageError(
          "unsupported footprint: the logical element count overflows");
  }
  if (logicalElements == 0)
    return PhysicalFootprint{0, {}, true};
  const uint64_t elementWidth = facts.bytes / logicalElements;

  llvm::SmallVector<int64_t, 4> physicalShape;
  if (!map || map.getNumResults() == 0) {
    if (!padding.empty())
      return storageError(
          "unsupported footprint: padding requires a layout map");
    physicalShape.append(shape->begin(), shape->end());
  } else {
    if (map.getNumDims() != shape->size())
      return storageError("unsupported footprint: the layout map's rank does "
                          "not match the value");
    if (map.getNumSymbols() != 0)
      return storageError(
          "unsupported footprint: the layout map names symbols");
    if (!padding.empty() && padding.size() != map.getNumResults())
      return storageError("unsupported footprint: the padding rank does not "
                          "match the layout map");
    llvm::Expected<llvm::SmallVector<int64_t, 4>> extents =
        imageExtents(map, *shape);
    if (!extents)
      return extents.takeError();
    physicalShape = std::move(*extents);
    for (unsigned j = 0; j < physicalShape.size(); ++j) {
      int64_t pad = padding.empty() ? 0 : padding[j];
      if (pad < 0)
        return storageError("unsupported footprint: negative padding");
      if (checkedAdd(physicalShape[j], pad, physicalShape[j]) ||
          physicalShape[j] < 0)
        return storageError("unsupported footprint: a padded extent overflows");
    }
  }

  uint64_t physicalElements = 1;
  for (int64_t extent : physicalShape) {
    if (extent < 0)
      return storageError("unsupported footprint: a negative physical extent");
    if (checkedMul(physicalElements, static_cast<uint64_t>(extent),
                   physicalElements))
      return storageError(
          "unsupported footprint: the physical element count overflows");
  }
  uint64_t bytes = 0;
  if (checkedMul(physicalElements, elementWidth, bytes))
    return storageError("unsupported footprint: the physical byte count "
                        "overflows");
  return PhysicalFootprint{bytes, std::move(physicalShape), true};
}

//===----------------------------------------------------------------------===//
// Peak storage
//===----------------------------------------------------------------------===//

llvm::Expected<std::map<MemoryNodeId, uint64_t>>
computePeakStorage(llvm::ArrayRef<StorageAllocation> allocations) {
  if (allocations.empty())
    return std::map<MemoryNodeId, uint64_t>{};

  llvm::DenseMap<uint64_t, size_t> indexById;
  for (size_t i = 0; i < allocations.size(); ++i) {
    if (allocations[i].beginStep > allocations[i].endStep)
      return storageError("storage plan: allocation " +
                          std::to_string(allocations[i].id) +
                          " ends before it begins");
    if (!indexById.insert({allocations[i].id, i}).second)
      return storageError("storage plan: duplicate allocation id " +
                          std::to_string(allocations[i].id));
  }

  // Resolve every allocation to the storage root its alias chain reaches, and
  // prove each alias compatible with that root.
  std::vector<size_t> root(allocations.size(), 0);
  for (size_t i = 0; i < allocations.size(); ++i) {
    size_t current = i;
    llvm::SmallSet<size_t, 8> seen;
    while (allocations[current].aliasOf) {
      if (!seen.insert(current).second)
        return storageError("storage plan: cyclic alias chain at allocation " +
                            std::to_string(allocations[current].id));
      auto it = indexById.find(*allocations[current].aliasOf);
      if (it == indexById.end())
        return storageError("storage plan: allocation " +
                            std::to_string(allocations[i].id) +
                            " aliases unknown allocation " +
                            std::to_string(*allocations[current].aliasOf));
      current = it->second;
    }
    root[i] = current;
    if (current == i)
      continue;
    if (allocations[i].memory != allocations[current].memory)
      return storageError(
          "storage plan: alias " + std::to_string(allocations[i].id) +
          " is in memory '" + allocations[i].memory + "' but its target " +
          std::to_string(allocations[current].id) + " is in '" +
          allocations[current].memory + "'");
    if (allocations[i].bytes > allocations[current].bytes)
      return storageError(
          "storage plan: alias " + std::to_string(allocations[i].id) +
          " needs " + std::to_string(allocations[i].bytes) +
          " bytes but its target " + std::to_string(allocations[current].id) +
          " holds only " + std::to_string(allocations[current].bytes));
  }

  PlanStepId minStep = allocations.front().beginStep;
  PlanStepId maxStep = allocations.front().endStep;
  for (const StorageAllocation &allocation : allocations) {
    minStep = std::min(minStep, allocation.beginStep);
    maxStep = std::max(maxStep, allocation.endStep);
  }
  if (maxStep - minStep > kMaxStorageSteps)
    return storageError(
        "storage plan: the step range is too large to summarize");

  std::map<MemoryNodeId, uint64_t> peak;
  for (PlanStepId step = minStep;; ++step) {
    // Each storage root contributes its own bytes once, however many of its
    // aliases are live at this step.
    std::map<std::pair<MemoryNodeId, size_t>, uint64_t> groups;
    for (size_t i = 0; i < allocations.size(); ++i) {
      const StorageAllocation &allocation = allocations[i];
      if (step < allocation.beginStep || step > allocation.endStep)
        continue;
      groups[{allocation.memory, root[i]}] = allocations[root[i]].bytes;
    }
    std::map<MemoryNodeId, uint64_t> live;
    for (const auto &group : groups) {
      uint64_t &held = live[group.first.first];
      if (checkedAdd(held, group.second, held))
        return storageError("storage plan: live bytes overflow for memory '" +
                            group.first.first + "'");
    }
    for (const auto &entry : live)
      peak[entry.first] = std::max(peak[entry.first], entry.second);
    if (step == maxStep)
      break;
  }
  return peak;
}

//===----------------------------------------------------------------------===//
// Storage-plan finalization
//===----------------------------------------------------------------------===//

llvm::Error finalizeStoragePlan(const WorkloadGraph &graph, CoveringPlan &plan,
                                const machine::MachineModel &machine) {
  // --- placements cover every node ------------------------------------------
  llvm::DenseMap<WorkloadNodeId, const PlanPlacement *> placementFor;
  for (const PlanPlacement &placement : plan.placements)
    if (!placementFor.insert({placement.node, &placement}).second)
      return storageError("storage plan: node " +
                          std::to_string(placement.node) +
                          " has several placements");
  for (const WorkloadNode &node : graph.getNodes())
    if (!placementFor.count(node.id))
      return storageError("storage plan: node " + std::to_string(node.id) +
                          " ('" + node.opName + "') has no placement");

  llvm::DenseMap<InstanceId, WorkloadNodeId> nodeForInstance;
  for (const PlanPlacement &placement : plan.placements)
    nodeForInstance[placement.instance] = placement.node;

  // --- value producer / consumer tables -------------------------------------
  llvm::DenseMap<WorkloadValueId, WorkloadNodeId> valueProducer;
  llvm::DenseMap<WorkloadValueId, std::vector<WorkloadNodeId>> valueConsumers;
  for (const WorkloadNode &node : graph.getNodes()) {
    for (const WorkloadPort &output : node.outputs)
      valueProducer[output.value] = node.id;
    for (const WorkloadPort &input : node.inputs) {
      std::vector<WorkloadNodeId> &consumers = valueConsumers[input.value];
      if (!llvm::is_contained(consumers, node.id))
        consumers.push_back(node.id);
    }
  }

  // --- deterministic dependency-ordered step DAG ----------------------------
  std::map<WorkloadNodeId, std::set<WorkloadNodeId>> successors;
  llvm::DenseMap<WorkloadNodeId, unsigned> indegree;
  for (const WorkloadNode &node : graph.getNodes())
    indegree[node.id] = 0;
  for (const WorkloadNode &node : graph.getNodes()) {
    for (const WorkloadPort &input : node.inputs) {
      auto it = valueProducer.find(input.value);
      if (it == valueProducer.end() || it->second == node.id)
        continue;
      successors[it->second].insert(node.id);
    }
  }
  for (const auto &entry : successors)
    for (WorkloadNodeId consumer : entry.second)
      ++indegree[consumer];

  std::set<WorkloadNodeId> ready;
  for (const WorkloadNode &node : graph.getNodes())
    if (indegree[node.id] == 0)
      ready.insert(node.id);
  std::vector<WorkloadNodeId> topo;
  while (!ready.empty()) {
    WorkloadNodeId node = *ready.begin();
    ready.erase(ready.begin());
    topo.push_back(node);
    auto it = successors.find(node);
    if (it == successors.end())
      continue;
    for (WorkloadNodeId consumer : it->second)
      if (--indegree[consumer] == 0)
        ready.insert(consumer);
  }
  if (topo.size() != graph.getNodes().size())
    return storageError("storage plan: the workload graph is cyclic");

  // Connections grouped by the value they carry, each group ordered by
  // connection id so step assignment does not depend on the plan's vector
  // order.
  std::map<WorkloadValueId, std::vector<size_t>> connectionsByValue;
  for (size_t i = 0; i < plan.connectionPlans.size(); ++i)
    connectionsByValue[plan.connectionPlans[i].value].push_back(i);
  for (auto &entry : connectionsByValue)
    llvm::sort(entry.second, [&](size_t lhs, size_t rhs) {
      return plan.connectionPlans[lhs].id < plan.connectionPlans[rhs].id;
    });

  // A connection that materializes movement needs a wait step and stages a copy
  // in its destination memory.
  auto materializesMovement = [](const PlanConnection &connection) {
    return connection.kind != ConnectionKind::Direct ||
           connection.transform.has_value() || connection.route.size() > 1;
  };

  PlanStepId nextStep = 0;
  llvm::DenseMap<WorkloadNodeId, PlanStepId> computeStep;
  llvm::DenseMap<ConnectionId, PlanStepId> connectionStep;
  llvm::DenseMap<ConnectionId, PlanStepId> syncStep;
  std::vector<char> emitted(plan.connectionPlans.size(), 0);
  std::vector<SynchronizationStep> synchronization;
  std::vector<PlanStep> steps;
  uint64_t nextSyncId = 0;

  auto addStep = [&](PlanStepKind kind, WorkloadNodeId node,
                     ConnectionId connection) {
    const PlanStepId id = nextStep++;
    steps.push_back(PlanStep{id, kind, node, connection});
    return id;
  };

  auto emitConnection = [&](size_t index) {
    const PlanConnection &connection = plan.connectionPlans[index];
    connectionStep[connection.id] =
        addStep(PlanStepKind::Movement, 0, connection.id);
    if (!materializesMovement(connection))
      return;
    SynchronizationStep sync;
    sync.id = nextSyncId++;
    sync.waitsFor.push_back(connection.id);
    sync.precedes.assign(connection.consumerPorts.begin(),
                         connection.consumerPorts.end());
    sync.requiresBarrier = connection.engines.size() > 1;
    synchronization.push_back(std::move(sync));
    // A wait occupies its own step, so a consumer's compute step follows it.
    syncStep[connection.id] =
        addStep(PlanStepKind::Synchronization, 0, connection.id);
  };

  for (WorkloadNodeId nodeId : topo) {
    const WorkloadNode *node = graph.findNode(nodeId);
    llvm::SmallSet<WorkloadValueId, 8> seenValues;
    for (const WorkloadPort &input : node->inputs) {
      if (!seenValues.insert(input.value).second)
        continue;
      auto it = connectionsByValue.find(input.value);
      if (it == connectionsByValue.end())
        continue;
      for (size_t index : it->second)
        if (!emitted[index]) {
          emitConnection(index);
          emitted[index] = 1;
        }
    }
    computeStep[nodeId] = addStep(PlanStepKind::Compute, nodeId, 0);
  }
  for (size_t i = 0; i < plan.connectionPlans.size(); ++i)
    if (!emitted[i]) {
      emitConnection(i);
      emitted[i] = 1;
    }

  // Dependency edges: every movement follows its producer's compute step and
  // precedes each consumer's compute step (through the wait when one exists).
  std::vector<PlanStepEdge> stepEdges;
  for (const PlanConnection &connection : plan.connectionPlans) {
    auto movement = connectionStep.find(connection.id);
    if (movement == connectionStep.end())
      continue;
    auto producer = valueProducer.find(connection.value);
    if (producer != valueProducer.end()) {
      auto step = computeStep.find(producer->second);
      if (step != computeStep.end())
        stepEdges.push_back(PlanStepEdge{step->second, movement->second});
    }
    PlanStepId tail = movement->second;
    auto sync = syncStep.find(connection.id);
    if (sync != syncStep.end()) {
      stepEdges.push_back(PlanStepEdge{movement->second, sync->second});
      tail = sync->second;
    }
    auto consumers = valueConsumers.find(connection.value);
    if (consumers == valueConsumers.end())
      continue;
    for (WorkloadNodeId consumer : consumers->second) {
      auto step = computeStep.find(consumer);
      if (step != computeStep.end())
        stepEdges.push_back(PlanStepEdge{tail, step->second});
    }
  }
  llvm::sort(stepEdges, [](const PlanStepEdge &lhs, const PlanStepEdge &rhs) {
    return lhs.from != rhs.from ? lhs.from < rhs.from : lhs.to < rhs.to;
  });
  stepEdges.erase(std::unique(stepEdges.begin(), stepEdges.end()),
                  stepEdges.end());

  // --- allocation construction ----------------------------------------------
  const bool strict = plan.materialized;
  std::vector<StorageAllocation> allocations;
  uint64_t nextAllocationId = 1;
  // Storage-plan diagnostics and per-connection storage ids, collected locally
  // so a failed finalization leaves the plan untouched rather than
  // half-annotated.
  std::vector<std::string> notes;
  std::vector<std::vector<uint64_t>> connectionStorage(
      plan.connectionPlans.size());

  auto sortedMemoryKeys = [](const llvm::StringMap<MemoryNodeId> &map) {
    std::vector<std::string> keys;
    for (const auto &entry : map)
      keys.push_back(entry.first().str());
    llvm::sort(keys);
    return keys;
  };

  auto primaryMemoryOf =
      [&](const PlanPlacement &placement) -> std::optional<MemoryNodeId> {
    for (const PortMemoryBinding &binding : placement.portMemoryBindings)
      return binding.memory;
    std::vector<std::string> keys = sortedMemoryKeys(placement.memories);
    if (!keys.empty())
      return placement.memories.lookup(keys.front());
    return std::nullopt;
  };

  auto outputValueAt = [&](WorkloadNodeId nodeId,
                           unsigned index) -> std::optional<WorkloadValueId> {
    const WorkloadNode *node = graph.findNode(nodeId);
    if (!node || index >= node->outputs.size())
      return std::nullopt;
    return node->outputs[index].value;
  };

  auto outputMemoryOf =
      [&](const PlanPlacement &placement, const PortRef &ref,
          WorkloadValueId value) -> std::optional<MemoryNodeId> {
    // The producing placement's own binding governs where the value is written;
    // a connection's destination is only where a *copy* of it may move.
    for (const PortMemoryBinding &binding : placement.portMemoryBindings)
      if (binding.port == ref)
        return binding.memory;
    for (const PortMemoryBinding &binding : placement.portMemoryBindings)
      if (binding.port.direction == PortDirection::Output) {
        std::optional<WorkloadValueId> bound =
            outputValueAt(placement.node, binding.port.index);
        if (bound && *bound == value)
          return binding.memory;
      }
    if (std::optional<MemoryNodeId> memory = primaryMemoryOf(placement))
      return memory;
    for (const PlanConnection &connection : plan.connectionPlans)
      if (connection.value == value && !connection.route.empty())
        return connection.route.back();
    return std::nullopt;
  };

  auto layoutMapFor = [&](const PlanPlacement &placement, const PortRef &ref,
                          WorkloadValueId value) -> mlir::AffineMap {
    std::vector<std::string> keys;
    for (const auto &entry : placement.layoutSolutions)
      keys.push_back(entry.first().str());
    llvm::sort(keys);
    for (const std::string &key : keys) {
      const SolvedLayout &solution = placement.layoutSolutions.lookup(key);
      if (solution.port && *solution.port == ref)
        return solution.map;
    }
    for (const std::string &key : keys) {
      const SolvedLayout &solution = placement.layoutSolutions.lookup(key);
      if (solution.portValue >= 0 &&
          static_cast<WorkloadValueId>(solution.portValue) == value)
        return solution.map;
    }
    return {};
  };

  auto footprintBytes =
      [&](WorkloadValueId value, const PortRef &ref,
          const PlanPlacement &placement) -> llvm::Expected<uint64_t> {
    const WorkloadValue *workloadValue = graph.findValue(value);
    if (!workloadValue)
      return storageError("storage plan: value " + std::to_string(value) +
                          " is not in the workload graph");
    llvm::Expected<PhysicalFootprint> footprint = physicalFootprintFor(
        workloadValue->type, layoutMapFor(placement, ref, value), {});
    if (footprint)
      return footprint->bytes;
    llvm::Error error = footprint.takeError();
    if (strict)
      return error;
    notes.push_back("storage plan: value " + std::to_string(value) + " ('" +
                    workloadValue->name +
                    "'): " + llvm::toString(std::move(error)) + "; assuming " +
                    std::to_string(kAssumedValueBytes) + " bytes for analysis");
    return kAssumedValueBytes;
  };

  auto lastReaderStep =
      [&](WorkloadValueId value) -> std::optional<PlanStepId> {
    auto it = valueConsumers.find(value);
    if (it == valueConsumers.end())
      return std::nullopt;
    std::optional<PlanStepId> best;
    for (WorkloadNodeId consumer : it->second) {
      auto step = computeStep.find(consumer);
      if (step == computeStep.end())
        continue;
      if (!best || step->second > *best)
        best = step->second;
    }
    return best;
  };

  auto allocate = [&](WorkloadValueId value, const std::string &memory,
                      uint64_t bytes, PlanStepId begin, PlanStepId end) {
    StorageAllocation allocation;
    allocation.id = nextAllocationId++;
    allocation.value = value;
    allocation.memory = memory;
    allocation.bytes = bytes;
    allocation.beginStep = begin;
    allocation.endStep = end;
    allocations.push_back(allocation);
    return allocation.id;
  };

  for (WorkloadNodeId nodeId : topo) {
    const WorkloadNode *node = graph.findNode(nodeId);
    const PlanPlacement *placement = placementFor.lookup(nodeId);

    // Execution multiplicity: how many times this node runs, recovered during
    // extraction from its enclosing structural ops. An unknown count is refused
    // by a strict (materialized) plan and reported by an analysis one.
    //
    // The analysis-mode fallback uses a single iteration. That is a *lower*
    // bound, not a conservative one: a genuinely unknown loop may run many
    // times, so an analysis-mode capacity result is advisory and must not be
    // read as a proof. No finite upper bound exists for an unknown trip count,
    // which is exactly why an executable (materialized) plan refuses instead of
    // guessing -- strict mode is the gate, and the note below names the node so
    // the under-report is never silent.
    const std::optional<uint64_t> multiplicity = node->executionMultiplicity;
    uint64_t scale = 1;
    if (multiplicity) {
      scale = *multiplicity;
    } else if (strict) {
      return storageError(
          "storage plan: node " + std::to_string(nodeId) + " ('" +
          node->opName +
          "') has unknown execution multiplicity; strict "
          "executable planning cannot assume a single iteration "
          "(recover the bound from the enclosing structural ops "
          "or use analysis mode)");
    } else {
      notes.push_back(
          "storage plan: node " + std::to_string(nodeId) + " ('" +
          node->opName +
          "') has unknown execution multiplicity; the single-iteration used "
          "here is a lower bound, so this analysis result is advisory");
    }

    for (unsigned index = 0; index < node->outputs.size(); ++index) {
      const WorkloadValueId value = node->outputs[index].value;
      const PortRef ref{nodeId, PortDirection::Output, index};
      std::optional<MemoryNodeId> memory =
          outputMemoryOf(*placement, ref, value);
      if (!memory)
        return storageError("storage plan: no memory is bound to value " +
                            std::to_string(value) + " written by node " +
                            std::to_string(nodeId) + " output " +
                            std::to_string(index));
      llvm::Expected<uint64_t> base = footprintBytes(value, ref, *placement);
      if (!base)
        return base.takeError();
      uint64_t bytes = 0;
      if (checkedMul(*base, scale, bytes))
        return storageError("storage plan: the footprint of value " +
                            std::to_string(value) + " overflows");
      const PlanStepId begin = computeStep.lookup(nodeId);
      std::optional<PlanStepId> last = lastReaderStep(value);
      const PlanStepId end = last ? std::max(begin, *last) : begin;
      const uint64_t id = allocate(value, *memory, bytes, begin, end);
      auto connections = connectionsByValue.find(value);
      if (connections != connectionsByValue.end())
        for (size_t connectionIndex : connections->second)
          if (!llvm::is_contained(connectionStorage[connectionIndex], id))
            connectionStorage[connectionIndex].push_back(id);
    }
  }

  // A movement the plan materializes stages a copy in its destination memory,
  // distinct from the producer's storage (a transform preserves its immutable
  // source), so the copy is charged its own interval.
  for (size_t index = 0; index < plan.connectionPlans.size(); ++index) {
    PlanConnection &connection = plan.connectionPlans[index];
    if (!materializesMovement(connection))
      continue;
    const WorkloadValue *workloadValue = graph.findValue(connection.value);
    if (!workloadValue)
      return storageError(
          "storage plan: connection " + std::to_string(connection.id) +
          " carries unknown value " + std::to_string(connection.value));

    std::optional<MemoryNodeId> memory;
    if (!connection.route.empty())
      memory = connection.route.back();
    else if (!connection.consumers.empty()) {
      auto node = nodeForInstance.find(connection.consumers.front());
      if (node != nodeForInstance.end())
        memory = primaryMemoryOf(*placementFor.lookup(node->second));
    }
    if (!memory)
      return storageError("storage plan: connection " +
                          std::to_string(connection.id) +
                          " has no destination memory for its staged copy");

    mlir::AffineMap destinationMap;
    if (connection.transform)
      destinationMap = connection.transform->dstMap;
    llvm::Expected<PhysicalFootprint> footprint =
        physicalFootprintFor(workloadValue->type, destinationMap, {});
    uint64_t bytes = 0;
    if (footprint) {
      bytes = footprint->bytes;
    } else {
      llvm::Error error = footprint.takeError();
      if (strict)
        return error;
      notes.push_back(
          "storage plan: connection " + std::to_string(connection.id) +
          " stage: " + llvm::toString(std::move(error)) + "; assuming " +
          std::to_string(kAssumedValueBytes) + " bytes for analysis");
      bytes = kAssumedValueBytes;
    }

    const PlanStepId begin = connectionStep.lookup(connection.id);
    std::optional<PlanStepId> last = lastReaderStep(connection.value);
    const PlanStepId end = last ? std::max(begin, *last) : begin;
    const uint64_t id = allocate(connection.value, *memory, bytes, begin, end);
    if (!llvm::is_contained(connectionStorage[index], id))
      connectionStorage[index].push_back(id);
  }

  // --- capacity validation and occupancy report -----------------------------
  llvm::Expected<std::map<MemoryNodeId, uint64_t>> peak =
      computePeakStorage(allocations);
  if (!peak)
    return peak.takeError();
  for (const auto &entry : *peak) {
    const machine::MemoryNode *memory = machine.findMemory(entry.first);
    if (!memory)
      return storageError("storage plan: the plan binds unknown memory '" +
                          entry.first + "'");
    if (entry.second > memory->capacityBytes)
      return storageError(
          "storage plan: memory '" + entry.first + "' over capacity (peak " +
          std::to_string(entry.second) + " bytes live, " +
          std::to_string(memory->capacityBytes) + " byte capacity)");
  }
  for (const auto &entry : *peak) {
    const machine::MemoryNode *memory = machine.findMemory(entry.first);
    notes.push_back("storage plan: memory '" + entry.first + "' peak " +
                    std::to_string(entry.second) + " bytes of " +
                    std::to_string(memory->capacityBytes));
  }

  plan.diagnostics.storageNotes = std::move(notes);
  for (size_t index = 0; index < plan.connectionPlans.size(); ++index)
    plan.connectionPlans[index].storageIds.assign(
        connectionStorage[index].begin(), connectionStorage[index].end());
  plan.steps = std::move(steps);
  plan.stepEdges = std::move(stepEdges);
  plan.allocations = std::move(allocations);
  plan.synchronization = std::move(synchronization);
  return llvm::Error::success();
}

} // namespace mlir::llk::mapping
