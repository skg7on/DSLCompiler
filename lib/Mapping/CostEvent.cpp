//===- CostEvent.cpp - Shared cost-event vocabulary -----------------------===//

#include "LLK/Mapping/CostEvent.h"

#include "LLK/Machine/MachineModel.h"
#include "LLK/Mapping/MappingPlan.h"
#include "LLK/Mapping/TileFacts.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Error.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <vector>

namespace mlir::llk::mapping {

namespace {
struct KindInfo {
  CostEventKind kind;
  llvm::StringLiteral name;
};

/// Declaration order is the canonical order.
constexpr std::array<KindInfo, 5> kKinds{{
    {CostEventKind::Compute, "compute"},
    {CostEventKind::TransferHop, "transfer_hop"},
    {CostEventKind::Transform, "transform"},
    {CostEventKind::Synchronization, "synchronization"},
    {CostEventKind::Capacity, "capacity"},
}};

llvm::Error planEventError(const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(), message);
}

/// The compute engine a placement's executor owns, preferring a vector engine
/// -- the same explicit default policy `pickVectorEngine`/`pickMatrixEngine`
/// use in the performance DAG. An executor that owns no compute node falls back
/// to the machine's declared first engine, so a plan event always names a real
/// pool. Null only when the machine declares no compute node at all.
const machine::ComputeNode *
engineForExecutor(const machine::MachineModel &machine,
                  llvm::StringRef executor) {
  if (!executor.empty()) {
    for (const machine::ComputeNode *node : machine.computesFor(executor))
      if (node->kind == "vector_engine")
        return node;
    for (const machine::ComputeNode *node : machine.computesFor(executor))
      if (node->kind == "matrix_engine")
        return node;
    for (const machine::ComputeNode *node : machine.computesFor(executor))
      return node;
  }
  std::vector<const machine::ComputeNode *> vectors =
      machine.computesOfKind("vector_engine");
  if (!vectors.empty())
    return vectors.front();
  std::vector<const machine::ComputeNode *> matrices =
      machine.computesOfKind("matrix_engine");
  if (!matrices.empty())
    return matrices.front();
  return machine.computes.empty() ? nullptr : &machine.computes.front();
}

/// The link joining two concrete memory nodes, or null when the machine
/// declares none. Node identity, never kind equality, decides.
const machine::LinkEdge *linkBetween(const machine::MachineModel &machine,
                                     llvm::StringRef from, llvm::StringRef to) {
  for (const machine::LinkEdge &edge : machine.links)
    if (edge.source == from && edge.destination == to)
      return &edge;
  return nullptr;
}

/// The executor whose engine runs a connection's transform: the first consumer
/// placement's, else the producer placement's. This is the resource the binder
/// stamps on the emitted `micro.transform` (task B8), so the plan's normalized
/// transform event and the materialized one name the same pool.
std::string transformExecutor(const CoveringPlan &plan,
                              const PlanConnection &connection) {
  for (const PlanPlacement &placement : plan.placements)
    for (InstanceId consumer : connection.consumers)
      if (placement.instance == consumer)
        return placement.executor;
  if (connection.producerPort)
    for (const PlanPlacement &placement : plan.placements)
      if (placement.node == connection.producerPort->node)
        return placement.executor;
  return {};
}

/// The node a connection serves, for the data dependency from a movement to the
/// consumer that reads it: the first consumer instance's covered node.
std::optional<WorkloadNodeId> consumerNodeOf(const CoveringPlan &plan,
                                             const PlanConnection &connection) {
  for (const PlanPlacement &placement : plan.placements)
    for (InstanceId consumer : connection.consumers)
      if (placement.instance == consumer)
        return placement.node;
  return std::nullopt;
}

} // namespace

llvm::StringRef stringifyCostEventKind(CostEventKind kind) {
  for (const KindInfo &info : kKinds)
    if (info.kind == kind)
      return info.name;
  return "";
}

std::optional<CostEventKind> symbolizeCostEventKind(llvm::StringRef text) {
  for (const KindInfo &info : kKinds)
    if (info.name == text)
      return info.kind;
  return std::nullopt;
}

PlanCostEvent makePlanCostEvent(CostEventKind kind, std::string resource,
                                double latencyCycles, uint64_t workItems,
                                uint64_t bytes, std::vector<uint32_t> deps) {
  PlanCostEvent normalized;
  normalized.event.kind = kind;
  normalized.event.resource = std::move(resource);
  normalized.event.cost.latencyCycles = latencyCycles;
  normalized.event.cost.localBytes = bytes;
  normalized.workItems = workItems;
  normalized.bytes = bytes;
  llvm::sort(deps);
  deps.erase(std::unique(deps.begin(), deps.end()), deps.end());
  normalized.deps = std::move(deps);
  return normalized;
}

llvm::Expected<PlanEventDAG>
buildPlanEvents(const CoveringPlan &plan,
                const machine::MachineModel &machine) {
  if (plan.steps.empty())
    return planEventError(
        "plan events: the plan carries no step DAG; finalize its storage plan "
        "before building event costs");

  // The placement covering each node, and the placement for each instance, so a
  // compute step and a connection's consumers resolve without the workload
  // graph.
  std::map<WorkloadNodeId, const PlanPlacement *> placementForNode;
  for (const PlanPlacement &placement : plan.placements)
    placementForNode[placement.node] = &placement;

  // The steps sorted by id: the storage plan builds them in a deterministic
  // topological order, so their ids are the execution order.
  std::vector<const PlanStep *> steps;
  steps.reserve(plan.steps.size());
  for (const PlanStep &step : plan.steps)
    steps.push_back(&step);
  llvm::sort(steps, [](const PlanStep *lhs, const PlanStep *rhs) {
    return lhs->id < rhs->id;
  });

  PlanEventDAG dag;
  // The events each step produced, and the connection each Movement step
  // materializes, so dependency edges and the data edge to a consumer resolve.
  std::map<PlanStepId, std::vector<uint32_t>> stepEvents;
  std::map<ConnectionId, PlanStepId> movementStep;

  auto add = [&](PlanCostEvent event) -> uint32_t {
    uint32_t id = static_cast<uint32_t>(dag.events.size());
    dag.events.push_back(std::move(event));
    return id;
  };

  for (const PlanStep *step : steps) {
    std::vector<uint32_t> produced;
    switch (step->kind) {
    case PlanStepKind::Compute: {
      auto placement = placementForNode.find(step->node);
      if (placement == placementForNode.end())
        return planEventError("plan events: compute step " +
                              llvm::Twine(step->id) + " names node " +
                              llvm::Twine(step->node) +
                              ", which no placement covers");
      const PlanPlacement &placed = *placement->second;
      const machine::ComputeNode *engine =
          engineForExecutor(machine, placed.executor);
      if (!engine)
        return planEventError("plan events: machine '" + machine.target +
                              "' declares no compute resource for node " +
                              llvm::Twine(step->node));
      // A compute event accounts for work, not traffic: the performance DAG's
      // vector/mma events carry no bytes, so this one carries none either and
      // the two streams stay comparable.
      produced.push_back(add(makePlanCostEvent(
          CostEventKind::Compute, engine->id, placed.cost.latencyCycles,
          placed.workItems, /*bytes=*/0)));
      break;
    }
    case PlanStepKind::Movement: {
      auto found = llvm::find_if(plan.connectionPlans,
                                 [&](const PlanConnection &connection) {
                                   return connection.id == step->connection;
                                 });
      if (found == plan.connectionPlans.end())
        return planEventError("plan events: movement step " +
                              llvm::Twine(step->id) + " names connection " +
                              llvm::Twine(step->connection) +
                              ", which the plan does not record");
      const PlanConnection &connection = *found;
      movementStep[connection.id] = step->id;

      // The transfer hops (and the wait each implies) a materialized movement
      // emits, one event per hop, chained so the data arrives in order.
      const auto emitHops = [&](uint64_t bytes, uint64_t workItems,
                                std::optional<uint32_t> chainFrom)
          -> llvm::Expected<std::vector<uint32_t>> {
        std::vector<uint32_t> hops;
        std::optional<uint32_t> previousWait = chainFrom;
        for (size_t hop = 1; hop < connection.route.size(); ++hop) {
          const machine::MemoryNode *from =
              machine.findMemory(connection.route[hop - 1]);
          const machine::MemoryNode *to =
              machine.findMemory(connection.route[hop]);
          if (!from || !to)
            return planEventError("plan events: connection " +
                                  llvm::Twine(connection.id) +
                                  " routes through unknown memory");
          const machine::LinkEdge *link =
              linkBetween(machine, from->id, to->id);
          if (!link)
            return planEventError(
                "plan events: connection " + llvm::Twine(connection.id) +
                " routes across a link the machine does not declare (" +
                from->id + " -> " + to->id + ")");
          uint64_t cycles = link->latencyCycles;
          if (link->bandwidthBytesPerCycle > 0)
            cycles += static_cast<uint64_t>(std::ceil(
                static_cast<double>(bytes) / link->bandwidthBytesPerCycle));
          std::string resource =
              link->transferEngines.empty() ? "dma" : link->transferEngines[0];
          // A hop follows the wait that preceded it -- the program-order edge
          // the materialized copy chain carries -- and the materialized
          // movement waits on the hop that just landed.
          std::vector<uint32_t> deps;
          if (previousWait)
            deps.push_back(*previousWait);
          uint32_t transfer = add(makePlanCostEvent(
              CostEventKind::TransferHop, resource, static_cast<double>(cycles),
              workItems, bytes, deps));
          hops.push_back(transfer);
          uint32_t wait = add(makePlanCostEvent(
              CostEventKind::Synchronization, "sync",
              static_cast<double>(machine.sync.waitCycles), 0, 0, {transfer}));
          hops.push_back(wait);
          previousWait = wait;
        }
        return hops;
      };

      std::optional<uint32_t> chainFrom;
      const uint64_t bytes = connection.cost.localBytes;
      const uint64_t workItems = connection.workItems;

      if (connection.kind == ConnectionKind::Reduce) {
        if (!connection.gatherSemantics)
          return planEventError("plan events: gather connection " +
                                llvm::Twine(connection.id) +
                                " declares no combination semantics");
        const size_t feeds = connection.producerPorts.size();
        if (feeds == 0)
          return planEventError("plan events: gather connection " +
                                llvm::Twine(connection.id) +
                                " records no producer feeds");
        std::vector<uint32_t> feedTails;
        for (size_t feed = 0; feed < feeds; ++feed) {
          llvm::Expected<std::vector<uint32_t>> hops =
              emitHops(bytes, workItems, std::nullopt);
          if (!hops)
            return hops.takeError();
          produced.insert(produced.end(), hops->begin(), hops->end());
          if (!hops->empty())
            feedTails.push_back(hops->back());
        }
        const machine::ComputeNode *engine =
            engineForExecutor(machine, transformExecutor(plan, connection));
        if (!engine)
          return planEventError(
              "plan events: machine '" + machine.target +
              "' declares no compute resource for gather connection " +
              llvm::Twine(connection.id));
        std::string dtype =
            elementTypeName(elementTypeOf(connection.valueType));
        produced.push_back(add(makePlanCostEvent(
            CostEventKind::Compute, engine->id,
            static_cast<double>(elementwiseCycles(*engine, dtype, workItems)),
            workItems, /*bytes=*/0, feedTails)));
        break;
      }

      llvm::Expected<std::vector<uint32_t>> hops =
          emitHops(bytes, workItems, chainFrom);
      if (!hops)
        return hops.takeError();
      produced.insert(produced.end(), hops->begin(), hops->end());
      if (!hops->empty())
        chainFrom = hops->back();

      // A layout conversion is real work; its cost is the *shared* estimate the
      // materialized kernel's event also uses, so the two paths cannot
      // disagree.
      if (connection.transform) {
        const machine::ComputeNode *engine =
            engineForExecutor(machine, transformExecutor(plan, connection));
        if (!engine)
          return planEventError(
              "plan events: machine '" + machine.target +
              "' declares no compute resource for transform connection " +
              llvm::Twine(connection.id));
        mlir::Type type = tileAsTensor(connection.valueType);
        if (!type)
          type = connection.valueType;
        TransformCostInput input;
        input.inputType = type;
        input.outputType = type;
        input.srcMap = connection.transform->srcMap;
        input.dstMap = connection.transform->dstMap;
        if (!connection.route.empty())
          input.memoryNode = connection.route.back();
        input.computeResource = engine->id;
        llvm::Expected<Cost> cost = estimateTransformCost(input, machine);
        if (!cost)
          return planEventError("plan events: transform connection " +
                                llvm::Twine(connection.id) + ": " +
                                llvm::toString(cost.takeError()));
        std::vector<uint32_t> deps;
        if (chainFrom)
          deps.push_back(*chainFrom);
        produced.push_back(add(makePlanCostEvent(
            CostEventKind::Transform, engine->id, cost->latencyCycles,
            workItems, cost->localBytes, deps)));
      }
      break;
    }
    case PlanStepKind::Synchronization: {
      // A movement's waits are emitted per hop above. The only additional
      // synchronization a step contributes is the barrier the plan asked for.
      bool barrier = false;
      for (const SynchronizationStep &sync : plan.synchronization)
        if (sync.requiresBarrier && sync.waitsFor.size() == 1 &&
            sync.waitsFor.front() == step->connection) {
          barrier = true;
          break;
        }
      if (!barrier)
        break;
      produced.push_back(add(makePlanCostEvent(
          CostEventKind::Synchronization, "sync",
          static_cast<double>(machine.sync.barrierCycles), 0, 0)));
      break;
    }
    }
    stepEvents[step->id] = std::move(produced);
  }

  // Inter-step dependencies: every event of a dependent step follows the last
  // event of each step it depends on.
  for (const PlanStepEdge &edge : plan.stepEdges) {
    auto from = stepEvents.find(edge.from);
    auto to = stepEvents.find(edge.to);
    if (from == stepEvents.end() || to == stepEvents.end() ||
        from->second.empty() || to->second.empty())
      continue;
    uint32_t predecessor = from->second.back();
    uint32_t &head = dag.events[to->second.front()].deps.emplace_back();
    head = predecessor;
  }

  // A movement's data dependency: the consumer compute event depends on the
  // movement event, not only on the wait that orders it (the same edge the
  // performance DAG derives from the consumer's operand).
  for (const PlanConnection &connection : plan.connectionPlans) {
    auto movement = movementStep.find(connection.id);
    if (movement == movementStep.end())
      continue;
    auto events = stepEvents.find(movement->second);
    if (events == stepEvents.end() || events->second.empty())
      continue;
    std::optional<WorkloadNodeId> consumerNode =
        consumerNodeOf(plan, connection);
    if (!consumerNode)
      continue;
    for (const PlanStep &step : plan.steps) {
      if (step.kind != PlanStepKind::Compute || step.node != *consumerNode)
        continue;
      auto computeEvents = stepEvents.find(step.id);
      if (computeEvents == stepEvents.end() || computeEvents->second.empty())
        continue;
      dag.events[computeEvents->second.front()].deps.push_back(
          events->second.back());
    }
  }

  for (PlanCostEvent &event : dag.events) {
    llvm::sort(event.deps);
    event.deps.erase(std::unique(event.deps.begin(), event.deps.end()),
                     event.deps.end());
  }
  return dag;
}

} // namespace mlir::llk::mapping
