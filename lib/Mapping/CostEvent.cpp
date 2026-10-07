//===- CostEvent.cpp - Shared cost-event vocabulary -----------------------===//

#include "LLK/Mapping/CostEvent.h"

#include "LLK/Machine/MachineModel.h"
#include "LLK/Mapping/EventSchedule.h"
#include "LLK/Mapping/LatencyProvider.h"
#include "LLK/Mapping/MappingPlan.h"
#include "LLK/Mapping/StoragePlan.h"
#include "LLK/Mapping/TileFacts.h"
#include "LLK/Mapping/WorkloadGraph.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

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

/// The compute node a placement *recorded* as selected, validated against the
/// machine before its id is used (issue #129, task R1). Returns:
///
///   - null when the placement recorded no compute binding -- a hand-built or
///     pre-#129 plan -- so the caller keeps the executor-based fallback;
///   - the selected node otherwise, preferring a vector engine, then a matrix
///     engine, then the first selection in sorted key order. That mirrors the
///     priority `engineForExecutor` applies to an executor's attachments, so a
///     single-capability placement normalizes exactly as it did before;
///   - an error when a recorded node is unknown to the machine, or when it is
///     not attached to the placement's executor. A recorded selection that no
///     longer resolves is never silently replaced by executor order: that is
///     the very re-derivation this task removes.
llvm::Expected<const machine::ComputeNode *>
recordedComputeNode(const machine::MachineModel &machine,
                    const PlanPlacement &placement) {
  if (placement.computeBindings.empty())
    return static_cast<const machine::ComputeNode *>(nullptr);
  // Sorted keys, so the choice is a function of the recorded content and never
  // of `StringMap` iteration order.
  std::vector<std::string> keys;
  keys.reserve(placement.computeBindings.size());
  for (const auto &entry : placement.computeBindings)
    keys.push_back(entry.first().str());
  llvm::sort(keys);
  const machine::ComputeNode *vector = nullptr;
  const machine::ComputeNode *matrix = nullptr;
  const machine::ComputeNode *other = nullptr;
  for (const std::string &key : keys) {
    const std::string &id = placement.computeBindings.lookup(key);
    const machine::ComputeNode *node = machine.findCompute(id);
    if (!node)
      return planEventError(
          "plan events: placement for node " + llvm::Twine(placement.node) +
          " records compute node '" + id + "', which machine '" +
          machine.target + "' does not declare");
    if (!placement.executor.empty() && node->attachedTo != placement.executor)
      return planEventError("plan events: placement for node " +
                            llvm::Twine(placement.node) +
                            " records compute node '" + id +
                            "', which is not attached to its executor '" +
                            placement.executor + "'");
    if (node->kind == "vector_engine" && !vector)
      vector = node;
    else if (node->kind == "matrix_engine" && !matrix)
      matrix = node;
    else if (!other)
      other = node;
  }
  if (vector)
    return vector;
  if (matrix)
    return matrix;
  return other;
}

/// The engine a compute event for `placement` runs on, with the "a required
/// kind must be recorded" diagnosis applied before any fallback is even
/// considered (issue #129, task R1).
///
/// A placement whose rule requires a compute kind but which records no concrete
/// node for it has had its selection dropped or tampered with. Returning null
/// there would let the caller re-derive an engine from the executor's first
/// attachment -- the exact silent substitution this task removes -- so it is an
/// error instead. Null is returned only for a placement that genuinely records
/// nothing to select: a rule that requires no capability (a copy or store
/// placement) or a hand-built plan.
llvm::Expected<const machine::ComputeNode *>
recordedEngineForPlacement(const machine::MachineModel &machine,
                           const PlanPlacement &placement) {
  if (std::optional<std::string> missing = missingComputeBinding(placement))
    return planEventError(
        "plan events: placement for node " + llvm::Twine(placement.node) +
        " requires a '" + *missing +
        "' compute capability but the plan records no concrete node for it; "
        "refusing to re-derive one from executor order");
  return recordedComputeNode(machine, placement);
}

/// The engine a connection's transform runs on: the concrete resource the
/// plan *recorded* for it (the same one `selectTransformResource` chose), never
/// a re-derivation from executor order. Falls back to the executor's first
/// attached engine only for a connection that recorded no resource at all
/// (issue #129, task R1).
llvm::Expected<const machine::ComputeNode *>
transformEngineFor(const PlanConnection &connection,
                   const machine::MachineModel &machine,
                   const CoveringPlan &plan) {
  if (connection.transform && !connection.transform->computeResource.empty()) {
    const std::string &recorded = connection.transform->computeResource;
    const machine::ComputeNode *node = machine.findCompute(recorded);
    if (!node)
      return planEventError(
          "plan events: transform connection " + llvm::Twine(connection.id) +
          " records compute resource '" + recorded + "', which machine '" +
          machine.target + "' does not declare");
    return node;
  }
  return engineForExecutor(machine, transformExecutorFor(plan, connection));
}

/// The first placement serving one of `connection`'s consumer instances, in
/// placement order -- the same placement `transformExecutorFor` takes the
/// executor from. Null when no recorded consumer instance resolves to a
/// placement.
const PlanPlacement *consumerPlacementOf(const CoveringPlan &plan,
                                         const PlanConnection &connection) {
  for (const PlanPlacement &placement : plan.placements)
    if (llvm::is_contained(connection.consumers, placement.instance))
      return &placement;
  return nullptr;
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

/// Checks a derived analysis stream before it is trusted (issue #129, task R6).
///
/// A snapshot is a claim about the plan's own work: the event stream the kernel
/// this plan bound actually does. It is accepted only when that claim is
/// internally consistent with the plan -- a non-empty stream, every resource
/// modelled by the machine, a real acyclic dependency order, and every
/// step/connection it names present in the plan. Anything else is an error
/// rather than a silently scored stream, because a stale or foreign snapshot
/// would make the plan's score describe work it does not do.
llvm::Error verifyPlanAnalysisStream(const CoveringPlan &plan,
                                     const PlanEventDAG &events,
                                     const machine::MachineModel &machine) {
  if (events.events.empty())
    return planEventError(
        "plan events: the plan's analysis snapshot carries no events");
  if (llvm::Error error = validateEventResources(events, machine))
    return planEventError("plan events: the plan's analysis snapshot is not a "
                          "valid stream: " +
                          llvm::toString(std::move(error)));

  // The recorded compute selection each placement claims (issue #129, task
  // R1) is part of what the snapshot is a snapshot *of*: a plan whose required
  // capability has no concrete recorded node, or whose recorded node the
  // machine does not declare, must be refused here exactly as the accumulation
  // path refuses it -- otherwise a tampered plan would quietly present the
  // materialized kernel's stream as if it were its own. Reusing
  // `recordedEngineForPlacement` keeps one wording for one defect.
  for (const PlanPlacement &placement : plan.placements)
    if (llvm::Expected<const machine::ComputeNode *> engine =
            recordedEngineForPlacement(machine, placement);
        !engine)
      return engine.takeError();

  std::vector<uint64_t> stepIds;
  stepIds.reserve(plan.steps.size());
  for (const PlanStep &step : plan.steps)
    stepIds.push_back(step.id);
  llvm::sort(stepIds);
  std::vector<uint64_t> connectionIds;
  connectionIds.reserve(plan.connectionPlans.size());
  for (const PlanConnection &connection : plan.connectionPlans)
    connectionIds.push_back(connection.id);
  llvm::sort(connectionIds);

  for (size_t index = 0; index < events.events.size(); ++index) {
    const PlanCostEvent &event = events.events[index];
    if (event.planStep &&
        !std::binary_search(stepIds.begin(), stepIds.end(), *event.planStep))
      return planEventError("plan events: the plan's analysis snapshot event " +
                            llvm::Twine(index) + " names plan step " +
                            llvm::Twine(*event.planStep) +
                            ", which the plan does not record");
    if (event.connectionId &&
        !std::binary_search(connectionIds.begin(), connectionIds.end(),
                            *event.connectionId))
      return planEventError("plan events: the plan's analysis snapshot event " +
                            llvm::Twine(index) + " names connection " +
                            llvm::Twine(*event.connectionId) +
                            ", which the plan does not record");
  }
  return llvm::Error::success();
}

} // namespace

llvm::StringRef stringifyPlanEventSource(PlanEventSource source) {
  switch (source) {
  case PlanEventSource::Accumulation:
    return "accumulation";
  case PlanEventSource::Snapshot:
    return "snapshot";
  }
  return "";
}

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

llvm::StringRef stringifyStorageAccess(StorageAccess access) {
  switch (access) {
  case StorageAccess::Read:
    return "read";
  case StorageAccess::Write:
    return "write";
  }
  return "";
}

llvm::Expected<PlanEventDAG>
buildPlanEvents(const CoveringPlan &plan, const machine::MachineModel &machine,
                const WorkloadGraph *graph) {
  // A plan whose kernel has been bound and analyzed carries the derived
  // normalized snapshot the shared selected-kernel analysis produced from that
  // kernel (issue #129, task R6). It is the materialized work's own event
  // stream -- machine formulas, owner pools, movement memories, dependency
  // edges -- so the planner's score and `micro-perf`'s prediction are one
  // analysis of one kernel rather than two estimates of the same work.
  //
  // The snapshot is used only for the *schedule-level* call. The graph-carrying
  // call is the storage-liveness path: it needs each allocation's identity and
  // access, which the kernel-side extraction does not record (a plan's
  // allocation ids are the planner's own), so it always derives the stream from
  // the plan's storage facts. A snapshot is never silently substituted where it
  // would be the wrong shape; it is verified or it is an error.
  if (plan.analysisEvents && graph == nullptr) {
    if (llvm::Error error =
            verifyPlanAnalysisStream(plan, *plan.analysisEvents, machine))
      return std::move(error);
    PlanEventDAG snapshot = *plan.analysisEvents;
    snapshot.source = PlanEventSource::Snapshot;
    return snapshot;
  }

  // The placement covering each node, and the placement for each instance, so a
  // compute step and a connection's consumers resolve without the workload
  // graph.
  std::map<WorkloadNodeId, const PlanPlacement *> placementForNode;
  std::map<InstanceId, WorkloadNodeId> nodeForInstance;
  for (const PlanPlacement &placement : plan.placements) {
    placementForNode[placement.node] = &placement;
    nodeForInstance[placement.instance] = placement.node;
  }

  // A plan decoded from persisted metadata carries no execution facts -- they
  // are provenance, not content -- so scoring it would silently undercharge
  // every event. Reject explicitly rather than emit zero-cycle events.
  if (plan.schemaVersion != 0) {
    for (const PlanPlacement &placement : plan.placements)
      if (placement.cost.latencyCycles == 0.0 &&
          placement.cost.localBytes == 0 && placement.workItems == 0)
        return planEventError(
            "plan events: node " + llvm::Twine(placement.node) +
            " carries no execution facts; a plan decoded from metadata must be "
            "re-scored from its search result");
  }

  // The execution structure. A finalized plan carries its storage step DAG; a
  // plan scored before finalization (the mapping search's candidate score) gets
  // a deterministic one synthesized from its placements and connections, so the
  // final score comes from the shared schedule without a separate, weaker
  // scoring path.
  std::vector<PlanStep> synthesizedSteps;
  std::vector<PlanStepEdge> synthesizedEdges;
  const bool haveSteps = !plan.steps.empty();
  if (!haveSteps) {
    PlanStepId next = 0;
    std::map<WorkloadNodeId, PlanStepId> computeStep;
    std::vector<WorkloadNodeId> nodes;
    for (const PlanPlacement &placement : plan.placements)
      nodes.push_back(placement.node);
    llvm::sort(nodes);
    nodes.erase(std::unique(nodes.begin(), nodes.end()), nodes.end());
    for (WorkloadNodeId node : nodes) {
      computeStep[node] = next;
      synthesizedSteps.push_back(
          PlanStep{next++, PlanStepKind::Compute, node, 0});
    }
    std::vector<const PlanConnection *> connections;
    for (const PlanConnection &connection : plan.connectionPlans)
      connections.push_back(&connection);
    llvm::sort(connections,
               [](const PlanConnection *lhs, const PlanConnection *rhs) {
                 return lhs->id < rhs->id;
               });
    for (const PlanConnection *connection : connections) {
      // A `Direct` connection materializes nothing, so it gets no step.
      if (connection->kind == ConnectionKind::Direct)
        continue;
      PlanStepId movement = next++;
      synthesizedSteps.push_back(
          PlanStep{movement, PlanStepKind::Movement, 0, connection->id});
      PlanStepId sync = next++;
      synthesizedSteps.push_back(
          PlanStep{sync, PlanStepKind::Synchronization, 0, connection->id});
      if (connection->producerPort) {
        auto producer = computeStep.find(connection->producerPort->node);
        if (producer != computeStep.end())
          synthesizedEdges.push_back(PlanStepEdge{producer->second, movement});
      }
      synthesizedEdges.push_back(PlanStepEdge{movement, sync});
      for (InstanceId consumer : connection->consumers) {
        auto node = nodeForInstance.find(consumer);
        if (node == nodeForInstance.end())
          continue;
        auto compute = computeStep.find(node->second);
        if (compute != computeStep.end())
          synthesizedEdges.push_back(PlanStepEdge{sync, compute->second});
      }
    }
  }
  const std::vector<PlanStep> &sourceSteps =
      haveSteps ? plan.steps : synthesizedSteps;
  const std::vector<PlanStepEdge> &sourceEdges =
      haveSteps ? plan.stepEdges : synthesizedEdges;

  // The steps sorted by id: the storage plan builds them in a deterministic
  // topological order, so their ids are the execution order.
  std::vector<const PlanStep *> steps;
  steps.reserve(sourceSteps.size());
  for (const PlanStep &step : sourceSteps)
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

  // --- storage facts (issue #129, task R5) ----------------------------------
  //
  // The allocations holding each value, in id order: a value can live in more
  // than one memory at once (a producer's own buffer, a movement stage's copy,
  // and one borrow per memory a boundary descriptor resolved to), and the
  // occurrence's own resolved memory is what picks between them. The lowest-id
  // match is the fallback when the memory cannot be resolved.

  // The value whose storage `value` occupies: a carried loop value is the value
  // the loop carries it from, followed transitively. Without a graph the value
  // resolves to itself.
  auto storageValueOf = [&](WorkloadValueId value) -> WorkloadValueId {
    if (!graph)
      return value;
    for (unsigned guard = 0; guard < 64; ++guard) {
      const WorkloadValue *entry = graph->findValue(value);
      if (!entry || !entry->carriedFrom)
        return value;
      WorkloadValueId next = *entry->carriedFrom;
      if (next == value)
        return value;
      value = next;
    }
    return value;
  };

  // One event's storage uses: every operand it reads and every result it
  // writes, expanded once per simultaneously-live occurrence of that slot, so
  // a consumer of a spatial loop's product sees the residency a serial loop
  // would not have. Each occurrence is attributed to the allocation that holds
  // its value *in the memory the occurrence resolved to*, so two same-kind
  // nodes (or a producer's buffer and a staged copy) are never confused.
  // `occurrenceBase` keeps the ids of two events distinct.
  auto storageUsesFor =
      [&](const PlanPlacement &placed, WorkloadNodeId node,
          uint64_t occurrenceBase) -> llvm::Expected<std::vector<StorageUse>> {
    std::vector<StorageUse> uses;
    if (!graph)
      return uses;
    const WorkloadNode *workloadNode = graph->findNode(node);
    if (!workloadNode)
      return uses;
    llvm::Error expansionError = llvm::Error::success();
    auto emit = [&](const WorkloadPort &port, PortDirection direction,
                    unsigned index, StorageAccess access) {
      const WorkloadValueId value = storageValueOf(port.value);
      std::optional<std::string> memory;
      if (llvm::Expected<EndpointMemory> resolved = resolveEndpointMemory(
              *graph, placed, PortRef{node, direction, index}, machine))
        memory = resolved->memory;
      else
        llvm::consumeError(resolved.takeError());
      const StorageAllocation *chosen = nullptr;
      for (const StorageAllocation &allocation : plan.allocations) {
        if (allocation.value != value)
          continue;
        if (!chosen)
          chosen = &allocation;
        if (memory && allocation.memory == *memory) {
          chosen = &allocation;
          break;
        }
      }
      if (!chosen)
        return;
      // One use per simultaneously resident occurrence, so liveness can count
      // them: an occurrence is what the event stream spells out. A residency
      // beyond the enumeration bound is an explicit incomplete fact rather
      // than a truncated stream (issue #129, task R5).
      if (chosen->simultaneousOccurrences > kMaxEnumeratedOccurrences) {
        if (expansionError)
          return;
        expansionError = planEventError(
            "plan events: allocation " + llvm::Twine(chosen->id) +
            " is resident " + llvm::Twine(chosen->simultaneousOccurrences) +
            " times, more than the " + llvm::Twine(kMaxEnumeratedOccurrences) +
            " occurrences the event stream enumerates");
        return;
      }
      for (uint64_t step = 0; step < chosen->simultaneousOccurrences; ++step)
        uses.push_back(StorageUse{chosen->id, occurrenceBase + step, access});
    };
    // Reads first, writes second, each in port order: deterministic and
    // independent of the allocation layout.
    for (unsigned index = 0; index < workloadNode->inputs.size(); ++index)
      emit(workloadNode->inputs[index], PortDirection::Input, index,
           StorageAccess::Read);
    for (unsigned index = 0; index < workloadNode->outputs.size(); ++index)
      emit(workloadNode->outputs[index], PortDirection::Output, index,
           StorageAccess::Write);
    if (expansionError)
      return std::move(expansionError);
    return uses;
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
      // The selected engine comes from the placement's *recorded* compute
      // binding when it has one; only a placement that recorded nothing to
      // select -- a rule requiring no compute capability, or a hand-built plan
      // -- falls back to the executor's first attached engine (issue #129, task
      // R1). A required kind with no recorded node, or a recorded node that no
      // longer resolves, is an error, never a silent re-derivation.
      llvm::Expected<const machine::ComputeNode *> recorded =
          recordedEngineForPlacement(machine, placed);
      if (!recorded)
        return recorded.takeError();
      const machine::ComputeNode *engine = *recorded;
      if (!engine)
        engine = engineForExecutor(machine, placed.executor);
      if (!engine)
        return planEventError("plan events: machine '" + machine.target +
                              "' declares no compute resource for node " +
                              llvm::Twine(step->node));
      // A compute event accounts for work, not traffic: the performance DAG's
      // vector/mma events carry no bytes, so this one carries none either and
      // the two streams stay comparable.
      //
      // The cycle input here is the selected instance's rule-local estimate.
      // That is the *accumulation fallback*'s value: a plan scored before its
      // kernel was bound has no operation kind (mma vs vector) to pick the
      // machine's formula from, so it can only charge the rule's declared cost.
      // The `AnalysisSnapshotEvents` path is what replaces it (issue #129, task
      // R6): once the shared selected-kernel analysis has extracted the bound
      // kernel, `buildPlanEvents` returns that stream and this estimate is
      // never what a final plan is scored on. The two consumers therefore agree
      // on every field of a finalized plan, and this estimate only ever
      // describes a stream explicitly labelled `PlanEventSource::Accumulation`.
      PlanCostEvent compute = makePlanCostEvent(
          CostEventKind::Compute, engine->id, placed.cost.latencyCycles,
          placed.workItems, /*bytes=*/0);
      compute.owner = placed.executor;
      compute.sourceNode = std::to_string(step->node);
      compute.planStep = step->id;
      // One occurrence namespace per node, so two steps' uses never collide.
      compute.occurrence =
          (static_cast<uint64_t>(step->node) << 32) | uint64_t{1};
      llvm::Expected<std::vector<StorageUse>> uses =
          storageUsesFor(placed, step->node, compute.occurrence);
      if (!uses)
        return uses.takeError();
      compute.storageUses = std::move(*uses);
      produced.push_back(add(std::move(compute)));
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

      // A conversion kind with no maps has no measurable footprint: rejected,
      // never silently emitted as a free event.
      if ((connection.kind == ConnectionKind::LayoutTransform ||
           connection.kind == ConnectionKind::TransferAndTransform) &&
          !connection.transform)
        return planEventError("plan events: transform connection " +
                              llvm::Twine(connection.id) +
                              " carries no maps to emit");

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
          PlanCostEvent transferEvent = makePlanCostEvent(
              CostEventKind::TransferHop, resource, static_cast<double>(cycles),
              workItems, bytes, deps);
          transferEvent.srcMemory = from->id;
          transferEvent.dstMemory = to->id;
          transferEvent.connectionId = connection.id;
          transferEvent.hopIndex = hop - 1;
          transferEvent.planStep = step->id;
          // The storage the hop reads and writes, when the plan recorded the
          // hop: a reader of the source slot and a writer of the destination
          // one. Each side is spelled out once per simultaneously resident
          // occurrence of the slot it names, exactly as a compute event's uses
          // are, so liveness counts one residency from the stream instead of
          // trusting a field (issue #129, task R5).
          const uint64_t occurrenceBase =
              (static_cast<uint64_t>(connection.id) << 32) | uint64_t{1};
          if (hop - 1 < connection.hops.size()) {
            const PlanMovementHop &recorded = connection.hops[hop - 1];
            auto hopUses = [&](uint64_t storageId,
                               StorageAccess access) -> llvm::Error {
              if (storageId == 0)
                return llvm::Error::success();
              uint64_t occurrences = 1;
              for (const StorageAllocation &allocation : plan.allocations)
                if (allocation.id == storageId) {
                  occurrences = allocation.simultaneousOccurrences;
                  break;
                }
              if (occurrences > kMaxEnumeratedOccurrences)
                return planEventError(
                    "plan events: allocation " + llvm::Twine(storageId) +
                    " is resident " + llvm::Twine(occurrences) +
                    " times, more than the " +
                    llvm::Twine(kMaxEnumeratedOccurrences) +
                    " occurrences the event stream enumerates");
              for (uint64_t index = 0; index < occurrences; ++index)
                transferEvent.storageUses.push_back(
                    StorageUse{storageId, occurrenceBase + index, access});
              return llvm::Error::success();
            };
            if (llvm::Error error =
                    hopUses(recorded.sourceStorageId, StorageAccess::Read))
              return std::move(error);
            if (llvm::Error error = hopUses(recorded.destinationStorageId,
                                            StorageAccess::Write))
              return std::move(error);
          }
          uint32_t transfer = add(std::move(transferEvent));
          hops.push_back(transfer);
          PlanCostEvent waitEvent = makePlanCostEvent(
              CostEventKind::Synchronization, "sync",
              static_cast<double>(machine.sync.waitCycles), 0, 0, {transfer});
          waitEvent.connectionId = connection.id;
          waitEvent.hopIndex = hop - 1;
          waitEvent.planStep = step->id;
          uint32_t wait = add(std::move(waitEvent));
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
        // A gather's combination is compute on the engine the *serving
        // placement* recorded, not on whatever the executor happens to list
        // first: the same selection rule the compute and transform events use
        // (issue #129, task R1).
        const machine::ComputeNode *engine = nullptr;
        if (const PlanPlacement *consumer =
                consumerPlacementOf(plan, connection)) {
          llvm::Expected<const machine::ComputeNode *> recorded =
              recordedEngineForPlacement(machine, *consumer);
          if (!recorded)
            return recorded.takeError();
          engine = *recorded;
        }
        if (!engine)
          engine = engineForExecutor(machine,
                                     transformExecutorFor(plan, connection));
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
        // The conversion runs on the resource the plan *recorded* for it --
        // `selectTransformResource` picked it by memory visibility, which can
        // disagree with the executor's declaration order. Only a connection
        // that recorded none falls back (issue #129, task R1).
        llvm::Expected<const machine::ComputeNode *> recorded =
            transformEngineFor(connection, machine, plan);
        if (!recorded)
          return recorded.takeError();
        const machine::ComputeNode *engine = *recorded;
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
      PlanCostEvent barrierEvent = makePlanCostEvent(
          CostEventKind::Synchronization, "sync",
          static_cast<double>(machine.sync.barrierCycles), 0, 0);
      barrierEvent.connectionId = step->connection;
      produced.push_back(add(std::move(barrierEvent)));
      break;
    }
    }
    stepEvents[step->id] = std::move(produced);
  }

  // Inter-step dependencies: every event of a dependent step follows the last
  // event of each step it depends on. A step cannot follow itself: a self-step
  // edge -- the reuse edge an in-place update contributes -- states which
  // single buffer two allocations share, and the step's own events are already
  // ordered by their emission sequence, so imposing it would only make an event
  // depend on itself (issue #129, task R5).
  for (const PlanStepEdge &edge : sourceEdges) {
    if (edge.from == edge.to)
      continue;
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
    for (const PlanStep &step : sourceSteps) {
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

llvm::Error attachPlanAnalysisEvents(CoveringPlan &plan, PlanEventDAG events,
                                     const machine::MachineModel &machine) {
  // A plan decoded from persisted metadata is a replay: its derived execution
  // facts were deliberately not persisted, so a snapshot handed to it would be
  // describing a kernel it has not re-derived. Reject rather than attach one
  // that a caller assumed was trustworthy.
  if (plan.schemaVersion != 0)
    return planEventError(
        "plan events: refusing to attach an analysis snapshot to a plan "
        "decoded from metadata; a replay must re-derive its events");

  events.source = PlanEventSource::Snapshot;
  if (llvm::Error error = verifyPlanAnalysisStream(plan, events, machine))
    return error;
  plan.analysisEvents = std::move(events);
  return llvm::Error::success();
}

ConnectionSignature
connectionSignatureFor(const ConnectionPlan &connection,
                       const WorkloadGraph &workload,
                       const machine::MachineModel &machine) {
  auto renderType = [](mlir::Type type) {
    if (!type)
      return std::string();
    std::string text;
    llvm::raw_string_ostream stream(text);
    type.print(stream);
    return stream.str();
  };
  auto renderMap = [](const mlir::AffineMap &map) {
    std::string text;
    llvm::raw_string_ostream stream(text);
    map.print(stream);
    return stream.str();
  };

  ConnectionSignature signature;
  signature.kind = stringifyConnectionKind(connection.kind).str();
  if (const WorkloadValue *value = workload.findValue(connection.value))
    signature.valueType = renderType(value->type);

  // Endpoint roles: a swapped producer/consumer assignment is different work.
  if (connection.producerPort)
    signature.producerEndpoint =
        "producer{" + canonicalPortRefString(*connection.producerPort) + "}";
  std::vector<std::string> consumers;
  for (const PortRef &port : connection.consumerPorts)
    consumers.push_back("consumer{" + canonicalPortRefString(port) + "}");
  llvm::sort(consumers);
  signature.consumerEndpoints = llvm::join(consumers, ",");

  // The ordered route and the links it crosses.
  std::vector<std::string> nodes(connection.memoryRoute.begin(),
                                 connection.memoryRoute.end());
  std::vector<std::string> links;
  for (size_t hop = 1; hop < connection.memoryRoute.size(); ++hop)
    for (const machine::LinkEdge &edge : machine.links)
      if (edge.source == connection.memoryRoute[hop - 1] &&
          edge.destination == connection.memoryRoute[hop]) {
        links.push_back(edge.id);
        break;
      }
  signature.route = llvm::join(nodes, ">");
  signature.links = llvm::join(links, ">");
  signature.engines = llvm::join(connection.transferEngines, ">");

  // The concrete affine relations on both sides, plus the transform's maps.
  std::string maps = "producer=";
  maps += connection.producerMap ? renderMap(*connection.producerMap)
                                 : std::string("<null>");
  std::vector<std::string> consumerMaps;
  for (const mlir::AffineMap &map : connection.consumerMaps)
    consumerMaps.push_back(renderMap(map));
  llvm::sort(consumerMaps);
  maps += ";consumers=";
  maps += llvm::join(consumerMaps, ",");
  if (connection.transform) {
    maps += ";transform_src=";
    maps += connection.transform->srcMap
                ? renderMap(connection.transform->srcMap)
                : std::string("<null>");
    maps += ";transform_dst=";
    maps += connection.transform->dstMap
                ? renderMap(connection.transform->dstMap)
                : std::string("<null>");
  }
  signature.maps = maps;

  // The concrete parameters: the transform's layout families *and the concrete
  // compute resource it runs on* (issue #129, task R1 -- a transform on two
  // different engines is different work), plus, for a gather, its declared
  // semantics and axis (execution-affecting content the canonical connection
  // string already folds).
  //
  // Every component is length-delimited (`<name>=<len>:<bytes>`), so a layout
  // id or resource id that itself contains the `;` or `=` separator cannot be
  // mistaken for a component boundary and collide two different connections on
  // one key.
  std::string parameters;
  auto component = [&](llvm::StringRef name, llvm::StringRef value) {
    if (!parameters.empty())
      parameters += ';';
    parameters += name.str();
    parameters += '=';
    parameters += std::to_string(value.size());
    parameters += ':';
    parameters.append(value.begin(), value.end());
  };
  if (connection.transform) {
    component("src", connection.transform->srcLayout);
    component("dst", connection.transform->dstLayout);
    if (!connection.transform->computeResource.empty())
      component("compute", connection.transform->computeResource);
  }
  if (connection.gatherSemantics) {
    component("gather", stringifyGatherSemantics(*connection.gatherSemantics));
    if (connection.concatAxis)
      component("axis", std::to_string(*connection.concatAxis));
  }
  signature.parameters = parameters;

  std::string storage;
  if (!connection.memoryRoute.empty())
    storage = connection.memoryRoute.back();
  storage += "#";
  storage += std::to_string(connection.cost.localBytes);
  signature.storage = storage;
  return signature;
}

} // namespace mlir::llk::mapping
