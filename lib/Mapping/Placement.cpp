//===- Placement.cpp - Placement and connection synthesis (D5) -----------===//

#include "LLK/Mapping/Placement.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/Error.h"

#include <string>
#include <utility>
#include <vector>

namespace mlir::llk::mapping {

namespace {

using machine::ComputeNode;
using machine::ExecutorNode;
using machine::MachineModel;
using machine::MemoryNode;

llvm::Error placementError(llvm::StringRef message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(), message);
}

/// Compute node ids attached to `executor`, sorted for comparison.
std::vector<std::string> attachedComputes(const MachineModel &machine,
                                          const ExecutorNode &executor) {
  std::vector<std::string> ids;
  for (const ComputeNode *node : machine.computesFor(executor.id))
    ids.push_back(node->id);
  llvm::sort(ids);
  return ids;
}

/// Memory node ids visible from `executor`, sorted for comparison.
std::vector<std::string> visibleMemories(const MachineModel &machine,
                                         const ExecutorNode &executor) {
  std::vector<std::string> ids;
  for (const MemoryNode &node : machine.memories)
    if (machine.isVisible(node.id, executor.id))
      ids.push_back(node.id);
  llvm::sort(ids);
  return ids;
}

/// Two executors are interchangeable when swapping them cannot change any
/// binding: same kind, same parent, and exactly the same attached compute and
/// visible memory nodes. Symmetry reduction is only sound under this rule.
bool interchangeable(const MachineModel &machine, const ExecutorNode &lhs,
                     const ExecutorNode &rhs) {
  return lhs.kind == rhs.kind && lhs.parent == rhs.parent &&
         attachedComputes(machine, lhs) == attachedComputes(machine, rhs) &&
         visibleMemories(machine, lhs) == visibleMemories(machine, rhs);
}

std::vector<const ExecutorNode *>
reduceSymmetric(const MachineModel &machine,
                const std::vector<const ExecutorNode *> &executors) {
  std::vector<const ExecutorNode *> representatives;
  for (const ExecutorNode *candidate : executors) {
    bool duplicate = false;
    for (const ExecutorNode *kept : representatives) {
      if (interchangeable(machine, *kept, *candidate)) {
        duplicate = true;
        break;
      }
    }
    if (!duplicate)
      representatives.push_back(candidate);
  }
  return representatives;
}

} // namespace

llvm::Expected<std::vector<CandidateInstance>>
enumeratePlacements(const MappingCandidate &candidate,
                    const MappingTarget &target, mlir::MLIRContext &context,
                    const LayoutContext &layoutContext,
                    const PlacementOptions &options, bool *truncated) {
  const MachineModel &machine = target.machine();

  // A layout requirement that cannot solve makes the candidate unplaceable,
  // independently of which executor would run it.
  for (const LayoutRequirement &requirement : candidate.layoutRequirements) {
    const LayoutDef *def = target.layouts().find(requirement.layoutClass);
    if (!def)
      return placementError("placement: unknown layout '" +
                            requirement.layoutClass + "'");
    llvm::Expected<LayoutSolveResult> solved =
        solveLayout(*def, machine, context, layoutContext);
    if (!solved)
      return solved.takeError();
    if (solved->solutions.empty())
      return std::vector<CandidateInstance>{};
  }

  std::vector<const ExecutorNode *> executors;
  for (const ExecutorNode &executor : machine.executors) {
    bool matches = true;
    for (const ExecutorRequirement &requirement :
         candidate.executorRequirements) {
      if (!machine.ownerMatches(requirement.capability, executor.id)) {
        matches = false;
        break;
      }
    }
    if (matches)
      executors.push_back(&executor);
  }
  if (options.reduceSymmetry)
    executors = reduceSymmetric(machine, executors);

  std::vector<CandidateInstance> instances;
  for (const ExecutorNode *executor : executors) {
    CandidateInstance instance;
    instance.candidate = candidate.id;
    instance.executorBindings["executor"] = executor->id;

    // Compute attachments: the first attached capability of each required kind.
    bool computesOk = true;
    for (const ComputeRequirement &requirement :
         candidate.computeRequirements) {
      const ComputeNode *match = nullptr;
      for (const ComputeNode *node : machine.computesFor(executor->id)) {
        if (node->kind == requirement.kind) {
          match = node;
          break;
        }
      }
      if (!match) {
        computesOk = false;
        break;
      }
      instance.computeBindings[requirement.kind] = match->id;
    }
    if (!computesOk)
      continue;

    // Memory attachments: the first visible memory of each required kind.
    bool memoriesOk = true;
    for (const MemoryRequirement &requirement : candidate.memoryRequirements) {
      const MemoryNode *match = nullptr;
      for (const MemoryNode &node : machine.memories) {
        if (node.kind == requirement.kind &&
            machine.isVisible(node.id, executor->id)) {
          match = &node;
          break;
        }
      }
      if (!match) {
        memoriesOk = false;
        break;
      }
      instance.memoryBindings[requirement.kind] = match->id;
    }
    if (!memoriesOk)
      continue;

    for (const LayoutRequirement &requirement : candidate.layoutRequirements)
      instance.layoutBindings[requirement.layoutClass] =
          requirement.layoutClass;

    instance.resourceUsage.executorSlots = 1;
    for (const MemoryRequirement &requirement : candidate.memoryRequirements)
      instance.resourceUsage.memoryBytes[requirement.kind] =
          requirement.minBytes;

    instance.localCost = candidate.lowerBound;
    instance.id = computeInstanceId(instance);
    instances.push_back(std::move(instance));
    if (instances.size() >= options.maxInstances) {
      // Conservative: also fires when the candidate has exactly maxInstances
      // legal placements. A caller may not claim optimality after touching the
      // cap (design §16.2).
      if (truncated)
        *truncated = true;
      break;
    }
  }
  return instances;
}

llvm::Expected<std::vector<ConnectionPlan>>
synthesizeConnections(const ConnectionRequest &request,
                      const MachineModel &machine,
                      const TopologyService &topology,
                      const PlacementOptions &options, bool *truncated) {
  if (!machine.findMemory(request.producerMemory))
    return placementError("connection: unknown memory '" +
                          request.producerMemory + "'");
  if (!machine.findMemory(request.consumerMemory))
    return placementError("connection: unknown memory '" +
                          request.consumerMemory + "'");
  if (request.bytes == 0)
    return placementError("connection: bytes must be positive");

  bool transformRequired = request.producerLayout && request.consumerLayout &&
                           *request.producerLayout != *request.consumerLayout;
  auto transformOf = [&]() -> std::optional<LayoutTransform> {
    if (!transformRequired)
      return std::nullopt;
    LayoutTransform transform;
    transform.srcLayout = *request.producerLayout;
    transform.dstLayout = *request.consumerLayout;
    return transform;
  };

  std::vector<ConnectionPlan> plans;
  if (request.producerMemory == request.consumerMemory) {
    // Nothing to move: either a direct connection or an in-place transform.
    ConnectionPlan plan;
    plan.producer = request.producer;
    plan.consumers.push_back(request.consumer);
    plan.value = request.value;
    plan.kind = transformRequired ? ConnectionKind::LayoutTransform
                                  : ConnectionKind::Direct;
    plan.memoryRoute.push_back(request.producerMemory);
    plan.transform = transformOf();
    plan.cost.localBytes = request.bytes;
    plan.id = computeConnectionId(plan);
    plans.push_back(std::move(plan));
    return plans;
  }

  RouteRequest route;
  route.source = request.producerMemory;
  route.destination = request.consumerMemory;
  route.bytes = request.bytes;
  route.alignmentBytes = request.alignmentBytes;
  llvm::Expected<llvm::SmallVector<MemoryRoute>> routes =
      topology.enumerateRoutes(route, options.maxRoutesPerConnection,
                               truncated);
  if (!routes) {
    // No route is a legal outcome: the pair is incompatible, not an error.
    llvm::consumeError(routes.takeError());
    return plans;
  }

  for (const MemoryRoute &memoryRoute : *routes) {
    ConnectionPlan plan;
    plan.producer = request.producer;
    plan.consumers.push_back(request.consumer);
    plan.value = request.value;
    plan.kind = transformRequired ? ConnectionKind::TransferAndTransform
                                  : ConnectionKind::Transfer;
    plan.memoryRoute = memoryRoute.nodes;
    plan.transferEngines = memoryRoute.transferEngines;
    plan.transform = transformOf();
    plan.cost = memoryRoute.cost;
    plan.id = computeConnectionId(plan);
    plans.push_back(std::move(plan));
  }
  return plans;
}

llvm::Expected<std::vector<ConnectionPlan>> synthesizeFanOut(
    const ConnectionRequest &base, llvm::ArrayRef<InstanceId> consumers,
    llvm::ArrayRef<MemoryNodeId> consumerMemories, const MachineModel &machine,
    const TopologyService &topology, const PlacementOptions &options) {
  if (consumers.size() != consumerMemories.size())
    return placementError(
        "fan-out: consumers and consumer memories must correspond");

  // Shared read: every consumer reads the memory the producer already wrote.
  bool allShareProducerMemory = true;
  for (const MemoryNodeId &memory : consumerMemories)
    if (memory != base.producerMemory)
      allShareProducerMemory = false;

  std::vector<ConnectionPlan> plans;
  if (allShareProducerMemory && !consumers.empty()) {
    ConnectionPlan plan;
    plan.producer = base.producer;
    plan.value = base.value;
    for (InstanceId consumer : consumers)
      plan.consumers.push_back(consumer);
    plan.kind = ConnectionKind::Direct;
    plan.memoryRoute.push_back(base.producerMemory);
    plan.cost.localBytes = base.bytes;
    plan.id = computeConnectionId(plan);
    plans.push_back(std::move(plan));
    return plans;
  }

  // Replication: each consumer gets its own connection.
  for (size_t index = 0; index < consumers.size(); ++index) {
    ConnectionRequest request = base;
    request.consumer = consumers[index];
    request.consumerMemory = consumerMemories[index];
    llvm::Expected<std::vector<ConnectionPlan>> replicated =
        synthesizeConnections(request, machine, topology, options);
    if (!replicated)
      return replicated.takeError();
    for (ConnectionPlan &plan : *replicated)
      plans.push_back(std::move(plan));
  }
  return plans;
}

ConnectionPlan synthesizeFanIn(llvm::ArrayRef<InstanceId> producers,
                               InstanceId consumer, WorkloadValueId value,
                               MemoryNodeId consumerMemory, uint64_t bytes) {
  ConnectionPlan plan;
  plan.kind = ConnectionKind::Reduce;
  plan.consumers.push_back(consumer);
  plan.producers.assign(producers.begin(), producers.end());
  plan.value = value;
  plan.memoryRoute.push_back(std::move(consumerMemory));
  plan.cost.localBytes = bytes;
  plan.id = computeConnectionId(plan);
  return plan;
}

} // namespace mlir::llk::mapping
