//===- Placement.cpp - Placement and connection synthesis (D5) -----------===//

#include "LLK/Mapping/Placement.h"

#include "mlir/IR/BuiltinTypes.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/Error.h"

#include <memory>
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

/// The element type a port type exposes for §10.2 comparison, or nullopt when
/// this target-independent core cannot read one. A modelled shaped type
/// (`tensor`, `memref`, `vector`) states its element type; a bare float or
/// integer type *is* an element type. A `!micro.tile` is opaque here -- the
/// core never names the Micro dialect -- so it yields nullopt rather than a
/// guessed element type.
std::optional<mlir::Type> comparableElementType(mlir::Type type) {
  if (!type)
    return std::nullopt;
  if (auto shaped = mlir::dyn_cast<mlir::ShapedType>(type))
    return shaped.getElementType();
  if (mlir::isa<mlir::FloatType, mlir::IntegerType, mlir::IndexType>(type))
    return type;
  return std::nullopt;
}

/// The static logical shape a port type exposes, or nullopt for a dynamic,
/// unranked, or opaque type.
std::optional<llvm::SmallVector<int64_t, 4>> staticShape(mlir::Type type) {
  if (!type)
    return std::nullopt;
  if (auto shaped = mlir::dyn_cast<mlir::ShapedType>(type))
    if (shaped.hasStaticShape())
      return llvm::SmallVector<int64_t, 4>(shaped.getShape());
  return std::nullopt;
}

/// §10.2: a changed element type or logical tile shape admits no alternative --
/// neither a layout transform nor a transfer rewrites it -- so it rejects the
/// pair outright. Only a fact both ends state is compared.
bool elementAndShapeCompatible(const ConnectionRequest &request) {
  std::optional<mlir::Type> producer =
      comparableElementType(request.elementType);
  std::optional<mlir::Type> consumer =
      comparableElementType(request.consumerType);
  if (producer && consumer && *producer != *consumer)
    return false;

  std::optional<llvm::SmallVector<int64_t, 4>> producerShape =
      staticShape(request.elementType);
  std::optional<llvm::SmallVector<int64_t, 4>> consumerShape =
      staticShape(request.consumerType);
  if (producerShape && consumerShape && *producerShape != *consumerShape)
    return false;
  return true;
}

/// §10.2: an index relation both ends state must match. When only one end
/// states one there is nothing to prove a mismatch from, so the pair is left
/// alone. Comparison goes through `simplifyAffineMap` and MLIR's own equality,
/// never a rendering (design §10.2 forbids ad hoc map comparison).
bool affineCompatible(const ConnectionRequest &request) {
  if (!request.producerMap || !request.consumerMap)
    return true;
  return mlir::simplifyAffineMap(*request.producerMap) ==
         mlir::simplifyAffineMap(*request.consumerMap);
}

/// The consumer can read the producer's memory only when its executor can
/// address that memory (design §11.2 `dominates`). An unstated executor is not
/// checkable, so it never rejects.
bool consumerSeesProducerMemory(const ConnectionRequest &request,
                                const MachineModel &machine) {
  if (!request.consumerExecutor)
    return true;
  return machine.isVisible(request.producerMemory, *request.consumerExecutor);
}

} // namespace

bool portsDirectCompatible(const ConnectionRequest &request,
                           const machine::MachineModel &machine) {
  if (!elementAndShapeCompatible(request))
    return false;
  // Different layouts are a transform, not a direct connection.
  if (request.producerLayout && request.consumerLayout &&
      *request.producerLayout != *request.consumerLayout)
    return false;
  if (!affineCompatible(request))
    return false;
  return consumerSeesProducerMemory(request, machine);
}

llvm::Expected<std::vector<CandidateInstance>>
enumeratePlacements(const MappingCandidate &candidate,
                    const MappingTarget &target, mlir::MLIRContext &context,
                    const LayoutContext &layoutContext,
                    const PlacementOptions &options, bool *truncated,
                    PlacementFailure *failure) {
  const MachineModel &machine = target.machine();

  auto reportFailure = [&](PlacementFailure reason) {
    if (failure)
      *failure = reason;
  };

  // A layout requirement that cannot solve makes the candidate unplaceable,
  // independently of which executor would run it. Solved through the
  // `LayoutSolver` interface (design §13.3), not the free function, so a future
  // backend drops in here without touching placement.
  //
  // Every requirement must solve before any executor is tried; the resolved
  // definition is kept so an instance can bind it. `layoutBindings` is a
  // `StringMap<LayoutId>`, so it carries the layout *definition* id only: that
  // is the registry-resolved id of the definition the requirement names, and is
  // therefore string-identical to `requirement.layoutClass` by construction. A
  // solve's concrete parameter assignment and affine map are not surfaced --
  // there is no field on `CandidateInstance` for them, and a solve may report
  // several, none of which placement selects (selection is the tuner's job).
  // Recording solved parameters would be a data-model change, not a binding
  // tweak.
  std::unique_ptr<LayoutSolver> layoutSolver = makeBoundedLayoutSolver();
  std::vector<const LayoutDef *> solvedDefs;
  solvedDefs.reserve(candidate.layoutRequirements.size());
  for (const LayoutRequirement &requirement : candidate.layoutRequirements) {
    const LayoutDef *def = target.layouts().find(requirement.layoutClass);
    if (!def)
      return placementError("placement: unknown layout '" +
                            requirement.layoutClass + "'");
    llvm::Expected<LayoutSolveResult> solved =
        layoutSolver->solve(*def, machine, context, layoutContext);
    if (!solved)
      return solved.takeError();
    if (solved->undecided || solved->solutions.empty()) {
      // A solve whose quantifier ran out of budget is *undecided*: an exhausted
      // quantifier yields 0, and `!undecided` reads as satisfied, so any
      // solution it reported could be illegal. Fail closed -- the candidate is
      // not legally placeable -- rather than accept an undecided layout. The
      // solve withholds those solutions too, so this guard is belt-and-braces.
      // (`truncated` alone is *not* a failure: it also means "more solutions
      // may exist", which is not a soundness problem.)
      reportFailure(PlacementFailure::NoLegalLayout);
      return std::vector<CandidateInstance>{};
    }
    solvedDefs.push_back(def);
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
  if (executors.empty())
    reportFailure(PlacementFailure::NoLegalExecutor);

  // The first failing executor in machine order decides the reported reason,
  // so the code does not depend on which executor happened to be visited last.
  bool executorReasonRecorded = false;
  bool instanceCapReached = false;
  std::vector<CandidateInstance> instances;
  for (const ExecutorNode *executor : executors) {
    // §15.1 step 2: enumerate every compatible compute attachment -- each
    // attached capability of each required kind, in machine declaration order.
    // Every one is a legal alternative, so all of them are kept.
    std::vector<std::vector<const ComputeNode *>> computeChoices;
    bool computesOk = true;
    for (const ComputeRequirement &requirement :
         candidate.computeRequirements) {
      std::vector<const ComputeNode *> matches;
      for (const ComputeNode *node : machine.computesFor(executor->id))
        if (node->kind == requirement.kind)
          matches.push_back(node);
      if (matches.empty()) {
        computesOk = false;
        break;
      }
      computeChoices.push_back(std::move(matches));
    }
    if (!computesOk) {
      if (!executorReasonRecorded) {
        reportFailure(PlacementFailure::UnsupportedComputeFragment);
        executorReasonRecorded = true;
      }
      continue;
    }

    // Memory attachments: every visible memory of each required kind, likewise
    // in machine declaration order.
    std::vector<std::vector<const MemoryNode *>> memoryChoices;
    bool memoriesOk = true;
    for (const MemoryRequirement &requirement : candidate.memoryRequirements) {
      std::vector<const MemoryNode *> matches;
      for (const MemoryNode &node : machine.memories)
        if (node.kind == requirement.kind &&
            machine.isVisible(node.id, executor->id))
          matches.push_back(&node);
      if (matches.empty()) {
        memoriesOk = false;
        break;
      }
      memoryChoices.push_back(std::move(matches));
    }
    if (!memoriesOk) {
      if (!executorReasonRecorded) {
        reportFailure(PlacementFailure::NoLegalMemory);
        executorReasonRecorded = true;
      }
      continue;
    }

    // The legal placements of this executor are the cartesian product of the
    // per-requirement choices. An odometer visits them in machine-declaration
    // order with the last requirement varying fastest; with no requirements it
    // yields the single unadorned placement.
    std::vector<size_t> sizes;
    sizes.reserve(computeChoices.size() + memoryChoices.size());
    for (const auto &choices : computeChoices)
      sizes.push_back(choices.size());
    for (const auto &choices : memoryChoices)
      sizes.push_back(choices.size());

    std::vector<size_t> pick(sizes.size(), 0);
    auto advance = [&]() {
      for (size_t dimension = sizes.size(); dimension-- > 0;) {
        if (++pick[dimension] < sizes[dimension])
          return true;
        pick[dimension] = 0; // wrapped: carry into the next dimension
      }
      return false; // every dimension wrapped: the product is exhausted
    };

    do {
      CandidateInstance instance;
      instance.candidate = candidate.id;
      instance.bundle = candidate.bundle;
      instance.executorBindings["executor"] = executor->id;
      for (size_t i = 0; i < computeChoices.size(); ++i)
        instance.computeBindings[candidate.computeRequirements[i].kind] =
            computeChoices[i][pick[i]]->id;
      for (size_t j = 0; j < memoryChoices.size(); ++j)
        instance.memoryBindings[candidate.memoryRequirements[j].kind] =
            memoryChoices[j][pick[computeChoices.size() + j]]->id;
      for (size_t i = 0; i < solvedDefs.size(); ++i)
        instance.layoutBindings[candidate.layoutRequirements[i].layoutClass] =
            solvedDefs[i]->id;

      instance.resourceUsage.executorSlots = 1;
      for (const MemoryRequirement &requirement : candidate.memoryRequirements)
        instance.resourceUsage.memoryBytes[requirement.kind] =
            requirement.minBytes;

      instance.localCost = candidate.lowerBound;
      // Compute utilization is the rule-local compute cycles over the cycles
      // the machine's workers had available in one sync period (design §17.2).
      // Left 0 when the machine models no sync period: a missing denominator is
      // not a fabricated one.
      if (std::optional<double> utilization = utilizationEstimate(
              instance.localCost.latencyCycles, machine, machine.workerThreads))
        instance.localCost.computeUtilization = *utilization;
      instance.id = computeInstanceId(instance);
      instances.push_back(std::move(instance));
      if (instances.size() >= options.maxInstances) {
        // Conservative: also fires when the candidate has exactly maxInstances
        // legal placements. A caller may not claim optimality after touching
        // the cap (design §16.2).
        if (truncated)
          *truncated = true;
        instanceCapReached = true;
        break;
      }
    } while (advance());

    if (instanceCapReached)
      break;
  }
  // Instances produced means the candidate placed; clear any reason recorded
  // while some other executor was tried and skipped.
  if (!instances.empty())
    reportFailure(PlacementFailure::None);
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

  // §10.2: an element type or index relation the two ends disagree on admits no
  // alternative -- no layout transform or transfer rewrites it -- so the pair
  // is rejected outright rather than given a plan.
  if (!elementAndShapeCompatible(request) || !affineCompatible(request))
    return std::vector<ConnectionPlan>{};

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
    // Nothing to move: either a direct connection or an in-place transform. A
    // consumer that cannot address the producer's memory cannot read the value
    // from it, and with both ends in one memory there is no transfer
    // alternative, so the pair is incompatible (design §10.2).
    if (!consumerSeesProducerMemory(request, machine))
      return std::vector<ConnectionPlan>{};

    ConnectionPlan plan;
    plan.producer = request.producer;
    plan.consumers.push_back(request.consumer);
    plan.value = request.value;
    plan.kind = portsDirectCompatible(request, machine)
                    ? ConnectionKind::Direct
                    : ConnectionKind::LayoutTransform;
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
  route.producerExecutor = request.producerExecutor;
  route.consumerExecutor = request.consumerExecutor;
  // The value leaves the producer in its layout, so that is the layout the hop
  // memories must support; the consumer's (possibly different) layout is
  // reached by the transform a later connection materializes. Threaded only
  // when the producer states it.
  if (request.producerLayout)
    route.layoutClass = *request.producerLayout;
  // liveBytesOnIntermediate stays unset: this layer keeps no occupancy state,
  // so it has no live-byte figure to supply rather than a zero that would
  // silently assert the intermediates are empty.
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
  if (allShareProducerMemory && !consumers.empty() &&
      portsDirectCompatible(base, machine)) {
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
