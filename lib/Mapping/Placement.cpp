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

/// True when the target declares `lhs` and `rhs` interchangeable (design §15.1:
/// "symmetric placements may be canonicalized when *the target declares*
/// executors equivalent"). A loaded model is guaranteed symmetric and same-kind
/// by `verifyMachineModel`; checking either direction keeps the rule robust for
/// a hand-built model, and a *valid* declaration read from YAML always holds
/// both ways. The kind-equality guard likewise duplicates a verified rule, but
/// `declaredEquivalent` is also reachable from a hand-built model that never
/// passed `verifyMachineModel` (a `MappingTarget` constructor does not verify),
/// and collapsing across kinds would bind the wrong executor class -- so the
/// check is repeated here rather than trusted.
bool declaredEquivalent(const ExecutorNode &lhs, const ExecutorNode &rhs) {
  if (lhs.kind != rhs.kind)
    return false;
  return llvm::is_contained(lhs.equivalentTo, rhs.id) ||
         llvm::is_contained(rhs.equivalentTo, lhs.id);
}

/// Two executors are interchangeable when swapping them cannot change a
/// binding, so symmetry reduction is sound. A target-declared equivalence
/// settles the question outright: it is the target's assertion, not a cost
/// proof -- the model cannot guarantee cost-invariance, and a declared group
/// may legitimately differ in concurrency -- so by design it collapses even
/// executors the structural heuristic would keep apart. With no declaration the
/// structural heuristic requires the same kind, parent, logical coordinates,
/// concurrency, and scheduling class, and exactly the same attached compute and
/// visible memory nodes. Coordinates, concurrency, and scheduling class are
/// compared because two executors that differ in any of them occupy a different
/// spatial/parallelism/overlap position -- collapsing such distinct-performance
/// executors into one representative (absent a declaration) would hide a
/// placement and misstate cost.
bool interchangeable(const MachineModel &machine, const ExecutorNode &lhs,
                     const ExecutorNode &rhs) {
  if (declaredEquivalent(lhs, rhs))
    return true;
  return lhs.kind == rhs.kind && lhs.parent == rhs.parent &&
         lhs.coordinates == rhs.coordinates &&
         lhs.concurrency == rhs.concurrency &&
         lhs.schedulingClass == rhs.schedulingClass &&
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

/// The hop a transform runs on (design §15.2 alternative 5, "transform placed
/// at a legal hop"). The value stays in the producer's layout up to
/// `nodes[hop]` and is in the consumer's layout from `nodes[hop + 1]` on, so
/// the transform is a property of one hop boundary rather than the whole plan.
/// The first legal hop is chosen, so the placement is deterministic. Nullopt
/// when no hop admits the transform -- the route cannot carry the value in
/// either layout regime and is not a legal alternative.
std::optional<size_t> legalTransformHop(llvm::ArrayRef<MemoryNodeId> nodes,
                                        const MachineModel &machine,
                                        llvm::StringRef producerLayout,
                                        llvm::StringRef consumerLayout) {
  if (nodes.size() < 2)
    return std::nullopt;
  for (size_t hop = 0; hop + 1 < nodes.size(); ++hop) {
    bool legal = true;
    // Up to and including the hop's source the value is stored in the layout
    // the producer wrote it in.
    for (size_t index = 0; index <= hop && legal; ++index) {
      const MemoryNode *memory = machine.findMemory(nodes[index]);
      legal = memory && memorySupportsLayout(*memory, producerLayout);
    }
    // From the hop's destination on it is stored in the consumer's layout.
    for (size_t index = hop + 1; index < nodes.size() && legal; ++index) {
      const MemoryNode *memory = machine.findMemory(nodes[index]);
      legal = memory && memorySupportsLayout(*memory, consumerLayout);
    }
    if (legal)
      return hop;
  }
  return std::nullopt;
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

  const bool transformRequired =
      request.producerLayout && request.consumerLayout &&
      *request.producerLayout != *request.consumerLayout;
  auto transformOf = [&]() -> std::optional<LayoutTransform> {
    if (!transformRequired)
      return std::nullopt;
    LayoutTransform transform;
    transform.srcLayout = *request.producerLayout;
    transform.dstLayout = *request.consumerLayout;
    return transform;
  };
  const MemoryNode *producerMemory = machine.findMemory(request.producerMemory);
  const bool sameMemory = request.producerMemory == request.consumerMemory;
  // Both endpoints must be able to reach the memory a layout-only transform
  // runs in. The producer wrote there; the check that matters is the
  // consumer's.
  const bool mutuallyVisible = consumerSeesProducerMemory(request, machine);

  // §15.2's alternatives are attempted in order, and every alternative that is
  // legal becomes its own plan so a caller can see all the shapes. The order of
  // `plans` is that attempt order.
  std::vector<ConnectionPlan> plans;

  // Alternative 1: a direct connection -- nothing is moved and nothing is
  // transformed. §10.2 defines direct compatibility by element type, logical
  // tile shape, memory *visibility*, and affine index relation, so the two
  // memories need not be the same node: a consumer that can address the
  // producer's memory reads the value there. That visibility check (and the
  // element-type, shape, and affine checks) is inside `portsDirectCompatible`,
  // so gating on it -- rather than on memory-id equality -- is what §10.2 asks.
  if (portsDirectCompatible(request, machine)) {
    ConnectionPlan plan;
    plan.producer = request.producer;
    plan.consumers.push_back(request.consumer);
    plan.value = request.value;
    plan.kind = ConnectionKind::Direct;
    plan.memoryRoute.push_back(request.producerMemory);
    plan.cost.localBytes = request.bytes;
    plan.id = computeConnectionId(plan);
    plans.push_back(std::move(plan));
  }

  // Alternative 2: a layout-only transform in a memory both endpoints can see.
  // The value already sits in the producer's memory, so the transform runs
  // there and the consumer reads the result -- there is no transfer. It is
  // emitted even when the two ends are *placed* on different memories, because
  // the consumer may legally read the producer's memory: `mutuallyVisible` is
  // exactly that `consumerSeesProducerMemory` check, the same visibility fact
  // alternative 1 relies on. The memory must be able to hold both layouts, or
  // the transform cannot occur in it. (Like the alt-5 transform hop, the plan
  // does not record that the consumer reads the producer's memory rather than
  // its own bound memory -- see the note on `ConnectionPlan` at the
  // chosen-hop site.)
  if (transformRequired && mutuallyVisible &&
      memorySupportsLayout(*producerMemory, *request.producerLayout) &&
      memorySupportsLayout(*producerMemory, *request.consumerLayout)) {
    ConnectionPlan plan;
    plan.producer = request.producer;
    plan.consumers.push_back(request.consumer);
    plan.value = request.value;
    plan.kind = ConnectionKind::LayoutTransform;
    plan.memoryRoute.push_back(request.producerMemory);
    plan.transform = transformOf();
    plan.cost.localBytes = request.bytes;
    plan.id = computeConnectionId(plan);
    plans.push_back(std::move(plan));
  }

  // Alternatives 3-5 are transfers over the topology. A route from a memory to
  // itself is not a movement, so a same-memory pair has no transfer to make and
  // the alternatives above stand alone. (Falling through rather than returning
  // early keeps a same-memory pair whose consumer cannot read the memory
  // incompatible without short-circuiting before the transfer path.)
  if (sameMemory)
    return plans;

  RouteRequest route;
  route.source = request.producerMemory;
  route.destination = request.consumerMemory;
  route.bytes = request.bytes;
  route.alignmentBytes = request.alignmentBytes;
  route.producerExecutor = request.producerExecutor;
  route.consumerExecutor = request.consumerExecutor;
  // A value that keeps one layout is carried in that layout over every hop, so
  // the route's every memory must support it. A transform changes the layout,
  // so its legality is *not* a plan-wide fact: rather than asking the router to
  // hold one layout on every hop, the transform's hop is validated per route
  // with `legalTransformHop` below, and the route is enumerated with no layout
  // filter. Threaded only when the producer states a layout.
  if (!transformRequired && request.producerLayout)
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

  // §15.2 orders a direct transfer (alternative 3/4) before a bounded
  // multi-hop transfer (alternative 5). D2 ranks routes cheapest-first, which
  // can interleave the two, so single-hop routes are emitted first; within each
  // group D2's cost-ranked order is preserved.
  std::vector<const MemoryRoute *> directRoutes;
  std::vector<const MemoryRoute *> multiHopRoutes;
  for (const MemoryRoute &memoryRoute : *routes)
    (memoryRoute.hopCount() == 1 ? directRoutes : multiHopRoutes)
        .push_back(&memoryRoute);

  auto emitRoute = [&](const MemoryRoute &memoryRoute) {
    if (transformRequired) {
      // A transform that no hop can carry makes this route no alternative at
      // all: the value cannot be held in either layout regime along it. The
      // chosen hop is not a `ConnectionPlan` field -- the plan carries the
      // route and the layout pair, and the boundary the two meet at is implied
      // by them -- so this validation is what fixes where the transform runs.
      // No current consumer materializes a transform (`PlanBinder` reports an
      // unmaterialized transform), so nothing is lost yet; a future transform
      // materializer must re-derive this first-legal-hop rule, or a hop field
      // must be added to `ConnectionPlan`, before it can place the transform.
      if (!legalTransformHop(memoryRoute.nodes, machine,
                             *request.producerLayout, *request.consumerLayout))
        return;
    }
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
  };
  for (const MemoryRoute *memoryRoute : directRoutes)
    emitRoute(*memoryRoute);
  for (const MemoryRoute *memoryRoute : multiHopRoutes)
    emitRoute(*memoryRoute);
  return plans;
}

llvm::Expected<std::vector<ConnectionPlan>>
synthesizeFanOut(const ConnectionRequest &base,
                 llvm::ArrayRef<ConnectionRequest> consumers,
                 const MachineModel &machine, const TopologyService &topology,
                 const PlacementOptions &options, bool *truncated,
                 const ObjectiveOrder &objective) {
  if (consumers.empty())
    return std::vector<ConnectionPlan>{};

  // §15.3(a): a shared read is legal exactly when every consumer can legally
  // access the producer's placement. That is a visibility fact, checked here
  // against *each* consumer's own request -- not a memory-equality shortcut and
  // not a check of one representative only.
  bool allShareProducerPlacement = true;
  for (const ConnectionRequest &consumer : consumers)
    if (!portsDirectCompatible(consumer, machine)) {
      allShareProducerPlacement = false;
      break;
    }

  if (allShareProducerPlacement) {
    ConnectionPlan plan;
    plan.producer = base.producer;
    plan.value = base.value;
    for (const ConnectionRequest &consumer : consumers)
      plan.consumers.push_back(consumer.consumer);
    llvm::sort(plan.consumers);
    plan.kind = ConnectionKind::Direct;
    plan.memoryRoute.push_back(base.producerMemory);
    plan.cost.localBytes = base.bytes;
    plan.id = computeConnectionId(plan);
    return std::vector<ConnectionPlan>{std::move(plan)};
  }

  // §15.3(b): replication. Consumers that share a destination memory share one
  // copy, so a copy's cost and capacity are counted once per memory rather
  // than once per consumer. Each consumer is still validated on its own
  // request, so a mismatched element type or affine relation is not missed.
  std::vector<MemoryNodeId> groupMemories;
  std::vector<std::vector<size_t>> groups;
  for (size_t index = 0; index < consumers.size(); ++index) {
    size_t group = 0;
    for (; group < groupMemories.size(); ++group)
      if (groupMemories[group] == consumers[index].consumerMemory)
        break;
    if (group == groupMemories.size()) {
      groupMemories.push_back(consumers[index].consumerMemory);
      groups.emplace_back();
    }
    groups[group].push_back(index);
  }

  std::vector<ConnectionPlan> plans;
  for (const std::vector<size_t> &group : groups) {
    // A group every member of which reads the producer's placement needs no
    // copy at all.
    bool groupSharesProducer = true;
    for (size_t index : group)
      if (!portsDirectCompatible(consumers[index], machine)) {
        groupSharesProducer = false;
        break;
      }
    if (groupSharesProducer) {
      ConnectionPlan plan;
      plan.producer = base.producer;
      plan.value = base.value;
      for (size_t index : group)
        plan.consumers.push_back(consumers[index].consumer);
      llvm::sort(plan.consumers);
      plan.kind = ConnectionKind::Direct;
      plan.memoryRoute.push_back(base.producerMemory);
      plan.cost.localBytes = base.bytes;
      plan.id = computeConnectionId(plan);
      plans.push_back(std::move(plan));
      continue;
    }

    // A copy into this memory serves the whole group. Validate every member and
    // pick the cheapest copy among their alternatives; group members share a
    // destination memory, so one copy's route is the group's route. Only plans
    // that actually move the value into the memory count -- a member's in-place
    // read or transform does not serve the group's non-sharing members.
    std::vector<ConnectionPlan> alternatives;
    for (size_t index : group) {
      llvm::Expected<std::vector<ConnectionPlan>> member =
          synthesizeConnections(consumers[index], machine, topology, options,
                                truncated);
      if (!member)
        return member.takeError();
      if (member->empty())
        return std::vector<ConnectionPlan>{}; // this consumer cannot be served
      for (ConnectionPlan &plan : *member) {
        if (plan.kind == ConnectionKind::Transfer ||
            plan.kind == ConnectionKind::TransferAndTransform) {
          plan.kind = ConnectionKind::Replicate;
          // `kind` is part of the canonical string, so the id is recomputed.
          plan.id = computeConnectionId(plan);
        }
        if (plan.kind == ConnectionKind::Replicate)
          alternatives.push_back(std::move(plan));
      }
    }
    if (alternatives.empty())
      return std::vector<ConnectionPlan>{};
    // §17.1: the copies are ranked by the declared objective, not by a
    // hard-coded dimension. `min_element` keeps the first alternative on an
    // exact tie, the same stable tie-break `CoveringSearch::pickBest` uses.
    auto best = llvm::min_element(alternatives, [&](const ConnectionPlan &lhs,
                                                    const ConnectionPlan &rhs) {
      return costLess(lhs.cost, rhs.cost, objective);
    });
    ConnectionPlan chosen = *best;
    chosen.consumers.clear();
    for (size_t index : group)
      chosen.consumers.push_back(consumers[index].consumer);
    llvm::sort(chosen.consumers);
    chosen.id = computeConnectionId(chosen);
    plans.push_back(std::move(chosen));
  }
  return plans;
}

ConnectionPlan synthesizeFanIn(llvm::ArrayRef<InstanceId> producers,
                               llvm::ArrayRef<InstanceId> consumers,
                               WorkloadValueId value,
                               MemoryNodeId consumerMemory, uint64_t bytes,
                               const Cost &feedCost) {
  ConnectionPlan plan;
  plan.kind = ConnectionKind::Reduce;
  plan.consumers.assign(consumers.begin(), consumers.end());
  plan.producers.assign(producers.begin(), producers.end());
  plan.value = value;
  plan.memoryRoute.push_back(std::move(consumerMemory));
  // A gather sums what its feeds cost. The gathered tile's `bytes` are charged
  // to its memory by the caller (capacity), not added again to `localBytes`,
  // which already carries each feed's staged value.
  plan.cost = feedCost;
  plan.id = computeConnectionId(plan);
  return plan;
}

} // namespace mlir::llk::mapping
