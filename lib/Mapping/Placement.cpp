//===- Placement.cpp - Placement and connection synthesis (D5) -----------===//

#include "LLK/Mapping/Placement.h"

#include "LLK/Mapping/SearchBinding.h"
#include "LLK/Mapping/TileFacts.h"

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

/// §10.2: a changed element type or logical tile shape admits no alternative --
/// neither a layout transform nor a transfer rewrites it -- so it rejects the
/// pair outright. Only a fact both ends state is compared: an unstated end (a
/// null type, or a type neither helper can read) yields no fact and can never
/// reject. The reads go through `TileFacts`, the one place a `!micro.tile` is
/// unwrapped to the tensor its head spells, so a tile-typed port is compared
/// rather than silently skipped -- the same element type and shape that size
/// the moving value are what §10.2 compares here.
bool elementAndShapeCompatible(const ConnectionRequest &request) {
  mlir::Type producer = elementTypeOf(request.elementType);
  mlir::Type consumer = elementTypeOf(request.consumerType);
  if (producer && consumer && producer != consumer)
    return false;

  std::optional<llvm::SmallVector<int64_t, 4>> producerShape =
      staticShapeOf(request.elementType);
  std::optional<llvm::SmallVector<int64_t, 4>> consumerShape =
      staticShapeOf(request.consumerType);
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

/// True when two solved layout parameterizations denote the same physical
/// representation: same names, same values. An absent entry on either side is a
/// different (empty) map, but two absent maps compare equal -- so a caller that
/// states only layout class ids is unaffected.
bool sameLayoutParameters(const llvm::StringMap<SearchValue> &lhs,
                          const llvm::StringMap<SearchValue> &rhs) {
  if (lhs.size() != rhs.size())
    return false;
  for (const auto &entry : lhs) {
    auto other = rhs.find(entry.first());
    if (other == rhs.end() || !(other->second == entry.second))
      return false;
  }
  return true;
}

} // namespace

bool portsDirectCompatible(const ConnectionRequest &request,
                           const machine::MachineModel &machine) {
  if (!elementAndShapeCompatible(request))
    return false;
  // A different layout is a transform, not a direct connection. "Different"
  // covers both the family id and the concrete parameterization: two endpoints
  // that both name `t.blocked` but solved `VW = 4` against `VW = 8` hold
  // different representations and cannot be read as one another. A caller that
  // states only class ids leaves both parameter maps empty, which compare
  // equal, so that behaviour is unchanged.
  if (request.producerLayout && request.consumerLayout &&
      (*request.producerLayout != *request.consumerLayout ||
       !sameLayoutParameters(request.producerLayoutParameters,
                             request.consumerLayoutParameters)))
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
  // definition is kept so an instance can bind it, and its first legal solution
  // is kept so the instance can record the parameterization it uses.
  // `layoutBindings` carries the layout *definition* id only: the
  // registry-resolved id of the definition the requirement names, and therefore
  // string-identical to `requirement.layoutClass` by construction. The concrete
  // assignment travels separately in `layoutSolutions`, because an instance
  // that says only "t.blocked" cannot tell a materializer whether it meant
  // `VW = 4` or `VW = 8`.
  //
  // A solve may report several solutions and placement does not rank them
  // (choosing between them by cost is the tuner's job). It binds the *first*
  // solution the solver reports: the solver enumerates the declared domains in
  // declaration order, so that pick is deterministic and a plan is
  // reproducible. A tuner that wants another solution needs a selection surface
  // here -- recording a different one silently would be the worse failure.
  std::unique_ptr<LayoutSolver> layoutSolver = makeBoundedLayoutSolver();
  std::vector<const LayoutDef *> solvedDefs;
  std::vector<LayoutSolution> solvedSolutions;
  solvedDefs.reserve(candidate.layoutRequirements.size());
  solvedSolutions.reserve(candidate.layoutRequirements.size());
  for (const LayoutRequirement &requirement : candidate.layoutRequirements) {
    const LayoutDef *def = target.layouts().find(requirement.layoutClass);
    if (!def)
      return placementError("placement: unknown layout '" +
                            requirement.layoutClass + "'");
    // Solve against the operand's own element type and rank when the candidate
    // resolved them; fall back to the caller's context otherwise (a hand-built
    // candidate leaves them unset). Solving every requirement against one
    // graph-wide context would reject a legal per-operand layout whenever the
    // graph's first value has a different dtype than the operand -- e.g. an
    // f32 vector op in a bf16 tile program.
    LayoutContext requirementContext = layoutContext;
    if (!requirement.elementType.empty())
      requirementContext.elementType = requirement.elementType;
    if (requirement.rank >= 0)
      requirementContext.rank = requirement.rank;
    llvm::Expected<LayoutSolveResult> solved =
        layoutSolver->solve(*def, machine, context, requirementContext);
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
    solvedSolutions.push_back(solved->solutions.front());
  }

  // How many requirements share each layout class. One class may be required by
  // more than one port of the same candidate (the same blocked layout on two
  // operands, say), and each requirement has its own port association and its
  // own solved parameterization -- so their solutions must not share a key.
  llvm::StringMap<unsigned> layoutClassCounts;
  for (const LayoutRequirement &requirement : candidate.layoutRequirements)
    ++layoutClassCounts[requirement.layoutClass];

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
      for (size_t j = 0; j < memoryChoices.size(); ++j) {
        const MemoryRequirement &requirement = candidate.memoryRequirements[j];
        const MemoryNodeId memory =
            memoryChoices[j][pick[computeChoices.size() + j]]->id;
        // A requirement that named a port records the occurrence it governs, so
        // two same-kind requirements keep distinct nodes. A bare requirement
        // stays keyed by kind, unchanged.
        if (requirement.port)
          instance.portMemoryBindings.push_back(
              PortMemoryBinding{*requirement.port, memory});
        else
          instance.memoryBindings[requirement.kind] = memory;
      }
      // Sorted by occurrence so map iteration order never reaches an id.
      llvm::sort(instance.portMemoryBindings, [](const PortMemoryBinding &lhs,
                                                 const PortMemoryBinding &rhs) {
        if (lhs.port.node != rhs.port.node)
          return lhs.port.node < rhs.port.node;
        if (lhs.port.direction != rhs.port.direction)
          return lhs.port.direction < rhs.port.direction;
        return lhs.port.index < rhs.port.index;
      });
      for (size_t i = 0; i < solvedDefs.size(); ++i) {
        const LayoutRequirement &requirement = candidate.layoutRequirements[i];
        const std::string &layoutClass = requirement.layoutClass;
        instance.layoutBindings[layoutClass] = solvedDefs[i]->id;
        // The solved assignment and its affine map, so the instance states
        // *which* parameterization of the bound layout it uses (`VW = 8`, not
        // merely "some legal `VW`"). `LayoutValue` and `SearchValue` are the
        // same variant, so the values carry over without conversion.
        //
        // A class required once keeps the bare class as its key, so a
        // single-requirement instance is byte-identical to before and a
        // class-keyed lookup still resolves. A class required by several ports
        // is keyed by class plus the requirement's index -- unique within the
        // candidate and deterministic -- so each requirement keeps its own
        // solved parameters and port association instead of the last one
        // overwriting the rest.
        std::string solutionKey = layoutClass;
        if (layoutClassCounts[layoutClass] > 1)
          solutionKey += "#" + std::to_string(i);
        SolvedLayout solvedLayout;
        solvedLayout.layoutClass = layoutClass;
        for (const auto &value : solvedSolutions[i].values)
          solvedLayout.parameters[value.first] = value.second;
        solvedLayout.map = solvedSolutions[i].map;
        // Which value the requirement was solved for, so an edge can ask for
        // *its* layout rather than the instance's only one.
        solvedLayout.portValue = requirement.portValue;
        // The occurrence the requirement named, so a connection request can ask
        // for *this* use's layout rather than a value-wide one that two uses
        // would have to share.
        solvedLayout.port = requirement.port;
        instance.layoutSolutions[solutionKey] = std::move(solvedLayout);
      }

      instance.resourceUsage.executorSlots = 1;
      // Only bare requirements contribute a kind-keyed byte figure; a
      // named-port requirement's storage is accounted per occurrence, not per
      // kind.
      for (const MemoryRequirement &requirement : candidate.memoryRequirements)
        if (!requirement.port)
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

  // A layout difference is a difference of family *or* of the concrete
  // parameterization solved for the value: two endpoints that both bound
  // `t.blocked` but solved `VW = 4` against `VW = 8` hold different
  // representations, and connecting them directly would be wrong.
  const bool transformRequired =
      request.producerLayout && request.consumerLayout &&
      (*request.producerLayout != *request.consumerLayout ||
       !sameLayoutParameters(request.producerLayoutParameters,
                             request.consumerLayoutParameters));
  auto transformOf = [&]() -> std::optional<LayoutTransform> {
    if (!transformRequired)
      return std::nullopt;
    LayoutTransform transform;
    transform.srcLayout = *request.producerLayout;
    transform.dstLayout = *request.consumerLayout;
    // The endpoints' solved maps, so the bound plan can materialize the
    // conversion as a target-neutral `micro.transform` rather than only naming
    // the two families.
    transform.srcMap = request.producerLayoutMap;
    transform.dstMap = request.consumerLayoutMap;
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

  // Copies the request's endpoint occurrences onto every synthesized plan, so
  // the connection names the *use* it serves -- not just the SSA value two uses
  // share -- and folds them into its canonical id. A request with no endpoints
  // leaves the plan's endpoint fields empty and its id as it was before the
  // migration.
  auto assignEndpoints = [&](ConnectionPlan &plan) {
    plan.producerPort = request.producerPort;
    if (request.consumerPort)
      plan.consumerPorts.push_back(*request.consumerPort);
  };

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
    assignEndpoints(plan);
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
    assignEndpoints(plan);
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
  // The caller's per-memory live bytes travel with the request: this layer
  // keeps no occupancy state of its own, so it forwards exactly the figure it
  // was given and asserts nothing otherwise. An empty map leaves every
  // intermediate treated as empty, the documented default.
  route.liveBytesByIntermediate = request.intermediateOccupancy;
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
    assignEndpoints(plan);
    plan.id = computeConnectionId(plan);
    plans.push_back(std::move(plan));
  };
  for (const MemoryRoute *memoryRoute : directRoutes)
    emitRoute(*memoryRoute);
  for (const MemoryRoute *memoryRoute : multiHopRoutes)
    emitRoute(*memoryRoute);
  return plans;
}

llvm::Expected<std::vector<ConnectionPlan>> synthesizeFanOut(
    const ConnectionRequest &base, llvm::ArrayRef<ConnectionRequest> consumers,
    const MachineModel &machine, const TopologyService &topology,
    const PlacementOptions &options, bool *truncated,
    const ObjectiveOrder &objective, bool *choseAmongAlternatives) {
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
    plan.producerPort = base.producerPort;
    for (const ConnectionRequest &consumer : consumers) {
      plan.consumers.push_back(consumer.consumer);
      // Every consumer *use* this shared read serves, so two ports of one
      // consumer both appear even though their instance id repeats.
      if (consumer.consumerPort)
        plan.consumerPorts.push_back(*consumer.consumerPort);
    }
    sortUnique(plan.consumers);
    plan.kind = ConnectionKind::Direct;
    plan.memoryRoute.push_back(base.producerMemory);
    plan.cost.localBytes = base.bytes;
    plan.id = computeConnectionId(plan);
    return std::vector<ConnectionPlan>{std::move(plan)};
  }

  // §15.3(b): replication. Consumers that share a destination memory *and* the
  // same required representation share one copy or transform, so its cost and
  // capacity are counted once per representation rather than once per consumer.
  // Each consumer is still validated on its own request, so a mismatched
  // element type or affine relation is not missed. The layout a consumer binds
  // partitions the group as well as the memory: grouping by memory alone would
  // assign one member's resulting layout to a consumer that requires a
  // different one, a representation it could never read.
  // The layout a consumer binds partitions the group by *family and concrete
  // parameterization*: two consumers that both name `t.blocked` but solved
  // `VW = 4` against `VW = 8` require different representations, so one copy
  // cannot serve both.
  struct FanOutKey {
    MemoryNodeId memory;
    std::optional<LayoutId> layout;
    std::string layoutParameters;
    bool operator==(const FanOutKey &other) const {
      return memory == other.memory && layout == other.layout &&
             layoutParameters == other.layoutParameters;
    }
  };
  std::vector<FanOutKey> groupKeys;
  std::vector<std::vector<size_t>> groups;
  for (size_t index = 0; index < consumers.size(); ++index) {
    FanOutKey key{
        consumers[index].consumerMemory, consumers[index].consumerLayout,
        canonicalSearchValueString(consumers[index].consumerLayoutParameters)};
    size_t group = 0;
    for (; group < groupKeys.size(); ++group)
      if (groupKeys[group] == key)
        break;
    if (group == groupKeys.size()) {
      groupKeys.push_back(key);
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
      plan.producerPort = base.producerPort;
      for (size_t index : group) {
        plan.consumers.push_back(consumers[index].consumer);
        if (consumers[index].consumerPort)
          plan.consumerPorts.push_back(*consumers[index].consumerPort);
      }
      sortUnique(plan.consumers);
      plan.kind = ConnectionKind::Direct;
      plan.memoryRoute.push_back(base.producerMemory);
      plan.cost.localBytes = base.bytes;
      plan.id = computeConnectionId(plan);
      plans.push_back(std::move(plan));
      continue;
    }

    // An in-place layout transform is a shared read of the producer's memory,
    // so it can serve the whole group only when *every* member can reach that
    // memory. A member that cannot would have offered a transfer instead, so
    // excluding it here keeps a shared transform from being assigned to a
    // consumer that cannot read where it lands.
    bool groupSeesProducer = true;
    for (size_t index : group)
      if (!consumerSeesProducerMemory(consumers[index], machine)) {
        groupSeesProducer = false;
        break;
      }

    // A copy into this memory serves the whole group. Validate every member and
    // pick the cheapest alternative among them; group members share a
    // destination memory, so one copy's route is the group's route. Only plans
    // that actually serve the group count -- a member's in-place read or
    // transform does not serve the group's non-sharing members unless every
    // member can read the producer's memory too.
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
        if (plan.kind == ConnectionKind::Replicate ||
            (plan.kind == ConnectionKind::LayoutTransform && groupSeesProducer))
          alternatives.push_back(std::move(plan));
      }
    }
    if (alternatives.empty())
      return std::vector<ConnectionPlan>{};
    // More than one way to serve this group, but only the cheapest is taken:
    // the caller can no longer claim an exhaustive joint search, so it is told.
    if (alternatives.size() > 1 && choseAmongAlternatives)
      *choseAmongAlternatives = true;
    // §17.1: the copies are ranked by the declared objective, not by a
    // hard-coded dimension. `min_element` keeps the first alternative on an
    // exact tie, the same stable tie-break `CoveringSearch::pickBest` uses.
    auto best = llvm::min_element(alternatives, [&](const ConnectionPlan &lhs,
                                                    const ConnectionPlan &rhs) {
      return costLess(lhs.cost, rhs.cost, objective);
    });
    ConnectionPlan chosen = *best;
    chosen.consumers.clear();
    chosen.consumerPorts.clear();
    chosen.producerPort = base.producerPort;
    for (size_t index : group) {
      chosen.consumers.push_back(consumers[index].consumer);
      if (consumers[index].consumerPort)
        chosen.consumerPorts.push_back(*consumers[index].consumerPort);
    }
    sortUnique(chosen.consumers);
    chosen.id = computeConnectionId(chosen);
    plans.push_back(std::move(chosen));
  }
  return plans;
}

ConnectionPlan synthesizeFanIn(llvm::ArrayRef<InstanceId> producers,
                               llvm::ArrayRef<InstanceId> consumers,
                               WorkloadValueId value,
                               MemoryNodeId consumerMemory, uint64_t bytes,
                               const Cost &feedCost,
                               llvm::ArrayRef<PortRef> consumerPorts) {
  ConnectionPlan plan;
  plan.kind = ConnectionKind::Reduce;
  plan.consumers.assign(consumers.begin(), consumers.end());
  plan.producers.assign(producers.begin(), producers.end());
  plan.consumerPorts.assign(consumerPorts.begin(), consumerPorts.end());
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
