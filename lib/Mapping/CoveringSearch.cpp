//===- CoveringSearch.cpp - Complete-plan search (D6) ---------------------===//
//
// Three searches over one table of legal instances:
//
//   deterministic  depth-first in canonical order, first complete plan wins
//   beam           level-synchronous best-first, bounded by beamWidth
//   exact          branch and bound on the lowest-id uncovered node
//
// Connections are synthesized lazily: choosing an instance for a node
// immediately tests every edge whose other end is already chosen, so a branch
// dies at the first pair that cannot be connected rather than at completion.
//
// The bound used for ordering and pruning is admissible: accumulated cost plus,
// for each uncovered node, the cheapest instance available for it. Connection
// costs are non-negative, so it never overestimates.
//
//===----------------------------------------------------------------------===//

#include "LLK/Mapping/CoveringSearch.h"

#include "LLK/Mapping/LatencyProvider.h"
#include "LLK/Mapping/Routing.h"
#include "LLK/Mapping/StableHash.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringMap.h"

#include <algorithm>
#include <limits>
#include <optional>
#include <string>
#include <utility>

namespace mlir::llk::mapping {

namespace {

using machine::MachineModel;
using machine::MemoryNode;

/// Byte count assumed for a value crossing a connection. The real size comes
/// from the value's type once the plan binder carries it (#50/D7); costing a
/// route needs a number now, and every plan is costed the same way.
constexpr uint64_t kAssumedValueBytes = 4096;

/// Alignment every fixture memory supports.
constexpr uint64_t kAssumedAlignment = 32;

/// A legal placement and the rule that produced it, so a selected plan can
/// name the rule and bundle it chose.
struct InstanceEntry {
  CandidateInstance instance;
  const RuleDef *rule = nullptr;
  /// The instance's cost, after any measurement the target provides. Legality
  /// was decided before this and is not revisited.
  Cost cost;
};

struct NodeTable {
  WorkloadNodeId node = 0;
  const WorkloadNode *workload = nullptr;
  std::vector<InstanceEntry> instances;
};

struct Edge {
  size_t producer = 0; // index into the node table
  size_t consumer = 0;
  /// The value the edge carries, so a connection names what it moves.
  WorkloadValueId value = 0;
};

/// A partial cover: one instance chosen per covered node.
struct Partial {
  std::vector<const CandidateInstance *> chosen; // null while uncovered
  std::vector<size_t> connections;               // indices into the plan pool
  Cost cost;
  double lowerBound = std::numeric_limits<double>::infinity();
  uint64_t id = 0;
  unsigned covered = 0;
  uint64_t executorSlots = 0;
  llvm::StringMap<uint64_t> memoryBytes;
};

MemoryNodeId primaryMemory(const MachineModel &machine,
                           const CandidateInstance &instance) {
  std::vector<std::pair<std::string, std::string>> bindings;
  for (const auto &entry : instance.memoryBindings)
    bindings.emplace_back(entry.first().str(), entry.second);
  llvm::sort(bindings);
  if (!bindings.empty())
    return bindings.front().second;
  std::string executor = instance.executorBindings.lookup("executor");
  for (const MemoryNode &memory : machine.memories)
    if (machine.isVisible(memory.id, executor))
      return memory.id;
  return MemoryNodeId();
}

/// Hashes the chosen instance ids, so a partial plan has a stable identity
/// independent of how it was reached.
uint64_t partialId(const std::vector<const CandidateInstance *> &chosen) {
  std::vector<uint64_t> ids;
  for (const CandidateInstance *instance : chosen)
    if (instance)
      ids.push_back(instance->id);
  llvm::sort(ids);
  uint64_t hash = 0xcbf29ce484222325ULL;
  for (uint64_t id : ids) {
    for (unsigned byte = 0; byte < 8; ++byte) {
      hash ^= (id >> (8 * byte)) & 0xff;
      hash *= 0x100000001b3ULL;
    }
  }
  return hash;
}

} // namespace

CoveringSearch::CoveringSearch(const WorkloadGraph &workload,
                               const MappingTarget &target,
                               mlir::MLIRContext &context,
                               const LayoutContext &layoutContext,
                               const MappingSearchOptions &options)
    : workload_(workload), target_(target), context_(context),
      layoutContext_(layoutContext), options_(options) {}

llvm::Expected<MappingSearchResult> CoveringSearch::search() {
  const MachineModel &machine = target_.machine();
  MappingSearchResult result;

  // --- node tables -----------------------------------------------------
  std::vector<const WorkloadNode *> ordered;
  for (const WorkloadNode &node : workload_.getNodes())
    ordered.push_back(&node);
  llvm::sort(ordered, [](const WorkloadNode *lhs, const WorkloadNode *rhs) {
    return lhs->id < rhs->id;
  });

  PlacementOptions placementOptions;
  placementOptions.reduceSymmetry = options_.enableSymmetryReduction;
  placementOptions.maxInstances = options_.maxInstancesPerCandidate;
  placementOptions.maxRoutesPerConnection = options_.maxRoutesPerConnection;

  std::vector<NodeTable> tables;
  for (const WorkloadNode *node : ordered) {
    NodeTable table;
    table.node = node->id;
    table.workload = node;

    std::vector<const RuleDef *> matches = matchRules(*node, target_.rules());
    if (matches.size() > options_.maxCandidatesPerNode) {
      matches.resize(options_.maxCandidatesPerNode);
      result.searchTruncated = true;
    }
    if (matches.empty()) {
      ++result.frontier.nodesWithoutRules;
      result.frontier.messages.push_back("node " + std::to_string(node->id) +
                                         " ('" + node->opName +
                                         "'): no matching rule");
    }
    for (const RuleDef *rule : matches) {
      MappingCandidate candidate = toMappingCandidate(*rule, *node);
      llvm::Expected<std::vector<CandidateInstance>> instances =
          enumeratePlacements(candidate, target_, context_, layoutContext_,
                              placementOptions, &result.searchTruncated);
      if (!instances)
        return instances.takeError();
      if (instances->empty()) {
        ++result.frontier.candidatesWithoutPlacement;
        result.frontier.messages.push_back("node " + std::to_string(node->id) +
                                           ": rule '" + rule->id +
                                           "' has no legal placement");
      }
      for (CandidateInstance &instance : *instances) {
        InstanceEntry entry{std::move(instance), rule, {}};
        entry.cost = entry.instance.localCost;
        if (options_.enableLatencyCache) {
          if (const LatencyProvider *provider = target_.latencyProvider()) {
            OperationSignature signature;
            signature.operation = node->opName;
            signature.rule = rule->id;
            signature.ruleVersion = rule->version;
            signature.bundle = rule->bundle;
            if (!entry.instance.layoutBindings.empty()) {
              std::vector<std::string> layouts;
              for (const auto &binding : entry.instance.layoutBindings)
                layouts.push_back(binding.second);
              llvm::sort(layouts);
              signature.layout = layouts.front();
            }
            if (const machine::ExecutorNode *executor = machine.findExecutor(
                    entry.instance.executorBindings.lookup("executor")))
              signature.placementClass = executor->kind;
            TargetContext context{target_.name().str(),
                                  hexId(machine.contentHash)};
            if (std::optional<double> measured =
                    provider->lookupCycles(signature, context))
              entry.cost.latencyCycles = *measured;
          }
        }
        table.instances.push_back(std::move(entry));
      }
    }
    tables.push_back(std::move(table));
  }

  // --- dataflow edges --------------------------------------------------
  llvm::DenseMap<WorkloadValueId, size_t> producerOf;
  for (size_t index = 0; index < tables.size(); ++index)
    for (const WorkloadPort &port : tables[index].workload->outputs)
      producerOf[port.value] = index;
  std::vector<Edge> edges;
  for (size_t index = 0; index < tables.size(); ++index)
    for (const WorkloadPort &port : tables[index].workload->inputs) {
      auto producer = producerOf.find(port.value);
      if (producer != producerOf.end() && producer->second != index)
        edges.push_back({producer->second, index, port.value});
    }

  TopologyService topology(machine,
                           RouteOptions{options_.maxRoutesPerConnection, 4});
  std::vector<ConnectionPlan> pool;
  // The extend lambda answers "is this branch legal", so an error is recorded
  // here and surfaced once the search returns.
  llvm::Error pendingError = llvm::Error::success();

  // Extends `partial` with `instance` for `nodeIndex`, synthesizing every
  // connection whose other endpoint is already chosen. Returns false when the
  // branch is illegal or over budget.
  auto extend = [&](Partial &partial, size_t nodeIndex,
                    const InstanceEntry &entry) -> bool {
    const CandidateInstance &instance = entry.instance;
    partial.chosen[nodeIndex] = &instance;
    ++partial.covered;
    partial.executorSlots += instance.resourceUsage.executorSlots;
    // A bound memory holds the tile this instance materializes, plus anything
    // the rule declared explicitly.
    for (const auto &entry : instance.memoryBindings)
      partial.memoryBytes[entry.second] += kAssumedValueBytes;
    uint64_t totalBytes = 0;
    for (const auto &entry : instance.resourceUsage.memoryBytes)
      partial.memoryBytes[entry.first()] += entry.second;
    for (const auto &entry : partial.memoryBytes)
      totalBytes += entry.second;
    if (totalBytes > options_.memoryBudgetBytes) {
      ++result.frontier.plansRejectedByCapacity;
      return false;
    }

    Cost cost = partial.cost;
    cost = addCost(cost, entry.cost);

    for (const Edge &edge : edges) {
      const CandidateInstance *other = nullptr;
      const CandidateInstance *producer = nullptr;
      const CandidateInstance *consumer = nullptr;
      if (edge.consumer == nodeIndex) {
        // The instance being added reads the edge, so the *other* end produces.
        other = partial.chosen[edge.producer];
        producer = other;
        consumer = &instance;
      } else if (edge.producer == nodeIndex) {
        other = partial.chosen[edge.consumer];
        producer = &instance;
        consumer = other;
      } else {
        continue;
      }
      if (!other)
        continue;

      ConnectionRequest request;
      request.producer = producer->id;
      request.consumer = consumer->id;
      request.value = edge.value;
      request.producerMemory = primaryMemory(machine, *producer);
      request.consumerMemory = primaryMemory(machine, *consumer);
      request.bytes = kAssumedValueBytes;
      request.alignmentBytes = kAssumedAlignment;
      llvm::Expected<std::vector<ConnectionPlan>> alternatives =
          synthesizeConnections(request, machine, topology, placementOptions,
                                &result.searchTruncated);
      if (!alternatives) {
        result.frontier.messages.push_back(
            "connection " + request.producerMemory + " -> " +
            request.consumerMemory + ": " +
            llvm::toString(alternatives.takeError()));
        return false;
      }
      if (alternatives->empty()) {
        ++result.frontier.incompatibleInstancePairs;
        result.frontier.messages.push_back(
            "connection " + request.producerMemory + " -> " +
            request.consumerMemory + ": no legal route");
        return false;
      }
      // The cheapest alternative is the one a plan would use.
      auto best =
          llvm::min_element(*alternatives, [](const ConnectionPlan &lhs,
                                              const ConnectionPlan &rhs) {
            return lhs.cost.latencyCycles < rhs.cost.latencyCycles;
          });
      pool.push_back(*best);
      partial.connections.push_back(pool.size() - 1);
      cost = addCost(cost, best->cost);
    }

    partial.cost = cost;
    partial.id = partialId(partial.chosen);
    result.expandedStates++;
    return true;
  };

  // Admissible bound: accumulated cost plus the cheapest instance still
  // available for every uncovered node.
  auto bound = [&](const Partial &partial) -> double {
    double total = partial.cost.latencyCycles;
    for (size_t index = 0; index < tables.size(); ++index) {
      if (partial.chosen[index])
        continue;
      double cheapest = std::numeric_limits<double>::infinity();
      for (const InstanceEntry &entry : tables[index].instances)
        cheapest = std::min(cheapest, entry.instance.localCost.latencyCycles);
      if (!std::isfinite(cheapest))
        return std::numeric_limits<double>::infinity();
      total += cheapest;
    }
    return total;
  };

  auto lowestUncovered = [&](const Partial &partial) -> std::optional<size_t> {
    for (size_t index = 0; index < tables.size(); ++index)
      if (!partial.chosen[index])
        return index;
    return std::nullopt;
  };

  std::vector<Partial> complete;

  if (options_.mode == SearchMode::Beam) {
    Partial start;
    start.chosen.assign(tables.size(), nullptr);
    std::vector<Partial> beam{std::move(start)};
    for (size_t depth = 0; depth < tables.size(); ++depth) {
      std::vector<Partial> next;
      for (const Partial &partial : beam) {
        std::optional<size_t> node = lowestUncovered(partial);
        if (!node)
          continue;
        for (const InstanceEntry &entry : tables[*node].instances) {
          Partial branch = partial;
          if (extend(branch, *node, entry)) {
            branch.lowerBound = bound(branch);
            next.push_back(std::move(branch));
          }
        }
      }
      llvm::sort(next, [](const Partial &lhs, const Partial &rhs) {
        if (lhs.lowerBound != rhs.lowerBound)
          return lhs.lowerBound < rhs.lowerBound;
        if (lhs.covered != rhs.covered)
          return lhs.covered > rhs.covered;
        return lhs.id < rhs.id;
      });
      if (next.size() > options_.beamWidth) {
        next.resize(options_.beamWidth);
        result.searchTruncated = true;
      }
      beam = std::move(next);
      if (beam.empty())
        break;
    }
    for (Partial &partial : beam)
      if (!lowestUncovered(partial))
        complete.push_back(std::move(partial));
  } else {
    // Deterministic and exact share a depth-first walk; they differ in when
    // they stop and in whether the bound prunes.
    bool exact = options_.mode == SearchMode::Exact;
    std::vector<double> bestCosts; // best complete costs, ascending
    std::function<void(Partial &, bool &)> visit = [&](Partial &partial,
                                                       bool &stop) {
      if (stop)
        return;
      std::optional<size_t> node = lowestUncovered(partial);
      if (!node) {
        complete.push_back(partial);
        bestCosts.push_back(partial.cost.latencyCycles);
        llvm::sort(bestCosts);
        if (bestCosts.size() > options_.topK)
          bestCosts.resize(options_.topK);
        return;
      }
      for (const InstanceEntry &entry : tables[*node].instances) {
        if (stop)
          return;
        Partial branch = partial;
        if (!extend(branch, *node, entry))
          continue;
        if (exact) {
          branch.lowerBound = bound(branch);
          if (bestCosts.size() >= options_.topK &&
              branch.lowerBound >= bestCosts.back()) {
            // A full top-K list makes this prune exact -- it cannot drop a
            // plan we would keep -- but the space was not exhausted, and the
            // caller is told so.
            result.searchTruncated = true;
            continue;
          }
        }
        visit(branch, stop);
        if (!exact && !complete.empty()) {
          // Deterministic mode is defined as the first legal plan.
          stop = true;
          return;
        }
      }
    };

    Partial start;
    start.chosen.assign(tables.size(), nullptr);
    bool stop = false;
    visit(start, stop);
  }

  if (pendingError)
    return std::move(pendingError);

  // --- finalize --------------------------------------------------------
  llvm::sort(complete, [](const Partial &lhs, const Partial &rhs) {
    if (lhs.cost.latencyCycles != rhs.cost.latencyCycles)
      return lhs.cost.latencyCycles < rhs.cost.latencyCycles;
    return lhs.id < rhs.id;
  });
  if (complete.size() > options_.topK) {
    complete.resize(options_.topK);
    result.searchTruncated = true;
  }

  for (const Partial &partial : complete) {
    CoveringPlan plan;
    std::vector<InstanceId> instances;
    for (const CandidateInstance *instance : partial.chosen)
      if (instance)
        instances.push_back(instance->id);
    llvm::sort(instances);
    plan.instances.assign(instances.begin(), instances.end());

    std::vector<ConnectionId> connections;
    for (size_t index : partial.connections)
      connections.push_back(pool[index].id);
    llvm::sort(connections);
    plan.connections.assign(connections.begin(), connections.end());

    // Resolve the same selection into the facts a materializer needs.
    for (size_t index = 0; index < partial.chosen.size(); ++index) {
      const CandidateInstance *instance = partial.chosen[index];
      if (!instance)
        continue;
      PlanPlacement placement;
      placement.node = tables[index].node;
      placement.instance = instance->id;
      for (const InstanceEntry &entry : tables[index].instances) {
        if (&entry.instance == instance) {
          placement.rule = entry.rule->id;
          placement.bundle = entry.rule->bundle;
          break;
        }
      }
      placement.executor = instance->executorBindings.lookup("executor");
      placement.memories = instance->memoryBindings;
      placement.layouts = instance->layoutBindings;
      plan.placements.push_back(std::move(placement));
    }
    llvm::sort(plan.placements,
               [](const PlanPlacement &lhs, const PlanPlacement &rhs) {
                 return lhs.node < rhs.node;
               });

    for (size_t index : partial.connections) {
      const ConnectionPlan &connection = pool[index];
      PlanConnection detail;
      detail.id = connection.id;
      detail.value = connection.value;
      detail.kind = connection.kind;
      detail.route = connection.memoryRoute;
      detail.engines = connection.transferEngines;
      detail.transform = connection.transform;
      plan.connectionPlans.push_back(std::move(detail));
    }
    llvm::sort(plan.connectionPlans,
               [](const PlanConnection &lhs, const PlanConnection &rhs) {
                 return lhs.id < rhs.id;
               });

    plan.totalCost = partial.cost;
    plan.diagnostics.searchTruncated = result.searchTruncated;
    plan.id = computePlanId(plan);
    result.plans.push_back(std::move(plan));
  }
  return result;
}

} // namespace mlir::llk::mapping
