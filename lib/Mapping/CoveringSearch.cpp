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
#include "llvm/ADT/StringSet.h"

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

/// One end of a dataflow value: the node table index that produces or consumes
/// it and the port it crosses, so a connection can state the element type,
/// logical shape, and affine relation §10.2 compares.
struct ValueEndpoint {
  size_t node = 0;
  const WorkloadPort *port = nullptr;
};

/// A logical value and every place it crosses. One producer and one consumer is
/// a plain edge; several consumers call for fan-out, and several producers call
/// for a gather (design §15.3).
struct ValueLink {
  WorkloadValueId value = 0;
  std::vector<ValueEndpoint> producers;
  std::vector<ValueEndpoint> consumers;
};

/// True when `node` is one of the value's ends.
bool valueInvolves(const ValueLink &link, size_t node) {
  for (const ValueEndpoint &endpoint : link.producers)
    if (endpoint.node == node)
      return true;
  for (const ValueEndpoint &endpoint : link.consumers)
    if (endpoint.node == node)
      return true;
  return false;
}

/// A partial cover: one instance chosen per covered node.
struct Partial {
  std::vector<const CandidateInstance *> chosen; // null while uncovered
  std::vector<size_t> connections;               // indices into the plan pool
  /// One flag per value link: set once the value's connections are synthesized
  /// (when every endpoint is chosen), so a fan-out is built exactly once.
  std::vector<char> linked;
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

/// A binding map's `name=value` entries, sorted. A placement's memory and
/// layout binding tuples compare through this, never through `StringMap`
/// iteration order (design §22.1).
template <typename ValueT>
std::vector<std::string>
sortedBindings(const llvm::StringMap<ValueT> &bindings) {
  std::vector<std::string> entries;
  entries.reserve(bindings.size());
  for (const auto &entry : bindings) {
    std::string text = entry.first().str();
    text += '=';
    text += entry.second;
    entries.push_back(std::move(text));
  }
  llvm::sort(entries);
  return entries;
}

/// A map's keys in sorted order, so an unordered container is never walked as
/// output order (design §22.1).
template <typename ValueT>
llvm::SmallVector<llvm::StringRef, 8>
sortedKeys(const llvm::StringMap<ValueT> &map) {
  llvm::SmallVector<llvm::StringRef, 8> keys;
  keys.reserve(map.size());
  for (const auto &entry : map)
    keys.push_back(entry.first());
  llvm::sort(keys);
  return keys;
}

/// Canonical placement order (design §22.1): the executor id, then the sorted
/// memory bindings, then the sorted layout bindings. Node and instance ids
/// break a tie so the order over a complete plan is total.
bool placementBefore(const PlanPlacement &lhs, const PlanPlacement &rhs) {
  if (lhs.executor != rhs.executor)
    return lhs.executor < rhs.executor;
  std::vector<std::string> lhsMemories = sortedBindings(lhs.memories);
  std::vector<std::string> rhsMemories = sortedBindings(rhs.memories);
  if (lhsMemories != rhsMemories)
    return lhsMemories < rhsMemories;
  std::vector<std::string> lhsLayouts = sortedBindings(lhs.layouts);
  std::vector<std::string> rhsLayouts = sortedBindings(rhs.layouts);
  if (lhsLayouts != rhsLayouts)
    return lhsLayouts < rhsLayouts;
  if (lhs.node != rhs.node)
    return lhs.node < rhs.node;
  return lhs.instance < rhs.instance;
}

/// The §22.3 code naming a placement failure. `NoLegalMemory` has no dedicated
/// code in the minimum set: the executor that matched cannot supply a memory
/// the rule requires, so the executor is what is not legal.
DiagnosticCode placementFailureCode(PlacementFailure failure) {
  switch (failure) {
  case PlacementFailure::NoLegalLayout:
    return DiagnosticCode::NoLegalLayout;
  case PlacementFailure::UnsupportedComputeFragment:
    return DiagnosticCode::UnsupportedComputeFragment;
  case PlacementFailure::NoLegalExecutor:
  case PlacementFailure::NoLegalMemory:
  case PlacementFailure::None:
    return DiagnosticCode::NoLegalExecutor;
  }
  return DiagnosticCode::NoLegalExecutor;
}

} // namespace

bool FailureFrontier::has(DiagnosticCode code) const {
  for (const Diagnostic &diagnostic : diagnostics)
    if (diagnostic.code == code)
      return true;
  return false;
}

CoveringSearch::CoveringSearch(const WorkloadGraph &workload,
                               const MappingTarget &target,
                               mlir::MLIRContext &context,
                               const LayoutContext &layoutContext,
                               const MappingSearchOptions &options,
                               std::optional<SearchBinding> binding)
    : workload_(workload), target_(target), context_(context),
      layoutContext_(layoutContext), options_(options),
      binding_(std::move(binding)) {}

llvm::Expected<MappingSearchResult> CoveringSearch::search() {
  const MachineModel &machine = target_.machine();
  MappingSearchResult result;

  // Records a stable-coded failure once per (code, message) pair. A cause that
  // recurs on every branch -- a memory overflow, a provider that never caches a
  // rule, a cap hit -- stays a single frontier entry; the category counts carry
  // the tally. Order here is deterministic, and the final sort makes it so even
  // if a future caller reaches these in a different order.
  llvm::StringSet<> recordedDiagnostics;
  auto report = [&](DiagnosticCode code, std::string message) {
    // Count every occurrence (design §22.2), even when the distinct
    // (code, message) pair is already recorded, so a recurring cause is
    // visible as a tally.
    ++result.frontier.codeCounts[code];
    std::string key = stringifyDiagnosticCode(code).str();
    key += '\x1f';
    key += message;
    if (recordedDiagnostics.insert(key).second)
      result.frontier.diagnostics.push_back({code, std::move(message)});
  };

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
      report(DiagnosticCode::SearchTruncated,
             "candidate cap reached (maxCandidatesPerNode=" +
                 std::to_string(options_.maxCandidatesPerNode) + ")");
    }
    if (matches.empty()) {
      ++result.frontier.nodesWithoutRules;
      report(DiagnosticCode::NoMatchingRule,
             "node " + std::to_string(node->id) + " ('" + node->opName +
                 "'): no matching rule");
    }
    // A rule whose `require` constraints no assignment satisfies produces no
    // candidate: that is a non-match, so a node with only such rules has no
    // rule in effect and is counted under `nodesWithoutRules` once, below. If
    // the constraint search hit its cap the match was not proven false, so the
    // result is reported as truncated rather than silently treated as absent.
    bool producedCandidate = false;
    for (const RuleDef *rule : matches) {
      std::string reason;
      bool truncated = false;
      std::optional<MappingCandidate> candidate = toMappingCandidate(
          *rule, *node, machine, layoutContext_, &reason, &truncated);
      if (!candidate) {
        if (truncated) {
          result.searchTruncated = true;
          report(DiagnosticCode::SearchTruncated,
                 "constraint assignment cap reached while matching rule '" +
                     rule->id + "'");
        }
        report(DiagnosticCode::NoMatchingRule,
               "node " + std::to_string(node->id) + ": rule '" + rule->id +
                   "' not applicable: " + reason);
        continue;
      }
      producedCandidate = true;
      ++result.candidateCount;
      PlacementFailure placementFailure = PlacementFailure::None;
      bool placementTruncated = false;
      llvm::Expected<std::vector<CandidateInstance>> instances =
          enumeratePlacements(*candidate, target_, context_, layoutContext_,
                              placementOptions, &placementTruncated,
                              &placementFailure);
      if (!instances)
        return instances.takeError();
      result.instanceCount += instances->size();
      if (placementTruncated) {
        result.searchTruncated = true;
        report(DiagnosticCode::SearchTruncated,
               "instance cap reached (maxInstancesPerCandidate=" +
                   std::to_string(options_.maxInstancesPerCandidate) + ")");
      }
      if (instances->empty()) {
        ++result.frontier.candidatesWithoutPlacement;
        DiagnosticCode code = placementFailureCode(placementFailure);
        report(code, "node " + std::to_string(node->id) + ": rule '" +
                         rule->id + "' has no legal placement (" +
                         stringifyDiagnosticCode(code).str() + ")");
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
            else
              report(DiagnosticCode::LatencyCacheMiss,
                     "rule '" + rule->id + "' (op '" + node->opName +
                         "'): no cached latency; static estimate retained");
          }
        }
        table.instances.push_back(std::move(entry));
      }
    }
    if (!producedCandidate && !matches.empty()) {
      ++result.frontier.nodesWithoutRules;
      report(DiagnosticCode::NoMatchingRule,
             "node " + std::to_string(node->id) + " ('" + node->opName +
                 "'): no rule satisfies its constraints");
    }
    tables.push_back(std::move(table));
  }

  // --- dataflow values -------------------------------------------------
  // Each value and every place it crosses. A value crossed once is a plain
  // edge; several consumers make it a fan-out, and several producers make it a
  // gather (design §15.3). Endpoints are canonicalized so a repeated port is
  // still one link.
  llvm::DenseMap<WorkloadValueId, size_t> valueIndex;
  std::vector<ValueLink> allLinks;
  auto linkIndexFor = [&](WorkloadValueId value) -> size_t {
    auto it = valueIndex.find(value);
    if (it != valueIndex.end())
      return it->second;
    valueIndex[value] = allLinks.size();
    ValueLink link;
    link.value = value;
    allLinks.push_back(std::move(link));
    return allLinks.size() - 1;
  };
  for (size_t index = 0; index < tables.size(); ++index) {
    for (const WorkloadPort &port : tables[index].workload->outputs)
      allLinks[linkIndexFor(port.value)].producers.push_back({index, &port});
    for (const WorkloadPort &port : tables[index].workload->inputs)
      allLinks[linkIndexFor(port.value)].consumers.push_back({index, &port});
  }
  auto canonicalEndpoints = [](std::vector<ValueEndpoint> &endpoints) {
    llvm::sort(endpoints,
               [](const ValueEndpoint &lhs, const ValueEndpoint &rhs) {
                 return lhs.node < rhs.node;
               });
    endpoints.erase(
        std::unique(endpoints.begin(), endpoints.end(),
                    [](const ValueEndpoint &lhs, const ValueEndpoint &rhs) {
                      return lhs.node == rhs.node;
                    }),
        endpoints.end());
  };
  std::vector<ValueLink> valueLinks;
  for (ValueLink &link : allLinks) {
    canonicalEndpoints(link.producers);
    canonicalEndpoints(link.consumers);
    // Nothing crosses when one side is empty, and a node feeding itself is not
    // a connection.
    if (link.producers.empty() || link.consumers.empty())
      continue;
    if (link.producers.size() == 1) {
      size_t self = link.producers[0].node;
      link.consumers.erase(std::remove_if(link.consumers.begin(),
                                          link.consumers.end(),
                                          [&](const ValueEndpoint &endpoint) {
                                            return endpoint.node == self;
                                          }),
                           link.consumers.end());
      if (link.consumers.empty())
        continue;
    }
    valueLinks.push_back(std::move(link));
  }
  llvm::sort(valueLinks, [](const ValueLink &lhs, const ValueLink &rhs) {
    return lhs.value < rhs.value;
  });

  TopologyService topology(machine,
                           RouteOptions{options_.maxRoutesPerConnection, 4});
  std::vector<ConnectionPlan> pool;
  // The extend lambda answers "is this branch legal", so an error is recorded
  // here and surfaced once the search returns.
  llvm::Error pendingError = llvm::Error::success();

  // Extends `partial` with `instance` for `nodeIndex`, synthesizing every value
  // whose endpoints are all chosen now. Returns false when the branch is
  // illegal or over budget.
  auto extend = [&](Partial &partial, size_t nodeIndex,
                    const InstanceEntry &entry) -> bool {
    const CandidateInstance &instance = entry.instance;
    partial.chosen[nodeIndex] = &instance;
    ++partial.covered;
    partial.executorSlots += instance.resourceUsage.executorSlots;
    if (partial.linked.size() != valueLinks.size())
      partial.linked.assign(valueLinks.size(), 0);
    // A bound memory holds the tile this instance materializes, plus anything
    // the rule declared explicitly. `memoryBytes` is keyed by memory *node*, so
    // each byte can be charged to the node that actually holds it: the binding
    // map resolves a requirement kind to the node placement chose, while
    // `resourceUsage` is keyed by that requirement kind. Bytes are accumulated
    // for the duration of the partial plan -- an honest conservative live-range
    // bound, since true expiry is not modelled.
    for (const auto &binding : instance.memoryBindings)
      partial.memoryBytes[binding.second] += kAssumedValueBytes;
    for (const auto &usage : instance.resourceUsage.memoryBytes) {
      MemoryNodeId node = instance.memoryBindings.lookup(usage.first());
      if (node.empty())
        node = usage.first().str(); // no binding recorded: charge the raw key
      partial.memoryBytes[node] += usage.second;
    }

    Cost cost = partial.cost;
    cost = addCost(cost, entry.cost);

    // The connection facts §10.2 compares, gathered from one producer/consumer
    // pair's ports.
    auto makeRequest = [&](const CandidateInstance &producer,
                           const CandidateInstance &consumer,
                           const WorkloadPort *producerPort,
                           const WorkloadPort *consumerPort,
                           WorkloadValueId value) {
      ConnectionRequest request;
      request.producer = producer.id;
      request.consumer = consumer.id;
      request.value = value;
      request.producerMemory = primaryMemory(machine, producer);
      request.consumerMemory = primaryMemory(machine, consumer);
      request.bytes = kAssumedValueBytes;
      request.alignmentBytes = kAssumedAlignment;
      if (const WorkloadValue *moved = workload_.findValue(value))
        request.elementType = moved->type;
      if (consumerPort)
        request.consumerType = consumerPort->type;
      if (producerPort)
        request.producerMap = producerPort->accessMap;
      if (consumerPort)
        request.consumerMap = consumerPort->accessMap;
      if (ExecutorId executor = producer.executorBindings.lookup("executor");
          !executor.empty())
        request.producerExecutor = executor;
      if (ExecutorId executor = consumer.executorBindings.lookup("executor");
          !executor.empty())
        request.consumerExecutor = executor;
      return request;
    };
    // The alternative the declared objective prefers. `min_element` keeps the
    // first on an exact tie, the stable deterministic tie-break.
    auto pickBest = [&](llvm::ArrayRef<ConnectionPlan> alternatives)
        -> const ConnectionPlan * {
      if (alternatives.empty())
        return nullptr;
      return &*llvm::min_element(alternatives, [&](const ConnectionPlan &lhs,
                                                   const ConnectionPlan &rhs) {
        return costLess(lhs.cost, rhs.cost, options_.objective);
      });
    };
    auto reportTruncation = [&](bool truncated) {
      if (!truncated)
        return;
      result.searchTruncated = true;
      report(DiagnosticCode::SearchTruncated,
             "route cap reached (maxRoutesPerConnection=" +
                 std::to_string(options_.maxRoutesPerConnection) + ")");
    };
    auto incompatible = [&](const std::string &message) {
      ++result.frontier.incompatibleInstancePairs;
      report(DiagnosticCode::NoMemoryRoute, message);
    };

    // Connections are staged locally and committed to the pool only after the
    // capacity check, so a rejected branch leaves no partial state behind.
    std::vector<ConnectionPlan> staged;
    llvm::StringMap<uint64_t> stagedBytes; // memory -> replica/gather bytes

    // Synthesizes one plain-edge connection, staging the chosen alternative.
    auto connect = [&](const ConnectionRequest &request) -> bool {
      bool connectionTruncated = false;
      llvm::Expected<std::vector<ConnectionPlan>> alternatives =
          synthesizeConnections(request, machine, topology, placementOptions,
                                &connectionTruncated);
      if (alternatives)
        result.routeCount += alternatives->size();
      reportTruncation(connectionTruncated);
      if (!alternatives) {
        // A malformed connection request: it fails this branch, and the closest
        // §22.3 code is the one that names a connection that cannot be built.
        report(DiagnosticCode::NoMemoryRoute,
               "connection " + request.producerMemory + " -> " +
                   request.consumerMemory + ": " +
                   llvm::toString(alternatives.takeError()));
        return false;
      }
      if (alternatives->empty()) {
        incompatible("connection " + request.producerMemory + " -> " +
                     request.consumerMemory + ": no legal route");
        return false;
      }
      staged.push_back(*pickBest(*alternatives));
      cost = addCost(cost, staged.back().cost);
      return true;
    };

    for (size_t index = 0; index < valueLinks.size(); ++index) {
      const ValueLink &link = valueLinks[index];
      if (partial.linked[index])
        continue;
      if (!valueInvolves(link, nodeIndex))
        continue;
      bool complete = true;
      for (const ValueEndpoint &producer : link.producers)
        complete &= partial.chosen[producer.node] != nullptr;
      for (const ValueEndpoint &consumer : link.consumers)
        complete &= partial.chosen[consumer.node] != nullptr;
      if (!complete)
        continue;
      partial.linked[index] = 1;

      if (link.producers.size() == 1) {
        const ValueEndpoint &producerEnd = link.producers[0];
        const CandidateInstance &producer = *partial.chosen[producerEnd.node];
        if (link.consumers.size() == 1) {
          // A plain edge.
          const ValueEndpoint &consumerEnd = link.consumers[0];
          if (!connect(makeRequest(producer, *partial.chosen[consumerEnd.node],
                                   producerEnd.port, consumerEnd.port,
                                   link.value)))
            return false;
          continue;
        }

        // Fan-out (design §15.3): a shared read when every consumer reads the
        // producer's placement, otherwise a per-consumer plan (replication).
        std::vector<InstanceId> consumerIds;
        std::vector<MemoryNodeId> consumerMemories;
        for (const ValueEndpoint &consumerEnd : link.consumers) {
          const CandidateInstance &consumer = *partial.chosen[consumerEnd.node];
          consumerIds.push_back(consumer.id);
          consumerMemories.push_back(primaryMemory(machine, consumer));
        }
        const ValueEndpoint &firstConsumer = link.consumers[0];
        ConnectionRequest base =
            makeRequest(producer, *partial.chosen[firstConsumer.node],
                        producerEnd.port, firstConsumer.port, link.value);
        bool fanOutTruncated = false;
        llvm::Expected<std::vector<ConnectionPlan>> alternatives =
            synthesizeFanOut(base, consumerIds, consumerMemories, machine,
                             topology, placementOptions, &fanOutTruncated);
        if (alternatives)
          result.routeCount += alternatives->size();
        reportTruncation(fanOutTruncated);
        if (!alternatives) {
          report(DiagnosticCode::NoMemoryRoute,
                 "fan-out " + base.producerMemory + ": " +
                     llvm::toString(alternatives.takeError()));
          return false;
        }
        if (alternatives->empty()) {
          incompatible("fan-out " + base.producerMemory + ": no legal route");
          return false;
        }
        // A shared read carries every consumer in one plan.
        if (alternatives->front().consumers.size() == consumerIds.size()) {
          staged.push_back(alternatives->front());
          cost = addCost(cost, staged.back().cost);
        } else {
          for (InstanceId consumerId : consumerIds) {
            std::vector<ConnectionPlan> mine;
            for (const ConnectionPlan &plan : *alternatives)
              if (llvm::is_contained(plan.consumers, consumerId))
                mine.push_back(plan);
            const ConnectionPlan *best = pickBest(mine);
            if (!best) {
              incompatible("fan-out " + base.producerMemory +
                           ": a consumer has no legal route");
              return false;
            }
            staged.push_back(*best);
            cost = addCost(cost, staged.back().cost);
            // A replicated copy occupies the memory that holds it.
            if (staged.back().kind == ConnectionKind::Replicate &&
                !staged.back().memoryRoute.empty())
              stagedBytes[staged.back().memoryRoute.back()] +=
                  kAssumedValueBytes;
          }
        }
        continue;
      }

      // Fan-in (design §15.3): several producers feed one value, so a gather
      // sums their feeds into one intermediate tile.
      std::vector<InstanceId> producerIds;
      for (const ValueEndpoint &producerEnd : link.producers)
        producerIds.push_back(partial.chosen[producerEnd.node]->id);
      llvm::sort(producerIds);
      for (const ValueEndpoint &consumerEnd : link.consumers) {
        const CandidateInstance &consumer = *partial.chosen[consumerEnd.node];
        Cost feedCost;
        std::vector<ExecutorId> engines;
        for (const ValueEndpoint &producerEnd : link.producers) {
          ConnectionRequest request =
              makeRequest(*partial.chosen[producerEnd.node], consumer,
                          producerEnd.port, consumerEnd.port, link.value);
          bool feedTruncated = false;
          llvm::Expected<std::vector<ConnectionPlan>> alternatives =
              synthesizeConnections(request, machine, topology,
                                    placementOptions, &feedTruncated);
          if (alternatives)
            result.routeCount += alternatives->size();
          reportTruncation(feedTruncated);
          if (!alternatives) {
            report(DiagnosticCode::NoMemoryRoute,
                   "gather: " + llvm::toString(alternatives.takeError()));
            return false;
          }
          if (alternatives->empty()) {
            incompatible("gather: a producer has no legal route");
            return false;
          }
          const ConnectionPlan *best = pickBest(*alternatives);
          feedCost = addCost(feedCost, best->cost);
          for (const ExecutorId &engine : best->transferEngines)
            if (!llvm::is_contained(engines, engine))
              engines.push_back(engine);
        }
        MemoryNodeId consumerMemory = primaryMemory(machine, consumer);
        ConnectionPlan reduce =
            synthesizeFanIn(producerIds, consumer.id, link.value,
                            consumerMemory, kAssumedValueBytes, feedCost);
        reduce.transferEngines.assign(engines.begin(), engines.end());
        staged.push_back(std::move(reduce));
        cost = addCost(cost, staged.back().cost);
        // The gather stages its reduced intermediate tile on the consumer.
        stagedBytes[consumerMemory] += kAssumedValueBytes;
      }
    }

    // §9.3: no memory's own capacity may be exceeded (per-node), on top of the
    // global byte ceiling (whole plan). Replicated copies and gathered
    // intermediate tiles are charged to the memory that holds them alongside
    // the instances' own bytes. Keys are walked sorted: which memory is
    // inspected first decides whether an unknown binding is reported as an
    // error or an over-capacity one merely rejects the branch, so `StringMap`
    // iteration order must not reach that decision (design §22.1).
    for (const auto &entry : stagedBytes)
      partial.memoryBytes[entry.first()] += entry.second;
    uint64_t totalBytes = 0;
    for (llvm::StringRef key : sortedKeys(partial.memoryBytes)) {
      uint64_t bytes = partial.memoryBytes.lookup(key);
      totalBytes += bytes;
      const MemoryNode *memory = machine.findMemory(key);
      if (!memory) {
        // Defensive: an id no machine node names cannot be checked. Surface it
        // as an error rather than silently treating it as unlimited.
        if (!pendingError)
          pendingError = llvm::createStringError(
              llvm::inconvertibleErrorCode(),
              "covering search: plan binds unknown memory '" + key.str() + "'");
        return false;
      }
      if (bytes > memory->capacityBytes) {
        ++result.frontier.plansRejectedByCapacity;
        report(DiagnosticCode::MemoryCapacityExceeded,
               "memory '" + key.str() + "' over capacity (" +
                   std::to_string(memory->capacityBytes) + " bytes)");
        return false;
      }
    }
    if (totalBytes > options_.memoryBudgetBytes) {
      ++result.frontier.plansRejectedByCapacity;
      report(DiagnosticCode::MemoryCapacityExceeded,
             "plan byte total exceeds the global budget (" +
                 std::to_string(options_.memoryBudgetBytes) + " bytes)");
      return false;
    }

    // The branch is legal: publish its connections to the shared pool.
    for (ConnectionPlan &plan : staged) {
      pool.push_back(std::move(plan));
      partial.connections.push_back(pool.size() - 1);
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
        report(DiagnosticCode::SearchTruncated,
               "beam width reached (beamWidth=" +
                   std::to_string(options_.beamWidth) + ")");
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
            report(DiagnosticCode::SearchTruncated,
                   "exact search pruned by the top-K bound (topK=" +
                       std::to_string(options_.topK) + ")");
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
  // The declared objective ranks complete plans; the stable id breaks an exact
  // tie so the order is deterministic (design §17.1).
  llvm::sort(complete, [&](const Partial &lhs, const Partial &rhs) {
    return ranksBefore(lhs.cost, lhs.id, rhs.cost, rhs.id, options_.objective);
  });
  // Tally complete plans before the top-K cap drops the tail (design §22.2).
  result.planCount = complete.size();
  if (complete.size() > options_.topK) {
    complete.resize(options_.topK);
    result.searchTruncated = true;
    report(DiagnosticCode::SearchTruncated,
           "top-K cap reached (topK=" + std::to_string(options_.topK) + ")");
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
      // The bundle travels on the placed instance, so it reaches the plan
      // unchanged and generic code never re-reads the rule for it.
      placement.bundle = instance->bundle;
      for (const InstanceEntry &entry : tables[index].instances) {
        if (&entry.instance == instance) {
          placement.rule = entry.rule->id;
          break;
        }
      }
      placement.executor = instance->executorBindings.lookup("executor");
      placement.memories = instance->memoryBindings;
      placement.layouts = instance->layoutBindings;
      plan.placements.push_back(std::move(placement));
    }
    llvm::sort(plan.placements, placementBefore);

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
    // Record the search point this plan came from before its content id is
    // folded, so two plans differing only by their binding do not collide.
    if (binding_) {
      plan.sourceBindingHash = binding_->stableHash;
      plan.globalParameters = binding_->values;
    }
    plan.id = computePlanId(plan);
    result.plans.push_back(std::move(plan));
  }

  // §22.1/§22.3: the frontier's codes are the stable interface, so their order
  // must not depend on the order branches happened to be explored.
  llvm::sort(result.frontier.diagnostics, diagnosticLess);
  return result;
}

} // namespace mlir::llk::mapping
