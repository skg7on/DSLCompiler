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
// The bound used for ordering and pruning is accumulated cost plus, for each
// uncovered node, the componentwise best still reachable -- the
// measured-or-static `entry.cost`, never the stale static estimate -- and its
// direction follows the declared objective. It omits connection costs, whose
// sign is non-negative; that omission keeps it admissible (never above a
// completion) for a minimize objective, while for maximize it makes the bound
// too small to prune on, so maximize explores fully and relies on the caps
// instead.
//
//===----------------------------------------------------------------------===//

#include "LLK/Mapping/CoveringSearch.h"

#include "LLK/Mapping/LatencyProvider.h"
#include "LLK/Mapping/MappingMetadata.h"
#include "LLK/Mapping/Routing.h"
#include "LLK/Mapping/StableHash.h"
#include "LLK/Mapping/StoragePlan.h"
#include "LLK/Mapping/TileFacts.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringSet.h"

#include <algorithm>
#include <cassert>
#include <optional>
#include <string>
#include <utility>

namespace mlir::llk::mapping {

namespace {

using machine::MachineModel;
using machine::MemoryNode;

/// Byte count and alignment assumed for a value whose tile size cannot be
/// derived (a dynamic shape, or a type the core does not model). These are the
/// pre-phase-3 constants, kept so an unmeasurable value is costed exactly as it
/// was -- but the fallback is *reported*, naming the value, never taken
/// silently (see `factsForValue`).
constexpr uint64_t kUnknownValueBytes = 4096;

constexpr uint64_t kUnknownAlignment = 32;

/// A legal placement and the rule that produced it, so a selected plan can
/// name the rule and bundle it chose.
struct InstanceEntry {
  CandidateInstance instance;
  const RuleDef *rule = nullptr;
  /// The instance's cost, after any measurement the target provides. Legality
  /// was decided before this and is not revisited.
  Cost cost;
  /// The resolved values of the rule's constraint-derived parameters, copied
  /// from the candidate so the selected plan can persist the assignment
  /// generation solved (task B1).
  llvm::StringMap<SearchValue> resolvedParameters;
};

struct NodeTable {
  WorkloadNodeId node = 0;
  const WorkloadNode *workload = nullptr;
  std::vector<InstanceEntry> instances;
};

/// One end of a dataflow value: the node table index that produces or consumes
/// it, the position of the port within that side, and the port it crosses, so a
/// connection can state the element type, logical shape, and affine relation
/// §10.2 compares. The index is what tells two operand uses of one value apart;
/// the node alone would collapse them.
struct ValueEndpoint {
  size_t node = 0;
  uint32_t index = 0;
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
  /// Optimistic completion cost, direction-aware and measured (`boundCost`).
  /// Multi-dimensional: the beam order always compares it through
  /// `boundIsBetterThan`, never as a bare latency. The exact prune compares it
  /// through the same predicate -- but only under a minimize objective, the
  /// only direction the bound is admissible for; an exact maximize run skips
  /// the prune block entirely.
  Cost lowerBound = infiniteCost();
  uint64_t id = 0;
  unsigned covered = 0;
  uint64_t executorSlots = 0;
  llvm::StringMap<uint64_t> memoryBytes;
  /// Per-value live-range accounting: the memory charges a value currently
  /// owes, as exact (memory, bytes) pairs so the release subtracts precisely
  /// what the charge added. Populated when the value's producer is placed and
  /// cleared when the value's *last* consumer is placed. A value the graph
  /// never fully consumes keeps its charges for the whole plan (it is not in
  /// `valueLinks`, so nothing ever releases it) -- the conservative direction.
  llvm::DenseMap<WorkloadValueId,
                 llvm::SmallVector<std::pair<MemoryNodeId, uint64_t>, 2>>
      liveValueCharges;
};

/// A named-port memory assignment list rendered `port=memory`, sorted. Shared
/// by the primary-memory pick and the placement-tie-break key so both agree on
/// order.
std::vector<std::string>
sortedPortMemories(const std::vector<PortMemoryBinding> &bindings) {
  std::vector<std::string> entries;
  entries.reserve(bindings.size());
  for (const PortMemoryBinding &binding : bindings) {
    std::string text = canonicalPortRefString(binding.port);
    text += '=';
    text += binding.memory;
    entries.push_back(std::move(text));
  }
  llvm::sort(entries);
  return entries;
}

MemoryNodeId primaryMemory(const MachineModel &machine,
                           const CandidateInstance &instance) {
  std::vector<std::pair<std::string, std::string>> bindings;
  for (const auto &entry : instance.memoryBindings)
    bindings.emplace_back(entry.first().str(), entry.second);
  llvm::sort(bindings);
  if (!bindings.empty())
    return bindings.front().second;
  // A named-port requirement files its node in `portMemoryBindings`, not in the
  // kind-keyed map, so the first occurrence's node is the primary memory when
  // there is no bare binding.
  if (!instance.portMemoryBindings.empty()) {
    const PortMemoryBinding *best = nullptr;
    std::string bestKey;
    for (const PortMemoryBinding &binding : instance.portMemoryBindings) {
      std::string key = canonicalPortRefString(binding.port);
      if (!best || key < bestKey) {
        best = &binding;
        bestKey = std::move(key);
      }
    }
    return best->memory;
  }
  std::string executor = instance.executorBindings.lookup("executor");
  for (const MemoryNode &memory : machine.memories)
    if (machine.isVisible(memory.id, executor))
      return memory.id;
  return MemoryNodeId();
}

/// The layout `instance` solved for the endpoint occurrence `port`, or nullopt
/// when it solved none for it.
///
/// The match is by *occurrence*, not by the SSA value the occurrence carries: a
/// rule's layout requirement names a port, and two operand uses of one value
/// can require different layouts. Matching by value would make those two
/// solutions ambiguous ("nothing governs it") and drop both, collapsing the
/// two uses into one unattributed request. Matching by the endpoint keeps each
/// use's obligation: the edge carrying a different occurrence -- or a
/// requirement that named no resolvable port, whose solution carries no
/// endpoint -- does not match. `avx2.mma_bf16` requires `lhs` to be
/// `avx2.row_major`, and `lhs` is an *input*, so the edge carrying the mma's
/// `result` must not inherit that layout.
///
/// Several solutions for one occurrence that *agree* (same class and
/// parameters) are one representation and resolve to it. Several that disagree
/// are a rule's alternative offers for one use (`require layout operand0
/// satisfies t.plain` and `... satisfies t.blocked` with no binding to choose
/// between them): no single layout governs the use, so the lookup reports none
/// -- exactly as it did by value -- rather than picking one by iteration order
/// or rejecting the whole candidate. This is a different fact from two
/// *occurrences* requiring different layouts, which each resolve on their own.
const SolvedLayout *boundSolvedLayoutForPort(const CandidateInstance &instance,
                                             const PortRef &port) {
  const SolvedLayout *found = nullptr;
  for (const auto &entry : instance.layoutSolutions) {
    if (!entry.second.port || !(*entry.second.port == port))
      continue;
    if (!found) {
      found = &entry.second;
      continue;
    }
    if (found->layoutClass != entry.second.layoutClass ||
        canonicalSearchValueString(found->parameters) !=
            canonicalSearchValueString(entry.second.parameters))
      return nullptr; // disagreeing alternatives for one use: nothing governs
                      // it
  }
  return found;
}

/// The layout *family* `instance` bound for `port`. Taken from the solved
/// layout's own class rather than its containing map's key, which is
/// index-disambiguated when one class is required by several ports.
std::optional<LayoutId> boundLayoutForPort(const CandidateInstance &instance,
                                           const PortRef &port) {
  const SolvedLayout *solved = boundSolvedLayoutForPort(instance, port);
  if (!solved)
    return std::nullopt;
  return solved->layoutClass;
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

/// A type's printed form, or "" for a null type.
std::string renderedType(mlir::Type type) {
  if (!type)
    return {};
  std::string text;
  llvm::raw_string_ostream stream(text);
  type.print(stream);
  return stream.str();
}

/// A node's port types, space-joined in declaration order.
std::string renderedPortTypes(llvm::ArrayRef<WorkloadPort> ports) {
  std::vector<std::string> texts;
  texts.reserve(ports.size());
  for (const WorkloadPort &port : ports)
    texts.push_back(renderedType(port.type));
  return llvm::join(texts, " ");
}

/// An attribute's printed form, or "" for a null attribute.
std::string renderedAttribute(mlir::Attribute attribute) {
  if (!attribute)
    return {};
  std::string text;
  llvm::raw_string_ostream stream(text);
  attribute.print(stream);
  return stream.str();
}

/// The instance's complete layout identity: for each bound requirement its
/// layout family and the parameters solved for it, canonically rendered and
/// sorted. Two placements that chose different parameterizations of one family
/// (`VW = 4` versus `VW = 8`) are different work and must not share a key.
std::string layoutIdentity(const CandidateInstance &instance) {
  std::vector<std::string> entries;
  entries.reserve(instance.layoutSolutions.size());
  for (const auto &entry : instance.layoutSolutions) {
    std::string text = entry.second.layoutClass;
    text += "(";
    text += canonicalSearchValueString(entry.second.parameters);
    text += ")";
    entries.push_back(std::move(text));
  }
  llvm::sort(entries);
  return llvm::join(entries, ",");
}

/// The instance's concrete placement identity: the executor it bound and its
/// sorted memory bindings. Two placements of one rule on different executors or
/// memories are different work.
std::string placementIdentity(const CandidateInstance &instance) {
  std::string text = "executor=";
  text += instance.executorBindings.lookup("executor");
  text += ",memories=";
  text += llvm::join(sortedBindings(instance.memoryBindings), ",");
  // A named-port assignment distinguishes two placements that bind the same
  // nodes to swapped occurrences, so it belongs in the placement key too.
  if (!instance.portMemoryBindings.empty()) {
    text += ",portmemories=";
    text += llvm::join(sortedPortMemories(instance.portMemoryBindings), ",");
  }
  return text;
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
  std::vector<std::string> lhsPortMemories =
      sortedPortMemories(lhs.portMemoryBindings);
  std::vector<std::string> rhsPortMemories =
      sortedPortMemories(rhs.portMemoryBindings);
  if (lhsPortMemories != rhsPortMemories)
    return lhsPortMemories < rhsPortMemories;
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

CoveringSearch::CoveringSearch(
    const WorkloadGraph &workload, const MappingTarget &target,
    mlir::MLIRContext &context, const LayoutContext &layoutContext,
    const MappingSearchOptions &options, std::optional<SearchBinding> binding,
    llvm::StringMap<std::string> boundLayouts, BoundAxes boundAxes)
    : workload_(workload), target_(target), context_(context),
      layoutContext_(layoutContext), options_(options),
      binding_(std::move(binding)), boundLayouts_(std::move(boundLayouts)),
      boundAxes_(std::move(boundAxes)) {}

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

  // The size of the tile a value carries, derived from its type. A dynamic
  // shape or a type the core does not model cannot be sized: fall back to the
  // pre-phase-3 constants and *report* it, naming the value, so an assumed size
  // is never silent again (phase-3 ruling R1).
  auto factsForValue = [&](WorkloadValueId value) -> TileFacts {
    const WorkloadValue *moved = workload_.findValue(value);
    TileFacts facts = moved ? tileFactsFor(moved->type) : TileFacts{};
    if (facts.known)
      return facts;
    report(DiagnosticCode::AssumedValueSize,
           "value " + std::to_string(value) + " ('" +
               (moved ? moved->name : std::string("<unknown>")) +
               "'): tile size unknown; assuming " +
               std::to_string(kUnknownValueBytes) + " bytes, " +
               std::to_string(kUnknownAlignment) + "-byte alignment");
    facts.bytes = kUnknownValueBytes;
    facts.alignment = kUnknownAlignment;
    return facts;
  };

  // The byte count for a node with no output to size: its first input's tile,
  // or -- for a node with no port at all -- the reported fallback. A node that
  // writes no output materializes nothing attributable, so such a charge is
  // never expired.
  auto portlessBytes = [&](const WorkloadNode &node) -> uint64_t {
    if (!node.inputs.empty())
      return factsForValue(node.inputs.front().value).bytes;
    report(DiagnosticCode::AssumedValueSize,
           "node " + std::to_string(node.id) +
               " has no port to size; assuming " +
               std::to_string(kUnknownValueBytes) + " bytes");
    return kUnknownValueBytes;
  };

  // The physical bytes an output occurrence occupies once the instance's solved
  // layout is applied: the bounding image of its layout map, not its logical
  // element count, so a padded or blocked image cannot slip under a capacity
  // check. A value the instance solved no layout for is charged its logical
  // image; a footprint that cannot be derived is *reported* and charged the
  // conservative fallback, never zero and never a logical lower bound.
  auto physicalBytesForOutput = [&](const CandidateInstance &instance,
                                    const WorkloadNode &node,
                                    unsigned index) -> uint64_t {
    const WorkloadPort &port = node.outputs[index];
    const WorkloadValue *value = workload_.findValue(port.value);
    if (!value)
      return factsForValue(port.value).bytes;
    std::vector<std::string> keys;
    for (const auto &entry : instance.layoutSolutions)
      keys.push_back(entry.first().str());
    llvm::sort(keys);
    const PortRef ref{node.id, PortDirection::Output, index};
    mlir::AffineMap map;
    for (const std::string &key : keys) {
      const SolvedLayout &solution = instance.layoutSolutions.lookup(key);
      if (solution.port && *solution.port == ref) {
        map = solution.map;
        break;
      }
    }
    if (!map) {
      for (const std::string &key : keys) {
        const SolvedLayout &solution = instance.layoutSolutions.lookup(key);
        if (solution.portValue >= 0 &&
            static_cast<WorkloadValueId>(solution.portValue) == port.value) {
          map = solution.map;
          break;
        }
      }
    }
    if (!map)
      return factsForValue(port.value).bytes;
    mlir::Type type = port.type ? port.type : value->type;
    llvm::Expected<PhysicalFootprint> footprint =
        physicalFootprintFor(type, map, {});
    if (footprint)
      return footprint->bytes;
    report(DiagnosticCode::AssumedValueSize,
           "value " + std::to_string(port.value) + " ('" + value->name +
               "'): unsupported footprint: " +
               llvm::toString(footprint.takeError()) + "; assuming " +
               std::to_string(kUnknownValueBytes) + " bytes");
    return kUnknownValueBytes;
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
    // A binding is authoritative for the rule parameters it names: each rule
    // resolves at that search-space point, so its pinned parameters take only
    // the bound value. With no binding the pointer is null and resolution is
    // exactly the pre-binding enumeration.
    const llvm::StringMap<SearchValue> *pinned =
        binding_ ? &binding_->values : nullptr;
    // The layouts the binding resolves to, one per role, resolved from the
    // space's `layout`-kind parameters by the caller. An empty map -- no
    // binding, or a binding with no layout-kind parameter -- leaves layout
    // selection unchanged.
    const llvm::StringMap<std::string> *boundLayouts =
        boundLayouts_.empty() ? nullptr : &boundLayouts_;
    // The owner_mapping/memory_path axes, resolved by parameter *kind* by the
    // caller, so a space that names the parameter differently is still
    // honoured.
    const BoundAxes *boundAxes = boundAxes_.empty() ? nullptr : &boundAxes_;
    for (const RuleDef *rule : matches) {
      std::string reason;
      bool truncated = false;
      std::optional<MappingCandidate> candidate =
          toMappingCandidate(*rule, *node, machine, layoutContext_, &reason,
                             &truncated, pinned, boundLayouts, boundAxes);
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
        InstanceEntry entry{std::move(instance), rule, {}, {}};
        entry.cost = entry.instance.localCost;
        entry.resolvedParameters = candidate->resolvedParameters;
        if (options_.enableLatencyCache) {
          if (const LatencyProvider *provider = target_.latencyProvider()) {
            OperationSignature signature;
            signature.operation = node->opName;
            // The operation's own types and attributes: two workloads on one
            // rule that differ in dtype, shape, or a predicate-relevant
            // attribute are different work at different costs.
            signature.operandTypes = renderedPortTypes(node->inputs);
            signature.resultTypes = renderedPortTypes(node->outputs);
            signature.attributes = renderedAttribute(node->attributes);
            signature.rule = rule->id;
            signature.ruleVersion = rule->version;
            signature.bundle = entry.instance.bundle.name.empty()
                                   ? rule->bundle
                                   : entry.instance.bundle.name;
            signature.bundleParameters =
                renderedAttribute(entry.instance.bundle.parameters);
            // Every bound requirement's family *and* solved parameters, not
            // merely the first family id.
            signature.layout = layoutIdentity(entry.instance);
            if (const machine::ExecutorNode *executor = machine.findExecutor(
                    entry.instance.executorBindings.lookup("executor")))
              signature.placementClass = executor->kind;
            signature.placement = placementIdentity(entry.instance);
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
    for (uint32_t portIndex = 0;
         portIndex < tables[index].workload->outputs.size(); ++portIndex)
      allLinks[linkIndexFor(tables[index].workload->outputs[portIndex].value)]
          .producers.push_back(
              {index, portIndex, &tables[index].workload->outputs[portIndex]});
    for (uint32_t portIndex = 0;
         portIndex < tables[index].workload->inputs.size(); ++portIndex)
      allLinks[linkIndexFor(tables[index].workload->inputs[portIndex].value)]
          .consumers.push_back(
              {index, portIndex, &tables[index].workload->inputs[portIndex]});
  }
  // One endpoint per *occurrence*. Deduplicating by node alone collapsed two
  // operand uses of one value into a single endpoint, so the second use lost
  // its layout obligation and only one connection was ever synthesized.
  auto canonicalEndpoints = [](std::vector<ValueEndpoint> &endpoints) {
    llvm::sort(endpoints,
               [](const ValueEndpoint &lhs, const ValueEndpoint &rhs) {
                 if (lhs.node != rhs.node)
                   return lhs.node < rhs.node;
                 return lhs.index < rhs.index;
               });
    endpoints.erase(
        std::unique(endpoints.begin(), endpoints.end(),
                    [](const ValueEndpoint &lhs, const ValueEndpoint &rhs) {
                      return lhs.node == rhs.node && lhs.index == rhs.index;
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
    // A bound memory holds the tiles this instance materializes, plus anything
    // the rule declared explicitly. `memoryBytes` is keyed by memory *node*, so
    // each byte is charged to the memory that actually holds it.
    //
    // Two association regimes. A rule may bind each output occurrence to its
    // own memory by naming the port (`require memory output "large" kind
    // dram`): the occurrence fixes the pairing, so each output is charged its
    // own bytes to its own node. Otherwise the legacy bare requirements apply,
    // and the graph fixes no output-to-binding pairing -- one binding holds
    // every output (its total, since sizing from the first output alone
    // under-charges), a single output may be charged to each binding
    // (over-charging, the safe direction), and several outputs with several
    // bindings are rejected rather than admitted on an unsafe lower bound.
    //
    // Each charged output is also recorded as a live range: the bytes are
    // released once its last consumer is placed (below), so a sequential
    // program is not charged as if every value were simultaneously live.
    const WorkloadNode &workload = *tables[nodeIndex].workload;
    if (!instance.portMemoryBindings.empty()) {
      // Explicit per-occurrence associations. Mixing them with bare
      // requirements leaves some occurrence unfixed, so the branch is rejected.
      if (!instance.memoryBindings.empty()) {
        report(
            DiagnosticCode::MemoryCapacityExceeded,
            "node " + std::to_string(workload.id) +
                ": memory requirements mix named-port and bare associations; "
                "placement is ambiguous and cannot be proven within capacity");
        return false;
      }
      bool anyOutputPort = false;
      for (const PortMemoryBinding &binding : instance.portMemoryBindings)
        anyOutputPort |= binding.port.direction == PortDirection::Output;
      if (workload.outputs.empty()) {
        // No output to attribute: charge the first-input fallback to each
        // distinct bound memory. A sink materializes nothing we can expire.
        const uint64_t bytes = portlessBytes(workload);
        std::vector<MemoryNodeId> charged;
        for (const PortMemoryBinding &binding : instance.portMemoryBindings)
          if (!llvm::is_contained(charged, binding.memory)) {
            charged.push_back(binding.memory);
            partial.memoryBytes[binding.memory] += bytes;
          }
      } else if (anyOutputPort) {
        // Every output must be associated with exactly one selected memory, and
        // charged its own bytes. An output with none, or with several, has no
        // proven capacity and rejects the branch.
        for (uint32_t index = 0; index < workload.outputs.size(); ++index) {
          const PortRef ref{workload.id, PortDirection::Output, index};
          const MemoryNodeId *memory = nullptr;
          bool ambiguous = false;
          for (const PortMemoryBinding &binding : instance.portMemoryBindings) {
            if (!(binding.port == ref))
              continue;
            if (memory) {
              ambiguous = true;
              break;
            }
            memory = &binding.memory;
          }
          if (!memory || ambiguous) {
            report(DiagnosticCode::MemoryCapacityExceeded,
                   "node " + std::to_string(workload.id) + ": output " +
                       std::to_string(index) + " has " +
                       (ambiguous ? "several memory associations"
                                  : "no memory association") +
                       "; placement is ambiguous and cannot be proven within "
                       "capacity");
            return false;
          }
          const uint64_t bytes =
              physicalBytesForOutput(instance, workload, index);
          partial.memoryBytes[*memory] += bytes;
          partial.liveValueCharges[workload.outputs[index].value].push_back(
              {*memory, bytes});
        }
      } else {
        // Every named requirement governs an *input* operand, so no output is
        // associated. The node still materializes its outputs, so they are
        // charged with the conservative bare policy over the distinct bound
        // memories -- never silently zero. (An input's own bytes are produced
        // elsewhere; an output's are written here.)
        std::vector<MemoryNodeId> memories;
        for (const PortMemoryBinding &binding : instance.portMemoryBindings)
          if (!llvm::is_contained(memories, binding.memory))
            memories.push_back(binding.memory);
        if (memories.size() == 1) {
          for (uint32_t index = 0; index < workload.outputs.size(); ++index) {
            const WorkloadPort &output = workload.outputs[index];
            const uint64_t bytes =
                physicalBytesForOutput(instance, workload, index);
            partial.memoryBytes[memories.front()] += bytes;
            partial.liveValueCharges[output.value].push_back(
                {memories.front(), bytes});
          }
        } else if (workload.outputs.size() > 1) {
          // Several outputs and several distinct input memories: no output to
          // memory pairing can be proven, so reject rather than under-charge.
          report(DiagnosticCode::MemoryCapacityExceeded,
                 "node " + std::to_string(workload.id) + ": " +
                     std::to_string(workload.outputs.size()) +
                     " outputs with " + std::to_string(memories.size()) +
                     " named input memories have no output-to-memory "
                     "association; placement is ambiguous and cannot be proven "
                     "within capacity");
          return false;
        } else {
          const WorkloadValueId first = workload.outputs.front().value;
          const uint64_t bytes = physicalBytesForOutput(instance, workload, 0);
          for (const MemoryNodeId &memory : memories) {
            partial.memoryBytes[memory] += bytes;
            partial.liveValueCharges[first].push_back({memory, bytes});
          }
        }
      }
    } else if (!instance.memoryBindings.empty()) {
      if (workload.outputs.empty()) {
        // No output to attribute: charge the first-input fallback for the whole
        // plan. A sink materializes nothing we can expire.
        const uint64_t bytes = portlessBytes(workload);
        for (const auto &binding : instance.memoryBindings)
          partial.memoryBytes[binding.second] += bytes;
      } else if (instance.memoryBindings.size() == 1) {
        const MemoryNodeId memory = instance.memoryBindings.begin()->second;
        for (uint32_t index = 0; index < workload.outputs.size(); ++index) {
          const WorkloadPort &output = workload.outputs[index];
          const uint64_t bytes =
              physicalBytesForOutput(instance, workload, index);
          partial.memoryBytes[memory] += bytes;
          partial.liveValueCharges[output.value].push_back({memory, bytes});
        }
      } else if (workload.outputs.size() > 1) {
        // Several outputs *and* several memory bindings. A bare memory
        // requirement names a kind, never the port that writes into it, so the
        // graph fixes no output-to-binding pairing: there is no sound way to
        // charge these outputs. Sizing every binding from the first output
        // alone is a lower bound that can admit an over-capacity plan -- a
        // 4-byte first output hiding a 4096-byte second -- so the branch is
        // rejected rather than admitted on an unsafe bound. A single binding
        // has no such ambiguity (one binding holds every output, charged
        // below), and a single output cannot be misattributed among bindings.
        // This is the legacy rejection an explicit named-port association is
        // what avoids.
        report(
            DiagnosticCode::MemoryCapacityExceeded,
            "node " + std::to_string(workload.id) + ": " +
                std::to_string(workload.outputs.size()) + " outputs with " +
                std::to_string(instance.memoryBindings.size()) +
                " memory bindings have no output-to-memory association; "
                "placement is ambiguous and cannot be proven within capacity");
        return false;
      } else {
        const WorkloadValueId first = workload.outputs.front().value;
        const uint64_t bytes = physicalBytesForOutput(instance, workload, 0);
        for (const auto &binding : instance.memoryBindings) {
          partial.memoryBytes[binding.second] += bytes;
          partial.liveValueCharges[first].push_back({binding.second, bytes});
        }
      }
    }
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
                           const ValueEndpoint *producerEnd,
                           const ValueEndpoint *consumerEnd,
                           WorkloadValueId value) {
      ConnectionRequest request;
      request.producer = producer.id;
      request.consumer = consumer.id;
      request.value = value;
      request.producerMemory = primaryMemory(machine, producer);
      request.consumerMemory = primaryMemory(machine, consumer);
      const WorkloadPort *producerPort =
          producerEnd ? producerEnd->port : nullptr;
      const WorkloadPort *consumerPort =
          consumerEnd ? consumerEnd->port : nullptr;
      // The endpoint occurrences, validated against the finalized graph so only
      // real ports reach the connection's identity.
      auto refFor = [&](const ValueEndpoint *endpoint,
                        PortDirection direction) -> std::optional<PortRef> {
        if (!endpoint)
          return std::nullopt;
        PortRef ref{tables[endpoint->node].node, direction, endpoint->index};
        if (!lookupPort(workload_, ref))
          return std::nullopt;
        return ref;
      };
      request.producerPort = refFor(producerEnd, PortDirection::Output);
      request.consumerPort = refFor(consumerEnd, PortDirection::Input);
      // The layout each endpoint solved for *its own occurrence*, so two
      // operand uses of one value keep their distinct obligations and a pair
      // that differs becomes a transform rather than an unattributed direct
      // connection (§15.2). Without the occurrence the lookup would give up on
      // a value with two solved classes and the transform would be
      // unreachable.
      if (request.producerPort)
        request.producerLayout =
            boundLayoutForPort(producer, *request.producerPort);
      if (request.consumerPort)
        request.consumerLayout =
            boundLayoutForPort(consumer, *request.consumerPort);
      // The concrete parameterization each endpoint solved, so a pair that
      // agrees on the class but not on its parameters is a transform rather
      // than a direct connection.
      if (request.producerPort)
        if (const SolvedLayout *solved =
                boundSolvedLayoutForPort(producer, *request.producerPort)) {
          request.producerLayoutParameters = solved->parameters;
          request.producerLayoutMap = solved->map;
        }
      if (request.consumerPort)
        if (const SolvedLayout *solved =
                boundSolvedLayoutForPort(consumer, *request.consumerPort)) {
          request.consumerLayoutParameters = solved->parameters;
          request.consumerLayoutMap = solved->map;
        }
      const TileFacts facts = factsForValue(value);
      request.bytes = facts.bytes;
      request.alignmentBytes = facts.alignment;
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
      // The partial plan's live bytes per memory, so a route that stages
      // through an already-occupied intermediate is rejected (§12.2's
      // intermediate capacity *and liveness*). The router only looks entries up
      // by node id and never iterates the map, so it cannot perturb
      // determinism. The pair's own memories are harmless to include: a route
      // never re-enters its source, and its destination is exempt.
      request.intermediateOccupancy = partial.memoryBytes;
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
    // The single place an exact-mode *choice collapse* is disclosed: when a
    // local decision has several legal alternatives, the search keeps only the
    // locally cheapest instead of branching over them. That is a restriction,
    // not a cap -- two individually cheapest routes can jointly exceed a shared
    // intermediate's capacity while more expensive direct routes would fit, so
    // a covering can be missed with no `searchTruncated` to explain it. Every
    // such collapse (plain edge, fan-out destination group, gather feed) routes
    // through here, so the flag and the stable §22.3 notice are decided in one
    // place, and a site that collapsed nothing cannot set either. `report`
    // deduplicates on (code, message), so a collapse repeated across branches
    // is one notice.
    auto discloseCollapsedChoice = [&](std::string detail) {
      if (options_.mode != SearchMode::Exact)
        return;
      result.connectionChoicesUnexplored = true;
      report(DiagnosticCode::ConnectionChoiceUnexplored, std::move(detail));
    };

    // Connections are staged locally and committed to the pool only after the
    // capacity check, so a rejected branch leaves no partial state behind.
    std::vector<ConnectionPlan> staged;
    // Bytes a connection adds beyond the instances' own tiles: a replicated
    // copy and a gather's intermediate tile are extra objects in memory, so
    // they are charged. A plain single-consumer `Transfer` is deliberately not
    // charged here: its destination tile is the consumer instance's own tile,
    // already counted through that instance's memory binding.
    //
    // Unlike the materialized values above, these bytes do *not* expire with
    // their value: the live range of a copy or gather intermediate is simply
    // not modelled yet, so it is charged for the whole partial plan. That is
    // the conservative direction -- an intermediate is assumed live from the
    // moment its copy is staged until the plan ends.
    llvm::StringMap<uint64_t> stagedBytes; // memory -> replica/gather bytes

    // Synthesizes one connection, staging the chosen alternative. When
    // `consumerPorts` is given the chosen plan serves every one of those
    // occurrences -- a single instance using the value through several equal
    // operand ports shares one connection, whose endpoint list names them all.
    auto connect = [&](const ConnectionRequest &request,
                       llvm::ArrayRef<PortRef> consumerPorts = {}) -> bool {
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
      // Several legal ways to connect this pair, but only the locally cheapest
      // is taken. In exact mode -- which otherwise implies an exhaustive joint
      // search -- that is a restriction, not a cap: two individually cheapest
      // routes through a shared intermediate can jointly exceed its capacity
      // while more expensive direct routes would fit, so a feasible covering
      // can be missed with no `searchTruncated` to explain it. Report it
      // explicitly rather than implying exhaustiveness.
      if (alternatives->size() > 1)
        discloseCollapsedChoice("connection " + request.producerMemory +
                                " -> " + request.consumerMemory +
                                ": chose the cheapest of " +
                                std::to_string(alternatives->size()) +
                                " alternatives without branching over them");
      staged.push_back(*pickBest(*alternatives));
      if (!consumerPorts.empty()) {
        staged.back().consumerPorts.assign(consumerPorts.begin(),
                                           consumerPorts.end());
        // `consumerPorts` is part of the canonical string, so the id is
        // recomputed for the shared plan.
        staged.back().id = computeConnectionId(staged.back());
      }
      cost = addCost(cost, staged.back().cost);
      // §9.3: a plain movement fills a destination buffer in the consumer's
      // memory -- the consumer's *input*, not its output tile (which the
      // consumer instance charges through its own memory binding). The two
      // sizes differ whenever the value is narrowed or reduced, so the
      // transferred bytes are charged here rather than assumed away as the
      // already-counted consumer tile. `Direct` and `LayoutTransform` move
      // nothing and charge nothing. The charge is a live range under the value,
      // so the capacity check below sees it and the link loop releases it in
      // the same placement: a transfer destination exists only while its
      // consumer runs. No aliasing with the consumer's own tile is assumed --
      // the model establishes none, and a lower charge is the unsafe direction.
      const ConnectionPlan &chosen = staged.back();
      if (chosen.kind == ConnectionKind::Transfer ||
          chosen.kind == ConnectionKind::TransferAndTransform) {
        const MemoryNodeId destination = chosen.memoryRoute.empty()
                                             ? request.consumerMemory
                                             : chosen.memoryRoute.back();
        partial.memoryBytes[destination] += request.bytes;
        partial.liveValueCharges[request.value].push_back(
            {destination, request.bytes});
      }
      return true;
    };

    // Values whose last endpoint this placement completes. Their bytes are
    // released only after the capacity check below, so a consumer and the value
    // it reads stay charged together while it is placed.
    std::vector<size_t> completedLinks;

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
      completedLinks.push_back(index);

      // The size of the value this link moves, derived once and shared by the
      // requests, the replica staging, and the gather's intermediate tile.
      const TileFacts linkFacts = factsForValue(link.value);

      // A zero-element value (a static 0 dimension) moves no bytes: it needs no
      // route, layout transform, or staging, and synthesizing a connection for
      // it would only trip the "bytes must be positive" rule. Budget it as
      // free, matching the zero capacity it is charged. The Micro tile verifier
      // rejects a 0 dimension, so this is reachable only through a modelled
      // tensor/memref/vector value type.
      if (linkFacts.bytes == 0)
        continue;

      // What makes two consumer *uses* interchangeable: same destination
      // memory, same solved representation (family and parameters), same
      // executor, and the same port type and affine relation (§10.2). Two
      // occurrences that agree on all of these can be served by one
      // connection; a difference in any is a different obligation. The key is
      // exactly what `makeRequest` reads from a use, so every member of a group
      // is equivalent to its representative by construction.
      struct ConsumerUseKey {
        MemoryNodeId memory;
        std::optional<LayoutId> layout;
        std::string layoutParameters;
        ExecutorId executor;
        mlir::Type portType;
        std::optional<mlir::AffineMap> portMap;
        bool operator==(const ConsumerUseKey &other) const {
          return memory == other.memory && layout == other.layout &&
                 layoutParameters == other.layoutParameters &&
                 executor == other.executor && portType == other.portType &&
                 portMap == other.portMap;
        }
      };
      auto useKeyFor = [&](const ValueEndpoint &consumerEnd) {
        ConsumerUseKey key;
        const CandidateInstance &consumer = *partial.chosen[consumerEnd.node];
        key.memory = primaryMemory(machine, consumer);
        PortRef ref{tables[consumerEnd.node].node, PortDirection::Input,
                    consumerEnd.index};
        if (lookupPort(workload_, ref)) {
          key.layout = boundLayoutForPort(consumer, ref);
          if (const SolvedLayout *solved =
                  boundSolvedLayoutForPort(consumer, ref))
            key.layoutParameters =
                canonicalSearchValueString(solved->parameters);
        }
        key.executor = consumer.executorBindings.lookup("executor");
        if (consumerEnd.port) {
          key.portType = consumerEnd.port->type;
          key.portMap = consumerEnd.port->accessMap;
        }
        return key;
      };
      // The occurrences of one group's value, in the order they were seen.
      auto addToUseGroup =
          [&](std::vector<std::vector<const ValueEndpoint *>> &groups,
              std::vector<ConsumerUseKey> &keys,
              const ValueEndpoint &consumerEnd) {
            ConsumerUseKey key = useKeyFor(consumerEnd);
            size_t group = 0;
            for (; group < keys.size(); ++group)
              if (keys[group] == key)
                break;
            if (group == keys.size()) {
              keys.push_back(std::move(key));
              groups.emplace_back();
            }
            groups[group].push_back(&consumerEnd);
          };

      if (link.producers.size() == 1) {
        const ValueEndpoint &producerEnd = link.producers[0];
        const CandidateInstance &producer = *partial.chosen[producerEnd.node];

        // One consumer *node* using the value through several operand ports is
        // not a fan-out: the uses belong to one instance. Group them by the
        // representation they need and synthesize one connection per group --
        // equal uses share it, incompatible uses (a plain and a blocked
        // operand, say) get their own. This is the path that keeps per-use
        // layout obligations; without it the two uses collapsed into one
        // request.
        bool distinctConsumerNodes = false;
        for (size_t i = 1; i < link.consumers.size(); ++i)
          if (link.consumers[i].node != link.consumers[0].node) {
            distinctConsumerNodes = true;
            break;
          }
        if (!distinctConsumerNodes) {
          std::vector<ConsumerUseKey> useKeys;
          std::vector<std::vector<const ValueEndpoint *>> useGroups;
          for (const ValueEndpoint &consumerEnd : link.consumers)
            addToUseGroup(useGroups, useKeys, consumerEnd);
          for (const std::vector<const ValueEndpoint *> &group : useGroups) {
            const ValueEndpoint &consumerEnd = *group.front();
            const CandidateInstance &consumer =
                *partial.chosen[consumerEnd.node];
            ConnectionRequest request = makeRequest(
                producer, consumer, &producerEnd, &consumerEnd, link.value);
            llvm::SmallVector<PortRef> consumerPorts;
            for (const ValueEndpoint *endpoint : group) {
              PortRef ref{tables[endpoint->node].node, PortDirection::Input,
                          endpoint->index};
              if (lookupPort(workload_, ref))
                consumerPorts.push_back(ref);
            }
            if (!connect(request, consumerPorts))
              return false;
          }
          continue;
        }

        // Fan-out (design §15.3). Each consumer is described by its own
        // request, so the shared-read decision and every §10.2 check see that
        // consumer's real element type, affine relation, and executor.
        std::vector<ConnectionRequest> consumerRequests;
        for (const ValueEndpoint &consumerEnd : link.consumers)
          consumerRequests.push_back(
              makeRequest(producer, *partial.chosen[consumerEnd.node],
                          &producerEnd, &consumerEnd, link.value));
        bool fanOutTruncated = false;
        bool fanOutChoseAmongAlternatives = false;
        llvm::Expected<std::vector<ConnectionPlan>> alternatives =
            synthesizeFanOut(consumerRequests.front(), consumerRequests,
                             machine, topology, placementOptions,
                             &fanOutTruncated, options_.objective,
                             &fanOutChoseAmongAlternatives);
        if (fanOutChoseAmongAlternatives)
          discloseCollapsedChoice(
              "fan-out from " + consumerRequests.front().producerMemory +
              ": chose the cheapest alternative per destination group "
              "without branching over them");
        if (alternatives)
          result.routeCount += alternatives->size();
        reportTruncation(fanOutTruncated);
        if (!alternatives) {
          report(DiagnosticCode::NoMemoryRoute,
                 "fan-out " + consumerRequests.front().producerMemory + ": " +
                     llvm::toString(alternatives.takeError()));
          return false;
        }
        if (alternatives->empty()) {
          incompatible("fan-out " + consumerRequests.front().producerMemory +
                       ": no legal route");
          return false;
        }
        // `synthesizeFanOut` has already selected the shapes: one shared read,
        // or one copy per destination memory (plus any in-place group reads).
        for (const ConnectionPlan &plan : *alternatives) {
          staged.push_back(plan);
          cost = addCost(cost, staged.back().cost);
          // A copy occupies every memory after its source: its destination and
          // any staging hop it passes through. The copy is one replica of the
          // moved value, so the value's own derived bytes size it.
          if (staged.back().kind == ConnectionKind::Replicate) {
            for (size_t hop = 1; hop < staged.back().memoryRoute.size(); ++hop)
              stagedBytes[staged.back().memoryRoute[hop]] += linkFacts.bytes;
          }
        }
        continue;
      }

      // Fan-in (design §15.3): several producers feed one value. Consumers
      // sharing a destination memory *and* the same required representation are
      // served by one gather, so the feeds and the intermediate tile are
      // counted once per representation, not once per consumer. Grouping by
      // memory alone validated the feeds against one representative and then
      // attached every consumer to the result, so a second consumer with a
      // different element type, affine relation, layout or executor was never
      // checked. `ConsumerUseKey` names exactly what `makeRequest` reads, so
      // every member of a group is equivalent to its representative by
      // construction.
      std::vector<InstanceId> producerIds;
      for (const ValueEndpoint &producerEnd : link.producers)
        producerIds.push_back(partial.chosen[producerEnd.node]->id);
      llvm::sort(producerIds);

      std::vector<ConsumerUseKey> gatherKeys;
      std::vector<std::vector<const ValueEndpoint *>> gatherGroups;
      for (const ValueEndpoint &consumerEnd : link.consumers)
        addToUseGroup(gatherGroups, gatherKeys, consumerEnd);

      for (size_t group = 0; group < gatherGroups.size(); ++group) {
        MemoryNodeId consumerMemory = gatherKeys[group].memory;
        const std::vector<const ValueEndpoint *> &groupEndpoints =
            gatherGroups[group];
        const ValueEndpoint &representative = *groupEndpoints.front();
        const CandidateInstance &consumer =
            *partial.chosen[representative.node];
        std::vector<InstanceId> consumerIds;
        llvm::SmallVector<PortRef> consumerPorts;
        for (const ValueEndpoint *endpoint : groupEndpoints) {
          consumerIds.push_back(partial.chosen[endpoint->node]->id);
          PortRef ref{tables[endpoint->node].node, PortDirection::Input,
                      endpoint->index};
          if (lookupPort(workload_, ref))
            consumerPorts.push_back(ref);
        }
        llvm::sort(consumerIds);

        Cost feedCost;
        std::vector<ExecutorId> engines;
        for (const ValueEndpoint &producerEnd : link.producers) {
          ConnectionRequest request =
              makeRequest(*partial.chosen[producerEnd.node], consumer,
                          &producerEnd, &representative, link.value);
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
          if (alternatives->size() > 1)
            discloseCollapsedChoice(
                "gather into " + consumerMemory + ": chose the cheapest of " +
                std::to_string(alternatives->size()) +
                " producer feed alternatives without branching over them");
          const ConnectionPlan *best = pickBest(*alternatives);
          feedCost = addCost(feedCost, best->cost);
          for (const ExecutorId &engine : best->transferEngines)
            if (!llvm::is_contained(engines, engine))
              engines.push_back(engine);
        }
        ConnectionPlan reduce = synthesizeFanIn(
            producerIds, consumerIds, link.value, consumerMemory,
            linkFacts.bytes, feedCost, consumerPorts);
        reduce.transferEngines.assign(engines.begin(), engines.end());
        staged.push_back(std::move(reduce));
        cost = addCost(cost, staged.back().cost);
        // The gather stages its reduced intermediate tile on the consumer.
        stagedBytes[consumerMemory] += linkFacts.bytes;
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

    // The branch fits, so the values whose last consumer this placement is have
    // now run: release the bytes they were charged. Releasing *after* the
    // capacity check keeps a consumer and the value it reads charged together
    // for the placement -- an operation reads its inputs and writes its outputs
    // in the same step, so they genuinely overlap. Where the graph fixes no
    // order, staying conservative is the rule (ruling R2): only a value whose
    // last consumer is placed expires, and its bytes are charged the whole time
    // every endpoint is still live.
    for (size_t index : completedLinks) {
      auto charges = partial.liveValueCharges.find(valueLinks[index].value);
      if (charges == partial.liveValueCharges.end())
        continue;
      for (const auto &charge : charges->second) {
        uint64_t &held = partial.memoryBytes[charge.first];
        // Subtract exactly what was charged. The charge can never exceed what
        // the memory still holds -- every release is a charge made earlier and
        // not yet released -- so `held >= charge.second` holds; `min` only
        // keeps a future accounting bug from underflowing instead of failing
        // loudly, and never masks drift by clamping a correct value.
        assert(held >= charge.second &&
               "live-value release exceeds the memory's charged bytes");
        held -= charge.second;
      }
      partial.liveValueCharges.erase(charges);
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

  // Optimistic completion cost: accumulated cost plus, for every uncovered
  // node, the componentwise best still reachable. "Best" is direction-aware --
  // a minimize objective wants the smallest reachable value, a maximize
  // objective the largest, so the bound favours the branch it bounds. The
  // per-instance value is `entry.cost` (measured-or-static), never
  // `instance.localCost`: a calibrated-down instance must lower the bound, not
  // be pruned by a stale static estimate. A node with no reachable instance
  // makes the bound infinite, which `boundIsBetterThan` always loses whichever
  // direction is declared.
  //
  // The bound omits connection costs, which are unknown until both endpoints of
  // a connection are chosen and are non-negative. That keeps it admissible for
  // minimize (omitted terms can only raise a completion); for maximize the
  // omission makes it too small, so the exact prune below is disabled unless
  // the objective minimizes.
  auto boundCost = [&](const Partial &partial) -> Cost {
    Cost total = partial.cost;
    for (size_t index = 0; index < tables.size(); ++index) {
      if (partial.chosen[index])
        continue;
      std::optional<Cost> best;
      for (const InstanceEntry &entry : tables[index].instances)
        best = best
                   ? bestCostForObjective(*best, entry.cost, options_.objective)
                   : entry.cost;
      if (!best)
        return infiniteCost();
      total = addCost(total, *best);
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
            branch.lowerBound = boundCost(branch);
            next.push_back(std::move(branch));
          }
        }
      }
      // Bound order: the more promising bound first, direction-aware. An exact
      // tie falls back to the deeper (more covered) plan, then the stable
      // partial id. The beam is a bounded heuristic that may sort on a bound
      // the exact prune would reject; it flags truncation only when a level
      // exceeds `beamWidth` and is cut below, not unconditionally.
      llvm::sort(next, [&](const Partial &lhs, const Partial &rhs) {
        if (boundIsBetterThan(lhs.lowerBound, rhs.lowerBound,
                              options_.objective))
          return true;
        if (boundIsBetterThan(rhs.lowerBound, lhs.lowerBound,
                              options_.objective))
          return false;
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
    // Best complete costs for the exact-minimize prune, ranked by the declared
    // objective (best first, so the worst kept plan is `back()`). Only the
    // prune reads it, and only a minimize objective arms that prune, so a
    // maximize or deterministic run skips this bookkeeping along with the
    // prune block below.
    std::vector<Cost> bestCosts;
    std::function<void(Partial &, bool &)> visit = [&](Partial &partial,
                                                       bool &stop) {
      if (stop)
        return;
      std::optional<size_t> node = lowestUncovered(partial);
      if (!node) {
        complete.push_back(partial);
        if (exact && options_.objective.minimize) {
          bestCosts.push_back(partial.cost);
          llvm::sort(bestCosts, [&](const Cost &lhs, const Cost &rhs) {
            return costLess(lhs, rhs, options_.objective);
          });
          if (bestCosts.size() > options_.topK)
            bestCosts.resize(options_.topK);
        }
        return;
      }
      for (const InstanceEntry &entry : tables[*node].instances) {
        if (stop)
          return;
        Partial branch = partial;
        if (!extend(branch, *node, entry))
          continue;
        // Bound-based pruning is sound only for a minimize objective. The bound
        // omits connection costs (a connection is synthesized only once both
        // endpoints are chosen), and that term is non-negative: for minimize an
        // omitted term can only raise a completion, so the bound stays at or
        // below it; for maximize the same omission makes the bound too small,
        // and a branch whose real completion would be largest can look poor.
        // Maximize therefore explores fully -- the caps (topK, instance,
        // candidate, and route caps) still bound the run and report
        // `searchTruncated`, so it never silently returns a non-best plan.
        if (exact && options_.objective.minimize) {
          branch.lowerBound = boundCost(branch);
          if (bestCosts.size() >= options_.topK &&
              !boundIsBetterThan(branch.lowerBound, bestCosts.back(),
                                 options_.objective)) {
            // A full top-K list makes this prune exact for a *strictly better*
            // cost: it cannot drop a completion strictly cheaper than the worst
            // kept plan. On an exact cost *tie* it is not exact -- the bound
            // compares against `bestCosts.back()`, which is cost-only, while
            // the final top-K trim keys on `(cost, plan.id)`, so a tie can
            // prune a plan whose smaller id the trim would have kept. Either
            // way the space was not exhausted, and the caller is told so
            // (`searchTruncated`).
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
  // Tally complete plans before the top-K cap drops the tail (design §22.2).
  result.planCount = complete.size();
  // The top-K cap, when it bites, is a search truncation like any other. Fold
  // it into the flag *before* any plan's content id is computed, because the
  // flag is part of that content (see `canonicalPlanString`); this keeps every
  // plan's id identical whether the cap was recorded before or after it was
  // built.
  if (complete.size() > options_.topK) {
    result.searchTruncated = true;
    report(DiagnosticCode::SearchTruncated,
           "top-K cap reached (topK=" + std::to_string(options_.topK) + ")");
  }

  // Build every complete plan before trimming, so each one's *exposed* content
  // id exists. The internal `Partial::id` is a search heuristic over partial
  // plans; the documented tie-break is the exposed plan id (design §22.1), so
  // the trim below must see the latter -- trimming on the partial hash would
  // keep whichever K the search happened to order first.
  std::vector<CoveringPlan> plans;
  plans.reserve(complete.size());
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
          // The resolved parameter assignment travels with the placement, so
          // the selected plan records *which* assignment generation solved and
          // verification can validate it rather than re-deriving one.
          placement.resolvedParameters = entry.resolvedParameters;
          break;
        }
      }
      placement.executor = instance->executorBindings.lookup("executor");
      placement.memories = instance->memoryBindings;
      // The per-occurrence memory assignments travel with the placement, so a
      // materializer can attribute a storage decision to the occurrence that
      // owns it.
      placement.portMemoryBindings = instance->portMemoryBindings;
      placement.layouts = instance->layoutBindings;
      // The solved parameterization travels with the binding, so a selected
      // plan says which one it chose rather than only naming the layout family.
      placement.layoutSolutions = instance->layoutSolutions;
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
      detail.consumers.assign(connection.consumers.begin(),
                              connection.consumers.end());
      llvm::sort(detail.consumers);
      // Endpoint occurrences, copied from the connection that owns them and
      // rendered sorted here if they ever join a canonical string. The
      // `consumers` instance projection cannot tell two operand uses of one
      // value apart, so this is what names the use a rewire must target.
      detail.producerPort = connection.producerPort;
      detail.consumerPorts.assign(connection.consumerPorts.begin(),
                                  connection.consumerPorts.end());
      llvm::sort(detail.consumerPorts,
                 [](const PortRef &lhs, const PortRef &rhs) {
                   if (lhs.node != rhs.node)
                     return lhs.node < rhs.node;
                   if (lhs.direction != rhs.direction)
                     return lhs.direction < rhs.direction;
                   return lhs.index < rhs.index;
                 });
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
      plan.sourceBindingCandidate = binding_->candidateId;
      plan.globalParameters = binding_->values;
    }
    plan.id = computePlanId(plan);
    plans.push_back(std::move(plan));
  }

  // §22.1: order and retain by the declared objective and then the *exposed*
  // plan id -- the documented key. The beam's `partialId` ordering stays where
  // it belongs, inside the beam's frontier heuristic; it never decides which K
  // survive a cap nor the emitted order. An exact cost tie breaks on `plan.id`,
  // so both are reproducible from `(totalCost, plan.id)` alone.
  llvm::sort(plans, [&](const CoveringPlan &lhs, const CoveringPlan &rhs) {
    return ranksBefore(lhs.totalCost, lhs.id, rhs.totalCost, rhs.id,
                       options_.objective);
  });
  if (plans.size() > options_.topK)
    plans.resize(options_.topK);
  result.plans = std::move(plans);

  // §22.1/§22.3: the frontier's codes are the stable interface, so their order
  // must not depend on the order branches happened to be explored.
  llvm::sort(result.frontier.diagnostics, diagnosticLess);
  // The canonical source-graph identity, so a report can require the same
  // pre-materialization workload before replaying the selected data.
  result.workloadHash = computeSourceGraphHash(workload_);

  return result;
}

} // namespace mlir::llk::mapping
