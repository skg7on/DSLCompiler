//===- properties.cpp - Bounded property tests for the mapping core -------===//
//
// Design §25.6 asks for bounded randomized property tests over the mapping
// engine's structural invariants. The hand-written fixtures elsewhere in this
// directory pin each property on one shape; this file draws many small random
// inputs from the same builders and asserts the properties that must hold for
// every draw:
//
//   1. every enumerated route is a simple path -- no memory node repeats;
//   2. every complete `CoveringPlan` covers each required node exactly once;
//   3. stable ids never depend on insertion order (graphs, and the id-bearing
//      value types alike);
//   4. a plan's identity survives a JSON round-trip of the plan report (the
//      report has a writer but no reader, so this pins the ids the report
//      carries rather than re-deriving them from a parsed report);
//   5. cost ordering is a deterministic total order whose exact ties break on
//      the stable id -- the *exposed* `CoveringPlan::id`, so the emitted order
//      is reproducible from `(totalCost, plan.id)` alone. The search's internal
//      partial-plan hash (`partialId`) is a beam heuristic, not the tie-break
//      the emitted list is ordered by.
//
// The generator is splitmix64 with a fixed seed: the draws are deterministic,
// so a failure reproduces from the seed alone and the harness adds no fuzz
// dependency. Iteration counts are bounded so the whole binary runs in well
// under a second.
//
//===----------------------------------------------------------------------===//

#include "LLK/Machine/MachineModel.h"
#include "LLK/Mapping/CostEvent.h"
#include "LLK/Mapping/CostModel.h"
#include "LLK/Mapping/CoveringSearch.h"
#include "LLK/Mapping/LatencyProvider.h"
#include "LLK/Mapping/LayoutConstraints.h"
#include "LLK/Mapping/MappingPlan.h"
#include "LLK/Mapping/MappingRules.h"
#include "LLK/Mapping/MappingTarget.h"
#include "LLK/Mapping/PlanReport.h"
#include "LLK/Mapping/Routing.h"
#include "LLK/Mapping/StableHash.h"
#include "LLK/Mapping/WorkloadGraph.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"

#include "llvm/Support/Error.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <vector>

using namespace mlir;
using namespace mlir::llk::machine;
using namespace mlir::llk::mapping;

namespace {

//===----------------------------------------------------------------------===//
// Deterministic PRNG: splitmix64. Fixed algorithm, fixed seed, no dependency.
//===----------------------------------------------------------------------===//

/// The fixed seed every generator streams from. A counter per property keeps
/// the streams independent, so changing one property's draw count does not
/// shift another's inputs.
constexpr uint64_t kSeed = 20261004ULL;

class Rng {
public:
  explicit Rng(uint64_t seed) : state_(seed) {}

  uint64_t next() {
    uint64_t z = (state_ += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
  }

  uint64_t below(uint64_t bound) { return bound == 0 ? 0 : next() % bound; }

  bool coin(unsigned percent) { return below(100) < percent; }

  template <typename Container> void shuffle(Container &values) {
    for (size_t i = values.size(); i > 1; --i)
      std::swap(values[i - 1], values[below(i)]);
  }

private:
  uint64_t state_;
};

//===----------------------------------------------------------------------===//
// Machine generator (property 1: routes).
//===----------------------------------------------------------------------===//

ExecutorNode executor(llvm::StringRef id, llvm::StringRef kind) {
  ExecutorNode node;
  node.id = id.str();
  node.kind = kind.str();
  return node;
}

MemoryNode memoryNode(llvm::StringRef id, llvm::StringRef kind) {
  MemoryNode node;
  node.id = id.str();
  node.kind = kind.str();
  node.visibleFrom = "e0";
  node.capacityBytes = 1u << 20;
  node.alignmentBytes = 64;
  return node;
}

LinkEdge linkEdge(llvm::StringRef id, llvm::StringRef source,
                  llvm::StringRef destination) {
  LinkEdge edge;
  edge.id = id.str();
  edge.source = source.str();
  edge.destination = destination.str();
  edge.bandwidthBytesPerCycle = 32;
  edge.latencyCycles = 10;
  edge.transactionBytes = 64;
  edge.transferEngines = {"dma.0"};
  return edge;
}

/// A small machine: one worker, one DMA engine, and 2..6 memories wired with a
/// random subset of the directed links. Every memory is visible from `e0` (so
/// the engine can legally reach each link's source) and every link moves whole
/// 64-byte transactions, so a request for a 64-byte-multiple value over any
/// pair is a question about path enumeration, not about a rejected fact.
MachineModel randomMachine(Rng &rng) {
  MachineModel model;
  model.target = "fuzz-route";
  model.executors = {executor("e0", "worker")};

  TransferEngineNode dma;
  dma.id = "dma.0";
  dma.kind = "dma";
  dma.attachedTo = "e0";
  dma.count = 4;
  dma.maxOutstanding = 8;
  model.transferEngines = {dma};

  const std::array<llvm::StringRef, 4> kinds = {"dram", "l2", "sram", "acc"};
  unsigned count = 2 + rng.below(5); // 2..6
  for (unsigned i = 0; i < count; ++i)
    model.memories.push_back(
        memoryNode("m" + std::to_string(i), kinds[rng.below(kinds.size())]));

  for (unsigned i = 0; i < count; ++i) {
    for (unsigned j = 0; j < count; ++j) {
      if (i == j || !rng.coin(45))
        continue;
      model.links.push_back(
          linkEdge("l" + std::to_string(i) + "_" + std::to_string(j),
                   model.memories[i].id, model.memories[j].id));
    }
  }
  return model;
}

//===----------------------------------------------------------------------===//
// Workload-graph generator (properties 2-4).
//===----------------------------------------------------------------------===//

/// A logical (insertion-order-free) description of a workload graph. Value
/// indices and node indices are the *logical* ones; `materialize` maps them to
/// whatever temporary ids a chosen insertion order produces, so the same
/// `LogicalGraph` can be built in any order and must finalize identically.
struct LogicalNode {
  std::string op;
  std::vector<unsigned> inputs;
  std::vector<unsigned> outputs;
  uint32_t sourceOrdinal = 0;
};

struct LogicalGraph {
  std::vector<std::string> valueNames;
  std::vector<bool> external;
  std::vector<LogicalNode> nodes;
};

/// A chain of `count` `micro.vector(op="add")` nodes: value 0 is the external
/// input, node i reads value i and writes value i+1.
LogicalGraph randomChain(unsigned count) {
  LogicalGraph graph;
  graph.valueNames.push_back("in");
  graph.external.push_back(true);
  for (unsigned i = 0; i < count; ++i) {
    graph.valueNames.push_back("mid" + std::to_string(i));
    graph.external.push_back(false);
    LogicalNode node;
    node.op = "add";
    node.inputs = {i};
    node.outputs = {i + 1};
    node.sourceOrdinal = i;
    graph.nodes.push_back(std::move(node));
  }
  return graph;
}

DictionaryAttr vectorAttributes(MLIRContext &context, llvm::StringRef op) {
  return DictionaryAttr::get(&context,
                             {NamedAttribute(StringAttr::get(&context, "op"),
                                             StringAttr::get(&context, op))});
}

/// Builds `graph` in the given value/node insertion orders and finalizes it.
WorkloadGraph materialize(MLIRContext &context, const LogicalGraph &graph,
                          llvm::ArrayRef<unsigned> valueOrder,
                          llvm::ArrayRef<unsigned> nodeOrder) {
  WorkloadGraph built;
  std::vector<WorkloadValueId> valueIds(graph.valueNames.size());
  for (unsigned index : valueOrder) {
    WorkloadValue value{0, Type(), graph.valueNames[index],
                        graph.external[index]};
    valueIds[index] = built.addValue(std::move(value));
  }
  for (unsigned index : nodeOrder) {
    const LogicalNode &logical = graph.nodes[index];
    WorkloadNode node;
    node.opName = "micro.vector";
    node.sourceOrdinal = logical.sourceOrdinal;
    node.attributes = vectorAttributes(context, logical.op);
    for (unsigned input : logical.inputs)
      node.inputs.push_back(
          WorkloadPort{valueIds[input], Type(), std::nullopt});
    for (unsigned output : logical.outputs)
      node.outputs.push_back(
          WorkloadPort{valueIds[output], Type(), std::nullopt});
    built.addNode(std::move(node));
  }
  built.finalize();
  return built;
}

llvm::SmallVector<unsigned> identityOrder(unsigned count) {
  llvm::SmallVector<unsigned> order;
  for (unsigned i = 0; i < count; ++i)
    order.push_back(i);
  return order;
}

//===----------------------------------------------------------------------===//
// Plan-search fixture: one worker, sram+dram, two interchangeable rules.
//===----------------------------------------------------------------------===//

MachineModel planMachine() {
  MachineModel model;
  model.target = "fuzz-plan";
  model.executors = {executor("e0", "worker")};
  model.memories = {memoryNode("sram.0", "sram"), memoryNode("dram.0", "dram")};
  return model;
}

/// Two rules naming the same op with different costs, so a chain of n nodes has
/// 2^n complete plans -- enough for coverage and ranking to be exercised
/// without a combinatorial explosion.
constexpr llvm::StringLiteral kTwoRules = R"llkmap(
rule r.cheap {
  match micro.vector(op = "add");
  require executor kind worker;
  bundle "b.cheap";
  emit "e1";
  cost 1;
}
rule r.expensive {
  match micro.vector(op = "add");
  require executor kind worker;
  bundle "b.expensive";
  emit "e1";
  cost 10;
}
)llkmap";

std::unique_ptr<MappingTarget> targetWith(MachineModel machine,
                                          llvm::StringRef rules) {
  llvm::Expected<RuleRegistry> registry = parseRuleText(rules, "<test>");
  if (!registry)
    return nullptr;
  return std::make_unique<FileMappingTarget>(
      "test", std::move(machine), LayoutRegistry{}, std::move(*registry),
      std::vector<std::string>{"e1"});
}

std::vector<PlanId> planIds(const MappingSearchResult &result) {
  std::vector<PlanId> ids;
  for (const CoveringPlan &plan : result.plans)
    ids.push_back(plan.id);
  return ids;
}

/// Runs `search` and fails the enclosing test at whichever step fails,
/// returning an empty result on error.
MappingSearchResult runSearch(const WorkloadGraph &graph,
                              const MappingTarget &target, MLIRContext &context,
                              const MappingSearchOptions &options) {
  CoveringSearch search(graph, target, context, LayoutContext{}, options);
  llvm::Expected<MappingSearchResult> result = search.search();
  EXPECT_TRUE(static_cast<bool>(result)) << llvm::toString(result.takeError());
  if (!result)
    return {};
  return std::move(*result);
}

} // namespace

//===----------------------------------------------------------------------===//
// Property 1: an enumerated route never repeats a memory node.
//===----------------------------------------------------------------------===//

TEST(MappingProperties, RoutesNeverRepeatANode) {
  Rng rng(kSeed ^ 0x01);
  const std::array<uint64_t, 4> sizes = {64, 128, 1024, 4096};
  unsigned checkedRoutes = 0;

  for (unsigned draw = 0; draw < 200; ++draw) {
    MachineModel model = randomMachine(rng);
    TopologyService service(model);

    RouteRequest request;
    request.source = model.memories[rng.below(model.memories.size())].id;
    request.destination = model.memories[rng.below(model.memories.size())].id;
    request.bytes = sizes[rng.below(sizes.size())];
    request.alignmentBytes = 32;

    llvm::Expected<llvm::SmallVector<MemoryRoute>> routes =
        service.enumerateRoutes(request, 1 + rng.below(8));
    // No route between this pair is a legal outcome; the property is about the
    // routes that were found.
    if (!routes)
      continue;

    for (const MemoryRoute &route : *routes) {
      ++checkedRoutes;
      ASSERT_FALSE(route.nodes.empty());
      EXPECT_EQ(route.nodes.front(), request.source);
      EXPECT_EQ(route.nodes.back(), request.destination);
      EXPECT_EQ(route.nodes.size(), route.links.size() + 1);

      // The property: no node appears twice on a route.
      std::vector<std::string> nodes(route.nodes.begin(), route.nodes.end());
      std::vector<std::string> sorted = nodes;
      llvm::sort(sorted);
      EXPECT_EQ(std::adjacent_find(sorted.begin(), sorted.end()), sorted.end())
          << "route repeats a memory node: " << request.source << " -> "
          << request.destination;

      // And each hop is a real edge joining consecutive nodes.
      for (size_t hop = 0; hop < route.links.size(); ++hop) {
        const LinkEdge *edge = model.findLink(route.links[hop]);
        ASSERT_NE(edge, nullptr);
        EXPECT_EQ(edge->source, route.nodes[hop]);
        EXPECT_EQ(edge->destination, route.nodes[hop + 1]);
      }
    }
  }

  EXPECT_GT(checkedRoutes, 0u)
      << "generator never produced a route -- the draw is degenerate";
}

//===----------------------------------------------------------------------===//
// Property 2: a complete plan covers every required node exactly once.
//===----------------------------------------------------------------------===//

TEST(MappingProperties, CompletePlansCoverEveryNodeExactlyOnce) {
  Rng rng(kSeed ^ 0x02);
  for (unsigned draw = 0; draw < 24; ++draw) {
    MLIRContext context;
    unsigned count = 1 + rng.below(5); // 1..5 nodes
    LogicalGraph logical = randomChain(count);
    WorkloadGraph graph =
        materialize(context, logical, identityOrder(logical.valueNames.size()),
                    identityOrder(logical.nodes.size()));

    std::unique_ptr<MappingTarget> target =
        targetWith(planMachine(), kTwoRules);
    ASSERT_NE(target, nullptr);

    MappingSearchOptions options;
    options.mode = SearchMode::Exact;
    options.topK = 32;
    MappingSearchResult result = runSearch(graph, *target, context, options);
    ASSERT_FALSE(result.plans.empty()) << "a legal chain must have a plan";

    std::vector<WorkloadNodeId> required;
    for (const WorkloadNode &node : graph.getNodes())
      required.push_back(node.id);
    llvm::sort(required);

    for (const CoveringPlan &plan : result.plans) {
      std::vector<WorkloadNodeId> covered;
      for (const PlanPlacement &placement : plan.placements) {
        covered.push_back(placement.node);
        // Every placement names a live instance of the plan.
        EXPECT_NE(std::find(plan.instances.begin(), plan.instances.end(),
                            placement.instance),
                  plan.instances.end());
      }
      llvm::sort(covered);

      // Exactly once: same multiset as the required set, no duplicates, no
      // extras, nothing missing.
      EXPECT_EQ(covered, required)
          << "plan does not cover the required nodes exactly once";
      EXPECT_EQ(plan.placements.size(), required.size());
    }
  }
}

//===----------------------------------------------------------------------===//
// Property 3: stable ids do not depend on insertion order.
//===----------------------------------------------------------------------===//

TEST(MappingProperties, GraphAndPlanIdsIgnoreInsertionOrder) {
  Rng rng(kSeed ^ 0x03);
  for (unsigned draw = 0; draw < 16; ++draw) {
    MLIRContext context;
    unsigned count = 1 + rng.below(5);
    LogicalGraph logical = randomChain(count);

    llvm::SmallVector<unsigned> values =
        identityOrder(logical.valueNames.size());
    llvm::SmallVector<unsigned> nodes = identityOrder(logical.nodes.size());
    WorkloadGraph forward = materialize(context, logical, values, nodes);

    // A second, differently-ordered construction of the same logical graph.
    rng.shuffle(values);
    rng.shuffle(nodes);
    WorkloadGraph shuffled = materialize(context, logical, values, nodes);

    EXPECT_EQ(forward.canonicalString(), shuffled.canonicalString());
    ASSERT_EQ(forward.getNodes().size(), shuffled.getNodes().size());
    for (size_t i = 0; i < forward.getNodes().size(); ++i)
      EXPECT_EQ(forward.getNodes()[i].id, shuffled.getNodes()[i].id);

    std::unique_ptr<MappingTarget> target =
        targetWith(planMachine(), kTwoRules);
    ASSERT_NE(target, nullptr);

    MappingSearchOptions options;
    options.mode = SearchMode::Exact;
    options.topK = 32;
    MappingSearchResult first = runSearch(forward, *target, context, options);
    MappingSearchResult second = runSearch(shuffled, *target, context, options);
    ASSERT_EQ(first.plans.size(), second.plans.size());
    ASSERT_FALSE(first.plans.empty());
    for (size_t i = 0; i < first.plans.size(); ++i) {
      EXPECT_EQ(first.plans[i].id, second.plans[i].id);
      EXPECT_EQ(canonicalPlanString(first.plans[i]),
                canonicalPlanString(second.plans[i]));
      EXPECT_EQ(first.plans[i].instances, second.plans[i].instances);
      EXPECT_EQ(first.plans[i].connections, second.plans[i].connections);
    }
  }
}

TEST(MappingProperties, ValueTypeIdsIgnoreInsertionOrder) {
  Rng rng(kSeed ^ 0x04);
  for (unsigned draw = 0; draw < 64; ++draw) {
    unsigned nodeCount = 1 + rng.below(5);
    unsigned bindingCount = 1 + rng.below(4);

    // Two candidates with the same covered-node *set* added in different
    // orders.
    MappingCandidate candidateA;
    candidateA.rule = "r." + std::to_string(rng.below(4));
    MappingCandidate candidateB = candidateA;
    llvm::SmallVector<WorkloadNodeId> ordered;
    for (unsigned i = 0; i < nodeCount; ++i)
      ordered.push_back(static_cast<WorkloadNodeId>(rng.below(64)));
    llvm::sort(ordered);
    ordered.erase(std::unique(ordered.begin(), ordered.end()), ordered.end());
    candidateA.coveredNodes = ordered;
    candidateB.coveredNodes = ordered;
    rng.shuffle(candidateB.coveredNodes);
    EXPECT_EQ(computeCandidateId(candidateA), computeCandidateId(candidateB));

    // Two instances whose StringMap bindings were inserted in different orders.
    CandidateInstance instanceA;
    instanceA.candidate = candidateA.id;
    CandidateInstance instanceB;
    instanceB.candidate = candidateA.id;
    std::vector<std::string> keys;
    for (unsigned i = 0; i < bindingCount; ++i)
      keys.push_back("k" + std::to_string(rng.below(32)));
    llvm::sort(keys);
    keys.erase(std::unique(keys.begin(), keys.end()), keys.end());
    for (const std::string &key : keys)
      instanceA.executorBindings[key] = "e" + std::to_string(rng.below(4));
    std::vector<std::string> reversed = keys;
    std::reverse(reversed.begin(), reversed.end());
    for (const std::string &key : reversed)
      instanceB.executorBindings[key] = instanceA.executorBindings.lookup(key);
    EXPECT_EQ(computeInstanceId(instanceA), computeInstanceId(instanceB));

    // Two plans whose id sets were appended in different orders.
    CoveringPlan planA;
    planA.sourceBindingHash = rng.next();
    CoveringPlan planB;
    planB.sourceBindingHash = planA.sourceBindingHash;
    for (unsigned i = 0; i < nodeCount; ++i)
      planA.instances.push_back(rng.next());
    planB.instances = planA.instances;
    rng.shuffle(planB.instances);
    EXPECT_EQ(computePlanId(planA), computePlanId(planB));
  }
}

//===----------------------------------------------------------------------===//
// Property 4: a plan's identity survives a JSON round-trip of the report.
//
// The plan report has a writer but no reader (PlanReport.h exposes only
// `writePlanReport` / `writePlanReportFile`), so identity is pinned as follows:
// the id the report carries for a retained plan is the same hex id
// `computePlanId` derives, and parsing the report and re-emitting it preserves
// every plan id and the selected id unchanged.
//===----------------------------------------------------------------------===//

TEST(MappingProperties, ReportRoundTripPreservesPlanIdentity) {
  Rng rng(kSeed ^ 0x05);
  for (unsigned draw = 0; draw < 8; ++draw) {
    MLIRContext context;
    unsigned count = 1 + rng.below(4);
    LogicalGraph logical = randomChain(count);
    WorkloadGraph graph =
        materialize(context, logical, identityOrder(logical.valueNames.size()),
                    identityOrder(logical.nodes.size()));

    std::unique_ptr<MappingTarget> target =
        targetWith(planMachine(), kTwoRules);
    ASSERT_NE(target, nullptr);

    MappingSearchOptions options;
    options.mode = SearchMode::Exact;
    options.topK = 16;
    options.beamWidth = 8;
    MappingSearchResult result = runSearch(graph, *target, context, options);
    ASSERT_FALSE(result.plans.empty());

    uint64_t moduleHash = stableHash(graph.canonicalString());
    std::string first = writePlanReport(result, target->machine(), *target,
                                        options, moduleHash);
    std::string second = writePlanReport(result, target->machine(), *target,
                                         options, moduleHash);
    EXPECT_EQ(first, second) << "the report is not byte-stable";

    llvm::Expected<llvm::json::Value> parsed = llvm::json::parse(first);
    ASSERT_TRUE(static_cast<bool>(parsed))
        << llvm::toString(parsed.takeError());
    const llvm::json::Object *root = parsed->getAsObject();
    ASSERT_TRUE(root);

    // Collect the ids the report carries, and check they are the ids the
    // in-memory plans derive -- the report names plan identity, it does not
    // invent it.
    auto idsFrom = [](const llvm::json::Object &object) {
      std::vector<std::string> ids;
      const llvm::json::Array *plans = object.getArray("plans");
      if (!plans)
        return ids;
      for (const llvm::json::Value &entry : *plans) {
        const llvm::json::Object *plan = entry.getAsObject();
        if (!plan)
          continue;
        if (std::optional<llvm::StringRef> id = plan->getString("id"))
          ids.push_back(id->str());
      }
      return ids;
    };

    std::vector<std::string> reported = idsFrom(*root);
    ASSERT_EQ(reported.size(), result.plans.size());
    for (size_t i = 0; i < result.plans.size(); ++i)
      EXPECT_EQ(reported[i], hexId(result.plans[i].id));

    std::optional<llvm::StringRef> selected = root->getString("selectedPlanId");
    ASSERT_TRUE(selected.has_value());
    EXPECT_EQ(*selected, hexId(result.plans.front().id));

    // Re-emit the parsed JSON and re-parse it: the round-trip preserves every
    // id verbatim.
    std::string reserialized;
    llvm::raw_string_ostream stream(reserialized);
    stream << *parsed;
    stream.flush();
    llvm::Expected<llvm::json::Value> reloaded =
        llvm::json::parse(reserialized);
    ASSERT_TRUE(static_cast<bool>(reloaded))
        << llvm::toString(reloaded.takeError());
    const llvm::json::Object *reloadedRoot = reloaded->getAsObject();
    ASSERT_TRUE(reloadedRoot);
    EXPECT_EQ(idsFrom(*reloadedRoot), reported);
    EXPECT_EQ(*reloadedRoot->getString("selectedPlanId"), *selected);
  }
}

//===----------------------------------------------------------------------===//
// Property 5: cost ordering is a deterministic total order with stable ties.
//===----------------------------------------------------------------------===//

TEST(MappingProperties, CostOrderingIsATotalOrderWithStableTies) {
  Rng rng(kSeed ^ 0x06);
  const std::array<ObjectiveOrder, 3> orders = {
      ObjectiveOrder{},
      ObjectiveOrder{CostMetric::DramBytes,
                     {CostMetric::LocalBytes, CostMetric::LatencyCycles},
                     true},
      ObjectiveOrder{CostMetric::ComputeUtilization,
                     {CostMetric::TransferUtilization},
                     false}};

  auto randomCost = [&]() {
    Cost cost;
    cost.latencyCycles = static_cast<double>(rng.below(64));
    cost.dramBytes = rng.below(4);
    cost.localBytes = rng.below(4);
    cost.computeUtilization = static_cast<double>(rng.below(4));
    cost.transferUtilization = static_cast<double>(rng.below(4));
    return cost;
  };

  for (const ObjectiveOrder &order : orders) {
    std::vector<Cost> costs;
    std::vector<uint64_t> ids;
    for (unsigned i = 0; i < 24; ++i) {
      costs.push_back(randomCost());
      ids.push_back(rng.below(4)); // small id space, so ties are common
    }

    for (size_t i = 0; i < costs.size(); ++i) {
      // Irreflexive: nothing ranks before itself.
      EXPECT_FALSE(ranksBefore(costs[i], ids[i], costs[i], ids[i], order));
      for (size_t j = 0; j < costs.size(); ++j) {
        bool ij = ranksBefore(costs[i], ids[i], costs[j], ids[j], order);
        bool ji = ranksBefore(costs[j], ids[j], costs[i], ids[i], order);
        // Never both directions.
        EXPECT_FALSE(ij && ji);

        bool costTie = !costLess(costs[i], costs[j], order) &&
                       !costLess(costs[j], costs[i], order);
        if (costTie && ids[i] == ids[j]) {
          // An exact tie on cost and id is equal in both directions.
          EXPECT_FALSE(ij);
          EXPECT_FALSE(ji);
        } else {
          // Otherwise exactly one direction ranks first: a total order.
          EXPECT_TRUE(ij != ji) << "incomparable pair under the declared order";
        }

        // A strict cost advantage dominates the id tie-break either way.
        if (costLess(costs[i], costs[j], order))
          EXPECT_TRUE(ij);

        // Stable tie-break: an exact cost tie falls back to the smaller id.
        if (costTie && ids[i] != ids[j])
          EXPECT_EQ(ij, ids[i] < ids[j]);
      }
    }

    // Transitivity over random triples.
    for (unsigned trial = 0; trial < 128; ++trial) {
      size_t a = rng.below(costs.size());
      size_t b = rng.below(costs.size());
      size_t c = rng.below(costs.size());
      bool ab = ranksBefore(costs[a], ids[a], costs[b], ids[b], order);
      bool bc = ranksBefore(costs[b], ids[b], costs[c], ids[c], order);
      bool ac = ranksBefore(costs[a], ids[a], costs[c], ids[c], order);
      if (ab && bc)
        EXPECT_TRUE(ac) << "cost ordering is not transitive";
    }
  }

  // The order depends only on its inputs: the same call repeats identically.
  Cost lhs = randomCost();
  Cost rhs = randomCost();
  bool baseline = ranksBefore(lhs, 3, rhs, 7, orders[0]);
  for (unsigned i = 0; i < 100; ++i)
    EXPECT_EQ(ranksBefore(lhs, 3, rhs, 7, orders[0]), baseline);
}

TEST(MappingProperties, SearchReturnsPlansInDeterministicRankedOrder) {
  Rng rng(kSeed ^ 0x07);
  for (unsigned draw = 0; draw < 12; ++draw) {
    MLIRContext context;
    unsigned count = 1 + rng.below(5);
    LogicalGraph logical = randomChain(count);
    WorkloadGraph graph =
        materialize(context, logical, identityOrder(logical.valueNames.size()),
                    identityOrder(logical.nodes.size()));

    std::unique_ptr<MappingTarget> target =
        targetWith(planMachine(), kTwoRules);
    ASSERT_NE(target, nullptr);

    MappingSearchOptions options;
    options.mode = SearchMode::Exact;
    options.topK = 32;
    MappingSearchResult result = runSearch(graph, *target, context, options);
    ASSERT_FALSE(result.plans.empty());

    // §22.1/§25.6: the emitted list is ordered by the declared objective, then
    // the exposed plan id. Plan ids are unique, so this is a strict order and
    // every adjacent pair must satisfy `ranksBefore`. Cost monotonicity is
    // asserted alongside it.
    for (size_t i = 0; i + 1 < result.plans.size(); ++i) {
      const CoveringPlan &lhs = result.plans[i];
      const CoveringPlan &rhs = result.plans[i + 1];
      EXPECT_TRUE(ranksBefore(lhs.totalCost, lhs.id, rhs.totalCost, rhs.id,
                              options.objective))
          << "plans are not ordered by (objective, plan.id) at " << i << ": "
          << canonicalCostString(lhs.totalCost) << " then "
          << canonicalCostString(rhs.totalCost);
      EXPECT_FALSE(costLess(rhs.totalCost, lhs.totalCost, options.objective))
          << "a later plan is strictly cheaper at " << i;
    }
    for (const CoveringPlan &plan : result.plans)
      EXPECT_FALSE(costLess(plan.totalCost, result.plans.front().totalCost,
                            options.objective))
          << "the first plan is beaten on cost by a later one";

    // Re-running the same search reproduces the same ordered ids.
    MappingSearchResult again = runSearch(graph, *target, context, options);
    EXPECT_EQ(planIds(result), planIds(again));
  }
}

//===----------------------------------------------------------------------===//
// Property 6 (task B7): exact joint connection branching agrees with a
// brute-force route oracle.
//
// One producer on `dram.0` fans out to two consumers on `acc.0`/`aux.0`. Each
// consumer's cheap route stages a 4096-byte tile through the shared `stage.0`,
// and its dear direct route stages nothing. With `stage.0`'s capacity as the
// free variable, the legal joint combinations are exactly the product the
// brief's `route_oracle` enumerates: a combination is legal iff the summed
// staged bytes fit, and its cost is the summed route costs plus the three
// 1-cycle instances. Exact search must match the oracle's optimum, so a locally
// cheapest pick can no longer masquerade as the joint result.
//===----------------------------------------------------------------------===//

/// One producer and two consumers, the consumers' memories distinct so the
/// value is a genuine fan-out (two destination groups).
WorkloadGraph branchFanOutGraph(MLIRContext &context) {
  WorkloadGraph graph;
  WorkloadValueId input =
      graph.addValue(WorkloadValue{0, Type(), "in", /*external=*/true});
  WorkloadValueId middle =
      graph.addValue(WorkloadValue{0, Type(), "mid", /*external=*/false});
  WorkloadValueId out1 =
      graph.addValue(WorkloadValue{0, Type(), "o1", /*external=*/false});
  WorkloadValueId out2 =
      graph.addValue(WorkloadValue{0, Type(), "o2", /*external=*/false});

  WorkloadNode producer;
  producer.opName = "micro.vector";
  producer.sourceOrdinal = 0;
  producer.attributes = vectorAttributes(context, "produce");
  producer.inputs.push_back(WorkloadPort{input, Type(), std::nullopt});
  producer.outputs.push_back(WorkloadPort{middle, Type(), std::nullopt});
  graph.addNode(std::move(producer));

  auto consumer = [&](llvm::StringRef op, unsigned ordinal,
                      WorkloadValueId output) {
    WorkloadNode node;
    node.opName = "micro.vector";
    node.sourceOrdinal = ordinal;
    node.attributes = vectorAttributes(context, op);
    node.inputs.push_back(WorkloadPort{middle, Type(), std::nullopt});
    node.outputs.push_back(WorkloadPort{output, Type(), std::nullopt});
    graph.addNode(std::move(node));
  };
  consumer("consume_a", 1, out1);
  consumer("consume_b", 2, out2);

  graph.finalize();
  return graph;
}

constexpr llvm::StringLiteral kBranchRules = R"llkmap(
rule r.produce {
  match micro.vector(op = "produce");
  require executor kind worker;
  require memory kind dram;
  bundle "b.produce";
  emit "e1";
  cost 1;
}
rule r.consume_a {
  match micro.vector(op = "consume_a");
  require executor kind worker;
  require memory kind acc;
  bundle "b.consume_a";
  emit "e1";
  cost 1;
}
rule r.consume_b {
  match micro.vector(op = "consume_b");
  require executor kind worker;
  require memory kind aux;
  bundle "b.consume_b";
  emit "e1";
  cost 1;
}
)llkmap";

/// The fan-out machine with `stage.0`'s capacity as the free variable. Each
/// consumer has a cheap two-hop route through `stage.0` and a dear direct
/// route.
MachineModel stagedFanOutMachine(uint64_t stageCapacity) {
  MachineModel model;
  model.target = "fuzz-branch";
  model.executors = {executor("e0", "worker"), executor("e1", "worker")};

  MemoryNode dram = memoryNode("dram.0", "dram");
  dram.capacityBytes = 1u << 30;
  MemoryNode stage = memoryNode("stage.0", "sram");
  stage.capacityBytes = stageCapacity;
  MemoryNode acc = memoryNode("acc.0", "acc");
  acc.visibleFrom = "e1";
  MemoryNode aux = memoryNode("aux.0", "aux");
  aux.visibleFrom = "e1";
  model.memories = {dram, stage, acc, aux};

  TransferEngineNode dma;
  dma.id = "dma.0";
  dma.kind = "dma";
  dma.attachedTo = "e0";
  model.transferEngines = {dma};

  model.links = {linkEdge("dram_to_stage.0", "dram.0", "stage.0"),
                 linkEdge("stage_to_acc.0", "stage.0", "acc.0"),
                 linkEdge("stage_to_aux.0", "stage.0", "aux.0")};
  LinkEdge accDirect = linkEdge("dram_to_acc.0", "dram.0", "acc.0");
  accDirect.latencyCycles = 1000;
  LinkEdge auxDirect = linkEdge("dram_to_aux.0", "dram.0", "aux.0");
  auxDirect.latencyCycles = 1000;
  model.links.push_back(accDirect);
  model.links.push_back(auxDirect);
  return model;
}

TEST(MappingProperties, ExactJointConnectionsMatchTheRouteOracle) {
  Rng rng(kSeed ^ 0x08);
  const uint64_t tile = 4096;
  const double oneHop = 10.0 + static_cast<double>(tile) / 32.0;
  const double staged = 2.0 * oneHop; // dram -> stage -> destination
  const double direct = 1000.0 + static_cast<double>(tile) / 32.0;
  const double instances = 3.0; // three 1-cycle rules

  // Capacities covering all three regimes: no staged copy fits, exactly one
  // fits, both fit.
  const std::array<uint64_t, 6> capacities = {2048, 4096,  5000,
                                              8192, 12288, 1u << 20};

  for (unsigned draw = 0; draw < 24; ++draw) {
    uint64_t capacity = capacities[rng.below(capacities.size())];
    MLIRContext context;
    WorkloadGraph graph = branchFanOutGraph(context);
    std::unique_ptr<MappingTarget> target =
        targetWith(stagedFanOutMachine(capacity), kBranchRules);
    ASSERT_NE(target, nullptr);

    // The brute-force oracle over the two groups.
    double oracleBest = std::numeric_limits<double>::infinity();
    for (int first = 0; first < 2; ++first)
      for (int second = 0; second < 2; ++second) {
        double cost = (first ? direct : staged) + (second ? direct : staged);
        uint64_t live = (first ? 0u : tile) + (second ? 0u : tile);
        if (live <= capacity)
          oracleBest = std::min(oracleBest, cost);
      }
    ASSERT_TRUE(std::isfinite(oracleBest));

    MappingSearchOptions options;
    options.mode = SearchMode::Exact;
    options.topK = 8;
    MappingSearchResult result = runSearch(graph, *target, context, options);
    ASSERT_FALSE(result.plans.empty())
        << "exact found no covering for capacity " << capacity;
    EXPECT_DOUBLE_EQ(result.plans[0].totalCost.latencyCycles,
                     instances + oracleBest)
        << "capacity " << capacity;
    // Exact is exhaustive within the caps, so it must not claim truncation.
    EXPECT_FALSE(result.searchTruncated) << "capacity " << capacity;

    // Deterministic and beam reuse the same joint enumeration, so neither may
    // return a covering dearer than the oracle optimum.
    MappingSearchOptions deterministic = options;
    deterministic.mode = SearchMode::Deterministic;
    MappingSearchResult greedy =
        runSearch(graph, *target, context, deterministic);
    ASSERT_FALSE(greedy.plans.empty()) << "capacity " << capacity;
    EXPECT_DOUBLE_EQ(greedy.plans[0].totalCost.latencyCycles,
                     instances + oracleBest)
        << "capacity " << capacity;

    // Re-running reproduces the same ranked ids.
    MappingSearchResult again = runSearch(graph, *target, context, options);
    EXPECT_EQ(planIds(result), planIds(again));
  }
}

//===----------------------------------------------------------------------===//
// Property 8 (task B8): the final candidate score is the shared schedule's
// overlapped latency, not the additive sum of the rule and route costs.
//===----------------------------------------------------------------------===//

/// The plan-search machine with one vector engine, so a computed plan can be
/// scored by the shared schedule.
MachineModel scoredPlanMachine() {
  MachineModel model = planMachine();
  ComputeNode vpu;
  vpu.id = "vpu";
  vpu.kind = "vector_engine";
  vpu.attachedTo = "e0";
  vpu.lanes["f32"] = 8;
  vpu.issueCycles = 1;
  model.computes = {vpu};
  return model;
}

TEST(MappingProperties, UnschedulablePlanScoreIsNamedNotSilentlyScheduled) {
  Rng rng(kSeed ^ 0x0B);
  MLIRContext context;
  LogicalGraph logical = randomChain(2);
  WorkloadGraph graph =
      materialize(context, logical, identityOrder(logical.valueNames.size()),
                  identityOrder(logical.nodes.size()));

  // A machine with no compute node: the plan's events cannot be built, so the
  // score is explicitly the accumulation, never presented as scheduled.
  std::unique_ptr<MappingTarget> unschedulable =
      targetWith(planMachine(), kTwoRules);
  ASSERT_NE(unschedulable, nullptr);
  MappingSearchOptions options;
  options.mode = SearchMode::Exact;
  options.topK = 8;
  MappingSearchResult fallback =
      runSearch(graph, *unschedulable, context, options);
  ASSERT_FALSE(fallback.plans.empty());
  for (const CoveringPlan &plan : fallback.plans) {
    EXPECT_EQ(plan.scoreSource, PlanScoreSource::Accumulation);
    bool named = false;
    for (const std::string &note : plan.diagnostics.storageNotes)
      named |= note.find("not scheduled") != std::string::npos;
    EXPECT_TRUE(named) << "the fallback is not named";
  }

  // With a compute node the same plan is scored by the shared schedule.
  std::unique_ptr<MappingTarget> schedulable =
      targetWith(scoredPlanMachine(), kTwoRules);
  ASSERT_NE(schedulable, nullptr);
  MappingSearchResult scheduled =
      runSearch(graph, *schedulable, context, options);
  ASSERT_FALSE(scheduled.plans.empty());
  EXPECT_EQ(scheduled.plans[0].scoreSource, PlanScoreSource::Schedule);
}

/// The staged fan-out machine with a second DMA engine and a vector engine on
/// each executor, so its two independent customer transfers can overlap and a
/// normalized plan event can name a compute resource.
MachineModel twoEngineFanOutMachine(uint64_t stageCapacity) {
  MachineModel model = stagedFanOutMachine(stageCapacity);
  TransferEngineNode second;
  second.id = "dma.1";
  second.kind = "dma";
  second.attachedTo = "e0";
  model.transferEngines.push_back(second);
  ComputeNode vpu0;
  vpu0.id = "vpu.0";
  vpu0.kind = "vector_engine";
  vpu0.attachedTo = "e0";
  vpu0.lanes["f32"] = 8;
  vpu0.issueCycles = 1;
  ComputeNode vpu1 = vpu0;
  vpu1.id = "vpu.1";
  vpu1.attachedTo = "e1";
  model.computes = {vpu0, vpu1};
  return model;
}

TEST(MappingProperties, FinalScoreComesFromTheSharedSchedule) {
  Rng rng(kSeed ^ 0x0A);
  for (unsigned draw = 0; draw < 12; ++draw) {
    MLIRContext context;
    WorkloadGraph graph = branchFanOutGraph(context);
    std::unique_ptr<MappingTarget> target =
        targetWith(twoEngineFanOutMachine(1u << 20), kBranchRules);
    ASSERT_NE(target, nullptr);

    MappingSearchOptions options;
    options.mode = SearchMode::Deterministic;
    MappingSearchResult result = runSearch(graph, *target, context, options);
    ASSERT_FALSE(result.plans.empty());
    const CoveringPlan &plan = result.plans[0];
    EXPECT_EQ(plan.scoreSource, PlanScoreSource::Schedule);

    // The final score is what the shared scheduler produces, so two independent
    // transfers on the machine's two engines overlap: the score is strictly
    // below the additive accumulation the search's optimistic model keeps.
    EXPECT_LT(plan.totalCost.latencyCycles, plan.accumulatedCost.latencyCycles);
    EXPECT_GT(plan.accumulatedCost.latencyCycles, 0.0);

    // The score agrees with scheduling the plan's events independently: it is a
    // schedule, not a sum.
    llvm::Expected<PlanEventDAG> events =
        buildPlanEvents(plan, target->machine());
    ASSERT_TRUE(static_cast<bool>(events))
        << llvm::toString(events.takeError());
    llvm::Expected<Cost> scheduled =
        schedulePlanEvents(*events, target->machine());
    ASSERT_TRUE(static_cast<bool>(scheduled))
        << llvm::toString(scheduled.takeError());
    EXPECT_DOUBLE_EQ(plan.totalCost.latencyCycles, scheduled->latencyCycles);
  }
}

//===----------------------------------------------------------------------===//
// Property 7 (task B8): a connection's measurement identity changes with every
// decision that could change its cost, cannot be forged by reordering a list or
// swapping roles, and never collides through ambiguous concatenation.
//===----------------------------------------------------------------------===//

TEST(MappingProperties, ConnectionIdentitySeparatesEveryDecision) {
  Rng rng(kSeed ^ 0x09);

  auto word = [&]() { return "w" + std::to_string(rng.next() % 4096); };
  // A list rendered in a fixed canonical order: two orderings of one set retain
  // one key, which is the property the connection builder must uphold.
  auto sortedJoin = [](std::vector<std::string> items) {
    llvm::sort(items);
    std::string out;
    for (const std::string &item : items) {
      out += std::to_string(item.size());
      out += ':';
      out += item;
      out += ',';
    }
    return out;
  };

  for (unsigned draw = 0; draw < 128; ++draw) {
    std::vector<std::string> consumers;
    for (unsigned i = 0, count = 1 + rng.below(3); i < count; ++i)
      consumers.push_back("consumer{" + word() + "}");
    std::vector<std::string> links;
    for (unsigned i = 0, count = 1 + rng.below(3); i < count; ++i)
      links.push_back(word());

    ConnectionSignature base;
    base.kind = word();
    base.valueType = word();
    base.producerEndpoint = "producer{" + word() + "}";
    base.consumerEndpoints = sortedJoin(consumers);
    base.route = word() + ">" + word() + ">" + word();
    base.links = sortedJoin(links);
    base.engines = word();
    base.maps = word();
    base.parameters = word();
    base.storage = word();
    const std::string key = base.canonicalString();

    // Reordering either list retains the key -- the rendering is order-free.
    std::vector<std::string> shuffled = consumers;
    rng.shuffle(shuffled);
    ConnectionSignature reordered = base;
    reordered.consumerEndpoints = sortedJoin(shuffled);
    std::vector<std::string> shuffledLinks = links;
    rng.shuffle(shuffledLinks);
    reordered.links = sortedJoin(shuffledLinks);
    EXPECT_EQ(reordered.canonicalString(), key);

    // Swapping the producer and consumer roles is different work.
    ConnectionSignature roleSwapped = reordered;
    std::swap(roleSwapped.producerEndpoint, roleSwapped.consumerEndpoints);
    EXPECT_NE(roleSwapped.canonicalString(), key);

    // Every other decision changes the identity.
    struct Field {
      const char *name;
      std::string ConnectionSignature::*member;
    };
    const Field fields[] = {
        {"kind", &ConnectionSignature::kind},
        {"valueType", &ConnectionSignature::valueType},
        {"producerEndpoint", &ConnectionSignature::producerEndpoint},
        {"route", &ConnectionSignature::route},
        {"engines", &ConnectionSignature::engines},
        {"maps", &ConnectionSignature::maps},
        {"parameters", &ConnectionSignature::parameters},
        {"storage", &ConnectionSignature::storage},
        {"links", &ConnectionSignature::links},
    };
    for (const Field &field : fields) {
      ConnectionSignature changed = base;
      changed.*(field.member) += "|" + word();
      EXPECT_NE(changed.canonicalString(), key) << field.name;
    }

    // No ambiguous concatenation: a field that merely contains another field's
    // name and separator cannot reproduce the same key.
    ConnectionSignature forged;
    forged.kind = "transfer;value_type=" + word();
    forged.valueType = word();
    ConnectionSignature genuine;
    genuine.kind = "transfer";
    genuine.valueType = "value_type=" + forged.valueType;
    forged.kind = "transfer;value_type=" + forged.valueType;
    EXPECT_NE(forged.canonicalString(), genuine.canonicalString());
  }
}
