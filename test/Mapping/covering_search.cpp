//===- covering_search.cpp - Complete-plan search (D6) -------------------===//

#include "LLK/Mapping/CoveringSearch.h"
#include "LLK/Mapping/LatencyProvider.h"

#include "llvm/Support/Error.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <memory>
#include <string>
#include <variant>
#include <vector>

using namespace mlir::llk::machine;
using namespace mlir::llk::mapping;

namespace {

MachineModel searchMachine() {
  MachineModel model;
  model.target = "search";
  model.executors = {{"e0", "worker", std::nullopt, {}, 1, {}}};
  MemoryNode sram;
  sram.id = "sram.0";
  sram.kind = "sram";
  sram.visibleFrom = "e0";
  sram.capacityBytes = 1u << 20;
  sram.alignmentBytes = 64;
  MemoryNode dram;
  dram.id = "dram.0";
  dram.kind = "dram";
  dram.visibleFrom = "e0";
  dram.capacityBytes = 1u << 30;
  dram.alignmentBytes = 64;
  model.memories = {sram, dram};
  return model;
}

/// Two interchangeable workers. With symmetry reduction off, a candidate that
/// requires a worker has two legal placements, so an instance cap of one
/// genuinely stops enumeration early rather than merely matching the count.
MachineModel twoWorkerMachine() {
  MachineModel model = searchMachine();
  model.executors = {{"e0", "worker", std::nullopt, {}, 1, {}},
                     {"e1", "worker", std::nullopt, {}, 1, {}}};
  return model;
}

/// `searchMachine` with the sram node shrunk to 5000 bytes. The search charges
/// 4096 bytes per bound instance (kAssumedValueBytes), so one instance fits the
/// node while two do not -- and two still fit the default global byte budget,
/// leaving the per-memory check as the only thing that can reject the pair.
MachineModel smallMemoryMachine() {
  MachineModel model = searchMachine();
  for (MemoryNode &memory : model.memories)
    if (memory.kind == "sram")
      memory.capacityBytes = 5000;
  return model;
}

/// A `micro.vector` with `op = "add"`, as the shipped rules predicate on.
mlir::DictionaryAttr vectorAttributes(mlir::MLIRContext &context) {
  return mlir::DictionaryAttr::get(
      &context, {mlir::NamedAttribute(mlir::StringAttr::get(&context, "op"),
                                      mlir::StringAttr::get(&context, "add"))});
}

/// producer -> consumer, both `micro.vector`.
WorkloadGraph twoNodeGraph(mlir::MLIRContext &context) {
  WorkloadGraph graph;
  WorkloadValueId input =
      graph.addValue(WorkloadValue{0, mlir::Type(), "in", /*external=*/true});
  WorkloadValueId middle =
      graph.addValue(WorkloadValue{0, mlir::Type(), "mid", /*external=*/false});
  WorkloadValueId output =
      graph.addValue(WorkloadValue{0, mlir::Type(), "out", /*external=*/false});

  WorkloadNode producer;
  producer.opName = "micro.vector";
  producer.attributes = vectorAttributes(context);
  producer.inputs.push_back(WorkloadPort{input, mlir::Type(), std::nullopt});
  producer.outputs.push_back(WorkloadPort{middle, mlir::Type(), std::nullopt});
  graph.addNode(std::move(producer));

  WorkloadNode consumer;
  consumer.opName = "micro.vector";
  consumer.attributes = vectorAttributes(context);
  consumer.inputs.push_back(WorkloadPort{middle, mlir::Type(), std::nullopt});
  consumer.outputs.push_back(WorkloadPort{output, mlir::Type(), std::nullopt});
  graph.addNode(std::move(consumer));

  graph.finalize();
  return graph;
}

constexpr llvm::StringLiteral kRules = R"llkmap(
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

constexpr llvm::StringLiteral kRulesWithMemory = R"llkmap(
rule r.cheap {
  match micro.vector(op = "add");
  require executor kind worker;
  require memory kind sram;
  bundle "b.cheap";
  emit "e1";
  cost 1;
}
)llkmap";

constexpr llvm::StringLiteral kNoPlacementRules = R"llkmap(
rule r.pe_only {
  match micro.vector(op = "add");
  require executor kind pe;
  bundle "b.pe";
  emit "e1";
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

/// A provider that answers from a table keyed by rule id, and records what it
/// was asked for.
class FixedLatencyProvider : public LatencyProvider {
public:
  std::map<std::string, double> byRule;
  mutable std::vector<std::string> lookups;
  /// How many times `lookupCycles` was entered, so a test can prove the search
  /// did (or did not) consult the provider at all.
  mutable size_t lookupCount = 0;

  std::optional<double> lookupCycles(const OperationSignature &signature,
                                     const TargetContext &) const override {
    ++lookupCount;
    lookups.push_back(signature.canonicalString());
    auto it = byRule.find(signature.rule);
    if (it == byRule.end())
      return std::nullopt;
    return it->second;
  }
};

std::unique_ptr<MappingTarget>
targetWithProvider(MachineModel machine, llvm::StringRef rules,
                   const LatencyProvider *provider) {
  llvm::Expected<RuleRegistry> registry = parseRuleText(rules, "<test>");
  if (!registry)
    return nullptr;
  return std::make_unique<FileMappingTarget>(
      "test", std::move(machine), LayoutRegistry{}, std::move(*registry),
      std::vector<std::string>{"e1"}, provider);
}

std::vector<PlanId> planIds(const MappingSearchResult &result) {
  std::vector<PlanId> ids;
  for (const CoveringPlan &plan : result.plans)
    ids.push_back(plan.id);
  return ids;
}

llvm::StringMap<SearchValue>
values(std::initializer_list<std::pair<llvm::StringRef, SearchValue>> entries) {
  llvm::StringMap<SearchValue> map;
  for (const auto &entry : entries)
    map[entry.first] = entry.second;
  return map;
}

/// The plan id the deterministic two-node fixture produced before bindings
/// were recorded. Pinned so the no-binding path cannot drift silently.
constexpr PlanId kNoBindingPlanId = 3353624054279393187ULL;

} // namespace

TEST(CoveringSearch, DeterministicReturnsTheFirstCompletePlan) {
  mlir::MLIRContext context;
  WorkloadGraph graph = twoNodeGraph(context);
  std::unique_ptr<MappingTarget> target = targetWith(searchMachine(), kRules);
  ASSERT_NE(target, nullptr);

  MappingSearchOptions options;
  options.mode = SearchMode::Deterministic;
  CoveringSearch search(graph, *target, context, LayoutContext{}, options);
  llvm::Expected<MappingSearchResult> result = search.search();
  ASSERT_TRUE(static_cast<bool>(result)) << llvm::toString(result.takeError());
  ASSERT_EQ(result->plans.size(), 1u);
  EXPECT_EQ(result->plans[0].instances.size(), 2u);
  EXPECT_EQ(result->plans[0].connections.size(), 1u);
  EXPECT_FALSE(result->searchTruncated);
}

TEST(CoveringSearch, ReportsNodesWithoutRules) {
  mlir::MLIRContext context;
  WorkloadGraph graph = twoNodeGraph(context);
  std::unique_ptr<MappingTarget> target = targetWith(searchMachine(), "");
  ASSERT_NE(target, nullptr);

  CoveringSearch search(graph, *target, context, LayoutContext{});
  llvm::Expected<MappingSearchResult> result = search.search();
  ASSERT_TRUE(static_cast<bool>(result)) << llvm::toString(result.takeError());
  EXPECT_TRUE(result->plans.empty());
  EXPECT_EQ(result->frontier.nodesWithoutRules, 2u);
  EXPECT_FALSE(result->frontier.messages.empty());
}

TEST(CoveringSearch, ReportsCandidatesWithoutPlacement) {
  mlir::MLIRContext context;
  WorkloadGraph graph = twoNodeGraph(context);
  std::unique_ptr<MappingTarget> target =
      targetWith(searchMachine(), kNoPlacementRules);
  ASSERT_NE(target, nullptr);

  CoveringSearch search(graph, *target, context, LayoutContext{});
  llvm::Expected<MappingSearchResult> result = search.search();
  ASSERT_TRUE(static_cast<bool>(result)) << llvm::toString(result.takeError());
  EXPECT_TRUE(result->plans.empty());
  EXPECT_EQ(result->frontier.candidatesWithoutPlacement, 2u);
}

TEST(CoveringSearch, WideBeamAndExactAgreeOnTheBestPlan) {
  mlir::MLIRContext context;
  WorkloadGraph graph = twoNodeGraph(context);
  std::unique_ptr<MappingTarget> target = targetWith(searchMachine(), kRules);
  ASSERT_NE(target, nullptr);

  MappingSearchOptions beamOptions;
  beamOptions.mode = SearchMode::Beam;
  beamOptions.beamWidth = 64;
  CoveringSearch beam(graph, *target, context, LayoutContext{}, beamOptions);
  llvm::Expected<MappingSearchResult> beamResult = beam.search();
  ASSERT_TRUE(static_cast<bool>(beamResult))
      << llvm::toString(beamResult.takeError());

  MappingSearchOptions exactOptions;
  exactOptions.mode = SearchMode::Exact;
  CoveringSearch exact(graph, *target, context, LayoutContext{}, exactOptions);
  llvm::Expected<MappingSearchResult> exactResult = exact.search();
  ASSERT_TRUE(static_cast<bool>(exactResult))
      << llvm::toString(exactResult.takeError());

  ASSERT_FALSE(beamResult->plans.empty());
  ASSERT_FALSE(exactResult->plans.empty());
  EXPECT_EQ(beamResult->plans[0].id, exactResult->plans[0].id);
  // The cheapest rule costs 1 per node, so the best plan is 2 (plus nothing:
  // both instances share a memory, so the connection is direct and free).
  EXPECT_DOUBLE_EQ(exactResult->plans[0].totalCost.latencyCycles, 2.0);
}

TEST(CoveringSearch, EveryPlanIsRankedAndCappedAtTopK) {
  mlir::MLIRContext context;
  WorkloadGraph graph = twoNodeGraph(context);
  std::unique_ptr<MappingTarget> target = targetWith(searchMachine(), kRules);
  ASSERT_NE(target, nullptr);

  MappingSearchOptions options;
  options.mode = SearchMode::Exact;
  options.topK = 1;
  CoveringSearch search(graph, *target, context, LayoutContext{}, options);
  llvm::Expected<MappingSearchResult> result = search.search();
  ASSERT_TRUE(static_cast<bool>(result));
  EXPECT_EQ(result->plans.size(), 1u);
  // Two rules per node give four plans; asking for one is a cap.
  EXPECT_TRUE(result->searchTruncated);
}

TEST(CoveringSearch, NarrowBeamDisclosesTruncationAndExactDoesNot) {
  mlir::MLIRContext context;
  WorkloadGraph graph = twoNodeGraph(context);
  std::unique_ptr<MappingTarget> target = targetWith(searchMachine(), kRules);
  ASSERT_NE(target, nullptr);

  MappingSearchOptions narrow;
  narrow.mode = SearchMode::Beam;
  narrow.beamWidth = 1;
  CoveringSearch beam(graph, *target, context, LayoutContext{}, narrow);
  llvm::Expected<MappingSearchResult> beamResult = beam.search();
  ASSERT_TRUE(static_cast<bool>(beamResult));
  EXPECT_TRUE(beamResult->searchTruncated);

  MappingSearchOptions exact;
  exact.mode = SearchMode::Exact;
  CoveringSearch exactSearch(graph, *target, context, LayoutContext{}, exact);
  llvm::Expected<MappingSearchResult> exactResult = exactSearch.search();
  ASSERT_TRUE(static_cast<bool>(exactResult));
  EXPECT_FALSE(exactResult->searchTruncated);
}

TEST(CoveringSearch, RepeatedRunsProduceIdenticalPlanIds) {
  mlir::MLIRContext context;
  WorkloadGraph graph = twoNodeGraph(context);
  std::unique_ptr<MappingTarget> target = targetWith(searchMachine(), kRules);
  ASSERT_NE(target, nullptr);

  MappingSearchOptions options;
  options.mode = SearchMode::Beam;
  CoveringSearch first(graph, *target, context, LayoutContext{}, options);
  CoveringSearch second(graph, *target, context, LayoutContext{}, options);
  llvm::Expected<MappingSearchResult> firstResult = first.search();
  llvm::Expected<MappingSearchResult> secondResult = second.search();
  ASSERT_TRUE(static_cast<bool>(firstResult));
  ASSERT_TRUE(static_cast<bool>(secondResult));
  EXPECT_EQ(planIds(*firstResult), planIds(*secondResult));
}

TEST(CoveringSearch, ReportsCapacityRejection) {
  mlir::MLIRContext context;
  WorkloadGraph graph = twoNodeGraph(context);
  std::unique_ptr<MappingTarget> target =
      targetWith(searchMachine(), kRulesWithMemory);
  ASSERT_NE(target, nullptr);

  MappingSearchOptions options;
  options.mode = SearchMode::Deterministic;
  options.memoryBudgetBytes = 1; // any bound memory overflows this
  CoveringSearch search(graph, *target, context, LayoutContext{}, options);
  llvm::Expected<MappingSearchResult> result = search.search();
  ASSERT_TRUE(static_cast<bool>(result)) << llvm::toString(result.takeError());
  EXPECT_TRUE(result->plans.empty());
  EXPECT_GT(result->frontier.plansRejectedByCapacity, 0u);
}

// §9.3: capacity is per memory, not one global pot. Each instance's 4096-byte
// tile fits both the 5000-byte sram node and the default byte budget; the two
// together exceed the node but not the budget, so only the per-memory check can
// reject the plan -- and it must.
TEST(CoveringSearch, PerMemoryCapacityRejectsOverSubscription) {
  mlir::MLIRContext context;
  WorkloadGraph graph = twoNodeGraph(context);
  std::unique_ptr<MappingTarget> target =
      targetWith(smallMemoryMachine(), kRulesWithMemory);
  ASSERT_NE(target, nullptr);
  // Guard the premise: one instance fits the node, two would not.
  const MemoryNode *sram = target->machine().findMemory("sram.0");
  ASSERT_NE(sram, nullptr);
  ASSERT_GE(sram->capacityBytes, 4096u);
  ASSERT_LT(sram->capacityBytes, 2u * 4096u);

  MappingSearchOptions options;
  options.mode = SearchMode::Deterministic;
  CoveringSearch search(graph, *target, context, LayoutContext{}, options);
  llvm::Expected<MappingSearchResult> result = search.search();
  ASSERT_TRUE(static_cast<bool>(result)) << llvm::toString(result.takeError());
  EXPECT_TRUE(result->plans.empty());
  EXPECT_GT(result->frontier.plansRejectedByCapacity, 0u);
}

TEST(CoveringSearch, ReportsTruncationWhenInstanceCapIsHit) {
  mlir::MLIRContext context;
  WorkloadGraph graph = twoNodeGraph(context);
  std::unique_ptr<MappingTarget> target =
      targetWith(twoWorkerMachine(), kRules);
  ASSERT_NE(target, nullptr);

  MappingSearchOptions options;
  options.mode = SearchMode::Deterministic;
  options.maxInstancesPerCandidate = 1;    // force the cap
  options.enableSymmetryReduction = false; // keep both workers legal
  CoveringSearch search(graph, *target, context, LayoutContext{}, options);
  llvm::Expected<MappingSearchResult> result = search.search();
  ASSERT_TRUE(static_cast<bool>(result)) << llvm::toString(result.takeError());
  EXPECT_TRUE(result->searchTruncated);
}

//===----------------------------------------------------------------------===//
// Measured latencies
//===----------------------------------------------------------------------===//

TEST(CoveringSearch, NoProviderLeavesTheStaticEstimate) {
  mlir::MLIRContext context;
  WorkloadGraph graph = twoNodeGraph(context);
  std::unique_ptr<MappingTarget> target = targetWith(searchMachine(), kRules);
  ASSERT_NE(target, nullptr);
  EXPECT_EQ(target->latencyProvider(), nullptr);

  MappingSearchOptions options;
  options.mode = SearchMode::Exact;
  CoveringSearch search(graph, *target, context, LayoutContext{}, options);
  llvm::Expected<MappingSearchResult> result = search.search();
  ASSERT_TRUE(static_cast<bool>(result));
  ASSERT_FALSE(result->plans.empty());
  // The cheapest declared rule is 1 per node.
  EXPECT_DOUBLE_EQ(result->plans[0].totalCost.latencyCycles, 2.0);
}

TEST(CoveringSearch, AMeasurementChangesTheCostAndTheRanking) {
  mlir::MLIRContext context;
  WorkloadGraph graph = twoNodeGraph(context);
  FixedLatencyProvider provider;
  // The declared-cheap rule is measured to be slow, and the other fast.
  provider.byRule["r.cheap"] = 100.0;
  provider.byRule["r.expensive"] = 2.0;

  std::unique_ptr<MappingTarget> target =
      targetWithProvider(searchMachine(), kRules, &provider);
  ASSERT_NE(target, nullptr);
  EXPECT_EQ(target->latencyProvider(), &provider);

  MappingSearchOptions options;
  options.mode = SearchMode::Exact;
  CoveringSearch search(graph, *target, context, LayoutContext{}, options);
  llvm::Expected<MappingSearchResult> result = search.search();
  ASSERT_TRUE(static_cast<bool>(result)) << llvm::toString(result.takeError());
  ASSERT_FALSE(result->plans.empty());
  // Two nodes at the measured 2 cycles each; the declared 1 no longer wins.
  EXPECT_DOUBLE_EQ(result->plans[0].totalCost.latencyCycles, 4.0);
  ASSERT_FALSE(result->plans[0].placements.empty());
  EXPECT_EQ(result->plans[0].placements[0].rule, "r.expensive");
  EXPECT_FALSE(provider.lookups.empty());
}

TEST(CoveringSearch, AnEntrylessProviderFallsBackToStaticCost) {
  mlir::MLIRContext context;
  WorkloadGraph graph = twoNodeGraph(context);
  FixedLatencyProvider provider; // no entries at all

  std::unique_ptr<MappingTarget> target =
      targetWithProvider(searchMachine(), kRules, &provider);
  ASSERT_NE(target, nullptr);
  MappingSearchOptions options;
  options.mode = SearchMode::Exact;
  CoveringSearch search(graph, *target, context, LayoutContext{}, options);
  llvm::Expected<MappingSearchResult> result = search.search();
  ASSERT_TRUE(static_cast<bool>(result));
  ASSERT_FALSE(result->plans.empty());
  EXPECT_DOUBLE_EQ(result->plans[0].totalCost.latencyCycles, 2.0);
  EXPECT_FALSE(provider.lookups.empty()); // it was asked, and declined
}

TEST(CoveringSearch, AnExpensiveMeasurementDoesNotMakeAPlanIllegal) {
  mlir::MLIRContext context;
  WorkloadGraph graph = twoNodeGraph(context);
  FixedLatencyProvider provider;
  provider.byRule["r.cheap"] = 1e9;
  provider.byRule["r.expensive"] = 1e9;

  std::unique_ptr<MappingTarget> target =
      targetWithProvider(searchMachine(), kRules, &provider);
  ASSERT_NE(target, nullptr);
  MappingSearchOptions options;
  options.mode = SearchMode::Exact;
  CoveringSearch search(graph, *target, context, LayoutContext{}, options);
  llvm::Expected<MappingSearchResult> result = search.search();
  ASSERT_TRUE(static_cast<bool>(result)) << llvm::toString(result.takeError());
  // Measurement changes what work costs, never what is allowed.
  EXPECT_FALSE(result->plans.empty());
}

TEST(CoveringSearch, LatencyCacheCanBeDisabled) {
  mlir::MLIRContext context;
  WorkloadGraph graph = twoNodeGraph(context);
  FixedLatencyProvider provider;
  // The declared-cheap rule is measured to be slow, and the other fast. With
  // the cache enabled this flips the ranking, exactly as the test above shows.
  provider.byRule["r.cheap"] = 100.0;
  provider.byRule["r.expensive"] = 2.0;

  std::unique_ptr<MappingTarget> target =
      targetWithProvider(searchMachine(), kRules, &provider);
  ASSERT_NE(target, nullptr);
  ASSERT_EQ(target->latencyProvider(), &provider);

  MappingSearchOptions disabled;
  disabled.mode = SearchMode::Exact;
  disabled.enableLatencyCache = false;
  CoveringSearch search(graph, *target, context, LayoutContext{}, disabled);
  llvm::Expected<MappingSearchResult> result = search.search();
  ASSERT_TRUE(static_cast<bool>(result)) << llvm::toString(result.takeError());
  ASSERT_FALSE(result->plans.empty());
  // The provider was never consulted, so the declared 1-cycle rule still wins
  // and the cost is the static estimate.
  EXPECT_EQ(provider.lookupCount, 0u);
  EXPECT_TRUE(provider.lookups.empty());
  EXPECT_DOUBLE_EQ(result->plans[0].totalCost.latencyCycles, 2.0);

  // The default (`true`) does consult it, and the measurement takes effect.
  FixedLatencyProvider measured;
  measured.byRule["r.cheap"] = 100.0;
  measured.byRule["r.expensive"] = 2.0;
  std::unique_ptr<MappingTarget> measuredTarget =
      targetWithProvider(searchMachine(), kRules, &measured);
  ASSERT_NE(measuredTarget, nullptr);

  MappingSearchOptions enabled;
  enabled.mode = SearchMode::Exact;
  CoveringSearch enabledSearch(graph, *measuredTarget, context, LayoutContext{},
                               enabled);
  llvm::Expected<MappingSearchResult> enabledResult = enabledSearch.search();
  ASSERT_TRUE(static_cast<bool>(enabledResult))
      << llvm::toString(enabledResult.takeError());
  ASSERT_FALSE(enabledResult->plans.empty());
  EXPECT_GT(measured.lookupCount, 0u);
  EXPECT_DOUBLE_EQ(enabledResult->plans[0].totalCost.latencyCycles, 4.0);
}

//===----------------------------------------------------------------------===//
// Source binding provenance
//===----------------------------------------------------------------------===//

TEST(CoveringSearch, PlansCarryTheSourceBindingHash) {
  mlir::MLIRContext context;
  WorkloadGraph graph = twoNodeGraph(context);
  std::unique_ptr<MappingTarget> target = targetWith(searchMachine(), kRules);
  ASSERT_NE(target, nullptr);

  SearchBinding binding = makeSearchBinding(
      "candidate_17",
      values({{"BM", int64_t{64}}, {"tile_layout", std::string("blocked")}}));

  MappingSearchOptions options;
  options.mode = SearchMode::Deterministic;
  CoveringSearch search(graph, *target, context, LayoutContext{}, options,
                        binding);
  llvm::Expected<MappingSearchResult> result = search.search();
  ASSERT_TRUE(static_cast<bool>(result)) << llvm::toString(result.takeError());
  ASSERT_FALSE(result->plans.empty());

  // Every emitted plan records the binding it came from (design §8.3/§9.5).
  EXPECT_EQ(result->plans[0].sourceBindingHash, binding.stableHash);
  EXPECT_NE(result->plans[0].sourceBindingHash, 0u);
  EXPECT_NE(result->plans[0].id, 0u);

  // ...and carries its parameters, so a plan is traceable to its search point.
  EXPECT_EQ(result->plans[0].globalParameters.size(), 2u);
  EXPECT_EQ(
      std::get<std::string>(result->plans[0].globalParameters["tile_layout"]),
      "blocked");
  EXPECT_EQ(std::get<int64_t>(result->plans[0].globalParameters["BM"]),
            int64_t{64});
}

TEST(CoveringSearch, DifferentBindingsYieldDifferentPlanIds) {
  mlir::MLIRContext context;
  WorkloadGraph graph = twoNodeGraph(context);
  std::unique_ptr<MappingTarget> target = targetWith(searchMachine(), kRules);
  ASSERT_NE(target, nullptr);

  SearchBinding bindingA = makeSearchBinding(
      "candidate_17",
      values({{"BM", int64_t{64}}, {"tile_layout", std::string("blocked")}}));
  SearchBinding bindingB = makeSearchBinding(
      "candidate_17",
      values({{"BM", int64_t{32}}, {"tile_layout", std::string("blocked")}}));

  MappingSearchOptions options;
  options.mode = SearchMode::Deterministic;
  CoveringSearch searchA(graph, *target, context, LayoutContext{}, options,
                         bindingA);
  CoveringSearch searchB(graph, *target, context, LayoutContext{}, options,
                         bindingB);
  llvm::Expected<MappingSearchResult> resultA = searchA.search();
  llvm::Expected<MappingSearchResult> resultB = searchB.search();
  ASSERT_TRUE(static_cast<bool>(resultA))
      << llvm::toString(resultA.takeError());
  ASSERT_TRUE(static_cast<bool>(resultB))
      << llvm::toString(resultB.takeError());
  ASSERT_FALSE(resultA->plans.empty());
  ASSERT_FALSE(resultB->plans.empty());

  // Two plans that differ only by their search point must not collide.
  EXPECT_NE(resultA->plans[0].sourceBindingHash,
            resultB->plans[0].sourceBindingHash);
  EXPECT_NE(resultA->plans[0].id, resultB->plans[0].id);
}

TEST(CoveringSearch, NoBindingLeavesTheHashZeroAndThePlanIdUnchanged) {
  mlir::MLIRContext context;
  WorkloadGraph graph = twoNodeGraph(context);
  std::unique_ptr<MappingTarget> target = targetWith(searchMachine(), kRules);
  ASSERT_NE(target, nullptr);

  MappingSearchOptions options;
  options.mode = SearchMode::Deterministic;
  CoveringSearch search(graph, *target, context, LayoutContext{}, options);
  llvm::Expected<MappingSearchResult> result = search.search();
  ASSERT_TRUE(static_cast<bool>(result)) << llvm::toString(result.takeError());
  ASSERT_FALSE(result->plans.empty());

  // The default (no binding) folds a zero hash and no parameters, so the plan
  // id is exactly what it was before bindings were recorded.
  EXPECT_EQ(result->plans[0].sourceBindingHash, 0u);
  EXPECT_TRUE(result->plans[0].globalParameters.empty());
  EXPECT_EQ(result->plans[0].id, kNoBindingPlanId);
}
