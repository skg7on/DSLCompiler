//===- plan_report.cpp - Versioned JSON plan report (§22.2) ---------------===//
//
// The plan report is reproducibility and diagnostic metadata: it names the
// exact inputs a search ran on (module, machine, layout and rule libraries),
// the options it used, how much of the space it covered, and the plans it
// retained. It must never change the IR.
//
// This test pins the report's shape and, above all, its byte-for-byte
// determinism: two runs over identical inputs have to produce the same bytes
// (§29.12), so no key may depend on iteration order, addresses, or the clock.
//
//===----------------------------------------------------------------------===//

#include "LLK/Mapping/CoveringSearch.h"
#include "LLK/Mapping/LayoutConstraints.h"
#include "LLK/Mapping/MappingRules.h"
#include "LLK/Mapping/MappingTarget.h"
#include "LLK/Mapping/PlanReport.h"
#include "LLK/Mapping/StableHash.h"
#include "LLK/Target/X86/Mapping/AVX2MappingTarget.h"

#include "LLK/Dialect/Micro/MicroDialect.h"

#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"

#include "llvm/ADT/StringSet.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"

#include <gtest/gtest.h>

#include <memory>
#include <string>
#include <vector>

using namespace mlir;
using namespace mlir::llk::mapping;

namespace avx2_mapping = mlir::llk::target::avx2;

#ifndef LLK_SOURCE_DIR
#error "LLK_SOURCE_DIR must name the repository root"
#endif

namespace {

/// Two nodes -- a staged copy and a vector add -- so the search has real work
/// to place and route.
constexpr llvm::StringLiteral kKernel = R"mlir(
module {
  micro.kernel @mapped {
    %ext = tensor.empty() : tensor<8x8xf32>
    %t, %tok = micro.async_copy %ext {src_memory = #micro.memory<dram>, dst_memory = #micro.memory<sram>} : tensor<8x8xf32> -> tensor<8x8xf32>, !micro.async_token
    %v = micro.tile_view %t {shape = array<i64: 8, 8>} : tensor<8x8xf32> -> !micro.tile<8x8xf32, memory = #micro.memory<sram>>
    %r = micro.vector "add" %v, %v : !micro.tile<8x8xf32, memory = #micro.memory<sram>>, !micro.tile<8x8xf32, memory = #micro.memory<sram>> -> !micro.tile<8x8xf32, memory = #micro.memory<sram>>
    micro.yield
  }
}
)mlir";

struct Parsed {
  std::unique_ptr<MLIRContext> context;
  OwningOpRef<ModuleOp> module;
  Operation *kernel = nullptr;
};

Parsed parseKernel(llvm::StringRef text) {
  Parsed parsed;
  parsed.context = std::make_unique<MLIRContext>();
  parsed.context->getOrLoadDialect<micro::MicroDialect>();
  parsed.context->getOrLoadDialect<tensor::TensorDialect>();
  parsed.module = parseSourceString<ModuleOp>(text, parsed.context.get());
  if (parsed.module)
    parsed.module->walk([&](Operation *op) {
      if (!parsed.kernel && op->getName().getStringRef() == "micro.kernel")
        parsed.kernel = op;
    });
  return parsed;
}

/// A search of `kernel` at the given options. Fails the test at whichever step
/// fails, returning the result.
MappingSearchResult searchKernel(MLIRContext &context, Operation *kernel,
                                 MappingTarget &target,
                                 const MappingSearchOptions &options) {
  llvm::Expected<WorkloadGraph> graph = extractWorkloadGraph(kernel);
  EXPECT_TRUE(static_cast<bool>(graph));
  if (!graph)
    return {};

  LayoutContext layoutContext;
  layoutContext.rank = 2;
  layoutContext.elementType = "f32";
  CoveringSearch search(*graph, target, context, layoutContext, options);
  llvm::Expected<MappingSearchResult> result = search.search();
  EXPECT_TRUE(static_cast<bool>(result)) << llvm::toString(result.takeError());
  if (!result)
    return {};
  return std::move(*result);
}

/// A complete, independent run: parse a fresh context and module, load a fresh
/// target from disk, search, and serialize the report. Calling it twice models
/// two separate process invocations, so comparing its outputs tests the whole
/// chain's determinism, not just `writePlanReport`'s.
std::string runReportOnce(llvm::StringRef kernel) {
  Parsed parsed = parseKernel(kernel);
  EXPECT_TRUE(parsed.module);
  if (!parsed.module)
    return {};

  llvm::Expected<std::unique_ptr<MappingTarget>> target =
      avx2_mapping::createMappingTarget(LLK_SOURCE_DIR);
  EXPECT_TRUE(static_cast<bool>(target)) << llvm::toString(target.takeError());
  if (!target)
    return {};

  MappingSearchOptions options;
  options.mode = SearchMode::Deterministic;
  MappingSearchResult result =
      searchKernel(*parsed.context, parsed.kernel, **target, options);
  return writePlanReport(result, (**target).machine(), **target, options,
                         stableHash(kernel));
}

} // namespace

// The report carries every field §22.2 names, and two independent runs are
// byte-identical.
TEST(MappingPlanReportTest, EmitsRequiredFieldsAndIsByteIdentical) {
  // Two full runs -- fresh context, fresh target loaded from disk -- must
  // agree byte for byte. This is the §29.12 contract; two calls on one
  // in-memory result could not catch search-level or cross-process
  // nondeterminism.
  std::string first = runReportOnce(kKernel);
  std::string second = runReportOnce(kKernel);
  ASSERT_FALSE(first.empty());
  EXPECT_EQ(first, second);

  llvm::Expected<llvm::json::Value> json = llvm::json::parse(first);
  ASSERT_TRUE(static_cast<bool>(json)) << llvm::toString(json.takeError());
  const llvm::json::Object *root = json->getAsObject();
  ASSERT_TRUE(root);

  EXPECT_TRUE(root->getInteger("version").has_value());
  EXPECT_EQ(*root->getInteger("version"), 2);
  EXPECT_TRUE(root->getString("compilerVersion").has_value());
  EXPECT_NE(*root->getString("compilerVersion"), "llk-compiler");
  EXPECT_TRUE(root->getInteger("costModelVersion").has_value());
  EXPECT_TRUE(root->getString("inputModuleHash").has_value());
  EXPECT_TRUE(root->getString("sourceBindingHash").has_value());
  // The candidate symbol is emitted even for a binding-free search, where it is
  // empty -- a fixed field set is part of the schema, not an omission.
  EXPECT_TRUE(root->getString("sourceBindingCandidate").has_value());
  EXPECT_EQ(*root->getString("sourceBindingCandidate"), "");
  EXPECT_TRUE(root->getString("sourceBinding").has_value());
  EXPECT_TRUE(root->getString("machineHash").has_value());
  EXPECT_TRUE(root->getString("layoutLibraryHash").has_value());
  EXPECT_TRUE(root->getString("ruleLibraryHash").has_value());
  EXPECT_TRUE(root->getBoolean("searchTruncated").has_value());
  EXPECT_TRUE(root->getString("selectedPlanId").has_value());

  // The input hash names the module the caller supplied.
  EXPECT_EQ(*root->getString("inputModuleHash"), hexId(stableHash(kKernel)));

  // Search options are recorded, not just their effect.
  const llvm::json::Object *searchOptions = root->getObject("searchOptions");
  ASSERT_TRUE(searchOptions);
  EXPECT_EQ(*searchOptions->getString("mode"), "deterministic");
  EXPECT_TRUE(searchOptions->getInteger("topK").has_value());

  // Counts are present and cohere with the retained plans.
  const llvm::json::Object *counts = root->getObject("counts");
  ASSERT_TRUE(counts);
  EXPECT_TRUE(counts->getInteger("candidates").has_value());
  EXPECT_TRUE(counts->getInteger("instances").has_value());
  EXPECT_TRUE(counts->getInteger("routes").has_value());
  EXPECT_TRUE(counts->getInteger("plans").has_value());
  EXPECT_GE(*counts->getInteger("instances"), 1);

  // Both coded-event arrays exist; each is grouped by code, one entry each.
  ASSERT_TRUE(root->getArray("rejections"));
  ASSERT_TRUE(root->getArray("notices"));

  // Top-K plans with their component costs.
  const llvm::json::Array *plans = root->getArray("plans");
  ASSERT_TRUE(plans);
  ASSERT_FALSE(plans->empty());
  EXPECT_GE(*counts->getInteger("plans"), static_cast<int64_t>(plans->size()));
  const llvm::json::Object *best = (*plans)[0].getAsObject();
  ASSERT_TRUE(best);
  EXPECT_TRUE(best->getString("totalCost").has_value());
  const llvm::json::Object *components = best->getObject("costComponents");
  ASSERT_TRUE(components);
  EXPECT_TRUE(components->getString("latencyCycles").has_value());
  EXPECT_TRUE(components->getInteger("dramBytes").has_value());
  EXPECT_TRUE(components->getString("computeUtilization").has_value());

  // The selected plan is the best plan.
  EXPECT_EQ(*root->getString("selectedPlanId"), *best->getString("id"));
}

// Rejections are grouped by code, one entry per code, counts positive -- and a
// truncation is a *notice*, not a rejection, so it never inflates the tally.
TEST(MappingPlanReportTest, RejectionCountsAreGroupedByCode) {
  Parsed parsed = parseKernel(kKernel);
  ASSERT_TRUE(parsed.module);

  llvm::Expected<std::unique_ptr<MappingTarget>> target =
      avx2_mapping::createMappingTarget(LLK_SOURCE_DIR);
  ASSERT_TRUE(static_cast<bool>(target)) << llvm::toString(target.takeError());

  // A zero candidate cap rejects every node, so the search records the same
  // stable codes on both nodes: the report must group them into one entry per
  // code, with the occurrence count, not one entry per event.
  MappingSearchOptions options;
  options.mode = SearchMode::Beam;
  options.topK = 1;
  options.maxCandidatesPerNode = 0;
  MappingSearchResult result =
      searchKernel(*parsed.context, parsed.kernel, **target, options);

  ASSERT_FALSE(result.frontier.codeCounts.empty());
  std::string report = writePlanReport(result, (**target).machine(), **target,
                                       options, stableHash(kKernel));
  llvm::Expected<llvm::json::Value> json = llvm::json::parse(report);
  ASSERT_TRUE(static_cast<bool>(json)) << llvm::toString(json.takeError());
  const llvm::json::Object *root = json->getAsObject();
  ASSERT_TRUE(root);

  const llvm::json::Array *rejections = root->getArray("rejections");
  ASSERT_TRUE(rejections);
  ASSERT_FALSE(rejections->empty());

  llvm::StringSet<> seen;
  bool sawNoMatchingRule = false;
  for (const llvm::json::Value &entry : *rejections) {
    const llvm::json::Object *object = entry.getAsObject();
    ASSERT_TRUE(object);
    std::optional<llvm::StringRef> code = object->getString("code");
    std::optional<int64_t> count = object->getInteger("count");
    ASSERT_TRUE(code.has_value());
    ASSERT_TRUE(count.has_value());
    EXPECT_GT(*count, 0);
    EXPECT_TRUE(seen.insert(*code).second) << "duplicate code " << code->str();
    if (*code == "search_truncated")
      ADD_FAILURE() << "a cap is a notice, not a rejection";
    if (*code == "no_matching_rule")
      sawNoMatchingRule = true;
  }
  EXPECT_TRUE(sawNoMatchingRule);

  // The truncation is reported, but under notices.
  const llvm::json::Array *notices = root->getArray("notices");
  ASSERT_TRUE(notices);
  bool sawTruncated = false;
  for (const llvm::json::Value &entry : *notices) {
    const llvm::json::Object *object = entry.getAsObject();
    ASSERT_TRUE(object);
    if (object->getString("code") == "search_truncated")
      sawTruncated = true;
  }
  EXPECT_TRUE(sawTruncated);
}

// Each stable code belongs to exactly one bucket, and that membership is a
// property of the code, not of the search that produced it -- so it is pinned
// per code here. In particular `assumed_value_size` is advice (the search
// sized a value it could not derive), never a refusal, so it must not inflate
// §22.2's rejection tally even on a run that reaches a plan.
TEST(MappingPlanReportTest, AssumedValueSizeIsANoticeNotARejection) {
  llvm::Expected<std::unique_ptr<MappingTarget>> target =
      avx2_mapping::createMappingTarget(LLK_SOURCE_DIR);
  ASSERT_TRUE(static_cast<bool>(target)) << llvm::toString(target.takeError());

  MappingSearchResult result;
  result.frontier.codeCounts[DiagnosticCode::AssumedValueSize] = 3;
  result.frontier.codeCounts[DiagnosticCode::MemoryCapacityExceeded] = 1;
  result.frontier.codeCounts[DiagnosticCode::SearchTruncated] = 2;
  result.frontier.codeCounts[DiagnosticCode::LatencyCacheMiss] = 4;

  MappingSearchOptions options;
  std::string report = writePlanReport(result, (**target).machine(), **target,
                                       options, stableHash("assumed"));
  llvm::Expected<llvm::json::Value> json = llvm::json::parse(report);
  ASSERT_TRUE(static_cast<bool>(json)) << llvm::toString(json.takeError());
  const llvm::json::Object *root = json->getAsObject();
  ASSERT_TRUE(root);

  auto codesIn = [](const llvm::json::Array *array) {
    llvm::StringSet<> codes;
    for (const llvm::json::Value &entry : *array)
      if (const llvm::json::Object *object = entry.getAsObject())
        if (std::optional<llvm::StringRef> code = object->getString("code"))
          codes.insert(*code);
    return codes;
  };
  const llvm::json::Array *rejections = root->getArray("rejections");
  const llvm::json::Array *notices = root->getArray("notices");
  ASSERT_TRUE(rejections);
  ASSERT_TRUE(notices);
  llvm::StringSet<> rejected = codesIn(rejections);
  llvm::StringSet<> noticed = codesIn(notices);

  EXPECT_EQ(rejected.find("assumed_value_size"), rejected.end());
  EXPECT_NE(noticed.find("assumed_value_size"), noticed.end());
  EXPECT_NE(rejected.find("memory_capacity_exceeded"), rejected.end());
  EXPECT_NE(noticed.find("search_truncated"), noticed.end());
  EXPECT_NE(noticed.find("latency_cache_miss"), noticed.end());
}

// A collapsed connection choice is a notice, never a rejection: like a
// truncation or an assumed size, `connection_choice_unexplored` belongs under
// `notices` and must not inflate §22.2's rejected tally. A restricted search is
// not a search that refused the plan, so reporting it as a rejection would make
// a feasible-but-collapsed run look like a failure.
TEST(MappingPlanReportTest, ConnectionChoiceUnexploredIsANoticeNotARejection) {
  llvm::Expected<std::unique_ptr<MappingTarget>> target =
      avx2_mapping::createMappingTarget(LLK_SOURCE_DIR);
  ASSERT_TRUE(static_cast<bool>(target)) << llvm::toString(target.takeError());

  MappingSearchResult result;
  result.connectionChoicesUnexplored = true;
  result.frontier.codeCounts[DiagnosticCode::ConnectionChoiceUnexplored] = 1;
  result.frontier.codeCounts[DiagnosticCode::MemoryCapacityExceeded] = 2;

  MappingSearchOptions options;
  std::string report = writePlanReport(result, (**target).machine(), **target,
                                       options, stableHash("collapse"));
  llvm::Expected<llvm::json::Value> json = llvm::json::parse(report);
  ASSERT_TRUE(static_cast<bool>(json)) << llvm::toString(json.takeError());
  const llvm::json::Object *root = json->getAsObject();
  ASSERT_TRUE(root);

  auto codesIn = [](const llvm::json::Array *array) {
    llvm::StringSet<> codes;
    for (const llvm::json::Value &entry : *array)
      if (const llvm::json::Object *object = entry.getAsObject())
        if (std::optional<llvm::StringRef> code = object->getString("code"))
          codes.insert(*code);
    return codes;
  };
  const llvm::json::Array *rejections = root->getArray("rejections");
  const llvm::json::Array *notices = root->getArray("notices");
  ASSERT_TRUE(rejections);
  ASSERT_TRUE(notices);
  llvm::StringSet<> rejected = codesIn(rejections);
  llvm::StringSet<> noticed = codesIn(notices);

  EXPECT_EQ(rejected.find("connection_choice_unexplored"), rejected.end());
  EXPECT_NE(noticed.find("connection_choice_unexplored"), noticed.end());
  EXPECT_NE(rejected.find("memory_capacity_exceeded"), rejected.end());
  EXPECT_EQ(noticed.find("memory_capacity_exceeded"), noticed.end());
}

// The library hashes are content-derived and stable, and differ between the
// two registries (a layout library is not a rule library).
TEST(MappingPlanReportTest, RegistryHashesAreContentDerivedAndStable) {
  llvm::Expected<std::unique_ptr<MappingTarget>> target =
      avx2_mapping::createMappingTarget(LLK_SOURCE_DIR);
  ASSERT_TRUE(static_cast<bool>(target)) << llvm::toString(target.takeError());

  uint64_t layoutHash = (**target).layouts().computeContentHash();
  uint64_t ruleHash = (**target).rules().computeContentHash();
  EXPECT_NE(layoutHash, 0u);
  EXPECT_NE(ruleHash, 0u);
  EXPECT_NE(layoutHash, ruleHash);
  EXPECT_EQ(layoutHash, (**target).layouts().computeContentHash());
}

// A library hash is content-derived: changing or adding a declaration changes
// it. A constant hash would pass the stability checks above, so this pins the
// property those checks cannot.
TEST(MappingPlanReportTest, RegistryHashChangesWithContent) {
  std::string error;

  // Same id, different body; and a different id. Both must move the hash.
  LayoutRegistry base;
  LayoutDef layout;
  layout.id = "test.layout";
  ASSERT_TRUE(base.add(layout, error)) << error;

  LayoutRegistry renamed;
  LayoutDef renamedDef = layout;
  renamedDef.id = "test.layout.other";
  ASSERT_TRUE(renamed.add(renamedDef, error)) << error;

  LayoutRegistry parameterized;
  LayoutDef parameterizedDef = layout;
  parameterizedDef.params.push_back(LayoutParam{"VW", /*symbolic=*/false});
  ASSERT_TRUE(parameterized.add(parameterizedDef, error)) << error;

  EXPECT_NE(base.computeContentHash(), renamed.computeContentHash());
  EXPECT_NE(base.computeContentHash(), parameterized.computeContentHash());
  EXPECT_EQ(base.computeContentHash(), base.computeContentHash());

  RuleRegistry ruleBase;
  RuleDef rule;
  rule.id = "test.rule";
  rule.matchOp = "micro.vector";
  ASSERT_TRUE(ruleBase.add(rule, error)) << error;

  RuleRegistry ruleChanged;
  RuleDef changedRule = rule;
  changedRule.bundle = "bundle.changed";
  ASSERT_TRUE(ruleChanged.add(changedRule, error)) << error;

  EXPECT_NE(ruleBase.computeContentHash(), ruleChanged.computeContentHash());
  EXPECT_EQ(ruleBase.computeContentHash(), ruleBase.computeContentHash());
}

//===----------------------------------------------------------------------===//
// Versioned replay (task B1)
//===----------------------------------------------------------------------===//

// A v2 report round-trips the selected state: reading it back reconstructs the
// same plan identity, endpoints, routes and solved-layout assignments.
TEST(MappingPlanReportTest, ReplayReportRoundTripsSelectedState) {
  Parsed parsed = parseKernel(kKernel);
  ASSERT_TRUE(parsed.module);
  llvm::Expected<std::unique_ptr<MappingTarget>> target =
      avx2_mapping::createMappingTarget(LLK_SOURCE_DIR);
  ASSERT_TRUE(static_cast<bool>(target)) << llvm::toString(target.takeError());

  MappingSearchOptions options;
  options.mode = SearchMode::Deterministic;
  MappingSearchResult result =
      searchKernel(*parsed.context, parsed.kernel, **target, options);
  ASSERT_FALSE(result.plans.empty());
  const CoveringPlan &selected = result.plans.front();

  std::string report = writePlanReport(result, (**target).machine(), **target,
                                       options, stableHash(kKernel));
  llvm::Expected<WorkloadGraph> graph = extractWorkloadGraph(parsed.kernel);
  ASSERT_TRUE(static_cast<bool>(graph)) << llvm::toString(graph.takeError());

  llvm::Expected<CoveringPlan> replay =
      readPlanReport(report, **target, *graph);
  ASSERT_TRUE(static_cast<bool>(replay)) << llvm::toString(replay.takeError());
  EXPECT_EQ(replay->id, selected.id);
  EXPECT_EQ(replay->schemaVersion, 2u);
  EXPECT_EQ(replay->sourceBindingHash, selected.sourceBindingHash);

  ASSERT_EQ(replay->placements.size(), selected.placements.size());
  for (size_t i = 0; i < selected.placements.size(); ++i) {
    EXPECT_EQ(replay->placements[i].node, selected.placements[i].node);
    EXPECT_EQ(replay->placements[i].instance, selected.placements[i].instance);
    EXPECT_EQ(replay->placements[i].rule, selected.placements[i].rule);
    EXPECT_EQ(replay->placements[i].executor, selected.placements[i].executor);
    EXPECT_EQ(replay->placements[i].layoutSolutions.size(),
              selected.placements[i].layoutSolutions.size());
    // An `AffineMap` is context-bound and a replay re-derives it from the
    // parameters; the report still records its printed form, which this checks.
    for (const auto &entry : selected.placements[i].layoutSolutions) {
      ASSERT_TRUE(replay->placements[i].layoutSolutions.count(entry.first()));
      EXPECT_EQ(
          canonicalSearchValueString(replay->placements[i]
                                         .layoutSolutions.lookup(entry.first())
                                         .parameters),
          canonicalSearchValueString(entry.second.parameters));
      EXPECT_EQ(
          replay->placements[i].layoutSolutions.lookup(entry.first()).port,
          entry.second.port);
    }
  }
  ASSERT_EQ(replay->connectionPlans.size(), selected.connectionPlans.size());
  for (size_t i = 0; i < selected.connectionPlans.size(); ++i) {
    EXPECT_EQ(replay->connectionPlans[i].id, selected.connectionPlans[i].id);
    EXPECT_EQ(replay->connectionPlans[i].kind,
              selected.connectionPlans[i].kind);
    EXPECT_EQ(replay->connectionPlans[i].route,
              selected.connectionPlans[i].route);
    EXPECT_EQ(replay->connectionPlans[i].consumerPorts,
              selected.connectionPlans[i].consumerPorts);
    EXPECT_EQ(replay->connectionPlans[i].producerPort,
              selected.connectionPlans[i].producerPort);
  }

  // The report persists each solved layout's concrete map in its printed form,
  // which is what a materializer re-derives the map from.
  auto renderMap = [](mlir::AffineMap map) {
    std::string text;
    llvm::raw_string_ostream stream(text);
    map.print(stream);
    return stream.str();
  };
  for (const PlanPlacement &placement : selected.placements)
    for (const auto &entry : placement.layoutSolutions)
      if (entry.second.map)
        EXPECT_NE(report.find(renderMap(entry.second.map)), std::string::npos);
}

// A report round-trips a gather connection's declared combination semantics,
// its concat axis, and the producer occurrences it combines (task I2). Without
// them a replayed `Reduce` connection degrades to `reduce_not_materialized`
// even though the report states what the gather means.
TEST(MappingPlanReportTest, ReplayRoundTripsGatherSemantics) {
  Parsed parsed = parseKernel(kKernel);
  ASSERT_TRUE(parsed.module);
  llvm::Expected<std::unique_ptr<MappingTarget>> target =
      avx2_mapping::createMappingTarget(LLK_SOURCE_DIR);
  ASSERT_TRUE(static_cast<bool>(target)) << llvm::toString(target.takeError());

  MappingSearchOptions options;
  options.mode = SearchMode::Deterministic;
  MappingSearchResult searched =
      searchKernel(*parsed.context, parsed.kernel, **target, options);
  ASSERT_FALSE(searched.plans.empty());

  llvm::Expected<WorkloadGraph> graph = extractWorkloadGraph(parsed.kernel);
  ASSERT_TRUE(static_cast<bool>(graph)) << llvm::toString(graph.takeError());

  // Two resolvable producer occurrences to feed the gather, taken from the
  // source graph so the reader's endpoint-resolution check accepts them.
  std::vector<PortRef> producers;
  for (const WorkloadNode &node : graph->getNodes())
    for (uint32_t index = 0; index < node.outputs.size(); ++index)
      producers.push_back(PortRef{node.id, PortDirection::Output, index});
  ASSERT_GE(producers.size(), 2u);

  MappingSearchResult result;
  result.workloadHash = searched.workloadHash;
  CoveringPlan plan = searched.plans.front();
  PlanConnection gather;
  gather.id = 0x1234;
  gather.kind = ConnectionKind::Reduce;
  gather.gatherSemantics = GatherSemantics::Sum;
  gather.producerPorts = {producers[0], producers[1]};
  plan.connectionPlans.clear();
  plan.connectionPlans.push_back(gather);
  result.plans.push_back(std::move(plan));

  std::string report = writePlanReport(result, (**target).machine(), **target,
                                       options, stableHash(kKernel));
  llvm::Expected<CoveringPlan> replay =
      readPlanReport(report, **target, *graph);
  ASSERT_TRUE(static_cast<bool>(replay)) << llvm::toString(replay.takeError());
  ASSERT_EQ(replay->connectionPlans.size(), 1u);
  const PlanConnection &replayed = replay->connectionPlans.front();
  EXPECT_EQ(replayed.kind, ConnectionKind::Reduce);
  ASSERT_TRUE(replayed.gatherSemantics.has_value());
  EXPECT_EQ(*replayed.gatherSemantics, GatherSemantics::Sum);
  EXPECT_EQ(replayed.producerPorts, gather.producerPorts);
}

// A report carries each placement's per-occurrence memory bindings, so a replay
// reconstructs them exactly (task B2 fix round 1).
TEST(MappingPlanReportTest, ReportRoundTripsNamedPortMemoryBindings) {
  Parsed parsed = parseKernel(kKernel);
  ASSERT_TRUE(parsed.module);
  llvm::Expected<std::unique_ptr<MappingTarget>> target =
      avx2_mapping::createMappingTarget(LLK_SOURCE_DIR);
  ASSERT_TRUE(static_cast<bool>(target)) << llvm::toString(target.takeError());

  MappingSearchOptions options;
  options.mode = SearchMode::Deterministic;
  MappingSearchResult result =
      searchKernel(*parsed.context, parsed.kernel, **target, options);
  ASSERT_FALSE(result.plans.empty());

  llvm::Expected<WorkloadGraph> graph = extractWorkloadGraph(parsed.kernel);
  ASSERT_TRUE(static_cast<bool>(graph)) << llvm::toString(graph.takeError());

  // Attach a per-occurrence binding to a placement whose node has an output the
  // graph can resolve.
  bool attached = false;
  for (PlanPlacement &placement : result.plans.front().placements) {
    const WorkloadNode *node = graph->findNode(placement.node);
    if (!node || node->outputs.empty())
      continue;
    placement.portMemoryBindings.push_back(PortMemoryBinding{
        PortRef{placement.node, PortDirection::Output, 0}, "sram.0"});
    attached = true;
    break;
  }
  ASSERT_TRUE(attached) << "no placement had a resolvable output occurrence";

  std::string report = writePlanReport(result, (**target).machine(), **target,
                                       options, stableHash(kKernel));
  llvm::Expected<CoveringPlan> replay =
      readPlanReport(report, **target, *graph);
  ASSERT_TRUE(static_cast<bool>(replay)) << llvm::toString(replay.takeError());

  for (const PlanPlacement &original : result.plans.front().placements) {
    if (original.portMemoryBindings.empty())
      continue;
    const PlanPlacement *match = nullptr;
    for (const PlanPlacement &candidate : replay->placements)
      if (candidate.node == original.node)
        match = &candidate;
    ASSERT_NE(match, nullptr);
    EXPECT_EQ(match->portMemoryBindings, original.portMemoryBindings);
  }
}

// A report replayed against a target whose content differs is rejected: the
// recorded target hash no longer matches.
TEST(MappingPlanReportTest, ReplayRejectsAChangedTarget) {
  Parsed parsed = parseKernel(kKernel);
  ASSERT_TRUE(parsed.module);
  llvm::Expected<std::unique_ptr<MappingTarget>> target =
      avx2_mapping::createMappingTarget(LLK_SOURCE_DIR);
  ASSERT_TRUE(static_cast<bool>(target)) << llvm::toString(target.takeError());
  MappingSearchOptions options;
  options.mode = SearchMode::Deterministic;
  MappingSearchResult result =
      searchKernel(*parsed.context, parsed.kernel, **target, options);
  std::string report = writePlanReport(result, (**target).machine(), **target,
                                       options, stableHash(kKernel));
  llvm::Expected<WorkloadGraph> graph = extractWorkloadGraph(parsed.kernel);
  ASSERT_TRUE(static_cast<bool>(graph));

  // The same machine and layouts but a different rule library: the target
  // content hash moves.
  llvm::Expected<RuleRegistry> rules =
      parseRuleText("rule extra { match micro.vector(); bundle \"b\"; emit "
                    "\"e1\"; }",
                    "<test>");
  ASSERT_TRUE(static_cast<bool>(rules)) << llvm::toString(rules.takeError());
  FileMappingTarget other("x86-avx2", (**target).machine(),
                          (**target).layouts(), std::move(*rules),
                          std::vector<std::string>{"e1"});
  llvm::Expected<CoveringPlan> replay = readPlanReport(report, other, *graph);
  ASSERT_FALSE(static_cast<bool>(replay));
  EXPECT_NE(llvm::toString(replay.takeError()).find("target"),
            std::string::npos);
}

// A report replayed against a graph whose workload changed semantically -- the
// same node ids and port shapes, but a different operation attribute -- is
// rejected by the source-graph content hash, not merely by node resolution.
TEST(MappingPlanReportTest, ReplayRejectsAChangedInputGraph) {
  Parsed parsed = parseKernel(kKernel);
  ASSERT_TRUE(parsed.module);
  llvm::Expected<std::unique_ptr<MappingTarget>> target =
      avx2_mapping::createMappingTarget(LLK_SOURCE_DIR);
  ASSERT_TRUE(static_cast<bool>(target)) << llvm::toString(target.takeError());
  MappingSearchOptions options;
  options.mode = SearchMode::Deterministic;
  MappingSearchResult result =
      searchKernel(*parsed.context, parsed.kernel, **target, options);
  std::string report = writePlanReport(result, (**target).machine(), **target,
                                       options, stableHash(kKernel));

  // The same kernel with one semantic attribute changed. Node ids, ports and
  // the placement's rule mnemonic all still resolve, so only a content hash can
  // distinguish it.
  Parsed changed = parseKernel(kKernel);
  ASSERT_TRUE(changed.module);
  changed.module->walk([&](Operation *op) {
    if (op->getName().getStringRef() == "micro.vector")
      op->setAttr("op", StringAttr::get(changed.context.get(), "mul"));
  });
  llvm::Expected<WorkloadGraph> changedGraph =
      extractWorkloadGraph(changed.kernel);
  ASSERT_TRUE(static_cast<bool>(changedGraph));

  llvm::Expected<CoveringPlan> replay =
      readPlanReport(report, **target, *changedGraph);
  ASSERT_FALSE(static_cast<bool>(replay));
  EXPECT_NE(llvm::toString(replay.takeError()).find("graph"),
            std::string::npos);
}
