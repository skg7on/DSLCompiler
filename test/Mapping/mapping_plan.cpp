//===- mapping_plan.cpp - Plan data model and canonical ids (D1) ---------===//

#include "LLK/Mapping/CoveringSearch.h"
#include "LLK/Mapping/MappingPlan.h"
#include "LLK/Mapping/MappingTarget.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"

#include "llvm/Support/Error.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

using namespace mlir::llk::mapping;

namespace {

/// One worker and no memories: enough to place a `micro.vector`.
mlir::llk::machine::MachineModel bundleMachine() {
  mlir::llk::machine::MachineModel model;
  model.target = "bundle";
  model.executors = {{"e0", "worker", std::nullopt, {}, 1, {}}};
  return model;
}

/// A one-node `micro.vector(op = "add")` graph. The node carries attributes, so
/// the rule-to-candidate bridge has an MLIR context to type the bundle
/// parameters with.
WorkloadGraph bundleGraph(mlir::MLIRContext &context) {
  WorkloadGraph graph;
  WorkloadValueId input =
      graph.addValue(WorkloadValue{0, mlir::Type(), "in", /*external=*/true});
  WorkloadValueId output =
      graph.addValue(WorkloadValue{0, mlir::Type(), "out", /*external=*/false});
  WorkloadNode node;
  node.opName = "micro.vector";
  node.attributes = mlir::DictionaryAttr::get(
      &context, {mlir::NamedAttribute(mlir::StringAttr::get(&context, "op"),
                                      mlir::StringAttr::get(&context, "add"))});
  node.inputs.push_back(WorkloadPort{input, mlir::Type(), std::nullopt});
  node.outputs.push_back(WorkloadPort{output, mlir::Type(), std::nullopt});
  graph.addNode(std::move(node));
  graph.finalize();
  return graph;
}

constexpr llvm::StringLiteral kBundleRules = R"llkmap(
rule r.bundle {
  match micro.vector(op = "add");
  require executor kind worker;
  bundle "b.tiled" { tile_m = 8, layout = "blocked_2d" };
  emit "e1";
}
)llkmap";

/// The typed parameters shared by the canonical-id tests.
mlir::DictionaryAttr typedParams(mlir::MLIRContext &context, int64_t tileM,
                                 int64_t tileN) {
  llvm::SmallVector<mlir::NamedAttribute> attributes;
  attributes.emplace_back(
      mlir::StringAttr::get(&context, "tile_m"),
      mlir::IntegerAttr::get(mlir::IntegerType::get(&context, 64), tileM));
  attributes.emplace_back(
      mlir::StringAttr::get(&context, "tile_n"),
      mlir::IntegerAttr::get(mlir::IntegerType::get(&context, 64), tileN));
  return mlir::DictionaryAttr::get(&context, attributes);
}

} // namespace

TEST(MappingPlan, BundleReachesPlacement) {
  mlir::MLIRContext context;
  WorkloadGraph graph = bundleGraph(context);
  llvm::Expected<RuleRegistry> registry = parseRuleText(kBundleRules, "<test>");
  ASSERT_TRUE(static_cast<bool>(registry))
      << llvm::toString(registry.takeError());
  FileMappingTarget target("test", bundleMachine(), LayoutRegistry{},
                           std::move(*registry),
                           std::vector<std::string>{"e1"});

  MappingSearchOptions options;
  options.mode = SearchMode::Deterministic;
  CoveringSearch search(graph, target, context, LayoutContext{}, options);
  llvm::Expected<MappingSearchResult> result = search.search();
  ASSERT_TRUE(static_cast<bool>(result)) << llvm::toString(result.takeError());
  ASSERT_FALSE(result->plans.empty());
  ASSERT_FALSE(result->plans.front().placements.empty());

  const TargetBundle &bundle = result->plans.front().placements.front().bundle;
  EXPECT_EQ(bundle.name, "b.tiled");
  EXPECT_EQ(bundle.emitterKey, "e1");
  ASSERT_TRUE(bundle.parameters);
  auto tileM = bundle.parameters.getAs<mlir::IntegerAttr>("tile_m");
  ASSERT_TRUE(tileM);
  EXPECT_EQ(tileM.getInt(), 8);
  auto layout = bundle.parameters.getAs<mlir::StringAttr>("layout");
  ASSERT_TRUE(layout);
  EXPECT_EQ(layout.getValue(), "blocked_2d");
}

TEST(MappingPlan, BundleParameterOrderDoesNotChangeTheId) {
  mlir::MLIRContext context;
  auto param = [&](llvm::StringRef name, mlir::Attribute value) {
    return mlir::NamedAttribute(mlir::StringAttr::get(&context, name), value);
  };
  auto intAttr = [&](int64_t value) {
    return mlir::IntegerAttr::get(mlir::IntegerType::get(&context, 64), value);
  };

  MappingCandidate a;
  a.rule = "r";
  a.bundle.name = "b";
  a.bundle.parameters = mlir::DictionaryAttr::get(
      &context, {param("tile_n", intAttr(16)), param("tile_m", intAttr(8))});
  MappingCandidate b;
  b.rule = "r";
  b.bundle.name = "b";
  b.bundle.parameters = mlir::DictionaryAttr::get(
      &context, {param("tile_m", intAttr(8)), param("tile_n", intAttr(16))});

  EXPECT_EQ(canonicalCandidateString(a), canonicalCandidateString(b));
  EXPECT_EQ(computeCandidateId(a), computeCandidateId(b));
}

TEST(MappingPlan, BundleParameterValueChangesTheCandidateId) {
  mlir::MLIRContext context;
  MappingCandidate baseline;
  baseline.rule = "r";
  baseline.bundle.name = "b";
  baseline.bundle.parameters = typedParams(context, 8, 16);

  MappingCandidate other;
  other.rule = "r";
  other.bundle.name = "b";
  other.bundle.parameters = typedParams(context, 16, 16);

  EXPECT_NE(computeCandidateId(baseline), computeCandidateId(other));
}

TEST(MappingPlan, BundleParameterTypeIsPartOfTheId) {
  mlir::MLIRContext context;
  MappingCandidate integer;
  integer.rule = "r";
  integer.bundle.name = "b";
  integer.bundle.parameters = mlir::DictionaryAttr::get(
      &context,
      {mlir::NamedAttribute(
          mlir::StringAttr::get(&context, "tile_m"),
          mlir::IntegerAttr::get(mlir::IntegerType::get(&context, 64), 8))});

  MappingCandidate text;
  text.rule = "r";
  text.bundle.name = "b";
  text.bundle.parameters = mlir::DictionaryAttr::get(
      &context, {mlir::NamedAttribute(mlir::StringAttr::get(&context, "tile_m"),
                                      mlir::StringAttr::get(&context, "8"))});

  EXPECT_NE(computeCandidateId(integer), computeCandidateId(text));
}

TEST(MappingPlan, CandidateIdIsOrderIndependent) {
  MappingCandidate a;
  a.rule = "r";
  a.coveredNodes = {2, 0, 1};
  MappingCandidate b;
  b.rule = "r";
  b.coveredNodes = {0, 1, 2};
  EXPECT_EQ(computeCandidateId(a), computeCandidateId(b));
}

TEST(MappingPlan, CandidateIdChangesWithContent) {
  MappingCandidate a;
  a.rule = "r";
  a.coveredNodes = {0};
  MappingCandidate b;
  b.rule = "r2";
  b.coveredNodes = {0};
  EXPECT_NE(computeCandidateId(a), computeCandidateId(b));
}

TEST(MappingPlan, InstanceIdIgnoresBindingInsertionOrder) {
  CandidateInstance a;
  a.candidate = 7;
  a.executorBindings["w0"] = "core.0";
  a.executorBindings["w1"] = "core.1";
  CandidateInstance b;
  b.candidate = 7;
  b.executorBindings["w1"] = "core.1";
  b.executorBindings["w0"] = "core.0";
  EXPECT_EQ(computeInstanceId(a), computeInstanceId(b));
}

TEST(MappingPlan, PlanIdChangesWithBindingHash) {
  CoveringPlan a;
  a.sourceBindingHash = 1;
  CoveringPlan b;
  b.sourceBindingHash = 2;
  EXPECT_NE(computePlanId(a), computePlanId(b));
}

TEST(MappingPlan, PlanIdIgnoresInstanceOrder) {
  CoveringPlan a;
  a.sourceBindingHash = 5;
  a.instances = {2, 1};
  CoveringPlan b;
  b.sourceBindingHash = 5;
  b.instances = {1, 2};
  EXPECT_EQ(computePlanId(a), computePlanId(b));
}

TEST(MappingPlan, SortUniqueRemovesDuplicatesAndSorts) {
  llvm::SmallVector<WorkloadNodeId> nodes{3, 1, 3, 2};
  sortUnique(nodes);
  ASSERT_EQ(nodes.size(), 3u);
  EXPECT_EQ(nodes[0], 1u);
  EXPECT_EQ(nodes[1], 2u);
  EXPECT_EQ(nodes[2], 3u);
}

TEST(MappingPlan, ConnectionKindRoundTrips) {
  for (ConnectionKind kind :
       {ConnectionKind::Direct, ConnectionKind::Transfer,
        ConnectionKind::LayoutTransform, ConnectionKind::TransferAndTransform,
        ConnectionKind::Replicate, ConnectionKind::Reduce}) {
    EXPECT_EQ(symbolizeConnectionKind(stringifyConnectionKind(kind)), kind);
  }
}
