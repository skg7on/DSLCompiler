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
#include <variant>
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

//===----------------------------------------------------------------------===//
// Solved layout assignments and identity (phase-3 T4, ruling R4)
//===----------------------------------------------------------------------===//

// R4: the solved parameter assignment IS part of the canonical instance and
// plan identity, deliberately. The bound layout *id* names the layout family;
// the solved assignment is what makes it concrete (`VW = 8` blocks differently
// from `VW = 4`, occupies different bytes, and materializes different code), so
// two instances that differ only in it are different placements and must not
// collide on one id. It is not a derived view of already-hashed content either:
// the solve also depends on the machine model and the solver limits, neither of
// which the candidate's content covers.
namespace {

/// An instance binding one layout class, solved with `vw`.
CandidateInstance solvedInstance(int64_t vw) {
  CandidateInstance instance;
  instance.candidate = 7;
  instance.layoutBindings["t.blocked"] = "t.blocked";
  instance.layoutSolutions["t.blocked"].parameters["VW"] = SearchValue(vw);
  return instance;
}

/// A plan with one placement, binding one layout class solved with `vw`.
CoveringPlan solvedPlan(int64_t vw) {
  CoveringPlan plan;
  PlanPlacement placement;
  placement.node = 0;
  placement.instance = 1;
  placement.rule = "r.blocked";
  placement.layouts["t.blocked"] = "t.blocked";
  placement.layoutSolutions["t.blocked"].parameters["VW"] = SearchValue(vw);
  plan.placements.push_back(std::move(placement));
  return plan;
}

} // namespace

TEST(MappingPlan, SolvedLayoutParametersArePartOfTheInstanceId) {
  EXPECT_NE(computeInstanceId(solvedInstance(8)),
            computeInstanceId(solvedInstance(4)));
}

TEST(MappingPlan, SolvedLayoutParametersArePartOfThePlanId) {
  EXPECT_NE(computePlanId(solvedPlan(8)), computePlanId(solvedPlan(4)));
}

// The rendering is sorted and type-tagged (the same canonical helper the search
// binding uses), so insertion order cannot leak into an id and an integer `8`
// never hashes like the string `"8"`.
TEST(MappingPlan, SolvedLayoutParameterOrderDoesNotChangeTheId) {
  // The same two parameters, filled in opposite orders.
  CandidateInstance ascending = solvedInstance(8);
  ascending.layoutSolutions["t.blocked"].parameters["N"] =
      SearchValue(int64_t{16});
  CandidateInstance descending;
  descending.candidate = 7;
  descending.layoutBindings["t.blocked"] = "t.blocked";
  descending.layoutSolutions["t.blocked"].parameters["N"] =
      SearchValue(int64_t{16});
  descending.layoutSolutions["t.blocked"].parameters["VW"] =
      SearchValue(int64_t{8});
  EXPECT_EQ(computeInstanceId(ascending), computeInstanceId(descending));
}

TEST(MappingPlan, SolvedLayoutParameterTypeIsPartOfTheId) {
  CandidateInstance integer = solvedInstance(8);
  CandidateInstance text;
  text.candidate = 7;
  text.layoutBindings["t.blocked"] = "t.blocked";
  text.layoutSolutions["t.blocked"].parameters["VW"] =
      SearchValue(std::string("8"));
  EXPECT_NE(computeInstanceId(integer), computeInstanceId(text));
}

// The affine map is carried (a materializer needs it) but is deliberately NOT
// part of the id: it is a pure function of the layout id and the solved
// assignment, both of which the canonical string already contains, so including
// its rendering would add a dependency on MLIR's map printer for no extra
// distinguishing power.
TEST(MappingPlan, TheSolvedAffineMapIsExcludedFromTheId) {
  mlir::MLIRContext context;
  CandidateInstance withoutMap = solvedInstance(8);
  CandidateInstance withMap = solvedInstance(8);
  withMap.layoutSolutions["t.blocked"].map =
      mlir::AffineMap::getMultiDimIdentityMap(2, &context);
  EXPECT_EQ(computeInstanceId(withoutMap), computeInstanceId(withMap));
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

//===----------------------------------------------------------------------===//
// Endpoint occurrence identity (A1)
//===----------------------------------------------------------------------===//

// An endpoint is an occurrence, not the value it carries: two operand specs
// that share a value and side but sit at different ports must not collide. A
// spec with no resolved endpoint keeps the legacy value-only rendering, so ids
// stay stable until endpoint resolution migrates in (B1 owns that migration).
namespace {

MappingCandidate candidateWithPort(std::optional<PortRef> endpoint) {
  MappingCandidate candidate;
  candidate.rule = "r";
  PortSpec spec;
  spec.name = "lhs";
  spec.value = 3;
  spec.isInput = true;
  spec.port = endpoint;
  candidate.ports.push_back(std::move(spec));
  return candidate;
}

ConnectionPlan
connectionWithEndpoints(std::optional<PortRef> producerPort,
                        llvm::SmallVector<PortRef> consumerPorts) {
  ConnectionPlan connection;
  connection.producer = 1;
  connection.value = 3;
  connection.producerPort = producerPort;
  connection.consumerPorts = std::move(consumerPorts);
  return connection;
}

CoveringPlan planWithConnection(const PlanConnection &connection) {
  CoveringPlan plan;
  plan.connectionPlans.push_back(connection);
  return plan;
}

} // namespace

TEST(MappingPlan, ResolvedEndpointIsPartOfTheCandidateId) {
  MappingCandidate plain = candidateWithPort(std::nullopt);
  MappingCandidate first =
      candidateWithPort(PortRef{7, PortDirection::Input, 0});
  MappingCandidate second =
      candidateWithPort(PortRef{7, PortDirection::Input, 1});

  // Unresolved: the legacy value-only key carries no endpoint marker.
  EXPECT_EQ(canonicalCandidateString(plain).find("node="), std::string::npos);
  // Resolved: the occurrence tells two uses of one value apart.
  EXPECT_NE(canonicalCandidateString(first), canonicalCandidateString(second));
  EXPECT_NE(computeCandidateId(first), computeCandidateId(second));
}

TEST(MappingPlan, ResolvedEndpointsArePartOfTheConnectionId) {
  ConnectionPlan plain = connectionWithEndpoints(std::nullopt, {});
  ConnectionPlan first =
      connectionWithEndpoints(PortRef{1, PortDirection::Output, 0},
                              {PortRef{2, PortDirection::Input, 0}});
  ConnectionPlan second =
      connectionWithEndpoints(PortRef{1, PortDirection::Output, 0},
                              {PortRef{2, PortDirection::Input, 1}});

  // Unresolved: neither endpoint marker is emitted, so the id is unchanged.
  EXPECT_EQ(canonicalConnectionString(plain).find("producerPort="),
            std::string::npos);
  EXPECT_EQ(canonicalConnectionString(plain).find("consumerPorts="),
            std::string::npos);
  // Resolved: the consumer occurrence is part of the connection's identity.
  EXPECT_NE(canonicalConnectionString(first),
            canonicalConnectionString(second));
  EXPECT_NE(computeConnectionId(first), computeConnectionId(second));
}

// The plan-level projection must expose the endpoint occurrences too. Two
// connections of one value whose `consumers` instance projection is identical
// (the repeated-operand case) are otherwise indistinguishable in a selected
// plan, so a materializer would rewiring both to the same read.
TEST(MappingPlan, PlanConnectionExposesEndpointOccurrences) {
  ConnectionPlan firstConnection =
      connectionWithEndpoints(PortRef{1, PortDirection::Output, 0},
                              {PortRef{2, PortDirection::Input, 0}});
  firstConnection.id = computeConnectionId(firstConnection);
  ConnectionPlan secondConnection = firstConnection;
  secondConnection.consumerPorts = {PortRef{2, PortDirection::Input, 1}};
  secondConnection.id = computeConnectionId(secondConnection);

  PlanConnection first;
  first.id = firstConnection.id;
  first.value = firstConnection.value;
  first.kind = firstConnection.kind;
  first.producerPort = firstConnection.producerPort;
  first.consumerPorts = firstConnection.consumerPorts;
  first.consumers = {9};
  PlanConnection second = first; // identical instance projection
  second.id = secondConnection.id;
  second.consumerPorts = secondConnection.consumerPorts;

  // The compatibility projection cannot tell the two uses apart ...
  EXPECT_EQ(first.consumers, second.consumers);
  // ... but the endpoint occurrences can, and they reach plan identity.
  EXPECT_EQ(first.producerPort, second.producerPort);
  ASSERT_EQ(first.consumerPorts.size(), 1u);
  ASSERT_EQ(second.consumerPorts.size(), 1u);
  EXPECT_NE(first.consumerPorts[0], second.consumerPorts[0]);
  EXPECT_NE(computePlanId(planWithConnection(first)),
            computePlanId(planWithConnection(second)));
}

// The control: a `PlanConnection` built without endpoint resolution stays
// byte-identical, so no endpoint marker is introduced into its plan's canonical
// string and its id is unchanged.
TEST(MappingPlan, PlanConnectionWithoutEndpointsKeepsTheLegacyPlanId) {
  PlanConnection connection;
  connection.id = 7;
  connection.value = 3;
  connection.kind = ConnectionKind::Direct;
  connection.consumers = {9};
  EXPECT_FALSE(connection.producerPort.has_value());
  EXPECT_TRUE(connection.consumerPorts.empty());

  const std::string canonical =
      canonicalPlanString(planWithConnection(connection));
  EXPECT_EQ(canonical.find("producerPort"), std::string::npos);
  EXPECT_EQ(canonical.find("consumerPort"), std::string::npos);
}
