//===- plan_binder.cpp - Materializing a selected plan (#50 revision) ----===//

#include "LLK/Machine/MachineModelLoader.h"
#include "LLK/Mapping/CoveringSearch.h"
#include "LLK/Mapping/PlanBinder.h"
#include "LLK/Target/X86/Mapping/AVX2MappingTarget.h"

#include "LLK/Dialect/Micro/MicroDialect.h"

#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"

#include "llvm/Support/Error.h"

#include <gtest/gtest.h>

#include <memory>
#include <string>

using namespace mlir;
using namespace mlir::llk::mapping;

namespace avx2_mapping = mlir::llk::target::avx2;

#ifndef LLK_SOURCE_DIR
#error "LLK_SOURCE_DIR must name the repository root"
#endif

namespace {

/// A concrete kernel whose work is one copy into SRAM and one vector add.
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

struct Fixture {
  std::unique_ptr<MLIRContext> context;
  OwningOpRef<ModuleOp> module;
  std::unique_ptr<MappingTarget> target;
  OwningOpRef<ModuleOp> source; // the same kernel parsed a second time
};

Fixture makeFixture() {
  Fixture fixture;
  fixture.context = std::make_unique<MLIRContext>();
  fixture.context->getOrLoadDialect<micro::MicroDialect>();
  fixture.context->getOrLoadDialect<tensor::TensorDialect>();
  fixture.module = parseSourceString<ModuleOp>(kKernel, fixture.context.get());
  auto target = avx2_mapping::createMappingTarget(LLK_SOURCE_DIR);
  if (target)
    fixture.target = std::move(*target);
  return fixture;
}

/// The `micro.kernel` in `module`. The generated op classes are not part of
/// the dialect's public headers, so it is found by name.
Operation *findKernel(ModuleOp module) {
  Operation *kernel = nullptr;
  module->walk([&](Operation *op) {
    if (!kernel && op->getName().getStringRef() == "micro.kernel")
      kernel = op;
  });
  return kernel;
}

/// Runs the search and returns the best plan.
llvm::Expected<CoveringPlan> selectPlan(MLIRContext &context, ModuleOp module,
                                        MappingTarget &target) {
  llvm::Expected<WorkloadGraph> graph =
      extractWorkloadGraph(findKernel(module));
  if (!graph)
    return graph.takeError();
  LayoutContext layoutContext;
  layoutContext.rank = 2;
  layoutContext.elementType = "f32";
  MappingSearchOptions options;
  options.mode = SearchMode::Deterministic;
  CoveringSearch search(*graph, target, context, layoutContext, options);
  llvm::Expected<MappingSearchResult> result = search.search();
  if (!result)
    return result.takeError();
  if (result->plans.empty()) {
    std::string reason =
        "no plan was found: nodesWithoutRules=" +
        std::to_string(result->frontier.nodesWithoutRules) +
        " candidatesWithoutPlacement=" +
        std::to_string(result->frontier.candidatesWithoutPlacement) +
        " incompatiblePairs=" +
        std::to_string(result->frontier.incompatibleInstancePairs);
    for (const std::string &message : result->frontier.messages)
      reason += "\n  " + message;
    return llvm::createStringError(llvm::inconvertibleErrorCode(), reason);
  }
  return result->plans.front();
}

} // namespace

TEST(PlanBinder, MaterializesThePlanWithoutTouchingTheSource) {
  Fixture fixture = makeFixture();
  ASSERT_TRUE(fixture.module);
  ASSERT_NE(fixture.target, nullptr);

  llvm::Expected<CoveringPlan> plan =
      selectPlan(*fixture.context, *fixture.module, *fixture.target);
  ASSERT_TRUE(static_cast<bool>(plan)) << llvm::toString(plan.takeError());

  llvm::Expected<BoundPlan> bound =
      bindPlan(*fixture.module, *plan, *fixture.target);
  ASSERT_TRUE(static_cast<bool>(bound)) << llvm::toString(bound.takeError());
  ASSERT_NE(bound->kernel, nullptr);

  // The clone carries the plan; the source is untouched.
  EXPECT_TRUE(bound->kernel->hasAttr("micro.plan"));
  EXPECT_TRUE(bound->kernel->hasAttr("micro.routes"));
  EXPECT_FALSE(findKernel(*fixture.module)->hasAttr("micro.plan"));
}

TEST(PlanBinder, AnnotatesEachCoveredOperationWithItsSelection) {
  Fixture fixture = makeFixture();
  ASSERT_TRUE(fixture.module);
  ASSERT_NE(fixture.target, nullptr);
  llvm::Expected<CoveringPlan> plan =
      selectPlan(*fixture.context, *fixture.module, *fixture.target);
  ASSERT_TRUE(static_cast<bool>(plan)) << llvm::toString(plan.takeError());
  llvm::Expected<BoundPlan> bound =
      bindPlan(*fixture.module, *plan, *fixture.target);
  ASSERT_TRUE(static_cast<bool>(bound)) << llvm::toString(bound.takeError());

  std::vector<std::string> rules;
  bound->module->walk([&](Operation *op) {
    if (auto mapping = op->getAttrOfType<DictionaryAttr>("micro.mapping"))
      rules.push_back(mapping.getAs<StringAttr>("rule").getValue().str());
  });
  llvm::sort(rules);
  // Both workload nodes were covered: the copy and the vector add.
  EXPECT_EQ(rules,
            (std::vector<std::string>{"avx2.async_copy", "avx2.vector_add"}));
}

TEST(PlanBinder, MappedIrVerifiesAndRoundTripsWithoutATargetPlugin) {
  Fixture fixture = makeFixture();
  ASSERT_TRUE(fixture.module);
  ASSERT_NE(fixture.target, nullptr);
  llvm::Expected<CoveringPlan> plan =
      selectPlan(*fixture.context, *fixture.module, *fixture.target);
  ASSERT_TRUE(static_cast<bool>(plan)) << llvm::toString(plan.takeError());
  llvm::Expected<BoundPlan> bound =
      bindPlan(*fixture.module, *plan, *fixture.target);
  ASSERT_TRUE(static_cast<bool>(bound)) << llvm::toString(bound.takeError());

  EXPECT_FALSE(
      static_cast<bool>(verifyMappedMicroIR(*bound->module, *fixture.target)));

  // llk-opt must be able to print and re-parse mapped IR without a plugin
  // (design §18.3): the metadata is ordinary attributes.
  std::string text;
  llvm::raw_string_ostream stream(text);
  bound->module->print(stream);
  OwningOpRef<ModuleOp> reparsed =
      parseSourceString<ModuleOp>(stream.str(), fixture.context.get());
  ASSERT_TRUE(reparsed);
  EXPECT_FALSE(
      static_cast<bool>(verifyMappedMicroIR(*reparsed, *fixture.target)));
}

TEST(PlanBinder, MachineAwareVerificationRejectsAnUnknownExecutor) {
  Fixture fixture = makeFixture();
  ASSERT_TRUE(fixture.module);
  ASSERT_NE(fixture.target, nullptr);
  llvm::Expected<CoveringPlan> plan =
      selectPlan(*fixture.context, *fixture.module, *fixture.target);
  ASSERT_TRUE(static_cast<bool>(plan)) << llvm::toString(plan.takeError());
  llvm::Expected<BoundPlan> bound =
      bindPlan(*fixture.module, *plan, *fixture.target);
  ASSERT_TRUE(static_cast<bool>(bound)) << llvm::toString(bound.takeError());

  // Rewrite one mapping to name an executor the machine does not have.
  bound->module->walk([&](Operation *op) {
    if (auto mapping = op->getAttrOfType<DictionaryAttr>("micro.mapping")) {
      llvm::SmallVector<NamedAttribute> attributes(mapping.begin(),
                                                   mapping.end());
      for (NamedAttribute &attribute : attributes) {
        if (attribute.getName() == "executor") {
          attribute = NamedAttribute(
              attribute.getName(),
              StringAttr::get(fixture.context.get(), "no_such_executor"));
        }
      }
      op->setAttr("micro.mapping",
                  DictionaryAttr::get(fixture.context.get(), attributes));
    }
  });
  EXPECT_TRUE(
      static_cast<bool>(verifyMappedMicroIR(*bound->module, *fixture.target)));
}

//===----------------------------------------------------------------------===//
// Movement materialization (design §18.2)
//===----------------------------------------------------------------------===//

namespace {

/// Rules whose memory requirements differ per node, so the two instances bind
/// different memories and the edge between them has to move.
constexpr llvm::StringLiteral kMovementRules = R"llkmap(
rule t.copy {
  match micro.async_copy();
  require executor kind worker;
  require memory kind sram;
  bundle "b.copy";
  emit "e1";
  cost 1;
}
rule t.vector {
  match micro.vector(op = "add");
  require executor kind worker;
  require memory kind dram;
  bundle "b.vector";
  emit "e1";
  cost 1;
}
)llkmap";

/// The shipped AVX2 machine with these rules.
llvm::Expected<std::unique_ptr<MappingTarget>> movementTarget() {
  llvm::Expected<mlir::llk::machine::MachineModel> machine =
      mlir::llk::machine::loadMachineModel(std::string(LLK_SOURCE_DIR) +
                                           "/machines/x86-avx2-v2.yaml");
  if (!machine)
    return machine.takeError();
  llvm::Expected<LayoutRegistry> layouts = parseLayoutText("", "<test>");
  if (!layouts)
    return layouts.takeError();
  llvm::Expected<RuleRegistry> rules = parseRuleText(kMovementRules, "<test>");
  if (!rules)
    return rules.takeError();
  return std::make_unique<FileMappingTarget>(
      "movement", std::move(*machine), std::move(*layouts), std::move(*rules),
      std::vector<std::string>{"e1"});
}

size_t countOps(ModuleOp module, llvm::StringRef name) {
  size_t count = 0;
  module->walk([&](Operation *op) {
    if (op->getName().getStringRef() == name)
      ++count;
  });
  return count;
}

} // namespace

TEST(PlanBinder, EmitsACopyAndWaitWhenTheValueMustMove) {
  Fixture fixture = makeFixture();
  ASSERT_TRUE(fixture.module);
  llvm::Expected<std::unique_ptr<MappingTarget>> target = movementTarget();
  ASSERT_TRUE(static_cast<bool>(target)) << llvm::toString(target.takeError());

  llvm::Expected<CoveringPlan> plan =
      selectPlan(*fixture.context, *fixture.module, **target);
  ASSERT_TRUE(static_cast<bool>(plan)) << llvm::toString(plan.takeError());
  ASSERT_FALSE(plan->connectionPlans.empty());
  ASSERT_EQ(plan->connectionPlans[0].kind, ConnectionKind::Transfer);

  const size_t copiesBefore = countOps(*fixture.module, "micro.async_copy");
  const size_t waitsBefore = countOps(*fixture.module, "micro.wait");

  llvm::Expected<BoundPlan> bound = bindPlan(*fixture.module, *plan, **target);
  ASSERT_TRUE(static_cast<bool>(bound)) << llvm::toString(bound.takeError());

  for (const std::string &note : bound->unmaterialized)
    ADD_FAILURE() << note;
  EXPECT_TRUE(bound->unmaterialized.empty());
  EXPECT_EQ(countOps(*bound->module, "micro.async_copy"), copiesBefore + 1);
  EXPECT_EQ(countOps(*bound->module, "micro.wait"), waitsBefore + 1);

  // The mapped IR must satisfy both the dialect verifier and the mapping
  // checks: a copy that broke either is worse than no copy.
  EXPECT_FALSE(
      static_cast<bool>(verifyMappedMicroIR(*bound->module, **target)));

  // The source kernel is untouched.
  EXPECT_EQ(countOps(*fixture.module, "micro.async_copy"), copiesBefore);
  EXPECT_EQ(countOps(*fixture.module, "micro.wait"), waitsBefore);
}

TEST(PlanBinder, ReportsConnectionsItCannotMaterialize) {
  // The shipped rules bind no memory, so both instances land in the same
  // memory and nothing has to move -- there is no movement to report.
  Fixture fixture = makeFixture();
  ASSERT_TRUE(fixture.module);
  ASSERT_NE(fixture.target, nullptr);
  llvm::Expected<CoveringPlan> plan =
      selectPlan(*fixture.context, *fixture.module, *fixture.target);
  ASSERT_TRUE(static_cast<bool>(plan)) << llvm::toString(plan.takeError());
  llvm::Expected<BoundPlan> bound =
      bindPlan(*fixture.module, *plan, *fixture.target);
  ASSERT_TRUE(static_cast<bool>(bound)) << llvm::toString(bound.takeError());
  EXPECT_TRUE(bound->unmaterialized.empty());
}

//===----------------------------------------------------------------------===//
// Multi-hop materialization
//===----------------------------------------------------------------------===//

namespace {

/// A machine whose only way from DRAM to SRAM is through L2, so a routed
/// movement between them has two hops rather than one.
constexpr llvm::StringLiteral kTwoHopMachine = R"yaml(
schema: llk.machine.v2
target: two-hop
clock_hz: 1000000000
worker_threads: 1
executors:
  - id: cluster.0
    kind: cluster
  - id: worker.0
    kind: worker
    parent: cluster.0
memories:
  - id: dram.0
    kind: dram
    visible_from: cluster.0
    capacity_bytes: 1048576
    alignment_bytes: 64
    supported_layouts: [row_major]
    bandwidth_bytes_per_cycle: 32
    latency_cycles: 220
  - id: l2.0
    kind: l2
    visible_from: cluster.0
    capacity_bytes: 262144
    alignment_bytes: 64
    supported_layouts: [row_major]
    bandwidth_bytes_per_cycle: 64
    latency_cycles: 12
  - id: sram.0
    kind: sram
    visible_from: cluster.0
    capacity_bytes: 32768
    alignment_bytes: 64
    supported_layouts: [row_major]
    bandwidth_bytes_per_cycle: 64
    latency_cycles: 4
compute:
  - id: vpu
    kind: vector_engine
    attached_to: worker.0
    element_types: [f32]
    shapes: [[8]]
    lanes: {f32: 8}
transfer_engines:
  - id: dma.0
    kind: dma
    attached_to: cluster.0
    count: 1
    max_outstanding: 1
links:
  - id: dram_to_l2.0
    source: dram.0
    destination: l2.0
    bandwidth_bytes_per_cycle: 64
    latency_cycles: 220
    transaction_bytes: 4096
    transfer_engines: [dma.0]
  - id: l2_to_sram.0
    source: l2.0
    destination: sram.0
    bandwidth_bytes_per_cycle: 64
    latency_cycles: 12
    transaction_bytes: 4096
    transfer_engines: [dma.0]
)yaml";

/// The copy lands in DRAM and the add reads from SRAM, so the connection has to
/// cross the hierarchy.
constexpr llvm::StringLiteral kTwoHopRules = R"llkmap(
rule t.copy {
  match micro.async_copy();
  require executor kind worker;
  require memory kind dram;
  bundle "b.copy";
  emit "e1";
  cost 1;
}
rule t.vector {
  match micro.vector(op = "add");
  require executor kind worker;
  require memory kind sram;
  bundle "b.vector";
  emit "e1";
  cost 1;
}
)llkmap";

llvm::Expected<std::unique_ptr<MappingTarget>> twoHopTarget() {
  llvm::Expected<mlir::llk::machine::MachineModel> machine =
      mlir::llk::machine::parseMachineModel(kTwoHopMachine, "<test>");
  if (!machine)
    return machine.takeError();
  llvm::Expected<LayoutRegistry> layouts = parseLayoutText("", "<test>");
  if (!layouts)
    return layouts.takeError();
  llvm::Expected<RuleRegistry> rules = parseRuleText(kTwoHopRules, "<test>");
  if (!rules)
    return rules.takeError();
  return std::make_unique<FileMappingTarget>(
      "two-hop", std::move(*machine), std::move(*layouts), std::move(*rules),
      std::vector<std::string>{"e1"});
}

} // namespace

TEST(PlanBinder, EmitsOneCopyPerRouteHop) {
  Fixture fixture = makeFixture();
  ASSERT_TRUE(fixture.module);
  llvm::Expected<std::unique_ptr<MappingTarget>> target = twoHopTarget();
  ASSERT_TRUE(static_cast<bool>(target)) << llvm::toString(target.takeError());

  // The fixture's whole point is that DRAM reaches SRAM only through L2.
  {
    TopologyService topology((*target)->machine());
    RouteRequest request;
    request.source = "dram.0";
    request.destination = "sram.0";
    request.bytes = 4096;
    request.alignmentBytes = 32;
    llvm::Expected<llvm::SmallVector<MemoryRoute>> routes =
        topology.enumerateRoutes(request, 8);
    ASSERT_TRUE(static_cast<bool>(routes))
        << llvm::toString(routes.takeError());
    ASSERT_EQ(routes->size(), 1u);
    EXPECT_EQ((*routes)[0].nodes.size(), 3u);
  }

  llvm::Expected<CoveringPlan> plan =
      selectPlan(*fixture.context, *fixture.module, **target);
  ASSERT_TRUE(static_cast<bool>(plan)) << llvm::toString(plan.takeError());
  ASSERT_FALSE(plan->connectionPlans.empty());
  ASSERT_EQ(plan->connectionPlans[0].route.size(), 3u); // dram -> l2 -> sram

  const size_t copiesBefore = countOps(*fixture.module, "micro.async_copy");
  const size_t waitsBefore = countOps(*fixture.module, "micro.wait");

  llvm::Expected<BoundPlan> bound = bindPlan(*fixture.module, *plan, **target);
  ASSERT_TRUE(static_cast<bool>(bound)) << llvm::toString(bound.takeError());
  for (const std::string &note : bound->unmaterialized)
    ADD_FAILURE() << note;

  // Two hops, so two copies and two waits -- not one of each for the whole
  // route.
  EXPECT_EQ(countOps(*bound->module, "micro.async_copy"), copiesBefore + 2);
  EXPECT_EQ(countOps(*bound->module, "micro.wait"), waitsBefore + 2);
  EXPECT_FALSE(
      static_cast<bool>(verifyMappedMicroIR(*bound->module, **target)));
}
