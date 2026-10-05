//===- plan_binder.cpp - Materializing a selected plan (#50 revision) ----===//

#include "LLK/Machine/MachineModelLoader.h"
#include "LLK/Mapping/CoveringSearch.h"
#include "LLK/Mapping/MappingMetadata.h"
#include "LLK/Mapping/PlanBinder.h"
#include "LLK/Mapping/PlanReport.h"
#include "LLK/Target/X86/Mapping/AVX2MappingTarget.h"

#include "MicroMappingCommon.h"

#include "LLK/Dialect/Micro/MicroDialect.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"

#include "llvm/Support/Error.h"

#include <gtest/gtest.h>

#include <memory>
#include <string>
#include <utility>

using namespace mlir;
using namespace mlir::llk::mapping;

namespace avx2_mapping = mlir::llk::target::avx2;

#ifndef LLK_SOURCE_DIR
#error "LLK_SOURCE_DIR must name the repository root"
#endif

namespace {

/// Binds `plan` with the canonical Micro materializer, exactly as the CLI does
/// (design §18.2): the target-neutral binder persists the selected state and
/// this dialect-aware component constructs the movements and transforms. A
/// fresh materializer per call keeps each test independent.
llvm::Expected<BoundPlan>
bindCanonical(ModuleOp module, const CoveringPlan &plan,
              const MappingTarget &target,
              BindContract contract = BindContract::Partial) {
  std::unique_ptr<PlanMaterializer> materializer =
      mlir::llk::micro_mapping_detail::createCanonicalPlanMaterializer();
  return bindPlan(module, plan, target, contract, materializer.get());
}

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
    for (const mlir::llk::mapping::Diagnostic &diagnostic :
         result->frontier.diagnostics)
      reason +=
          "\n  " +
          mlir::llk::mapping::stringifyDiagnosticCode(diagnostic.code).str() +
          ": " + diagnostic.message;
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
      bindCanonical(*fixture.module, *plan, *fixture.target);
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
      bindCanonical(*fixture.module, *plan, *fixture.target);
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
      bindCanonical(*fixture.module, *plan, *fixture.target);
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

// The bound IR must state the *concrete* layout instantiation the plan chose,
// not merely the family: `avx2.blocked_2d` alone cannot tell a materializer
// whether the plan meant `VW = 4` or `VW = 8`. The parameters are persisted as
// typed attributes under `layout_parameters`, alongside the family ids.
TEST(PlanBinder, PersistsTheSolvedLayoutParameters) {
  Fixture fixture = makeFixture();
  ASSERT_TRUE(fixture.module);
  ASSERT_NE(fixture.target, nullptr);
  llvm::Expected<CoveringPlan> plan =
      selectPlan(*fixture.context, *fixture.module, *fixture.target);
  ASSERT_TRUE(static_cast<bool>(plan)) << llvm::toString(plan.takeError());
  llvm::Expected<BoundPlan> bound =
      bindCanonical(*fixture.module, *plan, *fixture.target);
  ASSERT_TRUE(static_cast<bool>(bound)) << llvm::toString(bound.takeError());

  bool sawSolvedWidth = false;
  bound->kernel->walk([&](Operation *op) {
    auto mapping = op->getAttrOfType<DictionaryAttr>("micro.mapping");
    if (!mapping)
      return;
    auto layoutParameters = mapping.getAs<DictionaryAttr>("layout_parameters");
    if (!layoutParameters)
      return;
    for (const NamedAttribute &entry : layoutParameters) {
      auto parameters = dyn_cast<DictionaryAttr>(entry.getValue());
      ASSERT_TRUE(parameters) << "layout_parameters entry is not a dictionary";
      // The AVX2 blocked layout is parameterized by M, N, and VW; a solved
      // integer VW is what says which instantiation was selected.
      if (auto vw = parameters.getAs<IntegerAttr>("VW"))
        sawSolvedWidth = true;
    }
  });
  EXPECT_TRUE(sawSolvedWidth);

  // The persisted parameters are ordinary metadata, so phase-2 verification
  // still accepts the module.
  EXPECT_FALSE(
      static_cast<bool>(verifyMappedMicroIR(*bound->module, *fixture.target)));
}

// Regression (issue #67, stage A): the verifier checked the *container* types
// of `layout_parameters` but never re-solved the values, so a tampered solved
// assignment -- a vector width the target layout's own constraints forbid --
// verified successfully. The recorded assignment must be validated against the
// declaration it names, and never replaced by a different legal one.
TEST(PlanBinder, RejectsTamperedSolvedVectorWidth) {
  Fixture f = makeFixture();
  ASSERT_TRUE(f.module);
  ASSERT_TRUE(f.target);
  auto p = selectPlan(*f.context, *f.module, *f.target);
  ASSERT_TRUE(bool(p)) << llvm::toString(p.takeError());
  auto b = bindCanonical(*f.module, *p, *f.target);
  ASSERT_TRUE(bool(b)) << llvm::toString(b.takeError());
  auto legal = verifyMappedMicroIR(*b->module, *f.target);
  ASSERT_FALSE(bool(legal)) << llvm::toString(std::move(legal));
  unsigned mutations = 0;
  b->module->walk([&](Operation *op) {
    if (op->getName().getStringRef() != "micro.vector")
      return;
    auto mapping = op->getAttrOfType<DictionaryAttr>("micro.mapping");
    ASSERT_TRUE(mapping);
    auto layouts = mapping.getAs<DictionaryAttr>("layout_parameters");
    ASSERT_TRUE(layouts);
    NamedAttrList newLayouts(layouts);
    for (NamedAttribute entry : layouts) {
      auto parameters = dyn_cast<DictionaryAttr>(entry.getValue());
      if (!parameters || !parameters.getAs<IntegerAttr>("VW"))
        continue;
      NamedAttrList newParameters(parameters);
      newParameters.set(
          "VW", IntegerAttr::get(IntegerType::get(f.context.get(), 64), 4));
      newLayouts.set(entry.getName(),
                     newParameters.getDictionary(f.context.get()));
      ++mutations;
    }
    NamedAttrList newMapping(mapping);
    newMapping.set("layout_parameters",
                   newLayouts.getDictionary(f.context.get()));
    op->setAttr("micro.mapping", newMapping.getDictionary(f.context.get()));
  });
  ASSERT_GT(mutations, 0u);
  auto error = verifyMappedMicroIR(*b->module, *f.target);
  ASSERT_TRUE(bool(error));
  EXPECT_NE(llvm::toString(std::move(error)).find("no_legal_layout"),
            std::string::npos);
}

// A layout family the plan records must be the family the rule requires and
// the class's own declaration: pointing a class at some other legal layout is
// tampering the verifier must reject, not accept because the id resolves.
TEST(PlanBinder, RejectsTamperedLayoutFamily) {
  Fixture f = makeFixture();
  ASSERT_TRUE(f.module);
  ASSERT_TRUE(f.target);
  auto p = selectPlan(*f.context, *f.module, *f.target);
  ASSERT_TRUE(bool(p)) << llvm::toString(p.takeError());
  auto b = bindCanonical(*f.module, *p, *f.target);
  ASSERT_TRUE(bool(b)) << llvm::toString(b.takeError());
  ASSERT_FALSE(bool(verifyMappedMicroIR(*b->module, *f.target)));

  unsigned mutations = 0;
  b->module->walk([&](Operation *op) {
    if (op->getName().getStringRef() != "micro.vector")
      return;
    auto mapping = op->getAttrOfType<DictionaryAttr>("micro.mapping");
    ASSERT_TRUE(mapping);
    auto layouts = mapping.getAs<DictionaryAttr>("layouts");
    ASSERT_TRUE(layouts);
    NamedAttrList newLayouts(layouts);
    for (NamedAttribute entry : layouts) {
      // Repoint the recorded family at another layout the registry declares.
      newLayouts.set(entry.getName(),
                     StringAttr::get(f.context.get(), "avx2.row_major"));
      ++mutations;
    }
    NamedAttrList newMapping(mapping);
    newMapping.set("layouts", newLayouts.getDictionary(f.context.get()));
    op->setAttr("micro.mapping", newMapping.getDictionary(f.context.get()));
  });
  ASSERT_GT(mutations, 0u);

  auto error = verifyMappedMicroIR(*b->module, *f.target);
  ASSERT_TRUE(bool(error));
  EXPECT_NE(llvm::toString(std::move(error)).find("no_legal_layout"),
            std::string::npos);
}

TEST(PlanBinder, MachineAwareVerificationRejectsAnUnknownExecutor) {
  Fixture fixture = makeFixture();
  ASSERT_TRUE(fixture.module);
  ASSERT_NE(fixture.target, nullptr);
  llvm::Expected<CoveringPlan> plan =
      selectPlan(*fixture.context, *fixture.module, *fixture.target);
  ASSERT_TRUE(static_cast<bool>(plan)) << llvm::toString(plan.takeError());
  llvm::Expected<BoundPlan> bound =
      bindCanonical(*fixture.module, *plan, *fixture.target);
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

/// A machine whose two memories live in different visibility scopes, so a node
/// placed on one cannot read the other's memory: the edge between them is a
/// real transfer, not a direct read. (Under §10.2 a single-scope machine -- the
/// shipped AVX2 profile among them -- lets any worker read any memory, so every
/// cross-memory edge is a `Direct` read and nothing has to move; that is why
/// this fixture defines its own machine.)
constexpr llvm::StringLiteral kMovementMachine = R"yaml(
schema: llk.machine.v2
target: movement
clock_hz: 1000000000
worker_threads: 2
executors:
  - id: cluster.a
    kind: cluster
  - id: worker.a
    kind: worker
    parent: cluster.a
  - id: cluster.b
    kind: cluster
  - id: worker.b
    kind: worker
    parent: cluster.b
memories:
  - id: sram.0
    kind: sram
    visible_from: cluster.a
    capacity_bytes: 1048576
    alignment_bytes: 64
    supported_layouts: [row_major]
    bandwidth_bytes_per_cycle: 64
    latency_cycles: 4
  - id: dram.0
    kind: dram
    visible_from: cluster.b
    capacity_bytes: 1073741824
    alignment_bytes: 64
    supported_layouts: [row_major]
    bandwidth_bytes_per_cycle: 32
    latency_cycles: 220
compute:
  - id: vpu
    kind: vector_engine
    attached_to: worker.a
    element_types: [f32]
    shapes: [[8]]
    lanes: {f32: 8}
transfer_engines:
  - id: dma.a
    kind: dma
    attached_to: cluster.a
    count: 1
    max_outstanding: 1
links:
  - id: sram_to_dram.0
    source: sram.0
    destination: dram.0
    bandwidth_bytes_per_cycle: 64
    latency_cycles: 220
    transaction_bytes: 64
    transfer_engines: [dma.a]
)yaml";

/// The movement machine with these rules. The copy rule binds SRAM (visible
/// only in cluster A) and the vector rule binds DRAM (visible only in cluster
/// B), so the two instances sit on different executors and the consumer cannot
/// read the producer's memory.
llvm::Expected<std::unique_ptr<MappingTarget>> movementTarget() {
  llvm::Expected<mlir::llk::machine::MachineModel> machine =
      mlir::llk::machine::parseMachineModel(kMovementMachine, "<test>");
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

/// Rules that bind each node's *output* port to its own memory by name, so a
/// selected plan carries per-occurrence bindings (B2) that must survive
/// persistence, the report, and replay.
constexpr llvm::StringLiteral kNamedPortRules = R"llkmap(
rule t.copy {
  match micro.async_copy();
  require executor kind worker;
  require memory output "result" kind sram;
  output "result";
  bundle "b.copy";
  emit "e1";
  cost 1;
}
rule t.vector {
  match micro.vector(op = "add");
  require executor kind worker;
  require memory output "result" kind dram;
  output "result";
  bundle "b.vector";
  emit "e1";
  cost 1;
}
)llkmap";

/// `movementTarget` with named-port rules: the copy binds SRAM and the vector
/// binds DRAM, but each through an explicit output occurrence.
llvm::Expected<std::unique_ptr<MappingTarget>> namedPortTarget() {
  llvm::Expected<mlir::llk::machine::MachineModel> machine =
      mlir::llk::machine::parseMachineModel(kMovementMachine, "<test>");
  if (!machine)
    return machine.takeError();
  llvm::Expected<LayoutRegistry> layouts = parseLayoutText("", "<test>");
  if (!layouts)
    return layouts.takeError();
  llvm::Expected<RuleRegistry> rules = parseRuleText(kNamedPortRules, "<test>");
  if (!rules)
    return rules.takeError();
  return std::make_unique<FileMappingTarget>(
      "named-port", std::move(*machine), std::move(*layouts), std::move(*rules),
      std::vector<std::string>{"e1"});
}

/// The movement machine with a second route from SRAM to DRAM: a staged hop
/// through an L2. One value can then be carried two legal ways, which is what
/// exercises one copy chain per route.
constexpr llvm::StringLiteral kTwoRouteMachine = R"yaml(
schema: llk.machine.v2
target: two-route
clock_hz: 1000000000
worker_threads: 2
executors:
  - id: cluster.a
    kind: cluster
  - id: worker.a
    kind: worker
    parent: cluster.a
  - id: cluster.b
    kind: cluster
  - id: worker.b
    kind: worker
    parent: cluster.b
memories:
  - id: sram.0
    kind: sram
    visible_from: cluster.a
    capacity_bytes: 1048576
    alignment_bytes: 64
    supported_layouts: [row_major]
    bandwidth_bytes_per_cycle: 64
    latency_cycles: 4
  - id: stage.0
    kind: l2
    visible_from: cluster.a
    capacity_bytes: 1048576
    alignment_bytes: 64
    supported_layouts: [row_major]
    bandwidth_bytes_per_cycle: 64
    latency_cycles: 12
  - id: dram.0
    kind: dram
    visible_from: cluster.b
    capacity_bytes: 1073741824
    alignment_bytes: 64
    supported_layouts: [row_major]
    bandwidth_bytes_per_cycle: 32
    latency_cycles: 220
compute:
  - id: vpu
    kind: vector_engine
    attached_to: worker.a
    element_types: [f32]
    shapes: [[8]]
    lanes: {f32: 8}
transfer_engines:
  - id: dma.a
    kind: dma
    attached_to: cluster.a
    count: 1
    max_outstanding: 1
links:
  - id: sram_to_dram.0
    source: sram.0
    destination: dram.0
    bandwidth_bytes_per_cycle: 64
    latency_cycles: 220
    transaction_bytes: 64
    transfer_engines: [dma.a]
  - id: sram_to_stage.0
    source: sram.0
    destination: stage.0
    bandwidth_bytes_per_cycle: 64
    latency_cycles: 12
    transaction_bytes: 64
    transfer_engines: [dma.a]
  - id: stage_to_dram.0
    source: stage.0
    destination: dram.0
    bandwidth_bytes_per_cycle: 64
    latency_cycles: 220
    transaction_bytes: 64
    transfer_engines: [dma.a]
)yaml";

/// `movementTarget` over the two-route machine, so the same edge can be carried
/// by a direct hop or a staged one.
llvm::Expected<std::unique_ptr<MappingTarget>> twoRouteTarget() {
  llvm::Expected<mlir::llk::machine::MachineModel> machine =
      mlir::llk::machine::parseMachineModel(kTwoRouteMachine, "<test>");
  if (!machine)
    return machine.takeError();
  llvm::Expected<LayoutRegistry> layouts = parseLayoutText("", "<test>");
  if (!layouts)
    return layouts.takeError();
  llvm::Expected<RuleRegistry> rules = parseRuleText(kMovementRules, "<test>");
  if (!rules)
    return rules.takeError();
  return std::make_unique<FileMappingTarget>(
      "two-route", std::move(*machine), std::move(*layouts), std::move(*rules),
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

  llvm::Expected<BoundPlan> bound =
      bindCanonical(*fixture.module, *plan, **target);
  ASSERT_TRUE(static_cast<bool>(bound)) << llvm::toString(bound.takeError());

  for (const std::string &note : bound->unmaterialized)
    ADD_FAILURE() << note;
  EXPECT_TRUE(bound->unmaterialized.empty());
  EXPECT_EQ(countOps(*bound->module, "micro.async_copy"), copiesBefore + 1);
  EXPECT_EQ(countOps(*bound->module, "micro.wait"), waitsBefore + 1);

  // The mapped IR must satisfy both the dialect verifier and the mapping
  // checks: a copy that broke either is worse than no copy.
  {
    llvm::Error verification = verifyMappedMicroIR(*bound->module, **target);
    EXPECT_FALSE(static_cast<bool>(verification))
        << llvm::toString(std::move(verification));
  }

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
      bindCanonical(*fixture.module, *plan, *fixture.target);
  ASSERT_TRUE(static_cast<bool>(bound)) << llvm::toString(bound.takeError());
  EXPECT_TRUE(bound->unmaterialized.empty());
}

// A `LayoutTransform` connection moves nothing -- the value already sits in the
// memory the consumer reads -- and the conversion becomes one target-neutral
// `micro.transform`, which names the two layouts by their affine maps rather
// than by a target id (design §13.4).
TEST(PlanBinder, MaterializesALayoutTransformAsATransformOp) {
  Fixture fixture = makeFixture();
  ASSERT_TRUE(fixture.module);
  llvm::Expected<std::unique_ptr<MappingTarget>> target = movementTarget();
  ASSERT_TRUE(static_cast<bool>(target)) << llvm::toString(target.takeError());

  llvm::Expected<CoveringPlan> plan =
      selectPlan(*fixture.context, *fixture.module, **target);
  ASSERT_TRUE(static_cast<bool>(plan)) << llvm::toString(plan.takeError());
  ASSERT_FALSE(plan->connectionPlans.empty());

  // Turn the real movement into a transform-only connection: no hop, and a
  // conversion with a solved map on each side.
  mlir::MLIRContext *context = fixture.context.get();
  PlanConnection &connection = plan->connectionPlans.front();
  connection.kind = ConnectionKind::LayoutTransform;
  connection.route.resize(1); // same memory: nothing to move
  LayoutTransform transform;
  transform.srcLayout = "t.plain";
  transform.dstLayout = "t.blocked";
  transform.srcMap = mlir::AffineMap::getMultiDimIdentityMap(2, context);
  transform.dstMap = mlir::AffineMap::get(
      2, 0,
      {mlir::getAffineDimExpr(0, context),
       mlir::getAffineBinaryOpExpr(mlir::AffineExprKind::FloorDiv,
                                   mlir::getAffineDimExpr(1, context),
                                   mlir::getAffineConstantExpr(8, context))},
      context);
  connection.transform = transform;

  const size_t copiesBefore = countOps(*fixture.module, "micro.async_copy");
  const size_t transformsBefore = countOps(*fixture.module, "micro.transform");

  llvm::Expected<BoundPlan> bound =
      bindCanonical(*fixture.module, *plan, **target);
  ASSERT_TRUE(static_cast<bool>(bound)) << llvm::toString(bound.takeError());
  for (const std::string &note : bound->unmaterialized)
    ADD_FAILURE() << note;
  EXPECT_TRUE(bound->unmaterialized.empty());
  // A transform-only connection emits no copy, and exactly one transform.
  EXPECT_EQ(countOps(*bound->module, "micro.async_copy"), copiesBefore);
  EXPECT_EQ(countOps(*bound->module, "micro.transform"), transformsBefore + 1);
  EXPECT_FALSE(
      static_cast<bool>(verifyMappedMicroIR(*bound->module, **target)));
}

// A `TransferAndTransform` moves the value *and* converts it: the copies land
// it in the consumer's memory and one `micro.transform` re-represents it there.
TEST(PlanBinder, MaterializesATransferAndTransformAsCopiesPlusATransform) {
  Fixture fixture = makeFixture();
  ASSERT_TRUE(fixture.module);
  llvm::Expected<std::unique_ptr<MappingTarget>> target = movementTarget();
  ASSERT_TRUE(static_cast<bool>(target)) << llvm::toString(target.takeError());

  llvm::Expected<CoveringPlan> plan =
      selectPlan(*fixture.context, *fixture.module, **target);
  ASSERT_TRUE(static_cast<bool>(plan)) << llvm::toString(plan.takeError());
  ASSERT_FALSE(plan->connectionPlans.empty());

  mlir::MLIRContext *context = fixture.context.get();
  PlanConnection &connection = plan->connectionPlans.front();
  connection.kind = ConnectionKind::TransferAndTransform;
  const size_t hops = connection.route.size() - 1;
  LayoutTransform transform;
  transform.srcLayout = "t.plain";
  transform.dstLayout = "t.blocked";
  transform.srcMap = mlir::AffineMap::getMultiDimIdentityMap(2, context);
  connection.transform = transform;

  const size_t copiesBefore = countOps(*fixture.module, "micro.async_copy");

  llvm::Expected<BoundPlan> bound =
      bindCanonical(*fixture.module, *plan, **target);
  ASSERT_TRUE(static_cast<bool>(bound)) << llvm::toString(bound.takeError());
  EXPECT_TRUE(bound->unmaterialized.empty());
  EXPECT_EQ(countOps(*bound->module, "micro.async_copy"), copiesBefore + hops);
  EXPECT_EQ(countOps(*bound->module, "micro.transform"), 1u);
  EXPECT_FALSE(
      static_cast<bool>(verifyMappedMicroIR(*bound->module, **target)));
}

// The executable contract refuses exactly the plan the partial contract
// reports: a caller that will hand the result to a backend must not receive IR
// that silently omits a selected decision, while analysis/reporting keeps the
// partial plan.
TEST(PlanBinder, ExecutableContractRefusesAPlanThatOmitsADecision) {
  Fixture fixture = makeFixture();
  ASSERT_TRUE(fixture.module);
  ASSERT_NE(fixture.target, nullptr);

  // A `Reduce` connection has no Micro operation form, so it is the decision
  // the partial contract reports and the executable contract refuses.
  CoveringPlan plan;
  plan.id = 42;
  PlanConnection connection;
  connection.id = 1;
  connection.value = 7;
  connection.kind = ConnectionKind::Reduce;
  plan.connectionPlans.push_back(connection);

  // The partial contract binds it and reports the omission.
  llvm::Expected<BoundPlan> partial =
      bindCanonical(*fixture.module, plan, *fixture.target);
  ASSERT_TRUE(static_cast<bool>(partial))
      << llvm::toString(partial.takeError());
  EXPECT_FALSE(partial->unmaterialized.empty());

  // The executable contract refuses it, naming the decision.
  llvm::Expected<BoundPlan> executable = bindCanonical(
      *fixture.module, plan, *fixture.target, BindContract::Executable);
  ASSERT_FALSE(static_cast<bool>(executable));
  std::string error = llvm::toString(executable.takeError());
  EXPECT_NE(error.find("not fully executable"), std::string::npos) << error;
  EXPECT_NE(error.find("reduce_not_materialized"), std::string::npos) << error;
}

TEST(PlanBinder, ReduceIsStillReportedBecauseItHasNoMicroOperationForm) {
  // `Reduce` (a gather) is not a movement, so the binder emits nothing for it
  // and reports it rather than dropping it. `Replicate`, by contrast, is now
  // materialized (see below): it is a fan-out copy, the same chain a movement
  // uses. The plan is built by hand because the placement layer does not
  // produce a `Reduce` yet -- the contract is what is under test.
  Fixture fixture = makeFixture();
  ASSERT_TRUE(fixture.module);
  ASSERT_NE(fixture.target, nullptr);

  CoveringPlan plan;
  plan.id = 1;
  PlanConnection connection;
  connection.id = 1;
  connection.value = 9;
  connection.kind = ConnectionKind::Reduce;
  plan.connectionPlans.push_back(connection);

  llvm::Expected<BoundPlan> bound =
      bindCanonical(*fixture.module, plan, *fixture.target);
  ASSERT_TRUE(static_cast<bool>(bound)) << llvm::toString(bound.takeError());
  ASSERT_EQ(bound->unmaterialized.size(), 1u);
  EXPECT_EQ(bound->unmaterialized.front(), "value 9: reduce_not_materialized");
}

TEST(PlanBinder, MaterializesAReplicateConnectionAsACopyChain) {
  // A `Replicate` is a fan-out copy: the same copy chain a movement emits,
  // serving one destination group. It is materialized, not reported.
  Fixture fixture = makeFixture();
  ASSERT_TRUE(fixture.module);
  llvm::Expected<std::unique_ptr<MappingTarget>> target = movementTarget();
  ASSERT_TRUE(static_cast<bool>(target)) << llvm::toString(target.takeError());

  llvm::Expected<CoveringPlan> plan =
      selectPlan(*fixture.context, *fixture.module, **target);
  ASSERT_TRUE(static_cast<bool>(plan)) << llvm::toString(plan.takeError());
  ASSERT_FALSE(plan->connectionPlans.empty());
  ASSERT_EQ(plan->connectionPlans.front().kind, ConnectionKind::Transfer);
  const size_t hops = plan->connectionPlans.front().route.size() - 1;
  for (PlanConnection &connection : plan->connectionPlans)
    connection.kind = ConnectionKind::Replicate;

  // The source kernel already carries an `async_copy`, so the emitted chain is
  // counted against that baseline (as `EmitsACopyAndWaitWhenTheValueMustMove`
  // does).
  const size_t copiesBefore = countOps(*fixture.module, "micro.async_copy");

  llvm::Expected<BoundPlan> bound =
      bindCanonical(*fixture.module, *plan, **target);
  ASSERT_TRUE(static_cast<bool>(bound)) << llvm::toString(bound.takeError());
  EXPECT_TRUE(bound->unmaterialized.empty());
  EXPECT_EQ(countOps(*bound->module, "micro.async_copy"), copiesBefore + hops);
  EXPECT_FALSE(
      static_cast<bool>(verifyMappedMicroIR(*bound->module, **target)));
}

TEST(PlanBinder, RewiresOnlyTheConsumersItsConnectionNames) {
  // Rewiring is scoped to the connection's own recorded endpoints. A connection
  // with no recorded endpoint therefore emits its copy but redirects nobody --
  // the reader keeps reading the original value. (The old binder rewired
  // *every* reader of the value, which is wrong as soon as two connections
  // carry one value along different routes.)
  Fixture fixture = makeFixture();
  ASSERT_TRUE(fixture.module);
  llvm::Expected<std::unique_ptr<MappingTarget>> target = movementTarget();
  ASSERT_TRUE(static_cast<bool>(target)) << llvm::toString(target.takeError());

  llvm::Expected<CoveringPlan> plan =
      selectPlan(*fixture.context, *fixture.module, **target);
  ASSERT_TRUE(static_cast<bool>(plan)) << llvm::toString(plan.takeError());
  ASSERT_FALSE(plan->connectionPlans.empty());
  for (PlanConnection &connection : plan->connectionPlans) {
    connection.consumers.clear();
    connection.consumerPorts.clear();
  }

  llvm::Expected<BoundPlan> bound =
      bindCanonical(*fixture.module, *plan, **target);
  ASSERT_TRUE(static_cast<bool>(bound)) << llvm::toString(bound.takeError());
  EXPECT_TRUE(bound->unmaterialized.empty());

  // The copy exists, but nothing reads it: no consumer was named.
  Operation *copy = nullptr;
  bound->kernel->walk([&](Operation *op) {
    if (op->getName().getStringRef() == "micro.async_copy")
      copy = op;
  });
  ASSERT_NE(copy, nullptr);
  EXPECT_TRUE(copy->getResult(0).use_empty());
}

TEST(PlanBinder, EmitsOneChainPerRouteAndSurvivesBoth) {
  // One value reached by two connections with *different* routes gets one copy
  // chain per route instead of being reported as un-materializable: each chain
  // lands the value in its own memory, and each rewires only its own consumers.
  Fixture fixture = makeFixture();
  ASSERT_TRUE(fixture.module);
  llvm::Expected<std::unique_ptr<MappingTarget>> target = twoRouteTarget();
  ASSERT_TRUE(static_cast<bool>(target)) << llvm::toString(target.takeError());

  llvm::Expected<CoveringPlan> plan =
      selectPlan(*fixture.context, *fixture.module, **target);
  ASSERT_TRUE(static_cast<bool>(plan)) << llvm::toString(plan.takeError());
  ASSERT_FALSE(plan->connectionPlans.empty());
  const PlanConnection &first = plan->connectionPlans.front();
  ASSERT_EQ(first.kind, ConnectionKind::Transfer);
  ASSERT_GE(first.route.size(), 2u);

  // The other legal route between the same endpoints: whichever the search did
  // not take, the direct hop or the staged one.
  const llvm::SmallVector<MemoryNodeId> direct{"sram.0", "dram.0"};
  const llvm::SmallVector<MemoryNodeId> staged{"sram.0", "stage.0", "dram.0"};
  PlanConnection duplicate = first;
  duplicate.id = first.id + 1;
  duplicate.route = (first.route == direct) ? staged : direct;
  // The duplicate lands the value on a second route; it must not claim the
  // first connection's endpoints, or both movements would fight to be what the
  // same consumer reads. Endpoint rewiring is the authority, so it names none.
  duplicate.consumerPorts.clear();
  duplicate.consumers.clear();
  plan->connectionPlans.push_back(duplicate);

  const size_t expectedCopies =
      (first.route.size() - 1) + (duplicate.route.size() - 1);
  const size_t copiesBefore = countOps(*fixture.module, "micro.async_copy");

  llvm::Expected<BoundPlan> bound =
      bindCanonical(*fixture.module, *plan, **target);
  ASSERT_TRUE(static_cast<bool>(bound)) << llvm::toString(bound.takeError());
  EXPECT_TRUE(bound->unmaterialized.empty());
  EXPECT_EQ(countOps(*bound->module, "micro.async_copy"),
            copiesBefore + expectedCopies);
  EXPECT_FALSE(
      static_cast<bool>(verifyMappedMicroIR(*bound->module, **target)));
}

TEST(PlanBinder, MergesADuplicateValueConnectionWithTheSameRoute) {
  // The same duplicate with the same route is a genuine fan-out merge: one
  // chain serves both connections, so nothing is reported.
  Fixture fixture = makeFixture();
  ASSERT_TRUE(fixture.module);
  llvm::Expected<std::unique_ptr<MappingTarget>> target = movementTarget();
  ASSERT_TRUE(static_cast<bool>(target)) << llvm::toString(target.takeError());

  llvm::Expected<CoveringPlan> plan =
      selectPlan(*fixture.context, *fixture.module, **target);
  ASSERT_TRUE(static_cast<bool>(plan)) << llvm::toString(plan.takeError());
  ASSERT_FALSE(plan->connectionPlans.empty());
  const PlanConnection &first = plan->connectionPlans.front();
  ASSERT_EQ(first.kind, ConnectionKind::Transfer);

  PlanConnection duplicate = first;
  duplicate.id = first.id + 1;
  plan->connectionPlans.push_back(duplicate); // identical route

  llvm::Expected<BoundPlan> bound =
      bindCanonical(*fixture.module, *plan, **target);
  ASSERT_TRUE(static_cast<bool>(bound)) << llvm::toString(bound.takeError());
  EXPECT_TRUE(bound->unmaterialized.empty());
}

//===----------------------------------------------------------------------===//
// Multi-hop materialization
//===----------------------------------------------------------------------===//

namespace {

/// A machine whose only way from DRAM to SRAM is through L2, so a routed
/// movement between them has two hops rather than one. The two memories live in
/// different visibility scopes, so the node reading SRAM cannot read DRAM and
/// the edge between them is a real transfer (see `kMovementMachine`).
constexpr llvm::StringLiteral kTwoHopMachine = R"yaml(
schema: llk.machine.v2
target: two-hop
clock_hz: 1000000000
worker_threads: 2
executors:
  - id: cluster.a
    kind: cluster
  - id: worker.a
    kind: worker
    parent: cluster.a
  - id: cluster.b
    kind: cluster
  - id: worker.b
    kind: worker
    parent: cluster.b
memories:
  - id: dram.0
    kind: dram
    visible_from: cluster.b
    capacity_bytes: 1048576
    alignment_bytes: 64
    supported_layouts: [row_major]
    bandwidth_bytes_per_cycle: 32
    latency_cycles: 220
  - id: l2.0
    kind: l2
    visible_from: cluster.b
    capacity_bytes: 262144
    alignment_bytes: 64
    supported_layouts: [row_major]
    bandwidth_bytes_per_cycle: 64
    latency_cycles: 12
  - id: sram.0
    kind: sram
    visible_from: cluster.a
    capacity_bytes: 32768
    alignment_bytes: 64
    supported_layouts: [row_major]
    bandwidth_bytes_per_cycle: 64
    latency_cycles: 4
compute:
  - id: vpu
    kind: vector_engine
    attached_to: worker.a
    element_types: [f32]
    shapes: [[8]]
    lanes: {f32: 8}
transfer_engines:
  - id: dma.b
    kind: dma
    attached_to: cluster.b
    count: 1
    max_outstanding: 1
links:
  - id: dram_to_l2.0
    source: dram.0
    destination: l2.0
    bandwidth_bytes_per_cycle: 64
    latency_cycles: 220
    transaction_bytes: 64
    transfer_engines: [dma.b]
  - id: l2_to_sram.0
    source: l2.0
    destination: sram.0
    bandwidth_bytes_per_cycle: 64
    latency_cycles: 12
    transaction_bytes: 64
    transfer_engines: [dma.b]
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

  llvm::Expected<BoundPlan> bound =
      bindCanonical(*fixture.module, *plan, **target);
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

//===----------------------------------------------------------------------===//
// Full rule-legality verification (A3)
//===----------------------------------------------------------------------===//

namespace {

/// A machine with two `worker` executors, only one of which carries a
/// `vector_engine`. A rule that requires the capability is therefore legal on
/// exactly one of them, so verification can be pointed at a worker that has the
/// right *kind* but not the *capability*.
constexpr llvm::StringLiteral kComputeMachine = R"yaml(
schema: llk.machine.v2
target: compute
clock_hz: 1000000000
worker_threads: 2
executors:
  - id: cluster.c
    kind: cluster
  - id: worker.vec
    kind: worker
    parent: cluster.c
  - id: worker.plain
    kind: worker
    parent: cluster.c
memories:
  - id: sram.0
    kind: sram
    visible_from: cluster.c
    capacity_bytes: 1048576
    alignment_bytes: 64
    supported_layouts: [row_major]
    bandwidth_bytes_per_cycle: 64
    latency_cycles: 4
compute:
  - id: vpu
    kind: vector_engine
    attached_to: worker.vec
    element_types: [f32]
    shapes: [[8]]
    lanes: {f32: 8}
)yaml";

/// One rule whose executor requirement both workers satisfy but whose compute
/// requirement only one does.
constexpr llvm::StringLiteral kComputeRules = R"llkmap(
rule c.vector {
  match micro.vector(op = "add");
  require executor kind worker;
  require compute kind vector_engine;
  bundle "c.vector";
  emit "e1";
}
)llkmap";

/// A kernel whose only workload node is the vector add.
constexpr llvm::StringLiteral kVectorOnlyKernel = R"mlir(
module {
  micro.kernel @vector {
    %0 = tensor.empty() : tensor<8x8xf32>
    %t = micro.tile_view %0 {shape = array<i64: 8, 8>} : tensor<8x8xf32> -> !micro.tile<8x8xf32, memory = #micro.memory<sram>>
    %r = micro.vector "add" %t, %t : !micro.tile<8x8xf32, memory = #micro.memory<sram>>, !micro.tile<8x8xf32, memory = #micro.memory<sram>> -> !micro.tile<8x8xf32, memory = #micro.memory<sram>>
  }
}
)mlir";

llvm::Expected<std::unique_ptr<MappingTarget>> computeTarget() {
  llvm::Expected<mlir::llk::machine::MachineModel> machine =
      mlir::llk::machine::parseMachineModel(kComputeMachine, "<test>");
  if (!machine)
    return machine.takeError();
  llvm::Expected<LayoutRegistry> layouts = parseLayoutText("", "<test>");
  if (!layouts)
    return layouts.takeError();
  llvm::Expected<RuleRegistry> rules = parseRuleText(kComputeRules, "<test>");
  if (!rules)
    return rules.takeError();
  return std::make_unique<FileMappingTarget>(
      "compute", std::move(*machine), std::move(*layouts), std::move(*rules),
      std::vector<std::string>{"e1"});
}

/// Rewrites every recorded executor in `module` to `executor`.
void rewriteRecordedExecutor(ModuleOp module, MLIRContext &context,
                             llvm::StringRef executor) {
  module->walk([&](Operation *op) {
    auto mapping = op->getAttrOfType<DictionaryAttr>("micro.mapping");
    if (!mapping)
      return;
    llvm::SmallVector<NamedAttribute> attributes(mapping.begin(),
                                                 mapping.end());
    for (NamedAttribute &attribute : attributes)
      if (attribute.getName() == "executor")
        attribute = NamedAttribute(attribute.getName(),
                                   StringAttr::get(&context, executor));
    op->setAttr("micro.mapping", DictionaryAttr::get(&context, attributes));
  });
}

} // namespace

// The unmodified mapped fixture is the legal control for every rejection
// below: shared verification must not turn a valid selection into a failure.
TEST(PlanBinder, MappedFixtureVerifiesWithFullRuleLegality) {
  Fixture fixture = makeFixture();
  ASSERT_TRUE(fixture.module);
  ASSERT_NE(fixture.target, nullptr);
  llvm::Expected<CoveringPlan> plan =
      selectPlan(*fixture.context, *fixture.module, *fixture.target);
  ASSERT_TRUE(static_cast<bool>(plan)) << llvm::toString(plan.takeError());
  llvm::Expected<BoundPlan> bound =
      bindCanonical(*fixture.module, *plan, *fixture.target);
  ASSERT_TRUE(static_cast<bool>(bound)) << llvm::toString(bound.takeError());

  EXPECT_FALSE(
      static_cast<bool>(verifyMappedMicroIR(*bound->module, *fixture.target)));
}

// A recorded rule whose predicate no longer holds for the operation it labels
// is not a mapping: verification must re-evaluate the rule's predicates, not
// only confirm that its mnemonic matches. `micro.vector "mul"` labelled with
// the add rule is the canonical case.
TEST(PlanBinder, RejectsASelectedRuleWhosePredicateNoLongerMatches) {
  Fixture f = makeFixture();
  ASSERT_TRUE(f.module);
  ASSERT_NE(f.target, nullptr);
  llvm::Expected<CoveringPlan> p = selectPlan(*f.context, *f.module, *f.target);
  ASSERT_TRUE(bool(p)) << llvm::toString(p.takeError());
  llvm::Expected<BoundPlan> b = bindCanonical(*f.module, *p, *f.target);
  ASSERT_TRUE(bool(b)) << llvm::toString(b.takeError());
  b->module->walk([&](Operation *op) {
    if (op->getName().getStringRef() == "micro.vector")
      op->setAttr("op", StringAttr::get(f.context.get(), "mul"));
  });
  llvm::Error e = verifyMappedMicroIR(*b->module, *f.target);
  ASSERT_TRUE(bool(e));
  EXPECT_NE(llvm::toString(std::move(e)).find("no_matching_rule"),
            std::string::npos);
}

// The recorded executor must satisfy the rule's executor *kind*, not merely
// exist in the machine: `veng.0` is a real executor of kind `vector_engine`,
// which no worker rule accepts.
TEST(PlanBinder, RejectsAnExecutorOfTheWrongKind) {
  Fixture fixture = makeFixture();
  ASSERT_TRUE(fixture.module);
  ASSERT_NE(fixture.target, nullptr);
  llvm::Expected<CoveringPlan> plan =
      selectPlan(*fixture.context, *fixture.module, *fixture.target);
  ASSERT_TRUE(static_cast<bool>(plan)) << llvm::toString(plan.takeError());
  llvm::Expected<BoundPlan> bound =
      bindCanonical(*fixture.module, *plan, *fixture.target);
  ASSERT_TRUE(static_cast<bool>(bound)) << llvm::toString(bound.takeError());

  rewriteRecordedExecutor(*bound->module, *fixture.context, "veng.0");

  llvm::Error e = verifyMappedMicroIR(*bound->module, *fixture.target);
  ASSERT_TRUE(bool(e));
  EXPECT_NE(llvm::toString(std::move(e)).find("no_legal_executor"),
            std::string::npos);
}

// The recorded executor must actually offer the capability a rule requires:
// `worker.plain` is a legal worker, but no `vector_engine` is attached to it.
TEST(PlanBinder, RejectsAComputeRequirementTheExecutorCannotSupply) {
  MLIRContext context;
  context.getOrLoadDialect<micro::MicroDialect>();
  context.getOrLoadDialect<tensor::TensorDialect>();
  OwningOpRef<ModuleOp> module =
      parseSourceString<ModuleOp>(kVectorOnlyKernel, &context);
  ASSERT_TRUE(module);
  llvm::Expected<std::unique_ptr<MappingTarget>> target = computeTarget();
  ASSERT_TRUE(static_cast<bool>(target)) << llvm::toString(target.takeError());

  llvm::Expected<CoveringPlan> plan = selectPlan(context, *module, **target);
  ASSERT_TRUE(static_cast<bool>(plan)) << llvm::toString(plan.takeError());
  llvm::Expected<BoundPlan> bound = bindCanonical(*module, *plan, **target);
  ASSERT_TRUE(static_cast<bool>(bound)) << llvm::toString(bound.takeError());

  // The search placed the vector on `worker.vec`, the only worker with a
  // `vector_engine`. Naming `worker.plain` keeps the executor kind legal but
  // drops the capability.
  rewriteRecordedExecutor(*bound->module, context, "worker.plain");

  llvm::Error e = verifyMappedMicroIR(*bound->module, **target);
  ASSERT_TRUE(bool(e));
  EXPECT_NE(llvm::toString(std::move(e)).find("unsupported_compute_fragment"),
            std::string::npos);
}

// A `micro.mapping` must name an operation the workload graph classifies as a
// node: without the original endpoints the recorded rule cannot be re-checked,
// so pointing the metadata at a transparent op (here a `micro.tile_view`) is
// rejected rather than silently downgraded to the match-operation backstop.
TEST(PlanBinder, RejectsAMappedOperationThatIsNotAWorkloadNode) {
  Fixture fixture = makeFixture();
  ASSERT_TRUE(fixture.module);
  ASSERT_NE(fixture.target, nullptr);
  llvm::Expected<CoveringPlan> plan =
      selectPlan(*fixture.context, *fixture.module, *fixture.target);
  ASSERT_TRUE(static_cast<bool>(plan)) << llvm::toString(plan.takeError());
  llvm::Expected<BoundPlan> bound =
      bindCanonical(*fixture.module, *plan, *fixture.target);
  ASSERT_TRUE(static_cast<bool>(bound)) << llvm::toString(bound.takeError());

  bool stamped = false;
  bound->module->walk([&](Operation *op) {
    if (op->getName().getStringRef() != "micro.tile_view")
      return;
    mlir::MLIRContext *context = fixture.context.get();
    op->setAttr(
        "micro.mapping",
        DictionaryAttr::get(
            context,
            {NamedAttribute(StringAttr::get(context, "rule"),
                            StringAttr::get(context, "avx2.vector_add"))}));
    stamped = true;
  });
  ASSERT_TRUE(stamped);

  llvm::Error e = verifyMappedMicroIR(*bound->module, *fixture.target);
  ASSERT_TRUE(bool(e));
  EXPECT_NE(llvm::toString(std::move(e)).find("invalid_mapping_metadata"),
            std::string::npos);
}

//===----------------------------------------------------------------------===//
// Materialized-movement exemption validation (A5)
//===----------------------------------------------------------------------===//

namespace {

/// The binder-emitted movement copy in `module`: an `micro.async_copy` the
/// binder stamped with a destination node. The source kernel's own copy (when
/// it has one) carries no such stamp.
Operation *materializedCopy(ModuleOp module) {
  Operation *copy = nullptr;
  module->walk([&](Operation *op) {
    if (op->getName().getStringRef() == "micro.async_copy" &&
        op->hasAttr("micro.dst_node"))
      copy = op;
  });
  return copy;
}

/// Replaces the `engines` of every `micro.routes` entry with `engines`. Used to
/// point a genuine materialized movement at a transfer engine the machine does
/// not support.
void rewriteAllRouteEngines(Operation *kernel, MLIRContext &context,
                            llvm::ArrayRef<llvm::StringRef> engines) {
  auto routes = kernel->getAttrOfType<ArrayAttr>("micro.routes");
  ASSERT_TRUE(routes) << "the bound kernel carries no micro.routes";
  llvm::SmallVector<Attribute> rewritten;
  for (Attribute element : routes) {
    auto route = dyn_cast<DictionaryAttr>(element);
    ASSERT_TRUE(route) << "a route entry is not a dictionary";
    NamedAttrList attributes(route);
    llvm::SmallVector<Attribute> engineAttrs;
    for (llvm::StringRef engine : engines)
      engineAttrs.push_back(StringAttr::get(&context, engine));
    attributes.set("engines", ArrayAttr::get(&context, engineAttrs));
    rewritten.push_back(attributes.getDictionary(&context));
  }
  kernel->setAttr("micro.routes", ArrayAttr::get(&context, rewritten));
}

} // namespace

// Regression (issue #67, stage A): the completeness walk exempted *any* op
// carrying `micro.value` from the requirement that a workload operation be
// mapped. Deleting a compute op's `micro.mapping` and stamping an arbitrary
// movement attribute onto it therefore verified successfully. The exemption
// must require a recognized materialized movement op whose connection
// provenance resolves, so `micro.value` alone grants nothing.
TEST(PlanBinder, ValueStampCannotExemptAnUnmappedComputeOperation) {
  auto f = makeFixture();
  ASSERT_TRUE(f.module);
  ASSERT_TRUE(f.target);
  auto p = selectPlan(*f.context, *f.module, *f.target);
  ASSERT_TRUE(bool(p)) << llvm::toString(p.takeError());
  auto b = bindCanonical(*f.module, *p, *f.target);
  ASSERT_TRUE(bool(b)) << llvm::toString(b.takeError());
  auto legal = verifyMappedMicroIR(*b->module, *f.target);
  ASSERT_FALSE(bool(legal)) << llvm::toString(std::move(legal));
  unsigned mutations = 0;
  b->module->walk([&](Operation *op) {
    if (op->getName().getStringRef() != "micro.vector")
      return;
    op->removeAttr("micro.mapping");
    op->setAttr("micro.value",
                IntegerAttr::get(IntegerType::get(f.context.get(), 64), 0));
    ++mutations;
  });
  ASSERT_EQ(mutations, 1u);
  auto error = verifyMappedMicroIR(*b->module, *f.target);
  ASSERT_TRUE(bool(error));
  EXPECT_NE(llvm::toString(std::move(error)).find("no_matching_rule"),
            std::string::npos);
}

// The legal control for every rejection below: a genuine binder-emitted
// movement is still complete exactly because its connection provenance
// resolves. Validation must not turn a valid materialized connection into a
// completeness failure.
TEST(PlanBinder, MaterializedMovementKeepsItsCompletenessExemption) {
  Fixture f = makeFixture();
  ASSERT_TRUE(f.module);
  llvm::Expected<std::unique_ptr<MappingTarget>> target = movementTarget();
  ASSERT_TRUE(bool(target)) << llvm::toString(target.takeError());
  auto p = selectPlan(*f.context, *f.module, **target);
  ASSERT_TRUE(bool(p)) << llvm::toString(p.takeError());
  auto b = bindCanonical(*f.module, *p, **target);
  ASSERT_TRUE(bool(b)) << llvm::toString(b.takeError());
  ASSERT_NE(materializedCopy(*b->module), nullptr);
  auto error = verifyMappedMicroIR(*b->module, **target);
  EXPECT_FALSE(bool(error)) << llvm::toString(std::move(error));
}

// A forged value stamp: an emitted copy's connection bookkeeping is removed,
// leaving the bare `micro.value` the old exemption trusted. It must not grant
// the exemption.
TEST(PlanBinder, RejectsAMovementWhoseConnectionStampIsAbsent) {
  Fixture f = makeFixture();
  ASSERT_TRUE(f.module);
  llvm::Expected<std::unique_ptr<MappingTarget>> target = movementTarget();
  ASSERT_TRUE(bool(target)) << llvm::toString(target.takeError());
  auto p = selectPlan(*f.context, *f.module, **target);
  ASSERT_TRUE(bool(p)) << llvm::toString(p.takeError());
  auto b = bindCanonical(*f.module, *p, **target);
  ASSERT_TRUE(bool(b)) << llvm::toString(b.takeError());
  Operation *copy = materializedCopy(*b->module);
  ASSERT_NE(copy, nullptr);
  copy->removeAttr("micro.connection");
  llvm::Error e = verifyMappedMicroIR(*b->module, **target);
  ASSERT_TRUE(bool(e));
  EXPECT_NE(llvm::toString(std::move(e)).find("invalid_mapping_metadata"),
            std::string::npos);
}

// A movement naming a connection the kernel's `micro.routes` does not declare
// is spoofed, not materialized.
TEST(PlanBinder, RejectsAMovementNamingAnUnknownConnection) {
  Fixture f = makeFixture();
  ASSERT_TRUE(f.module);
  llvm::Expected<std::unique_ptr<MappingTarget>> target = movementTarget();
  ASSERT_TRUE(bool(target)) << llvm::toString(target.takeError());
  auto p = selectPlan(*f.context, *f.module, **target);
  ASSERT_TRUE(bool(p)) << llvm::toString(p.takeError());
  auto b = bindCanonical(*f.module, *p, **target);
  ASSERT_TRUE(bool(b)) << llvm::toString(b.takeError());
  Operation *copy = materializedCopy(*b->module);
  ASSERT_NE(copy, nullptr);
  copy->setAttr("micro.connection",
                IntegerAttr::get(IntegerType::get(f.context.get(), 64), 9999));
  llvm::Error e = verifyMappedMicroIR(*b->module, **target);
  ASSERT_TRUE(bool(e));
  EXPECT_NE(llvm::toString(std::move(e)).find("invalid_mapping_metadata"),
            std::string::npos);
}

// A hop index outside the resolved connection's route is spoofed bookkeeping.
TEST(PlanBinder, RejectsAMovementNamingAHopOutsideItsRoute) {
  Fixture f = makeFixture();
  ASSERT_TRUE(f.module);
  llvm::Expected<std::unique_ptr<MappingTarget>> target = movementTarget();
  ASSERT_TRUE(bool(target)) << llvm::toString(target.takeError());
  auto p = selectPlan(*f.context, *f.module, **target);
  ASSERT_TRUE(bool(p)) << llvm::toString(p.takeError());
  auto b = bindCanonical(*f.module, *p, **target);
  ASSERT_TRUE(bool(b)) << llvm::toString(b.takeError());
  Operation *copy = materializedCopy(*b->module);
  ASSERT_NE(copy, nullptr);
  copy->setAttr("micro.hop",
                IntegerAttr::get(IntegerType::get(f.context.get(), 64), 99));
  llvm::Error e = verifyMappedMicroIR(*b->module, **target);
  ASSERT_TRUE(bool(e));
  EXPECT_NE(llvm::toString(std::move(e)).find("invalid_mapping_metadata"),
            std::string::npos);
}

// The destination node stamp must be the memory the resolved hop actually lands
// in; a copy claiming some other destination is not the connection it names.
TEST(PlanBinder, RejectsAMovementWhoseDestinationDoesNotMatchItsRoute) {
  Fixture f = makeFixture();
  ASSERT_TRUE(f.module);
  llvm::Expected<std::unique_ptr<MappingTarget>> target = movementTarget();
  ASSERT_TRUE(bool(target)) << llvm::toString(target.takeError());
  auto p = selectPlan(*f.context, *f.module, **target);
  ASSERT_TRUE(bool(p)) << llvm::toString(p.takeError());
  auto b = bindCanonical(*f.module, *p, **target);
  ASSERT_TRUE(bool(b)) << llvm::toString(b.takeError());
  Operation *copy = materializedCopy(*b->module);
  ASSERT_NE(copy, nullptr);
  copy->setAttr("micro.dst_node", StringAttr::get(f.context.get(), "sram.9"));
  llvm::Error e = verifyMappedMicroIR(*b->module, **target);
  ASSERT_TRUE(bool(e));
  EXPECT_NE(llvm::toString(std::move(e)).find("invalid_mapping_metadata"),
            std::string::npos);
}

// A movement hop carried by a transfer engine the machine does not declare is
// not executable; the resolved route's engine set must be supported.
TEST(PlanBinder, RejectsAMovementOnAnUnsupportedTransferEngine) {
  Fixture f = makeFixture();
  ASSERT_TRUE(f.module);
  llvm::Expected<std::unique_ptr<MappingTarget>> target = movementTarget();
  ASSERT_TRUE(bool(target)) << llvm::toString(target.takeError());
  auto p = selectPlan(*f.context, *f.module, **target);
  ASSERT_TRUE(bool(p)) << llvm::toString(p.takeError());
  auto b = bindCanonical(*f.module, *p, **target);
  ASSERT_TRUE(bool(b)) << llvm::toString(b.takeError());
  ASSERT_NE(materializedCopy(*b->module), nullptr);
  rewriteAllRouteEngines(b->kernel, *f.context, {"no_such_engine"});
  llvm::Error e = verifyMappedMicroIR(*b->module, **target);
  ASSERT_TRUE(bool(e));
  // The materialized-movement path resolves the route and rejects the
  // unsupported engine before the route walk reaches it.
  std::string text = llvm::toString(std::move(e));
  EXPECT_NE(text.find("invalid_mapping_metadata"), std::string::npos) << text;
  EXPECT_NE(text.find("no_such_engine"), std::string::npos) << text;
}

//===----------------------------------------------------------------------===//
// Persisted schema-v2 selected state (task B1)
//===----------------------------------------------------------------------===//

namespace {

/// Rewrites the first `consumer_ports` endpoint index of every route.
void rewriteFirstConsumerPortIndex(Operation *kernel, MLIRContext &context,
                                   unsigned newIndex) {
  auto routes = kernel->getAttrOfType<ArrayAttr>("micro.routes");
  ASSERT_TRUE(routes);
  llvm::SmallVector<Attribute> rewritten;
  for (Attribute element : routes) {
    auto route = dyn_cast<DictionaryAttr>(element);
    ASSERT_TRUE(route);
    NamedAttrList attributes(route);
    auto ports = route.getAs<ArrayAttr>("consumer_ports");
    ASSERT_TRUE(ports);
    llvm::SmallVector<Attribute> portAttrs;
    for (size_t i = 0; i < ports.size(); ++i) {
      auto port = dyn_cast<DictionaryAttr>(ports[i]);
      ASSERT_TRUE(port);
      if (i == 0) {
        NamedAttrList fields(port);
        fields.set("index",
                   IntegerAttr::get(IntegerType::get(&context, 64), newIndex));
        portAttrs.push_back(fields.getDictionary(&context));
      } else {
        portAttrs.push_back(port);
      }
    }
    attributes.set("consumer_ports", ArrayAttr::get(&context, portAttrs));
    rewritten.push_back(attributes.getDictionary(&context));
  }
  kernel->setAttr("micro.routes", ArrayAttr::get(&context, rewritten));
}

} // namespace

TEST(PlanBinder, RoundTripsConnectionEndpointsAndResourceBindings) {
  Fixture f = makeFixture();
  ASSERT_TRUE(f.module);
  llvm::Expected<std::unique_ptr<MappingTarget>> target = movementTarget();
  ASSERT_TRUE(bool(target)) << llvm::toString(target.takeError());
  auto p = selectPlan(*f.context, *f.module, **target);
  ASSERT_TRUE(bool(p)) << llvm::toString(p.takeError());
  ASSERT_FALSE(p->connectionPlans.empty());
  auto b = bindCanonical(*f.module, *p, **target);
  ASSERT_TRUE(bool(b)) << llvm::toString(b.takeError());

  auto decoded = decodeSelectedPlan(*b->module, **target);
  ASSERT_TRUE(bool(decoded)) << llvm::toString(decoded.takeError());
  EXPECT_EQ(decoded->schemaVersion, 2u);
  EXPECT_EQ(decoded->id, p->id);

  ASSERT_EQ(decoded->connectionPlans.size(), p->connectionPlans.size());
  for (size_t i = 0; i < p->connectionPlans.size(); ++i) {
    EXPECT_EQ(decoded->connectionPlans[i].id, p->connectionPlans[i].id);
    EXPECT_EQ(decoded->connectionPlans[i].kind, p->connectionPlans[i].kind);
    EXPECT_EQ(decoded->connectionPlans[i].route, p->connectionPlans[i].route);
    EXPECT_EQ(decoded->connectionPlans[i].engines,
              p->connectionPlans[i].engines);
    EXPECT_EQ(decoded->connectionPlans[i].consumers,
              p->connectionPlans[i].consumers);
    ASSERT_EQ(decoded->connectionPlans[i].consumerPorts,
              p->connectionPlans[i].consumerPorts);
    EXPECT_EQ(decoded->connectionPlans[i].producerPort,
              p->connectionPlans[i].producerPort);
    EXPECT_FALSE(p->connectionPlans[i].consumerPorts.empty());
  }
  ASSERT_EQ(decoded->placements.size(), p->placements.size());
  for (size_t i = 0; i < p->placements.size(); ++i) {
    EXPECT_EQ(decoded->placements[i].node, p->placements[i].node);
    EXPECT_EQ(decoded->placements[i].instance, p->placements[i].instance);
    EXPECT_EQ(decoded->placements[i].rule, p->placements[i].rule);
    EXPECT_EQ(decoded->placements[i].executor, p->placements[i].executor);
    EXPECT_EQ(decoded->placements[i].layouts.size(),
              p->placements[i].layouts.size());
    EXPECT_EQ(decoded->placements[i].layoutSolutions.size(),
              p->placements[i].layoutSolutions.size());
  }
}

// The solved layout assignment -- the concrete parameters and the affine map
// they substitute into -- must survive a round trip, not only the family name.
TEST(PlanBinder, RoundTripsSolvedLayoutAssignmentsAndMaps) {
  Fixture f = makeFixture();
  ASSERT_TRUE(f.module);
  ASSERT_NE(f.target, nullptr);
  auto p = selectPlan(*f.context, *f.module, *f.target);
  ASSERT_TRUE(bool(p)) << llvm::toString(p.takeError());
  auto b = bindCanonical(*f.module, *p, *f.target);
  ASSERT_TRUE(bool(b)) << llvm::toString(b.takeError());

  auto decoded = decodeSelectedPlan(*b->module, *f.target);
  ASSERT_TRUE(bool(decoded)) << llvm::toString(decoded.takeError());
  ASSERT_EQ(decoded->placements.size(), p->placements.size());

  size_t solvedCount = 0;
  for (size_t i = 0; i < p->placements.size(); ++i) {
    ASSERT_EQ(decoded->placements[i].layoutSolutions.size(),
              p->placements[i].layoutSolutions.size());
    for (const auto &entry : p->placements[i].layoutSolutions) {
      ++solvedCount;
      ASSERT_TRUE(decoded->placements[i].layoutSolutions.count(entry.first()));
      const SolvedLayout &actual =
          decoded->placements[i].layoutSolutions.lookup(entry.first());
      EXPECT_EQ(actual.layoutClass, entry.second.layoutClass);
      EXPECT_EQ(canonicalSearchValueString(actual.parameters),
                canonicalSearchValueString(entry.second.parameters));
      EXPECT_EQ(actual.map, entry.second.map);
      EXPECT_EQ(actual.port, entry.second.port);
    }
  }
  EXPECT_GT(solvedCount, 0u);
}

// A plan bound for one target must not decode against another: the recorded
// target content hash no longer matches.
TEST(PlanBinder, DecodeRejectsAChangedTarget) {
  Fixture f = makeFixture();
  ASSERT_TRUE(f.module);
  ASSERT_NE(f.target, nullptr);
  auto p = selectPlan(*f.context, *f.module, *f.target);
  ASSERT_TRUE(bool(p)) << llvm::toString(p.takeError());
  auto b = bindCanonical(*f.module, *p, *f.target);
  ASSERT_TRUE(bool(b)) << llvm::toString(b.takeError());

  llvm::Expected<std::unique_ptr<MappingTarget>> other = movementTarget();
  ASSERT_TRUE(bool(other)) << llvm::toString(other.takeError());
  auto decoded = decodeSelectedPlan(*b->module, **other);
  ASSERT_FALSE(bool(decoded));
  EXPECT_NE(llvm::toString(decoded.takeError()).find("target_hash"),
            std::string::npos);
}

// A frozen plan cannot bypass current graph verification: changing a semantic
// workload attribute changes the source-graph hash and decoding fails.
TEST(PlanBinder, DecodeRejectsAChangedSourceGraph) {
  Fixture f = makeFixture();
  ASSERT_TRUE(f.module);
  ASSERT_NE(f.target, nullptr);
  auto p = selectPlan(*f.context, *f.module, *f.target);
  ASSERT_TRUE(bool(p)) << llvm::toString(p.takeError());
  auto b = bindCanonical(*f.module, *p, *f.target);
  ASSERT_TRUE(bool(b)) << llvm::toString(b.takeError());

  b->module->walk([&](Operation *op) {
    if (op->getName().getStringRef() == "micro.vector")
      op->setAttr("op", StringAttr::get(f.context.get(), "mul"));
  });

  auto decoded = decodeSelectedPlan(*b->module, *f.target);
  ASSERT_FALSE(bool(decoded));
  EXPECT_NE(llvm::toString(decoded.takeError()).find("graph_hash"),
            std::string::npos);
}

// A recorded endpoint with an out-of-range index is malformed metadata, not a
// plan that resolves.
TEST(PlanBinder, DecodeRejectsAMalformedEndpointIndex) {
  Fixture f = makeFixture();
  ASSERT_TRUE(f.module);
  llvm::Expected<std::unique_ptr<MappingTarget>> target = movementTarget();
  ASSERT_TRUE(bool(target)) << llvm::toString(target.takeError());
  auto p = selectPlan(*f.context, *f.module, **target);
  ASSERT_TRUE(bool(p)) << llvm::toString(p.takeError());
  auto b = bindCanonical(*f.module, *p, **target);
  ASSERT_TRUE(bool(b)) << llvm::toString(b.takeError());

  rewriteFirstConsumerPortIndex(b->kernel, *f.context.get(), /*newIndex=*/99);
  auto decoded = decodeSelectedPlan(*b->module, **target);
  ASSERT_FALSE(bool(decoded));
}

// A plan recording a metadata schema this reader does not understand must not
// be replayed under these semantics.
TEST(PlanBinder, DecodeRejectsAnUnsupportedSchemaVersion) {
  Fixture f = makeFixture();
  ASSERT_TRUE(f.module);
  ASSERT_NE(f.target, nullptr);
  auto p = selectPlan(*f.context, *f.module, *f.target);
  ASSERT_TRUE(bool(p)) << llvm::toString(p.takeError());
  auto b = bindCanonical(*f.module, *p, *f.target);
  ASSERT_TRUE(bool(b)) << llvm::toString(b.takeError());

  auto plan = b->kernel->getAttrOfType<DictionaryAttr>("micro.plan");
  ASSERT_TRUE(plan);
  NamedAttrList updated(plan);
  updated.set("schema_version",
              IntegerAttr::get(IntegerType::get(f.context.get(), 64), 3));
  b->kernel->setAttr("micro.plan", updated.getDictionary(f.context.get()));

  auto decoded = decodeSelectedPlan(*b->module, *f.target);
  ASSERT_FALSE(bool(decoded));
  EXPECT_NE(llvm::toString(decoded.takeError()).find("unsupported"),
            std::string::npos);
}

//===----------------------------------------------------------------------===//
// Stage-A A4: the layout container (or an explicit no-layout marker) is
// required
//===----------------------------------------------------------------------===//

// Regression (issue #67, stage A, A4): deleting the whole `layout_parameters`
// container skipped solved-layout validation. Under schema v2 the container is
// required, so the deletion is rejected.
TEST(PlanBinder, RejectsADeletedLayoutContainer) {
  Fixture f = makeFixture();
  ASSERT_TRUE(f.module);
  ASSERT_NE(f.target, nullptr);
  auto p = selectPlan(*f.context, *f.module, *f.target);
  ASSERT_TRUE(bool(p)) << llvm::toString(p.takeError());
  auto b = bindCanonical(*f.module, *p, *f.target);
  ASSERT_TRUE(bool(b)) << llvm::toString(b.takeError());
  ASSERT_FALSE(bool(verifyMappedMicroIR(*b->module, *f.target)));

  unsigned mutations = 0;
  b->module->walk([&](Operation *op) {
    if (op->getName().getStringRef() != "micro.vector")
      return;
    auto mapping = op->getAttrOfType<DictionaryAttr>("micro.mapping");
    if (!mapping || !mapping.get("layout_parameters"))
      return;
    llvm::SmallVector<NamedAttribute> kept;
    for (NamedAttribute attribute : mapping)
      if (attribute.getName() != "layout_parameters")
        kept.push_back(attribute);
    op->setAttr("micro.mapping", DictionaryAttr::get(f.context.get(), kept));
    ++mutations;
  });
  ASSERT_GT(mutations, 0u);

  auto error = verifyMappedMicroIR(*b->module, *f.target);
  ASSERT_TRUE(bool(error));
  EXPECT_NE(llvm::toString(std::move(error)).find("no_legal_layout"),
            std::string::npos);
}

namespace {

/// Rewrites the first `layout_entries` entry of the vector op's mapping,
/// applying `mutate` to its fields. Returns false when there is no such entry.
template <typename Fn>
bool mutateLayoutEntry(ModuleOp module, MLIRContext &context, Fn mutate) {
  bool mutated = false;
  module->walk([&](Operation *op) {
    if (op->getName().getStringRef() != "micro.vector")
      return;
    auto mapping = op->getAttrOfType<DictionaryAttr>("micro.mapping");
    if (!mapping)
      return;
    auto entries = mapping.getAs<ArrayAttr>("layout_entries");
    if (!entries || entries.empty())
      return;
    llvm::SmallVector<Attribute> rewritten;
    for (size_t i = 0; i < entries.size(); ++i) {
      auto entry = dyn_cast<DictionaryAttr>(entries[i]);
      if (i != 0) {
        rewritten.push_back(entry);
        continue;
      }
      NamedAttrList fields(entry);
      mutate(fields, context);
      rewritten.push_back(fields.getDictionary(&context));
    }
    NamedAttrList updated(mapping);
    updated.set("layout_entries", ArrayAttr::get(&context, rewritten));
    op->setAttr("micro.mapping", updated.getDictionary(&context));
    mutated = true;
  });
  return mutated;
}

} // namespace

// A v2 layout entry records the endpoint its solved assignment is attributed
// to; an endpoint index outside the source graph is malformed metadata.
TEST(PlanBinder, RejectsAMalformedLayoutEntryEndpoint) {
  Fixture f = makeFixture();
  ASSERT_TRUE(f.module);
  ASSERT_NE(f.target, nullptr);
  auto p = selectPlan(*f.context, *f.module, *f.target);
  ASSERT_TRUE(bool(p)) << llvm::toString(p.takeError());
  auto b = bindCanonical(*f.module, *p, *f.target);
  ASSERT_TRUE(bool(b)) << llvm::toString(b.takeError());
  ASSERT_FALSE(bool(verifyMappedMicroIR(*b->module, *f.target)));

  bool mutated = mutateLayoutEntry(
      *b->module, *f.context.get(), [](NamedAttrList &fields, MLIRContext &c) {
        auto port = dyn_cast<DictionaryAttr>(fields.get("port"));
        NamedAttrList updated(port);
        updated.set("index", IntegerAttr::get(IntegerType::get(&c, 64), 99));
        fields.set("port", updated.getDictionary(&c));
      });
  ASSERT_TRUE(mutated);
  auto error = verifyMappedMicroIR(*b->module, *f.target);
  ASSERT_TRUE(bool(error));
  EXPECT_NE(llvm::toString(std::move(error)).find("source graph"),
            std::string::npos);
}

// A v2 layout entry records the concrete map its parameters substitute into; a
// map that is not the one the recorded parameters rebuild is rejected.
TEST(PlanBinder, RejectsATamperedLayoutEntryMap) {
  Fixture f = makeFixture();
  ASSERT_TRUE(f.module);
  ASSERT_NE(f.target, nullptr);
  auto p = selectPlan(*f.context, *f.module, *f.target);
  ASSERT_TRUE(bool(p)) << llvm::toString(p.takeError());
  auto b = bindCanonical(*f.module, *p, *f.target);
  ASSERT_TRUE(bool(b)) << llvm::toString(b.takeError());
  ASSERT_FALSE(bool(verifyMappedMicroIR(*b->module, *f.target)));

  bool mutated = mutateLayoutEntry(
      *b->module, *f.context.get(), [](NamedAttrList &fields, MLIRContext &c) {
        fields.set("map", AffineMapAttr::get(
                              AffineMap::getMultiDimIdentityMap(2, &c)));
      });
  ASSERT_TRUE(mutated);
  auto error = verifyMappedMicroIR(*b->module, *f.target);
  ASSERT_TRUE(bool(error));
  EXPECT_NE(llvm::toString(std::move(error)).find("no_legal_layout"),
            std::string::npos);
}

//===----------------------------------------------------------------------===//
// Stage-A A5: recorded consumers must actually read the movement's value
//===----------------------------------------------------------------------===//

// Regression (issue #67, stage A, A5): consumer-side movement verification only
// checked that the consumer list was integer-shaped. A recorded consumer
// endpoint that does not read the connection's value is now rejected.
TEST(PlanBinder, RejectsAConsumerEndpointThatDoesNotReadTheValue) {
  Fixture f = makeFixture();
  ASSERT_TRUE(f.module);
  llvm::Expected<std::unique_ptr<MappingTarget>> target = movementTarget();
  ASSERT_TRUE(bool(target)) << llvm::toString(target.takeError());
  auto p = selectPlan(*f.context, *f.module, **target);
  ASSERT_TRUE(bool(p)) << llvm::toString(p.takeError());
  auto b = bindCanonical(*f.module, *p, **target);
  ASSERT_TRUE(bool(b)) << llvm::toString(b.takeError());
  ASSERT_NE(materializedCopy(*b->module), nullptr);
  ASSERT_FALSE(bool(verifyMappedMicroIR(*b->module, **target)));

  auto routes = b->kernel->getAttrOfType<ArrayAttr>("micro.routes");
  ASSERT_TRUE(routes);
  llvm::SmallVector<Attribute> rewritten;
  for (Attribute element : routes) {
    auto route = dyn_cast<DictionaryAttr>(element);
    ASSERT_TRUE(route);
    NamedAttrList attributes(route);
    auto ports = route.getAs<ArrayAttr>("consumer_ports");
    ASSERT_TRUE(ports);
    ASSERT_FALSE(ports.empty());
    auto producer = route.getAs<DictionaryAttr>("producer_port");
    ASSERT_TRUE(producer);
    auto producerNode = producer.getAs<IntegerAttr>("node");
    ASSERT_TRUE(producerNode);
    llvm::SmallVector<Attribute> portAttrs;
    for (size_t i = 0; i < ports.size(); ++i) {
      if (i != 0) {
        portAttrs.push_back(ports[i]);
        continue;
      }
      NamedAttrList fields(dyn_cast<DictionaryAttr>(ports[i]));
      fields.set("node", producerNode);
      fields.set("direction", StringAttr::get(f.context.get(), "input"));
      fields.set("index",
                 IntegerAttr::get(IntegerType::get(f.context.get(), 64), 0));
      portAttrs.push_back(fields.getDictionary(f.context.get()));
    }
    attributes.set("consumer_ports",
                   ArrayAttr::get(f.context.get(), portAttrs));
    rewritten.push_back(attributes.getDictionary(f.context.get()));
  }
  b->kernel->setAttr("micro.routes",
                     ArrayAttr::get(f.context.get(), rewritten));

  auto error = verifyMappedMicroIR(*b->module, **target);
  ASSERT_TRUE(bool(error));
  EXPECT_NE(llvm::toString(std::move(error)).find("consumer_ports"),
            std::string::npos);
}

//===----------------------------------------------------------------------===//
// Legacy (v1) metadata: uniquely recoverable vs ambiguous
//===----------------------------------------------------------------------===//

namespace {

/// A v1-mapped kernel: the plan records no schema version and the mapping omits
/// endpoint/instance associations. With one node and no connections its missing
/// associations are uniquely recoverable.
constexpr llvm::StringLiteral kV1Kernel = R"mlir(
module {
  micro.kernel @k attributes {micro.plan = {id = 7 : i64, binding_hash = 0 : i64, truncated = false}} {
    %0 = tensor.empty() : tensor<8x8xf32>
    %t = micro.tile_view %0 {shape = array<i64: 8, 8>} : tensor<8x8xf32> -> !micro.tile<8x8xf32, memory = #micro.memory<sram>>
    %r = micro.vector "add" %t, %t {micro.mapping = {rule = "avx2.vector_add", executor = "worker.0", layouts = {avx2.blocked_2d = "avx2.blocked_2d"}, layout_parameters = {avx2.blocked_2d = {M = 4 : i64, N = 8 : i64, VW = 8 : i64}}, bundle = "avx2.vector.add.f32", emitter = "avx2_vector_add"}} : !micro.tile<8x8xf32, memory = #micro.memory<sram>>, !micro.tile<8x8xf32, memory = #micro.memory<sram>> -> !micro.tile<8x8xf32, memory = #micro.memory<sram>>
  }
}
)mlir";

/// The same v1 kernel with a selected connection that records no endpoints: its
/// missing associations are not uniquely recoverable.
constexpr llvm::StringLiteral kV1AmbiguousKernel = R"mlir(
module {
  micro.kernel @k attributes {micro.plan = {id = 7 : i64, binding_hash = 0 : i64, truncated = false}, micro.routes = [{id = 1 : i64, value = 0 : i64, kind = "transfer", route = ["dram.0", "sram.0"], engines = ["dma.0"], consumers = [1 : i64]}]} {
    %0 = tensor.empty() : tensor<8x8xf32>
    %t = micro.tile_view %0 {shape = array<i64: 8, 8>} : tensor<8x8xf32> -> !micro.tile<8x8xf32, memory = #micro.memory<sram>>
    %r = micro.vector "add" %t, %t {micro.mapping = {rule = "avx2.vector_add", executor = "worker.0", layouts = {avx2.blocked_2d = "avx2.blocked_2d"}, layout_parameters = {avx2.blocked_2d = {M = 4 : i64, N = 8 : i64, VW = 8 : i64}}, bundle = "avx2.vector.add.f32", emitter = "avx2_vector_add"}} : !micro.tile<8x8xf32, memory = #micro.memory<sram>>, !micro.tile<8x8xf32, memory = #micro.memory<sram>> -> !micro.tile<8x8xf32, memory = #micro.memory<sram>>
  }
}
)mlir";

} // namespace

TEST(PlanBinder, DecodesUnambiguousV1Metadata) {
  Fixture f = makeFixture();
  ASSERT_TRUE(f.target);
  MLIRContext context;
  context.getOrLoadDialect<micro::MicroDialect>();
  context.getOrLoadDialect<tensor::TensorDialect>();
  OwningOpRef<ModuleOp> module =
      parseSourceString<ModuleOp>(kV1Kernel, &context);
  ASSERT_TRUE(module);

  auto decoded = decodeSelectedPlan(*module, *f.target);
  ASSERT_TRUE(bool(decoded)) << llvm::toString(decoded.takeError());
  EXPECT_EQ(decoded->schemaVersion, 1u);
  EXPECT_EQ(decoded->id, 7u);
  ASSERT_EQ(decoded->placements.size(), 1u);
  EXPECT_EQ(decoded->placements[0].rule, "avx2.vector_add");
  ASSERT_EQ(decoded->placements[0].layoutSolutions.size(), 1u);
  const SolvedLayout &solved =
      decoded->placements[0].layoutSolutions.lookup("avx2.blocked_2d");
  EXPECT_EQ(solved.parameters.size(), 3u);
  ASSERT_TRUE(solved.port.has_value());
  EXPECT_EQ(solved.port->direction, PortDirection::Input);
  EXPECT_EQ(solved.port->index, 0u);
}

TEST(PlanBinder, RejectsAmbiguousV1ExecutableReplay) {
  Fixture f = makeFixture();
  ASSERT_TRUE(f.target);
  MLIRContext context;
  context.getOrLoadDialect<micro::MicroDialect>();
  context.getOrLoadDialect<tensor::TensorDialect>();
  OwningOpRef<ModuleOp> module =
      parseSourceString<ModuleOp>(kV1AmbiguousKernel, &context);
  ASSERT_TRUE(module);

  auto decoded = decodeSelectedPlan(*module, *f.target);
  ASSERT_FALSE(bool(decoded));
  EXPECT_NE(llvm::toString(decoded.takeError()).find("ambiguous"),
            std::string::npos);
}

//===----------------------------------------------------------------------===//
// Fix round 1 — Critical 1: resolved rule parameters
//===----------------------------------------------------------------------===//

// The rule's resolved parameter assignment is part of the selected state: it
// must be persisted, and verification must validate the *recorded* assignment
// (not substitute a legal one). `avx2.vector_add` derives `VW` from
// `VW == machine.compute("vector_engine").lanes(element_type)`, so a recorded
// VW outside its domain is a tampered selection.
TEST(PlanBinder, PersistsAndValidatesResolvedRuleParameters) {
  Fixture f = makeFixture();
  ASSERT_TRUE(f.module);
  ASSERT_NE(f.target, nullptr);
  auto p = selectPlan(*f.context, *f.module, *f.target);
  ASSERT_TRUE(bool(p)) << llvm::toString(p.takeError());
  auto b = bindCanonical(*f.module, *p, *f.target);
  ASSERT_TRUE(bool(b)) << llvm::toString(b.takeError());
  ASSERT_FALSE(bool(verifyMappedMicroIR(*b->module, *f.target)));

  bool sawParameters = false;
  unsigned tampered = 0;
  mlir::MLIRContext *context = f.context.get();
  b->module->walk([&](Operation *op) {
    if (op->getName().getStringRef() != "micro.vector")
      return;
    auto mapping = op->getAttrOfType<DictionaryAttr>("micro.mapping");
    if (!mapping)
      return;
    auto parameters = mapping.getAs<DictionaryAttr>("rule_parameters");
    if (!parameters)
      return;
    sawParameters = true;
    NamedAttrList mutated(parameters);
    mutated.set("VW", IntegerAttr::get(IntegerType::get(context, 64), 99));
    NamedAttrList updated(mapping);
    updated.set("rule_parameters", mutated.getDictionary(context));
    op->setAttr("micro.mapping", updated.getDictionary(context));
    ++tampered;
  });
  EXPECT_TRUE(sawParameters)
      << "the selected rule's resolved parameters must be persisted";
  ASSERT_GT(tampered, 0u);
  auto error = verifyMappedMicroIR(*b->module, *f.target);
  ASSERT_TRUE(bool(error));
  EXPECT_NE(llvm::toString(std::move(error)).find("no_matching_rule"),
            std::string::npos);
}

//===----------------------------------------------------------------------===//
// Fix round 1 — Important 4: storage and synchronization state
//===----------------------------------------------------------------------===//

// B1 declares the storage/synchronization model; the selected state must carry
// it through encode/decode even when B3 has not yet populated it.
TEST(PlanBinder, RoundTripsStorageAndSynchronizationState) {
  Fixture f = makeFixture();
  ASSERT_TRUE(f.module);
  ASSERT_NE(f.target, nullptr);
  auto p = selectPlan(*f.context, *f.module, *f.target);
  ASSERT_TRUE(bool(p)) << llvm::toString(p.takeError());

  CoveringPlan plan = *p;
  StorageAllocation allocation;
  allocation.id = 5;
  allocation.value = 3;
  allocation.memory = "sram.0";
  allocation.bytes = 4096;
  allocation.beginStep = 1;
  allocation.endStep = 7;
  plan.allocations.push_back(allocation);
  StorageAllocation aliased;
  aliased.id = 6;
  aliased.value = 4;
  aliased.memory = "dram.0";
  aliased.bytes = 128;
  aliased.aliasOf = 5;
  plan.allocations.push_back(aliased);
  SynchronizationStep step;
  step.id = 9;
  step.waitsFor = {1};
  step.requiresBarrier = true;
  plan.synchronization.push_back(step);

  auto b = bindCanonical(*f.module, plan, *f.target);
  ASSERT_TRUE(bool(b)) << llvm::toString(b.takeError());
  auto decoded = decodeSelectedPlan(*b->module, *f.target);
  ASSERT_TRUE(bool(decoded)) << llvm::toString(decoded.takeError());

  ASSERT_EQ(decoded->allocations.size(), 2u);
  EXPECT_EQ(decoded->allocations[0].id, 5u);
  EXPECT_EQ(decoded->allocations[0].value, 3u);
  EXPECT_EQ(decoded->allocations[0].memory, "sram.0");
  EXPECT_EQ(decoded->allocations[0].bytes, 4096u);
  EXPECT_FALSE(decoded->allocations[0].aliasOf.has_value());
  EXPECT_EQ(decoded->allocations[0].beginStep, 1u);
  EXPECT_EQ(decoded->allocations[0].endStep, 7u);
  ASSERT_TRUE(decoded->allocations[1].aliasOf.has_value());
  EXPECT_EQ(*decoded->allocations[1].aliasOf, 5u);
  ASSERT_EQ(decoded->synchronization.size(), 1u);
  EXPECT_EQ(decoded->synchronization[0].id, 9u);
  EXPECT_EQ(decoded->synchronization[0].waitsFor,
            (std::vector<ConnectionId>{1}));
  EXPECT_TRUE(decoded->synchronization[0].requiresBarrier);
}

//===----------------------------------------------------------------------===//
// Fix round 1 — Important 5: never silently downgrade v2 to v1
//===----------------------------------------------------------------------===//

namespace {

/// A kernel whose only workload node is a `convert`, whose rule requires a
/// layout (`avx2.blocked_2d`) but declares no parameter constraint -- so
/// deleting the layout container would bypass solved-layout validation.
constexpr llvm::StringLiteral kConvertKernel = R"mlir(
module {
  micro.kernel @convert {
    %0 = tensor.empty() : tensor<8x8xf32>
    %t = micro.tile_view %0 {shape = array<i64: 8, 8>} : tensor<8x8xf32> -> !micro.tile<8x8xf32, memory = #micro.memory<sram>>
    %r = micro.vector "convert" %t : !micro.tile<8x8xf32, memory = #micro.memory<sram>> -> !micro.tile<8x8xf32, memory = #micro.memory<sram>>
  }
}
)mlir";

/// Reparses `text`, selects a plan against AVX2, and binds it.
llvm::Expected<BoundPlan> bindConvertKernel(MLIRContext &context,
                                            MappingTarget &target) {
  OwningOpRef<ModuleOp> module =
      parseSourceString<ModuleOp>(kConvertKernel, &context);
  if (!module)
    return llvm::createStringError("failed to parse the convert kernel");
  llvm::Expected<CoveringPlan> plan = selectPlan(context, *module, target);
  if (!plan)
    return plan.takeError();
  return bindCanonical(*module, *plan, target);
}

/// Removes `name` from the `micro.plan` dictionary on `kernel`.
void eraseFromPlan(Operation *kernel, MLIRContext &context,
                   llvm::StringRef name) {
  auto plan = kernel->getAttrOfType<DictionaryAttr>("micro.plan");
  ASSERT_TRUE(plan);
  llvm::SmallVector<NamedAttribute> kept;
  for (NamedAttribute attribute : plan)
    if (attribute.getName() != name)
      kept.push_back(attribute);
  kernel->setAttr("micro.plan", DictionaryAttr::get(&context, kept));
}

/// Removes `name` from the first `micro.vector`'s mapping.
void eraseFromVectorMapping(ModuleOp module, MLIRContext &context,
                            llvm::StringRef name) {
  module->walk([&](Operation *op) {
    if (op->getName().getStringRef() != "micro.vector")
      return;
    auto mapping = op->getAttrOfType<DictionaryAttr>("micro.mapping");
    if (!mapping)
      return;
    llvm::SmallVector<NamedAttribute> kept;
    for (NamedAttribute attribute : mapping)
      if (attribute.getName() != name)
        kept.push_back(attribute);
    op->setAttr("micro.mapping", DictionaryAttr::get(&context, kept));
  });
}

} // namespace

// Deleting `schema_version` must not downgrade a v2 binding to v1 and re-open
// the A4 container bypass.
TEST(PlanBinder, DeletingSchemaVersionDoesNotReopenTheLayoutBypass) {
  Fixture f = makeFixture();
  ASSERT_TRUE(f.target);
  auto b = bindConvertKernel(*f.context, *f.target);
  ASSERT_TRUE(bool(b)) << llvm::toString(b.takeError());
  ASSERT_FALSE(bool(verifyMappedMicroIR(*b->module, *f.target)));

  eraseFromPlan(b->kernel, *f.context.get(), "schema_version");
  eraseFromVectorMapping(*b->module, *f.context.get(), "layout_parameters");

  auto error = verifyMappedMicroIR(*b->module, *f.target);
  ASSERT_TRUE(bool(error))
      << "deleting schema_version must not disable the v2 layout requirement";
  EXPECT_NE(llvm::toString(std::move(error)).find("no_legal_layout"),
            std::string::npos);
}

// The v2 content-hash checks must also survive `schema_version` deletion: a
// frozen plan still cannot bypass current graph verification.
TEST(PlanBinder, DeletingSchemaVersionDoesNotDisableGraphHashChecks) {
  Fixture f = makeFixture();
  ASSERT_TRUE(f.module);
  ASSERT_NE(f.target, nullptr);
  auto p = selectPlan(*f.context, *f.module, *f.target);
  ASSERT_TRUE(bool(p)) << llvm::toString(p.takeError());
  auto b = bindCanonical(*f.module, *p, *f.target);
  ASSERT_TRUE(bool(b)) << llvm::toString(b.takeError());

  eraseFromPlan(b->kernel, *f.context.get(), "schema_version");
  b->module->walk([&](Operation *op) {
    if (op->getName().getStringRef() == "micro.vector")
      op->setAttr("op", StringAttr::get(f.context.get(), "mul"));
  });

  auto decoded = decodeSelectedPlan(*b->module, *f.target);
  ASSERT_FALSE(bool(decoded));
  EXPECT_NE(llvm::toString(decoded.takeError()).find("graph_hash"),
            std::string::npos);
}

//===----------------------------------------------------------------------===//
// Fix round 1 — minor 7: layout_entries must accompany layout_parameters
//===----------------------------------------------------------------------===//

TEST(PlanBinder, DecodeRejectsMissingLayoutEntries) {
  Fixture f = makeFixture();
  ASSERT_TRUE(f.module);
  ASSERT_NE(f.target, nullptr);
  auto p = selectPlan(*f.context, *f.module, *f.target);
  ASSERT_TRUE(bool(p)) << llvm::toString(p.takeError());
  auto b = bindCanonical(*f.module, *p, *f.target);
  ASSERT_TRUE(bool(b)) << llvm::toString(b.takeError());

  eraseFromVectorMapping(*b->module, *f.context.get(), "layout_entries");
  auto decoded = decodeSelectedPlan(*b->module, *f.target);
  ASSERT_FALSE(bool(decoded));
}

//===----------------------------------------------------------------------===//
// Fix round 2 — N-1: an empty recorded assignment must not downgrade a
// parametrized rule to generation's existential fallback
//===----------------------------------------------------------------------===//

namespace {

/// A machine with a single `worker` executor, sufficient for the custom rule.
constexpr llvm::StringLiteral kParamMachine = R"yaml(
schema: llk.machine.v2
target: param
clock_hz: 1000000000
worker_threads: 1
executors:
  - id: worker.0
    kind: worker
memories:
  - id: sram.0
    kind: sram
    visible_from: worker.0
    capacity_bytes: 1048576
    alignment_bytes: 64
    supported_layouts: [row_major]
    bandwidth_bytes_per_cycle: 64
    latency_cycles: 4
)yaml";

/// A rule whose `require` constraint references its declared parameter `VW` but
/// does not depend on the layout context, so a forged *empty* recorded
/// assignment still satisfies the existential fallback (the hole N-1 closes).
constexpr llvm::StringLiteral kParamRules = R"llkmap(
rule p.vector {
  match micro.vector(op = "add");
  param VW in [4..8];
  require VW % 4 == 0;
  require executor kind worker;
  bundle "p.vector";
  emit "e1";
  cost 1;
}
)llkmap";

llvm::Expected<std::unique_ptr<MappingTarget>> parameterizedTarget() {
  llvm::Expected<mlir::llk::machine::MachineModel> machine =
      mlir::llk::machine::parseMachineModel(kParamMachine, "<test>");
  if (!machine)
    return machine.takeError();
  llvm::Expected<LayoutRegistry> layouts = parseLayoutText("", "<test>");
  if (!layouts)
    return layouts.takeError();
  llvm::Expected<RuleRegistry> rules = parseRuleText(kParamRules, "<test>");
  if (!rules)
    return rules.takeError();
  return std::make_unique<FileMappingTarget>(
      "param", std::move(*machine), std::move(*layouts), std::move(*rules),
      std::vector<std::string>{"e1"});
}

} // namespace

TEST(PlanBinder, RejectsAnEmptyRecordedAssignmentForAParametrizedRule) {
  MLIRContext context;
  context.getOrLoadDialect<micro::MicroDialect>();
  context.getOrLoadDialect<tensor::TensorDialect>();
  OwningOpRef<ModuleOp> module =
      parseSourceString<ModuleOp>(kVectorOnlyKernel, &context);
  ASSERT_TRUE(module);
  llvm::Expected<std::unique_ptr<MappingTarget>> target = parameterizedTarget();
  ASSERT_TRUE(bool(target)) << llvm::toString(target.takeError());

  auto p = selectPlan(context, *module, **target);
  ASSERT_TRUE(bool(p)) << llvm::toString(p.takeError());
  auto b = bindCanonical(*module, *p, **target);
  ASSERT_TRUE(bool(b)) << llvm::toString(b.takeError());
  ASSERT_FALSE(bool(verifyMappedMicroIR(*b->module, **target)));

  // The rule derives `VW`, so the binding records a non-empty assignment.
  bool recorded = false;
  b->module->walk([&](Operation *op) {
    if (op->getName().getStringRef() != "micro.vector")
      return;
    auto mapping = op->getAttrOfType<DictionaryAttr>("micro.mapping");
    if (!mapping)
      return;
    auto parameters = mapping.getAs<DictionaryAttr>("rule_parameters");
    if (parameters && !parameters.empty())
      recorded = true;
    // Forge an empty assignment, which is *present* but records nothing.
    NamedAttrList updated(mapping);
    updated.set("rule_parameters", DictionaryAttr::get(&context, {}));
    op->setAttr("micro.mapping", updated.getDictionary(&context));
  });
  ASSERT_TRUE(recorded);

  auto error = verifyMappedMicroIR(*b->module, **target);
  ASSERT_TRUE(bool(error))
      << "an empty recorded assignment must not downgrade a parametrized rule";
  EXPECT_NE(llvm::toString(std::move(error)).find("recorded parameter"),
            std::string::npos);
}

//===----------------------------------------------------------------------===//
// Fix round 1 (task B2) — named-port memory associations round-trip and replay
//===----------------------------------------------------------------------===//

/// Rewrites the `memory` of every `port_memories` entry on an op whose rule is
/// `ruleId`, bypassing re-verification, so what is re-checked is the recorded
/// association.
bool retargetPortMemories(ModuleOp module, llvm::StringRef ruleId,
                          llvm::StringRef memory) {
  bool changed = false;
  mlir::MLIRContext *context = module.getContext();
  module->walk([&](Operation *op) {
    auto mapping = op->getAttrOfType<DictionaryAttr>("micro.mapping");
    if (!mapping)
      return;
    auto rule = mapping.getAs<StringAttr>("rule");
    if (!rule || rule.getValue() != ruleId)
      return;
    auto array = mapping.getAs<ArrayAttr>("port_memories");
    if (!array)
      return;
    SmallVector<Attribute> entries;
    for (Attribute element : array) {
      auto entry = mlir::cast<DictionaryAttr>(element);
      NamedAttrList fields(entry);
      fields.set("memory", StringAttr::get(context, memory));
      entries.push_back(DictionaryAttr::get(context, fields));
    }
    NamedAttrList updated(mapping);
    updated.set("port_memories", ArrayAttr::get(context, entries));
    op->setAttr("micro.mapping", updated.getDictionary(context));
    changed = true;
  });
  return changed;
}

// A selected plan whose rules name their output ports carries per-occurrence
// memory bindings, and both the metadata round trip and the replay retain them.
TEST(PlanBinder, RoundTripsNamedPortMemoryBindings) {
  Fixture fixture = makeFixture();
  ASSERT_TRUE(fixture.module);
  llvm::Expected<std::unique_ptr<MappingTarget>> target = namedPortTarget();
  ASSERT_TRUE(static_cast<bool>(target)) << llvm::toString(target.takeError());

  llvm::Expected<CoveringPlan> plan =
      selectPlan(*fixture.context, *fixture.module, **target);
  ASSERT_TRUE(static_cast<bool>(plan)) << llvm::toString(plan.takeError());
  bool sawPortBindings = false;
  for (const PlanPlacement &placement : plan->placements)
    sawPortBindings |= !placement.portMemoryBindings.empty();
  ASSERT_TRUE(sawPortBindings)
      << "the named-port rules must produce per-occurrence bindings";

  llvm::Expected<BoundPlan> bound =
      bindCanonical(*fixture.module, *plan, **target);
  ASSERT_TRUE(static_cast<bool>(bound)) << llvm::toString(bound.takeError());
  llvm::Error verify = verifyMappedMicroIR(*bound->module, **target);
  EXPECT_FALSE(static_cast<bool>(verify)) << llvm::toString(std::move(verify));

  llvm::Expected<CoveringPlan> decoded =
      decodeSelectedPlan(*bound->module, **target);
  ASSERT_TRUE(static_cast<bool>(decoded))
      << llvm::toString(decoded.takeError());
  for (const PlanPlacement &placement : plan->placements) {
    const PlanPlacement *match = nullptr;
    for (const PlanPlacement &candidate : decoded->placements)
      if (candidate.node == placement.node)
        match = &candidate;
    ASSERT_NE(match, nullptr)
        << "node " << placement.node << " was not decoded";
    EXPECT_EQ(match->portMemoryBindings, placement.portMemoryBindings);
  }
}

// Tampering the recorded port→memory association (pointing the vector node's
// output at SRAM instead of DRAM) is a memory violation, reported with the
// memory-specific code rather than an executor one.
TEST(PlanBinder, ReplayRejectsATamperedPortMemoryAssociation) {
  Fixture fixture = makeFixture();
  ASSERT_TRUE(fixture.module);
  llvm::Expected<std::unique_ptr<MappingTarget>> target = namedPortTarget();
  ASSERT_TRUE(static_cast<bool>(target)) << llvm::toString(target.takeError());

  llvm::Expected<CoveringPlan> plan =
      selectPlan(*fixture.context, *fixture.module, **target);
  ASSERT_TRUE(static_cast<bool>(plan)) << llvm::toString(plan.takeError());
  llvm::Expected<BoundPlan> bound =
      bindCanonical(*fixture.module, *plan, **target);
  ASSERT_TRUE(static_cast<bool>(bound)) << llvm::toString(bound.takeError());
  ASSERT_FALSE(
      static_cast<bool>(verifyMappedMicroIR(*bound->module, **target)));

  ASSERT_TRUE(retargetPortMemories(*bound->module, "t.vector", "sram.0"));
  auto error = verifyMappedMicroIR(*bound->module, **target);
  ASSERT_TRUE(static_cast<bool>(error));
  std::string text = llvm::toString(std::move(error));
  EXPECT_NE(text.find("no_memory_route"), std::string::npos) << text;
  EXPECT_EQ(text.find("no_legal_executor"), std::string::npos) << text;
}

// Deleting the recorded association of a rule that requires one is rejected
// (fail-closed), with the memory code -- a missing memory association is not an
// executor problem.
TEST(PlanBinder, ReplayWithoutThePortAssociationIsRejected) {
  Fixture fixture = makeFixture();
  ASSERT_TRUE(fixture.module);
  llvm::Expected<std::unique_ptr<MappingTarget>> target = namedPortTarget();
  ASSERT_TRUE(static_cast<bool>(target)) << llvm::toString(target.takeError());

  llvm::Expected<CoveringPlan> plan =
      selectPlan(*fixture.context, *fixture.module, **target);
  ASSERT_TRUE(static_cast<bool>(plan)) << llvm::toString(plan.takeError());
  llvm::Expected<BoundPlan> bound =
      bindCanonical(*fixture.module, *plan, **target);
  ASSERT_TRUE(static_cast<bool>(bound)) << llvm::toString(bound.takeError());
  ASSERT_FALSE(
      static_cast<bool>(verifyMappedMicroIR(*bound->module, **target)));

  // Strip `port_memories` from the mapped ops whose rules require one.
  mlir::MLIRContext *context = fixture.context.get();
  unsigned stripped = 0;
  bound->module->walk([&](Operation *op) {
    auto mapping = op->getAttrOfType<DictionaryAttr>("micro.mapping");
    if (!mapping || !mapping.get("port_memories"))
      return;
    NamedAttrList kept;
    for (NamedAttribute attribute : mapping)
      if (attribute.getName() != "port_memories")
        kept.push_back(attribute);
    op->setAttr("micro.mapping", DictionaryAttr::get(context, kept));
    ++stripped;
  });
  ASSERT_GT(stripped, 0u);

  auto error = verifyMappedMicroIR(*bound->module, **target);
  ASSERT_TRUE(static_cast<bool>(error));
  std::string text = llvm::toString(std::move(error));
  EXPECT_NE(text.find("no_memory_route"), std::string::npos) << text;
}

//===----------------------------------------------------------------------===//
// Canonical materialization of tile movements (task B4)
//===----------------------------------------------------------------------===//

namespace {

/// A machine with SRAM, L2 (both in cluster.a) and DRAM (cluster.b), linked
/// SRAM->L2 and SRAM->DRAM, so one SRAM value can be carried to two different
/// destinations.
constexpr llvm::StringLiteral kTileMachine = R"yaml(
schema: llk.machine.v2
target: tile
clock_hz: 1000000000
worker_threads: 2
executors:
  - id: cluster.a
    kind: cluster
  - id: worker.a
    kind: worker
    parent: cluster.a
  - id: cluster.b
    kind: cluster
  - id: worker.b
    kind: worker
    parent: cluster.b
memories:
  - id: sram.0
    kind: sram
    visible_from: cluster.a
    capacity_bytes: 1048576
    alignment_bytes: 64
    supported_layouts: [row_major]
    bandwidth_bytes_per_cycle: 64
    latency_cycles: 4
  - id: l2.0
    kind: l2
    visible_from: cluster.b
    capacity_bytes: 1048576
    alignment_bytes: 64
    supported_layouts: [row_major]
    bandwidth_bytes_per_cycle: 64
    latency_cycles: 12
  - id: dram.0
    kind: dram
    visible_from: cluster.b
    capacity_bytes: 1073741824
    alignment_bytes: 64
    supported_layouts: [row_major]
    bandwidth_bytes_per_cycle: 32
    latency_cycles: 220
compute:
  - id: vpu
    kind: vector_engine
    attached_to: worker.a
    element_types: [f32]
    shapes: [[8]]
    lanes: {f32: 8}
transfer_engines:
  - id: dma.a
    kind: dma
    attached_to: cluster.a
    count: 1
    max_outstanding: 1
links:
  - id: sram_to_l2.0
    source: sram.0
    destination: l2.0
    bandwidth_bytes_per_cycle: 64
    latency_cycles: 12
    transaction_bytes: 64
    transfer_engines: [dma.a]
  - id: sram_to_dram.0
    source: sram.0
    destination: dram.0
    bandwidth_bytes_per_cycle: 64
    latency_cycles: 220
    transaction_bytes: 64
    transfer_engines: [dma.a]
)yaml";

/// One rule per vector variant, each binding a distinct memory: add in SRAM,
/// mul in DRAM, sub in L2.
constexpr llvm::StringLiteral kTileRules = R"llkmap(
rule t.add {
  match micro.vector(op = "add");
  require executor kind worker;
  require memory kind sram;
  bundle "b.add";
  emit "e1";
  cost 1;
}
rule t.mul {
  match micro.vector(op = "mul");
  require executor kind worker;
  require memory kind dram;
  bundle "b.mul";
  emit "e1";
  cost 1;
}
rule t.sub {
  match micro.vector(op = "sub");
  require executor kind worker;
  require memory kind l2;
  bundle "b.sub";
  emit "e1";
  cost 1;
}
)llkmap";

llvm::Expected<std::unique_ptr<MappingTarget>> tileTarget() {
  llvm::Expected<mlir::llk::machine::MachineModel> machine =
      mlir::llk::machine::parseMachineModel(kTileMachine, "<test>");
  if (!machine)
    return machine.takeError();
  llvm::Expected<LayoutRegistry> layouts = parseLayoutText("", "<test>");
  if (!layouts)
    return layouts.takeError();
  llvm::Expected<RuleRegistry> rules = parseRuleText(kTileRules, "<test>");
  if (!rules)
    return rules.takeError();
  return std::make_unique<FileMappingTarget>(
      "tile", std::move(*machine), std::move(*layouts), std::move(*rules),
      std::vector<std::string>{"e1"});
}

/// A `micro.vector`-only tile kernel: an external SRAM tile feeds an add whose
/// tile result is consumed by `micro.vector`s the rules place in other
/// memories, so the edge is a real tile movement.
constexpr llvm::StringLiteral kTileKernel = R"mlir(
module {
  micro.kernel @tiled {
    %a = micro.tile_alloc : !micro.tile<8x8xf32, memory = #micro.memory<sram>>
    %r = micro.vector "add" %a, %a : !micro.tile<8x8xf32, memory = #micro.memory<sram>>, !micro.tile<8x8xf32, memory = #micro.memory<sram>> -> !micro.tile<8x8xf32, memory = #micro.memory<sram>>
    %m = micro.vector "mul" %r, %r : !micro.tile<8x8xf32, memory = #micro.memory<sram>>, !micro.tile<8x8xf32, memory = #micro.memory<sram>> -> !micro.tile<8x8xf32, memory = #micro.memory<sram>>
    %s = micro.vector "sub" %r, %r : !micro.tile<8x8xf32, memory = #micro.memory<sram>>, !micro.tile<8x8xf32, memory = #micro.memory<sram>> -> !micro.tile<8x8xf32, memory = #micro.memory<sram>>
    micro.yield
  }
}
)mlir";

/// The same kernel with one consumer, for the transform cases.
constexpr llvm::StringLiteral kTileKernelOneConsumer = R"mlir(
module {
  micro.kernel @tiled {
    %a = micro.tile_alloc : !micro.tile<8x8xf32, memory = #micro.memory<sram>>
    %r = micro.vector "add" %a, %a : !micro.tile<8x8xf32, memory = #micro.memory<sram>>, !micro.tile<8x8xf32, memory = #micro.memory<sram>> -> !micro.tile<8x8xf32, memory = #micro.memory<sram>>
    %m = micro.vector "mul" %r, %r : !micro.tile<8x8xf32, memory = #micro.memory<sram>>, !micro.tile<8x8xf32, memory = #micro.memory<sram>> -> !micro.tile<8x8xf32, memory = #micro.memory<sram>>
    micro.yield
  }
}
)mlir";

Fixture makeTileFixture(llvm::StringRef kernel = kTileKernel) {
  Fixture fixture;
  fixture.context = std::make_unique<MLIRContext>();
  fixture.context->getOrLoadDialect<micro::MicroDialect>();
  fixture.context->getOrLoadDialect<tensor::TensorDialect>();
  fixture.context->getOrLoadDialect<arith::ArithDialect>();
  fixture.module = parseSourceString<ModuleOp>(kernel, fixture.context.get());
  return fixture;
}

/// The printed form of `type`, for asserting a tile's memory.
std::string typeText(Type type) {
  std::string text;
  llvm::raw_string_ostream stream(text);
  type.print(stream);
  return stream.str();
}

/// The `micro.vector` operation with attribute `op = opName`, or null. Node ids
/// are content-canonical, so tests identify the consumer by what it is.
Operation *vectorOp(ModuleOp module, llvm::StringRef opName) {
  Operation *found = nullptr;
  module->walk([&](Operation *op) {
    if (found)
      return;
    if (op->getName().getStringRef() != "micro.vector")
      return;
    auto name = op->getAttrOfType<StringAttr>("op");
    if (name && name.getValue() == opName)
      found = op;
  });
  return found;
}

/// The materialized tile copy in `module` (the one the materializer stamped).
Operation *tileCopy(ModuleOp module) {
  Operation *copy = nullptr;
  module->walk([&](Operation *op) {
    if (op->getName().getStringRef() == "micro.tile_async_copy" &&
        op->hasAttr("micro.dst_node"))
      copy = op;
  });
  return copy;
}

} // namespace

// A tile-valued connection is materialized as a typed `micro.tile_async_copy`
// whose result is the tile retyped into the destination memory, followed by its
// `micro.wait`; the recorded consumer reads that result.
TEST(PlanBinder, MaterializesATileMovementAsATypedTileAsyncCopy) {
  Fixture f = makeTileFixture(kTileKernelOneConsumer);
  ASSERT_TRUE(f.module);
  llvm::Expected<std::unique_ptr<MappingTarget>> target = tileTarget();
  ASSERT_TRUE(bool(target)) << llvm::toString(target.takeError());
  auto p = selectPlan(*f.context, *f.module, **target);
  ASSERT_TRUE(bool(p)) << llvm::toString(p.takeError());

  auto b = bindCanonical(*f.module, *p, **target);
  ASSERT_TRUE(bool(b)) << llvm::toString(b.takeError());
  EXPECT_TRUE(b->unmaterialized.empty());

  Operation *copy = tileCopy(*b->module);
  ASSERT_NE(copy, nullptr);
  EXPECT_NE(typeText(copy->getResult(0).getType())
                .find("memory = #micro.memory<dram>"),
            std::string::npos);
  EXPECT_EQ(countOps(*b->module, "micro.wait"), 1u);
  // The mul consumer reads the destination-memory tile.
  Operation *mul = vectorOp(*b->module, "mul");
  ASSERT_NE(mul, nullptr);
  EXPECT_EQ(mul->getOperand(0), copy->getResult(0));
  {
    llvm::Error verification = verifyMappedMicroIR(*b->module, **target);
    EXPECT_FALSE(bool(verification)) << llvm::toString(std::move(verification));
  }
}

// A5 carry-over: graph-level value equality is not enough. Pointing a consumer
// back at the original value (same workload value, so the graph check still
// passes) while `micro.routes` still records it must be rejected -- the
// consumer no longer reads the materialized movement.
TEST(PlanBinder, RejectsAConsumerThatDoesNotReadTheEmittedMovement) {
  Fixture f = makeTileFixture(kTileKernelOneConsumer);
  ASSERT_TRUE(f.module);
  llvm::Expected<std::unique_ptr<MappingTarget>> target = tileTarget();
  ASSERT_TRUE(bool(target)) << llvm::toString(target.takeError());
  auto p = selectPlan(*f.context, *f.module, **target);
  ASSERT_TRUE(bool(p)) << llvm::toString(p.takeError());
  auto b = bindCanonical(*f.module, *p, **target);
  ASSERT_TRUE(bool(b)) << llvm::toString(b.takeError());

  // Legal control: the materialized consumer reads the movement.
  EXPECT_FALSE(bool(verifyMappedMicroIR(*b->module, **target)));

  Operation *copy = tileCopy(*b->module);
  ASSERT_NE(copy, nullptr);
  Operation *mul = vectorOp(*b->module, "mul");
  ASSERT_NE(mul, nullptr);
  Value original = copy->getOperand(0);
  for (OpOperand &use : mul->getOpOperands())
    use.set(original);

  auto error = verifyMappedMicroIR(*b->module, **target);
  ASSERT_TRUE(bool(error));
  std::string text = llvm::toString(std::move(error));
  EXPECT_NE(text.find("does not read the materialized movement"),
            std::string::npos)
      << text;
}

// One value with two consumers in different destinations gets one chain per
// destination, and each consumer reads its own chain -- the endpoint-scoped
// rewiring the instance projection cannot express.
TEST(PlanBinder, MaterializesOneChainPerDestinationForAReusedValue) {
  Fixture f = makeTileFixture(kTileKernel);
  ASSERT_TRUE(f.module);
  llvm::Expected<std::unique_ptr<MappingTarget>> target = tileTarget();
  ASSERT_TRUE(bool(target)) << llvm::toString(target.takeError());
  auto p = selectPlan(*f.context, *f.module, **target);
  ASSERT_TRUE(bool(p)) << llvm::toString(p.takeError());

  auto b = bindCanonical(*f.module, *p, **target);
  ASSERT_TRUE(bool(b)) << llvm::toString(b.takeError());
  EXPECT_TRUE(b->unmaterialized.empty());
  EXPECT_EQ(countOps(*b->module, "micro.tile_async_copy"), 2u);
  EXPECT_EQ(countOps(*b->module, "micro.wait"), 2u);

  Operation *mul = vectorOp(*b->module, "mul");
  Operation *sub = vectorOp(*b->module, "sub");
  ASSERT_NE(mul, nullptr);
  ASSERT_NE(sub, nullptr);
  EXPECT_NE(typeText(mul->getOperand(0).getType())
                .find("memory = #micro.memory<dram>"),
            std::string::npos);
  EXPECT_NE(
      typeText(sub->getOperand(0).getType()).find("memory = #micro.memory<l2>"),
      std::string::npos);
  EXPECT_FALSE(bool(verifyMappedMicroIR(*b->module, **target)));
}

// A tile LayoutTransform moves nothing and becomes one `micro.transform` on the
// tile, re-representing it under the solved maps.
TEST(PlanBinder, MaterializesATileLayoutTransformAsATransformOp) {
  Fixture f = makeTileFixture(kTileKernelOneConsumer);
  ASSERT_TRUE(f.module);
  llvm::Expected<std::unique_ptr<MappingTarget>> target = tileTarget();
  ASSERT_TRUE(bool(target)) << llvm::toString(target.takeError());
  auto p = selectPlan(*f.context, *f.module, **target);
  ASSERT_TRUE(bool(p)) << llvm::toString(p.takeError());
  ASSERT_FALSE(p->connectionPlans.empty());

  MLIRContext *context = f.context.get();
  PlanConnection &connection = p->connectionPlans.front();
  connection.kind = ConnectionKind::LayoutTransform;
  connection.route.resize(1);
  LayoutTransform transform;
  transform.srcLayout = "t.plain";
  transform.dstLayout = "t.blocked";
  transform.srcMap = AffineMap::getMultiDimIdentityMap(2, context);
  connection.transform = transform;

  auto b = bindCanonical(*f.module, *p, **target);
  ASSERT_TRUE(bool(b)) << llvm::toString(b.takeError());
  EXPECT_TRUE(b->unmaterialized.empty());
  EXPECT_EQ(countOps(*b->module, "micro.transform"), 1u);
  EXPECT_EQ(countOps(*b->module, "micro.tile_async_copy"), 0u);
  EXPECT_FALSE(bool(verifyMappedMicroIR(*b->module, **target)));
}

// A tile TransferAndTransform emits the typed tile copy and then the transform;
// the consumer reads the transformed destination tile.
TEST(PlanBinder, MaterializesATileTransferAndTransformAsCopyPlusTransform) {
  Fixture f = makeTileFixture(kTileKernelOneConsumer);
  ASSERT_TRUE(f.module);
  llvm::Expected<std::unique_ptr<MappingTarget>> target = tileTarget();
  ASSERT_TRUE(bool(target)) << llvm::toString(target.takeError());
  auto p = selectPlan(*f.context, *f.module, **target);
  ASSERT_TRUE(bool(p)) << llvm::toString(p.takeError());
  ASSERT_FALSE(p->connectionPlans.empty());

  MLIRContext *context = f.context.get();
  PlanConnection &connection = p->connectionPlans.front();
  connection.kind = ConnectionKind::TransferAndTransform;
  LayoutTransform transform;
  transform.srcLayout = "t.plain";
  transform.dstLayout = "t.blocked";
  transform.srcMap = AffineMap::getMultiDimIdentityMap(2, context);
  connection.transform = transform;

  auto b = bindCanonical(*f.module, *p, **target);
  ASSERT_TRUE(bool(b)) << llvm::toString(b.takeError());
  EXPECT_TRUE(b->unmaterialized.empty());
  EXPECT_EQ(countOps(*b->module, "micro.tile_async_copy"), 1u);
  EXPECT_EQ(countOps(*b->module, "micro.transform"), 1u);
  EXPECT_EQ(countOps(*b->module, "micro.wait"), 1u);
  EXPECT_FALSE(bool(verifyMappedMicroIR(*b->module, **target)));
}

// Without a materializer the binder is metadata-only: it still persists the
// selection but reports every connection that would need a new operation with
// the named `no_plan_materializer` reason, so a standalone caller gets an
// honest partial plan rather than IR silently missing the movement.
TEST(PlanBinder, WithoutAMaterializerThePlanIsBoundMetadataOnly) {
  Fixture f = makeTileFixture(kTileKernelOneConsumer);
  ASSERT_TRUE(f.module);
  llvm::Expected<std::unique_ptr<MappingTarget>> target = tileTarget();
  ASSERT_TRUE(bool(target)) << llvm::toString(target.takeError());
  auto p = selectPlan(*f.context, *f.module, **target);
  ASSERT_TRUE(bool(p)) << llvm::toString(p.takeError());

  auto b = bindPlan(*f.module, *p, **target);
  ASSERT_TRUE(bool(b)) << llvm::toString(b.takeError());
  ASSERT_FALSE(b->unmaterialized.empty());
  EXPECT_NE(b->unmaterialized.front().find("no_plan_materializer"),
            std::string::npos)
      << b->unmaterialized.front();
  EXPECT_EQ(countOps(*b->module, "micro.tile_async_copy"), 0u);
}

// An Executable caller must supply the canonical materializer: without one the
// plan cannot be made executable, so the binding is refused, naming the reason.
TEST(PlanBinder, ExecutableWithoutAMaterializerIsRejected) {
  Fixture f = makeTileFixture(kTileKernelOneConsumer);
  ASSERT_TRUE(f.module);
  llvm::Expected<std::unique_ptr<MappingTarget>> target = tileTarget();
  ASSERT_TRUE(bool(target)) << llvm::toString(target.takeError());
  auto p = selectPlan(*f.context, *f.module, **target);
  ASSERT_TRUE(bool(p)) << llvm::toString(p.takeError());

  auto b = bindPlan(*f.module, *p, **target, BindContract::Executable);
  ASSERT_FALSE(bool(b));
  std::string text = llvm::toString(b.takeError());
  EXPECT_NE(text.find("no_plan_materializer"), std::string::npos) << text;
}

// The positive control: with the canonical materializer the same plan is fully
// executable.
TEST(PlanBinder, ExecutableWithAMaterializerSucceeds) {
  Fixture f = makeTileFixture(kTileKernelOneConsumer);
  ASSERT_TRUE(f.module);
  llvm::Expected<std::unique_ptr<MappingTarget>> target = tileTarget();
  ASSERT_TRUE(bool(target)) << llvm::toString(target.takeError());
  auto p = selectPlan(*f.context, *f.module, **target);
  ASSERT_TRUE(bool(p)) << llvm::toString(p.takeError());

  auto b = bindCanonical(*f.module, *p, **target, BindContract::Executable);
  ASSERT_TRUE(bool(b)) << llvm::toString(b.takeError());
  EXPECT_TRUE(b->unmaterialized.empty());
}

// A tile whose element type has no Micro dtype cannot be retyped into the
// destination memory: the missing static fact is a named unresolved decision,
// reported by the partial contract and refused by the executable one -- never a
// silently wrong copy.
TEST(PlanBinder, AMissingStaticFactIsANamedUnresolvedDecision) {
  constexpr llvm::StringLiteral kDynamicKernel = R"mlir(
module {
  micro.kernel @tiled {
    %c = arith.constant 8 : index
    %e = tensor.empty(%c) : tensor<?x8xf32>
    %a = micro.tile_view %e {shape = array<i64: -9223372036854775808, 8>} : tensor<?x8xf32> -> !micro.tile<?x8xf32, memory = #micro.memory<sram>>
    %r = micro.vector "add" %a, %a : !micro.tile<?x8xf32, memory = #micro.memory<sram>>, !micro.tile<?x8xf32, memory = #micro.memory<sram>> -> !micro.tile<?x8xf32, memory = #micro.memory<sram>>
    %m = micro.vector "mul" %r, %r : !micro.tile<?x8xf32, memory = #micro.memory<sram>>, !micro.tile<?x8xf32, memory = #micro.memory<sram>> -> !micro.tile<?x8xf32, memory = #micro.memory<sram>>
    micro.yield
  }
}
)mlir";
  Fixture f = makeTileFixture(kDynamicKernel);
  ASSERT_TRUE(f.module);
  llvm::Expected<std::unique_ptr<MappingTarget>> target = tileTarget();
  ASSERT_TRUE(bool(target)) << llvm::toString(target.takeError());
  auto p = selectPlan(*f.context, *f.module, **target);
  ASSERT_TRUE(bool(p)) << llvm::toString(p.takeError());

  auto partial = bindCanonical(*f.module, *p, **target);
  ASSERT_TRUE(bool(partial)) << llvm::toString(partial.takeError());
  ASSERT_FALSE(partial->unmaterialized.empty());
  EXPECT_NE(partial->unmaterialized.front().find("missing_static_fact"),
            std::string::npos)
      << partial->unmaterialized.front();

  auto executable =
      bindCanonical(*f.module, *p, **target, BindContract::Executable);
  ASSERT_FALSE(bool(executable));
  std::string text = llvm::toString(executable.takeError());
  EXPECT_NE(text.find("missing_static_fact"), std::string::npos) << text;
}

// Tile memory is source-semantic: `micro.tile_alloc`/`micro.tile_view` declare
// it, and design §10.2 uses it to decide whether an edge is a Transfer at all.
// Two kernels identical but for a declared tile memory must therefore have
// different source-graph identities, and a plan/report bound for one must be
// rejected against the other -- the same class the loop-multiplicity fix
// closed.
TEST(PlanBinder, AChangedTileMemoryChangesTheGraphHashAndRejectsReplay) {
  // Identical to `kTileKernelOneConsumer` except the external tile is declared
  // in L2 rather than SRAM.
  constexpr llvm::StringLiteral kL2SourceKernel = R"mlir(
module {
  micro.kernel @tiled {
    %a = micro.tile_alloc : !micro.tile<8x8xf32, memory = #micro.memory<l2>>
    %r = micro.vector "add" %a, %a : !micro.tile<8x8xf32, memory = #micro.memory<l2>>, !micro.tile<8x8xf32, memory = #micro.memory<l2>> -> !micro.tile<8x8xf32, memory = #micro.memory<sram>>
    %m = micro.vector "mul" %r, %r : !micro.tile<8x8xf32, memory = #micro.memory<sram>>, !micro.tile<8x8xf32, memory = #micro.memory<sram>> -> !micro.tile<8x8xf32, memory = #micro.memory<sram>>
    micro.yield
  }
}
)mlir";
  Fixture a = makeTileFixture(kTileKernelOneConsumer);
  Fixture b = makeTileFixture(kL2SourceKernel);
  ASSERT_TRUE(a.module);
  ASSERT_TRUE(b.module);
  llvm::Expected<WorkloadGraph> graphA =
      extractWorkloadGraph(findKernel(*a.module));
  llvm::Expected<WorkloadGraph> graphB =
      extractWorkloadGraph(findKernel(*b.module));
  ASSERT_TRUE(bool(graphA)) << llvm::toString(graphA.takeError());
  ASSERT_TRUE(bool(graphB)) << llvm::toString(graphB.takeError());
  EXPECT_NE(computeSourceGraphHash(*graphA), computeSourceGraphHash(*graphB));

  llvm::Expected<std::unique_ptr<MappingTarget>> target = tileTarget();
  ASSERT_TRUE(bool(target)) << llvm::toString(target.takeError());
  LayoutContext layoutContext;
  layoutContext.rank = 2;
  layoutContext.elementType = "f32";
  MappingSearchOptions options;
  options.mode = SearchMode::Deterministic;
  CoveringSearch search(*graphA, **target, *a.context, layoutContext, options);
  llvm::Expected<MappingSearchResult> result = search.search();
  ASSERT_TRUE(bool(result)) << llvm::toString(result.takeError());
  ASSERT_FALSE(result->plans.empty());
  std::string report = writePlanReport(*result, (*target)->machine(), **target,
                                       options, /*moduleHash=*/0);

  // The report bound for the SRAM kernel is rejected against the L2 kernel.
  llvm::Expected<CoveringPlan> replayed =
      readPlanReport(report, **target, *graphB);
  EXPECT_FALSE(bool(replayed));

  // The control: it replays against its own graph.
  llvm::Expected<CoveringPlan> control =
      readPlanReport(report, **target, *graphA);
  EXPECT_TRUE(bool(control)) << llvm::toString(control.takeError());
}

//===----------------------------------------------------------------------===//
// Same-kind movement between distinct concrete memory nodes (task B5)
//===----------------------------------------------------------------------===//

namespace {

/// Two SRAM nodes of one abstract kind, each visible to its own executor, with
/// a single legal link sram.0 -> sram.1. The add rule can only run on worker_a
/// (which sees sram.0) and the mul rule only on worker_b (which sees sram.1),
/// so the edge between them is a real same-kind transfer: kind equality no
/// longer collapses it, because the two nodes are distinct concrete storage.
constexpr llvm::StringLiteral kSameKindMachine = R"yaml(
schema: llk.machine.v2
target: same-kind
clock_hz: 1000000000
worker_threads: 2
executors:
  - id: cluster.a
    kind: cluster
  - id: worker.a
    kind: core
    parent: cluster.a
  - id: cluster.b
    kind: cluster
  - id: worker.b
    kind: pe
    parent: cluster.b
memories:
  - id: sram.0
    kind: sram
    visible_from: cluster.a
    capacity_bytes: 1048576
    alignment_bytes: 64
    supported_layouts: [row_major]
    bandwidth_bytes_per_cycle: 64
    latency_cycles: 4
  - id: sram.1
    kind: sram
    visible_from: cluster.b
    capacity_bytes: 1048576
    alignment_bytes: 64
    supported_layouts: [row_major]
    bandwidth_bytes_per_cycle: 64
    latency_cycles: 55
compute:
  - id: vpu
    kind: vector_engine
    attached_to: worker.a
    element_types: [f32]
    shapes: [[8]]
    lanes: {f32: 8}
    issue_cycles: 1
    latency_cycles: 1
    supported_layouts: [row_major]
transfer_engines:
  - id: dma.a
    kind: dma
    attached_to: cluster.a
    count: 1
    max_outstanding: 1
links:
  - id: sram_to_sram.1
    source: sram.0
    destination: sram.1
    bandwidth_bytes_per_cycle: 64
    latency_cycles: 55
    transaction_bytes: 64
    transfer_engines: [dma.a]
)yaml";

/// Both rules bind the *same* abstract kind (`sram`); only the executor each
/// requires distinguishes the two concrete nodes. The executor kinds (`core`,
/// `pe`) are valid owner kinds that the rule predicate can name.
constexpr llvm::StringLiteral kSameKindRules = R"llkmap(
rule t.add {
  match micro.vector(op = "add");
  require executor kind core;
  require memory kind sram;
  bundle "b.add";
  emit "e1";
  cost 1;
}
rule t.mul {
  match micro.vector(op = "mul");
  require executor kind pe;
  require memory kind sram;
  bundle "b.mul";
  emit "e1";
  cost 1;
}
)llkmap";

llvm::Expected<std::unique_ptr<MappingTarget>>
makeSameKindTarget(llvm::StringRef machineText, llvm::StringRef name) {
  llvm::Expected<mlir::llk::machine::MachineModel> machine =
      mlir::llk::machine::parseMachineModel(machineText, "<test>");
  if (!machine)
    return machine.takeError();
  llvm::Expected<LayoutRegistry> layouts = parseLayoutText("", "<test>");
  if (!layouts)
    return layouts.takeError();
  llvm::Expected<RuleRegistry> rules = parseRuleText(kSameKindRules, "<test>");
  if (!rules)
    return rules.takeError();
  return std::make_unique<FileMappingTarget>(
      name.str(), std::move(*machine), std::move(*layouts), std::move(*rules),
      std::vector<std::string>{"e1"});
}

llvm::Expected<std::unique_ptr<MappingTarget>> sameKindTarget() {
  return makeSameKindTarget(kSameKindMachine, "same-kind");
}

/// The same machine with a third SRAM node between the two, reachable only
/// through `cluster.c`. No direct sram.0 -> sram.1 link exists, so the only
/// route is two same-kind hops: sram.0 -> sram.2 -> sram.1.
constexpr llvm::StringLiteral kSameKindStagedMachine = R"yaml(
schema: llk.machine.v2
target: same-kind-staged
clock_hz: 1000000000
worker_threads: 2
executors:
  - id: cluster.a
    kind: cluster
  - id: worker.a
    kind: core
    parent: cluster.a
  - id: cluster.b
    kind: cluster
  - id: worker.b
    kind: pe
    parent: cluster.b
  - id: cluster.c
    kind: cluster
  - id: worker.c
    kind: worker
    parent: cluster.c
memories:
  - id: sram.0
    kind: sram
    visible_from: cluster.a
    capacity_bytes: 1048576
    alignment_bytes: 64
    supported_layouts: [row_major]
    bandwidth_bytes_per_cycle: 64
    latency_cycles: 4
  - id: sram.2
    kind: sram
    visible_from: cluster.c
    capacity_bytes: 1048576
    alignment_bytes: 64
    supported_layouts: [row_major]
    bandwidth_bytes_per_cycle: 64
    latency_cycles: 4
  - id: sram.1
    kind: sram
    visible_from: cluster.b
    capacity_bytes: 1048576
    alignment_bytes: 64
    supported_layouts: [row_major]
    bandwidth_bytes_per_cycle: 64
    latency_cycles: 55
compute:
  - id: vpu
    kind: vector_engine
    attached_to: worker.a
    element_types: [f32]
    shapes: [[8]]
    lanes: {f32: 8}
    issue_cycles: 1
    latency_cycles: 1
    supported_layouts: [row_major]
transfer_engines:
  - id: dma.a
    kind: dma
    attached_to: cluster.a
    count: 1
    max_outstanding: 1
  - id: dma.c
    kind: dma
    attached_to: cluster.c
    count: 1
    max_outstanding: 1
links:
  - id: sram.0_to_sram.2
    source: sram.0
    destination: sram.2
    bandwidth_bytes_per_cycle: 64
    latency_cycles: 8
    transaction_bytes: 64
    transfer_engines: [dma.a]
  - id: sram.2_to_sram.1
    source: sram.2
    destination: sram.1
    bandwidth_bytes_per_cycle: 64
    latency_cycles: 12
    transaction_bytes: 64
    transfer_engines: [dma.c]
)yaml";

llvm::Expected<std::unique_ptr<MappingTarget>> sameKindStagedTarget() {
  return makeSameKindTarget(kSameKindStagedMachine, "same-kind-staged");
}

/// Every stamped movement in `module`, in order, with its source and
/// destination node ids.
struct StampedHop {
  std::string srcNode;
  std::string dstNode;
  uint64_t connection = 0;
  uint64_t hop = 0;
};

std::vector<StampedHop> stampedHops(ModuleOp module) {
  std::vector<StampedHop> hops;
  module->walk([&](Operation *op) {
    if (op->getName().getStringRef() != "micro.tile_async_copy")
      return;
    auto dst = op->getAttrOfType<StringAttr>("micro.dst_node");
    auto src = op->getAttrOfType<StringAttr>("micro.src_node");
    if (!dst || !src)
      return;
    StampedHop hook;
    hook.srcNode = src.getValue().str();
    hook.dstNode = dst.getValue().str();
    if (auto connection = op->getAttrOfType<IntegerAttr>("micro.connection"))
      hook.connection = connection.getValue().getZExtValue();
    if (auto hop = op->getAttrOfType<IntegerAttr>("micro.hop"))
      hook.hop = hop.getValue().getZExtValue();
    hops.push_back(std::move(hook));
  });
  return hops;
}

} // namespace

// A movement between two distinct concrete memories of one abstract kind is
// real work. Kind equality must not collapse it: the copy records the two node
// ids, so one SRAM value can move sram.0 -> sram.1 while both tile types stay
// `memory = sram`.
TEST(PlanBinder, MaterializesASameKindMovementBetweenDistinctNodes) {
  Fixture f = makeTileFixture(kTileKernelOneConsumer);
  ASSERT_TRUE(f.module);
  llvm::Expected<std::unique_ptr<MappingTarget>> target = sameKindTarget();
  ASSERT_TRUE(bool(target)) << llvm::toString(target.takeError());

  auto p = selectPlan(*f.context, *f.module, **target);
  ASSERT_TRUE(bool(p)) << llvm::toString(p.takeError());
  ASSERT_FALSE(p->connectionPlans.empty());
  const PlanConnection &connection = p->connectionPlans.front();
  ASSERT_EQ(connection.kind, ConnectionKind::Transfer);
  EXPECT_EQ(connection.route,
            (llvm::SmallVector<MemoryNodeId>{"sram.0", "sram.1"}));

  const size_t copiesBefore = countOps(*f.module, "micro.tile_async_copy");
  auto b = bindCanonical(*f.module, *p, **target);
  ASSERT_TRUE(bool(b)) << llvm::toString(b.takeError());
  for (const std::string &note : b->unmaterialized)
    ADD_FAILURE() << note;
  EXPECT_TRUE(b->unmaterialized.empty());
  EXPECT_EQ(countOps(*b->module, "micro.tile_async_copy"), copiesBefore + 1);

  std::vector<StampedHop> hops = stampedHops(*b->module);
  ASSERT_EQ(hops.size(), 1u);
  EXPECT_EQ(hops[0].srcNode, "sram.0");
  EXPECT_EQ(hops[0].dstNode, "sram.1");
  EXPECT_GT(hops[0].hop, 0u);
  EXPECT_NE(hops[0].connection, 0u);

  // Both tile types keep the abstract `sram` kind; only the node identity
  // distinguishes them.
  Operation *copy = tileCopy(*b->module);
  ASSERT_NE(copy, nullptr);
  EXPECT_NE(typeText(copy->getResult(0).getType())
                .find("memory = #micro.memory<sram>"),
            std::string::npos);

  {
    llvm::Error verification = verifyMappedMicroIR(*b->module, **target);
    EXPECT_FALSE(bool(verification)) << llvm::toString(std::move(verification));
  }
}

// The path is only real when the machine declares a link between the two
// concrete nodes. A same-kind route with no such link must not be materialized
// into an unverifiable copy.
TEST(PlanBinder, RejectsASameKindMovementWithNoLink) {
  Fixture f = makeTileFixture(kTileKernelOneConsumer);
  ASSERT_TRUE(f.module);
  llvm::Expected<std::unique_ptr<MappingTarget>> target = sameKindTarget();
  ASSERT_TRUE(bool(target)) << llvm::toString(target.takeError());
  auto p = selectPlan(*f.context, *f.module, **target);
  ASSERT_TRUE(bool(p)) << llvm::toString(p.takeError());
  ASSERT_FALSE(p->connectionPlans.empty());

  // The reverse of the only declared link: no sram.1 -> sram.0 hop exists.
  p->connectionPlans.front().route = {"sram.1", "sram.0"};

  auto b = bindCanonical(*f.module, *p, **target);
  ASSERT_TRUE(bool(b)) << llvm::toString(b.takeError());
  EXPECT_FALSE(b->unmaterialized.empty());
  llvm::Error error = verifyMappedMicroIR(*b->module, **target);
  ASSERT_TRUE(bool(error));
  EXPECT_NE(llvm::toString(std::move(error)).find("has no link"),
            std::string::npos);
}

// Two equal node ids are one concrete memory: a "movement" between them is not
// work, even though their kind is the one a same-kind route names.
TEST(PlanBinder, RejectsASameKindMovementBetweenOneNodeAndItself) {
  Fixture f = makeTileFixture(kTileKernelOneConsumer);
  ASSERT_TRUE(f.module);
  llvm::Expected<std::unique_ptr<MappingTarget>> target = sameKindTarget();
  ASSERT_TRUE(bool(target)) << llvm::toString(target.takeError());
  auto p = selectPlan(*f.context, *f.module, **target);
  ASSERT_TRUE(bool(p)) << llvm::toString(p.takeError());
  ASSERT_FALSE(p->connectionPlans.empty());

  p->connectionPlans.front().route = {"sram.0", "sram.0"};

  auto b = bindCanonical(*f.module, *p, **target);
  ASSERT_TRUE(bool(b)) << llvm::toString(b.takeError());
  EXPECT_FALSE(b->unmaterialized.empty());
  EXPECT_TRUE(stampedHops(*b->module).empty());
}

// A same-kind hop carried by a transfer engine the machine does not declare is
// not executable; the resolved route's engine set must be supported.
TEST(PlanBinder, RejectsASameKindMovementOnAnUnsupportedEngine) {
  Fixture f = makeTileFixture(kTileKernelOneConsumer);
  ASSERT_TRUE(f.module);
  llvm::Expected<std::unique_ptr<MappingTarget>> target = sameKindTarget();
  ASSERT_TRUE(bool(target)) << llvm::toString(target.takeError());
  auto p = selectPlan(*f.context, *f.module, **target);
  ASSERT_TRUE(bool(p)) << llvm::toString(p.takeError());
  auto b = bindCanonical(*f.module, *p, **target);
  ASSERT_TRUE(bool(b)) << llvm::toString(b.takeError());
  ASSERT_EQ(stampedHops(*b->module).size(), 1u);

  rewriteAllRouteEngines(b->kernel, *f.context, {"no_such_engine"});
  llvm::Error error = verifyMappedMicroIR(*b->module, **target);
  ASSERT_TRUE(bool(error));
  std::string text = llvm::toString(std::move(error));
  EXPECT_NE(text.find("no_such_engine"), std::string::npos) << text;
}

// A copy whose recorded source node is not the resolved hop's source is not the
// connection it names: the concrete node identity must match the route.
TEST(PlanBinder, RejectsASameKindMovementWhoseSourceDoesNotMatchItsRoute) {
  Fixture f = makeTileFixture(kTileKernelOneConsumer);
  ASSERT_TRUE(f.module);
  llvm::Expected<std::unique_ptr<MappingTarget>> target = sameKindTarget();
  ASSERT_TRUE(bool(target)) << llvm::toString(target.takeError());
  auto p = selectPlan(*f.context, *f.module, **target);
  ASSERT_TRUE(bool(p)) << llvm::toString(p.takeError());
  auto b = bindCanonical(*f.module, *p, **target);
  ASSERT_TRUE(bool(b)) << llvm::toString(b.takeError());

  Operation *copy = tileCopy(*b->module);
  ASSERT_NE(copy, nullptr);
  copy->setAttr("micro.src_node", StringAttr::get(f.context.get(), "sram.9"));
  llvm::Error error = verifyMappedMicroIR(*b->module, **target);
  ASSERT_TRUE(bool(error));
  EXPECT_NE(llvm::toString(std::move(error)).find("micro.src_node"),
            std::string::npos);
}

// A two-hop route through same-kind adjacent memories must materialize one
// correctly attributed copy per hop: sram.0 -> sram.2 -> sram.1.
TEST(PlanBinder, MaterializesATwoHopRouteOfSameKindNodes) {
  Fixture f = makeTileFixture(kTileKernelOneConsumer);
  ASSERT_TRUE(f.module);
  llvm::Expected<std::unique_ptr<MappingTarget>> target =
      sameKindStagedTarget();
  ASSERT_TRUE(bool(target)) << llvm::toString(target.takeError());

  auto p = selectPlan(*f.context, *f.module, **target);
  ASSERT_TRUE(bool(p)) << llvm::toString(p.takeError());
  ASSERT_FALSE(p->connectionPlans.empty());
  ASSERT_EQ(p->connectionPlans.front().kind, ConnectionKind::Transfer);
  EXPECT_EQ(p->connectionPlans.front().route,
            (llvm::SmallVector<MemoryNodeId>{"sram.0", "sram.2", "sram.1"}));

  const size_t copiesBefore = countOps(*f.module, "micro.tile_async_copy");
  auto b = bindCanonical(*f.module, *p, **target);
  ASSERT_TRUE(bool(b)) << llvm::toString(b.takeError());
  for (const std::string &note : b->unmaterialized)
    ADD_FAILURE() << note;
  EXPECT_TRUE(b->unmaterialized.empty());
  EXPECT_EQ(countOps(*b->module, "micro.tile_async_copy"), copiesBefore + 2);

  std::vector<StampedHop> hops = stampedHops(*b->module);
  ASSERT_EQ(hops.size(), 2u);
  EXPECT_EQ(hops[0].srcNode, "sram.0");
  EXPECT_EQ(hops[0].dstNode, "sram.2");
  EXPECT_EQ(hops[0].hop, 1u);
  EXPECT_EQ(hops[1].srcNode, "sram.2");
  EXPECT_EQ(hops[1].dstNode, "sram.1");
  EXPECT_EQ(hops[1].hop, 2u);

  {
    llvm::Error verification = verifyMappedMicroIR(*b->module, **target);
    EXPECT_FALSE(bool(verification)) << llvm::toString(std::move(verification));
  }
}

//===----------------------------------------------------------------------===//
// Explicit gather semantics and synchronization (task B6)
//===----------------------------------------------------------------------===//

namespace {

/// Two distinct producers feeding one consumer, so a gather can combine them.
/// The producers differ in their vector op, so extraction keeps them as two
/// nodes rather than folding identical content together.
constexpr llvm::StringLiteral kGatherKernel = R"mlir(
module {
  micro.kernel @gather {
    %a = micro.tile_alloc : !micro.tile<8x8xf32, memory = #micro.memory<sram>>
    %p1 = micro.vector "add" %a, %a : !micro.tile<8x8xf32, memory = #micro.memory<sram>>, !micro.tile<8x8xf32, memory = #micro.memory<sram>> -> !micro.tile<8x8xf32, memory = #micro.memory<sram>>
    %p2 = micro.vector "max" %a, %a : !micro.tile<8x8xf32, memory = #micro.memory<sram>>, !micro.tile<8x8xf32, memory = #micro.memory<sram>> -> !micro.tile<8x8xf32, memory = #micro.memory<sram>>
    %c = micro.vector "mul" %p1, %p2 : !micro.tile<8x8xf32, memory = #micro.memory<sram>>, !micro.tile<8x8xf32, memory = #micro.memory<sram>> -> !micro.tile<8x8xf32, memory = #micro.memory<sram>>
    micro.yield
  }
}
)mlir";

/// Producers of different extents, for the concatenation control (8 + 4 = 12
/// along the concatenated axis).
constexpr llvm::StringLiteral kConcatKernel = R"mlir(
module {
  micro.kernel @concat {
    %a = micro.tile_alloc : !micro.tile<8x8xf32, memory = #micro.memory<sram>>
    %b = micro.tile_alloc : !micro.tile<8x4xf32, memory = #micro.memory<sram>>
    %p1 = micro.vector "add" %a, %a : !micro.tile<8x8xf32, memory = #micro.memory<sram>>, !micro.tile<8x8xf32, memory = #micro.memory<sram>> -> !micro.tile<8x8xf32, memory = #micro.memory<sram>>
    %p2 = micro.vector "max" %b, %b : !micro.tile<8x4xf32, memory = #micro.memory<sram>>, !micro.tile<8x4xf32, memory = #micro.memory<sram>> -> !micro.tile<8x4xf32, memory = #micro.memory<sram>>
    micro.yield
  }
}
)mlir";

/// The id of the `micro.vector` node whose `op` attribute is `vecOp`, or
/// nullopt. Node ids are content-canonical, so the test identifies a node by
/// what it is rather than by position.
std::optional<WorkloadNodeId> vectorNodeId(const WorkloadGraph &graph,
                                           llvm::StringRef vecOp) {
  for (const WorkloadNode &node : graph.getNodes()) {
    if (node.opName != "micro.vector")
      continue;
    auto attr = node.attributes ? node.attributes.getAs<StringAttr>("op")
                                : StringAttr();
    if (attr && attr.getValue() == vecOp)
      return node.id;
  }
  return std::nullopt;
}

/// The `micro.gather` in `module`, or null.
Operation *gatherOp(ModuleOp module) {
  Operation *found = nullptr;
  module->walk([&](Operation *op) {
    if (!found && op->getName().getStringRef() == "micro.gather")
      found = op;
  });
  return found;
}

/// Builds a hand-built gather plan over `module`'s kernel: the two named
/// producer vector ops feed one gather with `semantics`, staged along the
/// SRAM -> L2 route, and (optionally) a named consumer reads the result.
CoveringPlan buildGatherPlan(ModuleOp module, llvm::StringRef producerA,
                             llvm::StringRef producerB,
                             GatherSemantics semantics,
                             std::optional<uint64_t> concatAxis,
                             std::optional<WorkloadNodeId> consumerNode) {
  WorkloadGraphBinding binding;
  llvm::Expected<WorkloadGraph> graph =
      extractWorkloadGraph(findKernel(module), &binding);
  EXPECT_TRUE(static_cast<bool>(graph));
  if (!graph)
    return CoveringPlan{};

  std::optional<WorkloadNodeId> a = vectorNodeId(*graph, producerA);
  std::optional<WorkloadNodeId> b = vectorNodeId(*graph, producerB);
  EXPECT_TRUE(a.has_value());
  EXPECT_TRUE(b.has_value());
  if (!a || !b)
    return CoveringPlan{};

  CoveringPlan plan;
  plan.id = 77;
  PlanConnection connection;
  connection.id = 7;
  connection.kind = ConnectionKind::Reduce;
  connection.value = graph->findNode(*a)->outputs.front().value;
  connection.gatherSemantics = semantics;
  connection.concatAxis = concatAxis;
  connection.route = llvm::SmallVector<MemoryNodeId>{"sram.0", "l2.0"};
  connection.producerPorts = {PortRef{*a, PortDirection::Output, 0},
                              PortRef{*b, PortDirection::Output, 0}};
  if (consumerNode)
    connection.consumerPorts = {
        PortRef{*consumerNode, PortDirection::Input, 0}};
  plan.connectionPlans.push_back(std::move(connection));
  return plan;
}

} // namespace

// A multi-producer connection with declared Sum semantics materializes a
// `micro.gather` whose operands are the staged feeds and whose result the
// recorded consumer reads. Nothing is inferred from the producer count.
TEST(PlanBinder, MaterializesAGatherUnderItsDeclaredSemantics) {
  Fixture f = makeTileFixture(kGatherKernel);
  ASSERT_TRUE(f.module);
  llvm::Expected<std::unique_ptr<MappingTarget>> target = tileTarget();
  ASSERT_TRUE(bool(target)) << llvm::toString(target.takeError());

  WorkloadGraphBinding binding;
  llvm::Expected<WorkloadGraph> graph =
      extractWorkloadGraph(findKernel(*f.module), &binding);
  ASSERT_TRUE(bool(graph)) << llvm::toString(graph.takeError());
  std::optional<WorkloadNodeId> consumer = vectorNodeId(*graph, "mul");
  ASSERT_TRUE(consumer.has_value());

  CoveringPlan plan = buildGatherPlan(
      *f.module, "add", "max", GatherSemantics::Sum, std::nullopt, consumer);
  ASSERT_FALSE(plan.connectionPlans.empty());

  const size_t copiesBefore = countOps(*f.module, "micro.tile_async_copy");
  llvm::Expected<BoundPlan> b = bindCanonical(*f.module, plan, **target);
  ASSERT_TRUE(bool(b)) << llvm::toString(b.takeError());
  for (const std::string &note : b->unmaterialized)
    ADD_FAILURE() << note;
  EXPECT_TRUE(b->unmaterialized.empty());

  Operation *gather = gatherOp(*b->module);
  ASSERT_NE(gather, nullptr);
  auto kind = gather->getAttrOfType<StringAttr>("kind");
  ASSERT_TRUE(kind);
  EXPECT_EQ(kind.getValue(), "sum");
  EXPECT_EQ(gather->getNumOperands(), 2u);
  EXPECT_FALSE(gather->hasAttr("axis"));

  // Each feed is staged over the selected route: two copies and two waits.
  EXPECT_EQ(countOps(*b->module, "micro.tile_async_copy"), copiesBefore + 2);
  EXPECT_EQ(countOps(*b->module, "micro.wait"), 2u);

  // The consumer reads the gathered result.
  Operation *mul = vectorOp(*b->module, "mul");
  ASSERT_NE(mul, nullptr);
  EXPECT_EQ(mul->getOperand(0), gather->getResult(0));
}

// The Max control: the same shape-compatible gather under a different declared
// semantics emits a `max` gather, not a sum.
TEST(PlanBinder, MaterializesAMaxGatherUnderItsSemantics) {
  Fixture f = makeTileFixture(kGatherKernel);
  ASSERT_TRUE(f.module);
  llvm::Expected<std::unique_ptr<MappingTarget>> target = tileTarget();
  ASSERT_TRUE(bool(target)) << llvm::toString(target.takeError());

  WorkloadGraphBinding binding;
  llvm::Expected<WorkloadGraph> graph =
      extractWorkloadGraph(findKernel(*f.module), &binding);
  ASSERT_TRUE(bool(graph)) << llvm::toString(graph.takeError());
  std::optional<WorkloadNodeId> consumer = vectorNodeId(*graph, "mul");
  ASSERT_TRUE(consumer.has_value());

  CoveringPlan plan = buildGatherPlan(
      *f.module, "add", "max", GatherSemantics::Max, std::nullopt, consumer);
  llvm::Expected<BoundPlan> b = bindCanonical(*f.module, plan, **target);
  ASSERT_TRUE(bool(b)) << llvm::toString(b.takeError());
  EXPECT_TRUE(b->unmaterialized.empty());

  Operation *gather = gatherOp(*b->module);
  ASSERT_NE(gather, nullptr);
  EXPECT_EQ(gather->getAttrOfType<StringAttr>("kind").getValue(), "max");
}

// The Concatenate control: the result extent along the named axis is exactly
// the sum of the input extents (8 + 4 = 12), and every other extent matches.
TEST(PlanBinder, MaterializesAConcatGatherWithTheExactResultingExtent) {
  Fixture f = makeTileFixture(kConcatKernel);
  ASSERT_TRUE(f.module);
  llvm::Expected<std::unique_ptr<MappingTarget>> target = tileTarget();
  ASSERT_TRUE(bool(target)) << llvm::toString(target.takeError());

  CoveringPlan plan =
      buildGatherPlan(*f.module, "add", "max", GatherSemantics::Concatenate,
                      uint64_t{1}, std::nullopt);
  llvm::Expected<BoundPlan> b = bindCanonical(*f.module, plan, **target);
  ASSERT_TRUE(bool(b)) << llvm::toString(b.takeError());
  for (const std::string &note : b->unmaterialized)
    ADD_FAILURE() << note;
  EXPECT_TRUE(b->unmaterialized.empty());

  Operation *gather = gatherOp(*b->module);
  ASSERT_NE(gather, nullptr);
  EXPECT_EQ(gather->getAttrOfType<StringAttr>("kind").getValue(), "concat");
  auto axis = gather->getAttrOfType<IntegerAttr>("axis");
  ASSERT_TRUE(axis);
  EXPECT_EQ(axis.getInt(), 1);
  // The concatenated tile is 8x(8+4) in the staging memory.
  EXPECT_NE(typeText(gather->getResult(0).getType()).find("8x12xf32"),
            std::string::npos)
      << typeText(gather->getResult(0).getType());
}

// A `Concatenate` without an axis cannot state its resulting extent, so it is
// not materialized: the partial contract reports it and the executable one
// refuses. Nothing is guessed.
TEST(PlanBinder, AConcatGatherWithoutAnAxisIsRejected) {
  Fixture f = makeTileFixture(kConcatKernel);
  ASSERT_TRUE(f.module);
  llvm::Expected<std::unique_ptr<MappingTarget>> target = tileTarget();
  ASSERT_TRUE(bool(target)) << llvm::toString(target.takeError());

  CoveringPlan plan =
      buildGatherPlan(*f.module, "add", "max", GatherSemantics::Concatenate,
                      std::nullopt, std::nullopt);
  llvm::Expected<BoundPlan> partial = bindCanonical(*f.module, plan, **target);
  ASSERT_TRUE(bool(partial)) << llvm::toString(partial.takeError());
  ASSERT_FALSE(partial->unmaterialized.empty());
  EXPECT_EQ(gatherOp(*partial->module), nullptr);

  llvm::Expected<BoundPlan> executable =
      bindCanonical(*f.module, plan, **target, BindContract::Executable);
  ASSERT_FALSE(bool(executable));
}

// A multi-producer connection with no declared semantics stays Partial-only
// under the stable `reduce_not_materialized` reason, and an executable contract
// refuses it: a topology alone is not arithmetic.
TEST(PlanBinder, MissingGatherSemanticsRejectsExecutableBinding) {
  Fixture f = makeTileFixture(kGatherKernel);
  ASSERT_TRUE(f.module);
  llvm::Expected<std::unique_ptr<MappingTarget>> target = tileTarget();
  ASSERT_TRUE(bool(target)) << llvm::toString(target.takeError());

  CoveringPlan plan =
      buildGatherPlan(*f.module, "add", "max", GatherSemantics::Sum,
                      std::nullopt, std::nullopt);
  ASSERT_FALSE(plan.connectionPlans.empty());
  plan.connectionPlans.front().gatherSemantics.reset(); // no declared semantics

  llvm::Expected<BoundPlan> partial = bindCanonical(*f.module, plan, **target);
  ASSERT_TRUE(bool(partial)) << llvm::toString(partial.takeError());
  ASSERT_EQ(partial->unmaterialized.size(), 1u);
  EXPECT_NE(partial->unmaterialized.front().find("reduce_not_materialized"),
            std::string::npos)
      << partial->unmaterialized.front();
  EXPECT_EQ(gatherOp(*partial->module), nullptr);

  llvm::Expected<BoundPlan> executable =
      bindCanonical(*f.module, plan, **target, BindContract::Executable);
  ASSERT_FALSE(bool(executable));
  EXPECT_NE(
      llvm::toString(executable.takeError()).find("reduce_not_materialized"),
      std::string::npos);
}

// A gather's declared semantics and axis are execution-affecting content, so
// they join the connection id: two connections differing only in what their
// combination means must not collapse to one id.
TEST(PlanBinder, GatherSemanticsJoinsTheConnectionIdentity) {
  ConnectionPlan base;
  base.value = 3;
  base.kind = ConnectionKind::Reduce;
  base.memoryRoute = {"sram.0", "l2.0"};

  ConnectionPlan sum = base;
  sum.gatherSemantics = GatherSemantics::Sum;
  ConnectionPlan max = base;
  max.gatherSemantics = GatherSemantics::Max;
  ConnectionPlan concat = base;
  concat.gatherSemantics = GatherSemantics::Concatenate;
  concat.concatAxis = 1;

  EXPECT_NE(computeConnectionId(base), computeConnectionId(sum));
  EXPECT_NE(computeConnectionId(sum), computeConnectionId(max));
  EXPECT_NE(computeConnectionId(sum), computeConnectionId(concat));
  EXPECT_NE(computeConnectionId(max), computeConnectionId(concat));

  // The axis is identity-bearing too: two concatenations along different axes
  // are different work.
  ConnectionPlan concatAxis0 = concat;
  concatAxis0.concatAxis = 0;
  EXPECT_NE(computeConnectionId(concat), computeConnectionId(concatAxis0));
}

// A gather's semantics, axis and producer occurrences survive the schema-v2
// round trip, and a tampered recorded semantics is rejected -- not silently
// downgraded to a semantics-less Partial gather.
TEST(PlanBinder, RoundTripsAndValidatesGatherSemantics) {
  Fixture f = makeTileFixture(kGatherKernel);
  ASSERT_TRUE(f.module);
  llvm::Expected<std::unique_ptr<MappingTarget>> target = tileTarget();
  ASSERT_TRUE(bool(target)) << llvm::toString(target.takeError());

  WorkloadGraphBinding binding;
  llvm::Expected<WorkloadGraph> graph =
      extractWorkloadGraph(findKernel(*f.module), &binding);
  ASSERT_TRUE(bool(graph)) << llvm::toString(graph.takeError());
  std::optional<WorkloadNodeId> consumer = vectorNodeId(*graph, "mul");
  ASSERT_TRUE(consumer.has_value());

  CoveringPlan plan =
      buildGatherPlan(*f.module, "add", "max", GatherSemantics::Concatenate,
                      uint64_t{0}, consumer);
  ASSERT_FALSE(plan.connectionPlans.empty());
  ASSERT_FALSE(plan.connectionPlans.front().producerPorts.empty());

  auto b = bindCanonical(*f.module, plan, **target);
  ASSERT_TRUE(bool(b)) << llvm::toString(b.takeError());

  llvm::Expected<CoveringPlan> decoded =
      decodeSelectedPlan(*b->module, **target);
  ASSERT_TRUE(bool(decoded)) << llvm::toString(decoded.takeError());
  const PlanConnection *reduce = nullptr;
  for (const PlanConnection &connection : decoded->connectionPlans)
    if (connection.kind == ConnectionKind::Reduce)
      reduce = &connection;
  ASSERT_NE(reduce, nullptr);
  ASSERT_TRUE(reduce->gatherSemantics.has_value());
  EXPECT_EQ(*reduce->gatherSemantics, GatherSemantics::Concatenate);
  ASSERT_TRUE(reduce->concatAxis.has_value());
  EXPECT_EQ(*reduce->concatAxis, 0u);
  EXPECT_EQ(reduce->producerPorts, plan.connectionPlans.front().producerPorts);

  // Tamper the recorded semantics: an unknown word must be rejected on decode,
  // not silently dropped.
  Operation *kernel = findKernel(*b->module);
  auto routes = kernel->getAttrOfType<ArrayAttr>("micro.routes");
  ASSERT_TRUE(routes);
  llvm::SmallVector<Attribute> updated;
  bool tampered = false;
  for (Attribute element : routes) {
    auto dict = cast<DictionaryAttr>(element);
    if (dict.get("gather_semantics")) {
      NamedAttrList attributes(dict);
      attributes.set("gather_semantics",
                     StringAttr::get(f.context.get(), "bogus"));
      updated.push_back(attributes.getDictionary(f.context.get()));
      tampered = true;
    } else {
      updated.push_back(element);
    }
  }
  ASSERT_TRUE(tampered);
  kernel->setAttr("micro.routes", ArrayAttr::get(f.context.get(), updated));
  llvm::Expected<CoveringPlan> rejected =
      decodeSelectedPlan(*b->module, **target);
  EXPECT_FALSE(bool(rejected));
}

// Sum/Max require compatible element types/shapes. A Sum over feeds of
// different extents is not materializable, so it is refused with a stable
// reason rather than emitting a gather that violates its own verifier.
TEST(PlanBinder, SumGatherWithMismatchedFeedShapesIsRejected) {
  Fixture f = makeTileFixture(kConcatKernel); // producers are 8x8 and 8x4
  ASSERT_TRUE(f.module);
  llvm::Expected<std::unique_ptr<MappingTarget>> target = tileTarget();
  ASSERT_TRUE(bool(target)) << llvm::toString(target.takeError());

  CoveringPlan plan =
      buildGatherPlan(*f.module, "add", "max", GatherSemantics::Sum,
                      std::nullopt, std::nullopt);
  llvm::Expected<BoundPlan> partial = bindCanonical(*f.module, plan, **target);
  ASSERT_TRUE(bool(partial)) << llvm::toString(partial.takeError());
  ASSERT_EQ(partial->unmaterialized.size(), 1u);
  EXPECT_NE(
      partial->unmaterialized.front().find("gather_feed_types_incompatible"),
      std::string::npos)
      << partial->unmaterialized.front();
  EXPECT_EQ(gatherOp(*partial->module), nullptr);

  llvm::Expected<BoundPlan> executable =
      bindCanonical(*f.module, plan, **target, BindContract::Executable);
  ASSERT_FALSE(bool(executable));
}

//===----------------------------------------------------------------------===//
// Synchronization decisions drive waits and barriers (task B6)
//===----------------------------------------------------------------------===//

// A plan whose synchronization requires a barrier emits one, over the tokens of
// the movement it orders; verification accepts it because the required
// synchronization is represented.
TEST(PlanBinder, AParallelEngineMovementEmitsABarrier) {
  Fixture f = makeTileFixture(kTileKernelOneConsumer);
  ASSERT_TRUE(f.module);
  llvm::Expected<std::unique_ptr<MappingTarget>> target = tileTarget();
  ASSERT_TRUE(bool(target)) << llvm::toString(target.takeError());
  auto p = selectPlan(*f.context, *f.module, **target);
  ASSERT_TRUE(bool(p)) << llvm::toString(p.takeError());
  ASSERT_FALSE(p->connectionPlans.empty());

  // The plan decides the movement must be barrier-synchronized.
  p->synchronization.clear();
  SynchronizationStep step;
  step.id = 0;
  step.waitsFor = {p->connectionPlans.front().id};
  step.requiresBarrier = true;
  p->synchronization.push_back(step);

  auto b = bindCanonical(*f.module, *p, **target);
  ASSERT_TRUE(bool(b)) << llvm::toString(b.takeError());
  for (const std::string &note : b->unmaterialized)
    ADD_FAILURE() << note;
  EXPECT_TRUE(b->unmaterialized.empty());
  EXPECT_EQ(countOps(*b->module, "micro.barrier"), 1u);
  EXPECT_FALSE(bool(verifyMappedMicroIR(*b->module, **target)));
}

// The same plan without the required barrier no longer represents the
// synchronization it recorded, so verification rejects it: a required decision
// cannot be silently dropped.
TEST(PlanBinder, ARequiredBarrierMustBeRepresented) {
  Fixture f = makeTileFixture(kTileKernelOneConsumer);
  ASSERT_TRUE(f.module);
  llvm::Expected<std::unique_ptr<MappingTarget>> target = tileTarget();
  ASSERT_TRUE(bool(target)) << llvm::toString(target.takeError());
  auto p = selectPlan(*f.context, *f.module, **target);
  ASSERT_TRUE(bool(p)) << llvm::toString(p.takeError());
  ASSERT_FALSE(p->connectionPlans.empty());

  p->synchronization.clear();
  SynchronizationStep step;
  step.id = 0;
  step.waitsFor = {p->connectionPlans.front().id};
  step.requiresBarrier = true;
  p->synchronization.push_back(step);

  auto b = bindCanonical(*f.module, *p, **target);
  ASSERT_TRUE(bool(b)) << llvm::toString(b.takeError());
  ASSERT_FALSE(bool(verifyMappedMicroIR(*b->module, **target)));

  Operation *barrier = nullptr;
  b->module->walk([&](Operation *op) {
    if (op->getName().getStringRef() == "micro.barrier")
      barrier = op;
  });
  ASSERT_NE(barrier, nullptr);
  barrier->erase();

  auto error = verifyMappedMicroIR(*b->module, **target);
  ASSERT_TRUE(bool(error));
  EXPECT_NE(llvm::toString(std::move(error)).find("does not express"),
            std::string::npos);
}

// Two independent movements that share no dependency record no barrier, so none
// is emitted: synchronization is added only where the plan requires it.
TEST(PlanBinder, IndependentCopiesReceiveNoBarrier) {
  Fixture f = makeTileFixture(kTileKernel); // add -> mul and add -> sub
  ASSERT_TRUE(f.module);
  llvm::Expected<std::unique_ptr<MappingTarget>> target = tileTarget();
  ASSERT_TRUE(bool(target)) << llvm::toString(target.takeError());
  auto p = selectPlan(*f.context, *f.module, **target);
  ASSERT_TRUE(bool(p)) << llvm::toString(p.takeError());
  ASSERT_GE(p->connectionPlans.size(), 2u);
  p->synchronization.clear();

  auto b = bindCanonical(*f.module, *p, **target);
  ASSERT_TRUE(bool(b)) << llvm::toString(b.takeError());
  EXPECT_TRUE(b->unmaterialized.empty());
  EXPECT_EQ(countOps(*b->module, "micro.barrier"), 0u);
  EXPECT_FALSE(bool(verifyMappedMicroIR(*b->module, **target)));
}
