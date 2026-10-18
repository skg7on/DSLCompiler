//===- framework_conformance.cpp - Movement/resource conformance ----------===//
//
// Part of the LLK Compiler Project, under the Apache License v2.0 with
// LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Framework C2 (issue #158). The #129 resource work already made storage,
// routing and selection concrete; these tests turn it into generic acceptance
// evidence: one strict source -> search -> bind -> shared-analysis chain over
// the shared fixtures, with literal identities, peaks and traffic, plus a
// control/transform pair proving a logical view moves nothing while a physical
// layout conversion costs real work and storage.

#include "resource_regression_fixture.h"

#include "CompletePlanEvaluation.h"

#include "LLK/Conversion/MappedCompilation.h"
#include "LLK/Dialect/LLKDialect.h"
#include "LLK/Dialect/Micro/MicroDialect.h"
#include "LLK/Mapping/CostEvent.h"
#include "LLK/Mapping/CoveringSearch.h"
#include "LLK/Mapping/MappingMetadata.h"
#include "LLK/Mapping/MappingPlan.h"
#include "LLK/Mapping/PlanReport.h"
#include "LLK/Perf/MicroDAG.h"
#include "LLK/Perf/SelectedKernelAnalysis.h"
#include "LLK/Target/GenericAccelerator/Mapping/GenericAcceleratorMappingTarget.h"
#include "LLK/Target/X86/Mapping/AVX2MappingTarget.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"

#include "gtest/gtest.h"

#include <set>
#include <string>
#include <utility>
#include <vector>

namespace {

using mlir::llk::mapping::ConnectionKind;
using mlir::llk::mapping::CostEventKind;
using mlir::llk::mapping::CoveringPlan;
using mlir::llk::mapping::MappingSearchOptions;
using mlir::llk::mapping::MappingSearchResult;
using mlir::llk::mapping::MappingTarget;
using mlir::llk::mapping::PlanConnection;
using mlir::llk::mapping::PlanCostEvent;
using mlir::llk::mapping::SearchMode;
using mlir::llk::mapping::StorageAllocation;

/// One strict conformance chain over a shared fixture: the searched and
/// finalized plan, the kernel the plan bound, and the shared analysis of that
/// kernel. The fixture owns the context all the MLIR objects live in.
struct ConformanceChain {
  issue129::ResourceCase fixture;
  MappingSearchResult search;
  CoveringPlan plan;
  mlir::OwningOpRef<mlir::ModuleOp> boundModule;
  mlir::Operation *kernel = nullptr;
  mlir::llk::perf::SelectedKernelAnalysis analysis;
};

llvm::Error conformanceError(const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(), message);
}

llvm::Expected<ConformanceChain> buildChain(llvm::StringRef caseName,
                                            unsigned topK) {
  auto fixture = issue129::resourceCase(caseName);
  if (!fixture)
    return fixture.takeError();

  MappingSearchOptions options;
  options.mode = SearchMode::Exact;
  options.topK = topK;
  auto search = issue129::searchCase(*fixture, options);
  if (!search)
    return search.takeError();
  if (search->plans.empty()) {
    std::string reason =
        "fixture '" + caseName.str() + "' searched no complete plan";
    if (!search->frontier.diagnostics.empty())
      reason += ": " + search->frontier.diagnostics.front().message;
    return conformanceError(reason);
  }

  ConformanceChain chain;
  chain.fixture = std::move(*fixture);
  chain.search = std::move(*search);
  chain.plan = chain.search.plans.front();

  // The plan the evaluator finalized is bound under the partial contract: the
  // chain is about static conformance, not about executable lowering.
  llk::MappedCompileOptions compile;
  compile.stop = llk::MappedStop::MappedMicro;
  compile.requireExecutable = false;
  auto bound = llk::compileMappedKernel(
      *chain.fixture.source, *chain.fixture.target, chain.plan, compile);
  if (!bound)
    return bound.takeError();
  chain.boundModule = std::move(bound->module);

  auto kernel = mlir::llk::perf::findMicroKernel(chain.boundModule.get(), "");
  if (!kernel)
    return kernel.takeError();
  chain.kernel = *kernel;

  auto analysis = mlir::llk::perf::analyzeSelectedKernel(
      chain.kernel, chain.fixture.target->machine(), /*requireComplete=*/true,
      chain.plan, chain.fixture.graph);
  if (!analysis)
    return analysis.takeError();
  chain.analysis = std::move(*analysis);
  return chain;
}

/// The parity every fixture must establish: the shared analysis is complete and
/// its scheduled cost is exactly the plan's attached score, not a second
/// derivation.
void expectStrictParity(const ConformanceChain &chain) {
  EXPECT_TRUE(chain.analysis.complete)
      << llvm::join(chain.analysis.incompleteReasons, "; ");
  EXPECT_DOUBLE_EQ(chain.analysis.cost.latencyCycles,
                   chain.plan.totalCost.latencyCycles);
  EXPECT_EQ(chain.plan.graphHash,
            mlir::llk::mapping::computeSourceGraphHash(chain.fixture.graph));
}

std::vector<const PlanCostEvent *> eventsOfKind(const ConformanceChain &chain,
                                                CostEventKind kind) {
  std::vector<const PlanCostEvent *> events;
  for (const PlanCostEvent &event : chain.analysis.events.events)
    if (event.event.kind == kind)
      events.push_back(&event);
  return events;
}

/// Reads one memory-keyed total without throwing on a missing model entry: a
/// missing key is a failed conformance fact, not a crashed test.
uint64_t memoryValue(const std::map<std::string, uint64_t> &values,
                     llvm::StringRef memory) {
  auto found = values.find(memory.str());
  EXPECT_NE(found, values.end())
      << "no model entry for memory '" << memory.str() << "'";
  return found == values.end() ? 0 : found->second;
}

unsigned allocationsOf(const CoveringPlan &plan, llvm::StringRef memory,
                       uint64_t bytes) {
  return llvm::count_if(plan.allocations, [&](const StorageAllocation &a) {
    return a.memory == memory && a.bytes == bytes;
  });
}

TEST(FrameworkConformance, TwoComputePreservesSelectedEngineIdentity) {
  auto chain = buildChain("two-compute", 1);
  ASSERT_TRUE(static_cast<bool>(chain)) << llvm::toString(chain.takeError());
  expectStrictParity(*chain);

  ASSERT_EQ(chain->plan.placements.size(), 1u);
  const auto &placement = chain->plan.placements.front();
  EXPECT_EQ(placement.rule, "issue129.vector_add");
  auto engine = placement.computeBindings.find("vector_engine");
  ASSERT_NE(engine, placement.computeBindings.end());
  EXPECT_TRUE(engine->second == "vpu.a" || engine->second == "vpu.b")
      << engine->second;
  EXPECT_EQ(placement.executor, "e0");
  // The rule names no memory: the tiles' own sram kind resolves to the only
  // sram node. Two 256-byte operand views and the 256-byte result are live.
  EXPECT_EQ(memoryValue(chain->analysis.peakBytes, "sram.0"), 768u);

  // With no retention cap both attached engines are enumerated, so the search
  // is selecting a concrete same-kind node rather than collapsing to one.
  auto full = buildChain("two-compute", 0);
  ASSERT_TRUE(static_cast<bool>(full)) << llvm::toString(full.takeError());
  std::set<std::string> engines;
  for (const CoveringPlan &plan : full->search.plans)
    for (const auto &planPlacement : plan.placements)
      engines.insert(planPlacement.computeBindings.lookup("vector_engine"));
  EXPECT_EQ(engines, (std::set<std::string>{"vpu.a", "vpu.b"}));
}

TEST(FrameworkConformance, NamedPortsPreservePerOccurrenceMemorySelections) {
  auto chain = buildChain("named-ports", 1);
  ASSERT_TRUE(static_cast<bool>(chain)) << llvm::toString(chain.takeError());
  expectStrictParity(*chain);

  ASSERT_EQ(chain->plan.placements.size(), 1u);
  const auto &placement = chain->plan.placements.front();
  ASSERT_EQ(placement.portMemoryBindings.size(), 3u);
  std::set<std::string> inputMemories;
  for (const auto &binding : placement.portMemoryBindings) {
    EXPECT_FALSE(binding.memory.empty());
    EXPECT_TRUE(binding.memory == "sram.a" || binding.memory == "sram.b")
        << binding.memory;
    if (binding.port.direction == mlir::llk::mapping::PortDirection::Input)
      inputMemories.insert(binding.memory);
  }

  // Boundary descriptors are borrowed at their own 8x8xf32 span, and the
  // borrowed memories are exactly the input occurrences the rule named.
  std::set<std::string> borrowedMemories;
  for (const StorageAllocation &allocation : chain->plan.allocations)
    if (allocation.borrowed) {
      EXPECT_EQ(allocation.bytes, 256u);
      borrowedMemories.insert(allocation.memory);
    }
  EXPECT_FALSE(borrowedMemories.empty());
  EXPECT_EQ(borrowedMemories, inputMemories);
}

TEST(FrameworkConformance, TwoHopPreservesHopsEnginesPeaksAndTraffic) {
  auto chain = buildChain("two-hop", 1);
  ASSERT_TRUE(static_cast<bool>(chain)) << llvm::toString(chain.takeError());
  expectStrictParity(*chain);

  ASSERT_EQ(chain->plan.connectionPlans.size(), 1u);
  const PlanConnection &movement = chain->plan.connectionPlans.front();
  ASSERT_EQ(movement.route.size(), 3u);
  EXPECT_EQ(movement.route[0], "sram.0");
  EXPECT_EQ(movement.route[1], "l2.0");
  EXPECT_EQ(movement.route[2], "dram.0");
  ASSERT_EQ(movement.hops.size(), 2u);
  EXPECT_EQ(movement.hops[0].srcMemory, "sram.0");
  EXPECT_EQ(movement.hops[0].dstMemory, "l2.0");
  EXPECT_EQ(movement.hops[0].engine, "dma.a");
  EXPECT_EQ(movement.hops[1].srcMemory, "l2.0");
  EXPECT_EQ(movement.hops[1].dstMemory, "dram.0");
  EXPECT_EQ(movement.hops[1].engine, "dma.b");
  EXPECT_EQ(movement.hops[1].sourceStorageId,
            movement.hops[0].destinationStorageId);

  // 256-byte intermediate and landed value.
  EXPECT_EQ(allocationsOf(chain->plan, "l2.0", 256), 1u);
  EXPECT_EQ(allocationsOf(chain->plan, "dram.0", 256), 1u);

  // One 256-byte value crosses two hops and is then written from DRAM to the
  // store's L2 destination: sram is read once (256), l2 is written and read by
  // the route and written once more by the store (768), and dram is written by
  // the route and read by the store (512). Traffic is keyed by memory space;
  // the peaks below are keyed by the concrete node the plan selected.
  EXPECT_EQ(memoryValue(chain->analysis.trafficBytes, "sram"), 256u);
  EXPECT_EQ(memoryValue(chain->analysis.trafficBytes, "l2"), 768u);
  EXPECT_EQ(memoryValue(chain->analysis.trafficBytes, "dram"), 512u);
  EXPECT_EQ(memoryValue(chain->analysis.peakBytes, "l2.0"), 256u);
  EXPECT_EQ(memoryValue(chain->analysis.peakBytes, "dram.0"), 256u);
  EXPECT_GT(eventsOfKind(*chain, CostEventKind::TransferHop).size(), 0u);
}

TEST(FrameworkConformance, PaddedLayoutReservesThePhysicalImage) {
  auto chain = buildChain("padded-layout", 1);
  ASSERT_TRUE(static_cast<bool>(chain)) << llvm::toString(chain.takeError());
  expectStrictParity(*chain);

  // 8x8xf32 with every logical column doubled is an 8x15 physical image: 120
  // elements, 480 bytes -- not the logical 256.
  EXPECT_EQ(memoryValue(chain->analysis.peakBytes, "l2.0"), 480u);
  EXPECT_EQ(allocationsOf(chain->plan, "l2.0", 480), 1u);
}

TEST(FrameworkConformance, InfeasibleCheaperPlanNeverMasksTheFeasibleOne) {
  auto chain = buildChain("capacity-topk", 1);
  ASSERT_TRUE(static_cast<bool>(chain)) << llvm::toString(chain.takeError());
  expectStrictParity(*chain);

  ASSERT_EQ(chain->plan.placements.size(), 1u);
  EXPECT_EQ(chain->plan.placements.front().rule, "r.large");
  EXPECT_GT(chain->search.frontier.plansRejectedByCapacity, 0u);
  // Four simultaneously-live 256-byte result versions (1024) plus the borrowed
  // 256-byte operand descriptor.
  EXPECT_EQ(memoryValue(chain->analysis.peakBytes, "dram.0"), 1280u);

  // Widening K cannot change the best feasible plan: the cheap SRAM covering
  // was rejected before retention, not trimmed after it.
  auto wide = buildChain("capacity-topk", 64);
  ASSERT_TRUE(static_cast<bool>(wide)) << llvm::toString(wide.takeError());
  EXPECT_EQ(wide->plan.id, chain->plan.id);
  EXPECT_EQ(wide->plan.placements.front().rule, "r.large");
}

TEST(FrameworkConformance, LogicalViewControlAddsNoTransformOrMovement) {
  auto chain = buildChain("transform-view", 1);
  ASSERT_TRUE(static_cast<bool>(chain)) << llvm::toString(chain.takeError());
  expectStrictParity(*chain);

  ASSERT_EQ(chain->plan.connectionPlans.size(), 1u);
  const PlanConnection &connection = chain->plan.connectionPlans.front();
  EXPECT_EQ(connection.kind, ConnectionKind::Direct);
  EXPECT_FALSE(connection.transform.has_value());
  EXPECT_TRUE(eventsOfKind(*chain, CostEventKind::Transform).empty());
  // The only transfer is the store's own SRAM -> L2 boundary write; the
  // producer->consumer view moves nothing.
  EXPECT_EQ(eventsOfKind(*chain, CostEventKind::TransferHop).size(), 1u);
}

TEST(FrameworkConformance, AddedLogicalViewPreservesAliasTrafficAndStorage) {
  auto viewed = buildChain("transform-view", 1);
  ASSERT_TRUE(static_cast<bool>(viewed)) << llvm::toString(viewed.takeError());
  unsigned viewCount = 0;
  mlir::Operation *producer = nullptr;
  mlir::Operation *view = nullptr;
  mlir::Operation *store = nullptr;
  viewed->fixture.source->walk([&](mlir::Operation *op) {
    const auto name = op->getName().getStringRef();
    if (name == "micro.vector")
      producer = op;
    if (name == "micro.tile_view") {
      ++viewCount;
      if (op->getOperand(0).getDefiningOp() == producer)
        view = op;
    }
    if (name == "micro.tile_store")
      store = op;
  });
  ASSERT_EQ(viewCount, 2u) << "the control needs a real producer-result view";
  ASSERT_NE(view, nullptr);
  ASSERT_NE(store, nullptr);
  EXPECT_EQ(store->getOperand(0), view->getResult(0));

  auto direct = buildChain("transform-direct", 1);
  ASSERT_TRUE(static_cast<bool>(direct)) << llvm::toString(direct.takeError());
  expectStrictParity(*viewed);
  expectStrictParity(*direct);

  // A view adds neither a workload node nor another value/storage version:
  // the store still reads the vector's result, so its data dependency survives.
  const auto &graph = viewed->fixture.graph;
  ASSERT_EQ(graph.getNodes().size(), 2u);
  const mlir::llk::mapping::WorkloadNode *vectorNode = nullptr;
  const mlir::llk::mapping::WorkloadNode *storeNode = nullptr;
  for (const auto &node : graph.getNodes()) {
    if (node.opName == "micro.vector")
      vectorNode = &node;
    if (node.opName == "micro.tile_store")
      storeNode = &node;
  }
  ASSERT_NE(vectorNode, nullptr);
  ASSERT_NE(storeNode, nullptr);
  ASSERT_EQ(vectorNode->outputs.size(), 1u);
  ASSERT_EQ(storeNode->inputs.size(), 1u);
  EXPECT_EQ(storeNode->inputs.front().value, vectorNode->outputs.front().value);
  EXPECT_EQ(graph.getValues().size(), direct->fixture.graph.getValues().size());

  // Only the store moves the 8x8xf32 result: 256 bytes read from SRAM and
  // written to L2. The 256-byte input and result share a 512-byte live peak.
  EXPECT_EQ(memoryValue(viewed->analysis.trafficBytes, "sram"), 256u);
  EXPECT_EQ(memoryValue(viewed->analysis.trafficBytes, "l2"), 256u);
  EXPECT_EQ(memoryValue(viewed->analysis.peakBytes, "sram.0"), 512u);
  EXPECT_EQ(viewed->analysis.trafficBytes, direct->analysis.trafficBytes);
  EXPECT_EQ(viewed->analysis.peakBytes, direct->analysis.peakBytes);
  EXPECT_EQ(viewed->plan.allocations.size(), direct->plan.allocations.size());
  EXPECT_EQ(allocationsOf(viewed->plan, "sram.0", 256),
            allocationsOf(direct->plan, "sram.0", 256));
  EXPECT_EQ(viewed->analysis.events.events.size(),
            direct->analysis.events.events.size());
  EXPECT_TRUE(eventsOfKind(*viewed, CostEventKind::Transform).empty());
  const auto computes = eventsOfKind(*viewed, CostEventKind::Compute);
  const auto transfers = eventsOfKind(*viewed, CostEventKind::TransferHop);
  ASSERT_EQ(computes.size(), 1u);
  ASSERT_EQ(transfers.size(), 1u);
  const auto computeIndex = static_cast<uint32_t>(
      computes.front() - viewed->analysis.events.events.data());
  EXPECT_TRUE(llvm::is_contained(transfers.front()->deps, computeIndex));
  EXPECT_DOUBLE_EQ(viewed->analysis.cost.latencyCycles,
                   direct->analysis.cost.latencyCycles);
}

TEST(FrameworkConformance, PhysicalTransformAddsCalculatedWorkAndStorage) {
  auto control = buildChain("transform-view", 1);
  ASSERT_TRUE(static_cast<bool>(control))
      << llvm::toString(control.takeError());
  auto physical = buildChain("transform-layout", 1);
  ASSERT_TRUE(static_cast<bool>(physical))
      << llvm::toString(physical.takeError());
  expectStrictParity(*control);
  expectStrictParity(*physical);

  // Binding clones: the original source module is never annotated.
  bool annotated = false;
  physical->fixture.source->walk([&](mlir::Operation *op) {
    annotated |= op->hasAttr("micro.mapping") || op->hasAttr("micro.plan");
  });
  EXPECT_FALSE(annotated) << "binding must clone the source, not mutate it";

  ASSERT_EQ(physical->plan.connectionPlans.size(), 1u);
  const PlanConnection &connection = physical->plan.connectionPlans.front();
  ASSERT_TRUE(connection.transform.has_value());
  EXPECT_EQ(connection.transform->srcLayout, "issue129.row_major_layout");
  EXPECT_EQ(connection.transform->dstLayout, "issue129.transposed_layout");

  // The 8x8xf32 tile is 64 elements / 8 f32 lanes = 8 issues on the one vector
  // engine, and 64 x 4 = 256 bytes.
  std::vector<const PlanCostEvent *> transforms =
      eventsOfKind(*physical, CostEventKind::Transform);
  ASSERT_EQ(transforms.size(), 1u);
  const PlanCostEvent &transform = *transforms.front();
  EXPECT_EQ(transform.event.resource, "vpu.0");
  EXPECT_EQ(transform.workItems, 64u);
  EXPECT_EQ(transform.bytes, 256u);
  EXPECT_EQ(transform.event.cost.localBytes, 256u);
  EXPECT_DOUBLE_EQ(transform.event.cost.latencyCycles, 8.0);

  // The conversion reserves its own post-transform 256-byte image; the direct
  // control does not.
  EXPECT_EQ(allocationsOf(physical->plan, "sram.0", 256),
            allocationsOf(control->plan, "sram.0", 256) + 1);
  EXPECT_DOUBLE_EQ(physical->analysis.cost.latencyCycles,
                   control->analysis.cost.latencyCycles + 8.0);
  EXPECT_EQ(physical->analysis.cost.localBytes,
            control->analysis.cost.localBytes + 256u);
}

//===----------------------------------------------------------------------===//
// C3: composed vectors and reduction on both target profiles
//===----------------------------------------------------------------------===//

enum class Profile { AVX2, GenericAccelerator };

llvm::StringRef profileName(Profile profile) {
  return profile == Profile::AVX2 ? "x86-avx2" : "generic-ai-accel";
}

mlir::DialectRegistry frameworkRegistry() {
  mlir::DialectRegistry registry;
  registry.insert<mlir::llk::LLKDialect, mlir::micro::MicroDialect,
                  mlir::func::FuncDialect, mlir::linalg::LinalgDialect,
                  mlir::tensor::TensorDialect, mlir::scf::SCFDialect,
                  mlir::arith::ArithDialect, mlir::memref::MemRefDialect>();
  return registry;
}

std::unique_ptr<MappingTarget> createProfile(Profile profile) {
  llvm::Expected<std::unique_ptr<MappingTarget>> target =
      profile == Profile::AVX2
          ? mlir::llk::target::avx2::createMappingTarget(LLK_SOURCE_DIR)
          : mlir::llk::target::generic_accel::createMappingTarget(
                LLK_SOURCE_DIR);
  if (!target) {
    ADD_FAILURE() << llvm::toString(target.takeError());
    return nullptr;
  }
  return std::move(*target);
}

mlir::OwningOpRef<mlir::ModuleOp> parseFramework(mlir::MLIRContext &context,
                                                 llvm::StringRef name) {
  std::string path = std::string(LLK_SOURCE_DIR) +
                     "/test/Execution/Inputs/framework/" + name.str();
  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> input =
      llvm::MemoryBuffer::getFile(path);
  if (!input)
    return {};
  return mlir::parseSourceString<mlir::ModuleOp>((*input)->getBuffer(),
                                                 &context);
}

std::string printModule(mlir::ModuleOp module) {
  std::string text;
  llvm::raw_string_ostream stream(text);
  module.print(stream);
  stream.flush();
  return text;
}

llvm::Expected<MappingSearchResult>
searchWithEvaluator(mlir::ModuleOp module,
                    const mlir::llk::mapping::WorkloadGraph &graph,
                    MappingTarget &target) {
  MappingSearchOptions options;
  options.mode = SearchMode::Exact;
  options.topK = 1;
  options.requireCompleteEvaluation = true;
  options.evaluateCompletePlan = [&](const CoveringPlan &proposal) {
    return mlir::llk::mapping::evaluateCompletePlan(
        module, graph, target, proposal,
        mlir::llk::mapping::BindContract::Partial, std::nullopt,
        mlir::llk::mapping::AnalysisRequirement::Complete);
  };
  mlir::llk::mapping::LayoutContext layoutContext;
  layoutContext.rank = 2;
  layoutContext.elementType = "f32";
  mlir::llk::mapping::CoveringSearch search(graph, target, *module.getContext(),
                                            layoutContext, options);
  return search.search();
}

/// One strict chain over a framework primitive fixture and a *real* target
/// package: the searched plan, the kernel it bound, and the shared analysis.
struct PackageChain {
  mlir::llk::mapping::WorkloadGraph graph;
  MappingSearchResult search;
  CoveringPlan plan;
  mlir::OwningOpRef<mlir::ModuleOp> boundModule;
  mlir::Operation *kernel = nullptr;
  mlir::llk::perf::SelectedKernelAnalysis analysis;
};

llvm::Expected<PackageChain> buildPackageChain(mlir::ModuleOp module,
                                               MappingTarget &target,
                                               llvm::StringRef symbol) {
  llvm::Expected<mlir::Operation *> kernel =
      mlir::llk::perf::findMicroKernel(module, symbol);
  if (!kernel)
    return kernel.takeError();
  llvm::Expected<mlir::llk::mapping::WorkloadGraph> graph =
      mlir::llk::mapping::extractWorkloadGraph(*kernel);
  if (!graph)
    return graph.takeError();
  auto search = searchWithEvaluator(module, *graph, target);
  if (!search)
    return search.takeError();
  if (search->plans.empty()) {
    std::string reason = "no complete plan for '" + symbol.str() + "'";
    if (!search->frontier.diagnostics.empty())
      reason += ": " + search->frontier.diagnostics.front().message;
    return llvm::createStringError(llvm::inconvertibleErrorCode(), reason);
  }

  PackageChain chain;
  chain.graph = std::move(*graph);
  chain.search = std::move(*search);
  chain.plan = chain.search.plans.front();

  llk::MappedCompileOptions compile;
  compile.stop = llk::MappedStop::MappedMicro;
  compile.requireExecutable = false;
  auto bound = llk::compileMappedKernel(module, target, chain.plan, compile);
  if (!bound)
    return bound.takeError();
  chain.boundModule = std::move(bound->module);
  auto boundKernel =
      mlir::llk::perf::findMicroKernel(chain.boundModule.get(), symbol);
  if (!boundKernel)
    return boundKernel.takeError();
  chain.kernel = *boundKernel;
  auto analysis = mlir::llk::perf::analyzeSelectedKernel(
      chain.kernel, target.machine(), /*requireComplete=*/true, chain.plan,
      chain.graph);
  if (!analysis)
    return analysis.takeError();
  chain.analysis = std::move(*analysis);
  return chain;
}

TEST(FrameworkConformance, FrameworkPrimitivesMapStaticallyOnBothProfiles) {
  struct FixtureCase {
    const char *file;
    const char *symbol;
    size_t nodes;
  };
  const FixtureCase fixtures[] = {{"reduce.mlir", "reduce_sum", 1},
                                  {"add_mul.mlir", "add_mul", 2},
                                  {"add_mul_reduce.mlir", "add_mul_reduce", 3}};

  for (Profile profile : {Profile::AVX2, Profile::GenericAccelerator})
    for (const FixtureCase &fixture : fixtures) {
      SCOPED_TRACE(profileName(profile).str() + "/" + fixture.file);
      mlir::MLIRContext context(frameworkRegistry());
      context.loadAllAvailableDialects();
      auto module = parseFramework(context, fixture.file);
      ASSERT_TRUE(module);
      ASSERT_TRUE(mlir::succeeded(mlir::verify(*module)));
      auto target = createProfile(profile);
      ASSERT_NE(target, nullptr);

      auto chain = buildPackageChain(*module, *target, fixture.symbol);
      ASSERT_TRUE(static_cast<bool>(chain))
          << llvm::toString(chain.takeError());
      EXPECT_EQ(chain->graph.getNodes().size(), fixture.nodes);
      ASSERT_EQ(chain->plan.placements.size(), fixture.nodes);
      for (const auto &placement : chain->plan.placements) {
        EXPECT_FALSE(placement.rule.empty());
        EXPECT_GT(placement.workItems, 0u);
      }
      EXPECT_TRUE(chain->analysis.complete);
      EXPECT_DOUBLE_EQ(chain->analysis.cost.latencyCycles,
                       chain->plan.totalCost.latencyCycles);
      EXPECT_TRUE(chain->plan.diagnostics.physicalComplete);
    }
}

TEST(FrameworkConformance, UnsupportedReductionVariantIsExplicitlyRejected) {
  constexpr llvm::StringLiteral kMaxReduce = R"mlir(
module {
  micro.kernel @reduce_max(%input: tensor<8x8xf32>) -> tensor<8xf32> {
    %tile = micro.tile_view %input {shape = array<i64: 8, 8>}
        : tensor<8x8xf32> -> !micro.tile<8x8xf32, memory = #micro.memory<sram>>
    %max = micro.reduce "max" %tile {axis = 1 : i64}
        : !micro.tile<8x8xf32, memory = #micro.memory<sram>>
       -> !micro.tile<8xf32, memory = #micro.memory<acc>>
    micro.yield %max : !micro.tile<8xf32, memory = #micro.memory<acc>>
  }
}
)mlir";

  for (Profile profile : {Profile::AVX2, Profile::GenericAccelerator})
    for (llvm::StringRef variant : {"max", "product"}) {
      SCOPED_TRACE(profileName(profile).str() + "/" + variant.str());
      std::string source = kMaxReduce.str();
      source.replace(source.find("\"max\""), 5, "\"" + variant.str() + "\"");
      mlir::MLIRContext context(frameworkRegistry());
      context.loadAllAvailableDialects();
      auto module = mlir::parseSourceString<mlir::ModuleOp>(source, &context);
      ASSERT_TRUE(module);
      ASSERT_TRUE(mlir::succeeded(mlir::verify(*module)));
      auto target = createProfile(profile);
      ASSERT_NE(target, nullptr);
      auto kernel = mlir::llk::perf::findMicroKernel(*module, "reduce_max");
      ASSERT_TRUE(static_cast<bool>(kernel))
          << llvm::toString(kernel.takeError());
      auto graph = mlir::llk::mapping::extractWorkloadGraph(*kernel);
      ASSERT_TRUE(static_cast<bool>(graph))
          << llvm::toString(graph.takeError());
      auto search = searchWithEvaluator(*module, *graph, *target);
      ASSERT_TRUE(static_cast<bool>(search))
          << llvm::toString(search.takeError());
      EXPECT_TRUE(search->plans.empty());
      EXPECT_TRUE(search->frontier.has(
          mlir::llk::mapping::DiagnosticCode::NoMatchingRule));
    }
}

TEST(FrameworkConformance, AcceleratorRejectsNonF32CoverageBeforeRanking) {
  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> input =
      llvm::MemoryBuffer::getFile(
          std::string(LLK_SOURCE_DIR) +
          "/test/Execution/Inputs/framework/add_mul.mlir");
  ASSERT_TRUE(static_cast<bool>(input)) << input.getError().message();
  const std::string source = (*input)->getBuffer().str();

  std::string bf16 = source;
  for (size_t at = bf16.find("f32"); at != std::string::npos;
       at = bf16.find("f32")) {
    bf16.replace(at, 3, "bf16");
  }

  mlir::MLIRContext context(frameworkRegistry());
  context.loadAllAvailableDialects();
  auto module = mlir::parseSourceString<mlir::ModuleOp>(bf16, &context);
  ASSERT_TRUE(module);
  ASSERT_TRUE(mlir::succeeded(mlir::verify(*module)));
  const std::string before = printModule(*module);

  auto target = createProfile(Profile::GenericAccelerator);
  ASSERT_NE(target, nullptr);
  auto kernel = mlir::llk::perf::findMicroKernel(*module, "add_mul");
  ASSERT_TRUE(static_cast<bool>(kernel)) << llvm::toString(kernel.takeError());
  auto graph = mlir::llk::mapping::extractWorkloadGraph(*kernel);
  ASSERT_TRUE(static_cast<bool>(graph)) << llvm::toString(graph.takeError());
  auto search = searchWithEvaluator(*module, *graph, *target);
  ASSERT_TRUE(static_cast<bool>(search)) << llvm::toString(search.takeError());

  EXPECT_TRUE(search->plans.empty());
  bool coverageFailure = false;
  for (const auto &diagnostic : search->frontier.diagnostics)
    coverageFailure |=
        diagnostic.code ==
            mlir::llk::mapping::DiagnosticCode::UnsupportedMaterialization &&
        (diagnostic.message.find("target_coverage") != std::string::npos ||
         diagnostic.message.find("f32") != std::string::npos);
  EXPECT_TRUE(coverageFailure)
      << (search->frontier.diagnostics.empty()
              ? "no diagnostic"
              : search->frontier.diagnostics.front().message);
  EXPECT_EQ(printModule(*module), before)
      << "coverage validation must not mutate the source IR";

  // f64 is outside the Micro tile type system: the dialect refuses it before
  // any target can consider coverage.
  std::string f64 = source;
  for (size_t at = f64.find("f32"); at != std::string::npos;
       at = f64.find("f32")) {
    f64.replace(at, 3, "f64");
  }
  auto f64Module = mlir::parseSourceString<mlir::ModuleOp>(f64, &context);
  if (f64Module)
    EXPECT_TRUE(mlir::failed(mlir::verify(*f64Module)))
        << "f64 tiles must never become a rankable plan";
}

TEST(FrameworkConformance,
     CompilationRejectsUnsupportedExplicitAndReplayedOperations) {
  for (llvm::StringRef fixture : {"reduce.mlir", "add_mul.mlir"}) {
    SCOPED_TRACE(fixture.str());
    auto input = llvm::MemoryBuffer::getFile(
        std::string(LLK_SOURCE_DIR) + "/test/Execution/Inputs/framework/" +
        fixture.str());
    ASSERT_TRUE(static_cast<bool>(input)) << input.getError().message();
    std::string source = (*input)->getBuffer().str();
    for (size_t at = source.find("f32"); at != std::string::npos;
         at = source.find("f32"))
      source.replace(at, 3, "bf16");

    mlir::MLIRContext context(frameworkRegistry());
    context.loadAllAvailableDialects();
    auto module = mlir::parseSourceString<mlir::ModuleOp>(source, &context);
    ASSERT_TRUE(module);
    ASSERT_TRUE(mlir::succeeded(mlir::verify(*module)));
    const std::string before = printModule(*module);
    auto target = createProfile(Profile::GenericAccelerator);
    ASSERT_NE(target, nullptr);
    auto kernel = mlir::llk::perf::findMicroKernel(*module, "");
    ASSERT_TRUE(static_cast<bool>(kernel))
        << llvm::toString(kernel.takeError());
    auto graph = mlir::llk::mapping::extractWorkloadGraph(*kernel);
    ASSERT_TRUE(static_cast<bool>(graph)) << llvm::toString(graph.takeError());

    // A caller can supply a plan without having run complete evaluation. The
    // compilation boundary must still enforce the target's f32-only contract.
    MappingSearchOptions searchOptions;
    searchOptions.mode = SearchMode::Exact;
    searchOptions.topK = 1;
    mlir::llk::mapping::LayoutContext layoutContext;
    layoutContext.rank = 2;
    layoutContext.elementType = "bf16";
    mlir::llk::mapping::CoveringSearch search(*graph, *target, context,
                                              layoutContext, searchOptions);
    auto searched = search.search();
    ASSERT_TRUE(static_cast<bool>(searched))
        << llvm::toString(searched.takeError());
    ASSERT_EQ(searched->plans.size(), 1u);
    const std::string report = mlir::llk::mapping::writePlanReport(
        *searched, target->machine(), *target, searchOptions, 0);
    auto replayed = mlir::llk::mapping::readPlanReport(report, *target, *graph);
    ASSERT_TRUE(static_cast<bool>(replayed))
        << llvm::toString(replayed.takeError());

    for (const auto *plan : {&searched->plans.front(), &*replayed})
      for (llk::MappedStop stop :
           {llk::MappedStop::MappedMicro, llk::MappedStop::Lowered}) {
        SCOPED_TRACE(plan == &*replayed ? "replayed" : "explicit");
        SCOPED_TRACE(static_cast<int>(stop));
        llk::MappedCompileOptions options;
        options.backend = llk::MappedBackend::Reference;
        options.stop = stop;
        options.requireExecutable = false;
        auto compiled =
            llk::compileMappedKernel(*module, *target, *plan, options);
        EXPECT_FALSE(static_cast<bool>(compiled));
        if (!compiled) {
          const std::string error = llvm::toString(compiled.takeError());
          EXPECT_NE(error.find("target_coverage:"), std::string::npos) << error;
          EXPECT_NE(error.find("f32"), std::string::npos) << error;
        }
        EXPECT_EQ(printModule(*module), before)
            << "rejected compilation must leave the caller's source unchanged";
      }
  }
}

} // namespace
