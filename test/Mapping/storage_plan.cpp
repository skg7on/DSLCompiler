//===- storage_plan.cpp - Physical storage footprints and value lifetimes
//--===//
//
// Task B3 (issue #67, stage B). Exercises the two storage-planning contracts:
// `computePeakStorage` summarizes a validated deterministic schedule's live
// ranges, and `finalizeStoragePlan` builds the dependency-aware intervals and
// checks per-memory capacity against real occupancy.

#include "LLK/Mapping/StoragePlan.h"

#include "LLK/Mapping/CoveringSearch.h"
#include "LLK/Mapping/MappingMetadata.h"
#include "LLK/Mapping/MappingPlan.h"
#include "LLK/Mapping/MappingTarget.h"
#include "LLK/Mapping/PlanReport.h"
#include "LLK/Mapping/TileFacts.h"

#include "LLK/Dialect/Micro/MicroDialect.h"

#include "mlir/AsmParser/AsmParser.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"

#include "llvm/Support/Error.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <vector>

using namespace mlir::llk::machine;
using namespace mlir::llk::mapping;

namespace {

MachineModel storageMachine() {
  MachineModel model;
  model.target = "storage";
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

mlir::Type tileType(mlir::MLIRContext &context, llvm::StringRef inner) {
  context.getOrLoadDialect<mlir::micro::MicroDialect>();
  std::string text = "!micro.tile<" + inner.str() + ">";
  return mlir::parseType(text, &context);
}

/// A three-node chain `in -> n0 -> v0 -> n1 -> v1 -> n2 -> out`, every node a
/// `micro.vector` over an 8x8xf32 tile. When `multiplicity` is set each node
/// carries it, so the strict planner has a known execution count; unset means
/// the count is unknown.
WorkloadGraph chainGraph(mlir::MLIRContext &context, mlir::Type tile,
                         std::optional<uint64_t> multiplicity) {
  WorkloadGraph graph;
  WorkloadValueId in =
      graph.addValue(WorkloadValue{0, tile, "in", /*external=*/true});
  WorkloadValueId v0 =
      graph.addValue(WorkloadValue{0, tile, "v0", /*external=*/false});
  WorkloadValueId v1 =
      graph.addValue(WorkloadValue{0, tile, "v1", /*external=*/false});
  WorkloadValueId out =
      graph.addValue(WorkloadValue{0, tile, "out", /*external=*/false});

  auto node = [&](unsigned ordinal, WorkloadValueId input,
                  WorkloadValueId output) {
    WorkloadNode n;
    n.opName = "micro.vector";
    n.sourceOrdinal = ordinal;
    n.executionMultiplicity = multiplicity;
    n.inputs.push_back(WorkloadPort{input, tile, std::nullopt});
    n.outputs.push_back(WorkloadPort{output, tile, std::nullopt});
    graph.addNode(std::move(n));
  };
  node(0, in, v0);
  node(1, v0, v1);
  node(2, v1, out);
  graph.finalize();
  return graph;
}

/// One `micro.vector` rule that binds every node to `sram`, so a chain of three
/// 256-byte values lives entirely in one memory.
constexpr llvm::StringLiteral kStorageRules = R"llkmap(
rule r.vec {
  match micro.vector();
  require executor kind worker;
  require memory kind sram;
  bundle "b.vec";
  emit "e1";
  cost 1;
}
)llkmap";

std::unique_ptr<MappingTarget> storageTarget(MachineModel machine) {
  llvm::Expected<RuleRegistry> registry =
      parseRuleText(kStorageRules, "<test>");
  if (!registry)
    return nullptr;
  return std::make_unique<FileMappingTarget>(
      "test", std::move(machine), LayoutRegistry{}, std::move(*registry),
      std::vector<std::string>{"e1"});
}

/// Runs a deterministic search and returns its best plan. `ASSERT_*` in the
/// caller guards the empty result.
std::optional<CoveringPlan> searchOne(const WorkloadGraph &graph,
                                      const MappingTarget &target,
                                      mlir::MLIRContext &context) {
  MappingSearchOptions options;
  options.mode = SearchMode::Deterministic;
  CoveringSearch search(graph, target, context, LayoutContext{}, options);
  llvm::Expected<MappingSearchResult> result = search.search();
  if (!result || result->plans.empty())
    return std::nullopt;
  return result->plans.front();
}

StorageAllocation alloc(uint64_t id, llvm::StringRef memory, uint64_t bytes,
                        PlanStepId begin, PlanStepId end) {
  StorageAllocation a;
  a.id = id;
  a.memory = memory.str();
  a.bytes = bytes;
  a.beginStep = begin;
  a.endStep = end;
  return a;
}

} // namespace

//===----------------------------------------------------------------------===//
// computePeakStorage
//===----------------------------------------------------------------------===//

TEST(StoragePlan, PreservesSourceWhileBothTransformsAreLive) {
  StorageAllocation a;
  a.id = 1;
  a.memory = "sram.0";
  a.bytes = 256;
  a.beginStep = 0;
  a.endStep = 4;
  StorageAllocation b;
  b.id = 2;
  b.memory = "sram.0";
  b.bytes = 256;
  b.beginStep = 1;
  b.endStep = 3;
  StorageAllocation c;
  c.id = 3;
  c.memory = "sram.0";
  c.bytes = 256;
  c.beginStep = 2;
  c.endStep = 4;
  auto peak = computePeakStorage({a, b, c});
  ASSERT_TRUE(bool(peak)) << llvm::toString(peak.takeError());
  EXPECT_EQ((*peak)["sram.0"], 768u);
}

TEST(StoragePlan, SequentialReuseFitsOneBuffer) {
  // The second allocation begins only after the first expires, so the two
  // share one buffer instead of summing.
  auto peak = computePeakStorage(
      {alloc(1, "sram.0", 256, 0, 1), alloc(2, "sram.0", 256, 2, 3)});
  ASSERT_TRUE(bool(peak)) << llvm::toString(peak.takeError());
  EXPECT_EQ((*peak)["sram.0"], 256u);
}

TEST(StoragePlan, OverlappingConsumersAccumulate) {
  auto peak = computePeakStorage(
      {alloc(1, "sram.0", 256, 0, 3), alloc(2, "sram.0", 256, 1, 2)});
  ASSERT_TRUE(bool(peak)) << llvm::toString(peak.takeError());
  EXPECT_EQ((*peak)["sram.0"], 512u);
}

TEST(StoragePlan, IntermediateStagingIsCountedInItsOwnMemory) {
  auto peak = computePeakStorage({alloc(1, "sram.0", 256, 0, 2),
                                  alloc(2, "sram.0", 128, 1, 4),
                                  alloc(3, "dram.0", 1024, 0, 4)});
  ASSERT_TRUE(bool(peak)) << llvm::toString(peak.takeError());
  EXPECT_EQ((*peak)["sram.0"], 384u);
  EXPECT_EQ((*peak)["dram.0"], 1024u);
}

TEST(StoragePlan, AnUnusedOutputIsReleasedAtItsProducer) {
  auto peak = computePeakStorage(
      {alloc(1, "sram.0", 256, 0, 0), alloc(2, "sram.0", 256, 1, 4)});
  ASSERT_TRUE(bool(peak)) << llvm::toString(peak.takeError());
  EXPECT_EQ((*peak)["sram.0"], 256u);
}

TEST(StoragePlan, AProvenAliasSharesStorage) {
  std::vector<StorageAllocation> allocations = {alloc(1, "sram.0", 256, 0, 5),
                                                alloc(2, "sram.0", 256, 2, 2)};
  allocations[1].aliasOf = 1;
  auto peak = computePeakStorage(allocations);
  ASSERT_TRUE(bool(peak)) << llvm::toString(peak.takeError());
  EXPECT_EQ((*peak)["sram.0"], 256u);
}

TEST(StoragePlan, RejectsAnIncompatibleAlias) {
  std::vector<StorageAllocation> allocations = {alloc(1, "sram.0", 256, 0, 5),
                                                alloc(2, "sram.0", 512, 2, 2)};
  allocations[1].aliasOf = 1;
  auto peak = computePeakStorage(allocations);
  ASSERT_FALSE(bool(peak));
  EXPECT_NE(llvm::toString(peak.takeError()).find("alias"), std::string::npos);
}

TEST(StoragePlan, RejectsAnAliasIntoAnotherMemory) {
  std::vector<StorageAllocation> allocations = {alloc(1, "sram.0", 256, 0, 5),
                                                alloc(2, "dram.0", 256, 2, 2)};
  allocations[1].aliasOf = 1;
  auto peak = computePeakStorage(allocations);
  ASSERT_FALSE(bool(peak));
}

TEST(StoragePlan, RejectsArithmeticOverflow) {
  const uint64_t huge = std::numeric_limits<uint64_t>::max();
  auto peak = computePeakStorage(
      {alloc(1, "sram.0", huge, 0, 3), alloc(2, "sram.0", huge, 1, 2)});
  ASSERT_FALSE(bool(peak));
  EXPECT_NE(llvm::toString(peak.takeError()).find("overflow"),
            std::string::npos);
}

TEST(StoragePlan, RejectsADuplicateAllocationId) {
  auto peak = computePeakStorage(
      {alloc(1, "sram.0", 256, 0, 1), alloc(1, "sram.0", 256, 2, 3)});
  ASSERT_FALSE(bool(peak));
}

//===----------------------------------------------------------------------===//
// finalizeStoragePlan
//===----------------------------------------------------------------------===//

TEST(StoragePlan, BuildsAllocationsAndOccupancyForASelectedPlan) {
  mlir::MLIRContext context;
  WorkloadGraph graph = chainGraph(context, tileType(context, "8x8xf32"), 1);
  std::unique_ptr<MappingTarget> target = storageTarget(storageMachine());
  ASSERT_NE(target, nullptr);
  std::optional<CoveringPlan> plan = searchOne(graph, *target, context);
  ASSERT_TRUE(plan.has_value());

  llvm::Error error = finalizeStoragePlan(graph, *plan, storageMachine());
  ASSERT_FALSE(bool(error)) << llvm::toString(std::move(error));

  // v0, v1 and out each occupy one 256-byte buffer in sram.0.
  ASSERT_EQ(plan->allocations.size(), 3u);
  for (const StorageAllocation &allocation : plan->allocations) {
    EXPECT_EQ(allocation.memory, "sram.0");
    EXPECT_EQ(allocation.bytes, 256u);
  }
  // The chain's live ranges overlap only across a consumer step, so the peak is
  // two buffers, not three.
  auto peak = computePeakStorage(plan->allocations);
  ASSERT_TRUE(bool(peak)) << llvm::toString(peak.takeError());
  EXPECT_EQ((*peak)["sram.0"], 512u);
  // Ordered producers: v0 begins at n0, v1 at n1, out at n2, so their begin
  // steps strictly increase.
  EXPECT_LT(plan->allocations[0].beginStep, plan->allocations[1].beginStep);
  EXPECT_LT(plan->allocations[1].beginStep, plan->allocations[2].beginStep);
  // The occupancy is recorded in the report, as an informational note that does
  // not enter the plan's identity.
  bool noted = false;
  for (const std::string &note : plan->diagnostics.storageNotes)
    noted |= note.find("sram.0") != std::string::npos;
  EXPECT_TRUE(noted);
  EXPECT_TRUE(plan->diagnostics.warnings.empty());
}

TEST(StoragePlan, ReleasesAProducerAllocationAfterItsLastReader) {
  mlir::MLIRContext context;
  WorkloadGraph graph = chainGraph(context, tileType(context, "8x8xf32"), 1);
  std::unique_ptr<MappingTarget> target = storageTarget(storageMachine());
  ASSERT_NE(target, nullptr);
  std::optional<CoveringPlan> plan = searchOne(graph, *target, context);
  ASSERT_TRUE(plan.has_value());
  ASSERT_FALSE(bool(finalizeStoragePlan(graph, *plan, storageMachine())));

  // The last value in the chain has no consumer, so it is released at its own
  // producer step.
  const StorageAllocation &last = plan->allocations.back();
  EXPECT_EQ(last.beginStep, last.endStep);
}

TEST(StoragePlan, PipelineMultiplicityMultipliesTheFootprint) {
  mlir::MLIRContext context;
  WorkloadGraph graph = chainGraph(context, tileType(context, "8x8xf32"), 4);
  std::unique_ptr<MappingTarget> target = storageTarget(storageMachine());
  ASSERT_NE(target, nullptr);
  std::optional<CoveringPlan> plan = searchOne(graph, *target, context);
  ASSERT_TRUE(plan.has_value());
  ASSERT_FALSE(bool(finalizeStoragePlan(graph, *plan, storageMachine())));
  for (const StorageAllocation &allocation : plan->allocations)
    EXPECT_EQ(allocation.bytes, 1024u);
}

TEST(StoragePlan, RejectsAPlanWhoseLiveRangeExceedsCapacity) {
  mlir::MLIRContext context;
  WorkloadGraph graph = chainGraph(context, tileType(context, "8x8xf32"), 1);
  std::unique_ptr<MappingTarget> target = storageTarget(storageMachine());
  ASSERT_NE(target, nullptr);
  std::optional<CoveringPlan> plan = searchOne(graph, *target, context);
  ASSERT_TRUE(plan.has_value());

  // The plan fits the search's large sram; a memory 300 bytes large does not
  // hold the 512-byte live peak, so finalization rejects it rather than
  // silently accepting a plan that cannot run.
  MachineModel tight = storageMachine();
  for (MemoryNode &memory : tight.memories)
    if (memory.kind == "sram")
      memory.capacityBytes = 300;
  llvm::Error error = finalizeStoragePlan(graph, *plan, tight);
  ASSERT_TRUE(bool(error));
  const std::string message = llvm::toString(std::move(error));
  EXPECT_NE(message.find("capacity"), std::string::npos);
  EXPECT_NE(message.find("sram.0"), std::string::npos);
}

TEST(StoragePlan, RejectsUnknownExecutionMultiplicityInStrictPlanning) {
  mlir::MLIRContext context;
  WorkloadGraph graph =
      chainGraph(context, tileType(context, "8x8xf32"), std::nullopt);
  std::unique_ptr<MappingTarget> target = storageTarget(storageMachine());
  ASSERT_NE(target, nullptr);
  std::optional<CoveringPlan> plan = searchOne(graph, *target, context);
  ASSERT_TRUE(plan.has_value());
  ASSERT_TRUE(plan->materialized); // strict executable planning

  llvm::Error error = finalizeStoragePlan(graph, *plan, storageMachine());
  ASSERT_TRUE(bool(error));
  EXPECT_NE(llvm::toString(std::move(error)).find("multiplicity"),
            std::string::npos);
}

TEST(StoragePlan, AnalysisModeReportsUnknownMultiplicityAndFallsBack) {
  mlir::MLIRContext context;
  WorkloadGraph graph =
      chainGraph(context, tileType(context, "8x8xf32"), std::nullopt);
  std::unique_ptr<MappingTarget> target = storageTarget(storageMachine());
  ASSERT_NE(target, nullptr);
  std::optional<CoveringPlan> plan = searchOne(graph, *target, context);
  ASSERT_TRUE(plan.has_value());
  // A partial (non-materialized) plan is explicit analysis: the conservative
  // single-iteration fallback stands, and the unknown multiplicity is reported.
  plan->materialized = false;

  llvm::Error error = finalizeStoragePlan(graph, *plan, storageMachine());
  ASSERT_FALSE(bool(error)) << llvm::toString(std::move(error));
  bool reported = false;
  for (const std::string &note : plan->diagnostics.storageNotes)
    reported |= note.find("multiplicity") != std::string::npos;
  EXPECT_TRUE(reported);
}

TEST(StoragePlan, RecordsStorageIdsOnTheConnectionsThatCarryAValue) {
  mlir::MLIRContext context;
  WorkloadGraph graph = chainGraph(context, tileType(context, "8x8xf32"), 1);
  std::unique_ptr<MappingTarget> target = storageTarget(storageMachine());
  ASSERT_NE(target, nullptr);
  std::optional<CoveringPlan> plan = searchOne(graph, *target, context);
  ASSERT_TRUE(plan.has_value());
  ASSERT_FALSE(bool(finalizeStoragePlan(graph, *plan, storageMachine())));

  ASSERT_FALSE(plan->connectionPlans.empty());
  bool anyStorage = false;
  for (const PlanConnection &connection : plan->connectionPlans)
    anyStorage |= !connection.storageIds.empty();
  EXPECT_TRUE(anyStorage);
}

TEST(StoragePlan, RejectsAPlanThatDoesNotCoverEveryNode) {
  mlir::MLIRContext context;
  WorkloadGraph graph = chainGraph(context, tileType(context, "8x8xf32"), 1);
  std::unique_ptr<MappingTarget> target = storageTarget(storageMachine());
  ASSERT_NE(target, nullptr);
  std::optional<CoveringPlan> plan = searchOne(graph, *target, context);
  ASSERT_TRUE(plan.has_value());
  plan->placements.clear();
  llvm::Error error = finalizeStoragePlan(graph, *plan, storageMachine());
  EXPECT_TRUE(bool(error));
}

//===----------------------------------------------------------------------===//
// Extraction recovers the execution multiplicity (fix round 1)
//===----------------------------------------------------------------------===//

namespace {

struct ParsedKernel {
  std::unique_ptr<mlir::MLIRContext> context;
  mlir::OwningOpRef<mlir::ModuleOp> module;
  mlir::Operation *kernel = nullptr;
};

ParsedKernel parseKernel(llvm::StringRef text) {
  ParsedKernel parsed;
  parsed.context = std::make_unique<mlir::MLIRContext>();
  parsed.context->getOrLoadDialect<mlir::micro::MicroDialect>();
  parsed.context->getOrLoadDialect<mlir::arith::ArithDialect>();
  parsed.module =
      mlir::parseSourceString<mlir::ModuleOp>(text, parsed.context.get());
  if (parsed.module)
    parsed.module->walk([&](mlir::Operation *op) {
      if (op->getName().getStringRef() == "micro.kernel")
        parsed.kernel = op;
    });
  return parsed;
}

/// A single `micro.vector` inside `micro.for(0, upper, 1)`. `upper` is a
/// constant when `constantBound`, otherwise a computed (non-constant) index.
constexpr llvm::StringLiteral kLoopedKernel = R"mlir(
module {
  micro.kernel @loop {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %upper = arith.addi %c4, %c1 : index
    micro.for %i = %c0 to %upper step %c1 {
      %t = micro.tile_alloc : !micro.tile<8x8xf32, memory = #micro.memory<sram>>
      %r = micro.vector "add" %t, %t : !micro.tile<8x8xf32, memory = #micro.memory<sram>>, !micro.tile<8x8xf32, memory = #micro.memory<sram>> -> !micro.tile<8x8xf32, memory = #micro.memory<sram>>
    }
    micro.yield
  }
}
)mlir";

constexpr llvm::StringLiteral kLoopedKernelConstant = R"mlir(
module {
  micro.kernel @loop {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    micro.for %i = %c0 to %c4 step %c1 {
      %t = micro.tile_alloc : !micro.tile<8x8xf32, memory = #micro.memory<sram>>
      %r = micro.vector "add" %t, %t : !micro.tile<8x8xf32, memory = #micro.memory<sram>>, !micro.tile<8x8xf32, memory = #micro.memory<sram>> -> !micro.tile<8x8xf32, memory = #micro.memory<sram>>
    }
    micro.yield
  }
}
)mlir";

} // namespace

TEST(StoragePlan, ExtractedKnownLoopMultiplicityFinalizesInStrictMode) {
  ParsedKernel parsed = parseKernel(kLoopedKernelConstant);
  ASSERT_TRUE(parsed.module);
  ASSERT_NE(parsed.kernel, nullptr);
  llvm::Expected<WorkloadGraph> graph = extractWorkloadGraph(parsed.kernel);
  ASSERT_TRUE(static_cast<bool>(graph)) << llvm::toString(graph.takeError());
  ASSERT_EQ(graph->getNodes().size(), 1u);
  // Extraction recovers the enclosing loop's trip count.
  ASSERT_TRUE(graph->getNodes()[0].executionMultiplicity.has_value());
  EXPECT_EQ(*graph->getNodes()[0].executionMultiplicity, 4u);

  std::unique_ptr<MappingTarget> target = storageTarget(storageMachine());
  ASSERT_NE(target, nullptr);
  std::optional<CoveringPlan> plan =
      searchOne(*graph, *target, *parsed.context);
  ASSERT_TRUE(plan.has_value());
  ASSERT_TRUE(plan->materialized); // strict executable planning

  llvm::Error error = finalizeStoragePlan(*graph, *plan, storageMachine());
  ASSERT_FALSE(bool(error)) << llvm::toString(std::move(error));
  ASSERT_EQ(plan->allocations.size(), 1u);
  // 8x8xf32 = 256 bytes per iteration, four iterations.
  EXPECT_EQ(plan->allocations[0].bytes, 1024u);
}

TEST(StoragePlan, AnUnresolvedLoopBoundStaysUnknownAndRefusesStrictPlanning) {
  ParsedKernel parsed = parseKernel(kLoopedKernel);
  ASSERT_TRUE(parsed.module);
  ASSERT_NE(parsed.kernel, nullptr);
  llvm::Expected<WorkloadGraph> graph = extractWorkloadGraph(parsed.kernel);
  ASSERT_TRUE(static_cast<bool>(graph)) << llvm::toString(graph.takeError());
  ASSERT_EQ(graph->getNodes().size(), 1u);
  // A computed bound is not statically recoverable, so it stays unknown rather
  // than being assumed to run once.
  EXPECT_FALSE(graph->getNodes()[0].executionMultiplicity.has_value());

  std::unique_ptr<MappingTarget> target = storageTarget(storageMachine());
  ASSERT_NE(target, nullptr);
  std::optional<CoveringPlan> plan =
      searchOne(*graph, *target, *parsed.context);
  ASSERT_TRUE(plan.has_value());
  llvm::Error error = finalizeStoragePlan(*graph, *plan, storageMachine());
  ASSERT_TRUE(bool(error));
  EXPECT_NE(llvm::toString(std::move(error)).find("multiplicity"),
            std::string::npos);
}

//===----------------------------------------------------------------------===//
// Fix round 1: id stability, the report round trip, and staged replicas
//===----------------------------------------------------------------------===//

TEST(StoragePlan, FinalizingTwiceIsIdempotentAndLeavesTheIdStable) {
  mlir::MLIRContext context;
  WorkloadGraph graph = chainGraph(context, tileType(context, "8x8xf32"), 1);
  std::unique_ptr<MappingTarget> target = storageTarget(storageMachine());
  ASSERT_NE(target, nullptr);
  std::optional<CoveringPlan> plan = searchOne(graph, *target, context);
  ASSERT_TRUE(plan.has_value());

  const PlanId before = computePlanId(*plan);
  ASSERT_FALSE(bool(finalizeStoragePlan(graph, *plan, storageMachine())));
  const size_t notes = plan->diagnostics.storageNotes.size();
  EXPECT_FALSE(plan->steps.empty());
  // Informational notes are excluded from the plan id, so finalizing does not
  // change it.
  EXPECT_EQ(computePlanId(*plan), before);

  ASSERT_FALSE(bool(finalizeStoragePlan(graph, *plan, storageMachine())));
  EXPECT_EQ(plan->diagnostics.storageNotes.size(), notes);
  EXPECT_EQ(computePlanId(*plan), before);
}

TEST(StoragePlan, ReportRoundTripsThePlanStepDag) {
  mlir::MLIRContext context;
  WorkloadGraph graph = chainGraph(context, tileType(context, "8x8xf32"), 1);
  std::unique_ptr<MappingTarget> target = storageTarget(storageMachine());
  ASSERT_NE(target, nullptr);

  MappingSearchOptions options;
  options.mode = SearchMode::Deterministic;
  CoveringSearch search(graph, *target, context, LayoutContext{}, options);
  llvm::Expected<MappingSearchResult> result = search.search();
  ASSERT_TRUE(static_cast<bool>(result)) << llvm::toString(result.takeError());
  ASSERT_FALSE(result->plans.empty());
  CoveringPlan &plan = result->plans.front();
  ASSERT_FALSE(bool(finalizeStoragePlan(graph, plan, storageMachine())));
  ASSERT_FALSE(plan.steps.empty());

  std::string report = writePlanReport(*result, target->machine(), *target,
                                       options, /*moduleHash=*/0);
  llvm::Expected<CoveringPlan> replay = readPlanReport(report, *target, graph);
  ASSERT_TRUE(static_cast<bool>(replay)) << llvm::toString(replay.takeError());
  ASSERT_EQ(replay->steps.size(), plan.steps.size());
  for (size_t i = 0; i < plan.steps.size(); ++i) {
    EXPECT_EQ(replay->steps[i].id, plan.steps[i].id);
    EXPECT_EQ(replay->steps[i].kind, plan.steps[i].kind);
    EXPECT_EQ(replay->steps[i].node, plan.steps[i].node);
    EXPECT_EQ(replay->steps[i].connection, plan.steps[i].connection);
  }
  EXPECT_EQ(replay->stepEdges, plan.stepEdges);
}

TEST(StoragePlan, ATransformedReplicaIsChargedToItsDestinationMemory) {
  mlir::MLIRContext context;
  mlir::Type tile = tileType(context, "8x8xf32");

  // n0 (sram) -> v0 -> n1 (dram), with v0 transformed in place of a plain copy.
  WorkloadGraph graph;
  WorkloadValueId v0 =
      graph.addValue(WorkloadValue{0, tile, "v0", /*external=*/false});
  WorkloadValueId v1 =
      graph.addValue(WorkloadValue{0, tile, "v1", /*external=*/false});
  WorkloadNode producer;
  producer.opName = "micro.vector";
  producer.sourceOrdinal = 0;
  producer.executionMultiplicity = 1;
  producer.outputs.push_back(WorkloadPort{v0, tile, std::nullopt});
  graph.addNode(std::move(producer));
  WorkloadNode consumer;
  consumer.opName = "micro.vector";
  consumer.sourceOrdinal = 1;
  consumer.executionMultiplicity = 1;
  consumer.inputs.push_back(WorkloadPort{v0, tile, std::nullopt});
  consumer.outputs.push_back(WorkloadPort{v1, tile, std::nullopt});
  graph.addNode(std::move(consumer));
  graph.finalize();

  const WorkloadNode *producerNode = nullptr;
  const WorkloadNode *consumerNode = nullptr;
  for (const WorkloadNode &node : graph.getNodes())
    (node.inputs.empty() ? producerNode : consumerNode) = &node;
  ASSERT_NE(producerNode, nullptr);
  ASSERT_NE(consumerNode, nullptr);
  const WorkloadValueId mid = producerNode->outputs[0].value;

  CoveringPlan plan;
  plan.materialized = true;
  PlanPlacement producerPlacement;
  producerPlacement.node = producerNode->id;
  producerPlacement.instance = 10;
  producerPlacement.memories["sram"] = "sram.0";
  plan.placements.push_back(producerPlacement);
  PlanPlacement consumerPlacement;
  consumerPlacement.node = consumerNode->id;
  consumerPlacement.instance = 20;
  consumerPlacement.memories["dram"] = "dram.0";
  plan.placements.push_back(consumerPlacement);

  mlir::AffineExpr d0 = mlir::getAffineDimExpr(0, &context);
  mlir::AffineExpr d1 = mlir::getAffineDimExpr(1, &context);
  PlanConnection connection;
  connection.id = 7;
  connection.value = mid;
  connection.kind = ConnectionKind::TransferAndTransform;
  connection.route = {"sram.0", "dram.0"};
  connection.consumers = {20};
  LayoutTransform transform;
  transform.srcLayout = "plain";
  transform.dstLayout = "blocked";
  transform.srcMap = mlir::AffineMap::get(2, 0, {d0, d1}, &context);
  // `(m, n) -> (m, n * 2)` doubles the second extent, so the replica's physical
  // image (8 x 15 x 4 = 480 bytes) is larger than the 256-byte logical tile.
  transform.dstMap = mlir::AffineMap::get(
      2, 0, {d0, d1 * mlir::getAffineConstantExpr(2, &context)}, &context);
  connection.transform = transform;
  plan.connectionPlans.push_back(connection);

  // 700 bytes in dram holds the producer's 256-byte output but not the 736-byte
  // live peak (the consumer output plus the 480-byte transformed replica).
  MachineModel tight = storageMachine();
  for (MemoryNode &memory : tight.memories)
    if (memory.kind == "dram")
      memory.capacityBytes = 700;
  llvm::Error rejected = finalizeStoragePlan(graph, plan, tight);
  ASSERT_TRUE(bool(rejected));
  const std::string message = llvm::toString(std::move(rejected));
  EXPECT_NE(message.find("dram.0"), std::string::npos);
  EXPECT_NE(message.find("capacity"), std::string::npos);

  // With room for the peak, the replica is charged to its own memory.
  ASSERT_FALSE(bool(finalizeStoragePlan(graph, plan, storageMachine())));
  bool sawReplica = false;
  bool sawSource = false;
  for (const StorageAllocation &allocation : plan.allocations) {
    if (allocation.memory == "dram.0" && allocation.bytes == 480u)
      sawReplica = true;
    if (allocation.memory == "sram.0" && allocation.bytes == 256u)
      sawSource = true;
  }
  EXPECT_TRUE(sawReplica)
      << "the transformed replica must be charged to dram.0";
  EXPECT_TRUE(sawSource)
      << "the immutable producer must stay charged in sram.0";
}

//===----------------------------------------------------------------------===//
// Fix round 2: multiplicity joins source-graph identity; metadata round trip
//===----------------------------------------------------------------------===//

namespace {

/// `kLoopedKernelConstant` with the trip count doubled to eight.
constexpr llvm::StringLiteral kLoopedKernelEight = R"mlir(
module {
  micro.kernel @loop {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index
    micro.for %i = %c0 to %c8 step %c1 {
      %t = micro.tile_alloc : !micro.tile<8x8xf32, memory = #micro.memory<sram>>
      %r = micro.vector "add" %t, %t : !micro.tile<8x8xf32, memory = #micro.memory<sram>>, !micro.tile<8x8xf32, memory = #micro.memory<sram>> -> !micro.tile<8x8xf32, memory = #micro.memory<sram>>
    }
    micro.yield
  }
}
)mlir";

} // namespace

TEST(StoragePlan, AChangedLoopTripCountChangesTheGraphHashAndRejectsReplay) {
  ParsedKernel four = parseKernel(kLoopedKernelConstant);
  ParsedKernel eight = parseKernel(kLoopedKernelEight);
  ASSERT_TRUE(four.module);
  ASSERT_TRUE(eight.module);
  llvm::Expected<WorkloadGraph> graphFour = extractWorkloadGraph(four.kernel);
  llvm::Expected<WorkloadGraph> graphEight = extractWorkloadGraph(eight.kernel);
  ASSERT_TRUE(static_cast<bool>(graphFour))
      << llvm::toString(graphFour.takeError());
  ASSERT_TRUE(static_cast<bool>(graphEight))
      << llvm::toString(graphEight.takeError());

  // A trip count is a semantic fact: two kernels identical but for it must not
  // share a source-graph identity.
  EXPECT_NE(computeSourceGraphHash(*graphFour),
            computeSourceGraphHash(*graphEight));

  std::unique_ptr<MappingTarget> target = storageTarget(storageMachine());
  ASSERT_NE(target, nullptr);
  MappingSearchOptions options;
  options.mode = SearchMode::Deterministic;
  CoveringSearch search(*graphFour, *target, *four.context, LayoutContext{},
                        options);
  llvm::Expected<MappingSearchResult> result = search.search();
  ASSERT_TRUE(static_cast<bool>(result)) << llvm::toString(result.takeError());
  ASSERT_FALSE(result->plans.empty());
  ASSERT_FALSE(bool(finalizeStoragePlan(*graphFour, result->plans.front(),
                                        storageMachine())));
  std::string report = writePlanReport(*result, target->machine(), *target,
                                       options, /*moduleHash=*/0);

  // The report bound for the four-iteration kernel is rejected against the
  // eight-iteration kernel's graph, whose storage reservation would be double.
  llvm::Expected<CoveringPlan> replayed =
      readPlanReport(report, *target, *graphEight);
  EXPECT_FALSE(static_cast<bool>(replayed));

  // The control: the same report replays against its own graph.
  llvm::Expected<CoveringPlan> control =
      readPlanReport(report, *target, *graphFour);
  EXPECT_TRUE(static_cast<bool>(control))
      << llvm::toString(control.takeError());
}

TEST(StoragePlan, DirectConnectionsProduceNoMovementStep) {
  mlir::MLIRContext context;
  WorkloadGraph graph = chainGraph(context, tileType(context, "8x8xf32"), 1);
  std::unique_ptr<MappingTarget> target = storageTarget(storageMachine());
  ASSERT_NE(target, nullptr);
  std::optional<CoveringPlan> plan = searchOne(graph, *target, context);
  ASSERT_TRUE(plan.has_value());
  ASSERT_FALSE(bool(finalizeStoragePlan(graph, *plan, storageMachine())));

  // Every connection in this all-sram chain is Direct, so the plan-step DAG is
  // compute steps only: a connection that materializes nothing gets no step.
  size_t computeSteps = 0;
  for (const PlanStep &step : plan->steps) {
    EXPECT_NE(step.kind, PlanStepKind::Movement);
    EXPECT_NE(step.kind, PlanStepKind::Synchronization);
    if (step.kind == PlanStepKind::Compute)
      ++computeSteps;
  }
  EXPECT_EQ(computeSteps, graph.getNodes().size());
}

TEST(StoragePlan, ModuleMetadataRoundTripsThePlanStepDag) {
  ParsedKernel parsed = parseKernel(kLoopedKernelConstant);
  ASSERT_TRUE(parsed.module);
  ASSERT_NE(parsed.kernel, nullptr);
  llvm::Expected<WorkloadGraph> graph = extractWorkloadGraph(parsed.kernel);
  ASSERT_TRUE(static_cast<bool>(graph)) << llvm::toString(graph.takeError());

  std::unique_ptr<MappingTarget> target = storageTarget(storageMachine());
  ASSERT_NE(target, nullptr);
  MappingSearchOptions options;
  options.mode = SearchMode::Deterministic;
  CoveringSearch search(*graph, *target, *parsed.context, LayoutContext{},
                        options);
  llvm::Expected<MappingSearchResult> result = search.search();
  ASSERT_TRUE(static_cast<bool>(result)) << llvm::toString(result.takeError());
  ASSERT_FALSE(result->plans.empty());
  CoveringPlan &plan = result->plans.front();
  ASSERT_FALSE(bool(finalizeStoragePlan(*graph, plan, storageMachine())));
  ASSERT_FALSE(plan.steps.empty());

  llvm::Error encoded = encodeSelectedPlan(parsed.module.get(), plan, *target);
  ASSERT_FALSE(bool(encoded)) << llvm::toString(std::move(encoded));
  llvm::Expected<CoveringPlan> decoded =
      decodeSelectedPlan(parsed.module.get(), *target);
  ASSERT_TRUE(static_cast<bool>(decoded))
      << llvm::toString(decoded.takeError());
  ASSERT_EQ(decoded->steps.size(), plan.steps.size());
  for (size_t i = 0; i < plan.steps.size(); ++i) {
    EXPECT_EQ(decoded->steps[i].id, plan.steps[i].id);
    EXPECT_EQ(decoded->steps[i].kind, plan.steps[i].kind);
    EXPECT_EQ(decoded->steps[i].node, plan.steps[i].node);
    EXPECT_EQ(decoded->steps[i].connection, plan.steps[i].connection);
  }
  EXPECT_EQ(decoded->stepEdges, plan.stepEdges);
}
