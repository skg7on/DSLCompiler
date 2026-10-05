//===- storage_plan.cpp - Physical storage footprints and value lifetimes
//--===//
//
// Task B3 (issue #67, stage B). Exercises the two storage-planning contracts:
// `computePeakStorage` summarizes a validated deterministic schedule's live
// ranges, and `finalizeStoragePlan` builds the dependency-aware intervals and
// checks per-memory capacity against real occupancy.

#include "LLK/Mapping/StoragePlan.h"

#include "LLK/Mapping/CoveringSearch.h"
#include "LLK/Mapping/MappingPlan.h"
#include "LLK/Mapping/MappingTarget.h"
#include "LLK/Mapping/TileFacts.h"

#include "LLK/Dialect/Micro/MicroDialect.h"

#include "mlir/AsmParser/AsmParser.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"

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

/// The node attributes a fixture carries: the `op` the rule predicates on and
/// the execution multiplicity the strict planner reads.
mlir::DictionaryAttr nodeAttributes(mlir::MLIRContext &context,
                                    uint64_t multiplicity) {
  return mlir::DictionaryAttr::get(
      &context,
      {mlir::NamedAttribute(mlir::StringAttr::get(&context, "op"),
                            mlir::StringAttr::get(&context, "add")),
       mlir::NamedAttribute(
           mlir::StringAttr::get(&context, kExecutionMultiplicityAttr),
           mlir::IntegerAttr::get(mlir::IntegerType::get(&context, 64),
                                  multiplicity))});
}

/// A three-node chain `in -> n0 -> v0 -> n1 -> v1 -> n2 -> out`, every node a
/// `micro.vector` over an 8x8xf32 tile. When `multiplicity` is set each node
/// declares it, so the strict planner has a known execution count.
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
    if (multiplicity)
      n.attributes = nodeAttributes(context, *multiplicity);
    else
      n.attributes = mlir::DictionaryAttr::get(
          &context,
          {mlir::NamedAttribute(mlir::StringAttr::get(&context, "op"),
                                mlir::StringAttr::get(&context, "add"))});
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
  // The occupancy is recorded in the report.
  bool noted = false;
  for (const std::string &warning : plan->diagnostics.warnings)
    noted |= warning.find("sram.0") != std::string::npos;
  EXPECT_TRUE(noted);
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
  for (const std::string &warning : plan->diagnostics.warnings)
    reported |= warning.find("multiplicity") != std::string::npos;
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
