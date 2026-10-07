//===- storage_liveness.cpp - Live storage from execution events ----------===//
//
// Task R5 (issue #129). Exercises `analyzeStorageLiveness`: the peak residency
// of each memory, derived from the plan's *scheduled* events, over fixtures
// whose expected peaks are literal values. The three structural fixtures make
// the residency distinction the task is about concrete:
//
//   * `sequential`     -- four temporal iterations of one 256-byte value reuse
//                         one buffer, so the peak is 256;
//   * `pipeline-four`  -- four overlapping pipeline stages keep four versions
//                         resident, so the peak is 1024;
//   * `parallel-overlap` -- two spatial occurrences run side by side, so the
//                         peak is 512 and a 256-byte memory cannot hold it.
//
// `padded-layout` adds the footprint rule: what a value occupies is its
// physical image, so an 8x8xf32 result under a row-major layout whose physical
// row is twice the logical one occupies an 8x15 image -- 480 bytes, not the
// logical 256.
//
//===----------------------------------------------------------------------===//

#include "LLK/Mapping/StorageLiveness.h"

#include "resource_regression_fixture.h"

#include "LLK/Machine/MachineModel.h"
#include "LLK/Mapping/CostEvent.h"
#include "LLK/Mapping/CoveringSearch.h"
#include "LLK/Mapping/EventSchedule.h"
#include "LLK/Mapping/MappingPlan.h"
#include "LLK/Mapping/MappingTarget.h"
#include "LLK/Mapping/StorageLiveness.h"
#include "LLK/Mapping/StoragePlan.h"

#include "llvm/Support/Error.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <optional>
#include <string>
#include <vector>

using namespace mlir::llk::machine;
using namespace mlir::llk::mapping;

namespace {

/// One fixture case searched, storage-finalized and analyzed, so a test asserts
/// on the liveness of a plan the compiler actually produced rather than on a
/// hand-written expectation.
struct Liveness {
  issue129::ResourceCase resource;
  CoveringPlan plan;
  PlanEventDAG dag;
  EventScheduleResult schedule;
  StorageLivenessResult live;
};

/// Searches `name`, finalizes its best plan and analyzes the liveness of the
/// resulting scheduled event stream.
llvm::Expected<Liveness> analyzeCase(llvm::StringRef name) {
  llvm::Expected<issue129::ResourceCase> built = issue129::resourceCase(name);
  if (!built)
    return built.takeError();
  Liveness out;
  out.resource = std::move(*built);
  MappingSearchOptions options;
  options.mode = SearchMode::Exact;
  llvm::Expected<MappingSearchResult> result =
      issue129::searchCase(out.resource, options);
  if (!result)
    return result.takeError();
  if (result->plans.empty())
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "the liveness fixture searched no plan");
  out.plan = result->plans.front();
  if (llvm::Error error = finalizeStoragePlan(out.resource.graph, out.plan,
                                              out.resource.target->machine()))
    return std::move(error);
  llvm::Expected<PlanEventDAG> events = buildPlanEvents(
      out.plan, out.resource.target->machine(), &out.resource.graph);
  if (!events)
    return events.takeError();
  out.dag = std::move(*events);
  out.schedule =
      scheduleNormalizedEvents(out.dag.events, out.resource.target->machine());
  llvm::Expected<StorageLivenessResult> live =
      analyzeStorageLiveness(out.plan, out.dag, out.schedule);
  if (!live)
    return live.takeError();
  out.live = std::move(*live);
  return out;
}

/// A minimal event carrying `uses` at `start`..`finish`.
void addEvent(PlanEventDAG &dag, EventScheduleResult &schedule, uint64_t start,
              uint64_t finish, uint64_t planStep,
              std::vector<StorageUse> uses) {
  PlanCostEvent event;
  event.planStep = planStep;
  event.storageUses = std::move(uses);
  const uint32_t id = static_cast<uint32_t>(dag.events.size());
  dag.events.push_back(std::move(event));
  ScheduledEvent placed;
  placed.id = id;
  placed.start = start;
  placed.finish = finish;
  schedule.entries.push_back(placed);
  schedule.predictedCycles = std::max(schedule.predictedCycles, finish);
}

} // namespace

//===----------------------------------------------------------------------===//
// Structural residency
//===----------------------------------------------------------------------===//

TEST(StorageLiveness, SequentialLoopReusesOneBuffer) {
  llvm::Expected<Liveness> built = analyzeCase("sequential");
  ASSERT_TRUE(bool(built)) << llvm::toString(built.takeError());
  // Four temporal iterations of one 256-byte result: the loop reuses the
  // buffer, so the peak is a single value, never four.
  EXPECT_EQ(built->live.peakBytes.at("l2.0"), 256u)
      << built->plan.diagnostics.storageNotes.front();
  // The borrowed DRAM operand is the caller's buffer, and its own 256 bytes.
  EXPECT_EQ(built->live.peakBytes.at("dram.0"), 256u);
}

TEST(StorageLiveness, PipelineResidencyMultipliesLiveVersions) {
  llvm::Expected<Liveness> built = analyzeCase("pipeline-four");
  ASSERT_TRUE(bool(built)) << llvm::toString(built.takeError());
  // Four overlapping stages keep four 256-byte versions resident: the same
  // loop shape as `sequential`, a different (and larger) peak.
  EXPECT_EQ(built->live.peakBytes.at("l2.0"), 1024u);
}

TEST(StorageLiveness, ParallelOccurrencesOverlapInTheirMemory) {
  llvm::Expected<Liveness> built = analyzeCase("parallel-overlap");
  ASSERT_TRUE(bool(built)) << llvm::toString(built.takeError());
  // Two spatial occurrences run side by side, so both 256-byte results are
  // live at once.
  EXPECT_EQ(built->live.peakBytes.at("l2.0"), 512u);
}

TEST(StorageLiveness, TheSamePlanIsRejectedWhenTheMemoryIsTooSmall) {
  llvm::Expected<issue129::ResourceCase> resource =
      issue129::resourceCase("parallel-overlap");
  ASSERT_TRUE(bool(resource)) << llvm::toString(resource.takeError());
  MappingSearchOptions options;
  options.mode = SearchMode::Exact;
  llvm::Expected<MappingSearchResult> result =
      issue129::searchCase(*resource, options);
  ASSERT_TRUE(bool(result)) << llvm::toString(result.takeError());
  ASSERT_FALSE(result->plans.empty());

  // L2 reduced to 256 bytes: the 512-byte live peak no longer fits, and the
  // rejection names the memory and its capacity rather than silently binding a
  // reservation that cannot be met.
  MachineModel small = resource->target->machine();
  for (MemoryNode &memory : small.memories)
    if (memory.id == "l2.0")
      memory.capacityBytes = 256;

  CoveringPlan plan = result->plans.front();
  llvm::Error error = finalizeStoragePlan(resource->graph, plan, small);
  ASSERT_TRUE(bool(error));
  const std::string message = llvm::toString(std::move(error));
  EXPECT_NE(message.find("l2.0"), std::string::npos) << message;
  EXPECT_NE(message.find("capacity"), std::string::npos) << message;
  EXPECT_NE(message.find("512"), std::string::npos) << message;
}

TEST(StorageLiveness, APaddedPhysicalImageIsWhatIsReserved) {
  llvm::Expected<Liveness> built = analyzeCase("padded-layout");
  ASSERT_TRUE(bool(built)) << llvm::toString(built.takeError());
  // 8x8xf32 under a row-major layout whose physical row is twice the logical
  // one: an 8x15 image, 120 elements, 480 bytes -- not the logical 256. What a
  // memory holds is the physical image, never the index space.
  EXPECT_EQ(built->live.peakBytes.at("l2.0"), 480u);
}

//===----------------------------------------------------------------------===//
// The relation itself
//===----------------------------------------------------------------------===//

// An allocation no scheduled event touches has no live range, so its footprint
// is unknown: a modelling gap is rejected rather than read as zero.
TEST(StorageLiveness, AnAllocationNoEventTouchesIsRejected) {
  CoveringPlan plan;
  StorageAllocation allocation;
  allocation.id = 7;
  allocation.memory = "l2.0";
  allocation.bytes = 256;
  plan.allocations.push_back(allocation);

  PlanEventDAG dag;
  EventScheduleResult schedule;
  llvm::Expected<StorageLivenessResult> live =
      analyzeStorageLiveness(plan, dag, schedule);
  ASSERT_FALSE(bool(live));
  const std::string message = llvm::toString(live.takeError());
  EXPECT_NE(message.find("touched by no scheduled event"), std::string::npos)
      << message;
}

// A reuse the plan chose is reported as the edge that makes it sound: the
// reused buffer's last use must complete before the new writer begins, and the
// caller materializes that ordering in the emitted step DAG.
TEST(StorageLiveness, AChosenReuseReportsItsOrderingEdge) {
  CoveringPlan plan;
  StorageAllocation first;
  first.id = 1;
  first.memory = "l2.0";
  first.bytes = 256;
  first.beginStep = 0;
  first.endStep = 5;
  StorageAllocation second;
  second.id = 2;
  second.memory = "l2.0";
  second.bytes = 256;
  second.beginStep = 9;
  second.endStep = 11;
  second.aliasOf = 1; // proven dead at step 5, rewritten at step 9
  plan.allocations = {first, second};

  PlanEventDAG dag;
  EventScheduleResult schedule;
  addEvent(dag, schedule, 0, 10, 0, {StorageUse{1, 0, StorageAccess::Write}});
  addEvent(dag, schedule, 20, 30, 5, {StorageUse{1, 0, StorageAccess::Read}});
  addEvent(dag, schedule, 40, 50, 9, {StorageUse{2, 0, StorageAccess::Write}});
  addEvent(dag, schedule, 60, 70, 11, {StorageUse{2, 0, StorageAccess::Read}});

  llvm::Expected<StorageLivenessResult> live =
      analyzeStorageLiveness(plan, dag, schedule);
  ASSERT_TRUE(bool(live)) << llvm::toString(live.takeError());
  // One buffer, not two: the alias contributes to its root's bytes once.
  EXPECT_EQ(live->peakBytes.at("l2.0"), 256u);
  ASSERT_EQ(live->requiredReuseEdges.size(), 1u);
  EXPECT_EQ(live->requiredReuseEdges[0].from, 5u);
  EXPECT_EQ(live->requiredReuseEdges[0].to, 9u);
}

// An *in-place* update -- an epilogue that consumes a buffer and writes its
// result into it -- aliases at a single step, so its ordering edge is a
// self-step edge. It is still reported: the peak relies on the two allocations
// being one buffer, and a reader of the plan must be able to see that.
TEST(StorageLiveness, AnInPlaceReuseReportsItsSelfStepEdge) {
  CoveringPlan plan;
  StorageAllocation accumulator;
  accumulator.id = 1;
  accumulator.memory = "acc.0";
  accumulator.bytes = 1024;
  accumulator.beginStep = 3;
  accumulator.endStep = 5;
  StorageAllocation epilogue;
  epilogue.id = 2;
  epilogue.memory = "acc.0";
  epilogue.bytes = 512;
  epilogue.beginStep = 5;
  epilogue.endStep = 6;
  epilogue.aliasOf = 1; // the epilogue writes into the accumulator it reads
  plan.allocations = {accumulator, epilogue};

  PlanEventDAG dag;
  EventScheduleResult schedule;
  addEvent(dag, schedule, 0, 10, 3, {StorageUse{1, 1, StorageAccess::Write}});
  addEvent(dag, schedule, 20, 30, 5,
           {StorageUse{1, 1, StorageAccess::Read},
            StorageUse{2, 1, StorageAccess::Write}});
  addEvent(dag, schedule, 40, 50, 6, {StorageUse{2, 1, StorageAccess::Read}});

  llvm::Expected<StorageLivenessResult> live =
      analyzeStorageLiveness(plan, dag, schedule);
  ASSERT_TRUE(bool(live)) << llvm::toString(live.takeError());
  // One buffer of 1024 bytes, not an accumulator plus an epilogue buffer.
  EXPECT_EQ(live->peakBytes.at("acc.0"), 1024u);
  ASSERT_EQ(live->requiredReuseEdges.size(), 1u);
  EXPECT_EQ(live->requiredReuseEdges[0].from, 5u);
  EXPECT_EQ(live->requiredReuseEdges[0].to, 5u);
  // The same relation is what the caller merges into the plan's step DAG, so
  // the ordering the peak assumes is the one the plan carries.
  EXPECT_EQ(requiredReuseEdgesFor(plan.allocations), live->requiredReuseEdges);
}
