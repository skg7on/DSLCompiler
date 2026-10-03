//===- placement.cpp - Placement enumeration (D5) ------------------------===//

#include "LLK/Mapping/Placement.h"

#include "llvm/Support/Error.h"

#include <gtest/gtest.h>

#include <string>
#include <vector>

using namespace mlir::llk::machine;
using namespace mlir::llk::mapping;

namespace {

ComputeNode compute(llvm::StringRef id, llvm::StringRef kind,
                    llvm::StringRef attachedTo) {
  ComputeNode node;
  node.id = id.str();
  node.kind = kind.str();
  node.attachedTo = attachedTo.str();
  node.shapes = {{8}};
  return node;
}

MemoryNode memory(llvm::StringRef id, llvm::StringRef kind,
                  llvm::StringRef visibleFrom) {
  MemoryNode node;
  node.id = id.str();
  node.kind = kind.str();
  node.visibleFrom = visibleFrom.str();
  node.capacityBytes = 1u << 20;
  node.alignmentBytes = 64;
  return node;
}

/// pkg.0 offers the worker owner directly; core.0 and core.1 refine it and
/// carry the compute and memory attachments.
MachineModel placementMachine() {
  MachineModel model;
  model.target = "placement";
  model.executors = {
      {"pkg.0", "worker", std::nullopt, {}, 1, {}},
      {"core.0", "core", std::string("pkg.0"), {0}, 1, {"worker"}},
      {"core.1", "core", std::string("pkg.0"), {1}, 1, {"worker"}}};
  model.computes = {compute("vec.0", "vector_engine", "core.0"),
                    compute("mxu.0", "matrix_engine", "core.0")};
  model.memories = {memory("sram.0", "sram", "core.0"),
                    memory("sram.1", "sram", "core.1"),
                    memory("dram.0", "dram", "pkg.0")};
  return model;
}

constexpr llvm::StringLiteral kLayouts = R"llkmap(
layout t.rank2(int N) {
  param N in [1..8];
  require rank == 2;
}
layout t.rank3(int N) {
  param N in [1..8];
  require rank == 3;
}
)llkmap";

std::unique_ptr<MappingTarget> targetFor(MachineModel machine,
                                         llvm::StringRef layouts = kLayouts) {
  llvm::Expected<LayoutRegistry> registry = parseLayoutText(layouts, "<test>");
  if (!registry)
    return nullptr;
  return std::make_unique<FileMappingTarget>(
      "test", std::move(machine), std::move(*registry), RuleRegistry{},
      std::vector<std::string>{});
}

MappingCandidate candidate() {
  MappingCandidate candidate;
  candidate.rule = "t.rule";
  candidate.coveredNodes = {0};
  ExecutorRequirement executor;
  executor.capability = "worker";
  candidate.executorRequirements.push_back(executor);
  return candidate;
}

std::vector<std::string>
boundExecutors(const std::vector<CandidateInstance> &instances) {
  std::vector<std::string> ids;
  for (const CandidateInstance &instance : instances)
    ids.push_back(instance.executorBindings.lookup("executor"));
  return ids;
}

} // namespace

TEST(Placement, PlacesOnEveryExecutorThatOffersTheCapability) {
  std::unique_ptr<MappingTarget> target = targetFor(placementMachine());
  ASSERT_NE(target, nullptr);
  mlir::MLIRContext context;
  LayoutContext layoutContext;
  layoutContext.rank = 2;

  llvm::Expected<std::vector<CandidateInstance>> instances =
      enumeratePlacements(candidate(), *target, context, layoutContext);
  ASSERT_TRUE(static_cast<bool>(instances))
      << llvm::toString(instances.takeError());
  // pkg.0 offers worker; core.0 and core.1 refine it.
  EXPECT_EQ(boundExecutors(*instances),
            (std::vector<std::string>{"pkg.0", "core.0", "core.1"}));
}

TEST(Placement, YieldsNothingWhenNoExecutorOffersTheCapability) {
  std::unique_ptr<MappingTarget> target = targetFor(placementMachine());
  ASSERT_NE(target, nullptr);
  MappingCandidate unplaceable = candidate();
  unplaceable.executorRequirements[0].capability = "pe";
  mlir::MLIRContext context;
  llvm::Expected<std::vector<CandidateInstance>> instances =
      enumeratePlacements(unplaceable, *target, context, LayoutContext{});
  ASSERT_TRUE(static_cast<bool>(instances));
  EXPECT_TRUE(instances->empty());
}

TEST(Placement, AttachesAComputeOfTheRequiredKind) {
  std::unique_ptr<MappingTarget> target = targetFor(placementMachine());
  ASSERT_NE(target, nullptr);
  MappingCandidate withCompute = candidate();
  ComputeRequirement requirement;
  requirement.kind = "vector_engine";
  withCompute.computeRequirements.push_back(requirement);
  mlir::MLIRContext context;

  llvm::Expected<std::vector<CandidateInstance>> instances =
      enumeratePlacements(withCompute, *target, context, LayoutContext{});
  ASSERT_TRUE(static_cast<bool>(instances))
      << llvm::toString(instances.takeError());
  // Only core.0 carries a vector engine.
  EXPECT_EQ(boundExecutors(*instances), (std::vector<std::string>{"core.0"}));
  EXPECT_EQ((*instances)[0].computeBindings.lookup("vector_engine"), "vec.0");
}

TEST(Placement, BindsAVisibleMemoryOfTheRequiredKind) {
  std::unique_ptr<MappingTarget> target = targetFor(placementMachine());
  ASSERT_NE(target, nullptr);
  MappingCandidate withMemory = candidate();
  MemoryRequirement requirement;
  requirement.kind = "sram";
  withMemory.memoryRequirements.push_back(requirement);
  mlir::MLIRContext context;

  llvm::Expected<std::vector<CandidateInstance>> instances =
      enumeratePlacements(withMemory, *target, context, LayoutContext{});
  ASSERT_TRUE(static_cast<bool>(instances))
      << llvm::toString(instances.takeError());
  // core.0 and core.1 each see their own sram; pkg.0 sees none.
  EXPECT_EQ(boundExecutors(*instances),
            (std::vector<std::string>{"core.0", "core.1"}));
  EXPECT_EQ((*instances)[0].memoryBindings.lookup("sram"), "sram.0");
  EXPECT_EQ((*instances)[1].memoryBindings.lookup("sram"), "sram.1");
}

TEST(Placement, RequiresTheLayoutToSolve) {
  std::unique_ptr<MappingTarget> target = targetFor(placementMachine());
  ASSERT_NE(target, nullptr);
  MappingCandidate withLayout = candidate();
  LayoutRequirement requirement;
  requirement.layoutClass = "t.rank3"; // needs rank 3
  withLayout.layoutRequirements.push_back(requirement);

  mlir::MLIRContext context;
  LayoutContext layoutContext;
  layoutContext.rank = 2; // so it cannot solve
  llvm::Expected<std::vector<CandidateInstance>> instances =
      enumeratePlacements(withLayout, *target, context, layoutContext);
  ASSERT_TRUE(static_cast<bool>(instances));
  EXPECT_TRUE(instances->empty());

  LayoutContext rank3;
  rank3.rank = 3;
  llvm::Expected<std::vector<CandidateInstance>> solvable =
      enumeratePlacements(withLayout, *target, context, rank3);
  ASSERT_TRUE(static_cast<bool>(solvable));
  EXPECT_EQ(solvable->size(), 3u);
  EXPECT_EQ((*solvable)[0].layoutBindings.lookup("t.rank3"), "t.rank3");
}

TEST(Placement, SymmetryReductionKeepsOneRepresentative) {
  // Three interchangeable executors: same kind, same parent, same attachments.
  MachineModel machine;
  machine.target = "symmetric";
  machine.executors = {{"c.0", "core", std::nullopt, {}, 1, {}},
                       {"c.1", "core", std::nullopt, {}, 1, {}},
                       {"c.2", "core", std::nullopt, {}, 1, {}}};
  std::unique_ptr<MappingTarget> target = targetFor(std::move(machine));
  ASSERT_NE(target, nullptr);

  MappingCandidate core = candidate();
  core.executorRequirements[0].capability = "core";
  mlir::MLIRContext context;

  PlacementOptions reduced;
  llvm::Expected<std::vector<CandidateInstance>> symmetric =
      enumeratePlacements(core, *target, context, LayoutContext{}, reduced);
  ASSERT_TRUE(static_cast<bool>(symmetric));
  EXPECT_EQ(boundExecutors(*symmetric), (std::vector<std::string>{"c.0"}));

  PlacementOptions full;
  full.reduceSymmetry = false;
  llvm::Expected<std::vector<CandidateInstance>> all =
      enumeratePlacements(core, *target, context, LayoutContext{}, full);
  ASSERT_TRUE(static_cast<bool>(all));
  EXPECT_EQ(boundExecutors(*all),
            (std::vector<std::string>{"c.0", "c.1", "c.2"}));
}

TEST(Placement, ReportsTruncationWhenTheInstanceCapIsHit) {
  // Three interchangeable executors and no reduction: a cap of one genuinely
  // stops enumeration before the other two legal placements are considered.
  MachineModel machine;
  machine.target = "symmetric";
  machine.executors = {{"c.0", "core", std::nullopt, {}, 1, {}},
                       {"c.1", "core", std::nullopt, {}, 1, {}},
                       {"c.2", "core", std::nullopt, {}, 1, {}}};
  std::unique_ptr<MappingTarget> target = targetFor(std::move(machine));
  ASSERT_NE(target, nullptr);

  MappingCandidate core = candidate();
  core.executorRequirements[0].capability = "core";
  mlir::MLIRContext context;

  PlacementOptions capped;
  capped.reduceSymmetry = false;
  capped.maxInstances = 1;
  bool truncated = false;
  llvm::Expected<std::vector<CandidateInstance>> instances =
      enumeratePlacements(core, *target, context, LayoutContext{}, capped,
                          &truncated);
  ASSERT_TRUE(static_cast<bool>(instances))
      << llvm::toString(instances.takeError());
  EXPECT_EQ(boundExecutors(*instances), (std::vector<std::string>{"c.0"}));
  EXPECT_TRUE(truncated);

  // A cap that admits every placement reports no truncation.
  PlacementOptions roomy;
  roomy.reduceSymmetry = false;
  bool notTruncated = false;
  llvm::Expected<std::vector<CandidateInstance>> all = enumeratePlacements(
      core, *target, context, LayoutContext{}, roomy, &notTruncated);
  ASSERT_TRUE(static_cast<bool>(all));
  EXPECT_EQ(all->size(), 3u);
  EXPECT_FALSE(notTruncated);
}

TEST(Placement, InstancesAreLegalAndStable) {
  std::unique_ptr<MappingTarget> target = targetFor(placementMachine());
  ASSERT_NE(target, nullptr);
  MappingCandidate withAll = candidate();
  ComputeRequirement computeRequirement;
  computeRequirement.kind = "vector_engine";
  withAll.computeRequirements.push_back(computeRequirement);
  MemoryRequirement memoryRequirement;
  memoryRequirement.kind = "sram";
  withAll.memoryRequirements.push_back(memoryRequirement);
  LayoutRequirement layoutRequirement;
  layoutRequirement.layoutClass = "t.rank2";
  withAll.layoutRequirements.push_back(layoutRequirement);
  mlir::MLIRContext context;
  LayoutContext layoutContext;
  layoutContext.rank = 2;

  llvm::Expected<std::vector<CandidateInstance>> first =
      enumeratePlacements(withAll, *target, context, layoutContext);
  ASSERT_TRUE(static_cast<bool>(first));
  llvm::Expected<std::vector<CandidateInstance>> second =
      enumeratePlacements(withAll, *target, context, layoutContext);
  ASSERT_TRUE(static_cast<bool>(second));
  ASSERT_EQ(first->size(), second->size());
  for (size_t i = 0; i < first->size(); ++i) {
    // Every binding names something the machine actually has.
    EXPECT_NE((*target).machine().findExecutor(
                  (*first)[i].executorBindings.lookup("executor")),
              nullptr);
    for (const auto &binding : (*first)[i].memoryBindings)
      EXPECT_NE((*target).machine().findMemory(binding.second), nullptr);
    for (const auto &binding : (*first)[i].computeBindings)
      EXPECT_NE((*target).machine().findCompute(binding.second), nullptr);
    EXPECT_EQ((*first)[i].id, (*second)[i].id);
  }
}

//===----------------------------------------------------------------------===//
// Cost dimensions (design §17.2)
//===----------------------------------------------------------------------===//

// Compute utilization is the candidate's rule-local compute cycles over the
// cycles the executors had available in one machine sync period
// (workerThreads x (barrier + wait) cycles).
TEST(Placement, ComputeUtilizationUsesTheSyncWindow) {
  MachineModel machine = placementMachine();
  machine.workerThreads = 4;
  // No clockHz: it cancels in the dimensionless cycle ratio (see
  // utilizationEstimate), so the sync period is the only fact this needs.
  machine.sync.barrierCycles = 200;
  machine.sync.waitCycles = 50;
  std::unique_ptr<MappingTarget> target = targetFor(machine);
  ASSERT_NE(target, nullptr);

  MappingCandidate withCompute = candidate();
  withCompute.lowerBound.latencyCycles = 500.0;
  mlir::MLIRContext context;

  llvm::Expected<std::vector<CandidateInstance>> instances =
      enumeratePlacements(withCompute, *target, context, LayoutContext{});
  ASSERT_TRUE(static_cast<bool>(instances))
      << llvm::toString(instances.takeError());
  ASSERT_FALSE(instances->empty());
  // 500 compute cycles over 4 workers x 250 cycles-per-period = 0.5.
  EXPECT_DOUBLE_EQ(instances->front().localCost.computeUtilization, 0.5);
}

// A machine without sync facts gives no denominator, so the dimension stays 0
// rather than a fabricated constant.
TEST(Placement, ComputeUtilizationStaysZeroWithoutSyncFacts) {
  std::unique_ptr<MappingTarget> target = targetFor(placementMachine());
  ASSERT_NE(target, nullptr);

  MappingCandidate withCompute = candidate();
  withCompute.lowerBound.latencyCycles = 500.0;
  mlir::MLIRContext context;

  llvm::Expected<std::vector<CandidateInstance>> instances =
      enumeratePlacements(withCompute, *target, context, LayoutContext{});
  ASSERT_TRUE(static_cast<bool>(instances))
      << llvm::toString(instances.takeError());
  ASSERT_FALSE(instances->empty());
  EXPECT_DOUBLE_EQ(instances->front().localCost.computeUtilization, 0.0);
}
