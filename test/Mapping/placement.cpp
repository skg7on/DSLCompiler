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

/// One executor carrying two compute capabilities and two memories: every
/// requirement has more than one compatible attachment, so placement must
/// enumerate the combinations rather than pick the first.
MachineModel attachmentMachine() {
  MachineModel model;
  model.target = "attachment";
  model.executors = {{"core.0", "core", std::nullopt, {0}, 1, {}}};
  model.computes = {compute("vec.a", "vector_engine", "core.0"),
                    compute("vec.b", "vector_engine", "core.0"),
                    compute("mxu.a", "matrix_engine", "core.0")};
  model.memories = {memory("sram.a", "sram", "core.0"),
                    memory("sram.b", "sram", "core.0")};
  return model;
}

/// Two executors of the same kind that differ in coordinates and concurrency,
/// so the structural heuristic would keep both; each declares the other
/// `equivalentTo`, so the target's declaration is what makes them
/// interchangeable.
MachineModel declaredEquivalentMachine() {
  MachineModel model;
  model.target = "declared";
  ExecutorNode lhs;
  lhs.id = "c.0";
  lhs.kind = "core";
  lhs.coordinates = {0};
  lhs.concurrency = 1;
  lhs.equivalentTo = {"c.1"};
  ExecutorNode rhs;
  rhs.id = "c.1";
  rhs.kind = "core";
  rhs.coordinates = {3};
  rhs.concurrency = 4;
  rhs.equivalentTo = {"c.0"};
  model.executors = {lhs, rhs};
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

/// A candidate that must run on `core`, so it matches the single-executor
/// attachment machine.
MappingCandidate coreCandidate() {
  MappingCandidate result = candidate();
  result.executorRequirements[0].capability = "core";
  return result;
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

TEST(Placement, EnumeratesEveryCompatibleComputeAttachment) {
  std::unique_ptr<MappingTarget> target = targetFor(attachmentMachine());
  ASSERT_NE(target, nullptr);
  MappingCandidate withCompute = coreCandidate();
  ComputeRequirement requirement;
  requirement.kind = "vector_engine";
  withCompute.computeRequirements.push_back(requirement);
  mlir::MLIRContext context;

  llvm::Expected<std::vector<CandidateInstance>> instances =
      enumeratePlacements(withCompute, *target, context, LayoutContext{});
  ASSERT_TRUE(static_cast<bool>(instances))
      << llvm::toString(instances.takeError());
  // core.0 carries two vector engines; both are compatible attachments, so the
  // candidate places twice rather than collapsing to the first.
  ASSERT_EQ(instances->size(), 2u);
  EXPECT_EQ((*instances)[0].computeBindings.lookup("vector_engine"), "vec.a");
  EXPECT_EQ((*instances)[1].computeBindings.lookup("vector_engine"), "vec.b");
}

TEST(Placement, EnumeratesEveryVisibleMemoryAttachment) {
  std::unique_ptr<MappingTarget> target = targetFor(attachmentMachine());
  ASSERT_NE(target, nullptr);
  MappingCandidate withMemory = coreCandidate();
  MemoryRequirement requirement;
  requirement.kind = "sram";
  withMemory.memoryRequirements.push_back(requirement);
  mlir::MLIRContext context;

  llvm::Expected<std::vector<CandidateInstance>> instances =
      enumeratePlacements(withMemory, *target, context, LayoutContext{});
  ASSERT_TRUE(static_cast<bool>(instances))
      << llvm::toString(instances.takeError());
  // core.0 sees two sram nodes; each is a legal attachment.
  ASSERT_EQ(instances->size(), 2u);
  EXPECT_EQ((*instances)[0].memoryBindings.lookup("sram"), "sram.a");
  EXPECT_EQ((*instances)[1].memoryBindings.lookup("sram"), "sram.b");
}

TEST(Placement, EnumeratesComputeMemoryAttachmentCombinations) {
  std::unique_ptr<MappingTarget> target = targetFor(attachmentMachine());
  ASSERT_NE(target, nullptr);
  MappingCandidate withAll = coreCandidate();
  ComputeRequirement computeRequirement;
  computeRequirement.kind = "vector_engine";
  withAll.computeRequirements.push_back(computeRequirement);
  MemoryRequirement memoryRequirement;
  memoryRequirement.kind = "sram";
  withAll.memoryRequirements.push_back(memoryRequirement);
  mlir::MLIRContext context;

  llvm::Expected<std::vector<CandidateInstance>> instances =
      enumeratePlacements(withAll, *target, context, LayoutContext{});
  ASSERT_TRUE(static_cast<bool>(instances))
      << llvm::toString(instances.takeError());
  // The two compute attachments and two memories cross to four instances, in
  // machine-declaration order with the last requirement varying fastest.
  ASSERT_EQ(instances->size(), 4u);
  EXPECT_EQ((*instances)[0].computeBindings.lookup("vector_engine"), "vec.a");
  EXPECT_EQ((*instances)[0].memoryBindings.lookup("sram"), "sram.a");
  EXPECT_EQ((*instances)[1].computeBindings.lookup("vector_engine"), "vec.a");
  EXPECT_EQ((*instances)[1].memoryBindings.lookup("sram"), "sram.b");
  EXPECT_EQ((*instances)[2].computeBindings.lookup("vector_engine"), "vec.b");
  EXPECT_EQ((*instances)[2].memoryBindings.lookup("sram"), "sram.a");
  EXPECT_EQ((*instances)[3].computeBindings.lookup("vector_engine"), "vec.b");
  EXPECT_EQ((*instances)[3].memoryBindings.lookup("sram"), "sram.b");
}

TEST(Placement, BindsTheSolvedLayoutForEveryEnumeratedAttachment) {
  std::unique_ptr<MappingTarget> target = targetFor(attachmentMachine());
  ASSERT_NE(target, nullptr);
  MappingCandidate withAll = coreCandidate();
  ComputeRequirement computeRequirement;
  computeRequirement.kind = "vector_engine";
  withAll.computeRequirements.push_back(computeRequirement);
  LayoutRequirement layoutRequirement;
  layoutRequirement.layoutClass = "t.rank2";
  withAll.layoutRequirements.push_back(layoutRequirement);
  mlir::MLIRContext context;
  LayoutContext layoutContext;
  layoutContext.rank = 2;

  llvm::Expected<std::vector<CandidateInstance>> instances =
      enumeratePlacements(withAll, *target, context, layoutContext);
  ASSERT_TRUE(static_cast<bool>(instances))
      << llvm::toString(instances.takeError());
  ASSERT_EQ(instances->size(), 2u);
  for (const CandidateInstance &instance : *instances) {
    // Each enumerated attachment binds the solved layout: the registry-resolved
    // definition id, identical to the requirement's declared id by
    // construction.
    const std::string bound = instance.layoutBindings.lookup("t.rank2");
    EXPECT_EQ(bound, "t.rank2");
    EXPECT_NE((*target).layouts().find(bound), nullptr);
  }
}

TEST(Placement, TheInstanceCapCapsEnumeratedAttachmentsAndReportsTruncation) {
  std::unique_ptr<MappingTarget> target = targetFor(attachmentMachine());
  ASSERT_NE(target, nullptr);
  MappingCandidate withAll = coreCandidate();
  ComputeRequirement computeRequirement;
  computeRequirement.kind = "vector_engine";
  withAll.computeRequirements.push_back(computeRequirement);
  MemoryRequirement memoryRequirement;
  memoryRequirement.kind = "sram";
  withAll.memoryRequirements.push_back(memoryRequirement);
  mlir::MLIRContext context;

  // Four legal combinations, capped at three: the cap genuinely stops
  // enumeration and must be reported.
  PlacementOptions capped;
  capped.maxInstances = 3;
  bool truncated = false;
  llvm::Expected<std::vector<CandidateInstance>> cappedInstances =
      enumeratePlacements(withAll, *target, context, LayoutContext{}, capped,
                          &truncated);
  ASSERT_TRUE(static_cast<bool>(cappedInstances))
      << llvm::toString(cappedInstances.takeError());
  EXPECT_EQ(cappedInstances->size(), 3u);
  EXPECT_TRUE(truncated);

  // A cap that admits every combination reports no truncation.
  PlacementOptions roomy;
  roomy.maxInstances = 8;
  bool notTruncated = false;
  llvm::Expected<std::vector<CandidateInstance>> all = enumeratePlacements(
      withAll, *target, context, LayoutContext{}, roomy, &notTruncated);
  ASSERT_TRUE(static_cast<bool>(all));
  EXPECT_EQ(all->size(), 4u);
  EXPECT_FALSE(notTruncated);
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
  // Every placed instance carries the solved layout: the binding is the layout
  // definition id the requirement resolved to, and it resolves in the target's
  // registry. A rule's `require layout p satisfies <id>` names that definition
  // id directly, so the bound value is string-identical to the requirement's
  // declared id by construction; the solved parameters are not carried
  // (`layoutBindings` is a `StringMap<LayoutId>`).
  const std::string bound = (*solvable)[0].layoutBindings.lookup("t.rank3");
  EXPECT_EQ(bound, "t.rank3");
  EXPECT_NE((*target).layouts().find(bound), nullptr);
}

TEST(Placement, TreatsATruncatedLayoutSolveAsUnplaceable) {
  // The domain is larger than the default quantifier budget, so the `forall`
  // is undecided and `!undecided` reads as satisfied -- the solve is truncated.
  // Placement must fail closed on the flag, not accept the undecided layout.
  constexpr llvm::StringLiteral kTruncating = R"llkmap(
layout t.truncating(int N) {
  param N in [1..200000];
  require !(forall v in domain(N) : v >= 1);
}
)llkmap";
  std::unique_ptr<MappingTarget> target =
      targetFor(placementMachine(), kTruncating);
  ASSERT_NE(target, nullptr);

  MappingCandidate withLayout = candidate();
  LayoutRequirement requirement;
  requirement.layoutClass = "t.truncating";
  withLayout.layoutRequirements.push_back(requirement);

  mlir::MLIRContext context;
  LayoutContext layoutContext;
  layoutContext.rank = 2;

  llvm::Expected<std::vector<CandidateInstance>> instances =
      enumeratePlacements(withLayout, *target, context, layoutContext);
  ASSERT_TRUE(static_cast<bool>(instances))
      << llvm::toString(instances.takeError());
  // The undecided solve yielded no trustworthy layout, so the candidate does
  // not place -- it is never accepted on the strength of `!undecided`.
  EXPECT_TRUE(instances->empty());
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

TEST(Placement, DeclaredEquivalenceCollapsesExecutorsToRepresentative) {
  // The target declares c.0 and c.1 equivalent even though their coordinates
  // and concurrency differ, so symmetry reduction collapses them to one
  // representative -- a declared group always keeps at least one member.
  std::unique_ptr<MappingTarget> target =
      targetFor(declaredEquivalentMachine());
  ASSERT_NE(target, nullptr);

  MappingCandidate core = candidate();
  core.executorRequirements[0].capability = "core";
  mlir::MLIRContext context;

  PlacementOptions reduced; // reduceSymmetry defaults on
  llvm::Expected<std::vector<CandidateInstance>> instances =
      enumeratePlacements(core, *target, context, LayoutContext{}, reduced);
  ASSERT_TRUE(static_cast<bool>(instances))
      << llvm::toString(instances.takeError());
  EXPECT_EQ(boundExecutors(*instances), (std::vector<std::string>{"c.0"}));
}

TEST(Placement, DeclaredEquivalenceRespectsTheSymmetrySwitch) {
  // The same declared-equivalent machine with reduction disabled enumerates
  // every representative, so the switch still governs the collapse.
  std::unique_ptr<MappingTarget> target =
      targetFor(declaredEquivalentMachine());
  ASSERT_NE(target, nullptr);

  MappingCandidate core = candidate();
  core.executorRequirements[0].capability = "core";
  mlir::MLIRContext context;

  PlacementOptions full;
  full.reduceSymmetry = false;
  llvm::Expected<std::vector<CandidateInstance>> instances =
      enumeratePlacements(core, *target, context, LayoutContext{}, full);
  ASSERT_TRUE(static_cast<bool>(instances))
      << llvm::toString(instances.takeError());
  EXPECT_EQ(boundExecutors(*instances),
            (std::vector<std::string>{"c.0", "c.1"}));
}

TEST(Placement, IgnoresCrossKindEquivalenceInAHandBuiltModel) {
  // A hand-built model bypasses `verifyMachineModel`, so placement must not
  // trust a cross-kind declaration: collapsing a `pe` onto a `worker` would
  // bind the wrong executor class. Both executors match the `worker` owner
  // (the `pe` refines it), but the kinds differ, so neither the declaration
  // nor the structural heuristic collapses them.
  MachineModel machine;
  machine.target = "hand-built";
  ExecutorNode worker;
  worker.id = "e0";
  worker.kind = "worker";
  worker.equivalentTo = {"e1"};
  ExecutorNode pe;
  pe.id = "e1";
  pe.kind = "pe";
  pe.refines = {"worker"};
  pe.equivalentTo = {"e0"};
  machine.executors = {worker, pe};
  std::unique_ptr<MappingTarget> target = targetFor(std::move(machine));
  ASSERT_NE(target, nullptr);

  MappingCandidate workerCandidate = candidate();
  workerCandidate.executorRequirements[0].capability = "worker";
  mlir::MLIRContext context;

  PlacementOptions reduced; // reduceSymmetry defaults on
  llvm::Expected<std::vector<CandidateInstance>> instances =
      enumeratePlacements(workerCandidate, *target, context, LayoutContext{},
                          reduced);
  ASSERT_TRUE(static_cast<bool>(instances))
      << llvm::toString(instances.takeError());
  EXPECT_EQ(boundExecutors(*instances), (std::vector<std::string>{"e0", "e1"}));
}

TEST(Placement, DoesNotCollapseExecutorsThatDifferInConcurrencyOrCoordinates) {
  // The structural heuristic must compare the facts that make two executors
  // behave differently: logical coordinates, concurrency, and scheduling class.
  // Executors that differ in any of them are not interchangeable, so collapsing
  // them to one representative would hide a distinct-performance placement.
  MachineModel machine;
  machine.target = "distinct";
  machine.executors = {
      {"c.0", "core", std::nullopt, {0}, 1, {}},
      {"c.1", "core", std::nullopt, {1}, 1, {}},
      {"c.2", "core", std::nullopt, {0}, 4, {}},
      {"c.3", "core", std::nullopt, {0}, 1, {}, SchedulingClass::OutOfOrder}};
  std::unique_ptr<MappingTarget> target = targetFor(std::move(machine));
  ASSERT_NE(target, nullptr);

  MappingCandidate core = candidate();
  core.executorRequirements[0].capability = "core";
  mlir::MLIRContext context;

  PlacementOptions reduced; // reduceSymmetry defaults on
  llvm::Expected<std::vector<CandidateInstance>> instances =
      enumeratePlacements(core, *target, context, LayoutContext{}, reduced);
  ASSERT_TRUE(static_cast<bool>(instances))
      << llvm::toString(instances.takeError());
  EXPECT_EQ(boundExecutors(*instances),
            (std::vector<std::string>{"c.0", "c.1", "c.2", "c.3"}));
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
