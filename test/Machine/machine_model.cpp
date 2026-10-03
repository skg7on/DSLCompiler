//===- machine_model.cpp - Topology, queries, and content hash (v2) -------===//

#include "LLK/Machine/MachineModel.h"

#include "llvm/Support/Error.h"

#include <gtest/gtest.h>

#include <string>

using namespace mlir::llk::machine;

namespace {

/// True when the model passes verification.
bool verifies(const MachineModel &model) {
  if (llvm::Error error = verifyMachineModel(model)) {
    llvm::consumeError(std::move(error));
    return false;
  }
  return true;
}

MachineModel twoCoreMachine() {
  MachineModel model;
  model.target = "test";
  model.executors = {
      {"package.0", "worker", std::nullopt, {}, 1, {}},
      {"core.0", "core", std::string("package.0"), {0}, 1, {"worker"}},
      {"core.1", "core", std::string("package.0"), {1}, 1, {"worker"}}};
  model.memories = {
      {"dram.0",
       "dram",
       "package.0",
       1u << 30,
       64,
       {"row_major"},
       std::nullopt},
      {"sram.0", "sram", "core.0", 1u << 15, 64, {"row_major", "blocked"}, 8u}};
  model.computes = {{"avx2.0",
                     "vector_engine",
                     "core.0",
                     {"f32"},
                     {"row_major", "vectorized"},
                     {{8}},
                     {{"f32", 8}},
                     1,
                     5,
                     16.0,
                     1}};
  model.transferEngines = {{"dma.0", "dma", "core.0", 1, 8}};
  model.links = {
      {"dram_to_sram.0", "dram.0", "sram.0", 32.0, 20, 64, {"dma.0"}, 1}};
  return model;
}

} // namespace

TEST(MachineModel, ContainmentIsAncestorWalk) {
  MachineModel model = twoCoreMachine();
  EXPECT_TRUE(model.isWithin("core.0", "package.0"));
  EXPECT_FALSE(model.isWithin("package.0", "core.0"));
  EXPECT_TRUE(model.isWithin("core.0", "core.0"));
}

TEST(MachineModel, OwnerMatchesKindOrRefinement) {
  MachineModel model = twoCoreMachine();
  EXPECT_TRUE(model.ownerMatches("core", "core.0"));
  EXPECT_TRUE(model.ownerMatches("worker", "core.0")); // declared refines
  EXPECT_FALSE(model.ownerMatches("pe", "core.0"));
}

TEST(MachineModel, VisibilityFollowsContainment) {
  MachineModel model = twoCoreMachine();
  EXPECT_TRUE(model.isVisible("sram.0", "core.0"));
  EXPECT_FALSE(model.isVisible("sram.0", "core.1")); // visible_from core.0 only
  EXPECT_TRUE(model.isVisible("dram.0", "core.1"));  // visible_from package.0
}

TEST(MachineModel, AttachmentQueries) {
  MachineModel model = twoCoreMachine();
  ASSERT_EQ(model.computesFor("core.0").size(), 1u);
  EXPECT_EQ(model.computesFor("core.0")[0]->id, "avx2.0");
  EXPECT_TRUE(model.computesFor("core.1").empty());
  ASSERT_EQ(model.transferEnginesFor("core.0").size(), 1u);
  EXPECT_TRUE(model.transferEnginesFor("core.1").empty());
}

TEST(MachineModel, FindersReturnNullForUnknown) {
  MachineModel model = twoCoreMachine();
  EXPECT_NE(model.findExecutor("core.0"), nullptr);
  EXPECT_EQ(model.findExecutor("nope"), nullptr);
  EXPECT_NE(model.findMemory("dram.0"), nullptr);
  EXPECT_NE(model.findLink("dram_to_sram.0"), nullptr);
}

TEST(MachineModel, ContentHashIsOrderIndependent) {
  MachineModel a = twoCoreMachine();
  MachineModel b = twoCoreMachine();
  std::swap(b.executors[1], b.executors[2]);
  EXPECT_EQ(computeContentHash(a), computeContentHash(b));
}

TEST(MachineModel, ContentHashDistinguishesContent) {
  MachineModel a = twoCoreMachine();
  MachineModel b = twoCoreMachine();
  b.memories[0].capacityBytes += 1;
  EXPECT_NE(computeContentHash(a), computeContentHash(b));
}

TEST(MachineModel, VerifyAcceptsTheTwoCoreMachine) {
  EXPECT_TRUE(verifies(twoCoreMachine()));
}

TEST(MachineModel, VerifyRejectsUnknownSchemaMajor) {
  MachineModel model = twoCoreMachine();
  model.schemaMajor = 1;
  EXPECT_FALSE(verifies(model));
}

TEST(MachineModel, VerifyRejectsEmptyTarget) {
  MachineModel model = twoCoreMachine();
  model.target.clear();
  EXPECT_FALSE(verifies(model));
}

TEST(MachineModel, VerifyRejectsDuplicateIds) {
  MachineModel model = twoCoreMachine();
  model.memories[0].id = "core.0"; // collides with an executor
  EXPECT_FALSE(verifies(model));
}

TEST(MachineModel, VerifyRejectsUnknownExecutorKind) {
  MachineModel model = twoCoreMachine();
  model.executors[1].kind = "warp_group";
  EXPECT_FALSE(verifies(model));
}

TEST(MachineModel, VerifyRejectsUnknownComputeKind) {
  MachineModel model = twoCoreMachine();
  model.computes[0].kind = "tensor_core"; // not Micro owner vocabulary
  EXPECT_FALSE(verifies(model));
}

TEST(MachineModel, VerifyRejectsUnknownTransferKind) {
  MachineModel model = twoCoreMachine();
  model.transferEngines[0].kind = "pcie";
  EXPECT_FALSE(verifies(model));
}

TEST(MachineModel, VerifyAcceptsMatrixEngineComputeKind) {
  MachineModel model = twoCoreMachine();
  model.computes[0].kind = "matrix_engine"; // a Micro owner capability
  EXPECT_TRUE(verifies(model));
}

TEST(MachineModel, VerifyRejectsExecutorOwnerAsComputeKind) {
  MachineModel model = twoCoreMachine();
  // `core` is valid Micro owner vocabulary, but it is an execution scope, not
  // a compute capability a machine may attach.
  model.computes[0].kind = "core";
  EXPECT_FALSE(verifies(model));
}

TEST(MachineModel, VerifyRejectsMissingParent) {
  MachineModel model = twoCoreMachine();
  model.executors[1].parent = "nope.0";
  EXPECT_FALSE(verifies(model));
}

TEST(MachineModel, VerifyRejectsContainmentCycle) {
  MachineModel model = twoCoreMachine();
  model.executors[0].parent = "core.0"; // package.0 -> core.0 -> package.0
  EXPECT_FALSE(verifies(model));
}

TEST(MachineModel, VerifyRejectsZeroCapacity) {
  MachineModel model = twoCoreMachine();
  model.memories[0].capacityBytes = 0;
  EXPECT_FALSE(verifies(model));
}

TEST(MachineModel, VerifyRejectsUnknownMemoryKind) {
  MachineModel model = twoCoreMachine();
  model.memories[0].kind = "hbm";
  EXPECT_FALSE(verifies(model));
}

TEST(MachineModel, VerifyRejectsInvisibleFromNonExecutor) {
  MachineModel model = twoCoreMachine();
  model.memories[0].visibleFrom = "nope.0";
  EXPECT_FALSE(verifies(model));
}

TEST(MachineModel, VerifyRejectsDanglingAttachment) {
  MachineModel model = twoCoreMachine();
  model.computes[0].attachedTo = "nope.0";
  EXPECT_FALSE(verifies(model));
}

TEST(MachineModel, VerifyRejectsDanglingLink) {
  MachineModel model = twoCoreMachine();
  model.links[0].destination = "l1.9";
  EXPECT_FALSE(verifies(model));
}

TEST(MachineModel, VerifyRejectsZeroBandwidthLink) {
  MachineModel model = twoCoreMachine();
  model.links[0].bandwidthBytesPerCycle = 0.0;
  EXPECT_FALSE(verifies(model));
}

TEST(MachineModel, VerifyRejectsDanglingLinkEngine) {
  MachineModel model = twoCoreMachine();
  model.links[0].transferEngines = {"nope.0"};
  EXPECT_FALSE(verifies(model));
}

TEST(MachineModel, LanesForComputeKindAndElementType) {
  MachineModel model = twoCoreMachine();
  model.computes[0].lanes = {{"f32", 8}, {"bf16", 16}};
  ASSERT_TRUE(model.lanesFor("vector_engine", "f32").has_value());
  EXPECT_EQ(*model.lanesFor("vector_engine", "f32"), 8);
  EXPECT_EQ(*model.lanesFor("vector_engine", "bf16"), 16);
  // Unknown dtype and unknown capability both answer "not modelled".
  EXPECT_FALSE(model.lanesFor("vector_engine", "f64").has_value());
  EXPECT_FALSE(model.lanesFor("matrix_engine", "f32").has_value());
}

TEST(MachineModel, ContentHashCoversLanes) {
  MachineModel a = twoCoreMachine();
  a.computes[0].lanes = {{"f32", 8}};
  MachineModel b = twoCoreMachine();
  b.computes[0].lanes = {{"f32", 16}};
  EXPECT_NE(computeContentHash(a), computeContentHash(b));
}

TEST(MachineModel, ContentHashCoversClockAndSync) {
  MachineModel a = twoCoreMachine();
  MachineModel b = twoCoreMachine();
  b.clockHz = 2000000000;
  EXPECT_NE(computeContentHash(a), computeContentHash(b));

  MachineModel c = twoCoreMachine();
  c.sync.barrierCycles = 128;
  EXPECT_NE(computeContentHash(a), computeContentHash(c));
}

TEST(MachineModel, VerifyRejectsADeclaredZeroClock) {
  MachineModel model = twoCoreMachine();
  model.clockHz = 0; // declared, and meaningless for a cycle estimate
  EXPECT_FALSE(verifies(model));
}

TEST(MachineModel, VerifyAcceptsAModelThatDoesNotDeclareAClock) {
  MachineModel model = twoCoreMachine();
  model.clockHz.reset();
  EXPECT_TRUE(verifies(model));
}

TEST(MachineModel, VerifyRejectsZeroWorkerThreads) {
  MachineModel model = twoCoreMachine();
  model.workerThreads = 0;
  EXPECT_FALSE(verifies(model));
}

//===----------------------------------------------------------------------===//
// Cost facts the performance simulator needs
//===----------------------------------------------------------------------===//

TEST(MachineModel, OwnerCountCountsExecutorsThatMatchTheKind) {
  MachineModel model = twoCoreMachine();
  // package.0 is `worker`; core.0 and core.1 refine it.
  EXPECT_EQ(model.ownerCount("worker"), 3u);
  EXPECT_EQ(model.ownerCount("core"), 2u);
  EXPECT_EQ(model.ownerCount("pe"), 0u);
}

TEST(MachineModel, FindMemoryOfKindReturnsTheFirstDeclaration) {
  MachineModel model = twoCoreMachine();
  ASSERT_NE(model.findMemoryOfKind("dram"), nullptr);
  EXPECT_EQ(model.findMemoryOfKind("dram")->id, "dram.0");
  EXPECT_EQ(model.findMemoryOfKind("sram")->id, "sram.0");
  EXPECT_EQ(model.findMemoryOfKind("hbm"), nullptr);
}

TEST(MachineModel, ContentHashCoversMemoryAccessCost) {
  MachineModel a = twoCoreMachine();
  MachineModel b = twoCoreMachine();
  b.memories[0].bandwidthBytesPerCycle = 128;
  EXPECT_NE(computeContentHash(a), computeContentHash(b));

  MachineModel c = twoCoreMachine();
  c.memories[0].latencyCycles = 7;
  EXPECT_NE(computeContentHash(a), computeContentHash(c));
}
