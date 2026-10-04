//===- machine_model_loader.cpp - v2 YAML loader tests (issue #82) --------===//

#include "LLK/Machine/MachineModelLoader.h"

#include "llvm/Support/Error.h"

#include <gtest/gtest.h>

#include <string>

using namespace mlir::llk::machine;

namespace {

llvm::Expected<MachineModel> parse(llvm::StringRef text) {
  return parseMachineModel(text, "<test>");
}

bool loads(llvm::StringRef text) {
  llvm::Expected<MachineModel> model = parse(text);
  if (!model) {
    llvm::consumeError(model.takeError());
    return false;
  }
  return true;
}

constexpr llvm::StringLiteral kValid = R"yaml(
schema: llk.machine.v2
target: test-avx2
description: a two-core test machine
executors:
  - id: package.0
    kind: worker
  - id: core.0
    kind: core
    parent: package.0
    coordinates: [0]
    refines: [worker]
memories:
  - id: dram.0
    kind: dram
    visible_from: package.0
    capacity_bytes: 1073741824
    alignment_bytes: 64
    supported_layouts: [row_major]
  - id: sram.0
    kind: sram
    visible_from: core.0
    capacity_bytes: 32768
    alignment_bytes: 64
    supported_layouts: [row_major, blocked]
    banks: 8
compute:
  - id: vec.0
    kind: vector_engine
    attached_to: core.0
    element_types: [f32]
    shapes: [[8]]
    issue_cycles: 1
    latency_cycles: 5
    supported_layouts: [row_major, vectorized]
transfer_engines:
  - id: dma.0
    kind: dma
    attached_to: core.0
    count: 1
    max_outstanding: 8
links:
  - id: dram_to_sram.0
    source: dram.0
    destination: sram.0
    bandwidth_bytes_per_cycle: 32
    latency_cycles: 20
    transaction_bytes: 64
    transfer_engines: [dma.0]
)yaml";

} // namespace

TEST(MachineModelLoader, ParsesAValidDocument) {
  llvm::Expected<MachineModel> model = parse(kValid);
  ASSERT_TRUE(static_cast<bool>(model)) << llvm::toString(model.takeError());
  EXPECT_EQ(model->target, "test-avx2");
  EXPECT_EQ(model->schemaMajor, 2u);
  EXPECT_EQ(model->schemaMinor, 0u);
  ASSERT_EQ(model->executors.size(), 2u);
  ASSERT_EQ(model->memories.size(), 2u);
  ASSERT_EQ(model->computes.size(), 1u);
  ASSERT_EQ(model->transferEngines.size(), 1u);
  ASSERT_EQ(model->links.size(), 1u);
  EXPECT_EQ(model->executors[1].coordinates.size(), 1u);
  EXPECT_EQ(model->memories[1].banks.has_value() ? *model->memories[1].banks
                                                 : 0u,
            8u);
  EXPECT_TRUE(model->ownerMatches("worker", "core.0"));
  EXPECT_TRUE(model->isVisible("sram.0", "core.0"));
  EXPECT_NE(computeContentHash(*model), 0u);
}

TEST(MachineModelLoader, RejectsMissingSchema) {
  EXPECT_FALSE(loads("target: t\n"));
}

TEST(MachineModelLoader, RejectsUnsupportedMajor) {
  EXPECT_FALSE(loads("schema: llk.machine.v1\ntarget: t\n"));
  EXPECT_FALSE(loads("schema: llk.machine.v3\ntarget: t\n"));
}

TEST(MachineModelLoader, AcceptsMinorVersionSuffix) {
  // A minor bump stays readable by this major, and the declared minor is
  // preserved on the model rather than discarded (design §11.6).
  std::string text = kValid.str();
  text.replace(text.find("llk.machine.v2"),
               std::string("llk.machine.v2").size(), "llk.machine.v2.1");
  llvm::Expected<MachineModel> model = parse(text);
  ASSERT_TRUE(static_cast<bool>(model)) << llvm::toString(model.takeError());
  EXPECT_EQ(model->schemaMajor, 2u);
  EXPECT_EQ(model->schemaMinor, 1u);
}

TEST(MachineModelLoader, RejectsMalformedMinorVersion) {
  // A "." with no minor digits is not a version.
  EXPECT_FALSE(loads("schema: llk.machine.v2.\ntarget: t\n"));
  EXPECT_FALSE(loads("schema: llk.machine.v2.x\ntarget: t\n"));
}

TEST(MachineModelLoader, ToleratesUnknownKeysUnderANewerMinor) {
  // Design §11.6: a minor addition must be optional/defaulted, so a file at a
  // newer minor may carry keys this build does not know. Ignoring them keeps
  // the file loadable instead of forcing a loader edit per minor bump.
  std::string text = kValid.str();
  text.replace(text.find("llk.machine.v2"),
               std::string("llk.machine.v2").size(), "llk.machine.v2.1");
  text += "\nfuture_section: 7\n";
  EXPECT_TRUE(loads(text));

  // The same tolerance applies inside a node section.
  std::string nested = kValid.str();
  nested.replace(nested.find("llk.machine.v2"),
                 std::string("llk.machine.v2").size(), "llk.machine.v2.1");
  nested.replace(nested.find("kind: worker"),
                 std::string("kind: worker").size(),
                 "kind: worker\n    future_flag: 1");
  EXPECT_TRUE(loads(nested));
}

TEST(MachineModelLoader, RejectsUnknownKeysAtTheCurrentMinor) {
  // At the current minor an unknown key is a typo, not a forward-compatible
  // addition, so it stays an error -- this is the diagnostic §11.6 relies on.
  std::string text = kValid.str();
  text += "\nfuture_section: 7\n";
  EXPECT_FALSE(loads(text));

  std::string nested = kValid.str();
  nested.replace(nested.find("kind: worker"),
                 std::string("kind: worker").size(),
                 "kind: worker\n    future_flag: 1");
  EXPECT_FALSE(loads(nested));
}

TEST(MachineModelLoader, ANewerMinorStillRejectsMissingRequiredKeys) {
  // Tolerance covers *unknown* keys only: a newer minor may add optional keys,
  // but the required structure is unchanged, so a missing `target` still fails.
  // Non-vacuous: it would load if tolerantMinor_ also suppressed requireKey.
  EXPECT_FALSE(loads(R"yaml(
schema: llk.machine.v2.1
executors:
  - {id: e0, kind: worker}
)yaml"));
}

TEST(MachineModelLoader, ANewerMinorStillRejectsWrongNodeTypes) {
  // `executors` must be a sequence; a newer minor does not relax that.
  EXPECT_FALSE(loads(R"yaml(
schema: llk.machine.v2.1
target: t
executors:
  id: e0
  kind: worker
)yaml"));
}

TEST(MachineModelLoader, ANewerMinorStillRejectsMalformedKnownValues) {
  // A *known* key with a bad value is a type error, not an unknown key.
  EXPECT_FALSE(loads(R"yaml(
schema: llk.machine.v2.1
target: t
clock_hz: not-a-number
)yaml"));
}

TEST(MachineModelLoader, ANewerMinorStillRejectsUnknownKinds) {
  // Kind validation is vocabulary, not an unknown key: a newer minor cannot
  // smuggle in a capability this build does not model.
  EXPECT_FALSE(loads(R"yaml(
schema: llk.machine.v2.1
target: t
executors:
  - {id: e0, kind: worker}
compute:
  - {id: c0, kind: tensor_core, attached_to: e0, element_types: [f32], shapes: [[8]]}
)yaml"));
}

TEST(MachineModelLoader, RejectsUnknownRootKey) {
  std::string text = kValid.str();
  text += "\nnonsense: 1\n";
  EXPECT_FALSE(loads(text));
}

TEST(MachineModelLoader, RejectsUnknownNodeKey) {
  EXPECT_FALSE(loads(R"yaml(
schema: llk.machine.v2
target: t
executors:
  - id: e0
    kind: worker
    warp_size: 32
)yaml"));
}

TEST(MachineModelLoader, RejectsMissingRequiredField) {
  EXPECT_FALSE(loads(R"yaml(
schema: llk.machine.v2
target: t
executors:
  - kind: worker
)yaml"));
}

TEST(MachineModelLoader, RejectsWrongNodeType) {
  EXPECT_FALSE(loads(R"yaml(
schema: llk.machine.v2
target: t
executors:
  id: e0
  kind: worker
)yaml"));
}

TEST(MachineModelLoader, RejectsDuplicateIds) {
  EXPECT_FALSE(loads(R"yaml(
schema: llk.machine.v2
target: t
executors:
  - {id: e0, kind: worker}
memories:
  - {id: e0, kind: dram, visible_from: e0, capacity_bytes: 1024, alignment_bytes: 1}
)yaml"));
}

TEST(MachineModelLoader, RejectsDanglingReference) {
  EXPECT_FALSE(loads(R"yaml(
schema: llk.machine.v2
target: t
executors:
  - {id: e0, kind: worker}
compute:
  - {id: c0, kind: vector_engine, attached_to: nope, element_types: [f32], shapes: [[8]]}
)yaml"));
}

TEST(MachineModelLoader, RejectsUnknownComputeKindAtLoad) {
  EXPECT_FALSE(loads(R"yaml(
schema: llk.machine.v2
target: t
executors:
  - {id: e0, kind: worker}
compute:
  - {id: c0, kind: tensor_core, attached_to: e0, element_types: [f32], shapes: [[8]]}
)yaml"));
}

TEST(MachineModelLoader, RejectsUnknownTransferKindAtLoad) {
  EXPECT_FALSE(loads(R"yaml(
schema: llk.machine.v2
target: t
executors:
  - {id: e0, kind: worker}
transfer_engines:
  - {id: t0, kind: pcie, attached_to: e0}
)yaml"));
}

TEST(MachineModelLoader, ParsesDeclaredEquivalence) {
  llvm::Expected<MachineModel> model = parse(R"yaml(
schema: llk.machine.v2
target: t
executors:
  - {id: e0, kind: worker, equivalent_to: [e1]}
  - {id: e1, kind: worker, equivalent_to: [e0]}
)yaml");
  ASSERT_TRUE(static_cast<bool>(model)) << llvm::toString(model.takeError());
  ASSERT_NE(model->findExecutor("e0"), nullptr);
  ASSERT_NE(model->findExecutor("e1"), nullptr);
  EXPECT_EQ(model->findExecutor("e0")->equivalentTo,
            (std::vector<std::string>{"e1"}));
  EXPECT_EQ(model->findExecutor("e1")->equivalentTo,
            (std::vector<std::string>{"e0"}));
}

TEST(MachineModelLoader, RejectsEquivalenceWithUnknownExecutor) {
  EXPECT_FALSE(loads(R"yaml(
schema: llk.machine.v2
target: t
executors:
  - {id: e0, kind: worker, equivalent_to: [nope]}
)yaml"));
}

TEST(MachineModelLoader, RejectsAsymmetricEquivalence) {
  // e0 declares e1 equivalent, but e1 declares nothing: the assertion is not
  // mutual, so it is not a sound canonicalization and is rejected.
  EXPECT_FALSE(loads(R"yaml(
schema: llk.machine.v2
target: t
executors:
  - {id: e0, kind: worker, equivalent_to: [e1]}
  - {id: e1, kind: worker}
)yaml"));
}

TEST(MachineModelLoader, RejectsEquivalenceAcrossKinds) {
  EXPECT_FALSE(loads(R"yaml(
schema: llk.machine.v2
target: t
executors:
  - {id: e0, kind: worker, equivalent_to: [e1]}
  - {id: e1, kind: core, equivalent_to: [e0]}
)yaml"));
}

TEST(MachineModelLoader, RejectsSelfEquivalence) {
  EXPECT_FALSE(loads(R"yaml(
schema: llk.machine.v2
target: t
executors:
  - {id: e0, kind: worker, equivalent_to: [e0]}
)yaml"));
}

#ifndef LLK_MACHINE_DIR
#error "LLK_MACHINE_DIR must name the shipped machines directory"
#endif

TEST(MachineModelLoader, LoadsShippedAvx2Profile) {
  llvm::Expected<MachineModel> model =
      loadMachineModel(std::string(LLK_MACHINE_DIR) + "/x86-avx2-v2.yaml");
  ASSERT_TRUE(static_cast<bool>(model)) << llvm::toString(model.takeError());
  EXPECT_FALSE(model->executors.empty());
  EXPECT_FALSE(model->memories.empty());
  EXPECT_FALSE(model->computes.empty());
  EXPECT_FALSE(model->links.empty());
  // The profile exposes a scope a kernel's abstract `worker` owner can match.
  bool hasWorkerScope = false;
  for (const ExecutorNode &executor : model->executors)
    hasWorkerScope |= model->ownerMatches("worker", executor.id);
  EXPECT_TRUE(hasWorkerScope);
}

TEST(MachineModelLoader, ParsesComputeLanes) {
  llvm::Expected<MachineModel> model = parse(R"yaml(
schema: llk.machine.v2
target: t
executors:
  - {id: e0, kind: worker}
compute:
  - id: c0
    kind: vector_engine
    attached_to: e0
    element_types: [f32]
    shapes: [[8]]
    lanes: {f32: 8, bf16: 16}
)yaml");
  ASSERT_TRUE(static_cast<bool>(model)) << llvm::toString(model.takeError());
  ASSERT_TRUE(model->lanesFor("vector_engine", "f32").has_value());
  EXPECT_EQ(*model->lanesFor("vector_engine", "f32"), 8);
  EXPECT_EQ(*model->lanesFor("vector_engine", "bf16"), 16);
}

TEST(MachineModelLoader, LoadsShippedGenericProfile) {
  llvm::Expected<MachineModel> model = loadMachineModel(
      std::string(LLK_MACHINE_DIR) + "/generic-ai-accel-v2.yaml");
  ASSERT_TRUE(static_cast<bool>(model)) << llvm::toString(model.takeError());
  // A PE declares it refines `worker`, so a worker-owned kernel can map onto it
  // without the generic code knowing the word "pe".
  ASSERT_NE(model->findExecutor("pe.0"), nullptr);
  EXPECT_TRUE(model->ownerMatches("pe", "pe.0"));
  EXPECT_TRUE(model->ownerMatches("worker", "pe.0"));
  // The two-hop memory path the routing fixture will need is present.
  EXPECT_NE(model->findLink("dram_to_sram.0"), nullptr);
  EXPECT_NE(model->findLink("sram_to_acc.0"), nullptr);
}

//===----------------------------------------------------------------------===//
// Performance facts
//===----------------------------------------------------------------------===//

TEST(MachineModelLoader, ParsesClockThreadsAndSyncCosts) {
  llvm::Expected<MachineModel> model = parse(R"yaml(
schema: llk.machine.v2
target: t
clock_hz: 3000000000
worker_threads: 8
sync:
  barrier_cycles: 64
  wait_cycles: 4
executors:
  - {id: e0, kind: worker}
)yaml");
  ASSERT_TRUE(static_cast<bool>(model)) << llvm::toString(model.takeError());
  ASSERT_TRUE(model->clockHz.has_value());
  EXPECT_EQ(*model->clockHz, 3000000000u);
  EXPECT_EQ(model->workerThreads, 8u);
  EXPECT_EQ(model->sync.barrierCycles, 64u);
  EXPECT_EQ(model->sync.waitCycles, 4u);
}

TEST(MachineModelLoader, LeavesTheClockUnsetWhenAbsent) {
  llvm::Expected<MachineModel> model = parse(R"yaml(
schema: llk.machine.v2
target: t
executors:
  - {id: e0, kind: worker}
)yaml");
  ASSERT_TRUE(static_cast<bool>(model)) << llvm::toString(model.takeError());
  EXPECT_FALSE(model->clockHz.has_value());
  EXPECT_EQ(model->workerThreads, 1u); // a machine still executes somewhere
}

TEST(MachineModelLoader, RejectsUnknownSyncKey) {
  EXPECT_FALSE(loads(R"yaml(
schema: llk.machine.v2
target: t
sync:
  barrier_cycles: 1
  spin_cycles: 2
)yaml"));
}

TEST(MachineModelLoader, RejectsADeclaredZeroClock) {
  EXPECT_FALSE(loads(R"yaml(
schema: llk.machine.v2
target: t
clock_hz: 0
)yaml"));
}

TEST(MachineModelLoader, ParsesAccessCostsSetupCyclesAndAccumulatorDtypes) {
  llvm::Expected<MachineModel> model = parse(R"yaml(
schema: llk.machine.v2
target: t
executors:
  - {id: e0, kind: worker}
memories:
  - id: dram.0
    kind: dram
    visible_from: e0
    capacity_bytes: 1073741824
    bandwidth_bytes_per_cycle: 32
    latency_cycles: 220
compute:
  - id: fma.0
    kind: matrix_engine
    attached_to: e0
    element_types: [bf16]
    accumulator_dtypes: [f32]
    shapes: [[4, 8, 8]]
transfer_engines:
  - id: dma.0
    kind: dma
    attached_to: e0
    setup_cycles: 16
)yaml");
  ASSERT_TRUE(static_cast<bool>(model)) << llvm::toString(model.takeError());
  const MemoryNode *dram = model->findMemoryOfKind("dram");
  ASSERT_NE(dram, nullptr);
  EXPECT_DOUBLE_EQ(dram->bandwidthBytesPerCycle, 32.0);
  EXPECT_EQ(dram->latencyCycles, 220u);
  ASSERT_EQ(model->computes.size(), 1u);
  ASSERT_EQ(model->computes[0].accumulatorDTypes.size(), 1u);
  EXPECT_EQ(model->computes[0].accumulatorDTypes[0], "f32");
  ASSERT_EQ(model->transferEngines.size(), 1u);
  EXPECT_EQ(model->transferEngines[0].setupCycles, 16u);
}

TEST(MachineModelLoader, ShippedProfilesDeclareMemoryAccessCosts) {
  for (const char *name : {"/x86-avx2-v2.yaml", "/generic-ai-accel-v2.yaml"}) {
    llvm::Expected<MachineModel> model =
        loadMachineModel(std::string(LLK_MACHINE_DIR) + name);
    ASSERT_TRUE(static_cast<bool>(model)) << llvm::toString(model.takeError());
    const MemoryNode *dram = model->findMemoryOfKind("dram");
    ASSERT_NE(dram, nullptr) << name;
    EXPECT_GT(dram->bandwidthBytesPerCycle, 0.0) << name;
    EXPECT_GT(dram->latencyCycles, 0u) << name;
  }
}

TEST(MachineModelLoader, ShippedProfilesDeclareTheirPerfFacts) {
  for (const char *name : {"/x86-avx2-v2.yaml", "/generic-ai-accel-v2.yaml"}) {
    llvm::Expected<MachineModel> model =
        loadMachineModel(std::string(LLK_MACHINE_DIR) + name);
    ASSERT_TRUE(static_cast<bool>(model)) << llvm::toString(model.takeError());
    EXPECT_TRUE(model->clockHz.has_value()) << name;
    EXPECT_GT(model->workerThreads, 0u) << name;
  }
}

//===----------------------------------------------------------------------===//
// Design 11.3 properties added after the v2 model shipped
//===----------------------------------------------------------------------===//

TEST(MachineModelLoader, AppliesDocumentedDefaultsForTheAddedProperties) {
  llvm::Expected<MachineModel> model = parse(kValid);
  ASSERT_TRUE(static_cast<bool>(model)) << llvm::toString(model.takeError());
  for (const ExecutorNode &executor : model->executors)
    EXPECT_EQ(executor.schedulingClass, SchedulingClass::InOrder);
  for (const MemoryNode &memory : model->memories)
    EXPECT_FALSE(memory.accessGranularityBytes.has_value());
  for (const ComputeNode &compute : model->computes)
    EXPECT_FALSE(compute.occupancyLimit.has_value());
  for (const LinkEdge &link : model->links)
    EXPECT_EQ(link.directionality, LinkDirectionality::Unidirectional);
}

TEST(MachineModelLoader,
     ParsesSchedulingGranularityOccupancyAndDirectionality) {
  llvm::Expected<MachineModel> model = parse(R"yaml(
schema: llk.machine.v2
target: t
executors:
  - id: e0
    kind: worker
    scheduling_class: out_of_order
memories:
  - id: m0
    kind: sram
    visible_from: e0
    capacity_bytes: 32768
    access_granularity_bytes: 32
compute:
  - id: c0
    kind: vector_engine
    attached_to: e0
    element_types: [f32]
    shapes: [[8]]
    occupancy_limit: 4
links:
  - id: l0
    source: m0
    destination: m0
    bandwidth_bytes_per_cycle: 64
    directionality: bidirectional
)yaml");
  ASSERT_TRUE(static_cast<bool>(model)) << llvm::toString(model.takeError());
  ASSERT_EQ(model->executors.size(), 1u);
  EXPECT_EQ(model->executors[0].schedulingClass, SchedulingClass::OutOfOrder);
  ASSERT_EQ(model->memories.size(), 1u);
  ASSERT_TRUE(model->memories[0].accessGranularityBytes.has_value());
  EXPECT_EQ(*model->memories[0].accessGranularityBytes, 32u);
  ASSERT_EQ(model->computes.size(), 1u);
  ASSERT_TRUE(model->computes[0].occupancyLimit.has_value());
  EXPECT_EQ(*model->computes[0].occupancyLimit, 4u);
  ASSERT_EQ(model->links.size(), 1u);
  EXPECT_EQ(model->links[0].directionality, LinkDirectionality::Bidirectional);
}

TEST(MachineModelLoader, RejectsUnknownSchedulingClass) {
  EXPECT_FALSE(loads(R"yaml(
schema: llk.machine.v2
target: t
executors:
  - {id: e0, kind: worker, scheduling_class: warp_ordered}
)yaml"));
}

TEST(MachineModelLoader, RejectsUnknownLinkDirectionality) {
  EXPECT_FALSE(loads(R"yaml(
schema: llk.machine.v2
target: t
executors:
  - {id: e0, kind: worker}
memories:
  - {id: m0, kind: sram, visible_from: e0, capacity_bytes: 32768}
links:
  - {id: l0, source: m0, destination: m0, bandwidth_bytes_per_cycle: 64, directionality: omni}
)yaml"));
}

TEST(MachineModelLoader, RejectsZeroMemoryAccessGranularity) {
  EXPECT_FALSE(loads(R"yaml(
schema: llk.machine.v2
target: t
executors:
  - {id: e0, kind: worker}
memories:
  - {id: m0, kind: sram, visible_from: e0, capacity_bytes: 32768, access_granularity_bytes: 0}
)yaml"));
}

TEST(MachineModelLoader, RejectsZeroOccupancyLimit) {
  EXPECT_FALSE(loads(R"yaml(
schema: llk.machine.v2
target: t
executors:
  - {id: e0, kind: worker}
compute:
  - id: c0
    kind: vector_engine
    attached_to: e0
    element_types: [f32]
    shapes: [[8]]
    occupancy_limit: 0
)yaml"));
}

TEST(MachineModelLoader, AddedPropertiesChangeTheContentHash) {
  llvm::Expected<MachineModel> base = parse(kValid);
  ASSERT_TRUE(static_cast<bool>(base)) << llvm::toString(base.takeError());

  // Rewrite the worker executor entry in place instead of appending.
  std::string scheduling = kValid.str();
  scheduling.replace(scheduling.find("    kind: worker\n"),
                     std::string("    kind: worker\n").size(),
                     "    kind: worker\n    scheduling_class: out_of_order\n");
  llvm::Expected<MachineModel> scheduled = parse(scheduling);
  ASSERT_TRUE(static_cast<bool>(scheduled))
      << llvm::toString(scheduled.takeError());
  EXPECT_NE(base->contentHash, scheduled->contentHash);

  std::string withOccupancy = kValid.str();
  withOccupancy.replace(withOccupancy.find("    shapes: [[8]]\n"),
                        std::string("    shapes: [[8]]\n").size(),
                        "    shapes: [[8]]\n    occupancy_limit: 4\n");
  llvm::Expected<MachineModel> occupied = parse(withOccupancy);
  ASSERT_TRUE(static_cast<bool>(occupied))
      << llvm::toString(occupied.takeError());
  EXPECT_NE(base->contentHash, occupied->contentHash);
}
