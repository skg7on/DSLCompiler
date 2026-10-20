//===- machine_model_loader.cpp - v2 YAML loader tests (issue #82) --------===//
//
// Part of the LLK Compiler Project, under the Apache License v2.0 with
// LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

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
    refines: [vector]
    attached_to: core.0
    element_types: [f32]
    shapes: [[8]]
    issue_cycles: 1
    latency_cycles: 5
    supported_layouts: [row_major, vectorized]
transfer_engines:
  - id: dma.0
    kind: dma
    refines: [transfer]
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
    refines: [vector]
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
// ARM SVE and SME profiles (issue #177)
//===----------------------------------------------------------------------===//

namespace {

llvm::Expected<MachineModel> loadShipped(llvm::StringRef name) {
  return loadMachineModel(std::string(LLK_MACHINE_DIR) + name.str());
}

/// Every link endpoint and transfer resource must name a node in the profile,
/// and every hop moves whole 64-byte transactions. A dangling reference would
/// only surface later as a routing failure, so the profiles are checked here.
void expectLinksResolve(const MachineModel &model) {
  ASSERT_FALSE(model.links.empty());
  for (const LinkEdge &link : model.links) {
    EXPECT_NE(model.findMemory(link.source), nullptr) << link.id;
    EXPECT_NE(model.findMemory(link.destination), nullptr) << link.id;
    EXPECT_EQ(link.transactionBytes, 64u) << link.id;
    EXPECT_FALSE(link.transferEngines.empty()) << link.id;
    for (const std::string &engine : link.transferEngines)
      EXPECT_NE(model.findTransferEngine(engine), nullptr) << link.id;
  }
}

} // namespace

TEST(MachineModelLoader, LoadsArmSve) {
  llvm::Expected<MachineModel> model = loadShipped("/arm-sve-v2.yaml");
  ASSERT_TRUE(static_cast<bool>(model)) << llvm::toString(model.takeError());
  EXPECT_EQ(model->target, "arm-sve");

  // A cluster group contains eight workers, each offering a 512-bit SVE scope.
  const ExecutorNode *cluster = model->findExecutor("cluster.0");
  ASSERT_NE(cluster, nullptr);
  EXPECT_EQ(cluster->kind, "cluster");
  EXPECT_TRUE(model->ownerMatches("group", "cluster.0"));
  EXPECT_EQ(cluster->concurrency, 1u);
  const ExecutorNode *worker = model->findExecutor("worker.0");
  ASSERT_NE(worker, nullptr);
  EXPECT_EQ(worker->parent, std::optional<std::string>("cluster.0"));
  EXPECT_EQ(worker->concurrency, 8u);
  const ExecutorNode *sve = model->findExecutor("sve_eng.0");
  ASSERT_NE(sve, nullptr);
  EXPECT_EQ(sve->kind, "sve_engine");
  EXPECT_EQ(sve->parent, std::optional<std::string>("worker.0"));
  EXPECT_EQ(sve->concurrency, 8u);
  // The target's own label resolves to an abstract class through `refines`.
  EXPECT_TRUE(model->ownerMatches("vector", "sve_eng.0"));

  // VL=512 makes 16 f32 / 32 bf16 / 64 i8 lanes.
  EXPECT_EQ(model->lanesFor("vector_engine", "f32"),
            std::optional<int64_t>(16));
  EXPECT_EQ(model->lanesFor("vector_engine", "bf16"),
            std::optional<int64_t>(32));
  EXPECT_EQ(model->lanesFor("vector_engine", "i8"), std::optional<int64_t>(64));

  const ComputeNode *svp = model->findCompute("svp");
  ASSERT_NE(svp, nullptr);
  EXPECT_EQ(svp->kind, "vector_engine");
  EXPECT_EQ(svp->attachedTo, "worker.0");
  EXPECT_EQ(svp->elementTypes, (std::vector<std::string>{"f32", "bf16", "i8"}));
  EXPECT_EQ(svp->accumulatorDTypes, (std::vector<std::string>{"f32"}));
  ASSERT_TRUE(svp->throughputPerCycle.has_value());
  EXPECT_DOUBLE_EQ(*svp->throughputPerCycle, 32.0);
  EXPECT_EQ(svp->issueCycles, 1u);
  EXPECT_EQ(svp->latencyCycles, 4u);
  EXPECT_EQ(svp->concurrency, 8u);

  // SVE is vector-only here, and the accumulator budget *is* the register
  // storage: a second `rf` pool would double-count it.
  EXPECT_TRUE(model->computesOfKind("matrix_engine").empty());
  EXPECT_EQ(model->findMemoryOfKind("rf"), nullptr);

  const MemoryNode *dram = model->findMemory("dram.0");
  ASSERT_NE(dram, nullptr);
  EXPECT_EQ(dram->capacityBytes, 68719476736u);
  EXPECT_DOUBLE_EQ(dram->bandwidthBytesPerCycle, 32.0);
  EXPECT_EQ(dram->latencyCycles, 220u);
  const MemoryNode *l2 = model->findMemory("l2.0");
  ASSERT_NE(l2, nullptr);
  EXPECT_EQ(l2->capacityBytes, 1048576u);
  EXPECT_DOUBLE_EQ(l2->bandwidthBytesPerCycle, 64.0);
  EXPECT_EQ(l2->latencyCycles, 12u);
  const MemoryNode *sram = model->findMemory("sram.0");
  ASSERT_NE(sram, nullptr);
  EXPECT_EQ(sram->capacityBytes, 65536u);
  EXPECT_DOUBLE_EQ(sram->bandwidthBytesPerCycle, 64.0);
  EXPECT_EQ(sram->latencyCycles, 4u);
  ASSERT_TRUE(sram->banks.has_value());
  EXPECT_EQ(*sram->banks, 1u);

  // 32 Z registers x 64 bytes, a modeled budget rather than a physical claim.
  const MemoryNode *acc = model->findMemory("acc.0");
  ASSERT_NE(acc, nullptr);
  EXPECT_EQ(acc->kind, "acc");
  EXPECT_EQ(acc->capacityBytes, 2048u);
  EXPECT_EQ(acc->alignmentBytes, 64u);
  ASSERT_TRUE(acc->banks.has_value());
  EXPECT_EQ(*acc->banks, 32u);
  EXPECT_DOUBLE_EQ(acc->bandwidthBytesPerCycle, 256.0);
  EXPECT_EQ(acc->latencyCycles, 1u);
  EXPECT_EQ(acc->supportedLayouts,
            (std::vector<std::string>{"row_major", "blocked"}));

  // Visibility declared on the worker is inherited by its descendants, and the
  // cluster-scoped memories remain reachable from the worker.
  EXPECT_TRUE(model->isVisible("dram.0", "worker.0"));
  EXPECT_TRUE(model->isVisible("l2.0", "worker.0"));
  EXPECT_TRUE(model->isVisible("sram.0", "worker.0"));
  EXPECT_TRUE(model->isVisible("acc.0", "worker.0"));

  const TransferEngineNode *dma = model->findTransferEngine("dma.0");
  ASSERT_NE(dma, nullptr);
  EXPECT_EQ(dma->attachedTo, "worker.0");
  EXPECT_EQ(dma->count, 1u);
  EXPECT_EQ(dma->maxOutstanding, 8u);
  EXPECT_EQ(dma->setupCycles, 0u);

  expectLinksResolve(*model);
  for (const char *id : {"dram_to_l2.0", "l2_to_sram.0", "dram_to_sram.0",
                         "sram_to_acc.0", "acc_to_sram.0", "sram_to_dram.0"})
    EXPECT_NE(model->findLink(id), nullptr) << id;
  EXPECT_EQ(model->findLink("sram_to_za.0"), nullptr);
  EXPECT_NE(computeContentHash(*model), 0u);
}

TEST(MachineModelLoader, LoadsArmSme) {
  llvm::Expected<MachineModel> model = loadShipped("/arm-sme-v2.yaml");
  ASSERT_TRUE(static_cast<bool>(model)) << llvm::toString(model.takeError());
  EXPECT_EQ(model->target, "arm-sme");

  // The common vector facts are unchanged: streaming mode does not widen the
  // ordinary vector lanes in this profile.
  EXPECT_EQ(model->lanesFor("vector_engine", "f32"),
            std::optional<int64_t>(16));
  EXPECT_EQ(model->lanesFor("vector_engine", "bf16"),
            std::optional<int64_t>(32));
  EXPECT_EQ(model->lanesFor("vector_engine", "i8"), std::optional<int64_t>(64));

  // Both refinements must resolve: the kernel can pick a vector or a matrix
  // owner, and neither path is hard-coded in the generic mapping code.
  const ExecutorNode *sve = model->findExecutor("sve_eng.0");
  ASSERT_NE(sve, nullptr);
  EXPECT_TRUE(model->ownerMatches("vector", "sve_eng.0"));
  const ExecutorNode *zaEng = model->findExecutor("za_eng.0");
  ASSERT_NE(zaEng, nullptr);
  EXPECT_EQ(zaEng->kind, "za_engine");
  EXPECT_EQ(zaEng->parent, std::optional<std::string>("worker.0"));
  EXPECT_TRUE(model->ownerMatches("matrix", "za_eng.0"));

  // One modeled 16x16x16 macro-work fragment, not one FMOPA instruction.
  std::vector<const ComputeNode *> matrices =
      model->computesOfKind("matrix_engine");
  ASSERT_EQ(matrices.size(), 1u);
  const ComputeNode *za = matrices[0];
  EXPECT_EQ(za->id, "za");
  EXPECT_EQ(za->attachedTo, "worker.0");
  EXPECT_EQ(za->elementTypes, (std::vector<std::string>{"f32", "bf16"}));
  EXPECT_EQ(za->accumulatorDTypes, (std::vector<std::string>{"f32"}));
  ASSERT_EQ(za->shapes.size(), 1u);
  EXPECT_EQ(za->shapes[0], (std::vector<int64_t>{16, 16, 16}));
  EXPECT_EQ(za->supportedLayouts,
            (std::vector<std::string>{"row_major", "blocked"}));
  ASSERT_TRUE(za->throughputPerCycle.has_value());
  EXPECT_DOUBLE_EQ(*za->throughputPerCycle, 512.0);
  EXPECT_EQ(za->issueCycles, 1u);
  EXPECT_EQ(za->latencyCycles, 5u);
  EXPECT_EQ(za->concurrency, 8u);

  // ZA capacity is (SVL/8)^2 = 4096 bytes; the bank count is an independent
  // declared seed, not a derivation from the four logical ZA.S tiles.
  EXPECT_EQ(model->findMemory("acc.0"), nullptr);
  const MemoryNode *zaMem = model->findMemory("za.0");
  ASSERT_NE(zaMem, nullptr);
  EXPECT_EQ(zaMem->kind, "acc");
  EXPECT_EQ(zaMem->capacityBytes, 4096u);
  EXPECT_EQ(zaMem->alignmentBytes, 64u);
  ASSERT_TRUE(zaMem->banks.has_value());
  EXPECT_EQ(*zaMem->banks, 4u);
  EXPECT_DOUBLE_EQ(zaMem->bandwidthBytesPerCycle, 256.0);
  EXPECT_EQ(zaMem->latencyCycles, 1u);
  EXPECT_EQ(zaMem->supportedLayouts,
            (std::vector<std::string>{"row_major", "blocked"}));
  EXPECT_TRUE(model->isVisible("za.0", "worker.0"));
  EXPECT_EQ(model->findMemoryOfKind("rf"), nullptr);

  expectLinksResolve(*model);
  EXPECT_NE(model->findLink("sram_to_za.0"), nullptr);
  EXPECT_NE(model->findLink("za_to_sram.0"), nullptr);
  EXPECT_EQ(model->findLink("sram_to_acc.0"), nullptr);
  EXPECT_EQ(model->findLink("acc_to_sram.0"), nullptr);
  EXPECT_NE(computeContentHash(*model), 0u);
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
    refines: [matrix]
    attached_to: e0
    element_types: [bf16]
    accumulator_dtypes: [f32]
    shapes: [[4, 8, 8]]
transfer_engines:
  - id: dma.0
    kind: dma
    refines: [transfer]
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
    refines: [vector]
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
    refines: [vector]
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
