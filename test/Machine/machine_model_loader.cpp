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
  // A minor bump stays readable by this major (design §11.6).
  std::string text = kValid.str();
  text.replace(text.find("llk.machine.v2"),
               std::string("llk.machine.v2").size(), "llk.machine.v2.1");
  EXPECT_TRUE(loads(text));
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
