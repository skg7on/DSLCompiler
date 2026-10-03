//===- connections.cpp - Connection synthesis (D5) -----------------------===//

#include "LLK/Mapping/Placement.h"
#include "LLK/Mapping/Routing.h"

#include "llvm/Support/Error.h"

#include <gtest/gtest.h>

#include <string>
#include <vector>

using namespace mlir::llk::machine;
using namespace mlir::llk::mapping;

namespace {

MemoryNode memory(llvm::StringRef id, llvm::StringRef kind) {
  MemoryNode node;
  node.id = id.str();
  node.kind = kind.str();
  node.visibleFrom = "e0";
  node.capacityBytes = 1u << 20;
  node.alignmentBytes = 64;
  return node;
}

LinkEdge link(llvm::StringRef id, llvm::StringRef source,
              llvm::StringRef destination) {
  LinkEdge edge;
  edge.id = id.str();
  edge.source = source.str();
  edge.destination = destination.str();
  edge.bandwidthBytesPerCycle = 32;
  edge.latencyCycles = 10;
  edge.transactionBytes = 64;
  edge.transferEngines = {"dma.0"};
  return edge;
}

/// dram can reach acc directly and in two hops through sram.
MachineModel connectionMachine() {
  MachineModel model;
  model.target = "connections";
  model.executors = {{"e0", "worker", std::nullopt, {}, 1, {}}};
  model.memories = {memory("dram.0", "dram"), memory("sram.0", "sram"),
                    memory("acc.0", "acc")};
  TransferEngineNode dma;
  dma.id = "dma.0";
  dma.kind = "dma";
  dma.attachedTo = "e0";
  model.transferEngines = {dma};
  model.links = {link("dram_to_acc.0", "dram.0", "acc.0"),
                 link("dram_to_sram.0", "dram.0", "sram.0"),
                 link("sram_to_acc.0", "sram.0", "acc.0")};
  return model;
}

ConnectionRequest baseRequest() {
  ConnectionRequest request;
  request.producer = 1;
  request.consumer = 2;
  request.value = 5;
  request.producerMemory = "dram.0";
  request.consumerMemory = "acc.0";
  request.producerLayout = "t.a";
  request.consumerLayout = "t.a";
  request.bytes = 1024;
  request.alignmentBytes = 32;
  return request;
}

} // namespace

TEST(Connections, DirectWhenMemoryAndLayoutAgree) {
  MachineModel machine = connectionMachine();
  TopologyService topology(machine);
  ConnectionRequest request = baseRequest();
  request.consumerMemory = request.producerMemory; // nothing to move

  llvm::Expected<std::vector<ConnectionPlan>> plans =
      synthesizeConnections(request, machine, topology);
  ASSERT_TRUE(static_cast<bool>(plans)) << llvm::toString(plans.takeError());
  ASSERT_EQ(plans->size(), 1u);
  EXPECT_EQ((*plans)[0].kind, ConnectionKind::Direct);
  EXPECT_TRUE((*plans)[0].transferEngines.empty());
  EXPECT_EQ((*plans)[0].memoryRoute.size(), 1u);
  EXPECT_EQ((*plans)[0].cost.latencyCycles, 0.0);
}

TEST(Connections, TransformOnlyWhenLayoutsDifferInPlace) {
  MachineModel machine = connectionMachine();
  TopologyService topology(machine);
  ConnectionRequest request = baseRequest();
  request.consumerMemory = request.producerMemory;
  request.consumerLayout = "t.b";

  llvm::Expected<std::vector<ConnectionPlan>> plans =
      synthesizeConnections(request, machine, topology);
  ASSERT_TRUE(static_cast<bool>(plans)) << llvm::toString(plans.takeError());
  ASSERT_EQ(plans->size(), 1u);
  EXPECT_EQ((*plans)[0].kind, ConnectionKind::LayoutTransform);
  EXPECT_TRUE((*plans)[0].transform.has_value());
  EXPECT_EQ((*plans)[0].transform->srcLayout, "t.a");
  EXPECT_EQ((*plans)[0].transform->dstLayout, "t.b");
}

TEST(Connections, TransferWhenOnlyMemoryDiffers) {
  MachineModel machine = connectionMachine();
  TopologyService topology(machine);
  ConnectionRequest request = baseRequest();

  llvm::Expected<std::vector<ConnectionPlan>> plans =
      synthesizeConnections(request, machine, topology);
  ASSERT_TRUE(static_cast<bool>(plans)) << llvm::toString(plans.takeError());
  ASSERT_FALSE(plans->empty());
  for (const ConnectionPlan &plan : *plans) {
    EXPECT_EQ(plan.kind, ConnectionKind::Transfer);
    EXPECT_FALSE(plan.transform.has_value());
    EXPECT_EQ(plan.producer, 1u);
    ASSERT_EQ(plan.consumers.size(), 1u);
    EXPECT_EQ(plan.consumers[0], 2u);
    EXPECT_EQ(plan.value, 5u);
  }
}

TEST(Connections, TransferAndTransformWhenBothDiffer) {
  MachineModel machine = connectionMachine();
  TopologyService topology(machine);
  ConnectionRequest request = baseRequest();
  request.consumerLayout = "t.b";

  llvm::Expected<std::vector<ConnectionPlan>> plans =
      synthesizeConnections(request, machine, topology);
  ASSERT_TRUE(static_cast<bool>(plans)) << llvm::toString(plans.takeError());
  ASSERT_FALSE(plans->empty());
  for (const ConnectionPlan &plan : *plans) {
    EXPECT_EQ(plan.kind, ConnectionKind::TransferAndTransform);
    ASSERT_TRUE(plan.transform.has_value());
  }
}

TEST(Connections, EmitsBothTheDirectAndTheTwoHopRoute) {
  MachineModel machine = connectionMachine();
  TopologyService topology(machine);
  ConnectionRequest request = baseRequest();

  llvm::Expected<std::vector<ConnectionPlan>> plans =
      synthesizeConnections(request, machine, topology);
  ASSERT_TRUE(static_cast<bool>(plans));
  ASSERT_EQ(plans->size(), 2u);
  // Cheapest first: the direct link beats the two-hop path.
  EXPECT_EQ((*plans)[0].memoryRoute.size(), 2u);
  EXPECT_EQ((*plans)[1].memoryRoute.size(), 3u);
  EXPECT_EQ((*plans)[1].memoryRoute[1], "sram.0");
  // Every hop's engine is recorded, so a cost model can see each one.
  EXPECT_FALSE((*plans)[1].transferEngines.empty());
}

TEST(Connections, NoRouteYieldsNoAlternatives) {
  MachineModel machine = connectionMachine();
  machine.links.clear();
  TopologyService topology(machine);
  ConnectionRequest request = baseRequest();

  llvm::Expected<std::vector<ConnectionPlan>> plans =
      synthesizeConnections(request, machine, topology);
  ASSERT_TRUE(static_cast<bool>(plans)) << llvm::toString(plans.takeError());
  EXPECT_TRUE(plans->empty());
}

TEST(Connections, UnknownMemoryIsAnError) {
  MachineModel machine = connectionMachine();
  TopologyService topology(machine);
  ConnectionRequest request = baseRequest();
  request.consumerMemory = "nowhere";

  llvm::Expected<std::vector<ConnectionPlan>> plans =
      synthesizeConnections(request, machine, topology);
  EXPECT_FALSE(static_cast<bool>(plans));
  if (!plans)
    llvm::consumeError(plans.takeError());
}

//===----------------------------------------------------------------------===//
// Fan-out and fan-in (design §15.3)
//===----------------------------------------------------------------------===//

TEST(Connections, FanOutSharesAReadWhenEveryConsumerReadsTheProducersMemory) {
  MachineModel machine = connectionMachine();
  TopologyService topology(machine);
  ConnectionRequest base = baseRequest();

  llvm::Expected<std::vector<ConnectionPlan>> plans = synthesizeFanOut(
      base, std::vector<InstanceId>{11, 12},
      std::vector<MemoryNodeId>{base.producerMemory, base.producerMemory},
      machine, topology);
  ASSERT_TRUE(static_cast<bool>(plans)) << llvm::toString(plans.takeError());
  ASSERT_EQ(plans->size(), 1u);
  EXPECT_EQ((*plans)[0].kind, ConnectionKind::Direct);
  ASSERT_EQ((*plans)[0].consumers.size(), 2u);
  EXPECT_EQ((*plans)[0].consumers[0], 11u);
  EXPECT_EQ((*plans)[0].consumers[1], 12u);
}

TEST(Connections, FanOutReplicatesWhenAConsumerNeedsADifferentMemory) {
  MachineModel machine = connectionMachine();
  TopologyService topology(machine);
  ConnectionRequest base = baseRequest();

  llvm::Expected<std::vector<ConnectionPlan>> plans =
      synthesizeFanOut(base, std::vector<InstanceId>{11, 12},
                       std::vector<MemoryNodeId>{base.producerMemory, "acc.0"},
                       machine, topology);
  ASSERT_TRUE(static_cast<bool>(plans)) << llvm::toString(plans.takeError());
  // Consumer 11 shares the producer's memory (direct); consumer 12 needs both
  // routes into acc.0. Nothing is shared, so every plan serves one consumer.
  ASSERT_EQ(plans->size(), 3u);
  size_t direct = 0;
  size_t transfers = 0;
  for (const ConnectionPlan &plan : *plans) {
    EXPECT_EQ(plan.consumers.size(), 1u);
    if (plan.kind == ConnectionKind::Direct)
      ++direct;
    else if (plan.kind == ConnectionKind::Transfer)
      ++transfers;
  }
  EXPECT_EQ(direct, 1u);
  EXPECT_EQ(transfers, 2u);
}

TEST(Connections, FanInProducesOneReducePlan) {
  ConnectionPlan plan =
      synthesizeFanIn(std::vector<InstanceId>{1, 2, 3}, /*consumer=*/9,
                      /*value=*/5, "acc.0", 1024);
  EXPECT_EQ(plan.kind, ConnectionKind::Reduce);
  ASSERT_EQ(plan.consumers.size(), 1u);
  EXPECT_EQ(plan.consumers[0], 9u);
  ASSERT_EQ(plan.producers.size(), 3u);
  EXPECT_EQ(plan.producers[2], 3u);
  EXPECT_NE(plan.id, 0u);
}
