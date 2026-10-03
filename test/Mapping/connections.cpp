//===- connections.cpp - Connection synthesis (D5) -----------------------===//

#include "LLK/Mapping/Placement.h"
#include "LLK/Mapping/Routing.h"

#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"

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

/// `(d0, d1) -> (d0, d1)`, the canonical identity relation.
mlir::AffineMap identity2(mlir::MLIRContext &context) {
  return mlir::AffineMap::getMultiDimIdentityMap(2, &context);
}

/// `(d0, d1) -> (d0 + 0, d1 * 1)`: equal to the identity only after MLIR
/// canonicalization. §10.2 requires comparing maps through that path, never by
/// string rendering.
mlir::AffineMap equivalentIdentity2(mlir::MLIRContext &context) {
  mlir::AffineExpr d0 = mlir::getAffineDimExpr(0, &context);
  mlir::AffineExpr d1 = mlir::getAffineDimExpr(1, &context);
  mlir::AffineExpr zero = mlir::getAffineConstantExpr(0, &context);
  mlir::AffineExpr one = mlir::getAffineConstantExpr(1, &context);
  return mlir::AffineMap::get(2, 0, {d0 + zero, d1 * one}, &context);
}

/// `(d0, d1) -> (d0 + 1, d1)`: a shifted window, not affinely equal to the
/// identity.
mlir::AffineMap shifted2(mlir::MLIRContext &context) {
  mlir::AffineExpr d0 = mlir::getAffineDimExpr(0, &context);
  mlir::AffineExpr d1 = mlir::getAffineDimExpr(1, &context);
  mlir::AffineExpr one = mlir::getAffineConstantExpr(1, &context);
  return mlir::AffineMap::get(2, 0, {d0 + one, d1}, &context);
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
// Direct compatibility (design §10.2): element type, logical tile shape,
// memory visibility, and affine index relation.
//===----------------------------------------------------------------------===//

TEST(Connections, RejectsElementTypeMismatch) {
  mlir::MLIRContext context;
  MachineModel machine = connectionMachine();
  TopologyService topology(machine);
  ConnectionRequest request = baseRequest();
  request.consumerMemory = request.producerMemory; // in place: nothing to move
  request.elementType = mlir::Float32Type::get(&context);
  request.consumerType = mlir::BFloat16Type::get(&context);

  llvm::Expected<std::vector<ConnectionPlan>> plans =
      synthesizeConnections(request, machine, topology);
  ASSERT_TRUE(static_cast<bool>(plans)) << llvm::toString(plans.takeError());
  EXPECT_TRUE(plans->empty());
}

TEST(Connections, RejectsShapedElementTypeMismatch) {
  mlir::MLIRContext context;
  MachineModel machine = connectionMachine();
  TopologyService topology(machine);
  ConnectionRequest request = baseRequest();
  request.consumerMemory = request.producerMemory;
  request.elementType =
      mlir::RankedTensorType::get({8, 8}, mlir::Float32Type::get(&context));
  request.consumerType =
      mlir::RankedTensorType::get({8, 8}, mlir::BFloat16Type::get(&context));

  llvm::Expected<std::vector<ConnectionPlan>> plans =
      synthesizeConnections(request, machine, topology);
  ASSERT_TRUE(static_cast<bool>(plans)) << llvm::toString(plans.takeError());
  EXPECT_TRUE(plans->empty());
}

TEST(Connections, RejectsLogicalTileShapeMismatch) {
  mlir::MLIRContext context;
  MachineModel machine = connectionMachine();
  TopologyService topology(machine);
  ConnectionRequest request = baseRequest();
  request.consumerMemory = request.producerMemory;
  request.elementType =
      mlir::RankedTensorType::get({8, 8}, mlir::Float32Type::get(&context));
  request.consumerType =
      mlir::RankedTensorType::get({4, 4}, mlir::Float32Type::get(&context));

  llvm::Expected<std::vector<ConnectionPlan>> plans =
      synthesizeConnections(request, machine, topology);
  ASSERT_TRUE(static_cast<bool>(plans)) << llvm::toString(plans.takeError());
  EXPECT_TRUE(plans->empty());
}

TEST(Connections, DirectWhenElementTypesAndShapesAgree) {
  mlir::MLIRContext context;
  MachineModel machine = connectionMachine();
  TopologyService topology(machine);
  ConnectionRequest request = baseRequest();
  request.consumerMemory = request.producerMemory;
  request.elementType =
      mlir::RankedTensorType::get({8, 8}, mlir::Float32Type::get(&context));
  request.consumerType =
      mlir::RankedTensorType::get({8, 8}, mlir::Float32Type::get(&context));

  llvm::Expected<std::vector<ConnectionPlan>> plans =
      synthesizeConnections(request, machine, topology);
  ASSERT_TRUE(static_cast<bool>(plans)) << llvm::toString(plans.takeError());
  ASSERT_EQ(plans->size(), 1u);
  EXPECT_EQ((*plans)[0].kind, ConnectionKind::Direct);
}

TEST(Connections, AcceptsAffinelyEquivalentMaps) {
  mlir::MLIRContext context;
  MachineModel machine = connectionMachine();
  TopologyService topology(machine);
  ConnectionRequest request = baseRequest();
  request.consumerMemory = request.producerMemory;
  request.producerMap = equivalentIdentity2(context);
  request.consumerMap = identity2(context);

  llvm::Expected<std::vector<ConnectionPlan>> plans =
      synthesizeConnections(request, machine, topology);
  ASSERT_TRUE(static_cast<bool>(plans)) << llvm::toString(plans.takeError());
  ASSERT_EQ(plans->size(), 1u);
  EXPECT_EQ((*plans)[0].kind, ConnectionKind::Direct);
}

TEST(Connections, RejectsAffinelyDifferentMaps) {
  mlir::MLIRContext context;
  MachineModel machine = connectionMachine();
  TopologyService topology(machine);
  ConnectionRequest request = baseRequest();
  request.consumerMemory = request.producerMemory;
  request.producerMap = identity2(context);
  request.consumerMap = shifted2(context);

  llvm::Expected<std::vector<ConnectionPlan>> plans =
      synthesizeConnections(request, machine, topology);
  ASSERT_TRUE(static_cast<bool>(plans)) << llvm::toString(plans.takeError());
  EXPECT_TRUE(plans->empty());
}

TEST(Connections, AcceptsMapsWhenOnlyOneEndpointDeclaresOne) {
  mlir::MLIRContext context;
  MachineModel machine = connectionMachine();
  TopologyService topology(machine);
  ConnectionRequest request = baseRequest();
  request.consumerMemory = request.producerMemory;
  request.producerMap = identity2(context); // consumer map unknown

  llvm::Expected<std::vector<ConnectionPlan>> plans =
      synthesizeConnections(request, machine, topology);
  ASSERT_TRUE(static_cast<bool>(plans)) << llvm::toString(plans.takeError());
  EXPECT_FALSE(plans->empty());
}

TEST(Connections, RejectsMemoryVisibilityMismatch) {
  MachineModel machine = connectionMachine();
  machine.executors.push_back({"e1", "worker", std::nullopt, {}, 1, {}});
  TopologyService topology(machine);
  ConnectionRequest request = baseRequest();
  request.consumerMemory = request.producerMemory; // in place
  request.producerExecutor = "e0";
  request.consumerExecutor = "e1"; // outside the memory's visibility scope

  llvm::Expected<std::vector<ConnectionPlan>> plans =
      synthesizeConnections(request, machine, topology);
  ASSERT_TRUE(static_cast<bool>(plans)) << llvm::toString(plans.takeError());
  EXPECT_TRUE(plans->empty());
}

TEST(Connections, DirectWhenTheConsumerSeesTheProducersMemory) {
  MachineModel machine = connectionMachine();
  machine.executors.push_back({"e1", "worker", std::nullopt, {}, 1, {}});
  TopologyService topology(machine);
  ConnectionRequest request = baseRequest();
  request.consumerMemory = request.producerMemory;
  request.producerExecutor = "e0";
  request.consumerExecutor = "e0";

  llvm::Expected<std::vector<ConnectionPlan>> plans =
      synthesizeConnections(request, machine, topology);
  ASSERT_TRUE(static_cast<bool>(plans)) << llvm::toString(plans.takeError());
  ASSERT_EQ(plans->size(), 1u);
  EXPECT_EQ((*plans)[0].kind, ConnectionKind::Direct);
}

TEST(Connections, VisibilityMismatchDoesNotForbidATransfer) {
  MachineModel machine = connectionMachine();
  machine.executors.push_back({"e1", "worker", std::nullopt, {}, 1, {}});
  // A memory only e1 can address, so the transfer has a legal destination even
  // though e1 cannot see the producer's memory.
  MemoryNode acc1 = memory("acc.1", "acc");
  acc1.visibleFrom = "e1";
  machine.memories.push_back(acc1);
  machine.links.push_back(link("dram_to_acc1.0", "dram.0", "acc.1"));
  TopologyService topology(machine);

  ConnectionRequest request = baseRequest();
  request.consumerMemory = "acc.1";
  request.producerExecutor = "e0";
  request.consumerExecutor = "e1";

  llvm::Expected<std::vector<ConnectionPlan>> plans =
      synthesizeConnections(request, machine, topology);
  ASSERT_TRUE(static_cast<bool>(plans)) << llvm::toString(plans.takeError());
  ASSERT_FALSE(plans->empty());
  for (const ConnectionPlan &plan : *plans)
    EXPECT_EQ(plan.kind, ConnectionKind::Transfer);
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
