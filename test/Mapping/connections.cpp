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

/// `(d0, d1) -> (d0 + d1 - d1, d1)`: affine-equal to the identity, but
/// *structurally distinct* from it. It must survive MLIR's construction-time
/// folding to do that, which is why the cancellation is split across a sum and
/// a difference: `d0 + 0` and `d1 * 1` are already collapsed by
/// `simplifyAdd`/`simplifyMul` as the expression is built, and so is a
/// same-operand `x + x - x`, so a map written either way is the identity before
/// `simplifyAffineMap` is ever consulted. Only a genuinely distinct map makes
/// the §10.2 comparison falsifiable: if the simplify step were removed, the raw
/// `operator==` would reject this pair.
mlir::AffineMap equivalentIdentity2(mlir::MLIRContext &context) {
  mlir::AffineExpr d0 = mlir::getAffineDimExpr(0, &context);
  mlir::AffineExpr d1 = mlir::getAffineDimExpr(1, &context);
  return mlir::AffineMap::get(2, 0, {d0 + d1 - d1, d1}, &context);
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

  // The whole point: the two maps must be *different objects* before
  // canonicalization, so accepting them can only come from simplifyAffineMap.
  ASSERT_NE(*request.producerMap, *request.consumerMap);
  EXPECT_EQ(mlir::simplifyAffineMap(*request.producerMap),
            mlir::simplifyAffineMap(*request.consumerMap));

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

//===----------------------------------------------------------------------===//
// Cost dimensions (design §17.2)
//===----------------------------------------------------------------------===//

// A route that touches the machine's DRAM-class memory records the bytes moved
// over the DRAM hop. The DRAM class is recognized from the memory node's
// declared `kind`, never from a hard-coded node id, so a profile may name its
// DRAM node anything.
TEST(Connections, DramRouteRecordsDramBytes) {
  MachineModel machine = connectionMachine();
  TopologyService topology(machine);
  ConnectionRequest request = baseRequest(); // dram.0 -> acc.0

  llvm::Expected<std::vector<ConnectionPlan>> plans =
      synthesizeConnections(request, machine, topology);
  ASSERT_TRUE(static_cast<bool>(plans)) << llvm::toString(plans.takeError());
  ASSERT_FALSE(plans->empty());
  for (const ConnectionPlan &plan : *plans) {
    ASSERT_EQ(plan.kind, ConnectionKind::Transfer);
    ASSERT_FALSE(plan.memoryRoute.empty());
    EXPECT_EQ(plan.memoryRoute.front(), "dram.0");
    EXPECT_EQ(plan.cost.dramBytes, request.bytes);
    // The staging bytes are the value itself.
    EXPECT_EQ(plan.cost.localBytes, request.bytes);
  }
}

// A route that never touches DRAM moves no DRAM bytes, but still stages the
// value locally.
TEST(Connections, NonDramRouteRecordsNoDramBytes) {
  MachineModel machine = connectionMachine();
  TopologyService topology(machine);
  ConnectionRequest request = baseRequest();
  request.producerMemory = "sram.0";
  request.consumerMemory = "acc.0";

  llvm::Expected<std::vector<ConnectionPlan>> plans =
      synthesizeConnections(request, machine, topology);
  ASSERT_TRUE(static_cast<bool>(plans)) << llvm::toString(plans.takeError());
  ASSERT_FALSE(plans->empty());
  EXPECT_EQ((*plans)[0].cost.dramBytes, 0u);
  EXPECT_EQ((*plans)[0].cost.localBytes, request.bytes);
}

// Spills are not modelled by the dialect, so the dimension is deliberately
// left unpopulated until a spill notion exists.
TEST(Connections, SpillBytesStayUnpopulated) {
  MachineModel machine = connectionMachine();
  TopologyService topology(machine);
  llvm::Expected<std::vector<ConnectionPlan>> plans =
      synthesizeConnections(baseRequest(), machine, topology);
  ASSERT_TRUE(static_cast<bool>(plans)) << llvm::toString(plans.takeError());
  ASSERT_FALSE(plans->empty());
  EXPECT_EQ((*plans)[0].cost.spillBytes, 0u);
}

// Transfer utilization is the route's transfer cycles over the cycles the
// machine's transfer engines had available in one sync period.
TEST(Connections, TransferUtilizationUsesTheSyncWindow) {
  MachineModel machine = connectionMachine();
  // No clockHz: it cancels in the dimensionless cycle ratio (see
  // utilizationEstimate), so the sync period is the only fact this needs.
  machine.sync.barrierCycles = 100; // one modelled sync period
  TopologyService topology(machine);
  ConnectionRequest request = baseRequest(); // dram.0 -> acc.0, 1024 bytes

  llvm::Expected<std::vector<ConnectionPlan>> plans =
      synthesizeConnections(request, machine, topology);
  ASSERT_TRUE(static_cast<bool>(plans)) << llvm::toString(plans.takeError());
  ASSERT_FALSE(plans->empty());
  // Cheapest route is the direct dram.0 -> acc.0 hop: 10 latency cycles plus
  // 1024 / 32 bytes-per-cycle = 42 cycles, over one engine x 100 cycles.
  EXPECT_EQ((*plans)[0].kind, ConnectionKind::Transfer);
  EXPECT_DOUBLE_EQ((*plans)[0].cost.transferUtilization, 0.42);
}

// Without a modelled sync period there is no denominator, so utilization stays
// 0 rather than a fabricated constant.
TEST(Connections, TransferUtilizationStaysZeroWithoutSyncFacts) {
  MachineModel machine = connectionMachine(); // sync costs are all zero
  TopologyService topology(machine);

  llvm::Expected<std::vector<ConnectionPlan>> plans =
      synthesizeConnections(baseRequest(), machine, topology);
  ASSERT_TRUE(static_cast<bool>(plans)) << llvm::toString(plans.takeError());
  ASSERT_FALSE(plans->empty());
  EXPECT_DOUBLE_EQ((*plans)[0].cost.transferUtilization, 0.0);
}
