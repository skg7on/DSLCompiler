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
  // The layouts `baseRequest` declares. Routing now checks the producer's
  // layout against every hop memory, so a memory that declares none would
  // reject the movement these tests exercise.
  node.supportedLayouts = {"t.a", "t.b"};
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

/// e0 owns dram.0 and e1 owns acc.0; neither sees the other's memory. A
/// consumer placed on acc.0 therefore cannot read the producer's dram.0 in
/// place, so serving it means a real copy over the single dram.0 -> acc.0 link:
/// replication, not a direct read.
MachineModel replicationMachine() {
  MachineModel model;
  model.target = "replication";
  model.executors = {{"e0", "worker", std::nullopt, {}, 1, {}},
                     {"e1", "worker", std::nullopt, {}, 1, {}}};
  MemoryNode dram = memory("dram.0", "dram");
  dram.visibleFrom = "e0";
  MemoryNode acc = memory("acc.0", "acc");
  acc.visibleFrom = "e1";
  model.memories = {dram, acc};
  TransferEngineNode dma;
  dma.id = "dma.0";
  dma.kind = "dma";
  dma.attachedTo = "e0";
  model.transferEngines = {dma};
  model.links = {link("dram_to_acc.0", "dram.0", "acc.0")};
  return model;
}

/// `baseRequest` against `replicationMachine`: the producer runs on e0/dram.0
/// and the consumer on e1/acc.0, so no direct read is possible.
ConnectionRequest replicationRequest() {
  ConnectionRequest request = baseRequest();
  request.producerMemory = "dram.0";
  request.consumerMemory = "acc.0";
  request.producerExecutor = "e0";
  request.consumerExecutor = "e1";
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

/// A linear machine with no shortcut: `dram.0 -> sram.0 -> acc.0`. The
/// producer's layout is supported only by the source memory and the consumer's
/// only by the memories after the first hop, so a plan-wide layout rule would
/// reject the route. Placing the transform at the first hop makes it legal.
MachineModel transformHopMachine() {
  MachineModel model;
  model.target = "transform-hop";
  model.executors = {{"e0", "worker", std::nullopt, {}, 1, {}}};
  MemoryNode dram = memory("dram.0", "dram");
  dram.supportedLayouts = {"t.a"};
  MemoryNode sram = memory("sram.0", "sram");
  sram.supportedLayouts = {"t.b"};
  MemoryNode acc = memory("acc.0", "acc");
  acc.supportedLayouts = {"t.b"};
  model.memories = {dram, sram, acc};
  TransferEngineNode dma;
  dma.id = "dma.0";
  dma.kind = "dma";
  dma.attachedTo = "e0";
  model.transferEngines = {dma};
  model.links = {link("dram_to_sram.0", "dram.0", "sram.0"),
                 link("sram_to_acc.0", "sram.0", "acc.0")};
  return model;
}

/// The §15.2 attempt a plan belongs to: 1 direct, 2 layout-only transform in a
/// mutually visible memory, 3 direct transfer, 4 transfer plus transform, 5
/// bounded multi-hop transfer. A multi-hop route is one with an intermediate
/// memory (more than two nodes).
unsigned sectionRank(const ConnectionPlan &plan) {
  switch (plan.kind) {
  case ConnectionKind::Direct:
    return 1;
  case ConnectionKind::LayoutTransform:
    return 2;
  case ConnectionKind::Transfer:
    return plan.memoryRoute.size() <= 2 ? 3 : 5;
  case ConnectionKind::TransferAndTransform:
    return plan.memoryRoute.size() <= 2 ? 4 : 5;
  case ConnectionKind::Replicate:
  case ConnectionKind::Reduce:
    return 6;
  }
  return 6;
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

TEST(Connections, DirectReadAndTransferRoutesWhenOnlyMemoryDiffers) {
  // §10.2 defines direct compatibility by memory *visibility*, not by the two
  // memories being the same node: a consumer that can address the producer's
  // memory reads the value there with no transfer. So a different-memory,
  // same-layout pair exposes a Direct alternative *and* the transfer routes.
  // (The previous version of this test asserted every plan was a Transfer,
  // enshrining the memory-id-equality reading.)
  MachineModel machine = connectionMachine();
  TopologyService topology(machine);
  ConnectionRequest request = baseRequest();

  llvm::Expected<std::vector<ConnectionPlan>> plans =
      synthesizeConnections(request, machine, topology);
  ASSERT_TRUE(static_cast<bool>(plans)) << llvm::toString(plans.takeError());
  ASSERT_FALSE(plans->empty());
  bool direct = false;
  size_t transfers = 0;
  for (const ConnectionPlan &plan : *plans) {
    EXPECT_FALSE(plan.transform.has_value());
    EXPECT_EQ(plan.producer, 1u);
    ASSERT_EQ(plan.consumers.size(), 1u);
    EXPECT_EQ(plan.consumers[0], 2u);
    EXPECT_EQ(plan.value, 5u);
    if (plan.kind == ConnectionKind::Direct) {
      direct = true;
      EXPECT_EQ(plan.memoryRoute.size(), 1u);
      EXPECT_EQ(plan.memoryRoute[0], "dram.0");
    } else {
      EXPECT_EQ(plan.kind, ConnectionKind::Transfer);
      ++transfers;
    }
  }
  EXPECT_TRUE(direct);
  EXPECT_EQ(transfers, 2u);
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
  // Different memories and different layouts: the route alternatives are
  // transfer-plus-transform. A layout-only transform in the producer's memory
  // (alternative 2) is also legal here and appears alongside them, so it is
  // skipped when checking the transfer routes.
  size_t transferAndTransform = 0;
  for (const ConnectionPlan &plan : *plans) {
    if (plan.kind == ConnectionKind::LayoutTransform)
      continue;
    EXPECT_EQ(plan.kind, ConnectionKind::TransferAndTransform);
    ASSERT_TRUE(plan.transform.has_value());
    ++transferAndTransform;
  }
  EXPECT_GE(transferAndTransform, 2u);
}

// (a) Same memory, different layouts: a direct connection is impossible and a
// layout-only transform in that shared memory is the alternative.
TEST(Connections, SameMemoryDifferentLayoutsCannotConnectDirectly) {
  MachineModel machine = connectionMachine();
  TopologyService topology(machine);
  ConnectionRequest request = baseRequest();
  request.consumerMemory = request.producerMemory; // one shared memory
  request.consumerLayout = "t.b";

  EXPECT_FALSE(portsDirectCompatible(request, machine));

  llvm::Expected<std::vector<ConnectionPlan>> plans =
      synthesizeConnections(request, machine, topology);
  ASSERT_TRUE(static_cast<bool>(plans)) << llvm::toString(plans.takeError());
  ASSERT_EQ(plans->size(), 1u);
  EXPECT_EQ((*plans)[0].kind, ConnectionKind::LayoutTransform);
  ASSERT_TRUE((*plans)[0].transform.has_value());
}

// (b) Different memories that both endpoints can reach: alongside the transfer
// routes, a layout-only transform can run in the producer's memory and the
// consumer can read the result there without a transfer.
TEST(Connections, DifferentMemoriesYieldTransferAndLayoutOnlyTransform) {
  MachineModel machine = connectionMachine();
  TopologyService topology(machine);
  ConnectionRequest request = baseRequest(); // dram.0 -> acc.0
  request.producerExecutor = "e0";
  request.consumerExecutor = "e0";
  request.consumerLayout = "t.b";

  llvm::Expected<std::vector<ConnectionPlan>> plans =
      synthesizeConnections(request, machine, topology);
  ASSERT_TRUE(static_cast<bool>(plans)) << llvm::toString(plans.takeError());

  bool layoutOnly = false;
  bool transfer = false;
  for (const ConnectionPlan &plan : *plans) {
    if (plan.kind == ConnectionKind::LayoutTransform) {
      layoutOnly = true;
      // No transfer: the transform runs where the value already is.
      ASSERT_EQ(plan.memoryRoute.size(), 1u);
      EXPECT_EQ(plan.memoryRoute[0], "dram.0");
      ASSERT_TRUE(plan.transform.has_value());
    } else {
      transfer = true;
    }
  }
  EXPECT_TRUE(layoutOnly);
  EXPECT_TRUE(transfer);
}

// (c) A multi-hop transfer whose transform is placed at a legal hop: the
// producer's layout is supported only by the source, the consumer's by every
// memory after it, so a plan-wide layout rule would reject the route. The hop
// boundary carries the transform.
TEST(Connections, MultiHopPlacesTheTransformAtALegalHop) {
  MachineModel machine = transformHopMachine();
  TopologyService topology(machine);
  ConnectionRequest request = baseRequest(); // dram.0 -> acc.0
  request.producerExecutor = "e0";
  request.consumerExecutor = "e0";
  request.producerLayout = "t.a";
  request.consumerLayout = "t.b";

  llvm::Expected<std::vector<ConnectionPlan>> plans =
      synthesizeConnections(request, machine, topology);
  ASSERT_TRUE(static_cast<bool>(plans)) << llvm::toString(plans.takeError());
  ASSERT_EQ(plans->size(), 1u);
  EXPECT_EQ((*plans)[0].kind, ConnectionKind::TransferAndTransform);
  ASSERT_EQ((*plans)[0].memoryRoute.size(), 3u);
  EXPECT_EQ((*plans)[0].memoryRoute[1], "sram.0");
  ASSERT_TRUE((*plans)[0].transform.has_value());
  EXPECT_EQ((*plans)[0].transform->srcLayout, "t.a");
  EXPECT_EQ((*plans)[0].transform->dstLayout, "t.b");
}

// (d) The five alternatives are attempted in the §15.2 order: direct, layout
// transform, direct transfer, transfer plus transform, multi-hop transfer.
TEST(Connections, AlternativesAreAttemptedInSection15Order) {
  MachineModel machine = connectionMachine();
  TopologyService topology(machine);

  auto ranksOf = [&](const ConnectionRequest &request) {
    llvm::Expected<std::vector<ConnectionPlan>> plans =
        synthesizeConnections(request, machine, topology);
    EXPECT_TRUE(static_cast<bool>(plans));
    std::vector<unsigned> ranks;
    if (plans)
      for (const ConnectionPlan &plan : *plans)
        ranks.push_back(sectionRank(plan));
    return ranks;
  };

  // Same memory, same layout: alternative 1 only.
  {
    ConnectionRequest request = baseRequest();
    request.consumerMemory = request.producerMemory;
    EXPECT_EQ(ranksOf(request), (std::vector<unsigned>{1u}));
  }
  // Same memory, different layout: alternative 2 only.
  {
    ConnectionRequest request = baseRequest();
    request.consumerMemory = request.producerMemory;
    request.consumerLayout = "t.b";
    EXPECT_EQ(ranksOf(request), (std::vector<unsigned>{2u}));
  }
  // Different memory, same layout: the direct read (alternative 1) precedes a
  // direct transfer (3) and a multi-hop transfer (5).
  {
    ConnectionRequest request = baseRequest();
    EXPECT_EQ(ranksOf(request), (std::vector<unsigned>{1u, 3u, 5u}));
  }
  // Different memory, different layout: the layout-only transform precedes the
  // direct transfer-plus-transform, which precedes the multi-hop one.
  {
    ConnectionRequest request = baseRequest();
    request.producerExecutor = "e0";
    request.consumerExecutor = "e0";
    request.consumerLayout = "t.b";
    EXPECT_EQ(ranksOf(request), (std::vector<unsigned>{2u, 4u, 5u}));
  }
}

TEST(Connections, EmitsTheDirectReadAndBothTransferRoutes) {
  MachineModel machine = connectionMachine();
  TopologyService topology(machine);
  ConnectionRequest request = baseRequest();

  llvm::Expected<std::vector<ConnectionPlan>> plans =
      synthesizeConnections(request, machine, topology);
  ASSERT_TRUE(static_cast<bool>(plans));
  ASSERT_EQ(plans->size(), 3u);
  // Alternative 1 first: the consumer reads the producer's memory directly.
  EXPECT_EQ((*plans)[0].kind, ConnectionKind::Direct);
  EXPECT_EQ((*plans)[0].memoryRoute.size(), 1u);
  // Then the transfer routes, cheapest first: the direct link beats the
  // two-hop path.
  EXPECT_EQ((*plans)[1].kind, ConnectionKind::Transfer);
  EXPECT_EQ((*plans)[1].memoryRoute.size(), 2u);
  EXPECT_EQ((*plans)[2].kind, ConnectionKind::Transfer);
  EXPECT_EQ((*plans)[2].memoryRoute.size(), 3u);
  EXPECT_EQ((*plans)[2].memoryRoute[1], "sram.0");
  // Every hop's engine is recorded, so a cost model can see each one.
  EXPECT_FALSE((*plans)[2].transferEngines.empty());
}

// A direct read needs no route, so clearing the links cannot remove it -- the
// pair is not incompatible.
TEST(Connections, NoRouteStillYieldsADirectReadWhenVisibilityAllows) {
  MachineModel machine = connectionMachine();
  machine.links.clear();
  TopologyService topology(machine);
  ConnectionRequest request = baseRequest();

  llvm::Expected<std::vector<ConnectionPlan>> plans =
      synthesizeConnections(request, machine, topology);
  ASSERT_TRUE(static_cast<bool>(plans)) << llvm::toString(plans.takeError());
  ASSERT_EQ(plans->size(), 1u);
  EXPECT_EQ((*plans)[0].kind, ConnectionKind::Direct);
}

// No route and no direct read: the pair is incompatible, which is an empty
// result, never an error.
TEST(Connections, NoRouteAndNoDirectReadYieldsNoAlternatives) {
  MachineModel machine = connectionMachine();
  machine.links.clear();
  machine.executors.push_back({"e1", "worker", std::nullopt, {}, 1, {}});
  // A destination only e1 can address. The route is gone (links cleared) and
  // the consumer cannot address the producer's memory, so there is no direct
  // read either: the pair is genuinely incompatible.
  MemoryNode acc1 = memory("acc.1", "acc");
  acc1.visibleFrom = "e1";
  machine.memories.push_back(acc1);
  TopologyService topology(machine);
  ConnectionRequest request = baseRequest();
  request.consumerMemory = "acc.1";
  request.producerExecutor = "e0";
  request.consumerExecutor = "e1"; // cannot address the producer's memory

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
  // Consumer 11 shares the producer's memory (one direct read). Consumer 12 is
  // placed on acc.0: it can read the producer's memory directly *and* it has
  // both transfer routes into acc.0. The transfer routes are copies, so they
  // are `Replicate`; the shared-memory read stays `Direct`.
  ASSERT_EQ(plans->size(), 4u);
  size_t direct = 0;
  size_t replicates = 0;
  for (const ConnectionPlan &plan : *plans) {
    EXPECT_EQ(plan.consumers.size(), 1u);
    if (plan.kind == ConnectionKind::Direct)
      ++direct;
    else if (plan.kind == ConnectionKind::Replicate)
      ++replicates;
  }
  EXPECT_EQ(direct, 2u);
  EXPECT_EQ(replicates, 2u);
}

// Two consumers that must be served from a memory their executors cannot read
// in place: every consumer gets a copy, and each copy is labelled `Replicate`.
// Replication multiplies the transfer cost, one transfer per consumer.
TEST(Connections, FanOutReplicatesWhenConsumersNeedDistinctPlacements) {
  MachineModel machine = replicationMachine();
  TopologyService topology(machine);
  ConnectionRequest base = replicationRequest();

  llvm::Expected<std::vector<ConnectionPlan>> plans = synthesizeFanOut(
      base, std::vector<InstanceId>{11, 12},
      std::vector<MemoryNodeId>{"acc.0", "acc.0"}, machine, topology);
  ASSERT_TRUE(static_cast<bool>(plans)) << llvm::toString(plans.takeError());
  ASSERT_EQ(plans->size(), 2u);
  const double perCopy = 10.0 + 1024.0 / 32.0; // link latency + bytes/bandwidth
  double total = 0.0;
  for (const ConnectionPlan &plan : *plans) {
    EXPECT_EQ(plan.kind, ConnectionKind::Replicate);
    ASSERT_EQ(plan.consumers.size(), 1u);
    ASSERT_FALSE(plan.memoryRoute.empty());
    EXPECT_EQ(plan.memoryRoute.back(), "acc.0");
    EXPECT_DOUBLE_EQ(plan.cost.latencyCycles, perCopy);
    total += plan.cost.latencyCycles;
  }
  // Two copies cost two transfers.
  EXPECT_DOUBLE_EQ(total, 2.0 * perCopy);
}

// The fan-out route cap must reach the caller: a route enumeration that hit
// `maxRoutesPerConnection` inside replication sets the shared `truncated`
// out-parameter rather than silently dropping alternatives.
TEST(Connections, FanOutReportsRouteCapThroughTruncated) {
  MachineModel machine = replicationMachine();
  TopologyService topology(machine);
  ConnectionRequest base = replicationRequest();
  PlacementOptions options;
  options.maxRoutesPerConnection = 1; // the single route saturates the cap

  bool truncated = false;
  llvm::Expected<std::vector<ConnectionPlan>> plans =
      synthesizeFanOut(base, std::vector<InstanceId>{11, 12},
                       std::vector<MemoryNodeId>{"acc.0", "acc.0"}, machine,
                       topology, options, &truncated);
  ASSERT_TRUE(static_cast<bool>(plans)) << llvm::toString(plans.takeError());
  EXPECT_TRUE(truncated);
  ASSERT_FALSE(plans->empty());
  EXPECT_EQ((*plans)[0].kind, ConnectionKind::Replicate);
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

// A gather sums what its feeds cost, and stages one intermediate tile on top.
TEST(Connections, FanInSumsTheFeedCost) {
  Cost feeds;
  feeds.latencyCycles = 42.0;
  feeds.dramBytes = 1024;
  feeds.localBytes = 1024;

  ConnectionPlan plan =
      synthesizeFanIn(std::vector<InstanceId>{1, 2},
                      /*consumer=*/9, /*value=*/5, "acc.0", 1024, feeds);
  EXPECT_EQ(plan.kind, ConnectionKind::Reduce);
  EXPECT_DOUBLE_EQ(plan.cost.latencyCycles, 42.0);
  EXPECT_EQ(plan.cost.dramBytes, 1024u);
  // The gathered tile's bytes join the feeds' staged bytes.
  EXPECT_EQ(plan.cost.localBytes, 1024u + 1024u);
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
  bool sawTransfer = false;
  for (const ConnectionPlan &plan : *plans) {
    if (plan.kind == ConnectionKind::Direct)
      continue; // a direct read touches no link, so no DRAM hop
    ASSERT_EQ(plan.kind, ConnectionKind::Transfer);
    ASSERT_FALSE(plan.memoryRoute.empty());
    EXPECT_EQ(plan.memoryRoute.front(), "dram.0");
    EXPECT_EQ(plan.cost.dramBytes, request.bytes);
    // The staging bytes are the value itself.
    EXPECT_EQ(plan.cost.localBytes, request.bytes);
    sawTransfer = true;
  }
  EXPECT_TRUE(sawTransfer);
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
  const ConnectionPlan *transfer = nullptr;
  for (const ConnectionPlan &plan : *plans)
    if (plan.kind == ConnectionKind::Transfer) {
      transfer = &plan;
      break;
    }
  ASSERT_NE(transfer, nullptr);
  EXPECT_EQ(transfer->cost.dramBytes, 0u);
  EXPECT_EQ(transfer->cost.localBytes, request.bytes);
}

// Spills are not modelled by the dialect, so the dimension is deliberately
// left unpopulated until a spill notion exists.
TEST(Connections, SpillBytesStayUnpopulated) {
  MachineModel machine = connectionMachine();
  TopologyService topology(machine);
  llvm::Expected<std::vector<ConnectionPlan>> plans =
      synthesizeConnections(baseRequest(), machine, topology);
  ASSERT_TRUE(static_cast<bool>(plans)) << llvm::toString(plans.takeError());
  // The transfer plan, not the rank-1 direct read: the direct read's
  // default-constructed cost would make this assertion vacuous.
  const ConnectionPlan *transfer = nullptr;
  for (const ConnectionPlan &plan : *plans)
    if (plan.kind == ConnectionKind::Transfer) {
      transfer = &plan;
      break;
    }
  ASSERT_NE(transfer, nullptr);
  EXPECT_EQ(transfer->cost.spillBytes, 0u);
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
  // The transfer routes are cost-ranked, so the first Transfer plan is the
  // cheapest route: the direct dram.0 -> acc.0 hop, 10 latency cycles plus
  // 1024 / 32 bytes-per-cycle = 42 cycles, over one engine x 100 cycles.
  const ConnectionPlan *cheapestTransfer = nullptr;
  for (const ConnectionPlan &plan : *plans)
    if (plan.kind == ConnectionKind::Transfer) {
      cheapestTransfer = &plan;
      break;
    }
  ASSERT_NE(cheapestTransfer, nullptr);
  EXPECT_DOUBLE_EQ(cheapestTransfer->cost.transferUtilization, 0.42);
}

// Without a modelled sync period there is no denominator, so utilization stays
// 0 rather than a fabricated constant.
TEST(Connections, TransferUtilizationStaysZeroWithoutSyncFacts) {
  MachineModel machine = connectionMachine(); // sync costs are all zero
  TopologyService topology(machine);

  llvm::Expected<std::vector<ConnectionPlan>> plans =
      synthesizeConnections(baseRequest(), machine, topology);
  ASSERT_TRUE(static_cast<bool>(plans)) << llvm::toString(plans.takeError());
  // The transfer plan, not the rank-1 direct read: the direct read's
  // default-constructed cost would make this assertion vacuous.
  const ConnectionPlan *transfer = nullptr;
  for (const ConnectionPlan &plan : *plans)
    if (plan.kind == ConnectionKind::Transfer) {
      transfer = &plan;
      break;
    }
  ASSERT_NE(transfer, nullptr);
  EXPECT_DOUBLE_EQ(transfer->cost.transferUtilization, 0.0);
}
