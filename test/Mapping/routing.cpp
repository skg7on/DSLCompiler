//===- routing.cpp - Deterministic topology routing (D2) -----------------===//

#include "LLK/Mapping/Routing.h"

#include "LLK/Machine/MachineModel.h"

#include "llvm/Support/Error.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <string>
#include <vector>

using namespace mlir::llk::machine;
using namespace mlir::llk::mapping;

namespace {

ExecutorNode executor(llvm::StringRef id, llvm::StringRef kind,
                      std::optional<std::string> parent = std::nullopt) {
  ExecutorNode node;
  node.id = id.str();
  node.kind = kind.str();
  node.parent = std::move(parent);
  return node;
}

MemoryNode memory(llvm::StringRef id, llvm::StringRef visibleFrom,
                  uint64_t capacity, uint64_t alignment = 64) {
  MemoryNode node;
  node.id = id.str();
  node.kind = id.starts_with("dram") ? "dram"
                                     : (id.starts_with("l2")    ? "l2"
                                        : id.starts_with("acc") ? "acc"
                                                                : "sram");
  node.visibleFrom = visibleFrom.str();
  node.capacityBytes = capacity;
  node.alignmentBytes = alignment;
  return node;
}

LinkEdge link(llvm::StringRef id, llvm::StringRef source,
              llvm::StringRef destination, double bandwidth, uint64_t latency,
              std::vector<std::string> engines = {"dma.0"},
              uint64_t transactionBytes = 4096) {
  LinkEdge edge;
  edge.id = id.str();
  edge.source = source.str();
  edge.destination = destination.str();
  edge.bandwidthBytesPerCycle = bandwidth;
  edge.latencyCycles = latency;
  // 4096 bytes by default, comfortably above the 1 KiB values these fixtures
  // move, so the §12.2 transaction-size check does not reject them. Tests that
  // exercise the check lower this explicitly.
  edge.transactionBytes = transactionBytes;
  edge.transferEngines = std::move(engines);
  return edge;
}

TransferEngineNode dma(llvm::StringRef attachedTo) {
  TransferEngineNode engine;
  engine.id = "dma.0";
  engine.kind = "dma";
  engine.attachedTo = attachedTo.str();
  engine.count = 4;
  engine.maxOutstanding = 8;
  return engine;
}

/// dram -> l2 -> sram -> acc, plus an expensive direct dram -> acc and a
/// back edge sram -> dram to exercise cycle freedom.
MachineModel diamond() {
  MachineModel model;
  model.target = "diamond";
  model.executors = {executor("package.0", "worker"),
                     executor("core.0", "core", std::string("package.0")),
                     executor("other.0", "core")};
  model.memories = {memory("dram", "package.0", 1u << 30),
                    memory("l2", "package.0", 1u << 20),
                    memory("sram", "core.0", 1u << 15),
                    memory("acc", "core.0", 4096, /*alignment=*/32)};
  model.transferEngines = {dma("core.0")};
  model.links = {link("dram_to_l2", "dram", "l2", 32, 100),
                 link("l2_to_sram", "l2", "sram", 64, 10),
                 link("sram_to_acc", "sram", "acc", 128, 2),
                 link("dram_to_acc", "dram", "acc", 8, 200),
                 link("sram_to_dram", "sram", "dram", 32, 100)};
  return model;
}

/// dram -> sram -> acc with no direct edge, so the only route must stage
/// through sram.
MachineModel chain(uint64_t sramCapacity) {
  MachineModel model;
  model.target = "chain";
  model.executors = {executor("package.0", "worker"),
                     executor("core.0", "core", std::string("package.0"))};
  model.memories = {memory("dram", "package.0", 1u << 30),
                    memory("sram", "core.0", sramCapacity),
                    memory("acc", "core.0", 4096, 32)};
  model.transferEngines = {dma("core.0")};
  model.links = {link("dram_to_sram", "dram", "sram", 32, 100),
                 link("sram_to_acc", "sram", "acc", 128, 2)};
  return model;
}

std::vector<std::string> linkIds(const MemoryRoute &route) {
  return std::vector<std::string>(route.links.begin(), route.links.end());
}

/// Builds a request by field so adding a field to `RouteRequest` never leaves a
/// partially-initialized aggregate in these tests.
RouteRequest routeRequest(llvm::StringRef source, llvm::StringRef destination,
                          uint64_t bytes, uint64_t alignment,
                          std::optional<ExecutorId> producer = std::nullopt,
                          std::optional<ExecutorId> consumer = std::nullopt) {
  RouteRequest request;
  request.source = source.str();
  request.destination = destination.str();
  request.bytes = bytes;
  request.alignmentBytes = alignment;
  request.producerExecutor = std::move(producer);
  request.consumerExecutor = std::move(consumer);
  return request;
}

} // namespace

TEST(Routing, PrefersCheaperMultiHopOverExpensiveDirect) {
  MachineModel model = diamond();
  TopologyService service(model);
  RouteRequest request = routeRequest("dram", "acc", 1024, 32);
  auto result = service.enumerateRoutes(request, 8);
  ASSERT_TRUE(static_cast<bool>(result));
  ASSERT_GE(result->size(), 2u);
  EXPECT_EQ(
      linkIds((*result)[0]),
      (std::vector<std::string>{"dram_to_l2", "l2_to_sram", "sram_to_acc"}));
  EXPECT_EQ(linkIds((*result)[1]), (std::vector<std::string>{"dram_to_acc"}));
  EXPECT_LT((*result)[0].cost.latencyCycles, (*result)[1].cost.latencyCycles);
  EXPECT_EQ((*result)[0].hopCount(), 3u);
}

TEST(Routing, RoutesAreCycleFree) {
  MachineModel model = diamond();
  TopologyService service(model);
  RouteRequest request = routeRequest("dram", "acc", 1024, 32);
  auto result = service.enumerateRoutes(request, 16);
  ASSERT_TRUE(static_cast<bool>(result));
  for (const MemoryRoute &route : *result) {
    std::vector<std::string> nodes(route.nodes.begin(), route.nodes.end());
    std::sort(nodes.begin(), nodes.end());
    EXPECT_EQ(std::adjacent_find(nodes.begin(), nodes.end()), nodes.end());
  }
}

TEST(Routing, RespectsTheLimit) {
  MachineModel model = diamond();
  TopologyService service(model);
  RouteRequest request = routeRequest("dram", "acc", 1024, 32);
  auto result = service.enumerateRoutes(request, 1);
  ASSERT_TRUE(static_cast<bool>(result));
  ASSERT_EQ(result->size(), 1u);
  EXPECT_EQ(
      linkIds((*result)[0]),
      (std::vector<std::string>{"dram_to_l2", "l2_to_sram", "sram_to_acc"}));
}

TEST(Routing, ReportsTruncationWhenTheRouteCapIsHit) {
  MachineModel model = diamond();
  TopologyService service(model);
  RouteRequest request = routeRequest("dram", "acc", 1024, 32);
  // diamond() has more than one route dram -> acc; asking for one reaches the
  // cap and must say so rather than implying the space was exhausted.
  bool truncated = false;
  auto result = service.enumerateRoutes(request, 1, &truncated);
  ASSERT_TRUE(static_cast<bool>(result));
  ASSERT_EQ(result->size(), 1u);
  EXPECT_TRUE(truncated);
}

TEST(Routing, ReportsNoTruncationWhenEveryRouteFits) {
  MachineModel model = diamond();
  TopologyService service(model);
  RouteRequest request = routeRequest("dram", "acc", 1024, 32);
  bool truncated = false;
  auto result = service.enumerateRoutes(request, 8, &truncated);
  ASSERT_TRUE(static_cast<bool>(result));
  EXPECT_LT(result->size(), 8u); // the whole space fit under the cap
  EXPECT_FALSE(truncated);
}

TEST(Routing, ReportsTruncationWhenTheHopCapPrunesALegalPath) {
  MachineModel model = chain(/*sramCapacity=*/32768);
  // dram -> acc needs two hops; a one-hop cap cuts the path at sram, which
  // still has a legal link to acc. The caller must hear that the search was
  // bounded, even though no route was found.
  TopologyService service(model, RouteOptions{/*maxRoutes=*/8, /*maxHops=*/1});
  RouteRequest request = routeRequest("dram", "acc", 1024, 32);
  bool truncated = false;
  auto result = service.enumerateRoutes(request, 8, &truncated);
  EXPECT_FALSE(static_cast<bool>(result));
  EXPECT_TRUE(truncated);
}

TEST(Routing, DoesNotReportTruncationWhenTheFrontierEmpties) {
  MachineModel model = chain(/*sramCapacity=*/32768);
  model.links.clear(); // nowhere to go at all
  TopologyService service(model, RouteOptions{/*maxRoutes=*/8, /*maxHops=*/1});
  RouteRequest request = routeRequest("dram", "acc", 1024, 32);
  bool truncated = false;
  auto result = service.enumerateRoutes(request, 8, &truncated);
  EXPECT_FALSE(static_cast<bool>(result));
  EXPECT_FALSE(truncated); // no legal hop was ever cut, so no truncation
}

TEST(Routing, DoesNotReportTruncationWhenTheHopCapHasNoLegalExtension) {
  // A two-node loop with an unreachable destination: the cap lands on sram,
  // whose only outgoing link returns to a memory already on the path. The cap
  // was reached, but nothing legal was pruned, so this is not a truncation.
  MachineModel model;
  model.target = "loop";
  model.executors = {executor("package.0", "worker"),
                     executor("core.0", "core", std::string("package.0"))};
  model.memories = {memory("dram", "package.0", 1u << 30),
                    memory("sram", "core.0", 32768),
                    memory("acc", "core.0", 4096, 32)};
  model.transferEngines = {dma("core.0")};
  model.links = {link("dram_to_sram", "dram", "sram", 32, 100),
                 link("sram_to_dram", "sram", "dram", 32, 100)};
  TopologyService service(model, RouteOptions{/*maxRoutes=*/8, /*maxHops=*/1});
  RouteRequest request = routeRequest("dram", "acc", 1024, 32);
  bool truncated = false;
  auto result = service.enumerateRoutes(request, 8, &truncated);
  EXPECT_FALSE(static_cast<bool>(result));
  EXPECT_FALSE(truncated);
}

TEST(Routing, SameMemoryIsATrivialRoute) {
  MachineModel model = diamond();
  TopologyService service(model);
  RouteRequest request = routeRequest("dram", "dram", 1024, 32);
  auto result = service.enumerateRoutes(request, 8);
  ASSERT_TRUE(static_cast<bool>(result));
  ASSERT_EQ(result->size(), 1u);
  EXPECT_TRUE(result->front().links.empty());
  EXPECT_EQ(result->front().cost.latencyCycles, 0.0);
}

TEST(Routing, ReportsNoRoute) {
  MachineModel model = diamond();
  model.links.clear(); // no edges at all
  TopologyService service(model);
  RouteRequest request = routeRequest("dram", "acc", 1024, 32);
  EXPECT_FALSE(static_cast<bool>(service.enumerateRoutes(request, 8)));
}

TEST(Routing, RejectsUnknownEndpoint) {
  MachineModel model = diamond();
  TopologyService service(model);
  RouteRequest request = routeRequest("dram", "nowhere", 1024, 32);
  EXPECT_FALSE(static_cast<bool>(service.enumerateRoutes(request, 8)));
}

TEST(Routing, RejectsInvisibleEndpoint) {
  MachineModel model = diamond();
  TopologyService service(model);
  RouteRequest request =
      routeRequest("dram", "acc", 1024, 32, std::string("other.0"));
  // other.0 is not within dram's visibility scope (package.0).
  EXPECT_FALSE(static_cast<bool>(service.enumerateRoutes(request, 8)));
}

TEST(Routing, AcceptsVisibleEndpoint) {
  MachineModel model = diamond();
  TopologyService service(model);
  RouteRequest request = routeRequest(
      "dram", "acc", 1024, 32, std::string("core.0"), std::string("core.0"));
  EXPECT_TRUE(static_cast<bool>(service.enumerateRoutes(request, 8)));
}

TEST(Routing, RejectsTooSmallIntermediate) {
  MachineModel model = chain(/*sramCapacity=*/512);
  TopologyService service(model);
  RouteRequest request = routeRequest("dram", "acc", 1024, 32);
  EXPECT_FALSE(static_cast<bool>(service.enumerateRoutes(request, 8)));
}

TEST(Routing, AcceptsFittingIntermediate) {
  MachineModel model = chain(/*sramCapacity=*/32768);
  TopologyService service(model);
  RouteRequest request = routeRequest("dram", "acc", 1024, 32);
  EXPECT_TRUE(static_cast<bool>(service.enumerateRoutes(request, 8)));
}

TEST(Routing, RejectsLayoutTheStagingMemoryDoesNotSupport) {
  MachineModel model = chain(/*sramCapacity=*/32768);
  // The endpoints can hold the value, but the sram the route must stage
  // through cannot: it only holds a tiled layout, not the row-major the value
  // declares.
  model.memories[0].supportedLayouts = {"row_major"}; // dram
  model.memories[1].supportedLayouts = {"tiled"};     // sram
  model.memories[2].supportedLayouts = {"row_major"}; // acc
  TopologyService service(model);
  RouteRequest request = routeRequest("dram", "acc", 1024, 32);
  request.layoutClass = "row_major";
  EXPECT_FALSE(static_cast<bool>(service.enumerateRoutes(request, 8)));
}

TEST(Routing, AcceptsLayoutEveryHopSupports) {
  MachineModel model = chain(/*sramCapacity=*/32768);
  for (MemoryNode &node : model.memories)
    node.supportedLayouts = {"row_major"};
  TopologyService service(model);
  RouteRequest request = routeRequest("dram", "acc", 1024, 32);
  request.layoutClass = "row_major";
  EXPECT_TRUE(static_cast<bool>(service.enumerateRoutes(request, 8)));
}

TEST(Routing, RejectsValueLargerThanTheLinkTransaction) {
  MachineModel model = chain(/*sramCapacity=*/32768);
  // The first hop can only move 64-byte transactions but the value is 1 KiB.
  model.links[0].transactionBytes = 64;
  TopologyService service(model);
  RouteRequest request = routeRequest("dram", "acc", 1024, 32);
  EXPECT_FALSE(static_cast<bool>(service.enumerateRoutes(request, 8)));
}

TEST(Routing, AcceptsValueFittingTheLinkTransaction) {
  MachineModel model = chain(/*sramCapacity=*/32768);
  model.links[0].transactionBytes = 1024;
  TopologyService service(model);
  RouteRequest request = routeRequest("dram", "acc", 1024, 32);
  EXPECT_TRUE(static_cast<bool>(service.enumerateRoutes(request, 8)));
}

TEST(Routing, RejectsIntermediateWithNoFreeSpace) {
  MachineModel model = chain(/*sramCapacity=*/32768);
  TopologyService service(model);
  RouteRequest request = routeRequest("dram", "acc", 1024, 32);
  // 512 bytes free in the staging sram, but the value needs 1024.
  request.liveBytesOnIntermediate = 32768 - 512;
  EXPECT_FALSE(static_cast<bool>(service.enumerateRoutes(request, 8)));
}

TEST(Routing, AcceptsIntermediateWithLiveBytesButRoomToSpare) {
  MachineModel model = chain(/*sramCapacity=*/32768);
  TopologyService service(model);
  RouteRequest request = routeRequest("dram", "acc", 1024, 32);
  request.liveBytesOnIntermediate = 32768 - 2048;
  EXPECT_TRUE(static_cast<bool>(service.enumerateRoutes(request, 8)));
}

TEST(Routing, RejectsHopWithoutEngine) {
  MachineModel model = chain(32768);
  for (LinkEdge &edge : model.links)
    edge.transferEngines.clear();
  TopologyService service(model);
  RouteRequest request = routeRequest("dram", "acc", 1024, 32);
  EXPECT_FALSE(static_cast<bool>(service.enumerateRoutes(request, 8)));
}

TEST(Routing, RejectsUnsupportedAlignment) {
  MachineModel model = diamond();
  TopologyService service(model);
  // No memory in the fixture aligns to 128 bytes.
  RouteRequest request = routeRequest("dram", "acc", 1024, 128);
  EXPECT_FALSE(static_cast<bool>(service.enumerateRoutes(request, 8)));
}

TEST(Routing, IdenticalRequestsProduceIdenticalRoutes) {
  MachineModel model = diamond();
  TopologyService service(model);
  RouteRequest request = routeRequest("dram", "acc", 1024, 32);
  auto first = service.enumerateRoutes(request, 8);
  auto second = service.enumerateRoutes(request, 8);
  ASSERT_TRUE(static_cast<bool>(first));
  ASSERT_TRUE(static_cast<bool>(second));
  ASSERT_EQ(first->size(), second->size());
  for (size_t i = 0; i < first->size(); ++i)
    EXPECT_EQ(linkIds((*first)[i]), linkIds((*second)[i]));
}
