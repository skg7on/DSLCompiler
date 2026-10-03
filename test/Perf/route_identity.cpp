//===- route_identity.cpp - Binder->perf route identity -------------------===//
//
// The binder materializes a selected multi-hop route as one `micro.async_copy`
// per hop (design §18.2). The performance model must charge each of those
// copies its own link -- identified by the connection value and destination
// node the binder stamps on it -- not the first link of the movement's endpoint
// memory *kind*. This test drives the real binder and then builds the DAG, so
// the identity is proven end to end rather than assumed from route order.
//
//===----------------------------------------------------------------------===//

#include "LLK/Dialect/Micro/MicroDialect.h"
#include "LLK/Machine/MachineModelLoader.h"
#include "LLK/Mapping/CoveringSearch.h"
#include "LLK/Mapping/MappingTarget.h"
#include "LLK/Mapping/PlanBinder.h"
#include "LLK/Mapping/WorkloadGraph.h"
#include "LLK/Perf/MicroDAG.h"

#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include "gtest/gtest.h"

#include <memory>
#include <string>
#include <vector>

using namespace mlir;
using namespace mlir::llk;

namespace {

/// A copy producer placed in DRAM and a vector consumer placed in SRAM, so the
/// tensor between them must cross the hierarchy. The moved value is a tensor,
/// which is what the binder can materialize as copies.
constexpr llvm::StringLiteral kKernel = R"mlir(
module {
  micro.kernel @mapped {
    %ext = tensor.empty() : tensor<8x8xf32>
    %t, %tok = micro.async_copy %ext {src_memory = #micro.memory<dram>, dst_memory = #micro.memory<sram>} : tensor<8x8xf32> -> tensor<8x8xf32>, !micro.async_token
    %v = micro.tile_view %t {shape = array<i64: 8, 8>} : tensor<8x8xf32> -> !micro.tile<8x8xf32, memory = #micro.memory<sram>>
    %r = micro.vector "add" %v, %v : !micro.tile<8x8xf32, memory = #micro.memory<sram>>, !micro.tile<8x8xf32, memory = #micro.memory<sram>> -> !micro.tile<8x8xf32, memory = #micro.memory<sram>>
    micro.yield
  }
}
)mlir";

/// The copy runs in DRAM, the vector in SRAM.
constexpr llvm::StringLiteral kRules = R"llkmap(
rule t.copy {
  match micro.async_copy();
  require executor kind worker;
  require memory kind dram;
  bundle "b";
  emit "e1";
  cost 1;
}
rule t.vector {
  match micro.vector(op = "add");
  require executor kind worker;
  require memory kind sram;
  bundle "b";
  emit "e1";
  cost 1;
}
)llkmap";

/// DRAM reaches SRAM only through L2. The second L2->SRAM link (to `sram.1`)
/// is declared *first* and costs far more, so a kind-first lookup charges the
/// wrong node; only the stamped destination node charges the right link.
constexpr llvm::StringLiteral kMachine = R"yaml(
schema: llk.machine.v2
target: route-identity
clock_hz: 1000000000
worker_threads: 1
executors:
  - id: cluster.0
    kind: cluster
  - id: worker.0
    kind: worker
    parent: cluster.0
memories:
  - id: dram.0
    kind: dram
    visible_from: cluster.0
    capacity_bytes: 1048576
    alignment_bytes: 64
    supported_layouts: [row_major]
    bandwidth_bytes_per_cycle: 64
    latency_cycles: 1
  - id: l2.0
    kind: l2
    visible_from: cluster.0
    capacity_bytes: 262144
    alignment_bytes: 64
    supported_layouts: [row_major]
    bandwidth_bytes_per_cycle: 64
    latency_cycles: 1
  - id: sram.0
    kind: sram
    visible_from: cluster.0
    capacity_bytes: 65536
    alignment_bytes: 64
    supported_layouts: [row_major]
    bandwidth_bytes_per_cycle: 64
    latency_cycles: 1
  - id: sram.1
    kind: sram
    visible_from: cluster.0
    capacity_bytes: 65536
    alignment_bytes: 64
    supported_layouts: [row_major]
    bandwidth_bytes_per_cycle: 64
    latency_cycles: 1
compute:
  - id: vpu
    kind: vector_engine
    attached_to: worker.0
    element_types: [f32]
    shapes: [[8]]
    lanes: {f32: 8}
    issue_cycles: 1
    latency_cycles: 1
    supported_layouts: [row_major]
transfer_engines:
  - id: dma.0
    kind: dma
    attached_to: cluster.0
    count: 1
    max_outstanding: 1
links:
  - id: dram_to_l2.0
    source: dram.0
    destination: l2.0
    bandwidth_bytes_per_cycle: 64
    latency_cycles: 220
    transaction_bytes: 64
    transfer_engines: [dma.0]
  - id: l2_to_sram.1
    source: l2.0
    destination: sram.1
    bandwidth_bytes_per_cycle: 64
    latency_cycles: 77
    transaction_bytes: 64
    transfer_engines: [dma.0]
  - id: l2_to_sram.0
    source: l2.0
    destination: sram.0
    bandwidth_bytes_per_cycle: 64
    latency_cycles: 12
    transaction_bytes: 64
    transfer_engines: [dma.0]
)yaml";

Operation *findKernel(ModuleOp module) {
  Operation *kernel = nullptr;
  module->walk([&](Operation *op) {
    if (!kernel && op->getName().getStringRef() == "micro.kernel")
      kernel = op;
  });
  return kernel;
}

} // namespace

TEST(RouteIdentity, BinderStampedCopiesChargeTheirOwnLinks) {
  MLIRContext context;
  context.getOrLoadDialect<micro::MicroDialect>();
  context.getOrLoadDialect<tensor::TensorDialect>();

  OwningOpRef<ModuleOp> module = parseSourceString<ModuleOp>(kKernel, &context);
  ASSERT_TRUE(module);

  llvm::Expected<machine::MachineModel> machine =
      machine::parseMachineModel(kMachine, "<test>");
  ASSERT_TRUE(static_cast<bool>(machine))
      << llvm::toString(machine.takeError());

  llvm::Expected<mapping::LayoutRegistry> layouts =
      mapping::parseLayoutText("", "<test>");
  ASSERT_TRUE(static_cast<bool>(layouts))
      << llvm::toString(layouts.takeError());
  llvm::Expected<mapping::RuleRegistry> rules =
      mapping::parseRuleText(kRules, "<test>");
  ASSERT_TRUE(static_cast<bool>(rules)) << llvm::toString(rules.takeError());
  mapping::FileMappingTarget target("route-identity", *machine,
                                    std::move(*layouts), std::move(*rules),
                                    std::vector<std::string>{"e1"});

  llvm::Expected<mapping::WorkloadGraph> graph =
      mapping::extractWorkloadGraph(findKernel(*module));
  ASSERT_TRUE(static_cast<bool>(graph)) << llvm::toString(graph.takeError());
  mapping::LayoutContext layoutContext;
  layoutContext.rank = 2;
  layoutContext.elementType = "f32";
  mapping::MappingSearchOptions options;
  options.mode = mapping::SearchMode::Deterministic;
  mapping::CoveringSearch search(*graph, target, context, layoutContext,
                                 options);
  llvm::Expected<mapping::MappingSearchResult> result = search.search();
  ASSERT_TRUE(static_cast<bool>(result)) << llvm::toString(result.takeError());
  ASSERT_EQ(result->plans.size(), 1u);
  const mapping::CoveringPlan &plan = result->plans.front();
  ASSERT_FALSE(plan.connectionPlans.empty());

  llvm::Expected<mapping::BoundPlan> bound =
      mapping::bindPlan(*module, plan, target);
  ASSERT_TRUE(static_cast<bool>(bound)) << llvm::toString(bound.takeError());
  for (const std::string &note : bound->unmaterialized)
    ADD_FAILURE() << note;

  // The connection went DRAM -> L2 -> SRAM, so the binder emitted two
  // materialized copies (the original kernel copy is not stamped), each
  // carrying its connection value and destination node.
  std::vector<std::string> stampedNodes;
  bound->module->walk([&](Operation *op) {
    if (op->getName().getStringRef() != "micro.async_copy")
      return;
    auto node = op->getAttrOfType<StringAttr>("micro.dst_node");
    if (!node)
      return; // the original kernel copy, not a materialized hop
    EXPECT_TRUE(op->getAttrOfType<IntegerAttr>("micro.value"))
        << "a stamped hop carries no connection value";
    stampedNodes.push_back(node.getValue().str());
  });
  EXPECT_EQ(stampedNodes, (std::vector<std::string>{"l2.0", "sram.0"}));

  // Charge the bound kernel. Each materialized hop is charged its own link:
  // DRAM->L2 (220 + 256/64 = 224) then L2->SRAM.0 (12 + 4 = 16). The original
  // kernel copy is unrouted but shares the DRAM->SRAM endpoint kinds, so it
  // charges the whole route -- the two hops again. The L2->SRAM.1 link (81) is
  // never reached: only a node-id match could pick it, and none should.
  auto dag = mlir::llk::perf::buildMicroDAG(bound->kernel, *machine);
  ASSERT_TRUE(static_cast<bool>(dag)) << llvm::toString(dag.takeError());

  std::vector<uint64_t> dmaCycles;
  for (const mlir::llk::perf::MicroEvent &event : dag->events)
    if (event.resource == mlir::llk::perf::ResourceKind::Dma)
      dmaCycles.push_back(event.minCycles);
  EXPECT_EQ(dmaCycles, (std::vector<uint64_t>{224, 16, 224, 16}));
  for (uint64_t cycles : dmaCycles)
    EXPECT_NE(cycles, 81u) << "the L2->SRAM.1 link was charged by mistake";
}
