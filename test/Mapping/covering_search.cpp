//===- covering_search.cpp - Complete-plan search (D6) -------------------===//

#include "LLK/Mapping/CoveringSearch.h"
#include "LLK/Mapping/Diagnostics.h"
#include "LLK/Mapping/LatencyProvider.h"
#include "LLK/Mapping/TileFacts.h"

#include "LLK/Dialect/Micro/MicroDialect.h"

#include "mlir/AsmParser/AsmParser.h"
#include "mlir/IR/MLIRContext.h"

#include "llvm/Support/Error.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <variant>
#include <vector>

using namespace mlir::llk::machine;
using namespace mlir::llk::mapping;

namespace {

MachineModel searchMachine() {
  MachineModel model;
  model.target = "search";
  model.executors = {{"e0", "worker", std::nullopt, {}, 1, {}}};
  MemoryNode sram;
  sram.id = "sram.0";
  sram.kind = "sram";
  sram.visibleFrom = "e0";
  sram.capacityBytes = 1u << 20;
  sram.alignmentBytes = 64;
  MemoryNode dram;
  dram.id = "dram.0";
  dram.kind = "dram";
  dram.visibleFrom = "e0";
  dram.capacityBytes = 1u << 30;
  dram.alignmentBytes = 64;
  model.memories = {sram, dram};
  return model;
}

/// Two interchangeable workers. With symmetry reduction off, a candidate that
/// requires a worker has two legal placements, so an instance cap of one
/// genuinely stops enumeration early rather than merely matching the count.
MachineModel twoWorkerMachine() {
  MachineModel model = searchMachine();
  model.executors = {{"e0", "worker", std::nullopt, {}, 1, {}},
                     {"e1", "worker", std::nullopt, {}, 1, {}}};
  return model;
}

/// `searchMachine` with the sram node shrunk to 5000 bytes. The fixtures here
/// carry untyped values, so the search falls back to 4096 bytes per bound
/// instance; one instance fits the node while two do not -- and two still fit
/// the default global byte budget, leaving the per-memory check as the only
/// thing that can reject the pair.
MachineModel smallMemoryMachine() {
  MachineModel model = searchMachine();
  for (MemoryNode &memory : model.memories)
    if (memory.kind == "sram")
      memory.capacityBytes = 5000;
  return model;
}

/// A `micro.vector` with `op = <op>`, as the shipped rules predicate on.
mlir::DictionaryAttr vectorAttributes(mlir::MLIRContext &context,
                                      llvm::StringRef op = "add") {
  return mlir::DictionaryAttr::get(
      &context, {mlir::NamedAttribute(mlir::StringAttr::get(&context, "op"),
                                      mlir::StringAttr::get(&context, op))});
}

/// producer -> consumer, both `micro.vector`. `producerOp` and `consumerOp` are
/// the nodes' `op` attributes, so a fixture can gate the two nodes onto
/// different rules (and thus different memories).
WorkloadGraph twoNodeGraph(mlir::MLIRContext &context,
                           llvm::StringRef producerOp = "add",
                           llvm::StringRef consumerOp = "add") {
  WorkloadGraph graph;
  WorkloadValueId input =
      graph.addValue(WorkloadValue{0, mlir::Type(), "in", /*external=*/true});
  WorkloadValueId middle =
      graph.addValue(WorkloadValue{0, mlir::Type(), "mid", /*external=*/false});
  WorkloadValueId output =
      graph.addValue(WorkloadValue{0, mlir::Type(), "out", /*external=*/false});

  WorkloadNode producer;
  producer.opName = "micro.vector";
  producer.attributes = vectorAttributes(context, producerOp);
  producer.inputs.push_back(WorkloadPort{input, mlir::Type(), std::nullopt});
  producer.outputs.push_back(WorkloadPort{middle, mlir::Type(), std::nullopt});
  graph.addNode(std::move(producer));

  WorkloadNode consumer;
  consumer.opName = "micro.vector";
  consumer.attributes = vectorAttributes(context, consumerOp);
  consumer.inputs.push_back(WorkloadPort{middle, mlir::Type(), std::nullopt});
  consumer.outputs.push_back(WorkloadPort{output, mlir::Type(), std::nullopt});
  graph.addNode(std::move(consumer));

  graph.finalize();
  return graph;
}

/// `twoNodeGraph` with every value and port carrying `tile`, so the search
/// derives real bytes for each materialized tile and each crossing instead of
/// sizing everything with a constant.
WorkloadGraph twoNodeTileGraph(mlir::MLIRContext &context, mlir::Type tile) {
  WorkloadGraph graph;
  WorkloadValueId input =
      graph.addValue(WorkloadValue{0, tile, "in", /*external=*/true});
  WorkloadValueId middle =
      graph.addValue(WorkloadValue{0, tile, "mid", /*external=*/false});
  WorkloadValueId output =
      graph.addValue(WorkloadValue{0, tile, "out", /*external=*/false});

  WorkloadNode producer;
  producer.opName = "micro.vector";
  producer.attributes = vectorAttributes(context, "add");
  producer.inputs.push_back(WorkloadPort{input, tile, std::nullopt});
  producer.outputs.push_back(WorkloadPort{middle, tile, std::nullopt});
  graph.addNode(std::move(producer));

  WorkloadNode consumer;
  consumer.opName = "micro.vector";
  consumer.attributes = vectorAttributes(context, "add");
  consumer.inputs.push_back(WorkloadPort{middle, tile, std::nullopt});
  consumer.outputs.push_back(WorkloadPort{output, tile, std::nullopt});
  graph.addNode(std::move(consumer));

  graph.finalize();
  return graph;
}

/// A three-node chain `in -> n0 -> v0 -> n1 -> v1 -> n2 -> out`, every node a
/// `micro.vector(op = "add")` over `tile`-typed values. Each node binds one
/// memory and materializes the derived tile size.
WorkloadGraph threeNodeChainGraph(mlir::MLIRContext &context, mlir::Type tile) {
  WorkloadGraph graph;
  WorkloadValueId input =
      graph.addValue(WorkloadValue{0, tile, "in", /*external=*/true});
  WorkloadValueId v0 =
      graph.addValue(WorkloadValue{0, tile, "v0", /*external=*/false});
  WorkloadValueId v1 =
      graph.addValue(WorkloadValue{0, tile, "v1", /*external=*/false});
  WorkloadValueId output =
      graph.addValue(WorkloadValue{0, tile, "out", /*external=*/false});

  auto node = [&](unsigned ordinal, WorkloadValueId in, WorkloadValueId out) {
    WorkloadNode n;
    n.opName = "micro.vector";
    n.sourceOrdinal = ordinal;
    n.attributes = vectorAttributes(context, "add");
    n.inputs.push_back(WorkloadPort{in, tile, std::nullopt});
    n.outputs.push_back(WorkloadPort{out, tile, std::nullopt});
    graph.addNode(std::move(n));
  };
  node(0, input, v0);
  node(1, v0, v1);
  node(2, v1, output);

  graph.finalize();
  return graph;
}

/// One `micro.vector` node with two output tiles of different sizes. The node
/// binds a single memory, which holds both outputs.
WorkloadGraph twoOutputTileGraph(mlir::MLIRContext &context, mlir::Type first,
                                 mlir::Type second) {
  WorkloadGraph graph;
  WorkloadValueId input =
      graph.addValue(WorkloadValue{0, first, "in", /*external=*/true});
  WorkloadValueId out0 =
      graph.addValue(WorkloadValue{0, first, "out0", /*external=*/false});
  WorkloadValueId out1 =
      graph.addValue(WorkloadValue{0, second, "out1", /*external=*/false});

  WorkloadNode node;
  node.opName = "micro.vector";
  node.attributes = vectorAttributes(context, "add");
  node.inputs.push_back(WorkloadPort{input, first, std::nullopt});
  node.outputs.push_back(WorkloadPort{out0, first, std::nullopt});
  node.outputs.push_back(WorkloadPort{out1, second, std::nullopt});
  graph.addNode(std::move(node));

  graph.finalize();
  return graph;
}

/// `searchMachine` with the sram node's capacity set to `bytes`, so a capacity
/// test can pin the exact live total it admits.
MachineModel sramCapacityMachine(uint64_t bytes) {
  MachineModel model = searchMachine();
  for (MemoryNode &memory : model.memories)
    if (memory.kind == "sram")
      memory.capacityBytes = bytes;
  return model;
}

/// Two executors whose ids sort opposite to their kinds, so ordering placements
/// by node and by the executor binding tuple give different answers.
MachineModel oppositeOrderMachine() {
  MachineModel model;
  model.target = "opposite";
  model.executors = {{"z0", "worker", std::nullopt, {}, 1, {}},
                     {"a0", "dma", std::nullopt, {}, 1, {}}};
  return model;
}

/// One rule per executor kind, each gated on the node's `op` attribute so a
/// node takes exactly one of them.
constexpr llvm::StringLiteral kExecutorRules = R"llkmap(
rule r.work {
  match micro.vector(op = "a");
  require executor kind worker;
  bundle "b.work";
  emit "e1";
}
rule r.dma {
  match micro.vector(op = "z");
  require executor kind dma;
  bundle "b.dma";
  emit "e1";
}
)llkmap";

/// Two independent `micro.vector` nodes: `op = "a"` (which binds executor `z0`)
/// and `op = "z"` (which binds executor `a0`). `finalize` orders nodes by
/// content, so the node bound to the *lexicographically smaller* executor gets
/// the *greater* node id -- node order and binding-tuple order disagree.
WorkloadGraph oppositeBindingGraph(mlir::MLIRContext &context) {
  WorkloadGraph graph;
  WorkloadValueId in0 =
      graph.addValue(WorkloadValue{0, mlir::Type(), "in0", /*external=*/true});
  WorkloadValueId out0 = graph.addValue(
      WorkloadValue{0, mlir::Type(), "out0", /*external=*/false});
  WorkloadValueId in1 =
      graph.addValue(WorkloadValue{0, mlir::Type(), "in1", /*external=*/true});
  WorkloadValueId out1 = graph.addValue(
      WorkloadValue{0, mlir::Type(), "out1", /*external=*/false});

  WorkloadNode first;
  first.opName = "micro.vector";
  first.attributes = vectorAttributes(context, "a");
  first.inputs.push_back(WorkloadPort{in0, mlir::Type(), std::nullopt});
  first.outputs.push_back(WorkloadPort{out0, mlir::Type(), std::nullopt});
  graph.addNode(std::move(first));

  WorkloadNode second;
  second.opName = "micro.vector";
  second.attributes = vectorAttributes(context, "z");
  second.inputs.push_back(WorkloadPort{in1, mlir::Type(), std::nullopt});
  second.outputs.push_back(WorkloadPort{out1, mlir::Type(), std::nullopt});
  graph.addNode(std::move(second));

  graph.finalize();
  return graph;
}

constexpr llvm::StringLiteral kRules = R"llkmap(
rule r.cheap {
  match micro.vector(op = "add");
  require executor kind worker;
  bundle "b.cheap";
  emit "e1";
  cost 1;
}
rule r.expensive {
  match micro.vector(op = "add");
  require executor kind worker;
  bundle "b.expensive";
  emit "e1";
  cost 10;
}
)llkmap";

/// Six interchangeable rules for one node, every one the same 1-cycle cost and
/// an interchangeable placement. Kept under a five-candidate cap,
/// cross-producing them over a two-node graph yields a large set of *cost-tied*
/// complete plans, so which K survive a top-K cap is decided purely by the
/// tie-break key rather than by cost.
constexpr llvm::StringLiteral kDenseTieRules = R"llkmap(
rule r.0 {
  match micro.vector(op = "add");
  require executor kind worker;
  bundle "b.0";
  emit "e1";
  cost 1;
}
rule r.1 {
  match micro.vector(op = "add");
  require executor kind worker;
  bundle "b.1";
  emit "e1";
  cost 1;
}
rule r.2 {
  match micro.vector(op = "add");
  require executor kind worker;
  bundle "b.2";
  emit "e1";
  cost 1;
}
rule r.3 {
  match micro.vector(op = "add");
  require executor kind worker;
  bundle "b.3";
  emit "e1";
  cost 1;
}
rule r.4 {
  match micro.vector(op = "add");
  require executor kind worker;
  bundle "b.4";
  emit "e1";
  cost 1;
}
rule r.5 {
  match micro.vector(op = "add");
  require executor kind worker;
  bundle "b.5";
  emit "e1";
  cost 1;
}
)llkmap";

constexpr llvm::StringLiteral kRulesWithMemory = R"llkmap(
rule r.cheap {
  match micro.vector(op = "add");
  require executor kind worker;
  require memory kind sram;
  bundle "b.cheap";
  emit "e1";
  cost 1;
}
)llkmap";

constexpr llvm::StringLiteral kNoPlacementRules = R"llkmap(
rule r.pe_only {
  match micro.vector(op = "add");
  require executor kind pe;
  bundle "b.pe";
  emit "e1";
}
)llkmap";

/// Two rules over the same node, each satisfied only by one value of the shared
/// parameter `VW`: a binding selects between them. Unpinned, both match (and
/// the cheaper `r.wide` wins); pinned to 4 only `r.narrow` matches.
constexpr llvm::StringLiteral kParameterRules = R"llkmap(
rule r.wide {
  match micro.vector(op = "add");
  param VW in [4..8];
  require VW == 8;
  require executor kind worker;
  bundle "b.wide";
  emit "e1";
  cost 1;
}
rule r.narrow {
  match micro.vector(op = "add");
  param VW in [4..8];
  require VW == 4;
  require executor kind worker;
  bundle "b.narrow";
  emit "e1";
  cost 2;
}
)llkmap";

/// One rule over a node, satisfied only by VW = 8, so a binding that pins any
/// other value leaves the node with no rule in effect.
constexpr llvm::StringLiteral kSingleParameterRule = R"llkmap(
rule r.wide {
  match micro.vector(op = "add");
  param VW in [4..8];
  require VW == 8;
  require executor kind worker;
  bundle "b.wide";
  emit "e1";
}
)llkmap";

/// One producer feeding two consumers across the value `mid`. `producerOp` and
/// `consumerOp` are the node attributes the rules predicate on, so a fixture
/// can gate producer and consumer rules apart.
WorkloadGraph fanOutGraph(mlir::MLIRContext &context,
                          llvm::StringRef producerOp,
                          llvm::StringRef consumerOp) {
  WorkloadGraph graph;
  WorkloadValueId input =
      graph.addValue(WorkloadValue{0, mlir::Type(), "in", /*external=*/true});
  WorkloadValueId middle =
      graph.addValue(WorkloadValue{0, mlir::Type(), "mid", /*external=*/false});
  WorkloadValueId out1 =
      graph.addValue(WorkloadValue{0, mlir::Type(), "o1", /*external=*/false});
  WorkloadValueId out2 =
      graph.addValue(WorkloadValue{0, mlir::Type(), "o2", /*external=*/false});

  WorkloadNode producer;
  producer.opName = "micro.vector";
  producer.sourceOrdinal = 0;
  producer.attributes = vectorAttributes(context, producerOp);
  producer.inputs.push_back(WorkloadPort{input, mlir::Type(), std::nullopt});
  producer.outputs.push_back(WorkloadPort{middle, mlir::Type(), std::nullopt});
  graph.addNode(std::move(producer));

  WorkloadNode first;
  first.opName = "micro.vector";
  first.sourceOrdinal = 1;
  first.attributes = vectorAttributes(context, consumerOp);
  first.inputs.push_back(WorkloadPort{middle, mlir::Type(), std::nullopt});
  first.outputs.push_back(WorkloadPort{out1, mlir::Type(), std::nullopt});
  graph.addNode(std::move(first));

  WorkloadNode second;
  second.opName = "micro.vector";
  second.sourceOrdinal = 2;
  second.attributes = vectorAttributes(context, consumerOp);
  second.inputs.push_back(WorkloadPort{middle, mlir::Type(), std::nullopt});
  second.outputs.push_back(WorkloadPort{out2, mlir::Type(), std::nullopt});
  graph.addNode(std::move(second));

  graph.finalize();
  return graph;
}

/// Two producers feeding one consumer across the value `mid`.
WorkloadGraph fanInGraph(mlir::MLIRContext &context) {
  WorkloadGraph graph;
  WorkloadValueId in1 =
      graph.addValue(WorkloadValue{0, mlir::Type(), "in1", /*external=*/true});
  WorkloadValueId in2 =
      graph.addValue(WorkloadValue{0, mlir::Type(), "in2", /*external=*/true});
  WorkloadValueId middle =
      graph.addValue(WorkloadValue{0, mlir::Type(), "mid", /*external=*/false});
  WorkloadValueId out =
      graph.addValue(WorkloadValue{0, mlir::Type(), "out", /*external=*/false});

  WorkloadNode first;
  first.opName = "micro.vector";
  first.sourceOrdinal = 0;
  first.attributes = vectorAttributes(context, "produce");
  first.inputs.push_back(WorkloadPort{in1, mlir::Type(), std::nullopt});
  first.outputs.push_back(WorkloadPort{middle, mlir::Type(), std::nullopt});
  graph.addNode(std::move(first));

  WorkloadNode second;
  second.opName = "micro.vector";
  second.sourceOrdinal = 1;
  second.attributes = vectorAttributes(context, "produce");
  second.inputs.push_back(WorkloadPort{in2, mlir::Type(), std::nullopt});
  second.outputs.push_back(WorkloadPort{middle, mlir::Type(), std::nullopt});
  graph.addNode(std::move(second));

  WorkloadNode consumer;
  consumer.opName = "micro.vector";
  consumer.sourceOrdinal = 2;
  consumer.attributes = vectorAttributes(context, "consume");
  consumer.inputs.push_back(WorkloadPort{middle, mlir::Type(), std::nullopt});
  consumer.outputs.push_back(WorkloadPort{out, mlir::Type(), std::nullopt});
  graph.addNode(std::move(consumer));

  graph.finalize();
  return graph;
}

/// Two workers with disjoint memory visibility: a producer placed on e0/dram.0
/// cannot be read in place by a consumer on e1/acc.0, so the value must move.
/// The single dram.0 -> acc.0 link is the only route.
MachineModel fanMachine() {
  MachineModel model;
  model.target = "fan";
  model.executors = {{"e0", "worker", std::nullopt, {}, 1, {}},
                     {"e1", "worker", std::nullopt, {}, 1, {}}};
  MemoryNode dram;
  dram.id = "dram.0";
  dram.kind = "dram";
  dram.visibleFrom = "e0";
  dram.capacityBytes = 1u << 30;
  dram.alignmentBytes = 64;
  MemoryNode acc;
  acc.id = "acc.0";
  acc.kind = "acc";
  acc.visibleFrom = "e1";
  acc.capacityBytes = 1u << 20;
  acc.alignmentBytes = 64;
  model.memories = {dram, acc};
  TransferEngineNode dma;
  dma.id = "dma.0";
  dma.kind = "dma";
  dma.attachedTo = "e0";
  model.transferEngines = {dma};
  LinkEdge edge;
  edge.id = "dram_to_acc.0";
  edge.source = "dram.0";
  edge.destination = "acc.0";
  edge.bandwidthBytesPerCycle = 32;
  edge.latencyCycles = 10;
  edge.transactionBytes = 64;
  edge.transferEngines = {"dma.0"};
  model.links = {edge};
  return model;
}

/// A producer rule gated on `op = "produce"` (dram on e0) and a consumer rule
/// gated on `op = "consume"` (acc on e1), so each node binds exactly one
/// placement and the fan shapes are the only thing under test.
constexpr llvm::StringLiteral kFanRules = R"llkmap(
rule r.produce {
  match micro.vector(op = "produce");
  require executor kind worker;
  require memory kind dram;
  bundle "b.produce";
  emit "e1";
  cost 1;
}
rule r.consume {
  match micro.vector(op = "consume");
  require executor kind worker;
  require memory kind acc;
  bundle "b.consume";
  emit "e1";
  cost 1;
}
)llkmap";

/// Every node binds `sram.0`, so a single producer can serve several consumers
/// from the one placement it wrote.
constexpr llvm::StringLiteral kSharedReadRules = R"llkmap(
rule r.sram {
  match micro.vector(op = "add");
  require executor kind worker;
  require memory kind sram;
  bundle "b.sram";
  emit "e1";
  cost 1;
}
)llkmap";

//===----------------------------------------------------------------------===//
// Layout transforms driven by the bound layouts (phase-3 T4)
//===----------------------------------------------------------------------===//

/// Two layouts that solve under the search's default (rank-0) context, and are
/// genuinely different: `t.plain` writes the logical indices unchanged while
/// `t.blocked` blocks the second dimension by the solved `VW`.
constexpr llvm::StringLiteral kTwoLayouts = R"llkmap(
layout t.plain(int N) {
  param N in [1..4];
  require N == 1;
  map (m, n) -> (m, n);
}
layout t.blocked(int VW) {
  param VW in [4..8];
  require VW == 8;
  map (m, n) -> (m, floordiv(n, VW), mod(n, VW));
}
)llkmap";

/// `searchMachine` with the sram node declaring both layouts, so a transform
/// between them is legal in the memory both instances bind -- and so the pair's
/// layouts are the only reason a transform is needed at all.
MachineModel transformMachine() {
  MachineModel model = searchMachine();
  for (MemoryNode &memory : model.memories)
    if (memory.kind == "sram")
      memory.supportedLayouts = {"t.plain", "t.blocked"};
  return model;
}

/// Producer and consumer bind *different* layouts for the value they exchange.
/// The producer names the port that carries the crossing value (`result`, its
/// only output) and the consumer names its input, so both bindings are
/// attributable to the edge. Both rules declare their ports: a layout
/// requirement is attributed to the value its named port carries, and a rule
/// that names no port attributes nothing.
constexpr llvm::StringLiteral kTransformRules = R"llkmap(
rule r.produce {
  match micro.vector(op = "produce");
  require executor kind worker;
  require memory kind sram;
  require layout result satisfies t.plain;
  input "operand0";
  output "result";
  bundle "b.produce";
  emit "e1";
  cost 1;
}
rule r.consume {
  match micro.vector(op = "consume");
  require executor kind worker;
  require memory kind sram;
  require layout operand0 satisfies t.blocked;
  input "operand0";
  output "result";
  bundle "b.consume";
  emit "e1";
  cost 1;
}
)llkmap";

/// The control: the same two rules, both requiring `t.plain` for the crossing
/// value, so the endpoints agree and the pair must stay a direct connection.
constexpr llvm::StringLiteral kSameLayoutRules = R"llkmap(
rule r.produce {
  match micro.vector(op = "produce");
  require executor kind worker;
  require memory kind sram;
  require layout result satisfies t.plain;
  input "operand0";
  output "result";
  bundle "b.produce";
  emit "e1";
  cost 1;
}
rule r.consume {
  match micro.vector(op = "consume");
  require executor kind worker;
  require memory kind sram;
  require layout operand0 satisfies t.plain;
  input "operand0";
  output "result";
  bundle "b.consume";
  emit "e1";
  cost 1;
}
)llkmap";

/// The `avx2.mma_bf16` shape: the producer's layout requirement names an
/// *input* port (`lhs`), while the value crossing the edge is its `result`. The
/// producer therefore says nothing about the result's layout, and the edge must
/// inherit nothing from it -- even though the consumer asks for a different
/// layout.
constexpr llvm::StringLiteral kMmaShapedRules = R"llkmap(
rule r.produce {
  match micro.vector(op = "produce");
  require executor kind worker;
  require memory kind sram;
  require layout lhs satisfies t.plain;
  input "lhs";
  output "result";
  bundle "b.produce";
  emit "e1";
  cost 1;
}
rule r.consume {
  match micro.vector(op = "consume");
  require executor kind worker;
  require memory kind sram;
  require layout operand0 satisfies t.blocked;
  input "operand0";
  output "result";
  bundle "b.consume";
  emit "e1";
  cost 1;
}
)llkmap";

/// One producer whose result feeds *both* operand ports of one consumer, over
/// the value `mid`. The two operand uses are distinct obligations even though
/// they carry one SSA value.
WorkloadGraph repeatedOperandGraph(mlir::MLIRContext &context) {
  WorkloadGraph graph;
  WorkloadValueId input =
      graph.addValue(WorkloadValue{0, mlir::Type(), "in", /*external=*/true});
  WorkloadValueId middle =
      graph.addValue(WorkloadValue{0, mlir::Type(), "mid", /*external=*/false});
  WorkloadValueId output =
      graph.addValue(WorkloadValue{0, mlir::Type(), "out", /*external=*/false});

  WorkloadNode producer;
  producer.opName = "micro.vector";
  producer.attributes = vectorAttributes(context, "produce");
  producer.inputs.push_back(WorkloadPort{input, mlir::Type(), std::nullopt});
  producer.outputs.push_back(WorkloadPort{middle, mlir::Type(), std::nullopt});
  graph.addNode(std::move(producer));

  WorkloadNode consumer;
  consumer.opName = "micro.vector";
  consumer.attributes = vectorAttributes(context, "consume");
  consumer.inputs.push_back(WorkloadPort{middle, mlir::Type(), std::nullopt});
  consumer.inputs.push_back(WorkloadPort{middle, mlir::Type(), std::nullopt});
  consumer.outputs.push_back(WorkloadPort{output, mlir::Type(), std::nullopt});
  graph.addNode(std::move(consumer));

  graph.finalize();
  return graph;
}

/// The producer's result is plain; the same value feeds both consumer ports,
/// `operand0` requiring plain and `operand1` requiring blocked. The two uses
/// are incompatible representations, so each must get its own connection.
constexpr llvm::StringLiteral kRepeatedConflictRules = R"llkmap(
rule r.produce {
  match micro.vector(op = "produce");
  require executor kind worker;
  require memory kind sram;
  require layout result satisfies t.plain;
  input "operand0";
  output "result";
  bundle "b.produce";
  emit "e1";
  cost 1;
}
rule r.consume {
  match micro.vector(op = "consume");
  require executor kind worker;
  require memory kind sram;
  require layout operand0 satisfies t.plain;
  require layout operand1 satisfies t.blocked;
  input "operand0";
  input "operand1";
  output "result";
  bundle "b.consume";
  emit "e1";
  cost 1;
}
)llkmap";

/// The same-family control: both uses require `t.blocked` (one family,
/// one solved parameterization) and the producer's result is blocked too, so
/// the two uses share one direct connection.
constexpr llvm::StringLiteral kRepeatedSameLayoutRules = R"llkmap(
rule r.produce {
  match micro.vector(op = "produce");
  require executor kind worker;
  require memory kind sram;
  require layout result satisfies t.blocked;
  input "operand0";
  output "result";
  bundle "b.produce";
  emit "e1";
  cost 1;
}
rule r.consume {
  match micro.vector(op = "consume");
  require executor kind worker;
  require memory kind sram;
  require layout operand0 satisfies t.blocked;
  require layout operand1 satisfies t.blocked;
  input "operand0";
  input "operand1";
  output "result";
  bundle "b.consume";
  emit "e1";
  cost 1;
}
)llkmap";

/// Two *different* values (`v0`, `v1`) feeding the two operand ports of one
/// consumer, each requiring the same family `t.plain`. Each edge resolves its
/// own layout independently.
WorkloadGraph twoValueSameLayoutGraph(mlir::MLIRContext &context) {
  WorkloadGraph graph;
  WorkloadValueId in0 =
      graph.addValue(WorkloadValue{0, mlir::Type(), "in0", /*external=*/true});
  WorkloadValueId in1 =
      graph.addValue(WorkloadValue{0, mlir::Type(), "in1", /*external=*/true});
  WorkloadValueId v0 =
      graph.addValue(WorkloadValue{0, mlir::Type(), "v0", /*external=*/false});
  WorkloadValueId v1 =
      graph.addValue(WorkloadValue{0, mlir::Type(), "v1", /*external=*/false});
  WorkloadValueId output =
      graph.addValue(WorkloadValue{0, mlir::Type(), "out", /*external=*/false});

  auto producer = [&](unsigned ordinal, WorkloadValueId input,
                      WorkloadValueId result) {
    WorkloadNode node;
    node.opName = "micro.vector";
    node.sourceOrdinal = ordinal;
    node.attributes = vectorAttributes(context, "produce");
    node.inputs.push_back(WorkloadPort{input, mlir::Type(), std::nullopt});
    node.outputs.push_back(WorkloadPort{result, mlir::Type(), std::nullopt});
    graph.addNode(std::move(node));
  };
  producer(0, in0, v0);
  producer(1, in1, v1);

  WorkloadNode consumer;
  consumer.opName = "micro.vector";
  consumer.sourceOrdinal = 2;
  consumer.attributes = vectorAttributes(context, "consume");
  consumer.inputs.push_back(WorkloadPort{v0, mlir::Type(), std::nullopt});
  consumer.inputs.push_back(WorkloadPort{v1, mlir::Type(), std::nullopt});
  consumer.outputs.push_back(WorkloadPort{output, mlir::Type(), std::nullopt});
  graph.addNode(std::move(consumer));

  graph.finalize();
  return graph;
}

/// Two producers, each plain, feeding one consumer whose two operands both want
/// `t.plain`: one family, two independent edges.
constexpr llvm::StringLiteral kTwoValuePlainRules = R"llkmap(
rule r.produce {
  match micro.vector(op = "produce");
  require executor kind worker;
  require memory kind sram;
  require layout result satisfies t.plain;
  input "operand0";
  output "result";
  bundle "b.produce";
  emit "e1";
  cost 1;
}
rule r.consume {
  match micro.vector(op = "consume");
  require executor kind worker;
  require memory kind sram;
  require layout operand0 satisfies t.plain;
  require layout operand1 satisfies t.plain;
  input "operand0";
  input "operand1";
  output "result";
  bundle "b.consume";
  emit "e1";
  cost 1;
}
)llkmap";

/// The search cost of one producer -> consumer transfer over `fanMachine`:
/// the link latency (10) plus 4096 assumed bytes at 32 bytes/cycle.
constexpr double kFanTransferCycles = 10.0 + 4096.0 / 32.0;

/// Two producers and two consumers of one value: the multi-producer,
/// multi-consumer shape.
WorkloadGraph fanInFanOutGraph(mlir::MLIRContext &context) {
  WorkloadGraph graph;
  WorkloadValueId in1 =
      graph.addValue(WorkloadValue{0, mlir::Type(), "in1", /*external=*/true});
  WorkloadValueId in2 =
      graph.addValue(WorkloadValue{0, mlir::Type(), "in2", /*external=*/true});
  WorkloadValueId middle =
      graph.addValue(WorkloadValue{0, mlir::Type(), "mid", /*external=*/false});
  WorkloadValueId out1 =
      graph.addValue(WorkloadValue{0, mlir::Type(), "o1", /*external=*/false});
  WorkloadValueId out2 =
      graph.addValue(WorkloadValue{0, mlir::Type(), "o2", /*external=*/false});

  auto producer = [&](WorkloadValueId input, unsigned ordinal) {
    WorkloadNode node;
    node.opName = "micro.vector";
    node.sourceOrdinal = ordinal;
    node.attributes = vectorAttributes(context, "produce");
    node.inputs.push_back(WorkloadPort{input, mlir::Type(), std::nullopt});
    node.outputs.push_back(WorkloadPort{middle, mlir::Type(), std::nullopt});
    graph.addNode(std::move(node));
  };
  producer(in1, 0);
  producer(in2, 1);

  auto consumer = [&](WorkloadValueId output, unsigned ordinal) {
    WorkloadNode node;
    node.opName = "micro.vector";
    node.sourceOrdinal = ordinal;
    node.attributes = vectorAttributes(context, "consume");
    node.inputs.push_back(WorkloadPort{middle, mlir::Type(), std::nullopt});
    node.outputs.push_back(WorkloadPort{output, mlir::Type(), std::nullopt});
    graph.addNode(std::move(node));
  };
  consumer(out1, 2);
  consumer(out2, 3);

  graph.finalize();
  return graph;
}

/// One producer and two consumers of one value, where each consumer is gated
/// by its own operation so it binds a distinct destination memory.
WorkloadGraph twoConsumerGraph(mlir::MLIRContext &context) {
  WorkloadGraph graph;
  WorkloadValueId input =
      graph.addValue(WorkloadValue{0, mlir::Type(), "in", /*external=*/true});
  WorkloadValueId middle =
      graph.addValue(WorkloadValue{0, mlir::Type(), "mid", /*external=*/false});
  WorkloadValueId out1 =
      graph.addValue(WorkloadValue{0, mlir::Type(), "o1", /*external=*/false});
  WorkloadValueId out2 =
      graph.addValue(WorkloadValue{0, mlir::Type(), "o2", /*external=*/false});

  WorkloadNode producer;
  producer.opName = "micro.vector";
  producer.sourceOrdinal = 0;
  producer.attributes = vectorAttributes(context, "produce");
  producer.inputs.push_back(WorkloadPort{input, mlir::Type(), std::nullopt});
  producer.outputs.push_back(WorkloadPort{middle, mlir::Type(), std::nullopt});
  graph.addNode(std::move(producer));

  WorkloadNode first;
  first.opName = "micro.vector";
  first.sourceOrdinal = 1;
  first.attributes = vectorAttributes(context, "consume_a");
  first.inputs.push_back(WorkloadPort{middle, mlir::Type(), std::nullopt});
  first.outputs.push_back(WorkloadPort{out1, mlir::Type(), std::nullopt});
  graph.addNode(std::move(first));

  WorkloadNode second;
  second.opName = "micro.vector";
  second.sourceOrdinal = 2;
  second.attributes = vectorAttributes(context, "consume_b");
  second.inputs.push_back(WorkloadPort{middle, mlir::Type(), std::nullopt});
  second.outputs.push_back(WorkloadPort{out2, mlir::Type(), std::nullopt});
  graph.addNode(std::move(second));

  graph.finalize();
  return graph;
}

/// Two consumers that must each copy, into two distinct destination memories,
/// both reached only through the same staging memory: dram.0 -> stage.0 ->
/// {acc.0, aux.0}. `stage.0` is large enough for the router to accept each hop
/// (>= 4096 bytes) but too small to hold both staged copies, so whether the
/// search charges intermediate hops is observable.
MachineModel twoDestinationHopMachine() {
  MachineModel model;
  model.target = "fan-hop";
  model.executors = {{"e0", "worker", std::nullopt, {}, 1, {}},
                     {"e1", "worker", std::nullopt, {}, 1, {}}};
  auto memory = [](llvm::StringRef id, llvm::StringRef kind,
                   llvm::StringRef visible, uint64_t capacity) {
    MemoryNode node;
    node.id = id.str();
    node.kind = kind.str();
    node.visibleFrom = visible.str();
    node.capacityBytes = capacity;
    node.alignmentBytes = 64;
    return node;
  };
  model.memories = {memory("dram.0", "dram", "e0", 1u << 30),
                    memory("stage.0", "sram", "e0", 5000),
                    memory("acc.0", "acc", "e1", 1u << 20),
                    memory("aux.0", "aux", "e1", 1u << 20)};
  TransferEngineNode dma;
  dma.id = "dma.0";
  dma.kind = "dma";
  dma.attachedTo = "e0";
  model.transferEngines = {dma};
  auto link = [](llvm::StringRef id, llvm::StringRef source,
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
  };
  model.links = {link("dram_to_stage.0", "dram.0", "stage.0"),
                 link("stage_to_acc.0", "stage.0", "acc.0"),
                 link("stage_to_aux.0", "stage.0", "aux.0")};
  return model;
}

/// One producer rule and two consumer rules, each consumer gated on its own
/// operation so it binds a different memory kind (acc vs aux).
constexpr llvm::StringLiteral kTwoConsumerRules = R"llkmap(
rule r.produce {
  match micro.vector(op = "produce");
  require executor kind worker;
  require memory kind dram;
  bundle "b.produce";
  emit "e1";
  cost 1;
}
rule r.consume_a {
  match micro.vector(op = "consume_a");
  require executor kind worker;
  require memory kind acc;
  bundle "b.consume_a";
  emit "e1";
  cost 1;
}
rule r.consume_b {
  match micro.vector(op = "consume_b");
  require executor kind worker;
  require memory kind aux;
  bundle "b.consume_b";
  emit "e1";
  cost 1;
}
)llkmap";

/// One producer rule and one consumer rule over the `twoDestinationHopMachine`
/// topology, plus a third rule that parks an instance on the staging sram. The
/// `a_`/`m_`/`z_` operation-name prefixes sort the content-keyed node order to
/// hold -> produce -> consume, so `a_hold`'s live tile is charged to the
/// staging memory before the producer -> consumer connection is synthesized.
constexpr llvm::StringLiteral kOccupiedStagingRules = R"llkmap(
rule r.hold {
  match micro.vector(op = "a_hold");
  require executor kind worker;
  require memory kind sram;
  bundle "b.hold";
  emit "e1";
  cost 1;
}
rule r.produce {
  match micro.vector(op = "m_produce");
  require executor kind worker;
  require memory kind dram;
  bundle "b.produce";
  emit "e1";
  cost 1;
}
rule r.consume {
  match micro.vector(op = "z_consume");
  require executor kind worker;
  require memory kind acc;
  bundle "b.consume";
  emit "e1";
  cost 1;
}
)llkmap";

/// `a_hold -> {hold_out}`, `m_produce -> mid -> z_consume`, over `tile`.
/// `a_hold` binds the staging sram and its output tile stays live there (no
/// consumer ever releases it); `m_produce` binds dram and `z_consume` binds
/// acc, so the one crossing value `mid` can only reach its consumer by staging
/// through the memory `a_hold` already occupies.
WorkloadGraph occupiedStagingGraph(mlir::MLIRContext &context,
                                   mlir::Type tile) {
  WorkloadGraph graph;
  WorkloadValueId holdIn =
      graph.addValue(WorkloadValue{0, tile, "hold_in", /*external=*/true});
  WorkloadValueId holdOut =
      graph.addValue(WorkloadValue{0, tile, "hold_out", /*external=*/false});
  WorkloadValueId in =
      graph.addValue(WorkloadValue{0, tile, "in", /*external=*/true});
  WorkloadValueId mid =
      graph.addValue(WorkloadValue{0, tile, "mid", /*external=*/false});
  WorkloadValueId out =
      graph.addValue(WorkloadValue{0, tile, "out", /*external=*/false});

  auto node = [&](unsigned ordinal, llvm::StringRef op, WorkloadValueId input,
                  WorkloadValueId output) {
    WorkloadNode n;
    n.opName = "micro.vector";
    n.sourceOrdinal = ordinal;
    n.attributes = vectorAttributes(context, op);
    n.inputs.push_back(WorkloadPort{input, tile, std::nullopt});
    n.outputs.push_back(WorkloadPort{output, tile, std::nullopt});
    graph.addNode(std::move(n));
  };
  node(0, "a_hold", holdIn, holdOut);
  node(1, "m_produce", in, mid);
  node(2, "z_consume", mid, out);

  graph.finalize();
  return graph;
}

std::unique_ptr<MappingTarget> targetWith(MachineModel machine,
                                          llvm::StringRef rules) {
  llvm::Expected<RuleRegistry> registry = parseRuleText(rules, "<test>");
  if (!registry)
    return nullptr;
  return std::make_unique<FileMappingTarget>(
      "test", std::move(machine), LayoutRegistry{}, std::move(*registry),
      std::vector<std::string>{"e1"});
}

/// `targetWith`, with a layout library so a rule's layout requirement can be
/// solved (or fail to be).
std::unique_ptr<MappingTarget> targetWithLayouts(MachineModel machine,
                                                 llvm::StringRef rules,
                                                 llvm::StringRef layouts) {
  llvm::Expected<RuleRegistry> ruleRegistry = parseRuleText(rules, "<test>");
  if (!ruleRegistry)
    return nullptr;
  llvm::Expected<LayoutRegistry> layoutRegistry =
      parseLayoutText(layouts, "<test>");
  if (!layoutRegistry)
    return nullptr;
  return std::make_unique<FileMappingTarget>(
      "test", std::move(machine), std::move(*layoutRegistry),
      std::move(*ruleRegistry), std::vector<std::string>{"e1"});
}

/// A provider that answers from a table keyed by rule id, and records what it
/// was asked for.
class FixedLatencyProvider : public LatencyProvider {
public:
  std::map<std::string, double> byRule;
  mutable std::vector<std::string> lookups;
  /// How many times `lookupCycles` was entered, so a test can prove the search
  /// did (or did not) consult the provider at all.
  mutable size_t lookupCount = 0;

  std::optional<double> lookupCycles(const OperationSignature &signature,
                                     const TargetContext &) const override {
    ++lookupCount;
    lookups.push_back(signature.canonicalString());
    auto it = byRule.find(signature.rule);
    if (it == byRule.end())
      return std::nullopt;
    return it->second;
  }
};

std::unique_ptr<MappingTarget>
targetWithProvider(MachineModel machine, llvm::StringRef rules,
                   const LatencyProvider *provider) {
  llvm::Expected<RuleRegistry> registry = parseRuleText(rules, "<test>");
  if (!registry)
    return nullptr;
  return std::make_unique<FileMappingTarget>(
      "test", std::move(machine), LayoutRegistry{}, std::move(*registry),
      std::vector<std::string>{"e1"}, provider);
}

std::vector<PlanId> planIds(const MappingSearchResult &result) {
  std::vector<PlanId> ids;
  for (const CoveringPlan &plan : result.plans)
    ids.push_back(plan.id);
  return ids;
}

bool hasDiagnostic(const MappingSearchResult &result, DiagnosticCode code) {
  for (const Diagnostic &diagnostic : result.frontier.diagnostics)
    if (diagnostic.code == code)
      return true;
  return false;
}

llvm::StringMap<SearchValue>
values(std::initializer_list<std::pair<llvm::StringRef, SearchValue>> entries) {
  llvm::StringMap<SearchValue> map;
  for (const auto &entry : entries)
    map[entry.first] = entry.second;
  return map;
}

/// The plan id the deterministic two-node fixture produces when a zero binding
/// hash and no parameters are folded in. Pinned so the no-binding path cannot
/// drift silently; it changes only when the canonical plan form does.
///
/// Re-baselined for phase-3 T4 (ruling R4): the canonical instance and plan
/// forms each gained a `solvedlayout=` field carrying the solved layout
/// parameter assignment. The field is present for *every* instance and
/// placement -- empty when the rule requires no layout -- so every plan id
/// moves, not only those of plans that bind a layout. This fixture binds none,
/// which is exactly why it is still the right pin for "the form changed and
/// nothing else did".
// Pinned with endpoint occurrences in the identity: a candidate's ports and a
// connection's producer/consumer ports now join their content keys, so two uses
// of one value are distinct. The plan id changed once when those were resolved
// (task A2); from then on it is content-stable.
constexpr PlanId kNoBindingPlanId = 16999786886447551871ULL;

//===----------------------------------------------------------------------===//
// Search bound fixtures
//===----------------------------------------------------------------------===//

/// Both rules declare the *same* static 100-cycle cost, so a bound that reads
/// the static estimate cannot tell them apart -- only the measured cost can.
/// `r.a_static` sorts first, which pins the depth-first instance order.
constexpr llvm::StringLiteral kMeasuredBoundRules = R"llkmap(
rule r.a_static {
  match micro.vector(op = "add");
  require executor kind worker;
  bundle "b.a";
  emit "e1";
  cost 100;
}
rule r.b_measured {
  match micro.vector(op = "add");
  require executor kind worker;
  bundle "b.b";
  emit "e1";
  cost 100;
}
)llkmap";

/// Two workers under a shared `root` scope, joined by a near-free
/// `dram.0 -> sram.0` link. Both workers can address `sram.0`, but only `e0`
/// can address `dram.0`, so a consumer on `e1` must *move* a DRAM value rather
/// than read it in place. The route latency stays small, so latency and DRAM
/// pull in opposite directions.
MachineModel objectiveMachine() {
  MachineModel model;
  model.target = "objective";
  model.executors = {{"root", "cluster", std::nullopt, {}, 1, {}},
                     {"e0", "a", std::string("root"), {}, 1, {}},
                     {"e1", "b", std::string("root"), {}, 1, {}}};
  MemoryNode sram;
  sram.id = "sram.0";
  sram.kind = "sram";
  sram.visibleFrom = "root";
  sram.capacityBytes = 1u << 20;
  sram.alignmentBytes = 64;
  MemoryNode dram;
  dram.id = "dram.0";
  dram.kind = "dram";
  dram.visibleFrom = "e0";
  dram.capacityBytes = 1u << 30;
  dram.alignmentBytes = 64;
  model.memories = {sram, dram};
  TransferEngineNode dma;
  dma.id = "dma.0";
  dma.kind = "dma";
  dma.attachedTo = "e0";
  model.transferEngines = {dma};
  LinkEdge edge;
  edge.id = "dram_to_sram.0";
  edge.source = "dram.0";
  edge.destination = "sram.0";
  edge.bandwidthBytesPerCycle = 4096;
  edge.latencyCycles = 0;
  edge.transactionBytes = 4096;
  edge.transferEngines = {"dma.0"};
  model.links = {edge};
  return model;
}

/// The consumer is pinned to a 1-cycle SRAM placement on `e1`. The producer
/// chooses between a 1-cycle DRAM placement (which the consumer cannot read in
/// place, so 4096 bytes cross DRAM) and a 50-cycle SRAM placement (read in
/// place, 0 DRAM bytes). Minimizing latency picks the DRAM plan, minimizing
/// DRAM the SRAM one -- opposite answers, so the bound must follow the metric.
constexpr llvm::StringLiteral kObjectiveRules = R"llkmap(
rule r.p_dram {
  match micro.vector(op = "produce");
  require executor kind a;
  require memory kind dram;
  bundle "b.pd";
  emit "e1";
  cost 1;
}
rule r.p_sram {
  match micro.vector(op = "produce");
  require executor kind a;
  require memory kind sram;
  bundle "b.ps";
  emit "e1";
  cost 50;
}
rule r.c_sram {
  match micro.vector(op = "consume");
  require executor kind b;
  require memory kind sram;
  bundle "b.c";
  emit "e1";
  cost 1;
}
)llkmap";

/// Both producer placements reach the consumer in 51 cycles, but on different
/// metrics: `r.p_dram` pays 49 in place plus a 1-cycle 4096-byte move across
/// `dram.0 -> sram.0`, `r.p_sram` pays 50 and the consumer reads it in place.
/// The two plans tie on latency and differ only on `dram_bytes` (4096 vs 0), so
/// a primary-only comparison cannot tell them apart -- the declared secondary
/// metric must. `r.p_dram` is declared first (and sorts first), so its plan
/// completes first and becomes the plan a tie-breaking prune would defend.
constexpr llvm::StringLiteral kTieRules = R"llkmap(
rule r.p_dram {
  match micro.vector(op = "produce");
  require executor kind a;
  require memory kind dram;
  bundle "b.pd";
  emit "e1";
  cost 49;
}
rule r.p_sram {
  match micro.vector(op = "produce");
  require executor kind a;
  require memory kind sram;
  bundle "b.ps";
  emit "e1";
  cost 50;
}
rule r.c_sram {
  match micro.vector(op = "consume");
  require executor kind b;
  require memory kind sram;
  bundle "b.c";
  emit "e1";
  cost 1;
}
)llkmap";

/// Two producer destinations feeding one consumer memory over links of very
/// different latency (11 vs 101 cycles for the 4096-byte value). The consumer
/// on `e1` sees neither producer memory, so every plan moves the value. A
/// plan's connection cost is what can make it largest, and that cost is unknown
/// while any node is still uncovered.
MachineModel maximizeMachine() {
  MachineModel model;
  model.target = "maximize";
  model.executors = {{"e0", "a", std::nullopt, {}, 1, {}},
                     {"e1", "b", std::nullopt, {}, 1, {}}};
  MemoryNode dram;
  dram.id = "dram.0";
  dram.kind = "dram";
  dram.visibleFrom = "e0";
  dram.capacityBytes = 1u << 30;
  dram.alignmentBytes = 64;
  MemoryNode acc;
  acc.id = "acc.0";
  acc.kind = "acc";
  acc.visibleFrom = "e0";
  acc.capacityBytes = 1u << 30;
  acc.alignmentBytes = 64;
  MemoryNode sram;
  sram.id = "sram.0";
  sram.kind = "sram";
  sram.visibleFrom = "e1";
  sram.capacityBytes = 1u << 20;
  sram.alignmentBytes = 64;
  model.memories = {dram, acc, sram};
  TransferEngineNode dma;
  dma.id = "dma.0";
  dma.kind = "dma";
  dma.attachedTo = "e0";
  model.transferEngines = {dma};
  LinkEdge fastEdge;
  fastEdge.id = "dram_to_sram.0";
  fastEdge.source = "dram.0";
  fastEdge.destination = "sram.0";
  fastEdge.bandwidthBytesPerCycle = 4096;
  fastEdge.latencyCycles = 10;
  fastEdge.transactionBytes = 4096;
  fastEdge.transferEngines = {"dma.0"};
  LinkEdge slowEdge;
  slowEdge.id = "acc_to_sram.0";
  slowEdge.source = "acc.0";
  slowEdge.destination = "sram.0";
  slowEdge.bandwidthBytesPerCycle = 4096;
  slowEdge.latencyCycles = 100;
  slowEdge.transactionBytes = 4096;
  slowEdge.transferEngines = {"dma.0"};
  model.links = {fastEdge, slowEdge};
  return model;
}

/// `r.p_fast` reaches the consumer in 11 cycles, `r.p_slow` in 101; both
/// declare the same 1-cycle instance cost, so only the connection distinguishes
/// them. The operation names sort the producer to node 0, pinning the
/// depth-first order the exact search follows.
constexpr llvm::StringLiteral kMaximizeRules = R"llkmap(
rule r.p_fast {
  match micro.vector(op = "a_produce");
  require executor kind a;
  require memory kind dram;
  bundle "b.pf";
  emit "e1";
  cost 1;
}
rule r.p_slow {
  match micro.vector(op = "a_produce");
  require executor kind a;
  require memory kind acc;
  bundle "b.ps";
  emit "e1";
  cost 1;
}
rule r.c_dest {
  match micro.vector(op = "z_consume");
  require executor kind b;
  require memory kind sram;
  bundle "b.c";
  emit "e1";
  cost 1;
}
)llkmap";

} // namespace

TEST(CoveringSearch, DeterministicReturnsTheFirstCompletePlan) {
  mlir::MLIRContext context;
  WorkloadGraph graph = twoNodeGraph(context);
  std::unique_ptr<MappingTarget> target = targetWith(searchMachine(), kRules);
  ASSERT_NE(target, nullptr);

  MappingSearchOptions options;
  options.mode = SearchMode::Deterministic;
  CoveringSearch search(graph, *target, context, LayoutContext{}, options);
  llvm::Expected<MappingSearchResult> result = search.search();
  ASSERT_TRUE(static_cast<bool>(result)) << llvm::toString(result.takeError());
  ASSERT_EQ(result->plans.size(), 1u);
  EXPECT_EQ(result->plans[0].instances.size(), 2u);
  EXPECT_EQ(result->plans[0].connections.size(), 1u);
  EXPECT_FALSE(result->searchTruncated);
}

TEST(CoveringSearch, PlacementsAreOrderedByTheirBindingTuple) {
  mlir::MLIRContext context;
  WorkloadGraph graph = oppositeBindingGraph(context);
  std::unique_ptr<MappingTarget> target =
      targetWith(oppositeOrderMachine(), kExecutorRules);
  ASSERT_NE(target, nullptr);

  MappingSearchOptions options;
  options.mode = SearchMode::Deterministic;
  CoveringSearch search(graph, *target, context, LayoutContext{}, options);
  llvm::Expected<MappingSearchResult> result = search.search();
  ASSERT_TRUE(static_cast<bool>(result)) << llvm::toString(result.takeError());
  ASSERT_FALSE(result->plans.empty());
  const llvm::SmallVector<PlanPlacement> &placements =
      result->plans[0].placements;
  ASSERT_EQ(placements.size(), 2u);

  // §22.1 orders placements by the (executor, memory, layout) binding tuple,
  // not by node: executor `a0` sorts before `z0`, and it belongs to the node
  // with the greater id.
  EXPECT_EQ(placements[0].executor, "a0");
  EXPECT_EQ(placements[0].node, 1u);
  EXPECT_EQ(placements[1].executor, "z0");
  EXPECT_EQ(placements[1].node, 0u);
}

TEST(CoveringSearch, ReportsNodesWithoutRules) {
  mlir::MLIRContext context;
  WorkloadGraph graph = twoNodeGraph(context);
  std::unique_ptr<MappingTarget> target = targetWith(searchMachine(), "");
  ASSERT_NE(target, nullptr);

  CoveringSearch search(graph, *target, context, LayoutContext{});
  llvm::Expected<MappingSearchResult> result = search.search();
  ASSERT_TRUE(static_cast<bool>(result)) << llvm::toString(result.takeError());
  EXPECT_TRUE(result->plans.empty());
  EXPECT_EQ(result->frontier.nodesWithoutRules, 2u);
  EXPECT_FALSE(result->frontier.diagnostics.empty());
}

TEST(CoveringSearch, ReportsCandidatesWithoutPlacement) {
  mlir::MLIRContext context;
  WorkloadGraph graph = twoNodeGraph(context);
  std::unique_ptr<MappingTarget> target =
      targetWith(searchMachine(), kNoPlacementRules);
  ASSERT_NE(target, nullptr);

  CoveringSearch search(graph, *target, context, LayoutContext{});
  llvm::Expected<MappingSearchResult> result = search.search();
  ASSERT_TRUE(static_cast<bool>(result)) << llvm::toString(result.takeError());
  EXPECT_TRUE(result->plans.empty());
  EXPECT_EQ(result->frontier.candidatesWithoutPlacement, 2u);
}

TEST(CoveringSearch, ReportsARuleWhoseConstraintSearchWasTruncated) {
  mlir::MLIRContext context;
  WorkloadGraph graph = twoNodeGraph(context);
  // A rule whose parameter space exceeds the assignment cap, with a constraint
  // no assignment satisfies. The rule is not proven inapplicable, so the
  // search is truncated rather than silently reporting "no matching rule".
  constexpr llvm::StringLiteral kBigRule = R"llkmap(
rule r.big {
  match micro.vector(op = "add");
  param A in [1..400];
  param B in [1..400];
  require A == 999999;
  bundle "b.big";
  emit "e1";
}
)llkmap";
  std::unique_ptr<MappingTarget> target = targetWith(searchMachine(), kBigRule);
  ASSERT_NE(target, nullptr);

  CoveringSearch search(graph, *target, context, LayoutContext{});
  llvm::Expected<MappingSearchResult> result = search.search();
  ASSERT_TRUE(static_cast<bool>(result)) << llvm::toString(result.takeError());
  EXPECT_TRUE(result->plans.empty());
  EXPECT_TRUE(result->searchTruncated);
  EXPECT_EQ(result->frontier.nodesWithoutRules, 2u);
  ASSERT_FALSE(result->frontier.diagnostics.empty());
  // The cap reason reaches the frontier, where the pass surfaces it.
  bool mentionsCap = false;
  for (const Diagnostic &diagnostic : result->frontier.diagnostics)
    mentionsCap |= diagnostic.message.find("assignments") != std::string::npos;
  EXPECT_TRUE(mentionsCap);
}

TEST(CoveringSearch, WideBeamAndExactAgreeOnTheBestPlan) {
  mlir::MLIRContext context;
  WorkloadGraph graph = twoNodeGraph(context);
  std::unique_ptr<MappingTarget> target = targetWith(searchMachine(), kRules);
  ASSERT_NE(target, nullptr);

  MappingSearchOptions beamOptions;
  beamOptions.mode = SearchMode::Beam;
  beamOptions.beamWidth = 64;
  CoveringSearch beam(graph, *target, context, LayoutContext{}, beamOptions);
  llvm::Expected<MappingSearchResult> beamResult = beam.search();
  ASSERT_TRUE(static_cast<bool>(beamResult))
      << llvm::toString(beamResult.takeError());

  MappingSearchOptions exactOptions;
  exactOptions.mode = SearchMode::Exact;
  CoveringSearch exact(graph, *target, context, LayoutContext{}, exactOptions);
  llvm::Expected<MappingSearchResult> exactResult = exact.search();
  ASSERT_TRUE(static_cast<bool>(exactResult))
      << llvm::toString(exactResult.takeError());

  ASSERT_FALSE(beamResult->plans.empty());
  ASSERT_FALSE(exactResult->plans.empty());
  EXPECT_EQ(beamResult->plans[0].id, exactResult->plans[0].id);
  // The cheapest rule costs 1 per node, so the best plan is 2 (plus nothing:
  // both instances share a memory, so the connection is direct and free).
  EXPECT_DOUBLE_EQ(exactResult->plans[0].totalCost.latencyCycles, 2.0);
}

TEST(CoveringSearch, EveryPlanIsRankedAndCappedAtTopK) {
  mlir::MLIRContext context;
  WorkloadGraph graph = twoNodeGraph(context);
  std::unique_ptr<MappingTarget> target = targetWith(searchMachine(), kRules);
  ASSERT_NE(target, nullptr);

  MappingSearchOptions options;
  options.mode = SearchMode::Exact;
  options.topK = 1;
  CoveringSearch search(graph, *target, context, LayoutContext{}, options);
  llvm::Expected<MappingSearchResult> result = search.search();
  ASSERT_TRUE(static_cast<bool>(result));
  EXPECT_EQ(result->plans.size(), 1u);
  // Two rules per node give four plans; asking for one is a cap.
  EXPECT_TRUE(result->searchTruncated);
}

// §22.1: the retained top-K is chosen by the documented (objective, plan id)
// key. The beam's frontier order over *partial* plans is a search heuristic and
// must not decide which K survive: with many cost-tied plans the two orders
// disagree, so trimming before the exposed id exists keeps the wrong K.
TEST(CoveringSearch, TopKRetainsThePlanIdSmallestAmongCostTies) {
  mlir::MLIRContext context;
  WorkloadGraph graph = twoNodeGraph(context);
  std::unique_ptr<MappingTarget> target =
      targetWith(searchMachine(), kDenseTieRules);
  ASSERT_NE(target, nullptr);

  // Six rules match but the candidate cap keeps five, so the search is already
  // truncated before the top-K decision -- both runs below fold the same
  // `truncated` bit into every plan id, which is what makes their ids directly
  // comparable.
  MappingSearchOptions wideOptions;
  wideOptions.mode = SearchMode::Beam;
  wideOptions.beamWidth = 64;
  wideOptions.topK = 64;
  wideOptions.maxCandidatesPerNode = 5;
  CoveringSearch wide(graph, *target, context, LayoutContext{}, wideOptions);
  llvm::Expected<MappingSearchResult> wideResult = wide.search();
  ASSERT_TRUE(static_cast<bool>(wideResult))
      << llvm::toString(wideResult.takeError());
  ASSERT_TRUE(wideResult->searchTruncated);
  // Five interchangeable instances per node over two nodes: twenty-five
  // complete plans, every one tied at the same cost.
  ASSERT_EQ(wideResult->plans.size(), 25u);
  ASSERT_EQ(wideResult->planCount, 25u);
  // With every cost equal, the emitted order is exactly the plan-id order.
  for (size_t i = 1; i < wideResult->plans.size(); ++i)
    EXPECT_LT(wideResult->plans[i - 1].id, wideResult->plans[i].id);

  // A capped run over the same complete set must retain the plan-id-smallest K.
  const unsigned k = 2;
  MappingSearchOptions cappedOptions = wideOptions;
  cappedOptions.topK = k;
  CoveringSearch capped(graph, *target, context, LayoutContext{},
                        cappedOptions);
  llvm::Expected<MappingSearchResult> cappedResult = capped.search();
  ASSERT_TRUE(static_cast<bool>(cappedResult))
      << llvm::toString(cappedResult.takeError());
  ASSERT_EQ(cappedResult->plans.size(), k);
  // The tally still counts every complete plan before the cap (design §22.2).
  EXPECT_EQ(cappedResult->planCount, 25u);
  EXPECT_TRUE(cappedResult->searchTruncated);
  EXPECT_TRUE(hasDiagnostic(*cappedResult, DiagnosticCode::SearchTruncated));
  for (unsigned i = 0; i < k; ++i)
    EXPECT_EQ(cappedResult->plans[i].id, wideResult->plans[i].id);
}

TEST(CoveringSearch, NarrowBeamDisclosesTruncationAndExactDoesNot) {
  mlir::MLIRContext context;
  WorkloadGraph graph = twoNodeGraph(context);
  std::unique_ptr<MappingTarget> target = targetWith(searchMachine(), kRules);
  ASSERT_NE(target, nullptr);

  MappingSearchOptions narrow;
  narrow.mode = SearchMode::Beam;
  narrow.beamWidth = 1;
  CoveringSearch beam(graph, *target, context, LayoutContext{}, narrow);
  llvm::Expected<MappingSearchResult> beamResult = beam.search();
  ASSERT_TRUE(static_cast<bool>(beamResult));
  EXPECT_TRUE(beamResult->searchTruncated);

  MappingSearchOptions exact;
  exact.mode = SearchMode::Exact;
  CoveringSearch exactSearch(graph, *target, context, LayoutContext{}, exact);
  llvm::Expected<MappingSearchResult> exactResult = exactSearch.search();
  ASSERT_TRUE(static_cast<bool>(exactResult));
  EXPECT_FALSE(exactResult->searchTruncated);
}

TEST(CoveringSearch, RepeatedRunsProduceIdenticalPlanIds) {
  mlir::MLIRContext context;
  WorkloadGraph graph = twoNodeGraph(context);
  std::unique_ptr<MappingTarget> target = targetWith(searchMachine(), kRules);
  ASSERT_NE(target, nullptr);

  MappingSearchOptions options;
  options.mode = SearchMode::Beam;
  CoveringSearch first(graph, *target, context, LayoutContext{}, options);
  CoveringSearch second(graph, *target, context, LayoutContext{}, options);
  llvm::Expected<MappingSearchResult> firstResult = first.search();
  llvm::Expected<MappingSearchResult> secondResult = second.search();
  ASSERT_TRUE(static_cast<bool>(firstResult));
  ASSERT_TRUE(static_cast<bool>(secondResult));
  EXPECT_EQ(planIds(*firstResult), planIds(*secondResult));
}

TEST(CoveringSearch, ReportsCapacityRejection) {
  mlir::MLIRContext context;
  WorkloadGraph graph = twoNodeGraph(context);
  std::unique_ptr<MappingTarget> target =
      targetWith(searchMachine(), kRulesWithMemory);
  ASSERT_NE(target, nullptr);

  MappingSearchOptions options;
  options.mode = SearchMode::Deterministic;
  options.memoryBudgetBytes = 1; // any bound memory overflows this
  CoveringSearch search(graph, *target, context, LayoutContext{}, options);
  llvm::Expected<MappingSearchResult> result = search.search();
  ASSERT_TRUE(static_cast<bool>(result)) << llvm::toString(result.takeError());
  EXPECT_TRUE(result->plans.empty());
  EXPECT_GT(result->frontier.plansRejectedByCapacity, 0u);
}

// §9.3: capacity is per memory, not one global pot. Each instance's 4096-byte
// tile fits both the 5000-byte sram node and the default byte budget; the two
// together exceed the node but not the budget, so only the per-memory check can
// reject the plan -- and it must.
TEST(CoveringSearch, PerMemoryCapacityRejectsOverSubscription) {
  mlir::MLIRContext context;
  WorkloadGraph graph = twoNodeGraph(context);
  std::unique_ptr<MappingTarget> target =
      targetWith(smallMemoryMachine(), kRulesWithMemory);
  ASSERT_NE(target, nullptr);
  // Guard the premise: one instance fits the node, two would not.
  const MemoryNode *sram = target->machine().findMemory("sram.0");
  ASSERT_NE(sram, nullptr);
  ASSERT_GE(sram->capacityBytes, 4096u);
  ASSERT_LT(sram->capacityBytes, 2u * 4096u);

  MappingSearchOptions options;
  options.mode = SearchMode::Deterministic;
  CoveringSearch search(graph, *target, context, LayoutContext{}, options);
  llvm::Expected<MappingSearchResult> result = search.search();
  ASSERT_TRUE(static_cast<bool>(result)) << llvm::toString(result.takeError());
  EXPECT_TRUE(result->plans.empty());
  EXPECT_GT(result->frontier.plansRejectedByCapacity, 0u);
}

// §9.3: the bytes a memory is charged are the value's own, not a constant. Two
// `!micro.tile<8x32xf32>` tiles are 1024 bytes each, so 2048 fits the
// 5000-byte sram node -- where the old 4096-byte-per-instance assumption
// (8192) would have rejected the pair outright. The derived size is what makes
// the plan legal.
TEST(CoveringSearch, CapacityUsesTheRealValueSizeNotAConstant) {
  mlir::MLIRContext context;
  context.getOrLoadDialect<mlir::micro::MicroDialect>();
  mlir::Type tile = mlir::parseType("!micro.tile<8x32xf32>", &context);
  ASSERT_TRUE(static_cast<bool>(tile));
  WorkloadGraph graph = twoNodeTileGraph(context, tile);
  std::unique_ptr<MappingTarget> target =
      targetWith(smallMemoryMachine(), kRulesWithMemory);
  ASSERT_NE(target, nullptr);

  // The premise: two real 1024-byte tiles fit the node, two 4096-byte
  // assumptions would not -- so the old constant could not have produced this
  // legal plan.
  const MemoryNode *sram = target->machine().findMemory("sram.0");
  ASSERT_NE(sram, nullptr);
  ASSERT_GE(sram->capacityBytes, 2u * 1024u);
  ASSERT_LT(sram->capacityBytes, 2u * 4096u);

  MappingSearchOptions options;
  options.mode = SearchMode::Deterministic;
  CoveringSearch search(graph, *target, context, LayoutContext{}, options);
  llvm::Expected<MappingSearchResult> result = search.search();
  ASSERT_TRUE(static_cast<bool>(result)) << llvm::toString(result.takeError());
  EXPECT_FALSE(result->plans.empty());
}

// Phase-3 T2: a value's bytes are released once its last consumer is placed.
// Three 1024-byte tiles chained n0 -> n1 -> n2 each fit sram.0 *sequentially*,
// but never all three at once: the live peak is two tiles (the one a consumer
// reads plus the one it writes). Monotonic accumulation charged all three and
// rejected the chain; live-range expiry makes it legal. The tiles are typed and
// the capacity is tied to their derived size, so the test cannot keep passing
// after a derivation change that would make three tiles fit.
TEST(CoveringSearch, ExpiresAValueWhenItsLastConsumerIsPlaced) {
  mlir::MLIRContext context;
  context.getOrLoadDialect<mlir::micro::MicroDialect>();
  mlir::Type tile = mlir::parseType("!micro.tile<8x32xf32>", &context);
  ASSERT_TRUE(static_cast<bool>(tile));
  WorkloadGraph graph = threeNodeChainGraph(context, tile);

  // Task 1's derivation, not the fallback: the test exercises the real-size
  // path the rest of the stack uses.
  const uint64_t tileBytes = tileFactsFor(tile).bytes;
  ASSERT_EQ(tileBytes, 1024u); // 8 * 32 * 4
  std::unique_ptr<MappingTarget> target =
      targetWith(sramCapacityMachine(2u * 1024u), kSharedReadRules);
  ASSERT_NE(target, nullptr);
  // Premise: two tiles fit, three do not. Only then is monotonic accumulation
  // guaranteed to reject while the live peak of two is admitted; if the derived
  // tile ever shrank, this fails loudly rather than letting the test pass
  // without detecting the regression.
  const MemoryNode *sram = target->machine().findMemory("sram.0");
  ASSERT_NE(sram, nullptr);
  ASSERT_GE(sram->capacityBytes, 2u * tileBytes);
  ASSERT_LT(sram->capacityBytes, 3u * tileBytes);

  MappingSearchOptions options;
  options.mode = SearchMode::Deterministic;
  CoveringSearch search(graph, *target, context, LayoutContext{}, options);
  llvm::Expected<MappingSearchResult> result = search.search();
  ASSERT_TRUE(static_cast<bool>(result)) << llvm::toString(result.takeError());
  EXPECT_FALSE(result->plans.empty());
  EXPECT_EQ(result->frontier.plansRejectedByCapacity, 0u);
}

// Expiry must not become unsound: a value stays charged until its *last*
// consumer is placed. A fan-out's value and both consumers' own output tiles
// are live together, so three 4096-byte tiles exceed a two-tile memory at the
// moment the second consumer is placed. Releasing on the first consumer would
// wrongly admit it -- both consumers genuinely overlap the shared value.
TEST(CoveringSearch, AnOverlappingValueStaysChargedUntilEveryConsumerIsPlaced) {
  mlir::MLIRContext context;
  WorkloadGraph graph = fanOutGraph(context, /*producerOp=*/"add",
                                    /*consumerOp=*/"add");
  std::unique_ptr<MappingTarget> target =
      targetWith(sramCapacityMachine(2u * 4096u), kSharedReadRules);
  ASSERT_NE(target, nullptr);

  MappingSearchOptions options;
  options.mode = SearchMode::Deterministic;
  CoveringSearch search(graph, *target, context, LayoutContext{}, options);
  llvm::Expected<MappingSearchResult> result = search.search();
  ASSERT_TRUE(static_cast<bool>(result)) << llvm::toString(result.takeError());
  EXPECT_TRUE(result->plans.empty());
  EXPECT_GT(result->frontier.plansRejectedByCapacity, 0u);
}

// Carried from T1: a node's single memory binding holds *every* output tile it
// writes, not just the first. A 1024-byte and a 2048-byte output are 3072
// together; sizing the binding from the first output alone (1024) under-charged
// and admitted a 2500-byte sram. 2500 must reject the pair, 4096 must admit it.
TEST(CoveringSearch, ANodeChargesItsMemoryForEveryMaterializedOutput) {
  mlir::MLIRContext context;
  context.getOrLoadDialect<mlir::micro::MicroDialect>();
  mlir::Type small = mlir::parseType("!micro.tile<8x32xf32>", &context);
  mlir::Type large = mlir::parseType("!micro.tile<16x32xf32>", &context);
  ASSERT_TRUE(static_cast<bool>(small));
  ASSERT_TRUE(static_cast<bool>(large));
  WorkloadGraph graph = twoOutputTileGraph(context, small, large);

  MappingSearchOptions options;
  options.mode = SearchMode::Deterministic;

  {
    std::unique_ptr<MappingTarget> target =
        targetWith(sramCapacityMachine(2500), kRulesWithMemory);
    ASSERT_NE(target, nullptr);
    CoveringSearch search(graph, *target, context, LayoutContext{}, options);
    llvm::Expected<MappingSearchResult> result = search.search();
    ASSERT_TRUE(static_cast<bool>(result))
        << llvm::toString(result.takeError());
    EXPECT_TRUE(result->plans.empty());
    EXPECT_GT(result->frontier.plansRejectedByCapacity, 0u);
  }
  {
    // The complement: 4096 admits the derived 3072, so the rejection above was
    // the summed output bytes and not some unrelated failure.
    std::unique_ptr<MappingTarget> target =
        targetWith(sramCapacityMachine(4096), kRulesWithMemory);
    ASSERT_NE(target, nullptr);
    CoveringSearch search(graph, *target, context, LayoutContext{}, options);
    llvm::Expected<MappingSearchResult> result = search.search();
    ASSERT_TRUE(static_cast<bool>(result))
        << llvm::toString(result.takeError());
    EXPECT_FALSE(result->plans.empty());
  }
}

// Phase-3 ruling R1: a value whose size cannot be derived is still costed, but
// the fallback is *reported*, naming the value -- an assumed size must never be
// silent again.
TEST(CoveringSearch, AnUnknownValueSizeIsReportedNotSilent) {
  mlir::MLIRContext context;
  WorkloadGraph graph = twoNodeGraph(context); // untyped values
  std::unique_ptr<MappingTarget> target = targetWith(searchMachine(), kRules);
  ASSERT_NE(target, nullptr);

  MappingSearchOptions options;
  options.mode = SearchMode::Deterministic;
  CoveringSearch search(graph, *target, context, LayoutContext{}, options);
  llvm::Expected<MappingSearchResult> result = search.search();
  ASSERT_TRUE(static_cast<bool>(result)) << llvm::toString(result.takeError());
  ASSERT_FALSE(result->plans.empty());
  EXPECT_TRUE(hasDiagnostic(*result, DiagnosticCode::AssumedValueSize));

  bool namedTheValue = false;
  for (const Diagnostic &diagnostic : result->frontier.diagnostics)
    if (diagnostic.code == DiagnosticCode::AssumedValueSize &&
        diagnostic.message.find("'mid'") != std::string::npos)
      namedTheValue = true;
  EXPECT_TRUE(namedTheValue);
}

// The complement: a value the graph types is sized exactly, so no assumption is
// reported.
TEST(CoveringSearch, AKnownValueSizeIsNotReported) {
  mlir::MLIRContext context;
  context.getOrLoadDialect<mlir::micro::MicroDialect>();
  mlir::Type tile = mlir::parseType("!micro.tile<8x32xf32>", &context);
  ASSERT_TRUE(static_cast<bool>(tile));
  WorkloadGraph graph = twoNodeTileGraph(context, tile);
  std::unique_ptr<MappingTarget> target =
      targetWith(smallMemoryMachine(), kRulesWithMemory);
  ASSERT_NE(target, nullptr);

  MappingSearchOptions options;
  options.mode = SearchMode::Deterministic;
  CoveringSearch search(graph, *target, context, LayoutContext{}, options);
  llvm::Expected<MappingSearchResult> result = search.search();
  ASSERT_TRUE(static_cast<bool>(result)) << llvm::toString(result.takeError());
  ASSERT_FALSE(result->plans.empty());
  EXPECT_FALSE(hasDiagnostic(*result, DiagnosticCode::AssumedValueSize));
}

// A zero-element value (a static 0 dimension -- reachable only through a
// modelled tensor, since the Micro tile verifier rejects 0) moves no bytes, so
// it takes no connection rather than tripping the "bytes must be positive"
// route rule. The plan still exists, charged zero for the tile.
TEST(CoveringSearch, AZeroElementValueTakesNoConnection) {
  mlir::MLIRContext context;
  mlir::Type empty =
      mlir::RankedTensorType::get({0, 32}, mlir::Float32Type::get(&context));
  WorkloadGraph graph = twoNodeTileGraph(context, empty);
  std::unique_ptr<MappingTarget> target =
      targetWith(searchMachine(), kRulesWithMemory);
  ASSERT_NE(target, nullptr);

  MappingSearchOptions options;
  options.mode = SearchMode::Deterministic;
  CoveringSearch search(graph, *target, context, LayoutContext{}, options);
  llvm::Expected<MappingSearchResult> result = search.search();
  ASSERT_TRUE(static_cast<bool>(result)) << llvm::toString(result.takeError());
  ASSERT_FALSE(result->plans.empty());
  EXPECT_TRUE(result->plans[0].connections.empty());
  EXPECT_FALSE(hasDiagnostic(*result, DiagnosticCode::AssumedValueSize));
}

// A connection across two *different* memories must run producer -> consumer,
// not the reverse. The #97 bug reversed the endpoints when the connection was
// synthesized as the consumer instance completed the edge, and every fixture
// that placed both ends in one memory hid it: a same-memory route has one node,
// so reversing it is invisible. Here the producer is placed on dram.0 (visible
// to e0) and the consumer on acc.0 (visible to e1), so the only legal route is
// the single dram.0 -> acc.0 link. The route's endpoints pin the direction.
TEST(CoveringSearch, CrossMemoryConnectionRunsFromProducerToConsumer) {
  mlir::MLIRContext context;
  WorkloadGraph graph = twoNodeGraph(context, /*producerOp=*/"produce",
                                     /*consumerOp=*/"consume");
  std::unique_ptr<MappingTarget> target = targetWith(fanMachine(), kFanRules);
  ASSERT_NE(target, nullptr);

  MappingSearchOptions options;
  options.mode = SearchMode::Deterministic;
  CoveringSearch search(graph, *target, context, LayoutContext{}, options);
  llvm::Expected<MappingSearchResult> result = search.search();
  ASSERT_TRUE(static_cast<bool>(result)) << llvm::toString(result.takeError());
  ASSERT_FALSE(result->plans.empty());
  const CoveringPlan &plan = result->plans[0];

  // The two endpoints are placed in genuinely different memories, named by each
  // placement's own rule so the direction claim does not rest on node ids.
  const PlanPlacement *producer = nullptr;
  const PlanPlacement *consumer = nullptr;
  for (const PlanPlacement &placement : plan.placements) {
    if (placement.rule == "r.produce")
      producer = &placement;
    else if (placement.rule == "r.consume")
      consumer = &placement;
  }
  ASSERT_NE(producer, nullptr);
  ASSERT_NE(consumer, nullptr);
  const MemoryNodeId producerMemory = producer->memories.lookup("dram");
  const MemoryNodeId consumerMemory = consumer->memories.lookup("acc");
  EXPECT_EQ(producerMemory, "dram.0");
  EXPECT_EQ(consumerMemory, "acc.0");
  ASSERT_NE(producerMemory, consumerMemory);

  // One connection whose route starts at the producer's memory and ends at the
  // consumer's -- producer first.
  ASSERT_EQ(plan.connections.size(), 1u);
  ASSERT_EQ(plan.connectionPlans.size(), 1u);
  const PlanConnection &connection = plan.connectionPlans[0];
  ASSERT_FALSE(connection.route.empty());
  EXPECT_EQ(connection.route.front(), producerMemory);
  EXPECT_EQ(connection.route.back(), consumerMemory);
}

// Phase-3 T4: the search hands each connection request the layouts its
// endpoints *bound*, so a pair placed under different layouts selects the
// transform alternative inside the search. Before this, `makeRequest` left both
// endpoints' layouts unset, so `synthesizeConnections` could only ever take its
// direct path from the search -- the transform path existed but was unreachable
// from it.
TEST(CoveringSearch, DifferingBoundLayoutsSelectTheTransformFromTheSearch) {
  mlir::MLIRContext context;
  WorkloadGraph graph = twoNodeGraph(context, /*producerOp=*/"produce",
                                     /*consumerOp=*/"consume");
  std::unique_ptr<MappingTarget> target =
      targetWithLayouts(transformMachine(), kTransformRules, kTwoLayouts);
  ASSERT_NE(target, nullptr);

  MappingSearchOptions options;
  options.mode = SearchMode::Deterministic;
  CoveringSearch search(graph, *target, context, LayoutContext{}, options);
  llvm::Expected<MappingSearchResult> result = search.search();
  ASSERT_TRUE(static_cast<bool>(result)) << llvm::toString(result.takeError());
  ASSERT_FALSE(result->plans.empty());
  const CoveringPlan &plan = result->plans[0];
  ASSERT_EQ(plan.connections.size(), 1u);
  ASSERT_EQ(plan.connectionPlans.size(), 1u);
  const PlanConnection &connection = plan.connectionPlans[0];
  // Both endpoints sit in sram.0, which supports both layouts, so the transform
  // runs in place: no transfer, one layout conversion, producer's layout first.
  EXPECT_EQ(connection.kind, ConnectionKind::LayoutTransform);
  EXPECT_EQ(connection.route, (llvm::SmallVector<MemoryNodeId>{"sram.0"}));
  ASSERT_TRUE(connection.transform.has_value());
  EXPECT_EQ(connection.transform->srcLayout, "t.plain");
  EXPECT_EQ(connection.transform->dstLayout, "t.blocked");

  // The placements carry the solved assignment each endpoint bound -- the
  // parameters the transform's two layouts were instantiated with.
  const PlanPlacement *producer = nullptr;
  const PlanPlacement *consumer = nullptr;
  for (const PlanPlacement &placement : plan.placements) {
    if (placement.rule == "r.produce")
      producer = &placement;
    else if (placement.rule == "r.consume")
      consumer = &placement;
  }
  ASSERT_NE(producer, nullptr);
  ASSERT_NE(consumer, nullptr);
  auto producerLayout = producer->layoutSolutions.find("t.plain");
  ASSERT_NE(producerLayout, producer->layoutSolutions.end());
  auto consumerLayout = consumer->layoutSolutions.find("t.blocked");
  ASSERT_NE(consumerLayout, consumer->layoutSolutions.end());
  ASSERT_EQ(consumerLayout->second.parameters.size(), 1u);
  const int64_t *vw = std::get_if<int64_t>(
      &consumerLayout->second.parameters.find("VW")->second);
  ASSERT_NE(vw, nullptr);
  EXPECT_EQ(*vw, 8);
}

// The control for the test above: the same graph and topology with both rules
// requiring the *same* layout stays a direct connection, so the transform above
// was the differing endpoint layouts and not the fixture.
TEST(CoveringSearch, MatchingBoundLayoutsStayDirect) {
  mlir::MLIRContext context;
  WorkloadGraph graph = twoNodeGraph(context, /*producerOp=*/"produce",
                                     /*consumerOp=*/"consume");
  std::unique_ptr<MappingTarget> target =
      targetWithLayouts(transformMachine(), kSameLayoutRules, kTwoLayouts);
  ASSERT_NE(target, nullptr);

  MappingSearchOptions options;
  options.mode = SearchMode::Deterministic;
  CoveringSearch search(graph, *target, context, LayoutContext{}, options);
  llvm::Expected<MappingSearchResult> result = search.search();
  ASSERT_TRUE(static_cast<bool>(result)) << llvm::toString(result.takeError());
  ASSERT_FALSE(result->plans.empty());
  const CoveringPlan &plan = result->plans[0];
  ASSERT_EQ(plan.connectionPlans.size(), 1u);
  EXPECT_EQ(plan.connectionPlans[0].kind, ConnectionKind::Direct);
  EXPECT_FALSE(plan.connectionPlans[0].transform.has_value());
}

// The other direction of the same attribution rule, and the reason it is by
// value rather than by "the instance's one layout": a layout named on a port
// that does *not* carry the crossing value must not be inherited by the edge.
// This is the shipped `avx2.mma_bf16` shape -- `require layout lhs satisfies
// avx2.row_major` names an input, while the edge carries the mma's `result`.
// Attributing `lhs`'s layout to the result would fabricate a transform (and,
// because differing layouts suppress `Direct`, could drop a valid connection).
TEST(CoveringSearch, ALayoutOnAnotherPortIsNotAttributedToTheEdge) {
  mlir::MLIRContext context;
  WorkloadGraph graph = twoNodeGraph(context, /*producerOp=*/"produce",
                                     /*consumerOp=*/"consume");
  std::unique_ptr<MappingTarget> target =
      targetWithLayouts(transformMachine(), kMmaShapedRules, kTwoLayouts);
  ASSERT_NE(target, nullptr);

  MappingSearchOptions options;
  options.mode = SearchMode::Deterministic;
  CoveringSearch search(graph, *target, context, LayoutContext{}, options);
  llvm::Expected<MappingSearchResult> result = search.search();
  ASSERT_TRUE(static_cast<bool>(result)) << llvm::toString(result.takeError());
  ASSERT_FALSE(result->plans.empty());
  const CoveringPlan &plan = result->plans[0];
  ASSERT_EQ(plan.connectionPlans.size(), 1u);
  // The producer bound `t.plain`, but for its *input*; the edge carries its
  // result, whose layout it never named. So the edge is not a transform: the
  // producer's layout is unknown, not `t.plain`.
  EXPECT_EQ(plan.connectionPlans[0].kind, ConnectionKind::Direct);
  EXPECT_FALSE(plan.connectionPlans[0].transform.has_value());

  // The premise: the producer really did bind `t.plain` (for its input), so
  // this is an attribution decision and not a missing solve.
  const PlanPlacement *producer = nullptr;
  for (const PlanPlacement &placement : plan.placements)
    if (placement.rule == "r.produce")
      producer = &placement;
  ASSERT_NE(producer, nullptr);
  auto solved = producer->layoutSolutions.find("t.plain");
  ASSERT_NE(solved, producer->layoutSolutions.end());
  const WorkloadNode *producerNode = graph.findNode(producer->node);
  ASSERT_NE(producerNode, nullptr);
  ASSERT_EQ(producerNode->inputs.size(), 1u);
  ASSERT_EQ(producerNode->outputs.size(), 1u);
  EXPECT_EQ(solved->second.portValue,
            static_cast<int64_t>(producerNode->inputs[0].value));
  EXPECT_NE(solved->second.portValue,
            static_cast<int64_t>(producerNode->outputs[0].value));
}

// P1 regression: one value feeding two operand ports of the same consumer is
// *two* uses, not one. Here the producer's result is plain, `operand0` requires
// plain and `operand1` requires blocked. Resolving a layout by SSA value made
// the two solved classes ambiguous ("nothing governs it"), so the pair became
// one unattributed direct connection -- silently dropping operand1's blocked
// obligation. Each use must get its own connection under its own layout.
TEST(CoveringSearch, DistinctOperandUsesOfOneValueKeepTheirOwnLayouts) {
  mlir::MLIRContext context;
  WorkloadGraph graph = repeatedOperandGraph(context);
  std::unique_ptr<MappingTarget> target = targetWithLayouts(
      transformMachine(), kRepeatedConflictRules, kTwoLayouts);
  ASSERT_NE(target, nullptr);

  MappingSearchOptions options;
  options.mode = SearchMode::Deterministic;
  CoveringSearch search(graph, *target, context, LayoutContext{}, options);
  llvm::Expected<MappingSearchResult> result = search.search();
  ASSERT_TRUE(static_cast<bool>(result)) << llvm::toString(result.takeError());
  ASSERT_FALSE(result->plans.empty());
  const CoveringPlan &plan = result->plans.front();

  // Two connections, one per operand use -- not one collapsed direct read.
  ASSERT_EQ(plan.connectionPlans.size(), 2u);

  // The plain use (operand0) connects directly; the blocked use (operand1)
  // needs a conversion into `t.blocked`. The transform's destination layout
  // identifies which connection serves `operand1`, so no plan-level port field
  // is needed to tell them apart.
  const PlanConnection *direct = nullptr;
  const PlanConnection *toBlocked = nullptr;
  for (const PlanConnection &connection : plan.connectionPlans) {
    if (connection.kind == ConnectionKind::Direct)
      direct = &connection;
    if (connection.transform && connection.transform->dstLayout == "t.blocked")
      toBlocked = &connection;
  }
  ASSERT_NE(direct, nullptr);
  ASSERT_NE(toBlocked, nullptr);
  EXPECT_EQ(toBlocked->kind, ConnectionKind::LayoutTransform);
  ASSERT_TRUE(toBlocked->transform.has_value());
  EXPECT_EQ(toBlocked->transform->srcLayout, "t.plain");
  EXPECT_EQ(toBlocked->transform->dstLayout, "t.blocked");
  EXPECT_TRUE(direct->transform == std::nullopt);
  // Both movements serve the same consumer instance; the compatibility
  // projection cannot tell the two uses apart, which is why the endpoint
  // occurrence is the rewiring authority. The exposed plan connections carry
  // it: `direct` serves `operand0`, the transform serves `operand1`.
  EXPECT_EQ(direct->consumers, toBlocked->consumers);
  ASSERT_TRUE(direct->producerPort.has_value());
  EXPECT_EQ(direct->producerPort->direction, PortDirection::Output);
  ASSERT_EQ(direct->consumerPorts.size(), 1u);
  EXPECT_EQ(direct->consumerPorts[0].direction, PortDirection::Input);
  EXPECT_EQ(direct->consumerPorts[0].index, 0u);
  ASSERT_TRUE(toBlocked->producerPort.has_value());
  ASSERT_EQ(toBlocked->consumerPorts.size(), 1u);
  EXPECT_EQ(toBlocked->consumerPorts[0].direction, PortDirection::Input);
  EXPECT_EQ(toBlocked->consumerPorts[0].index, 1u);
}

// The same-family, equal-parameterization control: both operand uses require
// `t.blocked` and so does the producer, so the two uses share one legal direct
// connection. Equal layouts must stay shareable -- the fix must not turn every
// repeated operand into a transform.
TEST(CoveringSearch, EqualLayoutsOnRepeatedOperandUsesShareOneConnection) {
  mlir::MLIRContext context;
  WorkloadGraph graph = repeatedOperandGraph(context);
  std::unique_ptr<MappingTarget> target = targetWithLayouts(
      transformMachine(), kRepeatedSameLayoutRules, kTwoLayouts);
  ASSERT_NE(target, nullptr);

  MappingSearchOptions options;
  options.mode = SearchMode::Deterministic;
  CoveringSearch search(graph, *target, context, LayoutContext{}, options);
  llvm::Expected<MappingSearchResult> result = search.search();
  ASSERT_TRUE(static_cast<bool>(result)) << llvm::toString(result.takeError());
  ASSERT_FALSE(result->plans.empty());
  const CoveringPlan &plan = result->plans.front();
  ASSERT_EQ(plan.connectionPlans.size(), 1u);
  EXPECT_EQ(plan.connectionPlans.front().kind, ConnectionKind::Direct);
  EXPECT_FALSE(plan.connectionPlans.front().transform.has_value());
  // One shared read serves both operand uses, and its endpoint list names them
  // both even though the instance projection repeats.
  ASSERT_EQ(plan.connectionPlans.front().consumerPorts.size(), 2u);
  EXPECT_EQ(plan.connectionPlans.front().consumerPorts[0].index, 0u);
  EXPECT_EQ(plan.connectionPlans.front().consumerPorts[1].index, 1u);
}

// Two *different* values feeding the two operands, both requiring one family
// `t.plain`: each edge resolves its own solved layout independently and stays
// direct. This is the legal control for the by-occurrence lookup; resolving by
// value must not conflate the two edges.
TEST(CoveringSearch, DistinctValuesNeedingOneFamilyStayIndependent) {
  mlir::MLIRContext context;
  WorkloadGraph graph = twoValueSameLayoutGraph(context);
  std::unique_ptr<MappingTarget> target =
      targetWithLayouts(transformMachine(), kTwoValuePlainRules, kTwoLayouts);
  ASSERT_NE(target, nullptr);

  MappingSearchOptions options;
  options.mode = SearchMode::Deterministic;
  CoveringSearch search(graph, *target, context, LayoutContext{}, options);
  llvm::Expected<MappingSearchResult> result = search.search();
  ASSERT_TRUE(static_cast<bool>(result)) << llvm::toString(result.takeError());
  ASSERT_FALSE(result->plans.empty());
  const CoveringPlan &plan = result->plans.front();
  ASSERT_EQ(plan.connectionPlans.size(), 2u);
  for (const PlanConnection &connection : plan.connectionPlans)
    EXPECT_EQ(connection.kind, ConnectionKind::Direct);
}

//===----------------------------------------------------------------------===//
// Fan-out and fan-in in the search (design §15.3)
//===----------------------------------------------------------------------===//

// Two consumers that can both read the producer's placement share a single
// read plan instead of getting one transfer each.
TEST(CoveringSearch, SharedReadSynthesizesOnePlanForBothConsumers) {
  mlir::MLIRContext context;
  WorkloadGraph graph = fanOutGraph(context, /*producerOp=*/"add",
                                    /*consumerOp=*/"add");
  std::unique_ptr<MappingTarget> target =
      targetWith(searchMachine(), kSharedReadRules);
  ASSERT_NE(target, nullptr);

  MappingSearchOptions options;
  options.mode = SearchMode::Deterministic;
  CoveringSearch search(graph, *target, context, LayoutContext{}, options);
  llvm::Expected<MappingSearchResult> result = search.search();
  ASSERT_TRUE(static_cast<bool>(result)) << llvm::toString(result.takeError());
  ASSERT_FALSE(result->plans.empty());
  const CoveringPlan &plan = result->plans[0];
  // One shared read for the value, not one connection per consumer.
  ASSERT_EQ(plan.connections.size(), 1u);
  ASSERT_EQ(plan.connectionPlans.size(), 1u);
  EXPECT_EQ(plan.connectionPlans[0].kind, ConnectionKind::Direct);
  // Three instances at one cycle each; the shared read is free.
  EXPECT_DOUBLE_EQ(plan.totalCost.latencyCycles, 3.0);
}

// Two consumers whose placements their executors cannot read in place must be
// served by a copy. They share a destination memory, so one `Replicate` copy
// serves both: the transfer cost is counted once, not once per consumer.
TEST(CoveringSearch, ReplicatedConsumersOnOneMemoryShareACopy) {
  mlir::MLIRContext context;
  WorkloadGraph graph = fanOutGraph(context, /*producerOp=*/"produce",
                                    /*consumerOp=*/"consume");
  std::unique_ptr<MappingTarget> target = targetWith(fanMachine(), kFanRules);
  ASSERT_NE(target, nullptr);

  MappingSearchOptions options;
  options.mode = SearchMode::Deterministic;
  CoveringSearch search(graph, *target, context, LayoutContext{}, options);
  llvm::Expected<MappingSearchResult> result = search.search();
  ASSERT_TRUE(static_cast<bool>(result)) << llvm::toString(result.takeError());
  ASSERT_FALSE(result->plans.empty());
  const CoveringPlan &plan = result->plans[0];
  ASSERT_EQ(plan.connections.size(), 1u);
  ASSERT_EQ(plan.connectionPlans.size(), 1u);
  EXPECT_EQ(plan.connectionPlans[0].kind, ConnectionKind::Replicate);
  // Three instances at one cycle, plus one shared copy.
  EXPECT_DOUBLE_EQ(plan.totalCost.latencyCycles, 3.0 + kFanTransferCycles);
}

// Two producers and two consumers of one value: consumers sharing a
// destination memory are served by a single gather, so the feeds and the
// intermediate tile are counted once per memory, not once per consumer.
TEST(CoveringSearch, MultiProducerMultiConsumerDedupesToOneReduce) {
  mlir::MLIRContext context;
  WorkloadGraph graph = fanInFanOutGraph(context);
  std::unique_ptr<MappingTarget> target = targetWith(fanMachine(), kFanRules);
  ASSERT_NE(target, nullptr);

  MappingSearchOptions options;
  options.mode = SearchMode::Deterministic;
  CoveringSearch search(graph, *target, context, LayoutContext{}, options);
  llvm::Expected<MappingSearchResult> result = search.search();
  ASSERT_TRUE(static_cast<bool>(result)) << llvm::toString(result.takeError());
  ASSERT_FALSE(result->plans.empty());
  const CoveringPlan &plan = result->plans[0];
  ASSERT_EQ(plan.connections.size(), 1u);
  ASSERT_EQ(plan.connectionPlans.size(), 1u);
  EXPECT_EQ(plan.connectionPlans[0].kind, ConnectionKind::Reduce);
  // Four instances at one cycle, plus the two summed feeds.
  EXPECT_DOUBLE_EQ(plan.totalCost.latencyCycles,
                   4.0 + 2.0 * kFanTransferCycles);
}

// A value fed by two producers is gathered into one `Reduce` connection whose
// cost is the sum of the two feeds.
TEST(CoveringSearch, GatherSumsTheProducerFeeds) {
  mlir::MLIRContext context;
  WorkloadGraph graph = fanInGraph(context);
  std::unique_ptr<MappingTarget> target = targetWith(fanMachine(), kFanRules);
  ASSERT_NE(target, nullptr);

  MappingSearchOptions options;
  options.mode = SearchMode::Deterministic;
  CoveringSearch search(graph, *target, context, LayoutContext{}, options);
  llvm::Expected<MappingSearchResult> result = search.search();
  ASSERT_TRUE(static_cast<bool>(result)) << llvm::toString(result.takeError());
  ASSERT_FALSE(result->plans.empty());
  const CoveringPlan &plan = result->plans[0];
  ASSERT_EQ(plan.connections.size(), 1u);
  ASSERT_EQ(plan.connectionPlans.size(), 1u);
  EXPECT_EQ(plan.connectionPlans[0].kind, ConnectionKind::Reduce);
  // Three instances at one cycle, plus the two summed transfer feeds.
  EXPECT_DOUBLE_EQ(plan.totalCost.latencyCycles,
                   3.0 + 2.0 * kFanTransferCycles);
}

// Exact mode does not branch over a gather feed's legal alternatives; it keeps
// only the locally cheapest per producer. That is a choice collapse, not a cap,
// so it must disclose itself -- `connectionChoicesUnexplored` set and the
// stable notice filed -- while every cap remains lifted (`searchTruncated`
// clear). Before the fix the gather path reported neither.
TEST(CoveringSearch, GatherExactChoicesAreDisclosed) {
  mlir::MLIRContext context;
  WorkloadGraph graph = fanInGraph(context);
  MachineModel machine = fanMachine();
  // A second parallel dram.0 -> acc.0 route. The original (latency 10) stays
  // cheaper than the copy (latency 20), so each feed has two legal alternatives
  // and exact mode keeps one without branching over them.
  LinkEdge second = machine.links.front();
  second.id = "second_route";
  second.latencyCycles = 20;
  machine.links.push_back(second);

  std::unique_ptr<MappingTarget> target =
      targetWith(std::move(machine), kFanRules);
  ASSERT_NE(target, nullptr);

  MappingSearchOptions options;
  options.mode = SearchMode::Exact;
  CoveringSearch search(graph, *target, context, LayoutContext{}, options);
  llvm::Expected<MappingSearchResult> result = search.search();
  ASSERT_TRUE(static_cast<bool>(result)) << llvm::toString(result.takeError());
  ASSERT_FALSE(result->plans.empty());
  EXPECT_TRUE(result->connectionChoicesUnexplored);
  EXPECT_FALSE(result->searchTruncated);
  EXPECT_TRUE(
      hasDiagnostic(*result, DiagnosticCode::ConnectionChoiceUnexplored));
  EXPECT_FALSE(hasDiagnostic(*result, DiagnosticCode::SearchTruncated));
  // The disclosure is a single stable notice, however many feeds collapsed it:
  // `report` deduplicates on (code, message), so the distinct-notice list does
  // not grow per producer or per branch.
  size_t notices = 0;
  for (const Diagnostic &diagnostic : result->frontier.diagnostics)
    if (diagnostic.code == DiagnosticCode::ConnectionChoiceUnexplored)
      ++notices;
  EXPECT_EQ(notices, 1u);
}

// Control: a gather whose every feed has exactly one legal route collapses no
// choice, so it must not claim an unexplored one. `fanMachine` offers a single
// dram.0 -> acc.0 link.
TEST(CoveringSearch, SingleRouteGatherReportsNoUnexploredChoices) {
  mlir::MLIRContext context;
  WorkloadGraph graph = fanInGraph(context);
  std::unique_ptr<MappingTarget> target = targetWith(fanMachine(), kFanRules);
  ASSERT_NE(target, nullptr);

  MappingSearchOptions options;
  options.mode = SearchMode::Exact;
  CoveringSearch search(graph, *target, context, LayoutContext{}, options);
  llvm::Expected<MappingSearchResult> result = search.search();
  ASSERT_TRUE(static_cast<bool>(result)) << llvm::toString(result.takeError());
  ASSERT_FALSE(result->plans.empty());
  EXPECT_FALSE(result->connectionChoicesUnexplored);
  EXPECT_FALSE(result->searchTruncated);
}

// Control: a route cap is a cap, not a choice collapse. Capping the feed's
// routes to one reports truncation through the ordinary path, so the two
// disclosures stay independent.
TEST(CoveringSearch, GatherRouteCapReportsTruncation) {
  mlir::MLIRContext context;
  WorkloadGraph graph = fanInGraph(context);
  MachineModel machine = fanMachine();
  LinkEdge second = machine.links.front();
  second.id = "second_route";
  second.latencyCycles = 20;
  machine.links.push_back(second);

  std::unique_ptr<MappingTarget> target =
      targetWith(std::move(machine), kFanRules);
  ASSERT_NE(target, nullptr);

  MappingSearchOptions options;
  options.mode = SearchMode::Exact;
  options.maxRoutesPerConnection = 1; // two parallel routes exceed the cap
  CoveringSearch search(graph, *target, context, LayoutContext{}, options);
  llvm::Expected<MappingSearchResult> result = search.search();
  ASSERT_TRUE(static_cast<bool>(result)) << llvm::toString(result.takeError());
  ASSERT_FALSE(result->plans.empty());
  EXPECT_TRUE(result->searchTruncated);
  EXPECT_TRUE(hasDiagnostic(*result, DiagnosticCode::SearchTruncated));
}

// A route cap reached inside fan-out replication reaches `searchTruncated`
// through the fan-out out-parameter.
TEST(CoveringSearch, FanOutRouteCapSetsSearchTruncated) {
  mlir::MLIRContext context;
  WorkloadGraph graph = fanOutGraph(context, /*producerOp=*/"produce",
                                    /*consumerOp=*/"consume");
  std::unique_ptr<MappingTarget> target = targetWith(fanMachine(), kFanRules);
  ASSERT_NE(target, nullptr);

  MappingSearchOptions options;
  options.mode = SearchMode::Deterministic;
  options.maxRoutesPerConnection = 1; // the single route saturates the cap
  CoveringSearch search(graph, *target, context, LayoutContext{}, options);
  llvm::Expected<MappingSearchResult> result = search.search();
  ASSERT_TRUE(static_cast<bool>(result)) << llvm::toString(result.takeError());
  EXPECT_TRUE(result->searchTruncated);
  ASSERT_FALSE(result->plans.empty());
  bool replicated = false;
  for (const PlanConnection &connection : result->plans[0].connectionPlans)
    replicated |= connection.kind == ConnectionKind::Replicate;
  EXPECT_TRUE(replicated);
}

// Replicated bytes are charged to the memory that holds the copy: two consumer
// instances fit acc.0, but the one copy they share does not.
TEST(CoveringSearch, ReplicationBytesCountAgainstCapacity) {
  mlir::MLIRContext context;
  WorkloadGraph graph = fanOutGraph(context, /*producerOp=*/"produce",
                                    /*consumerOp=*/"consume");
  MachineModel machine = fanMachine();
  // The two consumer instances charge 4096 each (8192); the one shared copy
  // pushes acc.0 to 12288. 12000 admits the instances but not the copy.
  for (MemoryNode &memory : machine.memories)
    if (memory.kind == "acc")
      memory.capacityBytes = 12000;
  std::unique_ptr<MappingTarget> target = targetWith(machine, kFanRules);
  ASSERT_NE(target, nullptr);

  MappingSearchOptions options;
  options.mode = SearchMode::Deterministic;
  CoveringSearch search(graph, *target, context, LayoutContext{}, options);
  llvm::Expected<MappingSearchResult> result = search.search();
  ASSERT_TRUE(static_cast<bool>(result)) << llvm::toString(result.takeError());
  EXPECT_TRUE(result->plans.empty());
  EXPECT_GT(result->frontier.plansRejectedByCapacity, 0u);
}

// A replica staged through an intermediate memory is charged there too, not
// only at its destination: two copies into acc.0 and aux.0 both stage through
// the 5000-byte stage.0, which holds one 4096-byte tile but not two.
TEST(CoveringSearch, ReplicateIntermediateHopsCountAgainstCapacity) {
  mlir::MLIRContext context;
  WorkloadGraph graph = twoConsumerGraph(context);
  std::unique_ptr<MappingTarget> target =
      targetWith(twoDestinationHopMachine(), kTwoConsumerRules);
  ASSERT_NE(target, nullptr);

  MappingSearchOptions options;
  options.mode = SearchMode::Deterministic;
  CoveringSearch search(graph, *target, context, LayoutContext{}, options);
  llvm::Expected<MappingSearchResult> result = search.search();
  ASSERT_TRUE(static_cast<bool>(result)) << llvm::toString(result.takeError());
  EXPECT_TRUE(result->plans.empty());
  EXPECT_GT(result->frontier.plansRejectedByCapacity, 0u);
}

// Phase-3 T3: a connection whose only route stages through a memory that
// already holds another instance's live tile is illegal. `a_hold` writes its
// 1024-byte tile to the staging sram and nothing ever releases it, so when
// `z_consume` completes the `m_produce -> z_consume` crossing the router must
// charge that live tile against the staging memory: 1500 - 1024 < 1024, so the
// staging hop is rejected. The occupancy is the partial plan's real live bytes,
// derived from the tile type, not a constant. With the occupancy left unset
// (the pre-fix behaviour) the route fits and a plan is found.
TEST(CoveringSearch, OccupiedStagingMemoryRejectsTheConnection) {
  mlir::MLIRContext context;
  context.getOrLoadDialect<mlir::micro::MicroDialect>();
  mlir::Type tile = mlir::parseType("!micro.tile<8x32xf32>", &context);
  ASSERT_TRUE(static_cast<bool>(tile));
  const uint64_t tileBytes = tileFactsFor(tile).bytes;
  ASSERT_EQ(tileBytes, 1024u); // 8 * 32 * 4, so no magic constant

  WorkloadGraph graph = occupiedStagingGraph(context, tile);
  MachineModel machine = twoDestinationHopMachine();
  for (MemoryNode &memory : machine.memories)
    if (memory.kind == "sram")
      memory.capacityBytes = 1500;
  // Premise: one staging tile fits, the live tile plus the staged value do not.
  const MemoryNode *stage = machine.findMemory("stage.0");
  ASSERT_NE(stage, nullptr);
  ASSERT_GE(stage->capacityBytes, tileBytes);
  ASSERT_LT(stage->capacityBytes, 2u * tileBytes);

  std::unique_ptr<MappingTarget> target =
      targetWith(std::move(machine), kOccupiedStagingRules);
  ASSERT_NE(target, nullptr);

  MappingSearchOptions options;
  options.mode = SearchMode::Deterministic;
  CoveringSearch search(graph, *target, context, LayoutContext{}, options);
  llvm::Expected<MappingSearchResult> result = search.search();
  ASSERT_TRUE(static_cast<bool>(result)) << llvm::toString(result.takeError());
  EXPECT_TRUE(result->plans.empty());
}

// Control for the test above: the same graph and topology with a staging memory
// large enough to hold the live tile *and* the staged value yields a legal
// plan. The rejection above is therefore the occupancy, not the route's shape.
TEST(CoveringSearch, StagingMemoryWithRoomAdmitsTheConnection) {
  mlir::MLIRContext context;
  context.getOrLoadDialect<mlir::micro::MicroDialect>();
  mlir::Type tile = mlir::parseType("!micro.tile<8x32xf32>", &context);
  ASSERT_TRUE(static_cast<bool>(tile));
  const uint64_t tileBytes = tileFactsFor(tile).bytes;

  WorkloadGraph graph = occupiedStagingGraph(context, tile);
  MachineModel machine = twoDestinationHopMachine();
  for (MemoryNode &memory : machine.memories)
    if (memory.kind == "sram")
      memory.capacityBytes = 4u * tileBytes;
  const MemoryNode *stage = machine.findMemory("stage.0");
  ASSERT_NE(stage, nullptr);
  ASSERT_GE(stage->capacityBytes, 2u * tileBytes);

  std::unique_ptr<MappingTarget> target =
      targetWith(std::move(machine), kOccupiedStagingRules);
  ASSERT_NE(target, nullptr);

  MappingSearchOptions options;
  options.mode = SearchMode::Deterministic;
  CoveringSearch search(graph, *target, context, LayoutContext{}, options);
  llvm::Expected<MappingSearchResult> result = search.search();
  ASSERT_TRUE(static_cast<bool>(result)) << llvm::toString(result.takeError());
  EXPECT_FALSE(result->plans.empty());
}

// Issue #109 defect 3: exact mode implies an exhaustive joint search, but each
// connection's alternative is chosen locally -- only the cheapest is taken.
// When more than one alternative exists the search now *says so*, rather than
// letting exact mode's name imply a completeness it does not have. Two routes
// (a direct hop and a staged one) make the choice observable.
TEST(CoveringSearch, ExactModeReportsUnexploredConnectionChoices) {
  mlir::MLIRContext context;
  context.getOrLoadDialect<mlir::micro::MicroDialect>();
  mlir::Type tile = mlir::parseType("!micro.tile<8x32xf32>", &context);
  ASSERT_TRUE(static_cast<bool>(tile));
  WorkloadGraph graph = occupiedStagingGraph(context, tile);

  MachineModel machine = twoDestinationHopMachine();
  for (MemoryNode &memory : machine.memories)
    memory.capacityBytes = 1u << 20; // every alternative fits
  // A second route from dram.0 to acc.0: the direct hop, alongside the staged
  // one through stage.0 the topology already offers.
  LinkEdge direct;
  direct.id = "dram_to_acc.0";
  direct.source = "dram.0";
  direct.destination = "acc.0";
  direct.bandwidthBytesPerCycle = 32;
  direct.latencyCycles = 10;
  direct.transactionBytes = 64;
  direct.transferEngines = {"dma.0"};
  machine.links.push_back(direct);

  std::unique_ptr<MappingTarget> target =
      targetWith(std::move(machine), kOccupiedStagingRules);
  ASSERT_NE(target, nullptr);

  MappingSearchOptions options;
  options.mode = SearchMode::Exact;
  CoveringSearch search(graph, *target, context, LayoutContext{}, options);
  llvm::Expected<MappingSearchResult> result = search.search();
  ASSERT_TRUE(static_cast<bool>(result)) << llvm::toString(result.takeError());
  ASSERT_FALSE(result->plans.empty());
  EXPECT_TRUE(result->connectionChoicesUnexplored);
  EXPECT_TRUE(
      hasDiagnostic(*result, DiagnosticCode::ConnectionChoiceUnexplored));
}

// Control: the beam and deterministic modes are heuristic by contract, so they
// make no exhaustiveness claim and leave the flag clear.
TEST(CoveringSearch, DeterministicModeDoesNotReportUnexploredChoices) {
  mlir::MLIRContext context;
  context.getOrLoadDialect<mlir::micro::MicroDialect>();
  mlir::Type tile = mlir::parseType("!micro.tile<8x32xf32>", &context);
  ASSERT_TRUE(static_cast<bool>(tile));
  WorkloadGraph graph = occupiedStagingGraph(context, tile);

  MachineModel machine = twoDestinationHopMachine();
  for (MemoryNode &memory : machine.memories)
    memory.capacityBytes = 1u << 20;
  LinkEdge direct;
  direct.id = "dram_to_acc.0";
  direct.source = "dram.0";
  direct.destination = "acc.0";
  direct.bandwidthBytesPerCycle = 32;
  direct.latencyCycles = 10;
  direct.transactionBytes = 64;
  direct.transferEngines = {"dma.0"};
  machine.links.push_back(direct);

  std::unique_ptr<MappingTarget> target =
      targetWith(std::move(machine), kOccupiedStagingRules);
  ASSERT_NE(target, nullptr);

  MappingSearchOptions options;
  options.mode = SearchMode::Deterministic;
  CoveringSearch search(graph, *target, context, LayoutContext{}, options);
  llvm::Expected<MappingSearchResult> result = search.search();
  ASSERT_TRUE(static_cast<bool>(result)) << llvm::toString(result.takeError());
  EXPECT_FALSE(result->connectionChoicesUnexplored);
}

// A gather's intermediate tile is charged to the consumer's memory too: the
// consumer instance fits acc.0, but the gathered tile it produces does not.
TEST(CoveringSearch, GatherIntermediateCountsAgainstCapacity) {
  mlir::MLIRContext context;
  WorkloadGraph graph = fanInGraph(context);
  MachineModel machine = fanMachine();
  // The consumer instance charges 4096 to acc.0; the gather's intermediate tile
  // adds another 4096. 5000 admits the instance but not the tile.
  for (MemoryNode &memory : machine.memories)
    if (memory.kind == "acc")
      memory.capacityBytes = 5000;
  std::unique_ptr<MappingTarget> target = targetWith(machine, kFanRules);
  ASSERT_NE(target, nullptr);

  MappingSearchOptions options;
  options.mode = SearchMode::Deterministic;
  CoveringSearch search(graph, *target, context, LayoutContext{}, options);
  llvm::Expected<MappingSearchResult> result = search.search();
  ASSERT_TRUE(static_cast<bool>(result)) << llvm::toString(result.takeError());
  EXPECT_TRUE(result->plans.empty());
  EXPECT_GT(result->frontier.plansRejectedByCapacity, 0u);
}

/// Producer -> consumer across `fanMachine`'s dram.0 -> acc.0 where the value
/// the consumer reads is `large` and the tile it writes is `small`. The two
/// sizes differ so the transfer's destination input buffer is distinguishable
/// from the consumer's own output tile.
WorkloadGraph transferCapacityGraph(mlir::MLIRContext &context,
                                    mlir::Type large, mlir::Type small) {
  WorkloadGraph graph;
  WorkloadValueId input =
      graph.addValue(WorkloadValue{0, large, "in", /*external=*/true});
  WorkloadValueId middle =
      graph.addValue(WorkloadValue{0, large, "mid", /*external=*/false});
  WorkloadValueId output =
      graph.addValue(WorkloadValue{0, small, "out", /*external=*/false});

  WorkloadNode producer;
  producer.opName = "micro.vector";
  producer.attributes = vectorAttributes(context, "produce");
  producer.inputs.push_back(WorkloadPort{input, large, std::nullopt});
  producer.outputs.push_back(WorkloadPort{middle, large, std::nullopt});
  graph.addNode(std::move(producer));

  WorkloadNode consumer;
  consumer.opName = "micro.vector";
  consumer.sourceOrdinal = 1;
  consumer.attributes = vectorAttributes(context, "consume");
  consumer.inputs.push_back(WorkloadPort{middle, large, std::nullopt});
  consumer.outputs.push_back(WorkloadPort{output, small, std::nullopt});
  graph.addNode(std::move(consumer));

  graph.finalize();
  return graph;
}

// §9.3: a transfer's destination memory holds the consumer's *input* buffer,
// which is not the consumer's output tile. A 4096-byte value moved into a
// 1024-byte acc.0 must be charged there even though the consumer writes only 4
// bytes -- the old accounting only counted the consumer's outputs, so a
// narrowing (or reducing) consumer let an oversized input through.
TEST(CoveringSearch, ATransferChargesItsDestinationInputBuffer) {
  mlir::MLIRContext context;
  mlir::Type large =
      mlir::RankedTensorType::get({1024}, mlir::Float32Type::get(&context));
  mlir::Type small =
      mlir::RankedTensorType::get({1}, mlir::Float32Type::get(&context));
  WorkloadGraph graph = transferCapacityGraph(context, large, small);

  MachineModel machine = fanMachine();
  // The 4-byte output fits; the 4096-byte input buffer it reads does not.
  machine.memories[1].capacityBytes = 1024;
  std::unique_ptr<MappingTarget> target = targetWith(machine, kFanRules);
  ASSERT_NE(target, nullptr);

  MappingSearchOptions options;
  options.mode = SearchMode::Exact;
  CoveringSearch search(graph, *target, context, LayoutContext{}, options);
  llvm::Expected<MappingSearchResult> result = search.search();
  ASSERT_TRUE(static_cast<bool>(result)) << llvm::toString(result.takeError());
  EXPECT_TRUE(result->plans.empty());
  EXPECT_TRUE(hasDiagnostic(*result, DiagnosticCode::MemoryCapacityExceeded));
  EXPECT_GT(result->frontier.plansRejectedByCapacity, 0u);
}

// Control: sizing the destination for the input buffer *and* the consumer's
// output admits the same plan, so the rejection above is the charged input
// buffer and not some unrelated failure.
TEST(CoveringSearch, ATransferWithinItsDestinationCapacityIsAdmitted) {
  mlir::MLIRContext context;
  mlir::Type large =
      mlir::RankedTensorType::get({1024}, mlir::Float32Type::get(&context));
  mlir::Type small =
      mlir::RankedTensorType::get({1}, mlir::Float32Type::get(&context));
  WorkloadGraph graph = transferCapacityGraph(context, large, small);

  MachineModel machine = fanMachine();
  machine.memories[1].capacityBytes = 4096u + 4u; // input buffer + output tile
  std::unique_ptr<MappingTarget> target = targetWith(machine, kFanRules);
  ASSERT_NE(target, nullptr);

  MappingSearchOptions options;
  options.mode = SearchMode::Exact;
  CoveringSearch search(graph, *target, context, LayoutContext{}, options);
  llvm::Expected<MappingSearchResult> result = search.search();
  ASSERT_TRUE(static_cast<bool>(result)) << llvm::toString(result.takeError());
  EXPECT_FALSE(result->plans.empty());
}

/// One node writing two outputs and reading nothing: the simplest shape whose
/// capacity accounting depends on how its outputs are attributed to memory.
WorkloadGraph multiOutputGraph(mlir::MLIRContext &context, mlir::Type first,
                               mlir::Type second) {
  WorkloadGraph graph;
  WorkloadValueId a =
      graph.addValue(WorkloadValue{0, first, "a", /*external=*/false});
  WorkloadValueId b =
      graph.addValue(WorkloadValue{0, second, "b", /*external=*/false});
  WorkloadNode node;
  node.opName = "micro.vector";
  node.attributes = vectorAttributes(context);
  node.outputs.push_back(WorkloadPort{a, first, std::nullopt});
  node.outputs.push_back(WorkloadPort{b, second, std::nullopt});
  graph.addNode(std::move(node));
  graph.finalize();
  return graph;
}

/// A rule binding two memory kinds, so a placed node carries two memory
/// bindings and the graph fixes no output-to-binding association.
constexpr llvm::StringLiteral kTwoMemoryRules = R"llkmap(
rule r.two_memory {
  match micro.vector(op = "add");
  require executor kind worker;
  require memory kind sram;
  require memory kind dram;
  bundle "b";
  emit "e1";
  cost 1;
}
)llkmap";

/// The same node with a single memory binding: no association is needed, so
/// every output is charged to that one binding.
constexpr llvm::StringLiteral kOneMemoryRules = R"llkmap(
rule r.one_memory {
  match micro.vector(op = "add");
  require executor kind worker;
  require memory kind sram;
  bundle "b";
  emit "e1";
  cost 1;
}
)llkmap";

// Issue #109 defect 1: a node with several outputs and several memory bindings
// has no output-to-binding association, so its capacity cannot be established.
// Charging only the *first* output to every binding is a lower bound, and a
// 4-byte first output hid a 4096-byte second: the old search admitted the plan
// into two 1024-byte memories. Ambiguous placement is now rejected rather than
// admitted on that unsafe bound -- and here with capacity to spare, because the
// ambiguity is structural, not a matter of size.
TEST(CoveringSearch, AmbiguousMultiOutputMultiMemoryPlacementIsRejected) {
  mlir::MLIRContext context;
  mlir::Type small =
      mlir::RankedTensorType::get({1}, mlir::Float32Type::get(&context));
  mlir::Type large =
      mlir::RankedTensorType::get({1024}, mlir::Float32Type::get(&context));
  WorkloadGraph graph = multiOutputGraph(context, small, large);

  MachineModel machine = searchMachine();
  for (MemoryNode &memory : machine.memories)
    memory.capacityBytes = 1u << 30; // plenty: rejection is not about size
  std::unique_ptr<MappingTarget> target = targetWith(machine, kTwoMemoryRules);
  ASSERT_NE(target, nullptr);

  MappingSearchOptions options;
  options.mode = SearchMode::Exact;
  CoveringSearch search(graph, *target, context, LayoutContext{}, options);
  llvm::Expected<MappingSearchResult> result = search.search();
  ASSERT_TRUE(static_cast<bool>(result)) << llvm::toString(result.takeError());
  EXPECT_TRUE(result->plans.empty());
  EXPECT_TRUE(hasDiagnostic(*result, DiagnosticCode::MemoryCapacityExceeded));
}

// Control: one memory binding leaves nothing ambiguous -- the binding holds
// every output -- so a two-output node whose outputs both fit is admitted.
TEST(CoveringSearch, MultiOutputSingleMemoryIsAdmittedWhenTheTotalFits) {
  mlir::MLIRContext context;
  mlir::Type small =
      mlir::RankedTensorType::get({1}, mlir::Float32Type::get(&context));
  WorkloadGraph graph = multiOutputGraph(context, small, small);

  MachineModel machine = searchMachine();
  for (MemoryNode &memory : machine.memories)
    memory.capacityBytes = 1024;
  std::unique_ptr<MappingTarget> target = targetWith(machine, kOneMemoryRules);
  ASSERT_NE(target, nullptr);

  MappingSearchOptions options;
  options.mode = SearchMode::Exact;
  CoveringSearch search(graph, *target, context, LayoutContext{}, options);
  llvm::Expected<MappingSearchResult> result = search.search();
  ASSERT_TRUE(static_cast<bool>(result)) << llvm::toString(result.takeError());
  EXPECT_FALSE(result->plans.empty());
}

// Control: the same single-binding node is rejected once its *later* output is
// large -- proving every output is charged, not only the first.
TEST(CoveringSearch, MultiOutputSingleMemoryRejectsAnOversizedLaterOutput) {
  mlir::MLIRContext context;
  mlir::Type small =
      mlir::RankedTensorType::get({1}, mlir::Float32Type::get(&context));
  mlir::Type large =
      mlir::RankedTensorType::get({1024}, mlir::Float32Type::get(&context));
  WorkloadGraph graph = multiOutputGraph(context, small, large);

  MachineModel machine = searchMachine();
  for (MemoryNode &memory : machine.memories)
    memory.capacityBytes = 1024;
  std::unique_ptr<MappingTarget> target = targetWith(machine, kOneMemoryRules);
  ASSERT_NE(target, nullptr);

  MappingSearchOptions options;
  options.mode = SearchMode::Exact;
  CoveringSearch search(graph, *target, context, LayoutContext{}, options);
  llvm::Expected<MappingSearchResult> result = search.search();
  ASSERT_TRUE(static_cast<bool>(result)) << llvm::toString(result.takeError());
  EXPECT_TRUE(result->plans.empty());
  EXPECT_TRUE(hasDiagnostic(*result, DiagnosticCode::MemoryCapacityExceeded));
}

/// `searchMachine` with explicit sram and dram capacities, so a per-port
/// capacity test can pin exactly what each bound output affords.
MachineModel sizedMemoryMachine(uint64_t sramBytes, uint64_t dramBytes) {
  MachineModel model = searchMachine();
  for (MemoryNode &memory : model.memories)
    memory.capacityBytes = memory.kind == "sram" ? sramBytes : dramBytes;
  return model;
}

/// `searchMachine` with two visible sram nodes of different capacities, so two
/// same-kind roles can be bound to different nodes.
MachineModel twoSramMachine(uint64_t smallBytes, uint64_t largeBytes) {
  MachineModel model;
  model.target = "samekind";
  model.executors = {{"e0", "worker", std::nullopt, {}, 1, {}}};
  MemoryNode small;
  small.id = "sram.small";
  small.kind = "sram";
  small.visibleFrom = "e0";
  small.capacityBytes = smallBytes;
  MemoryNode large;
  large.id = "sram.large";
  large.kind = "sram";
  large.visibleFrom = "e0";
  large.capacityBytes = largeBytes;
  model.memories = {small, large};
  return model;
}

/// A rule that binds each of its two output ports to its own memory kind by
/// name. The output-to-memory association is explicit, so the search charges
/// each output to the memory its port selected rather than replicating the
/// first output across every binding.
constexpr llvm::StringLiteral kPortedMemoryRules = R"llkmap(
rule r.ported {
  match micro.vector(op = "add");
  require executor kind worker;
  require memory output "small" kind sram;
  require memory output "large" kind dram;
  output "small";
  output "large";
  bundle "b";
  emit "e1";
  cost 1;
}
)llkmap";

/// The same two ports with their memory kinds swapped: the small output is
/// bound to dram and the large one to sram.
constexpr llvm::StringLiteral kSwappedMemoryRules = R"llkmap(
rule r.swapped {
  match micro.vector(op = "add");
  require executor kind worker;
  require memory output "small" kind dram;
  require memory output "large" kind sram;
  output "small";
  output "large";
  bundle "b";
  emit "e1";
  cost 1;
}
)llkmap";

/// Both output ports bound to the same memory kind, so the roles are
/// distinguished by port, not by kind.
constexpr llvm::StringLiteral kSameKindPortRules = R"llkmap(
rule r.same_kind {
  match micro.vector(op = "add");
  require executor kind worker;
  require memory output "small" kind sram;
  require memory output "large" kind sram;
  output "small";
  output "large";
  bundle "b";
  emit "e1";
  cost 1;
}
)llkmap";

/// Finds the memory bound to an output port occurrence in a placement, or "".
std::string placementMemoryForPort(const PlanPlacement &placement,
                                   uint32_t index) {
  for (const PortMemoryBinding &binding : placement.portMemoryBindings)
    if (binding.port.direction == PortDirection::Output &&
        binding.port.index == index)
      return binding.memory;
  return {};
}

// The worked example: a 4-byte output bound to a 1024-byte SRAM and a
// 4096-byte output bound to an 8192-byte DRAM is legal, and each output is
// charged to the memory its own port selected.
TEST(CoveringSearch, ExplicitNamedPortMemoriesAreChargedPerPort) {
  mlir::MLIRContext context;
  mlir::Type small =
      mlir::RankedTensorType::get({1}, mlir::Float32Type::get(&context));
  mlir::Type large =
      mlir::RankedTensorType::get({1024}, mlir::Float32Type::get(&context));
  WorkloadGraph graph = multiOutputGraph(context, small, large);

  std::unique_ptr<MappingTarget> target =
      targetWith(sizedMemoryMachine(1024, 8192), kPortedMemoryRules);
  ASSERT_NE(target, nullptr);

  MappingSearchOptions options;
  options.mode = SearchMode::Exact;
  CoveringSearch search(graph, *target, context, LayoutContext{}, options);
  llvm::Expected<MappingSearchResult> result = search.search();
  ASSERT_TRUE(static_cast<bool>(result)) << llvm::toString(result.takeError());
  ASSERT_FALSE(result->plans.empty());
  ASSERT_FALSE(result->plans[0].placements.empty());
  const PlanPlacement &placement = result->plans[0].placements[0];
  EXPECT_EQ(placementMemoryForPort(placement, 0), "sram.0");
  EXPECT_EQ(placementMemoryForPort(placement, 1), "dram.0");
}

// Swapping which port gets which memory rejects: the 4096-byte output no longer
// fits the 1024-byte SRAM it is now bound to. Charging each port to its own
// memory is what makes this observable; the old first-output-only accounting
// would have charged both ports the 4-byte first output and admitted it.
TEST(CoveringSearch, SwappedNamedPortMemoriesAreRejected) {
  mlir::MLIRContext context;
  mlir::Type small =
      mlir::RankedTensorType::get({1}, mlir::Float32Type::get(&context));
  mlir::Type large =
      mlir::RankedTensorType::get({1024}, mlir::Float32Type::get(&context));
  WorkloadGraph graph = multiOutputGraph(context, small, large);

  std::unique_ptr<MappingTarget> target =
      targetWith(sizedMemoryMachine(1024, 8192), kSwappedMemoryRules);
  ASSERT_NE(target, nullptr);

  MappingSearchOptions options;
  options.mode = SearchMode::Exact;
  CoveringSearch search(graph, *target, context, LayoutContext{}, options);
  llvm::Expected<MappingSearchResult> result = search.search();
  ASSERT_TRUE(static_cast<bool>(result)) << llvm::toString(result.takeError());
  EXPECT_TRUE(result->plans.empty());
  EXPECT_TRUE(hasDiagnostic(*result, DiagnosticCode::MemoryCapacityExceeded));
}

// Both memories too small for the large port rejects, even though they are
// plentiful for the small one.
TEST(CoveringSearch, NamedPortMemoryTooSmallRejects) {
  mlir::MLIRContext context;
  mlir::Type small =
      mlir::RankedTensorType::get({1}, mlir::Float32Type::get(&context));
  mlir::Type large =
      mlir::RankedTensorType::get({1024}, mlir::Float32Type::get(&context));
  WorkloadGraph graph = multiOutputGraph(context, small, large);

  std::unique_ptr<MappingTarget> target =
      targetWith(sizedMemoryMachine(1024, 1024), kPortedMemoryRules);
  ASSERT_NE(target, nullptr);

  MappingSearchOptions options;
  options.mode = SearchMode::Exact;
  CoveringSearch search(graph, *target, context, LayoutContext{}, options);
  llvm::Expected<MappingSearchResult> result = search.search();
  ASSERT_TRUE(static_cast<bool>(result)) << llvm::toString(result.takeError());
  EXPECT_TRUE(result->plans.empty());
  EXPECT_TRUE(hasDiagnostic(*result, DiagnosticCode::MemoryCapacityExceeded));
}

// Both ports require the same memory kind, so they are distinguished by port
// alone. A 4-byte output lands in the small sram and a 4096-byte output in the
// large one; the kind-keyed binding map could not have told them apart.
TEST(CoveringSearch, SameKindNamedPortsBindDifferentMemories) {
  mlir::MLIRContext context;
  mlir::Type small =
      mlir::RankedTensorType::get({1}, mlir::Float32Type::get(&context));
  mlir::Type large =
      mlir::RankedTensorType::get({1024}, mlir::Float32Type::get(&context));
  WorkloadGraph graph = multiOutputGraph(context, small, large);

  std::unique_ptr<MappingTarget> target =
      targetWith(twoSramMachine(1024, 8192), kSameKindPortRules);
  ASSERT_NE(target, nullptr);

  MappingSearchOptions options;
  options.mode = SearchMode::Exact;
  CoveringSearch search(graph, *target, context, LayoutContext{}, options);
  llvm::Expected<MappingSearchResult> result = search.search();
  ASSERT_TRUE(static_cast<bool>(result)) << llvm::toString(result.takeError());
  ASSERT_FALSE(result->plans.empty());
  // At least one legal placement binds the large output to the large node and
  // the small output to the small node.
  bool sawSplit = false;
  for (const CoveringPlan &plan : result->plans)
    for (const PlanPlacement &placement : plan.placements)
      sawSplit |= placementMemoryForPort(placement, 0) == "sram.small" &&
                  placementMemoryForPort(placement, 1) == "sram.large";
  EXPECT_TRUE(sawSplit);
}

/// One `micro.vector` node reading a `tile`-typed input and writing `output`,
/// so a rule that names only the input still has an output to charge.
WorkloadGraph oneInOneOutGraph(mlir::MLIRContext &context, mlir::Type input,
                               mlir::Type output) {
  WorkloadGraph graph;
  WorkloadValueId in =
      graph.addValue(WorkloadValue{0, input, "in", /*external=*/true});
  WorkloadValueId out =
      graph.addValue(WorkloadValue{0, output, "out", /*external=*/false});
  WorkloadNode node;
  node.opName = "micro.vector";
  node.attributes = vectorAttributes(context);
  node.inputs.push_back(WorkloadPort{in, input, std::nullopt});
  node.outputs.push_back(WorkloadPort{out, output, std::nullopt});
  graph.addNode(std::move(node));
  graph.finalize();
  return graph;
}

/// A rule that names only its *input* port's memory. The node it covers still
/// produces an output, which must be charged rather than left unchecked.
constexpr llvm::StringLiteral kInputOnlyPortRule = R"llkmap(
rule r.in_ported {
  match micro.vector(op = "add");
  require executor kind worker;
  require memory input "operand0" kind sram;
  input "operand0";
  bundle "b";
  emit "e1";
  cost 1;
}
)llkmap";

// A named-port requirement that names only an input must not silently exempt
// the node's outputs from capacity: the 4096-byte output does not fit the
// 1024-byte sram the rule binds, so the plan rejects. (An input's own bytes are
// produced elsewhere; the output's are not.)
TEST(CoveringSearch, InputOnlyNamedMemoryStillChargesTheOutput) {
  mlir::MLIRContext context;
  mlir::Type small =
      mlir::RankedTensorType::get({1}, mlir::Float32Type::get(&context));
  mlir::Type large =
      mlir::RankedTensorType::get({1024}, mlir::Float32Type::get(&context));
  WorkloadGraph graph = oneInOneOutGraph(context, small, large);

  std::unique_ptr<MappingTarget> target =
      targetWith(sizedMemoryMachine(1024, 1u << 30), kInputOnlyPortRule);
  ASSERT_NE(target, nullptr);

  MappingSearchOptions options;
  options.mode = SearchMode::Exact;
  CoveringSearch search(graph, *target, context, LayoutContext{}, options);
  llvm::Expected<MappingSearchResult> result = search.search();
  ASSERT_TRUE(static_cast<bool>(result)) << llvm::toString(result.takeError());
  EXPECT_TRUE(result->plans.empty());
  EXPECT_TRUE(hasDiagnostic(*result, DiagnosticCode::MemoryCapacityExceeded));
}

// Control: the same input-only rule admits the plan once the bound memory holds
// the output, proving the charge is real rather than a blanket rejection.
TEST(CoveringSearch, InputOnlyNamedMemoryAdmitsWhenTheOutputFits) {
  mlir::MLIRContext context;
  mlir::Type small =
      mlir::RankedTensorType::get({1}, mlir::Float32Type::get(&context));
  mlir::Type large =
      mlir::RankedTensorType::get({1024}, mlir::Float32Type::get(&context));
  WorkloadGraph graph = oneInOneOutGraph(context, small, large);

  std::unique_ptr<MappingTarget> target =
      targetWith(sizedMemoryMachine(8192, 1u << 30), kInputOnlyPortRule);
  ASSERT_NE(target, nullptr);

  MappingSearchOptions options;
  options.mode = SearchMode::Exact;
  CoveringSearch search(graph, *target, context, LayoutContext{}, options);
  llvm::Expected<MappingSearchResult> result = search.search();
  ASSERT_TRUE(static_cast<bool>(result)) << llvm::toString(result.takeError());
  EXPECT_FALSE(result->plans.empty());
}

/// One `micro.vector` node reading a single input and writing nothing: a sink.
WorkloadGraph sinkGraph(mlir::MLIRContext &context, mlir::Type input) {
  WorkloadGraph graph;
  WorkloadValueId value =
      graph.addValue(WorkloadValue{0, input, "in", /*external=*/true});
  WorkloadNode node;
  node.opName = "micro.vector";
  node.attributes = vectorAttributes(context);
  node.inputs.push_back(WorkloadPort{value, input, std::nullopt});
  graph.addNode(std::move(node));
  graph.finalize();
  return graph;
}

// A sink writes no output to attribute, so a single bare binding is charged the
// first-input fallback and the plan is admitted when it fits. This is the
// control that keeps the port-association change from rejecting every
// outputless rule.
TEST(CoveringSearch, OutputlessSinkWithOneBindingIsAdmitted) {
  mlir::MLIRContext context;
  mlir::Type tile =
      mlir::RankedTensorType::get({8}, mlir::Float32Type::get(&context));
  WorkloadGraph graph = sinkGraph(context, tile);

  std::unique_ptr<MappingTarget> target =
      targetWith(searchMachine(), kRulesWithMemory);
  ASSERT_NE(target, nullptr);

  MappingSearchOptions options;
  options.mode = SearchMode::Exact;
  CoveringSearch search(graph, *target, context, LayoutContext{}, options);
  llvm::Expected<MappingSearchResult> result = search.search();
  ASSERT_TRUE(static_cast<bool>(result)) << llvm::toString(result.takeError());
  EXPECT_FALSE(result->plans.empty());
}

/// Two independent producer -> consumer pairs over `fanMachine`, each moving a
/// `value`-sized tile from dram.0 into acc.0 and writing a 4-byte result. The
/// pairs share no value, so the only thing linking them is the destination
/// memory both transfers fill.
WorkloadGraph twoSequentialTransferGraph(mlir::MLIRContext &context,
                                         mlir::Type value) {
  mlir::Type small =
      mlir::RankedTensorType::get({1}, mlir::Float32Type::get(&context));
  WorkloadGraph graph;
  auto pair = [&](llvm::StringRef tag, unsigned ordinal) {
    WorkloadValueId input = graph.addValue(
        WorkloadValue{0, value, tag.str() + ".in", /*external=*/true});
    WorkloadValueId middle = graph.addValue(
        WorkloadValue{0, value, tag.str() + ".mid", /*external=*/false});
    WorkloadValueId output = graph.addValue(
        WorkloadValue{0, small, tag.str() + ".out", /*external=*/false});

    WorkloadNode producer;
    producer.opName = "micro.vector";
    producer.sourceOrdinal = ordinal;
    producer.attributes = vectorAttributes(context, "produce");
    producer.inputs.push_back(WorkloadPort{input, value, std::nullopt});
    producer.outputs.push_back(WorkloadPort{middle, value, std::nullopt});
    graph.addNode(std::move(producer));

    WorkloadNode consumer;
    consumer.opName = "micro.vector";
    consumer.sourceOrdinal = ordinal + 1;
    consumer.attributes = vectorAttributes(context, "consume");
    consumer.inputs.push_back(WorkloadPort{middle, value, std::nullopt});
    consumer.outputs.push_back(WorkloadPort{output, small, std::nullopt});
    graph.addNode(std::move(consumer));
  };
  pair("a", 0);
  pair("b", 2);
  graph.finalize();
  return graph;
}

// A transfer destination is released once its consumer is placed, so two
// sequential transfers into one memory need room for only one buffer at a time.
// acc.0 holds both 4-byte outputs plus one 4096-byte input buffer (4104), not
// both buffers (8200): charging the destination without its lifetime would
// reject this legal plan, the over-charging direction the report warns about.
TEST(CoveringSearch, SequentialTransfersReuseTheirDestinationBuffer) {
  mlir::MLIRContext context;
  mlir::Type value =
      mlir::RankedTensorType::get({1024}, mlir::Float32Type::get(&context));
  WorkloadGraph graph = twoSequentialTransferGraph(context, value);

  MachineModel machine = fanMachine();
  machine.memories[1].capacityBytes = 4200; // one 4096 buffer + both outputs
  std::unique_ptr<MappingTarget> target = targetWith(machine, kFanRules);
  ASSERT_NE(target, nullptr);

  MappingSearchOptions options;
  options.mode = SearchMode::Deterministic;
  CoveringSearch search(graph, *target, context, LayoutContext{}, options);
  llvm::Expected<MappingSearchResult> result = search.search();
  ASSERT_TRUE(static_cast<bool>(result)) << llvm::toString(result.takeError());
  EXPECT_FALSE(result->plans.empty());
}

/// Two producers and two consumers of one `middle` value over `fanMachine`'s
/// dram.0 -> acc.0. The second consumer's input port carries `secondPortType`,
/// so a caller can make it incompatible with the `f32` value both consumers
/// read.
WorkloadGraph twoConsumerGatherGraph(mlir::MLIRContext &context,
                                     mlir::Type secondPortType) {
  mlir::Type f32 =
      mlir::RankedTensorType::get({16}, mlir::Float32Type::get(&context));
  WorkloadGraph graph;
  WorkloadValueId in1 =
      graph.addValue(WorkloadValue{0, f32, "in1", /*external=*/true});
  WorkloadValueId in2 =
      graph.addValue(WorkloadValue{0, f32, "in2", /*external=*/true});
  WorkloadValueId middle =
      graph.addValue(WorkloadValue{0, f32, "mid", /*external=*/false});
  WorkloadValueId out1 =
      graph.addValue(WorkloadValue{0, f32, "o1", /*external=*/false});
  WorkloadValueId out2 =
      graph.addValue(WorkloadValue{0, f32, "o2", /*external=*/false});

  auto producer = [&](WorkloadValueId input, unsigned ordinal) {
    WorkloadNode node;
    node.opName = "micro.vector";
    node.sourceOrdinal = ordinal;
    node.attributes = vectorAttributes(context, "produce");
    node.inputs.push_back(WorkloadPort{input, f32, std::nullopt});
    node.outputs.push_back(WorkloadPort{middle, f32, std::nullopt});
    graph.addNode(std::move(node));
  };
  producer(in1, 0);
  producer(in2, 1);

  auto consumer = [&](WorkloadValueId output, mlir::Type portType,
                      unsigned ordinal) {
    WorkloadNode node;
    node.opName = "micro.vector";
    node.sourceOrdinal = ordinal;
    node.attributes = vectorAttributes(context, "consume");
    node.inputs.push_back(WorkloadPort{middle, portType, std::nullopt});
    node.outputs.push_back(WorkloadPort{output, f32, std::nullopt});
    graph.addNode(std::move(node));
  };
  consumer(out1, f32, 2);
  consumer(out2, secondPortType, 3);
  graph.finalize();
  return graph;
}

// §15.3: a gather validates *every* consumer, not one representative. When the
// second consumer's port carries a different element type, no feed can legally
// serve it, so the whole gather must fail rather than attach it to a plan built
// from the first consumer's compatible facts. The guard first proves the
// fixture is viable with two compatible consumers, so the empty result below is
// the mismatched port. `WorkloadGraph::finalize` orders nodes by content key;
// the compatible consumer sorts first here, which is the order that puts the
// representative's facts in agreement while the other consumer disagrees.
TEST(CoveringSearch, GatherValidatesEveryConsumerNotJustTheFirst) {
  mlir::MLIRContext context;
  mlir::Type f32 =
      mlir::RankedTensorType::get({16}, mlir::Float32Type::get(&context));
  mlir::Type incompatible =
      mlir::RankedTensorType::get({16}, mlir::Float64Type::get(&context));

  std::unique_ptr<MappingTarget> target = targetWith(fanMachine(), kFanRules);
  ASSERT_NE(target, nullptr);
  MappingSearchOptions options;
  options.mode = SearchMode::Deterministic;

  {
    // Guard: two compatible consumers are a legal gather.
    WorkloadGraph viable = twoConsumerGatherGraph(context, f32);
    CoveringSearch search(viable, *target, context, LayoutContext{}, options);
    llvm::Expected<MappingSearchResult> result = search.search();
    ASSERT_TRUE(static_cast<bool>(result))
        << llvm::toString(result.takeError());
    EXPECT_FALSE(result->plans.empty());
  }
  {
    WorkloadGraph split = twoConsumerGatherGraph(context, incompatible);
    CoveringSearch search(split, *target, context, LayoutContext{}, options);
    llvm::Expected<MappingSearchResult> result = search.search();
    ASSERT_TRUE(static_cast<bool>(result))
        << llvm::toString(result.takeError());
    EXPECT_TRUE(result->plans.empty());
  }
}

TEST(CoveringSearch, ReportsTruncationWhenInstanceCapIsHit) {
  mlir::MLIRContext context;
  WorkloadGraph graph = twoNodeGraph(context);
  std::unique_ptr<MappingTarget> target =
      targetWith(twoWorkerMachine(), kRules);
  ASSERT_NE(target, nullptr);

  MappingSearchOptions options;
  options.mode = SearchMode::Deterministic;
  options.maxInstancesPerCandidate = 1;    // force the cap
  options.enableSymmetryReduction = false; // keep both workers legal
  CoveringSearch search(graph, *target, context, LayoutContext{}, options);
  llvm::Expected<MappingSearchResult> result = search.search();
  ASSERT_TRUE(static_cast<bool>(result)) << llvm::toString(result.takeError());
  EXPECT_TRUE(result->searchTruncated);
}

//===----------------------------------------------------------------------===//
// Measured latencies
//===----------------------------------------------------------------------===//

TEST(CoveringSearch, NoProviderLeavesTheStaticEstimate) {
  mlir::MLIRContext context;
  WorkloadGraph graph = twoNodeGraph(context);
  std::unique_ptr<MappingTarget> target = targetWith(searchMachine(), kRules);
  ASSERT_NE(target, nullptr);
  EXPECT_EQ(target->latencyProvider(), nullptr);

  MappingSearchOptions options;
  options.mode = SearchMode::Exact;
  CoveringSearch search(graph, *target, context, LayoutContext{}, options);
  llvm::Expected<MappingSearchResult> result = search.search();
  ASSERT_TRUE(static_cast<bool>(result));
  ASSERT_FALSE(result->plans.empty());
  // The cheapest declared rule is 1 per node.
  EXPECT_DOUBLE_EQ(result->plans[0].totalCost.latencyCycles, 2.0);
}

TEST(CoveringSearch, AMeasurementChangesTheCostAndTheRanking) {
  mlir::MLIRContext context;
  WorkloadGraph graph = twoNodeGraph(context);
  FixedLatencyProvider provider;
  // The declared-cheap rule is measured to be slow, and the other fast.
  provider.byRule["r.cheap"] = 100.0;
  provider.byRule["r.expensive"] = 2.0;

  std::unique_ptr<MappingTarget> target =
      targetWithProvider(searchMachine(), kRules, &provider);
  ASSERT_NE(target, nullptr);
  EXPECT_EQ(target->latencyProvider(), &provider);

  MappingSearchOptions options;
  options.mode = SearchMode::Exact;
  CoveringSearch search(graph, *target, context, LayoutContext{}, options);
  llvm::Expected<MappingSearchResult> result = search.search();
  ASSERT_TRUE(static_cast<bool>(result)) << llvm::toString(result.takeError());
  ASSERT_FALSE(result->plans.empty());
  // Two nodes at the measured 2 cycles each; the declared 1 no longer wins.
  EXPECT_DOUBLE_EQ(result->plans[0].totalCost.latencyCycles, 4.0);
  ASSERT_FALSE(result->plans[0].placements.empty());
  EXPECT_EQ(result->plans[0].placements[0].rule, "r.expensive");
  ASSERT_FALSE(provider.lookups.empty());
  // The search fills the whole §17.4 key, not just the rule: the operation's
  // attributes and its placement class are part of what it asked about.
  EXPECT_NE(provider.lookups.front().find("attributes={op = \"add\"}"),
            std::string::npos)
      << provider.lookups.front();
  EXPECT_NE(provider.lookups.front().find("placement_class=worker"),
            std::string::npos)
      << provider.lookups.front();
}

// A typed fixture proves the cache key carries the operation's types and the
// concrete placement, not only the rule: the fields are filled from the real
// node and instance rather than left blank.
TEST(CoveringSearch, LatencyKeyCarriesTheOperationsTypesAndPlacement) {
  mlir::MLIRContext context;
  mlir::Type small =
      mlir::RankedTensorType::get({1}, mlir::Float32Type::get(&context));
  WorkloadGraph graph = multiOutputGraph(context, small, small);
  FixedLatencyProvider provider;

  std::unique_ptr<MappingTarget> target =
      targetWithProvider(searchMachine(), kOneMemoryRules, &provider);
  ASSERT_NE(target, nullptr);
  MappingSearchOptions options;
  options.mode = SearchMode::Exact;
  CoveringSearch search(graph, *target, context, LayoutContext{}, options);
  llvm::Expected<MappingSearchResult> result = search.search();
  ASSERT_TRUE(static_cast<bool>(result)) << llvm::toString(result.takeError());
  ASSERT_FALSE(provider.lookups.empty());

  const std::string &key = provider.lookups.front();
  EXPECT_NE(key.find("result_types=tensor<1xf32>"), std::string::npos) << key;
  EXPECT_NE(key.find("placement=executor=e0,memories=sram=sram.0"),
            std::string::npos)
      << key;
}

// §17.4: the measurement cache key must separate any two pieces of work whose
// measured cost could differ. Each pair below changes exactly one component and
// must therefore produce a different key; an unchanged copy must not.
TEST(CoveringSearch, LatencyKeysDistinguishWorkThatCanDiffer) {
  OperationSignature base;
  base.operation = "micro.vector";
  base.operandTypes = "tensor<8x8xf32>";
  base.resultTypes = "tensor<8x8xf32>";
  base.attributes = "{op = \"add\"}";
  base.rule = "r";
  base.ruleVersion = 1;
  base.bundle = "b";
  base.bundleParameters = "{VW = 8 : i64}";
  base.layout = "t.blocked(VW=8)";
  base.placementClass = "core";
  base.placement = "executor=e0,memories=sram.0";
  base.routeClass = "direct";

  const std::string key = base.canonicalString();

  OperationSignature operand = base;
  operand.operandTypes = "tensor<8x8xbf16>";
  EXPECT_NE(operand.canonicalString(), key);

  OperationSignature result = base;
  result.resultTypes = "tensor<4x4xf32>";
  EXPECT_NE(result.canonicalString(), key);

  OperationSignature attributes = base;
  attributes.attributes = "{op = \"mul\"}";
  EXPECT_NE(attributes.canonicalString(), key);

  OperationSignature bundleParameters = base;
  bundleParameters.bundleParameters = "{VW = 4 : i64}";
  EXPECT_NE(bundleParameters.canonicalString(), key);

  OperationSignature layout = base;
  layout.layout = "t.blocked(VW=4)";
  EXPECT_NE(layout.canonicalString(), key);

  OperationSignature placement = base;
  placement.placement = "executor=e1,memories=acc.0";
  EXPECT_NE(placement.canonicalString(), key);

  OperationSignature unchanged = base;
  EXPECT_EQ(unchanged.canonicalString(), key);
}

TEST(CoveringSearch, AnEntrylessProviderFallsBackToStaticCost) {
  mlir::MLIRContext context;
  WorkloadGraph graph = twoNodeGraph(context);
  FixedLatencyProvider provider; // no entries at all

  std::unique_ptr<MappingTarget> target =
      targetWithProvider(searchMachine(), kRules, &provider);
  ASSERT_NE(target, nullptr);
  MappingSearchOptions options;
  options.mode = SearchMode::Exact;
  CoveringSearch search(graph, *target, context, LayoutContext{}, options);
  llvm::Expected<MappingSearchResult> result = search.search();
  ASSERT_TRUE(static_cast<bool>(result));
  ASSERT_FALSE(result->plans.empty());
  EXPECT_DOUBLE_EQ(result->plans[0].totalCost.latencyCycles, 2.0);
  EXPECT_FALSE(provider.lookups.empty()); // it was asked, and declined
}

TEST(CoveringSearch, AnExpensiveMeasurementDoesNotMakeAPlanIllegal) {
  mlir::MLIRContext context;
  WorkloadGraph graph = twoNodeGraph(context);
  FixedLatencyProvider provider;
  provider.byRule["r.cheap"] = 1e9;
  provider.byRule["r.expensive"] = 1e9;

  std::unique_ptr<MappingTarget> target =
      targetWithProvider(searchMachine(), kRules, &provider);
  ASSERT_NE(target, nullptr);
  MappingSearchOptions options;
  options.mode = SearchMode::Exact;
  CoveringSearch search(graph, *target, context, LayoutContext{}, options);
  llvm::Expected<MappingSearchResult> result = search.search();
  ASSERT_TRUE(static_cast<bool>(result)) << llvm::toString(result.takeError());
  // Measurement changes what work costs, never what is allowed.
  EXPECT_FALSE(result->plans.empty());
}

TEST(CoveringSearch, LatencyCacheCanBeDisabled) {
  mlir::MLIRContext context;
  WorkloadGraph graph = twoNodeGraph(context);
  FixedLatencyProvider provider;
  // The declared-cheap rule is measured to be slow, and the other fast. With
  // the cache enabled this flips the ranking, exactly as the test above shows.
  provider.byRule["r.cheap"] = 100.0;
  provider.byRule["r.expensive"] = 2.0;

  std::unique_ptr<MappingTarget> target =
      targetWithProvider(searchMachine(), kRules, &provider);
  ASSERT_NE(target, nullptr);
  ASSERT_EQ(target->latencyProvider(), &provider);

  MappingSearchOptions disabled;
  disabled.mode = SearchMode::Exact;
  disabled.enableLatencyCache = false;
  CoveringSearch search(graph, *target, context, LayoutContext{}, disabled);
  llvm::Expected<MappingSearchResult> result = search.search();
  ASSERT_TRUE(static_cast<bool>(result)) << llvm::toString(result.takeError());
  ASSERT_FALSE(result->plans.empty());
  // The provider was never consulted, so the declared 1-cycle rule still wins
  // and the cost is the static estimate.
  EXPECT_EQ(provider.lookupCount, 0u);
  EXPECT_TRUE(provider.lookups.empty());
  EXPECT_DOUBLE_EQ(result->plans[0].totalCost.latencyCycles, 2.0);

  // The default (`true`) does consult it, and the measurement takes effect.
  FixedLatencyProvider measured;
  measured.byRule["r.cheap"] = 100.0;
  measured.byRule["r.expensive"] = 2.0;
  std::unique_ptr<MappingTarget> measuredTarget =
      targetWithProvider(searchMachine(), kRules, &measured);
  ASSERT_NE(measuredTarget, nullptr);

  MappingSearchOptions enabled;
  enabled.mode = SearchMode::Exact;
  CoveringSearch enabledSearch(graph, *measuredTarget, context, LayoutContext{},
                               enabled);
  llvm::Expected<MappingSearchResult> enabledResult = enabledSearch.search();
  ASSERT_TRUE(static_cast<bool>(enabledResult))
      << llvm::toString(enabledResult.takeError());
  ASSERT_FALSE(enabledResult->plans.empty());
  EXPECT_GT(measured.lookupCount, 0u);
  EXPECT_DOUBLE_EQ(enabledResult->plans[0].totalCost.latencyCycles, 4.0);
}

//===----------------------------------------------------------------------===//
// Objective-declared ranking (§17.1)
//===----------------------------------------------------------------------===//

// `micro.objective` supplies the comparison order, so the search must obey the
// declared objective rather than a hard-coded latency minimum. Maximizing
// latency makes the effect visible: the pricier rule per node must rank first.
TEST(CoveringSearch, TheDeclaredObjectiveDirectionDecidesTheRanking) {
  mlir::MLIRContext context;
  WorkloadGraph graph = twoNodeGraph(context);
  std::unique_ptr<MappingTarget> target = targetWith(searchMachine(), kRules);
  ASSERT_NE(target, nullptr);

  MappingSearchOptions options;
  options.mode = SearchMode::Beam;
  options.objective =
      ObjectiveOrder{CostMetric::LatencyCycles, {}, /*minimize=*/false};
  CoveringSearch search(graph, *target, context, LayoutContext{}, options);
  llvm::Expected<MappingSearchResult> result = search.search();
  ASSERT_TRUE(static_cast<bool>(result)) << llvm::toString(result.takeError());
  ASSERT_FALSE(result->plans.empty());

  // The 10-cycle rule per node is the *worst* under the default objective and
  // the best under this declared one -- proving the option reaches the final
  // ranking instead of a hard-coded minimum.
  ASSERT_FALSE(result->plans[0].placements.empty());
  EXPECT_EQ(result->plans[0].placements[0].rule, "r.expensive");
  EXPECT_DOUBLE_EQ(result->plans[0].totalCost.latencyCycles, 20.0);
}

// The default objective is the pre-existing behaviour: latency minimized.
TEST(CoveringSearch, TheDefaultObjectiveStillMinimizesLatency) {
  mlir::MLIRContext context;
  WorkloadGraph graph = twoNodeGraph(context);
  std::unique_ptr<MappingTarget> target = targetWith(searchMachine(), kRules);
  ASSERT_NE(target, nullptr);

  MappingSearchOptions options;
  options.mode = SearchMode::Beam;
  CoveringSearch search(graph, *target, context, LayoutContext{}, options);
  llvm::Expected<MappingSearchResult> result = search.search();
  ASSERT_TRUE(static_cast<bool>(result)) << llvm::toString(result.takeError());
  ASSERT_FALSE(result->plans.empty());
  ASSERT_FALSE(result->plans[0].placements.empty());
  EXPECT_EQ(result->plans[0].placements[0].rule, "r.cheap");
  EXPECT_DOUBLE_EQ(result->plans[0].totalCost.latencyCycles, 2.0);
}

// Two plans that differ in DRAM: a slow plan moving fewer bytes ranks ahead of
// a fast plan moving more, because the objective names DRAM first. An exact
// cost tie falls back to the smaller stable id.
TEST(CoveringSearch, DramObjectiveRanksBeforeLatency) {
  Cost fastManyDram;
  fastManyDram.latencyCycles = 1.0;
  fastManyDram.dramBytes = 100;
  Cost slowFewDram;
  slowFewDram.latencyCycles = 9.0;
  slowFewDram.dramBytes = 10;

  ObjectiveOrder byDram{CostMetric::DramBytes, {}, true};
  EXPECT_TRUE(ranksBefore(slowFewDram, 2, fastManyDram, 1, byDram));
  EXPECT_FALSE(ranksBefore(fastManyDram, 1, slowFewDram, 2, byDram));

  Cost a;
  a.latencyCycles = 5.0;
  Cost b;
  b.latencyCycles = 5.0;
  ObjectiveOrder byLatency{CostMetric::LatencyCycles, {}, true};
  EXPECT_TRUE(ranksBefore(a, 1, b, 2, byLatency));
  EXPECT_FALSE(ranksBefore(b, 2, a, 1, byLatency));
}

//===----------------------------------------------------------------------===//
// The search bound: multi-dimensional, measured, direction-aware
//===----------------------------------------------------------------------===//

// The bound must follow the *measured* cost. Both rules declare a static
// 100-cycle cost; a provider measures `r.b_measured` down to 1. The genuinely
// cheapest plan therefore places `r.b_measured` on both nodes -- and the bound
// must not over-estimate it from the static 100 and prune it.
TEST(CoveringSearch, TheBoundUsesTheMeasuredCostNotTheStaticOne) {
  mlir::MLIRContext context;
  WorkloadGraph graph = twoNodeGraph(context);
  FixedLatencyProvider provider;
  provider.byRule["r.b_measured"] = 1.0; // `r.a_static` keeps the static 100
  std::unique_ptr<MappingTarget> target =
      targetWithProvider(searchMachine(), kMeasuredBoundRules, &provider);
  ASSERT_NE(target, nullptr);

  MappingSearchOptions options;
  options.mode = SearchMode::Exact;
  options.topK = 1; // a full top-K list is what arms the prune
  CoveringSearch search(graph, *target, context, LayoutContext{}, options);
  llvm::Expected<MappingSearchResult> result = search.search();
  ASSERT_TRUE(static_cast<bool>(result)) << llvm::toString(result.takeError());
  ASSERT_FALSE(result->plans.empty());
  // 1 + 1 measured cycles; a static-only bound prunes this branch at 101.
  EXPECT_DOUBLE_EQ(result->plans[0].totalCost.latencyCycles, 2.0);
  ASSERT_FALSE(result->plans[0].placements.empty());
  EXPECT_EQ(result->plans[0].placements[0].rule, "r.b_measured");
}

// The bound must answer to the *declared* metric. The latency-cheaper plan
// moves 4096 bytes through DRAM; the plan that minimizes DRAM is 50 cycles
// slower in place. A latency-only bound prunes the DRAM-cheap plan; a bound
// that reads the DramBytes dimension keeps it, and it ranks first.
TEST(CoveringSearch, TheBoundIsObjectiveAware) {
  mlir::MLIRContext context;
  WorkloadGraph graph = twoNodeGraph(context, "produce", "consume");
  std::unique_ptr<MappingTarget> target =
      targetWith(objectiveMachine(), kObjectiveRules);
  ASSERT_NE(target, nullptr);

  MappingSearchOptions options;
  options.mode = SearchMode::Exact;
  options.topK = 1;
  options.objective = ObjectiveOrder{CostMetric::DramBytes, {}, true};
  CoveringSearch search(graph, *target, context, LayoutContext{}, options);
  llvm::Expected<MappingSearchResult> result = search.search();
  ASSERT_TRUE(static_cast<bool>(result)) << llvm::toString(result.takeError());
  ASSERT_FALSE(result->plans.empty());
  // The SRAM producer reads in place: no DRAM traffic, 51 cycles rather than
  // the DRAM plan's 3. A latency-only bound prunes it and reports the 4096-byte
  // plan instead.
  EXPECT_EQ(result->plans[0].totalCost.dramBytes, 0u);
  bool usesDramCheap = false;
  for (const PlanPlacement &placement : result->plans[0].placements)
    usesDramCheap |= placement.rule == "r.p_sram";
  EXPECT_TRUE(usesDramCheap);
}

// §17.1: the objective's secondary metrics are tie-breakers, not decoration.
// Two complete plans here tie at 51 cycles on the primary metric and differ
// only on `dram_bytes`; the DRAM-free plan must rank first even though the
// primary sees a tie. With `topK = 1` the exact prune is armed, so a prune that
// compared only the primary -- treating an exact primary tie as "not better" --
// would drop the DRAM-free branch. It must survive on the secondary metric.
TEST(CoveringSearch, SecondaryMetricBreaksAPrimaryTie) {
  mlir::MLIRContext context;
  WorkloadGraph graph = twoNodeGraph(context, "produce", "consume");
  std::unique_ptr<MappingTarget> target =
      targetWith(objectiveMachine(), kTieRules);
  ASSERT_NE(target, nullptr);

  ObjectiveOrder order{CostMetric::LatencyCycles,
                       {CostMetric::DramBytes},
                       /*minimize=*/true};

  // A wide top-K keeps both plans, so the tie and the tie-break are both
  // observable rather than inferred from which plan survived a prune.
  {
    MappingSearchOptions options;
    options.mode = SearchMode::Exact;
    options.topK = 2;
    options.objective = order;
    CoveringSearch search(graph, *target, context, LayoutContext{}, options);
    llvm::Expected<MappingSearchResult> result = search.search();
    ASSERT_TRUE(static_cast<bool>(result))
        << llvm::toString(result.takeError());
    ASSERT_EQ(result->plans.size(), 2u);
    EXPECT_DOUBLE_EQ(result->plans[0].totalCost.latencyCycles, 51.0);
    EXPECT_DOUBLE_EQ(result->plans[1].totalCost.latencyCycles, 51.0);
    // ...and the only difference is the secondary metric, which ranks the
    // DRAM-free plan first.
    EXPECT_EQ(result->plans[0].totalCost.dramBytes, 0u);
    EXPECT_EQ(result->plans[1].totalCost.dramBytes, 4096u);
  }

  // A full top-K list arms the exact prune. The DRAM plan completes first, so
  // the DRAM-free branch's bound ties it on latency; only the secondary metric
  // can keep that branch alive.
  {
    MappingSearchOptions options;
    options.mode = SearchMode::Exact;
    options.topK = 1;
    options.objective = order;
    CoveringSearch search(graph, *target, context, LayoutContext{}, options);
    llvm::Expected<MappingSearchResult> result = search.search();
    ASSERT_TRUE(static_cast<bool>(result))
        << llvm::toString(result.takeError());
    ASSERT_EQ(result->plans.size(), 1u);
    EXPECT_DOUBLE_EQ(result->plans[0].totalCost.latencyCycles, 51.0);
    EXPECT_EQ(result->plans[0].totalCost.dramBytes, 0u);
    bool usesDramFree = false;
    for (const PlanPlacement &placement : result->plans[0].placements)
      usesDramFree |= placement.rule == "r.p_sram";
    EXPECT_TRUE(usesDramFree);
  }
}

// The bound must be direction-aware. Maximizing latency keeps the *largest*
// reachable completion; a componentwise-min bound (the minimize rule) would
// keep the 1-cycle branch and prune the 10-cycle plan the objective asks for.
TEST(CoveringSearch, TheBoundFollowsTheObjectiveDirection) {
  mlir::MLIRContext context;
  WorkloadGraph graph = twoNodeGraph(context);
  std::unique_ptr<MappingTarget> target = targetWith(searchMachine(), kRules);
  ASSERT_NE(target, nullptr);

  MappingSearchOptions options;
  options.mode = SearchMode::Beam;
  options.beamWidth = 1; // the sort, not the final rank, decides what survives
  options.objective =
      ObjectiveOrder{CostMetric::LatencyCycles, {}, /*minimize=*/false};
  CoveringSearch search(graph, *target, context, LayoutContext{}, options);
  llvm::Expected<MappingSearchResult> result = search.search();
  ASSERT_TRUE(static_cast<bool>(result)) << llvm::toString(result.takeError());
  ASSERT_FALSE(result->plans.empty());
  ASSERT_FALSE(result->plans[0].placements.empty());
  EXPECT_EQ(result->plans[0].placements[0].rule, "r.expensive");
  EXPECT_DOUBLE_EQ(result->plans[0].totalCost.latencyCycles, 20.0);
}

// The bound omits connection costs, so it is only admissible for a minimize
// objective. Plan B is largest (103 cycles) only *after* its 101-cycle route is
// added; while the consumer is still uncovered the bound sees 1 + 1 = 2. A
// maximize objective must not prune on that bound: exact mode explores fully
// and returns B, rather than reporting the 13-cycle plan as exact.
TEST(CoveringSearch, ExactMaximizeDoesNotPruneOnAnInadmissibleBound) {
  mlir::MLIRContext context;
  WorkloadGraph graph = twoNodeGraph(context, "a_produce", "z_consume");
  std::unique_ptr<MappingTarget> target =
      targetWith(maximizeMachine(), kMaximizeRules);
  ASSERT_NE(target, nullptr);

  MappingSearchOptions options;
  options.mode = SearchMode::Exact;
  options.topK = 1;
  options.objective =
      ObjectiveOrder{CostMetric::LatencyCycles, {}, /*minimize=*/false};
  CoveringSearch search(graph, *target, context, LayoutContext{}, options);
  llvm::Expected<MappingSearchResult> result = search.search();
  ASSERT_TRUE(static_cast<bool>(result)) << llvm::toString(result.takeError());
  ASSERT_FALSE(result->plans.empty());
  // Producer on acc.0: 1 + 1 + (100 + 4096 / 4096) = 103 cycles.
  EXPECT_DOUBLE_EQ(result->plans[0].totalCost.latencyCycles, 103.0);
  bool usesSlow = false;
  for (const PlanPlacement &placement : result->plans[0].placements)
    usesSlow |= placement.rule == "r.p_slow";
  EXPECT_TRUE(usesSlow);
}

//===----------------------------------------------------------------------===//
// Source binding provenance
//===----------------------------------------------------------------------===//

TEST(CoveringSearch, PlansCarryTheSourceBindingHash) {
  mlir::MLIRContext context;
  WorkloadGraph graph = twoNodeGraph(context);
  std::unique_ptr<MappingTarget> target = targetWith(searchMachine(), kRules);
  ASSERT_NE(target, nullptr);

  SearchBinding binding = makeSearchBinding(
      "candidate_17",
      values({{"BM", int64_t{64}}, {"tile_layout", std::string("blocked")}}));

  MappingSearchOptions options;
  options.mode = SearchMode::Deterministic;
  CoveringSearch search(graph, *target, context, LayoutContext{}, options,
                        binding);
  llvm::Expected<MappingSearchResult> result = search.search();
  ASSERT_TRUE(static_cast<bool>(result)) << llvm::toString(result.takeError());
  ASSERT_FALSE(result->plans.empty());

  // Every emitted plan records the binding it came from (design §8.3/§9.5).
  EXPECT_EQ(result->plans[0].sourceBindingHash, binding.stableHash);
  EXPECT_NE(result->plans[0].sourceBindingHash, 0u);
  EXPECT_NE(result->plans[0].id, 0u);

  // ...and carries its parameters, so a plan is traceable to its search point.
  EXPECT_EQ(result->plans[0].globalParameters.size(), 2u);
  EXPECT_EQ(
      std::get<std::string>(result->plans[0].globalParameters["tile_layout"]),
      "blocked");
  EXPECT_EQ(std::get<int64_t>(result->plans[0].globalParameters["BM"]),
            int64_t{64});
}

TEST(CoveringSearch, DifferentBindingsYieldDifferentPlanIds) {
  mlir::MLIRContext context;
  WorkloadGraph graph = twoNodeGraph(context);
  std::unique_ptr<MappingTarget> target = targetWith(searchMachine(), kRules);
  ASSERT_NE(target, nullptr);

  SearchBinding bindingA = makeSearchBinding(
      "candidate_17",
      values({{"BM", int64_t{64}}, {"tile_layout", std::string("blocked")}}));
  SearchBinding bindingB = makeSearchBinding(
      "candidate_17",
      values({{"BM", int64_t{32}}, {"tile_layout", std::string("blocked")}}));

  MappingSearchOptions options;
  options.mode = SearchMode::Deterministic;
  CoveringSearch searchA(graph, *target, context, LayoutContext{}, options,
                         bindingA);
  CoveringSearch searchB(graph, *target, context, LayoutContext{}, options,
                         bindingB);
  llvm::Expected<MappingSearchResult> resultA = searchA.search();
  llvm::Expected<MappingSearchResult> resultB = searchB.search();
  ASSERT_TRUE(static_cast<bool>(resultA))
      << llvm::toString(resultA.takeError());
  ASSERT_TRUE(static_cast<bool>(resultB))
      << llvm::toString(resultB.takeError());
  ASSERT_FALSE(resultA->plans.empty());
  ASSERT_FALSE(resultB->plans.empty());

  // Two plans that differ only by their search point must not collide.
  EXPECT_NE(resultA->plans[0].sourceBindingHash,
            resultB->plans[0].sourceBindingHash);
  EXPECT_NE(resultA->plans[0].id, resultB->plans[0].id);
}

TEST(CoveringSearch, NoBindingLeavesTheHashZeroAndThePlanIdUnchanged) {
  mlir::MLIRContext context;
  WorkloadGraph graph = twoNodeGraph(context);
  std::unique_ptr<MappingTarget> target = targetWith(searchMachine(), kRules);
  ASSERT_NE(target, nullptr);

  MappingSearchOptions options;
  options.mode = SearchMode::Deterministic;
  CoveringSearch search(graph, *target, context, LayoutContext{}, options);
  llvm::Expected<MappingSearchResult> result = search.search();
  ASSERT_TRUE(static_cast<bool>(result)) << llvm::toString(result.takeError());
  ASSERT_FALSE(result->plans.empty());

  // The default (no binding) folds a zero hash and no parameters, so the plan
  // id is exactly what it was before bindings were recorded.
  EXPECT_EQ(result->plans[0].sourceBindingHash, 0u);
  EXPECT_TRUE(result->plans[0].globalParameters.empty());
  EXPECT_EQ(result->plans[0].id, kNoBindingPlanId);
}

//===----------------------------------------------------------------------===//
// Bound owner_mapping / memory_path axes are projected, not ignored
//===----------------------------------------------------------------------===//

namespace {

/// Two executors of different owner kinds, one visible memory system, so a
/// bound `owner_mapping` axis can decide which executor a rule that names no
/// owner of its own places on.
MachineModel ownerMachine() {
  MachineModel model;
  model.target = "owner";
  model.executors = {
      {"cluster.0", "cluster", std::nullopt, {}, 1, {}},
      {"w0", "worker", std::string("cluster.0"), {}, 1, {}},
      {"v0", "vector_engine", std::string("cluster.0"), {}, 1, {}}};
  MemoryNode sram;
  sram.id = "sram.0";
  sram.kind = "sram";
  sram.visibleFrom = "cluster.0";
  sram.capacityBytes = 1u << 20;
  MemoryNode dram;
  dram.id = "dram.0";
  dram.kind = "dram";
  dram.visibleFrom = "cluster.0";
  dram.capacityBytes = 1u << 30;
  model.memories = {sram, dram};
  return model;
}

/// A rule that declares no owner of its own, so the only thing that can select
/// an executor is the bound owner_mapping axis.
constexpr llvm::StringLiteral kOwnerFreeRules = R"llkmap(
rule r.free {
  match micro.vector(op = "add");
  bundle "b";
  emit "e1";
  cost 1;
}
)llkmap";

/// A rule that pins the executor kind, so a contradictory bound owner rejects.
constexpr llvm::StringLiteral kVectorOwnerRule = R"llkmap(
rule r.vec {
  match micro.vector(op = "add");
  require executor kind vector_engine;
  bundle "b";
  emit "e1";
  cost 1;
}
)llkmap";

} // namespace

// A bound owner_mapping projects onto an abstract executor requirement, so it
// changes which executor the plan places on -- rather than being recorded as
// provenance and ignored.
TEST(CoveringSearch, ABoundOwnerMappingSelectsThePlacement) {
  mlir::MLIRContext context;
  WorkloadGraph graph = twoNodeGraph(context);
  std::unique_ptr<MappingTarget> target =
      targetWith(ownerMachine(), kOwnerFreeRules);
  ASSERT_NE(target, nullptr);

  MappingSearchOptions options;
  options.mode = SearchMode::Deterministic;

  auto searchWith = [&](llvm::StringRef owner) {
    SearchBinding binding =
        makeSearchBinding("c", values({{"owner_mapping", std::string(owner)}}));
    CoveringSearch search(graph, *target, context, LayoutContext{}, options,
                          binding);
    return search.search();
  };

  llvm::Expected<MappingSearchResult> worker = searchWith("worker");
  ASSERT_TRUE(static_cast<bool>(worker)) << llvm::toString(worker.takeError());
  ASSERT_FALSE(worker->plans.empty());
  for (const PlanPlacement &placement : worker->plans[0].placements)
    EXPECT_EQ(placement.executor, "w0");

  llvm::Expected<MappingSearchResult> vector = searchWith("vector_engine");
  ASSERT_TRUE(static_cast<bool>(vector)) << llvm::toString(vector.takeError());
  ASSERT_FALSE(vector->plans.empty());
  for (const PlanPlacement &placement : vector->plans[0].placements)
    EXPECT_EQ(placement.executor, "v0");
}

// An owner_mapping naming an owner the machine does not model is rejected
// explicitly: the node has no rule in effect rather than the value being
// silently dropped.
TEST(CoveringSearch, AnUnmodeledBoundOwnerMappingIsRejected) {
  mlir::MLIRContext context;
  WorkloadGraph graph = twoNodeGraph(context);
  std::unique_ptr<MappingTarget> target =
      targetWith(ownerMachine(), kOwnerFreeRules);
  ASSERT_NE(target, nullptr);

  SearchBinding binding =
      makeSearchBinding("c", values({{"owner_mapping", std::string("pe")}}));
  MappingSearchOptions options;
  options.mode = SearchMode::Deterministic;
  CoveringSearch search(graph, *target, context, LayoutContext{}, options,
                        binding);
  llvm::Expected<MappingSearchResult> result = search.search();
  ASSERT_TRUE(static_cast<bool>(result)) << llvm::toString(result.takeError());
  EXPECT_TRUE(result->plans.empty());
  EXPECT_TRUE(hasDiagnostic(*result, DiagnosticCode::NoMatchingRule));
}

// A rule that requires a different executor kind than the bound owner is a
// contradiction and rejects.
TEST(CoveringSearch, AContradictoryBoundOwnerMappingIsRejected) {
  mlir::MLIRContext context;
  WorkloadGraph graph = twoNodeGraph(context);
  std::unique_ptr<MappingTarget> target =
      targetWith(ownerMachine(), kVectorOwnerRule);
  ASSERT_NE(target, nullptr);

  SearchBinding binding = makeSearchBinding(
      "c", values({{"owner_mapping", std::string("worker")}}));
  MappingSearchOptions options;
  options.mode = SearchMode::Deterministic;
  CoveringSearch search(graph, *target, context, LayoutContext{}, options,
                        binding);
  llvm::Expected<MappingSearchResult> result = search.search();
  ASSERT_TRUE(static_cast<bool>(result)) << llvm::toString(result.takeError());
  EXPECT_TRUE(result->plans.empty());
  EXPECT_TRUE(hasDiagnostic(*result, DiagnosticCode::NoMatchingRule));
}

// A bound memory_path is the set of allowed memory kinds: a rule whose memory
// requirement sits off the path rejects, one on the path is admitted, and a
// path naming an unmodeled level rejects explicitly.
TEST(CoveringSearch, ABoundMemoryPathGatesMemoryRequirements) {
  mlir::MLIRContext context;
  WorkloadGraph graph = twoNodeGraph(context);
  std::unique_ptr<MappingTarget> target =
      targetWith(searchMachine(), kRulesWithMemory);
  ASSERT_NE(target, nullptr);

  MappingSearchOptions options;
  options.mode = SearchMode::Deterministic;

  auto searchWith = [&](llvm::StringRef path) {
    SearchBinding binding =
        makeSearchBinding("c", values({{"memory_path", std::string(path)}}));
    CoveringSearch search(graph, *target, context, LayoutContext{}, options,
                          binding);
    return search.search();
  };

  // sram is off this path, so the requirement cannot be met.
  llvm::Expected<MappingSearchResult> offPath = searchWith("dram");
  ASSERT_TRUE(static_cast<bool>(offPath))
      << llvm::toString(offPath.takeError());
  EXPECT_TRUE(offPath->plans.empty());
  EXPECT_TRUE(hasDiagnostic(*offPath, DiagnosticCode::NoMatchingRule));

  // sram is on this path, so the rule places as before.
  llvm::Expected<MappingSearchResult> onPath = searchWith("dram:sram");
  ASSERT_TRUE(static_cast<bool>(onPath)) << llvm::toString(onPath.takeError());
  EXPECT_FALSE(onPath->plans.empty());

  // An unmodeled level is an unsupported axis, rejected rather than ignored.
  llvm::Expected<MappingSearchResult> bogus = searchWith("dram:bogus");
  ASSERT_TRUE(static_cast<bool>(bogus)) << llvm::toString(bogus.takeError());
  EXPECT_TRUE(bogus->plans.empty());
  EXPECT_TRUE(hasDiagnostic(*bogus, DiagnosticCode::NoMatchingRule));
}

// A kind-resolved axis is honoured however the space spelled its parameter: the
// caller resolves the binding's axis by parameter *kind* (only it can see the
// search space) and passes the value explicitly, so a binding keyed by an
// arbitrary name -- here "owner" rather than "owner_mapping" -- is projected
// rather than silently ignored.
TEST(CoveringSearch, AKindResolvedOwnerAxisIsHonouredRegardlessOfItsName) {
  mlir::MLIRContext context;
  WorkloadGraph graph = twoNodeGraph(context);
  std::unique_ptr<MappingTarget> target =
      targetWith(ownerMachine(), kOwnerFreeRules);
  ASSERT_NE(target, nullptr);

  MappingSearchOptions options;
  options.mode = SearchMode::Deterministic;
  SearchBinding binding =
      makeSearchBinding("c", values({{"owner", std::string("worker")}}));

  BoundAxes axes;
  axes.ownerMapping = "worker";
  CoveringSearch search(graph, *target, context, LayoutContext{}, options,
                        binding, {}, axes);
  llvm::Expected<MappingSearchResult> result = search.search();
  ASSERT_TRUE(static_cast<bool>(result)) << llvm::toString(result.takeError());
  ASSERT_FALSE(result->plans.empty());
  for (const PlanPlacement &placement : result->plans[0].placements)
    EXPECT_EQ(placement.executor, "w0");
}

//===----------------------------------------------------------------------===//
// A binding constrains rule parameter resolution (phase-4 T2)
//===----------------------------------------------------------------------===//

TEST(CoveringSearch, ABindingDecidesWhichRuleResolvesForANode) {
  mlir::MLIRContext context;
  WorkloadGraph graph = twoNodeGraph(context);
  std::unique_ptr<MappingTarget> target =
      targetWith(searchMachine(), kParameterRules);
  ASSERT_NE(target, nullptr);

  MappingSearchOptions options;
  options.mode = SearchMode::Deterministic;

  // With no binding both rules match -- the enumeration resolves VW = 8 for
  // `r.wide` and VW = 4 for `r.narrow` -- and deterministic mode keeps the
  // first in canonical (rule-id) order, `r.narrow`.
  {
    CoveringSearch search(graph, *target, context, LayoutContext{}, options);
    llvm::Expected<MappingSearchResult> result = search.search();
    ASSERT_TRUE(static_cast<bool>(result))
        << llvm::toString(result.takeError());
    ASSERT_FALSE(result->plans.empty());
    for (const PlanPlacement &placement : result->plans[0].placements)
      EXPECT_EQ(placement.rule, "r.narrow");
  }

  // Pinned to 8, `r.narrow` (which needs VW = 4) is a non-match for every node,
  // so `r.wide` is the only rule left: the binding, not canonical order,
  // decides. This is the case that is red before the binding constrains
  // resolution -- the search would pick `r.narrow` regardless.
  {
    SearchBinding wide =
        makeSearchBinding("candidate_8", values({{"VW", int64_t{8}}}));
    CoveringSearch search(graph, *target, context, LayoutContext{}, options,
                          wide);
    llvm::Expected<MappingSearchResult> result = search.search();
    ASSERT_TRUE(static_cast<bool>(result))
        << llvm::toString(result.takeError());
    ASSERT_FALSE(result->plans.empty());
    ASSERT_EQ(result->plans[0].placements.size(), 2u);
    for (const PlanPlacement &placement : result->plans[0].placements)
      EXPECT_EQ(placement.rule, "r.wide");
    // The rule the binding pinned out of range is reported, not silently
    // dropped.
    EXPECT_GE(result->frontier.codeCounts[DiagnosticCode::NoMatchingRule], 1u);
  }

  // Pinned to 4, symmetric: `r.wide` is a non-match and `r.narrow` remains.
  {
    SearchBinding narrow =
        makeSearchBinding("candidate_4", values({{"VW", int64_t{4}}}));
    CoveringSearch search(graph, *target, context, LayoutContext{}, options,
                          narrow);
    llvm::Expected<MappingSearchResult> result = search.search();
    ASSERT_TRUE(static_cast<bool>(result))
        << llvm::toString(result.takeError());
    ASSERT_FALSE(result->plans.empty());
    for (const PlanPlacement &placement : result->plans[0].placements)
      EXPECT_EQ(placement.rule, "r.narrow");
    EXPECT_GE(result->frontier.codeCounts[DiagnosticCode::NoMatchingRule], 1u);
  }
}

TEST(CoveringSearch, ABindingThatFailsEveryRuleLeavesTheNodeWithoutARule) {
  mlir::MLIRContext context;
  WorkloadGraph graph = twoNodeGraph(context);
  std::unique_ptr<MappingTarget> target =
      targetWith(searchMachine(), kSingleParameterRule);
  ASSERT_NE(target, nullptr);

  MappingSearchOptions options;
  options.mode = SearchMode::Deterministic;

  // Pinned to the value the rule rejects, no rule applies to either node: the
  // nodes fall into the failure frontier, and the search is not truncated.
  SearchBinding binding =
      makeSearchBinding("candidate_4", values({{"VW", int64_t{4}}}));
  CoveringSearch search(graph, *target, context, LayoutContext{}, options,
                        binding);
  llvm::Expected<MappingSearchResult> result = search.search();
  ASSERT_TRUE(static_cast<bool>(result)) << llvm::toString(result.takeError());
  EXPECT_TRUE(result->plans.empty());
  EXPECT_FALSE(result->searchTruncated);
  EXPECT_EQ(result->frontier.nodesWithoutRules, 2u);
  EXPECT_TRUE(hasDiagnostic(*result, DiagnosticCode::NoMatchingRule));

  // The same fixture with the satisfying value pinned does produce plans.
  SearchBinding matching =
      makeSearchBinding("candidate_8", values({{"VW", int64_t{8}}}));
  CoveringSearch matchingSearch(graph, *target, context, LayoutContext{},
                                options, matching);
  llvm::Expected<MappingSearchResult> matchingResult = matchingSearch.search();
  ASSERT_TRUE(static_cast<bool>(matchingResult))
      << llvm::toString(matchingResult.takeError());
  EXPECT_FALSE(matchingResult->plans.empty());
}

//===----------------------------------------------------------------------===//
// A binding constrains layout selection (phase-4 T3)
//===----------------------------------------------------------------------===//

/// One rule that offers two layouts for the same port (`operand0`). With no
/// binding both requirements are materialized, exactly as before; a bound
/// layout selects the one it names and supersedes the other.
constexpr llvm::StringLiteral kOneRuleTwoLayouts = R"llkmap(
rule r.laid_out {
  match micro.vector(op = "add");
  require executor kind worker;
  require memory kind sram;
  require layout operand0 satisfies t.plain;
  require layout operand0 satisfies t.blocked;
  input "operand0";
  output "result";
  bundle "b.laid_out";
  emit "e1";
  cost 1;
}
)llkmap";

/// Two sibling rules, each offering exactly one layout. `r.a_plain` sorts
/// before `r.b_blocked`, so a binding that names `t.blocked` is observable: the
/// canonical-first rule offers only `t.plain` and is a non-match, while the
/// sibling still matches.
constexpr llvm::StringLiteral kSiblingLayoutRules = R"llkmap(
rule r.a_plain {
  match micro.vector(op = "add");
  require executor kind worker;
  require memory kind sram;
  require layout operand0 satisfies t.plain;
  input "operand0";
  output "result";
  bundle "b.plain";
  emit "e1";
  cost 1;
}
rule r.b_blocked {
  match micro.vector(op = "add");
  require executor kind worker;
  require memory kind sram;
  require layout operand0 satisfies t.blocked;
  input "operand0";
  output "result";
  bundle "b.blocked";
  emit "e1";
  cost 1;
}
)llkmap";

// (a) + (c): a rule offers two layout ids; a binding that names one selects it,
// and with no binding the choice is unchanged.
TEST(CoveringSearch, ABoundLayoutSelectsAmongTheLayoutsARuleOffers) {
  mlir::MLIRContext context;
  WorkloadGraph graph = twoNodeGraph(context);
  std::unique_ptr<MappingTarget> target =
      targetWithLayouts(transformMachine(), kOneRuleTwoLayouts, kTwoLayouts);
  ASSERT_NE(target, nullptr);

  MappingSearchOptions options;
  options.mode = SearchMode::Deterministic;

  // (c) No binding: both declared layouts are materialized, as before. The
  // plan is still found (the two layouts for one value are unattributable to
  // an edge, which only costs the transform alternative, not the plan).
  {
    CoveringSearch search(graph, *target, context, LayoutContext{}, options);
    llvm::Expected<MappingSearchResult> result = search.search();
    ASSERT_TRUE(static_cast<bool>(result))
        << llvm::toString(result.takeError());
    ASSERT_FALSE(result->plans.empty());
    ASSERT_FALSE(result->plans[0].placements.empty());
    for (const PlanPlacement &placement : result->plans[0].placements) {
      EXPECT_EQ(placement.layouts.size(), 2u);
      EXPECT_NE(placement.layouts.find("t.plain"), placement.layouts.end());
      EXPECT_NE(placement.layouts.find("t.blocked"), placement.layouts.end());
    }
  }

  // (a) The binding names `t.blocked`: that layout is selected and `t.plain`,
  // which the rule also offered, is not. Red before the fix -- both remain.
  {
    SearchBinding binding =
        makeSearchBinding("candidate_blocked",
                          values({{"tile_layout", std::string("t.blocked")}}));
    CoveringSearch search(graph, *target, context, LayoutContext{}, options,
                          binding,
                          llvm::StringMap<std::string>{{"", "t.blocked"}});
    llvm::Expected<MappingSearchResult> result = search.search();
    ASSERT_TRUE(static_cast<bool>(result))
        << llvm::toString(result.takeError());
    ASSERT_FALSE(result->plans.empty());
    ASSERT_FALSE(result->plans[0].placements.empty());
    for (const PlanPlacement &placement : result->plans[0].placements) {
      ASSERT_EQ(placement.layouts.size(), 1u);
      EXPECT_NE(placement.layouts.find("t.blocked"), placement.layouts.end());
      EXPECT_EQ(placement.layouts.find("t.plain"), placement.layouts.end());
    }
  }
}

// (b): a binding naming a layout the rule does not offer yields no candidate
// for that node (a non-match, not an error), and a sibling rule that offers it
// still matches.
TEST(CoveringSearch, ABoundLayoutARuleDoesNotOfferIsANonMatch) {
  mlir::MLIRContext context;
  WorkloadGraph graph = twoNodeGraph(context);
  std::unique_ptr<MappingTarget> target =
      targetWithLayouts(transformMachine(), kSiblingLayoutRules, kTwoLayouts);
  ASSERT_NE(target, nullptr);

  MappingSearchOptions options;
  options.mode = SearchMode::Deterministic;

  // No binding: both rules match; deterministic mode keeps the canonical
  // first, `r.a_plain`.
  {
    CoveringSearch search(graph, *target, context, LayoutContext{}, options);
    llvm::Expected<MappingSearchResult> result = search.search();
    ASSERT_TRUE(static_cast<bool>(result))
        << llvm::toString(result.takeError());
    ASSERT_FALSE(result->plans.empty());
    for (const PlanPlacement &placement : result->plans[0].placements)
      EXPECT_EQ(placement.rule, "r.a_plain");
  }

  // Bound to `t.blocked`: `r.a_plain` offers only `t.plain`, so it is a
  // non-match for every node; `r.b_blocked` remains. Red before the fix --
  // `r.a_plain` is picked regardless.
  {
    SearchBinding binding =
        makeSearchBinding("candidate_blocked",
                          values({{"tile_layout", std::string("t.blocked")}}));
    CoveringSearch search(graph, *target, context, LayoutContext{}, options,
                          binding,
                          llvm::StringMap<std::string>{{"", "t.blocked"}});
    llvm::Expected<MappingSearchResult> result = search.search();
    ASSERT_TRUE(static_cast<bool>(result))
        << llvm::toString(result.takeError());
    ASSERT_FALSE(result->plans.empty());
    ASSERT_FALSE(result->plans[0].placements.empty());
    for (const PlanPlacement &placement : result->plans[0].placements)
      EXPECT_EQ(placement.rule, "r.b_blocked");
    // The rule the binding rejected is reported, not silently dropped.
    EXPECT_GE(result->frontier.codeCounts[DiagnosticCode::NoMatchingRule], 1u);
  }
}

// Ruling S7: a bound layout is a contradiction test, not a requirement test. A
// rule that declares *no* layout requirement takes on no layout obligation, so
// it matches unchanged under any bound layout -- the layout-axis analogue of
// T2's "a name the rule does not declare is ignored". This is the shipped-AVX2
// regression in miniature: its `reduce_sum_f32`/`async_copy`/`tile_async_copy`/
// `tile_store` rules carry no `require layout`, and vetoing them would make a
// movement-and-reduce kernel unmappable the moment a candidate binds a layout.
TEST(CoveringSearch, ABoundLayoutLeavesLayoutAgnosticRulesUnchanged) {
  mlir::MLIRContext context;
  WorkloadGraph graph = twoNodeGraph(context);
  std::unique_ptr<MappingTarget> target = targetWith(searchMachine(), kRules);
  ASSERT_NE(target, nullptr);

  MappingSearchOptions options;
  options.mode = SearchMode::Deterministic;
  SearchBinding binding = makeSearchBinding(
      "candidate_blocked", values({{"tile_layout", std::string("t.blocked")}}));

  // No bound layout: the layout-agnostic rules match and plans are found.
  CoveringSearch plain(graph, *target, context, LayoutContext{}, options);
  llvm::Expected<MappingSearchResult> plainResult = plain.search();
  ASSERT_TRUE(static_cast<bool>(plainResult))
      << llvm::toString(plainResult.takeError());
  ASSERT_FALSE(plainResult->plans.empty());

  // The same rules under a bound layout still match, selecting the same rules:
  // nothing about them contradicts `t.blocked`. (The plan *id* legitimately
  // differs -- the binding's provenance is folded into it -- so compare the
  // selections, not the id.)
  CoveringSearch boundSearch(graph, *target, context, LayoutContext{}, options,
                             binding,
                             llvm::StringMap<std::string>{{"", "t.blocked"}});
  llvm::Expected<MappingSearchResult> boundResult = boundSearch.search();
  ASSERT_TRUE(static_cast<bool>(boundResult))
      << llvm::toString(boundResult.takeError());
  ASSERT_FALSE(boundResult->plans.empty());
  EXPECT_EQ(boundResult->frontier.nodesWithoutRules, 0u);
  EXPECT_FALSE(hasDiagnostic(*boundResult, DiagnosticCode::NoMatchingRule));
  ASSERT_EQ(boundResult->plans[0].placements.size(),
            plainResult->plans[0].placements.size());
  for (size_t i = 0; i < boundResult->plans[0].placements.size(); ++i)
    EXPECT_EQ(boundResult->plans[0].placements[i].rule,
              plainResult->plans[0].placements[i].rule);
}

//===----------------------------------------------------------------------===//
// Stable diagnostic codes (design §22.3)
//===----------------------------------------------------------------------===//

// Every code has a stable string, and that string maps back to the same code.
TEST(MappingDiagnostics, EveryCodeRoundTripsThroughItsString) {
  const DiagnosticCode codes[] = {
      DiagnosticCode::NoMatchingRule,
      DiagnosticCode::NoLegalLayout,
      DiagnosticCode::NoLegalExecutor,
      DiagnosticCode::MemoryCapacityExceeded,
      DiagnosticCode::UnsupportedComputeFragment,
      DiagnosticCode::NoMemoryRoute,
      DiagnosticCode::NoLayoutTransform,
      DiagnosticCode::GlobalConstraintFailed,
      DiagnosticCode::SearchTruncated,
      DiagnosticCode::LatencyCacheMiss,
      DiagnosticCode::TargetBundleInvalid,
      DiagnosticCode::AssumedValueSize,
      DiagnosticCode::InvalidMappingMetadata,
      DiagnosticCode::ConnectionChoiceUnexplored,
  };
  for (DiagnosticCode code : codes) {
    llvm::StringRef text = stringifyDiagnosticCode(code);
    EXPECT_FALSE(text.empty());
    std::optional<DiagnosticCode> back = symbolizeDiagnosticCode(text);
    ASSERT_TRUE(back.has_value()) << text.str();
    EXPECT_EQ(*back, code);
  }
  // An unknown string is not silently mapped to a code.
  EXPECT_FALSE(symbolizeDiagnosticCode("not_a_code").has_value());
}

TEST(MappingDiagnostics, NoRuleGraphCarriesNoMatchingRule) {
  mlir::MLIRContext context;
  WorkloadGraph graph = twoNodeGraph(context);
  std::unique_ptr<MappingTarget> target = targetWith(searchMachine(), "");
  ASSERT_NE(target, nullptr);

  CoveringSearch search(graph, *target, context, LayoutContext{});
  llvm::Expected<MappingSearchResult> result = search.search();
  ASSERT_TRUE(static_cast<bool>(result)) << llvm::toString(result.takeError());
  EXPECT_TRUE(result->plans.empty());
  EXPECT_TRUE(hasDiagnostic(*result, DiagnosticCode::NoMatchingRule));
}

TEST(MappingDiagnostics, PlacementFailureCarriesNoLegalExecutor) {
  mlir::MLIRContext context;
  WorkloadGraph graph = twoNodeGraph(context);
  std::unique_ptr<MappingTarget> target =
      targetWith(searchMachine(), kNoPlacementRules);
  ASSERT_NE(target, nullptr);

  CoveringSearch search(graph, *target, context, LayoutContext{});
  llvm::Expected<MappingSearchResult> result = search.search();
  ASSERT_TRUE(static_cast<bool>(result)) << llvm::toString(result.takeError());
  EXPECT_TRUE(result->plans.empty());
  EXPECT_TRUE(hasDiagnostic(*result, DiagnosticCode::NoLegalExecutor));
}

TEST(MappingDiagnostics, UnsolvableLayoutCarriesNoLegalLayout) {
  mlir::MLIRContext context;
  WorkloadGraph graph = twoNodeGraph(context);
  constexpr llvm::StringLiteral kLayoutRule = R"llkmap(
rule r.layout {
  match micro.vector(op = "add");
  require layout operand0 satisfies t.rank3;
  bundle "b.layout";
  emit "e1";
}
)llkmap";
  // The layout needs rank 3; the search's default context is rank 0, so the
  // requirement cannot solve and no placement is legal.
  constexpr llvm::StringLiteral kRank3Layout = R"llkmap(
layout t.rank3(int N) {
  param N in [1..8];
  require rank == 3;
}
)llkmap";
  std::unique_ptr<MappingTarget> target =
      targetWithLayouts(searchMachine(), kLayoutRule, kRank3Layout);
  ASSERT_NE(target, nullptr);

  CoveringSearch search(graph, *target, context, LayoutContext{});
  llvm::Expected<MappingSearchResult> result = search.search();
  ASSERT_TRUE(static_cast<bool>(result)) << llvm::toString(result.takeError());
  EXPECT_TRUE(result->plans.empty());
  EXPECT_TRUE(hasDiagnostic(*result, DiagnosticCode::NoLegalLayout));
}

TEST(MappingDiagnostics, MissingComputeCapabilityIsUnsupportedComputeFragment) {
  mlir::MLIRContext context;
  WorkloadGraph graph = twoNodeGraph(context);
  // The rule needs a matrix engine; the machine's only executor attaches no
  // compute node, so the executor matches but its fragment support does not.
  constexpr llvm::StringLiteral kComputeRule = R"llkmap(
rule r.compute {
  match micro.vector(op = "add");
  require compute kind matrix_engine;
  bundle "b.compute";
  emit "e1";
}
)llkmap";
  std::unique_ptr<MappingTarget> target =
      targetWith(searchMachine(), kComputeRule);
  ASSERT_NE(target, nullptr);

  CoveringSearch search(graph, *target, context, LayoutContext{});
  llvm::Expected<MappingSearchResult> result = search.search();
  ASSERT_TRUE(static_cast<bool>(result)) << llvm::toString(result.takeError());
  EXPECT_TRUE(result->plans.empty());
  EXPECT_TRUE(
      hasDiagnostic(*result, DiagnosticCode::UnsupportedComputeFragment));
}

TEST(MappingDiagnostics, CapacityRejectionCarriesMemoryCapacityExceeded) {
  mlir::MLIRContext context;
  WorkloadGraph graph = twoNodeGraph(context);
  std::unique_ptr<MappingTarget> target =
      targetWith(searchMachine(), kRulesWithMemory);
  ASSERT_NE(target, nullptr);

  MappingSearchOptions options;
  options.mode = SearchMode::Deterministic;
  options.memoryBudgetBytes = 1; // any bound memory overflows this
  CoveringSearch search(graph, *target, context, LayoutContext{}, options);
  llvm::Expected<MappingSearchResult> result = search.search();
  ASSERT_TRUE(static_cast<bool>(result)) << llvm::toString(result.takeError());
  EXPECT_TRUE(result->plans.empty());
  EXPECT_TRUE(hasDiagnostic(*result, DiagnosticCode::MemoryCapacityExceeded));
}

TEST(MappingDiagnostics, TruncationCarriesSearchTruncated) {
  mlir::MLIRContext context;
  WorkloadGraph graph = twoNodeGraph(context);
  std::unique_ptr<MappingTarget> target = targetWith(searchMachine(), kRules);
  ASSERT_NE(target, nullptr);

  MappingSearchOptions options;
  options.mode = SearchMode::Exact;
  options.topK = 1; // four plans exist; asking for one is a cap
  CoveringSearch search(graph, *target, context, LayoutContext{}, options);
  llvm::Expected<MappingSearchResult> result = search.search();
  ASSERT_TRUE(static_cast<bool>(result)) << llvm::toString(result.takeError());
  EXPECT_TRUE(result->searchTruncated);
  EXPECT_TRUE(hasDiagnostic(*result, DiagnosticCode::SearchTruncated));
}

TEST(MappingDiagnostics, LatencyMissCarriesLatencyCacheMiss) {
  mlir::MLIRContext context;
  WorkloadGraph graph = twoNodeGraph(context);
  FixedLatencyProvider provider; // entryless: every lookup misses

  std::unique_ptr<MappingTarget> target =
      targetWithProvider(searchMachine(), kRules, &provider);
  ASSERT_NE(target, nullptr);

  MappingSearchOptions options;
  options.mode = SearchMode::Exact;
  CoveringSearch search(graph, *target, context, LayoutContext{}, options);
  llvm::Expected<MappingSearchResult> result = search.search();
  ASSERT_TRUE(static_cast<bool>(result)) << llvm::toString(result.takeError());
  EXPECT_FALSE(provider.lookups.empty()); // it was asked, and declined
  EXPECT_TRUE(hasDiagnostic(*result, DiagnosticCode::LatencyCacheMiss));
}

// The codes are the stable interface, so a caller can group by them rather
// than parse prose. Two distinct root causes produce two distinct codes.
TEST(MappingDiagnostics, DistinctFailuresCarryDistinctCodes) {
  mlir::MLIRContext context;
  WorkloadGraph graph = twoNodeGraph(context);
  std::unique_ptr<MappingTarget> target = targetWith(searchMachine(), "");
  ASSERT_NE(target, nullptr);

  CoveringSearch search(graph, *target, context, LayoutContext{});
  llvm::Expected<MappingSearchResult> result = search.search();
  ASSERT_TRUE(static_cast<bool>(result)) << llvm::toString(result.takeError());
  EXPECT_TRUE(hasDiagnostic(*result, DiagnosticCode::NoMatchingRule));
  EXPECT_FALSE(hasDiagnostic(*result, DiagnosticCode::SearchTruncated));
  // Codes are deterministically ordered by (code, message), so two identical
  // searches produce byte-identical frontier diagnostics.
  CoveringSearch again(graph, *target, context, LayoutContext{});
  llvm::Expected<MappingSearchResult> againResult = again.search();
  ASSERT_TRUE(static_cast<bool>(againResult))
      << llvm::toString(againResult.takeError());
  ASSERT_EQ(result->frontier.diagnostics.size(),
            againResult->frontier.diagnostics.size());
  for (size_t index = 0; index < result->frontier.diagnostics.size(); ++index) {
    EXPECT_EQ(result->frontier.diagnostics[index].code,
              againResult->frontier.diagnostics[index].code);
    EXPECT_EQ(result->frontier.diagnostics[index].message,
              againResult->frontier.diagnostics[index].message);
  }
}

//===----------------------------------------------------------------------===//
// Physical (layout-image) capacity charging (task B3)
//===----------------------------------------------------------------------===//

namespace {

/// A blocked layout whose physical image rounds the padded second dimension up
/// to a whole block.
constexpr llvm::StringLiteral kPaddedLayout = R"llkmap(
layout t.block(int VW) {
  param VW in [4..4];
  map (m, n) -> (m, floordiv(n, VW), mod(n, VW));
}
)llkmap";

/// One `micro.vector` whose `result` carries `tile` in `t.block`. The rule
/// names the output port, so the solved layout is attributable to the crossing
/// value.
constexpr llvm::StringLiteral kPaddedRule = R"llkmap(
rule r.pad {
  match micro.vector();
  require executor kind worker;
  require memory kind sram;
  require layout result satisfies t.block;
  input "operand0";
  output "result";
  bundle "b.pad";
  emit "e1";
  cost 1;
}
)llkmap";

/// `searchMachine` with sram declaring `t.block` and holding `capacity` bytes.
MachineModel paddedLayoutMachine(uint64_t capacity) {
  MachineModel model = searchMachine();
  for (MemoryNode &memory : model.memories) {
    if (memory.kind == "sram") {
      memory.supportedLayouts = {"t.block"};
      memory.capacityBytes = capacity;
    }
  }
  return model;
}

/// One node `in -> out` over an 8x6xf32 tile: 192 logical bytes, but the
/// blocked map's physical image is 8*2*4 = 64 elements = 256 bytes.
WorkloadGraph raggedTileGraph(mlir::MLIRContext &context, mlir::Type tile) {
  WorkloadGraph graph;
  WorkloadValueId input =
      graph.addValue(WorkloadValue{0, tile, "in", /*external=*/true});
  WorkloadValueId output =
      graph.addValue(WorkloadValue{0, tile, "out", /*external=*/false});
  WorkloadNode node;
  node.opName = "micro.vector";
  node.attributes = vectorAttributes(context, "add");
  node.inputs.push_back(WorkloadPort{input, tile, std::nullopt});
  node.outputs.push_back(WorkloadPort{output, tile, std::nullopt});
  graph.addNode(std::move(node));
  graph.finalize();
  return graph;
}

} // namespace

TEST(CoveringSearch, ChargesThePhysicalLayoutImageNotLogicalElements) {
  mlir::MLIRContext context;
  context.getOrLoadDialect<mlir::micro::MicroDialect>();
  mlir::Type tile = mlir::parseType("!micro.tile<8x6xf32>", &context);
  ASSERT_TRUE(static_cast<bool>(tile));
  WorkloadGraph graph = raggedTileGraph(context, tile);

  // 200 bytes admits the 192 logical bytes but not the 256-byte physical image,
  // so a plan that fits the padded layout does not exist.
  std::unique_ptr<MappingTarget> tight =
      targetWithLayouts(paddedLayoutMachine(200), kPaddedRule, kPaddedLayout);
  ASSERT_NE(tight, nullptr);
  CoveringSearch tightSearch(graph, *tight, context, LayoutContext{});
  llvm::Expected<MappingSearchResult> rejected = tightSearch.search();
  ASSERT_TRUE(static_cast<bool>(rejected))
      << llvm::toString(rejected.takeError());
  EXPECT_TRUE(rejected->plans.empty());
  EXPECT_TRUE(hasDiagnostic(*rejected, DiagnosticCode::MemoryCapacityExceeded));

  // The same rule admits the plan once the memory holds the physical image.
  std::unique_ptr<MappingTarget> roomy =
      targetWithLayouts(paddedLayoutMachine(256), kPaddedRule, kPaddedLayout);
  ASSERT_NE(roomy, nullptr);
  CoveringSearch roomySearch(graph, *roomy, context, LayoutContext{});
  llvm::Expected<MappingSearchResult> admitted = roomySearch.search();
  ASSERT_TRUE(static_cast<bool>(admitted))
      << llvm::toString(admitted.takeError());
  EXPECT_FALSE(admitted->plans.empty());
}
