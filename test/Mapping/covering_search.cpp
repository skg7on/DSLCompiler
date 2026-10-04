//===- covering_search.cpp - Complete-plan search (D6) -------------------===//

#include "LLK/Mapping/CoveringSearch.h"
#include "LLK/Mapping/Diagnostics.h"
#include "LLK/Mapping/LatencyProvider.h"

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

/// `searchMachine` with the sram node shrunk to 5000 bytes. The search charges
/// 4096 bytes per bound instance (kAssumedValueBytes), so one instance fits the
/// node while two do not -- and two still fit the default global byte budget,
/// leaving the per-memory check as the only thing that can reject the pair.
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
constexpr PlanId kNoBindingPlanId = 13971994565763734923ULL;

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
  EXPECT_FALSE(provider.lookups.empty());
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
