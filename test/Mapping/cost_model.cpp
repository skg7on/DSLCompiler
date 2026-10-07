//===- cost_model.cpp - Multi-dimensional Cost tests (issue #80 / D1) ----===//

#include "LLK/Machine/MachineModel.h"
#include "LLK/Mapping/CostEvent.h"
#include "LLK/Mapping/CostModel.h"
#include "LLK/Mapping/EventSchedule.h"
#include "LLK/Mapping/LatencyProvider.h"
#include "LLK/Mapping/MappingPlan.h"
#include "LLK/Mapping/WorkloadGraph.h"

#include "resource_regression_fixture.h"

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"

#include "llvm/Support/Error.h"

#include <gtest/gtest.h>

#include <set>
#include <string>

using namespace mlir::llk::mapping;

namespace {
Cost cycles(double value) {
  Cost cost;
  cost.latencyCycles = value;
  return cost;
}

/// One vector engine (8 f32 lanes, one cycle per issue) and one SRAM node, so a
/// transform's cost can be read off a capability rather than a constant.
mlir::llk::machine::MachineModel transformMachine() {
  mlir::llk::machine::MachineModel machine;
  machine.target = "transform-machine";
  machine.workerThreads = 1;
  machine.sync.barrierCycles = 1;
  machine.sync.waitCycles = 1;

  mlir::llk::machine::ComputeNode vpu;
  vpu.id = "vpu";
  vpu.kind = "vector_engine";
  vpu.attachedTo = "worker.0";
  machine.executors = {{"worker.0", "worker", std::nullopt, {}, 1, {}}};
  vpu.lanes["f32"] = 8;
  vpu.issueCycles = 1;
  machine.computes.push_back(vpu);

  mlir::llk::machine::MemoryNode sram;
  sram.id = "sram.0";
  sram.kind = "sram";
  sram.visibleFrom = "worker.0";
  machine.memories.push_back(sram);
  return machine;
}

/// The `(d0,d1) -> (d1,d0)` index exchange of a transpose, spelled explicitly
/// so the unsigned/int64_t `getPermutationMap` overloads are not ambiguous.
mlir::AffineMap transposeMap(mlir::MLIRContext *context) {
  return mlir::AffineMap::getPermutationMap(llvm::ArrayRef<unsigned>{1u, 0u},
                                            context);
}

/// A machine with one worker/vector engine, one DMA engine, and a
/// `dram.0 -> l2.0 -> sram.0` hierarchy, so a two-hop movement and a layout
/// conversion can both be normalized.
mlir::llk::machine::MachineModel planEventMachine() {
  using namespace mlir::llk::machine;
  MachineModel machine;
  machine.target = "plan-events";
  machine.workerThreads = 1;
  machine.sync.waitCycles = 1;
  machine.sync.barrierCycles = 3;

  ExecutorNode worker;
  worker.id = "w0";
  worker.kind = "worker";
  machine.executors.push_back(worker);

  ComputeNode vpu;
  vpu.id = "vpu";
  vpu.kind = "vector_engine";
  vpu.attachedTo = "w0";
  vpu.lanes["f32"] = 8;
  vpu.issueCycles = 1;
  machine.computes.push_back(vpu);

  for (const auto &[id, kind] :
       {std::pair<const char *, const char *>{"d0", "dram"},
        {"l0", "l2"},
        {"s0", "sram"}}) {
    MemoryNode memory;
    memory.id = id;
    memory.kind = kind;
    // Every memory is addressable from the worker, so the transform cost's
    // resource/memory visibility check accepts the transform this machine
    // models; without it the estimator fails a footprintless-in-memory
    // conversion the plan legitimately selected.
    memory.visibleFrom = "w0";
    machine.memories.push_back(memory);
  }

  TransferEngineNode dma;
  dma.id = "dma.0";
  dma.kind = "dma";
  machine.transferEngines.push_back(dma);

  LinkEdge first;
  first.id = "d0_l0";
  first.source = "d0";
  first.destination = "l0";
  first.latencyCycles = 10;
  first.bandwidthBytesPerCycle = 8;
  first.transferEngines = {"dma.0"};
  machine.links.push_back(first);

  LinkEdge second = first;
  second.id = "l0_s0";
  second.source = "l0";
  second.destination = "s0";
  second.latencyCycles = 20;
  machine.links.push_back(second);
  return machine;
}

/// A two-node plan: node 0 feeds node 1 through a two-hop transfer connection.
CoveringPlan twoHopPlan() {
  CoveringPlan plan;
  PlanPlacement producer;
  producer.node = 0;
  producer.instance = 100;
  producer.executor = "w0";
  producer.cost.latencyCycles = 8;
  producer.workItems = 16;
  plan.placements.push_back(producer);

  PlanPlacement consumer;
  consumer.node = 1;
  consumer.instance = 101;
  consumer.executor = "w0";
  consumer.cost.latencyCycles = 4;
  consumer.workItems = 16;
  plan.placements.push_back(consumer);

  PlanConnection connection;
  connection.id = 500;
  connection.value = 7;
  connection.kind = ConnectionKind::Transfer;
  connection.route = {"d0", "l0", "s0"};
  connection.engines = {"dma.0"};
  connection.cost.localBytes = 64;
  connection.workItems = 16;
  connection.consumers = {101};
  connection.producerPort = PortRef{0, PortDirection::Output, 0};
  plan.connectionPlans.push_back(connection);

  plan.steps = {PlanStep{0, PlanStepKind::Compute, 0, 0},
                PlanStep{1, PlanStepKind::Movement, 0, 500},
                PlanStep{2, PlanStepKind::Synchronization, 0, 500},
                PlanStep{3, PlanStepKind::Compute, 1, 0}};
  plan.stepEdges = {PlanStepEdge{0, 1}, PlanStepEdge{1, 2}, PlanStepEdge{2, 3}};
  return plan;
}

/// A provider that implements only the operation lookup, so the default
/// nullopt connection overload is the one exercised.
class OperationOnlyProvider : public LatencyProvider {
public:
  // A derived class that overrides one `lookupCycles` overload hides the other,
  // so a provider written before connections existed must bring the default
  // connection overload back into scope to keep it reachable.
  using LatencyProvider::lookupCycles;
  std::optional<double> lookupCycles(const OperationSignature &,
                                     const TargetContext &) const override {
    return std::nullopt;
  }
};
} // namespace

TEST(CostModel, AddsDimensionwise) {
  Cost a = cycles(10.0);
  a.dramBytes = 100;
  a.localBytes = 4;
  Cost b = cycles(2.5);
  b.dramBytes = 50;
  b.localBytes = 1;

  Cost sum = addCost(a, b);
  EXPECT_DOUBLE_EQ(sum.latencyCycles, 12.5);
  EXPECT_EQ(sum.dramBytes, 150u);
  EXPECT_EQ(sum.localBytes, 5u);
}

// Utilization is an aggregate load factor, not a per-component fraction: both
// dimensions share one machine-global denominator, so `addCost` sums them and
// the total may exceed 1 to signal oversubscription. Two components each using
// half a window therefore add to one full window.
TEST(CostModel, UtilizationAddsAsAnAggregateLoadFactor) {
  Cost half;
  half.computeUtilization = 0.5;
  half.transferUtilization = 0.5;
  Cost other = half;

  Cost sum = addCost(half, other);
  EXPECT_DOUBLE_EQ(sum.computeUtilization, 1.0);
  EXPECT_DOUBLE_EQ(sum.transferUtilization, 1.0);

  // The sum is not clamped to 1: an overloaded plan reads above 1.
  Cost overloaded = addCost(sum, half);
  EXPECT_DOUBLE_EQ(overloaded.computeUtilization, 1.5);
  EXPECT_DOUBLE_EQ(overloaded.transferUtilization, 1.5);
}

TEST(CostModel, MinimizeOrdersByPrimaryThenSecondary) {
  ObjectiveOrder order{
      CostMetric::LatencyCycles, {CostMetric::DramBytes}, true};

  Cost fastManyDram = cycles(10.0);
  fastManyDram.dramBytes = 999;
  Cost slowNoDram = cycles(11.0);
  slowNoDram.dramBytes = 0;
  // Primary latency wins regardless of the secondary byte count.
  EXPECT_TRUE(costLess(fastManyDram, slowNoDram, order));
  EXPECT_FALSE(costLess(slowNoDram, fastManyDram, order));

  Cost fastFew = cycles(10.0);
  fastFew.dramBytes = 1;
  Cost fastMany = cycles(10.0);
  fastMany.dramBytes = 2;
  // Equal primary: the declared secondary decides.
  EXPECT_TRUE(costLess(fastFew, fastMany, order));
  EXPECT_FALSE(costLess(fastMany, fastFew, order));
}

TEST(CostModel, ExactTieIsNotLessInEitherDirection) {
  ObjectiveOrder order{CostMetric::LatencyCycles, {}, true};
  Cost a = cycles(10.0);
  Cost b = cycles(10.0);
  EXPECT_FALSE(costLess(a, b, order));
  EXPECT_FALSE(costLess(b, a, order));
}

TEST(CostModel, MaximizeInvertsTheComparison) {
  ObjectiveOrder order{CostMetric::ComputeUtilization, {}, false};
  Cost low;
  low.computeUtilization = 0.5;
  Cost high;
  high.computeUtilization = 0.9;
  // Maximizing utilization: the higher value ranks ahead.
  EXPECT_TRUE(costLess(high, low, order));
  EXPECT_FALSE(costLess(low, high, order));
}

TEST(CostModel, CanonicalStringIsStableAndDistinct) {
  Cost a = cycles(1.0);
  Cost b = cycles(1.0);
  b.dramBytes = 1;
  EXPECT_EQ(canonicalCostString(a), canonicalCostString(cycles(1.0)));
  EXPECT_NE(canonicalCostString(a), canonicalCostString(b));
}

// The bridge that carries a `micro.objective`'s declared metric and direction
// into the mapping's comparison order. Every spelling the Micro verifier
// accepts must resolve (design §17.1).
TEST(CostModel, ObjectiveBridgeResolvesEveryDialectMetricSpelling) {
  struct Case {
    const char *spelling;
    CostMetric metric;
  };
  const Case cases[] = {
      {"latency_cycles", CostMetric::LatencyCycles},
      {"dram_bytes", CostMetric::DramBytes},
      {"sram_bytes", CostMetric::LocalBytes},
      {"matrix_utilization", CostMetric::ComputeUtilization},
      {"dma_utilization", CostMetric::TransferUtilization},
      {"capacity_spill_bytes", CostMetric::SpillBytes},
  };
  for (const Case &entry : cases) {
    llvm::Expected<ObjectiveOrder> order =
        objectiveOrderFromMicro(entry.spelling, /*minimize=*/true);
    ASSERT_TRUE(static_cast<bool>(order)) << entry.spelling;
    EXPECT_EQ(order->primary, entry.metric) << entry.spelling;
    EXPECT_TRUE(order->minimize);
    EXPECT_TRUE(order->secondary.empty());
  }

  llvm::Expected<ObjectiveOrder> maximize =
      objectiveOrderFromMicro("dram_bytes", /*minimize=*/false);
  ASSERT_TRUE(static_cast<bool>(maximize));
  EXPECT_EQ(maximize->primary, CostMetric::DramBytes);
  EXPECT_FALSE(maximize->minimize);
}

// Secondary metrics become tie-breakers, kept in the declared order (§17.1).
TEST(CostModel, ObjectiveBridgeKeepsSecondaryMetricsInDeclaredOrder) {
  llvm::StringRef secondary[] = {"matrix_utilization", "dram_bytes"};
  llvm::Expected<ObjectiveOrder> order =
      objectiveOrderFromMicro("latency_cycles", /*minimize=*/true, secondary);
  ASSERT_TRUE(static_cast<bool>(order));
  EXPECT_EQ(order->primary, CostMetric::LatencyCycles);
  ASSERT_EQ(order->secondary.size(), 2u);
  EXPECT_EQ(order->secondary[0], CostMetric::ComputeUtilization);
  EXPECT_EQ(order->secondary[1], CostMetric::DramBytes);
}

// A declared objective the cost model cannot honor is rejected, never silently
// replaced by the default.
TEST(CostModel, ObjectiveBridgeRejectsAnUnknownMetric) {
  llvm::Expected<ObjectiveOrder> unknown =
      objectiveOrderFromMicro("not_a_metric", /*minimize=*/true);
  EXPECT_FALSE(static_cast<bool>(unknown));
  if (!unknown)
    llvm::consumeError(unknown.takeError());

  llvm::StringRef badSecondary[] = {"not_a_metric"};
  llvm::Expected<ObjectiveOrder> unknownSecondary = objectiveOrderFromMicro(
      "latency_cycles", /*minimize=*/true, badSecondary);
  EXPECT_FALSE(static_cast<bool>(unknownSecondary));
  if (!unknownSecondary)
    llvm::consumeError(unknownSecondary.takeError());
}

TEST(CostEvent, EveryKindRoundTripsThroughItsName) {
  for (CostEventKind kind :
       {CostEventKind::Compute, CostEventKind::TransferHop,
        CostEventKind::Transform, CostEventKind::Synchronization,
        CostEventKind::Capacity}) {
    EXPECT_EQ(symbolizeCostEventKind(stringifyCostEventKind(kind)), kind);
  }
  EXPECT_FALSE(symbolizeCostEventKind("not_a_kind").has_value());
}

// The shared transform estimate: a nonidentity conversion is charged what the
// selected capability takes to issue it, read off the machine -- not a
// per-layer constant. 64 f32 elements over 8 lanes is 8 one-cycle issues.
TEST(CostModel, TransformCostScalesWithTheSelectedCapability) {
  mlir::MLIRContext context;
  auto type =
      mlir::RankedTensorType::get({8, 8}, mlir::Float32Type::get(&context));

  TransformCostInput input;
  input.inputType = type;
  input.outputType = type;
  input.srcMap = mlir::AffineMap::getMultiDimIdentityMap(2, &context);
  input.dstMap = transposeMap(&context);
  input.memoryNode = "sram.0";
  input.computeResource = "vpu";

  llvm::Expected<Cost> cost = estimateTransformCost(input, transformMachine());
  ASSERT_TRUE(static_cast<bool>(cost)) << llvm::toString(cost.takeError());
  EXPECT_EQ(cost->latencyCycles, 8.0);
  EXPECT_EQ(cost->localBytes, 256u);
}

// An identity source/destination map pair is an explicit re-representation:
// modeled as zero arithmetic while still materializing the output buffer.
TEST(CostModel, IdentityTransformCostsNoArithmetic) {
  mlir::MLIRContext context;
  auto type =
      mlir::RankedTensorType::get({8, 8}, mlir::Float32Type::get(&context));
  mlir::AffineMap identity =
      mlir::AffineMap::getMultiDimIdentityMap(2, &context);

  TransformCostInput input;
  input.inputType = type;
  input.outputType = type;
  input.srcMap = identity;
  input.dstMap = identity;
  input.computeResource = "vpu";

  llvm::Expected<Cost> cost = estimateTransformCost(input, transformMachine());
  ASSERT_TRUE(static_cast<bool>(cost)) << llvm::toString(cost.takeError());
  EXPECT_EQ(cost->latencyCycles, 0.0);
  EXPECT_EQ(cost->localBytes, 256u);
}

// An unknown resource or footprint is reported, never silently costed as zero.
TEST(CostModel, TransformCostRejectsUnknownResourcesAndFootprints) {
  mlir::MLIRContext context;
  auto type =
      mlir::RankedTensorType::get({8, 8}, mlir::Float32Type::get(&context));

  TransformCostInput input;
  input.inputType = type;
  input.outputType = type;
  input.srcMap = mlir::AffineMap::getMultiDimIdentityMap(2, &context);
  input.dstMap = transposeMap(&context);
  input.computeResource = "missing.vpu";

  llvm::Expected<Cost> unknownCompute =
      estimateTransformCost(input, transformMachine());
  EXPECT_FALSE(static_cast<bool>(unknownCompute));
  if (!unknownCompute)
    llvm::consumeError(unknownCompute.takeError());

  input.computeResource = "vpu";
  input.memoryNode = "missing.memory";
  llvm::Expected<Cost> unknownMemory =
      estimateTransformCost(input, transformMachine());
  EXPECT_FALSE(static_cast<bool>(unknownMemory));
  if (!unknownMemory)
    llvm::consumeError(unknownMemory.takeError());

  TransformCostInput noFootprint;
  llvm::Expected<Cost> unknownFootprint =
      estimateTransformCost(noFootprint, transformMachine());
  EXPECT_FALSE(static_cast<bool>(unknownFootprint));
  if (!unknownFootprint)
    llvm::consumeError(unknownFootprint.takeError());
}

//===----------------------------------------------------------------------===//
// Task B8: normalized plan events
//===----------------------------------------------------------------------===//

TEST(CostEvent, PlanEventsNormalizeSelectedWork) {
  CoveringPlan plan = twoHopPlan();
  llvm::Expected<PlanEventDAG> dag = buildPlanEvents(plan, planEventMachine());
  ASSERT_TRUE(static_cast<bool>(dag)) << llvm::toString(dag.takeError());
  ASSERT_EQ(dag->events.size(), 6u);

  const CostEventKind kinds[] = {
      CostEventKind::Compute,         CostEventKind::TransferHop,
      CostEventKind::Synchronization, CostEventKind::TransferHop,
      CostEventKind::Synchronization, CostEventKind::Compute};
  const char *resources[] = {"vpu", "dma.0", "sync", "dma.0", "sync", "vpu"};
  const uint64_t bytes[] = {0, 64, 0, 64, 0, 0};
  const uint64_t work[] = {16, 16, 0, 16, 0, 16};
  // Hop 1: 10 + ceil(64/8); hop 2: 20 + ceil(64/8); waits: one cycle each.
  const double cycles[] = {8, 18, 1, 28, 1, 4};
  for (size_t i = 0; i < dag->events.size(); ++i) {
    EXPECT_EQ(dag->events[i].event.kind, kinds[i]) << i;
    EXPECT_EQ(dag->events[i].event.resource, resources[i]) << i;
    EXPECT_EQ(dag->events[i].bytes, bytes[i]) << i;
    EXPECT_EQ(dag->events[i].workItems, work[i]) << i;
    EXPECT_DOUBLE_EQ(dag->events[i].event.cost.latencyCycles, cycles[i]) << i;
  }
  // First hop follows the producer's compute; the consumer follows the wait.
  EXPECT_EQ(dag->events[1].deps, (std::vector<uint32_t>{0}));
  EXPECT_EQ(dag->events[5].deps, (std::vector<uint32_t>{4}));
}

TEST(CostEvent, LayoutTransformEventUsesTheSharedEstimate) {
  mlir::MLIRContext context;
  CoveringPlan plan = twoHopPlan();
  PlanConnection &connection = plan.connectionPlans.front();
  connection.kind = ConnectionKind::LayoutTransform;
  connection.route = {"s0"};
  connection.valueType =
      mlir::RankedTensorType::get({4, 4}, mlir::Float32Type::get(&context));
  LayoutTransform transform;
  transform.srcMap = mlir::AffineMap::getMultiDimIdentityMap(2, &context);
  transform.dstMap = transposeMap(&context);
  connection.transform = transform;
  // The producer also runs on the same executor, so the transform resolves to
  // the shared estimate's default vector engine.
  plan.placements[0].executor = "w0";

  llvm::Expected<PlanEventDAG> dag = buildPlanEvents(plan, planEventMachine());
  ASSERT_TRUE(static_cast<bool>(dag)) << llvm::toString(dag.takeError());
  ASSERT_EQ(dag->events.size(), 3u);
  const PlanCostEvent &transformEvent = dag->events[1];
  EXPECT_EQ(transformEvent.event.kind, CostEventKind::Transform);
  EXPECT_EQ(transformEvent.event.resource, "vpu");
  EXPECT_EQ(transformEvent.bytes, 64u);
  // 16 elements over 8 f32 lanes: 2 one-cycle issues.
  EXPECT_DOUBLE_EQ(transformEvent.event.cost.latencyCycles, 2.0);
}

TEST(CostEvent, PlanEventsNormalizeAGather) {
  mlir::MLIRContext context;
  CoveringPlan plan;
  PlanPlacement consumer;
  consumer.node = 1;
  consumer.instance = 101;
  consumer.executor = "w0";
  consumer.cost.latencyCycles = 1;
  consumer.workItems = 8;
  plan.placements.push_back(consumer);

  PlanConnection gather;
  gather.id = 600;
  gather.value = 9;
  gather.kind = ConnectionKind::Reduce;
  gather.gatherSemantics = GatherSemantics::Sum;
  gather.workItems = 8;
  gather.valueType =
      mlir::RankedTensorType::get({8}, mlir::Float32Type::get(&context));
  gather.producerPorts = {PortRef{0, PortDirection::Output, 0},
                          PortRef{1, PortDirection::Output, 0}};
  plan.connectionPlans.push_back(gather);
  plan.steps = {PlanStep{0, PlanStepKind::Movement, 0, 600},
                PlanStep{1, PlanStepKind::Synchronization, 0, 600},
                PlanStep{2, PlanStepKind::Compute, 1, 0}};

  llvm::Expected<PlanEventDAG> dag = buildPlanEvents(plan, planEventMachine());
  ASSERT_TRUE(static_cast<bool>(dag)) << llvm::toString(dag.takeError());
  // The gather is compute on the selected vector engine; 8 f32 elements over 8
  // lanes is one issue.
  ASSERT_FALSE(dag->events.empty());
  EXPECT_EQ(dag->events[0].event.kind, CostEventKind::Compute);
  EXPECT_EQ(dag->events[0].event.resource, "vpu");
  EXPECT_EQ(dag->events[0].workItems, 8u);
  EXPECT_DOUBLE_EQ(dag->events[0].event.cost.latencyCycles, 1.0);
}

// A plan scored before storage finalization gets a synthesized step DAG. Its
// dependency edges must be real: the movement follows the producer's compute
// and the consumer follows the movement -- a step-id collision once made both
// edges vanish, so every event started at t=0.
TEST(CostEvent, SynthesizedStepsKeepTheirDependencyEdges) {
  CoveringPlan plan = twoHopPlan();
  plan.steps.clear();
  plan.stepEdges.clear();

  llvm::Expected<PlanEventDAG> dag = buildPlanEvents(plan, planEventMachine());
  ASSERT_TRUE(static_cast<bool>(dag)) << llvm::toString(dag.takeError());
  ASSERT_EQ(dag->events.size(), 6u);

  // The first transfer hop (event 2) follows the producer's compute (event 0).
  ASSERT_EQ(dag->events[2].event.kind, CostEventKind::TransferHop);
  EXPECT_EQ(dag->events[2].deps, (std::vector<uint32_t>{0}));
  // The consumer's compute (event 1) follows the movement's last wait (5).
  ASSERT_EQ(dag->events[1].event.kind, CostEventKind::Compute);
  EXPECT_EQ(dag->events[1].deps, (std::vector<uint32_t>{5}));

  // Scheduling confirms it: no event starts before an event it depends on has
  // finished, so the movement is not charged as if it began at zero.
  EventScheduleResult schedule =
      scheduleNormalizedEvents(dag->events, planEventMachine());
  for (const ScheduledEvent &event : schedule.entries)
    for (uint32_t dep : dag->events[event.id].deps)
      EXPECT_LE(schedule.entries[dep].finish, event.start) << event.id;
  // The consumer cannot start before the producer's compute and both hops.
  EXPECT_GE(schedule.entries[1].start, schedule.entries[5].finish);
  EXPECT_GT(schedule.entries[1].start, 0u);
}

TEST(CostEvent, PlanEventsRejectAnUnknownStrictFact) {
  // A hop across a link the machine does not declare cannot be charged and is
  // rejected rather than silently charged a single iteration.
  CoveringPlan missingRoute = twoHopPlan();
  missingRoute.connectionPlans.front().route = {"d0", "s0"};
  llvm::Expected<PlanEventDAG> unrouted =
      buildPlanEvents(missingRoute, planEventMachine());
  EXPECT_FALSE(static_cast<bool>(unrouted));
  if (!unrouted)
    llvm::consumeError(unrouted.takeError());

  // A transform with no maps has no measurable footprint.
  CoveringPlan noTransform = twoHopPlan();
  noTransform.connectionPlans.front().kind = ConnectionKind::LayoutTransform;
  noTransform.connectionPlans.front().route = {"s0"};
  llvm::Expected<PlanEventDAG> unmapped =
      buildPlanEvents(noTransform, planEventMachine());
  EXPECT_FALSE(static_cast<bool>(unmapped));
  if (!unmapped)
    llvm::consumeError(unmapped.takeError());

  // A gather with no declared semantics cannot be materialized.
  CoveringPlan noGather = twoHopPlan();
  PlanConnection &gather = noGather.connectionPlans.front();
  gather.kind = ConnectionKind::Reduce;
  gather.route = {"s0"};
  gather.producerPorts = {PortRef{0, PortDirection::Output, 0},
                          PortRef{1, PortDirection::Output, 0}};
  llvm::Expected<PlanEventDAG> undecided =
      buildPlanEvents(noGather, planEventMachine());
  EXPECT_FALSE(static_cast<bool>(undecided));
  if (!undecided)
    llvm::consumeError(undecided.takeError());
}

//===----------------------------------------------------------------------===//
// Task B8: connection measurement identity
//===----------------------------------------------------------------------===//

TEST(CostEvent, ConnectionSignatureDistinguishesEveryDecision) {
  ConnectionSignature base;
  base.kind = "transfer";
  base.valueType = "tensor<4x4xf32>";
  base.producerEndpoint = "producer{0.0}";
  base.consumerEndpoints = "consumer{1.0}";
  base.route = "d0>l0>s0";
  base.links = "d0_l0>l0_s0";
  base.engines = "dma.0";
  base.maps = "producer=(d0,d1)->(d0,d1)";
  base.parameters = "row_major->blocked";
  base.storage = "s0#64";
  const std::string reference = base.canonicalString();

  struct Mutation {
    const char *name;
    std::string ConnectionSignature::*field;
  };
  const Mutation mutations[] = {
      {"kind", &ConnectionSignature::kind},
      {"valueType", &ConnectionSignature::valueType},
      {"producerEndpoint", &ConnectionSignature::producerEndpoint},
      {"consumerEndpoints", &ConnectionSignature::consumerEndpoints},
      {"route", &ConnectionSignature::route},
      {"links", &ConnectionSignature::links},
      {"engines", &ConnectionSignature::engines},
      {"maps", &ConnectionSignature::maps},
      {"parameters", &ConnectionSignature::parameters},
      {"storage", &ConnectionSignature::storage},
  };
  for (const Mutation &mutation : mutations) {
    ConnectionSignature changed = base;
    changed.*(mutation.field) += "#changed";
    EXPECT_NE(changed.canonicalString(), reference) << mutation.name;
  }

  // Swapping the two roles must not collide: the endpoint fields carry an
  // explicit role, so a moved producer/consumer assignment is different work.
  ConnectionSignature swapped = base;
  swapped.producerEndpoint = base.consumerEndpoints;
  swapped.consumerEndpoints = base.producerEndpoint;
  EXPECT_NE(swapped.canonicalString(), reference);
}

TEST(CostEvent, ConnectionSignatureFoldsConsumerMapsAndGatherSemantics) {
  mlir::MLIRContext context;
  WorkloadGraph workload;
  mlir::llk::machine::MachineModel machine = planEventMachine();

  mlir::AffineMap identity =
      mlir::AffineMap::getMultiDimIdentityMap(2, &context);
  mlir::AffineMap transpose = transposeMap(&context);

  ConnectionPlan base;
  base.id = 1;
  base.value = 0;
  base.kind = ConnectionKind::Reduce;
  base.gatherSemantics = GatherSemantics::Sum;
  base.memoryRoute = {"s0"};
  base.producerMap = identity;
  base.consumerMaps = {identity, transpose};
  const std::string reference =
      connectionSignatureFor(base, workload, machine).canonicalString();

  // Sum vs Max vs Concatenate are different work.
  ConnectionPlan max = base;
  max.gatherSemantics = GatherSemantics::Max;
  EXPECT_NE(connectionSignatureFor(max, workload, machine).canonicalString(),
            reference);

  ConnectionPlan concat0 = base;
  concat0.gatherSemantics = GatherSemantics::Concatenate;
  concat0.concatAxis = 0;
  ConnectionPlan concat1 = concat0;
  concat1.concatAxis = 1;
  const std::string axisZero =
      connectionSignatureFor(concat0, workload, machine).canonicalString();
  EXPECT_NE(axisZero, reference);
  EXPECT_NE(
      connectionSignatureFor(concat1, workload, machine).canonicalString(),
      axisZero);

  // A changed consumer-side affine relation changes the identity.
  ConnectionPlan differentConsumerMaps = base;
  differentConsumerMaps.consumerMaps = {transpose, transpose};
  EXPECT_NE(connectionSignatureFor(differentConsumerMaps, workload, machine)
                .canonicalString(),
            reference);

  // The consumer maps are rendered in a canonical (sorted) order, so a
  // reordered description retains one key.
  ConnectionPlan reordered = base;
  reordered.consumerMaps = {transpose, identity};
  EXPECT_EQ(
      connectionSignatureFor(reordered, workload, machine).canonicalString(),
      reference);
}

TEST(CostEvent, ConnectionSignatureIsNotAmbiguousConcatenation) {
  // Two different connection descriptions whose fields, naively concatenated,
  // would produce the same bytes. The length-delimited rendering must keep them
  // apart.
  ConnectionSignature first;
  first.kind = "transfer|value_type=x";
  first.valueType = "y";
  ConnectionSignature second;
  second.kind = "transfer";
  second.valueType = "x|value_type=y";
  EXPECT_NE(first.canonicalString(), second.canonicalString());

  // A provider that implements only the operation lookup misses a connection
  // through the default overload -- a miss is not a legality verdict.
  OperationOnlyProvider provider;
  TargetContext context{"target", "machine", "rules", "layouts"};
  EXPECT_FALSE(provider.lookupCycles(first, context).has_value());
}

//===----------------------------------------------------------------------===//
// Issue #129, task R1: normalized events use the recorded compute selection
//===----------------------------------------------------------------------===//

/// The first-engine probe. Two attached vector engines produce two complete
/// plans; each plan's compute event must name the engine *that plan selected*,
/// not the executor's first attached capability. Before the repair both plans
/// normalized their event resource as `vpu.a`.
TEST(CostModel, Issue129PlanEventsNameTheRecordedComputeSelection) {
  auto c = issue129::resourceCase("two-compute");
  ASSERT_TRUE(bool(c)) << llvm::toString(c.takeError());
  MappingSearchOptions options;
  options.mode = SearchMode::Exact;
  options.topK = 8;
  auto result = issue129::searchCase(*c, options);
  ASSERT_TRUE(bool(result)) << llvm::toString(result.takeError());
  ASSERT_EQ(result->plans.size(), 2u);

  std::set<std::string> resources;
  for (const CoveringPlan &plan : result->plans) {
    llvm::Expected<PlanEventDAG> dag =
        buildPlanEvents(plan, c->target->machine());
    ASSERT_TRUE(bool(dag)) << llvm::toString(dag.takeError());
    ASSERT_FALSE(dag->events.empty());
    EXPECT_EQ(dag->events.front().event.kind, CostEventKind::Compute);
    resources.insert(dag->events.front().event.resource);
  }
  EXPECT_EQ(resources, (std::set<std::string>{"vpu.a", "vpu.b"}));
}

/// A recorded selection that no longer resolves is an error, never silently
/// replaced by the executor's first engine. The same plan with a tampered
/// engine id must not normalize to a *different* engine.
TEST(CostModel, Issue129AnUnknownRecordedComputeNodeIsRejected) {
  auto c = issue129::resourceCase("two-compute");
  ASSERT_TRUE(bool(c)) << llvm::toString(c.takeError());
  MappingSearchOptions options;
  options.mode = SearchMode::Exact;
  options.topK = 8;
  auto result = issue129::searchCase(*c, options);
  ASSERT_TRUE(bool(result)) << llvm::toString(result.takeError());
  ASSERT_FALSE(result->plans.empty());

  CoveringPlan plan = result->plans.front();
  ASSERT_FALSE(plan.placements.empty());
  plan.placements.front().computeBindings["vector_engine"] = "vpu.unknown";
  llvm::Expected<PlanEventDAG> dag =
      buildPlanEvents(plan, c->target->machine());
  ASSERT_FALSE(bool(dag));
  EXPECT_NE(llvm::toString(dag.takeError()).find("vpu.unknown"),
            std::string::npos);
}

/// Two placements that differ only in the selected engine are different work,
/// so a measured cost must not be reused across them: the operation signature's
/// compute field is what separates the keys.
TEST(CostModel, Issue129OperationSignatureSeparatesSelectedEngines) {
  OperationSignature a;
  a.operation = "micro.vector";
  a.rule = "issue129.vector_add";
  a.compute = "vector_engine=vpu.a";
  OperationSignature b = a;
  b.compute = "vector_engine=vpu.b";
  EXPECT_NE(a.canonicalString(), b.canonicalString());
  OperationSignature same = a;
  EXPECT_EQ(a.canonicalString(), same.canonicalString());
}
