//===- cost_model.cpp - Multi-dimensional Cost tests (issue #80 / D1) ----===//

#include "LLK/Mapping/CostEvent.h"
#include "LLK/Mapping/CostModel.h"

#include <gtest/gtest.h>

using namespace mlir::llk::mapping;

namespace {
Cost cycles(double value) {
  Cost cost;
  cost.latencyCycles = value;
  return cost;
}
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
