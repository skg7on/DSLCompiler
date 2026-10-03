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

TEST(CostEvent, EveryKindRoundTripsThroughItsName) {
  for (CostEventKind kind :
       {CostEventKind::Compute, CostEventKind::TransferHop,
        CostEventKind::Transform, CostEventKind::Synchronization,
        CostEventKind::Capacity}) {
    EXPECT_EQ(symbolizeCostEventKind(stringifyCostEventKind(kind)), kind);
  }
  EXPECT_FALSE(symbolizeCostEventKind("not_a_kind").has_value());
}
