//===- CostModel.cpp - Multi-dimensional mapping cost ---------------------===//

#include "LLK/Mapping/CostModel.h"

#include <array>
#include <cstdio>

namespace mlir::llk::mapping {

namespace {
struct MetricInfo {
  CostMetric metric;
  llvm::StringRef name;
};

/// Declaration order is the canonical order used by `canonicalCostString`.
constexpr std::array<MetricInfo, 6> kMetrics{{
    {CostMetric::LatencyCycles, "latency_cycles"},
    {CostMetric::DramBytes, "dram_bytes"},
    {CostMetric::LocalBytes, "local_bytes"},
    {CostMetric::SpillBytes, "spill_bytes"},
    {CostMetric::ComputeUtilization, "compute_utilization"},
    {CostMetric::TransferUtilization, "transfer_utilization"},
}};
} // namespace

llvm::StringRef stringifyCostMetric(CostMetric metric) {
  for (const MetricInfo &info : kMetrics)
    if (info.metric == metric)
      return info.name;
  return "";
}

std::optional<CostMetric> symbolizeCostMetric(llvm::StringRef text) {
  for (const MetricInfo &info : kMetrics)
    if (info.name == text)
      return info.metric;
  return std::nullopt;
}

double costMetric(const Cost &cost, CostMetric metric) {
  switch (metric) {
  case CostMetric::LatencyCycles:
    return cost.latencyCycles;
  case CostMetric::DramBytes:
    return static_cast<double>(cost.dramBytes);
  case CostMetric::LocalBytes:
    return static_cast<double>(cost.localBytes);
  case CostMetric::SpillBytes:
    return static_cast<double>(cost.spillBytes);
  case CostMetric::ComputeUtilization:
    return cost.computeUtilization;
  case CostMetric::TransferUtilization:
    return cost.transferUtilization;
  }
  return 0.0;
}

Cost addCost(const Cost &lhs, const Cost &rhs) {
  Cost sum;
  sum.latencyCycles = lhs.latencyCycles + rhs.latencyCycles;
  sum.dramBytes = lhs.dramBytes + rhs.dramBytes;
  sum.localBytes = lhs.localBytes + rhs.localBytes;
  sum.spillBytes = lhs.spillBytes + rhs.spillBytes;
  sum.computeUtilization = lhs.computeUtilization + rhs.computeUtilization;
  sum.transferUtilization = lhs.transferUtilization + rhs.transferUtilization;
  return sum;
}

bool costLess(const Cost &lhs, const Cost &rhs, const ObjectiveOrder &order) {
  auto compare = [&](CostMetric metric) -> int {
    double l = costMetric(lhs, metric);
    double r = costMetric(rhs, metric);
    if (l == r)
      return 0;
    bool lhsFirst = order.minimize ? (l < r) : (l > r);
    return lhsFirst ? -1 : 1;
  };

  if (int verdict = compare(order.primary); verdict != 0)
    return verdict < 0;
  for (CostMetric metric : order.secondary)
    if (int verdict = compare(metric); verdict != 0)
      return verdict < 0;
  return false;
}

bool ranksBefore(const Cost &lhs, uint64_t lhsId, const Cost &rhs,
                 uint64_t rhsId, const ObjectiveOrder &order) {
  if (costLess(lhs, rhs, order))
    return true;
  if (costLess(rhs, lhs, order))
    return false;
  return lhsId < rhsId;
}

std::optional<ObjectiveOrder> objectiveOrderFromMicro(llvm::StringRef metric,
                                                      bool minimize) {
  std::optional<CostMetric> primary = symbolizeCostMetric(metric);
  if (!primary)
    return std::nullopt;
  ObjectiveOrder order;
  order.primary = *primary;
  order.minimize = minimize;
  return order;
}

std::string canonicalCostString(const Cost &cost) {
  std::string out;
  for (const MetricInfo &info : kMetrics) {
    if (!out.empty())
      out += ';';
    double value = costMetric(cost, info.metric);
    char buffer[32];
    std::snprintf(buffer, sizeof(buffer), "%.6f", value);
    out += info.name.str();
    out += '=';
    out += buffer;
  }
  return out;
}

} // namespace mlir::llk::mapping
