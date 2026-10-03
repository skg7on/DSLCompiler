//===- CostModel.cpp - Multi-dimensional mapping cost ---------------------===//

#include "LLK/Mapping/CostModel.h"

#include "llvm/ADT/Twine.h"

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

/// Resolves a `micro.objective` metric spelling. The Micro dialect verifier
/// (`ObjectiveOp::verify`) accepts six names; only two coincide with a
/// `CostMetric` spelling, so the four that differ are aliased here. A
/// `CostMetric` spelling is also accepted, so the bridge is usable from either
/// vocabulary.
std::optional<CostMetric> symbolizeMicroMetric(llvm::StringRef text) {
  if (text == "latency_cycles")
    return CostMetric::LatencyCycles;
  if (text == "dram_bytes")
    return CostMetric::DramBytes;
  if (text == "sram_bytes")
    return CostMetric::LocalBytes;
  if (text == "matrix_utilization")
    return CostMetric::ComputeUtilization;
  if (text == "dma_utilization")
    return CostMetric::TransferUtilization;
  if (text == "capacity_spill_bytes")
    return CostMetric::SpillBytes;
  return symbolizeCostMetric(text);
}
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

llvm::Expected<ObjectiveOrder>
objectiveOrderFromMicro(llvm::StringRef metric, bool minimize,
                        llvm::ArrayRef<llvm::StringRef> secondary) {
  std::optional<CostMetric> primary = symbolizeMicroMetric(metric);
  if (!primary)
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        llvm::Twine("micro.objective names an unknown metric '") + metric +
            "'");

  ObjectiveOrder order;
  order.primary = *primary;
  order.minimize = minimize;
  order.secondary.reserve(secondary.size());
  for (llvm::StringRef spelling : secondary) {
    std::optional<CostMetric> resolved = symbolizeMicroMetric(spelling);
    if (!resolved)
      return llvm::createStringError(
          llvm::inconvertibleErrorCode(),
          llvm::Twine("micro.objective names an unknown secondary metric '") +
              spelling + "'");
    order.secondary.push_back(*resolved);
  }
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
