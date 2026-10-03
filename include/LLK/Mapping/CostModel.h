//===- CostModel.h - Multi-dimensional mapping cost -----------------------===//
//
// Part of the target-independent mapping core (issue #80, epic #67 D1).
//
// Mapping cost is deliberately multi-dimensional. A candidate is never
// reduced to a scalar here: `micro.objective` declares which dimension ranks
// first and in what order the rest break ties, so the same set of costs can
// be ranked differently by different workloads. `costLess` applies that
// declared order; it never invents one.
//
// Dimension spellings match the `micro.objective` metric vocabulary so a
// report, a constraint, and an objective all name the same thing.
//
//===----------------------------------------------------------------------===//

#ifndef LLK_MAPPING_COSTMODEL_H
#define LLK_MAPPING_COSTMODEL_H

#include "llvm/ADT/StringRef.h"

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace mlir::llk::mapping {

/// One cost estimate, kept unranked until an objective orders it. Optimistic
/// values (used as pruning lower bounds) are the caller's responsibility: this
/// type only carries the numbers.
struct Cost {
  double latencyCycles = 0.0;
  uint64_t dramBytes = 0;
  uint64_t localBytes = 0;
  uint64_t spillBytes = 0;
  double computeUtilization = 0.0;
  double transferUtilization = 0.0;
};

/// The rankable dimensions of a `Cost`. The spellings mirror the metric names
/// a `micro.objective` can declare.
enum class CostMetric {
  LatencyCycles,
  DramBytes,
  LocalBytes,
  SpillBytes,
  ComputeUtilization,
  TransferUtilization
};

llvm::StringRef stringifyCostMetric(CostMetric metric);
std::optional<CostMetric> symbolizeCostMetric(llvm::StringRef text);

/// Reads one dimension as a double; integer dimensions are widened so every
/// metric is comparable through one accessor.
double costMetric(const Cost &cost, CostMetric metric);

/// Component-wise sum. Used to combine rule-local, route, and transform costs
/// into a plan total without collapsing the dimensions.
Cost addCost(const Cost &lhs, const Cost &rhs);

/// How an objective ranks costs: primary metric first, then each secondary
/// metric in the declared order. `minimize` false means larger is better.
struct ObjectiveOrder {
  CostMetric primary = CostMetric::LatencyCycles;
  std::vector<CostMetric> secondary;
  bool minimize = true;
};

/// True when `lhs` ranks strictly ahead of `rhs` under `order`. Ties on every
/// listed metric return false in both directions, leaving the caller to break
/// the tie by stable id.
bool costLess(const Cost &lhs, const Cost &rhs, const ObjectiveOrder &order);

/// Fixed-format rendering of every dimension, byte-stable across runs and
/// platforms so it can key hashes and reports.
std::string canonicalCostString(const Cost &cost);

} // namespace mlir::llk::mapping

#endif // LLK_MAPPING_COSTMODEL_H
