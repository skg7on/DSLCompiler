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

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace mlir::llk::machine {
struct MachineModel;
} // namespace mlir::llk::machine

namespace mlir::llk::mapping {

/// One cost estimate, kept unranked until an objective orders it. Optimistic
/// values (used as pruning lower bounds) are the caller's responsibility: this
/// type only carries the numbers.
struct Cost {
  double latencyCycles = 0.0;
  uint64_t dramBytes = 0;
  uint64_t localBytes = 0;
  /// Deliberately never populated: the Micro dialect models no spill, so there
  /// is nothing to count. It stays 0 until a spill notion exists rather than
  /// inventing a model for one (design §17.2).
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

/// First-order component utilization (design §17.2): the `busyCycles` a
/// resource spent over the cycles it had available in one machine sync period,
/// spread across `parallelUnits` interchangeable units. The sync period is what
/// the model says one barrier and one wait cost; a machine that models neither
/// leaves the denominator unknown, so this returns `nullopt` and the caller
/// leaves the dimension 0 rather than inventing a window. Returns `nullopt`
/// likewise when the machine offers no such unit.
///
/// The window's length in seconds is `syncPeriodCycles / clockHz`, so its
/// available cycles are `clockHz * (syncPeriodCycles / clockHz) * units` =
/// `syncPeriodCycles * units`. `clockHz` therefore cancels in the
/// dimensionless ratio: an absent clock still permits a cycle ratio, while an
/// absent sync period does not.
std::optional<double> utilizationEstimate(double busyCycles,
                                          const machine::MachineModel &machine,
                                          uint32_t parallelUnits);

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

/// Total order used to rank plans and partial plans: the declared objective
/// decides, and an exact cost tie falls back to the smaller stable id, so the
/// same inputs always produce the same order.
bool ranksBefore(const Cost &lhs, uint64_t lhsId, const Cost &rhs,
                 uint64_t rhsId, const ObjectiveOrder &order);

/// Builds the comparison order a `micro.objective` declares from its metric
/// spelling, direction (`minimize == false` means maximize), and optional
/// secondary metrics, which are kept in the declared order as tie-breakers
/// (design §17.1). Every metric spelling the Micro dialect verifier accepts is
/// resolved to its `CostMetric`; a spelling the cost model cannot honor is an
/// error, never a silent fall back to the default -- §17.1 forbids replacing a
/// declared objective.
llvm::Expected<ObjectiveOrder>
objectiveOrderFromMicro(llvm::StringRef metric, bool minimize,
                        llvm::ArrayRef<llvm::StringRef> secondary = {});

/// Fixed-format rendering of every dimension, byte-stable across runs and
/// platforms so it can key hashes and reports.
std::string canonicalCostString(const Cost &cost);

} // namespace mlir::llk::mapping

#endif // LLK_MAPPING_COSTMODEL_H
