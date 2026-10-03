//===- LatencyProvider.h - Optional measured latencies --------------------===//
//
// Part of the target-independent mapping core (epic #67).
//
// Static cost is always available and always admissible for pruning; a
// `LatencyProvider` is an *optional* better estimate from calibration or
// measurement (design §17.3). Its absence is not a verdict: a provider that has
// no entry for a piece of work leaves the static estimate in place, and no
// provider at all leaves every estimate static. Neither makes a candidate
// illegal -- measurement changes what work *costs*, never what is allowed.
//
// A measurement is only reusable while everything that could change it still
// holds, so the lookup carries the whole cache key (design §17.4): the machine
// profile, the rule and its version, the operation, the bundle, the layout and
// placement classes, the route class, and the cost-model version. A caller
// that cannot fill a field leaves it empty rather than guessing, which makes
// the entry unreachable rather than wrong.
//
//===----------------------------------------------------------------------===//

#ifndef LLK_MAPPING_LATENCYPROVIDER_H
#define LLK_MAPPING_LATENCYPROVIDER_H

#include "LLK/Mapping/CostModel.h"

#include "llvm/ADT/StringRef.h"

#include <cstdint>
#include <optional>
#include <string>

namespace mlir::llk::mapping {

/// Bumped whenever a static estimate would mean something different, so a
/// measurement taken under an older model is not silently reused.
inline constexpr uint64_t kCostModelVersion = 1;

/// What work is being looked up. Every field is part of the cache key; an
/// empty field means "not modelled", never "any".
struct OperationSignature {
  std::string operation;
  std::string rule;
  uint64_t ruleVersion = 0;
  std::string bundle;
  std::string layout;
  std::string placementClass;
  std::string routeClass;
  uint64_t costModelVersion = kCostModelVersion;

  /// Canonical rendering, so two lookups that describe the same work produce
  /// the same key.
  std::string canonicalString() const;
};

/// The target a measurement belongs to: a measurement from another machine, or
/// from the same machine before its profile changed, is not this measurement.
struct TargetContext {
  std::string target;
  std::string machineHash;
};

/// An optional source of measured or calibrated cycle counts.
class LatencyProvider {
public:
  virtual ~LatencyProvider() = default;

  /// The measured cycles for this work, or nullopt when there is no entry.
  /// Nullopt is not a legality verdict.
  virtual std::optional<double>
  lookupCycles(const OperationSignature &signature,
               const TargetContext &context) const = 0;
};

} // namespace mlir::llk::mapping

#endif // LLK_MAPPING_LATENCYPROVIDER_H
