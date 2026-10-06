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
///
/// Version 2 (task B8) adds endpoint *roles* to the keys and renders every key
/// length-delimited, so a field that merely contains a separator can no longer
/// make two different pieces of work collide. A measurement taken under the
/// ambiguous version-1 key is therefore not reused.
inline constexpr uint64_t kCostModelVersion = 2;

/// Bumped independently of `kCostModelVersion` whenever a *connection*
/// measurement's identity changes. Version 2 (task B8) gained ordered
/// node/link/engine paths, concrete maps/parameters and storage rendering;
/// version 3 folded the consumer-side affine maps and a gather's declared
/// semantics/axis in, so two connections differing only there no longer
/// collide.
inline constexpr uint64_t kConnectionKeyVersion = 3;

/// What work is being looked up. Every field is part of the cache key; an
/// empty field means "not modelled", never "any".
///
/// The key must distinguish any two pieces of work whose measured cost could
/// differ, so it carries the operation's *types* and attributes alongside its
/// rule, and the *concrete* bundle parameters, layout parameterization, and
/// placement it resolved to -- not merely the rule and the layout families.
/// Two candidates that share a rule but differ in dtype, shape, an attribute,
/// a bundle parameter, a solved layout parameter, or the executor/memory they
/// bound are different work; keying them together would reuse a measurement
/// taken under conditions that no longer hold (design §17.4).
struct OperationSignature {
  std::string operation;
  /// Canonical rendering of the consuming node's operand and result types --
  /// dtype and shape -- and of its attributes.
  std::string operandTypes;
  std::string resultTypes;
  std::string attributes;
  std::string rule;
  uint64_t ruleVersion = 0;
  std::string bundle;
  /// The resolved bundle's typed parameters, canonically rendered.
  std::string bundleParameters;
  /// The instance's complete layout identity: each bound requirement's family
  /// *and* the parameters solved for it, canonically rendered and sorted.
  std::string layout;
  /// The executor's kind (the placement *class*) and, separately, the concrete
  /// placement: the executor id bound and its memory bindings.
  std::string placementClass;
  std::string placement;
  std::string routeClass;
  uint64_t costModelVersion = kCostModelVersion;

  /// Canonical rendering, so two lookups that describe the same work produce
  /// the same key.
  std::string canonicalString() const;
};

/// The target a measurement belongs to: a measurement from another machine, or
/// from the same machine before its profile changed, is not this measurement.
///
/// The machine hash alone is not the whole target identity (task B8): the rule
/// library and the layout library are target content too, and a change to
/// either changes what a piece of work *means*. Both extra hashes default to
/// the empty string, so a caller that only knows the machine keeps source
/// compatibility; a non-empty hash is required before a measurement can be
/// reused.
struct TargetContext {
  std::string target;
  std::string machineHash;
  std::string ruleHash;
  std::string layoutHash;
};

/// One connection decision's complete measurement identity (task B8): what kind
/// of connection it is, the value it carries, its endpoint *roles*, its ordered
/// node/link/engine path, the concrete layout maps and parameters it applies,
/// the storage it resolves to, and the connection-key version. The mapping
/// search and the performance evaluator describe the same movement through this
/// one shape, so a measured cost is only reused for the connection it was taken
/// on.
///
/// Every field is a canonical, length-delimited rendering (`<len>:<bytes>`), so
/// two descriptions collide only when they are the same description -- swapping
/// a producer's and a consumer's role, or merely reordering a list, cannot
/// produce the same key.
struct ConnectionSignature {
  /// `stringifyConnectionKind` of the connection.
  std::string kind;
  /// Canonical rendering of the carried value's type.
  std::string valueType;
  /// The producer-side endpoint, prefixed with its `producer` role.
  std::string producerEndpoint;
  /// The consumer-side endpoints, each prefixed with its `consumer` role and
  /// rendered sorted.
  std::string consumerEndpoints;
  /// The ordered route as concrete memory *node* ids, `a>link>b>link>c` form.
  std::string route;
  /// The ordered link ids the route crosses.
  std::string links;
  /// The ordered transfer engine ids the route uses.
  std::string engines;
  /// The concrete source/destination affine maps the connection applies.
  std::string maps;
  /// The concrete layout and bundle parameters the connection resolved.
  std::string parameters;
  /// The storage the connection resolves to (memory plus allocation bytes).
  std::string storage;
  uint64_t keyVersion = kConnectionKeyVersion;

  /// Canonical, length-delimited rendering, so two lookups that describe the
  /// same connection produce the same key and no two different connections do.
  std::string canonicalString() const;
};

struct ConnectionPlan;
class WorkloadGraph;

/// Builds the measurement identity of one synthesized connection (task B8)
/// from the same facts the search decided: kind, value type, endpoint roles,
/// ordered route/links/engines, the producer and consumer affine maps, the
/// transform's layouts, the gather semantics/axis, and the destination storage.
/// Public so a provider and its tests agree on exactly what a key contains.
ConnectionSignature
connectionSignatureFor(const ConnectionPlan &connection,
                       const WorkloadGraph &workload,
                       const machine::MachineModel &machine);

/// An optional source of measured or calibrated cycle counts.
class LatencyProvider {
public:
  virtual ~LatencyProvider() = default;

  /// The measured cycles for this work, or nullopt when there is no entry.
  /// Nullopt is not a legality verdict.
  virtual std::optional<double>
  lookupCycles(const OperationSignature &signature,
               const TargetContext &context) const = 0;

  /// The measured cycles for a connection, or nullopt when there is no entry.
  ///
  /// Nullopt is *not* a legality verdict and does not change which connections
  /// are legal: a miss leaves the static estimate in place, and a hit only
  /// replaces that estimate with a calibrated number (task B8). The default
  /// implementation always misses, so a provider written before connections
  /// existed keeps its exact behaviour.
  ///
  /// A subclass that overrides one overload hides the other, so it must bring
  /// the other into scope with `using LatencyProvider::lookupCycles;`.
  virtual std::optional<double>
  lookupCycles(const ConnectionSignature &signature,
               const TargetContext &context) const {
    (void)signature;
    (void)context;
    return std::nullopt;
  }
};

} // namespace mlir::llk::mapping

#endif // LLK_MAPPING_LATENCYPROVIDER_H
