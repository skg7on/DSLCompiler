//===- MappingPlan.h - Mapping candidates, instances, connections, plans --===//
//
// Part of the target-independent mapping core (issue #80, epic #67 D1).
//
// These are the transient structures the mapping search produces (design
// §9.2-§9.5). D1 defines the shapes and their canonical ids; the algorithms
// that populate them are later workstreams (D4 rules, D5 placement, D6
// covering). Everything here is a plain value type so a plan can be compared,
// hashed, and reported without a live MLIR context held open -- only
// `AffineMap` and `DictionaryAttr` fields reference MLIR objects.
//
// A `MappingCandidate` is an *unplaced* rule match. A `CandidateInstance`
// places one on concrete machine resources. A `ConnectionPlan` moves one value
// from a producer instance to its consumers. A `CoveringPlan` is a complete,
// ranked proposal. Concrete executor ids live only in instances and plans;
// `!micro.tile`'s owner stays the abstract class (design §5.3).
//
// Canonical ids are derived from content, with insertion-order-insensitive
// fields (covered node sets, consumer sets, route sequences) sorted before
// hashing, so two equivalently-built objects share an id.
//
//===----------------------------------------------------------------------===//

#ifndef LLK_MAPPING_MAPPINGPLAN_H
#define LLK_MAPPING_MAPPINGPLAN_H

#include "LLK/Mapping/CostModel.h"
#include "LLK/Mapping/SearchBinding.h"
#include "LLK/Mapping/WorkloadGraph.h"

#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BuiltinAttributes.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"

#include <algorithm>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace mlir::llk::mapping {

// Target-owned string ids and content-hash numeric ids (design §9.1).
using RuleId = std::string;
using ExecutorId = std::string;
using MemoryNodeId = std::string;
using LinkId = std::string;
using LayoutId = std::string;
using CandidateId = uint64_t;
using InstanceId = uint64_t;
using ConnectionId = uint64_t;
using PlanId = uint64_t;

/// An opaque target-owned implementation name plus typed parameters and the
/// emitter key its plugin understands. Only the target's emitter interprets
/// `name` and `parameters`; generic mapping code may compare, hash, report, and
/// hand the bundle to a plugin, but never reads a field as target semantics
/// (design §14.3). The plan value types carry one so a selected bundle reaches
/// the materializer unchanged; it is content-addressed by
/// `canonicalCandidateString` / `canonicalPlanString`.
struct TargetBundle {
  std::string name;
  mlir::DictionaryAttr parameters;
  std::string emitterKey;
};

/// One value crossing a candidate's boundary. `isInput` distinguishes an
/// operand the candidate consumes from a result it produces.
struct PortSpec {
  std::string name;
  WorkloadValueId value = 0;
  bool isInput = false;
  /// The operand or result occurrence this spec names, once resolved against a
  /// finalized graph. Unset while endpoint resolution is still migrating;
  /// when set it joins the candidate's canonical identity, so two uses of one
  /// value are not the same port.
  std::optional<PortRef> port;
};

/// An abstract capability a placement must satisfy. `capability` is a target
/// vocabulary word, never a concrete executor id.
struct ExecutorRequirement {
  std::string capability;
  llvm::StringMap<std::string> attrs;
};

struct MemoryRequirement {
  std::string kind;
  uint64_t minBytes = 0;
};

struct LayoutRequirement {
  std::string layoutClass;
  /// The element type and rank the requirement is solved against, taken from
  /// the operand the rule names. A layout applies to that operand, so its
  /// legality must be decided against the operand's own type -- not a single
  /// graph-wide context, which for a mixed-dtype kernel names a different
  /// value. Unset (empty element type, negative rank) means the caller's
  /// `LayoutContext` is used, which is what a hand-built candidate expects.
  /// Not part of the candidate's canonical string: the id still depends only on
  /// the layout class, so this resolution never changes a plan id.
  std::string elementType;
  int64_t rank = -1;
  /// The workload value the named port carries, so the solved binding can later
  /// be attributed to the edge that carries that value rather than to every
  /// edge touching the instance. `-1` when the rule names no port, or names one
  /// its node does not have -- in which case the binding is attributable to
  /// nothing and a connection request leaves the endpoint's layout unset.
  /// Excluded from the canonical string for the same reason as `elementType`:
  /// it is a resolution fact, and the id already depends only on the class.
  int64_t portValue = -1;
  /// The occurrence `portValue` was read from, once resolved against a
  /// finalized graph. Unset while endpoint resolution is still migrating; like
  /// `portValue`, it is a resolution fact and stays out of the canonical id.
  std::optional<PortRef> port;
};

/// An abstract compute capability a placement must attach (design §15.1).
struct ComputeRequirement {
  std::string kind;
};

/// Concrete resource consumption of one placed instance.
struct ResourceUsage {
  uint64_t executorSlots = 0;
  llvm::StringMap<uint64_t> memoryBytes;
};

/// A layout conversion over a value: the value read through `srcMap` is written
/// through `dstMap`. Both are the logical-to-physical maps of the endpoints'
/// *solved* layouts, so the conversion is concrete (`VW = 8`) rather than a
/// family pair, and it is expressed as affine maps rather than target ids --
/// which is what lets the binder emit a target-neutral `micro.transform`
/// (design §13.4 keeps target layout ids out of `#micro.layout`).
///
/// A map is null when the corresponding layout declares no map clause, leaving
/// that side's index relation to the target. `srcLayout`/`dstLayout` name the
/// families, for diagnostics and for a reader that wants the id.
struct LayoutTransform {
  std::string srcLayout;
  std::string dstLayout;
  AffineMap srcMap;
  AffineMap dstMap;
  std::string memoryNode{};
  std::string computeResource{};
};

/// An unplaced rule match.
struct MappingCandidate {
  CandidateId id = 0;
  RuleId rule;
  llvm::SmallVector<WorkloadNodeId> coveredNodes;
  TargetBundle bundle;
  llvm::SmallVector<PortSpec> ports;
  llvm::SmallVector<ExecutorRequirement> executorRequirements;
  llvm::SmallVector<MemoryRequirement> memoryRequirements;
  llvm::SmallVector<LayoutRequirement> layoutRequirements;
  llvm::SmallVector<ComputeRequirement> computeRequirements;
  llvm::StringMap<SearchValue> resolvedParameters;
  Cost lowerBound;
};

/// One layout class's solved instantiation: the parameter values the solver
/// chose and the affine map they substitute into. `layoutBindings` names the
/// layout *family*; this is what makes the binding concrete, so a materializer
/// (and a report) can state which parameterization was selected instead of
/// implying "some legal one".
///
/// The parameter map is the identity-bearing field: it is rendered sorted and
/// type-tagged by `canonicalSearchValueString`, and it participates in the
/// instance and plan ids (ruling R4 -- two parameterizations of one layout id
/// are different placements). The affine map is carried but deliberately *not*
/// hashed: it is a pure function of the bound layout id and these values, so
/// including its rendering would only add a dependency on MLIR's map printer.
///
/// Keying: `enumeratePlacements` files each solution under its requirement's
/// layout class. When one class is required by several ports of the same
/// candidate, each requirement has its own solved parameters and port
/// association, so the later ones are keyed by class plus the requirement's
/// index (for example `t.blocked#1`) rather than overwriting the first. A class
/// required once keeps the bare class as its key, so a class-keyed lookup still
/// resolves for the common case; `SolvedLayout::portValue` is what an edge
/// matches on, not the key.
struct SolvedLayout {
  /// The layout family this solution instantiates. Carried explicitly rather
  /// than inferred from the containing map's key, because a class required by
  /// several ports has index-disambiguated keys while the class itself is what
  /// a connection and its transform name.
  std::string layoutClass;
  llvm::StringMap<SearchValue> parameters;
  /// The logical-to-physical map with the integer parameters substituted; null
  /// when the layout declaration carries no map clause.
  mlir::AffineMap map;
  /// The workload value this layout was solved for, copied from the
  /// requirement's `portValue`; `-1` when the rule named no resolvable port.
  /// This is what lets a connection request ask for "the layout bound for
  /// *this* edge's value" instead of taking whatever one layout the instance
  /// happens to hold -- a rule that names an input port must not have its
  /// layout attributed to the edge carrying the result. Like `parameters`, it
  /// is a resolution fact, not content, so it is not hashed either.
  int64_t portValue = -1;
  /// The occurrence this solution was solved for, copied from the
  /// requirement's `port`; unset while endpoint resolution is still migrating.
  /// Also a resolution fact, so it is likewise not hashed.
  std::optional<PortRef> port;
};

/// One mapping candidate placed on concrete resources.
struct CandidateInstance {
  InstanceId id = 0;
  CandidateId candidate = 0;
  /// The candidate's opaque bundle, carried unchanged so a materializer reads
  /// it off the placed instance rather than re-looking-up the rule.
  TargetBundle bundle;
  llvm::StringMap<ExecutorId> executorBindings;
  llvm::StringMap<MemoryNodeId> memoryBindings;
  llvm::StringMap<std::string> computeBindings;
  llvm::StringMap<LayoutId> layoutBindings;
  /// The solved instantiation of each layout requirement, keyed by layout class
  /// (index-disambiguated when one class is required by several ports; see
  /// `SolvedLayout`). Empty when the candidate requires no layout.
  llvm::StringMap<SolvedLayout> layoutSolutions;
  ResourceUsage resourceUsage;
  Cost localCost;
};

enum class ConnectionKind {
  Direct,
  Transfer,
  LayoutTransform,
  TransferAndTransform,
  Replicate,
  Reduce
};

llvm::StringRef stringifyConnectionKind(ConnectionKind kind);
std::optional<ConnectionKind> symbolizeConnectionKind(llvm::StringRef text);

/// How one value reaches one or more consumers.
struct ConnectionPlan {
  ConnectionId id = 0;
  InstanceId producer = 0;
  llvm::SmallVector<InstanceId> consumers;
  /// The producer-side result occurrence this connection reads. Unset while
  /// endpoint resolution is still migrating; once set it joins the connection's
  /// canonical identity, so two connections that read the same value through
  /// different endpoints are distinct. The instance list above remains a
  /// compatibility projection, not the rewiring authority.
  std::optional<PortRef> producerPort;
  /// The consumer-side operand occurrences this connection serves. Rendered
  /// sorted into the canonical string; empty while resolution migrates.
  llvm::SmallVector<PortRef> consumerPorts;
  /// The several producers a `Reduce` gathers; empty for every other kind,
  /// where `producer` is the single source.
  llvm::SmallVector<InstanceId> producers;
  WorkloadValueId value = 0;
  ConnectionKind kind = ConnectionKind::Direct;
  llvm::SmallVector<MemoryNodeId> memoryRoute;
  llvm::SmallVector<ExecutorId> transferEngines;
  std::optional<AffineMap> producerMap;
  llvm::SmallVector<AffineMap> consumerMaps;
  std::optional<LayoutTransform> transform;
  Cost cost;
};

/// Structured search outcome. `searchTruncated` is set whenever a cap ended
/// the search early, so a caller never reads a truncated result as optimal.
struct PlanDiagnostics {
  std::vector<std::string> errors;
  std::vector<std::string> warnings;
  bool searchTruncated = false;
};

/// One selected placement: which node an instance covers, and the target facts
/// a materializer needs (design §18.1). Everything here is a target-neutral
/// container -- a rule id, an opaque bundle, a machine node id, a layout id.
struct PlanPlacement {
  WorkloadNodeId node = 0;
  InstanceId instance = 0;
  std::string rule;
  TargetBundle bundle;
  ExecutorId executor;
  llvm::StringMap<MemoryNodeId> memories;
  llvm::StringMap<LayoutId> layouts;
  /// The solved instantiation of each layout in `layouts`, so the selected
  /// plan states the parameterization it chose, not just the family name.
  llvm::StringMap<SolvedLayout> layoutSolutions;
};

/// One selected connection, with the route it takes.
struct PlanConnection {
  ConnectionId id = 0;
  WorkloadValueId value = 0;
  ConnectionKind kind = ConnectionKind::Direct;
  llvm::SmallVector<MemoryNodeId> route;
  llvm::SmallVector<ExecutorId> engines;
  std::optional<LayoutTransform> transform;
  /// The instances this connection serves, sorted and unique. A materializer
  /// rewires exactly these consumers to the connection's result; without them
  /// it could only redirect *every* reader of the value, which is wrong as soon
  /// as two connections carry one value along different routes. Empty for a
  /// plan built without consumer associations.
  llvm::SmallVector<InstanceId> consumers;
  /// The producer-side result occurrence this connection reads and the
  /// consumer-side operand occurrences it serves, copied from the
  /// `ConnectionPlan`. The `consumers` instance list above is a compatibility
  /// projection that cannot tell two operand uses of one value apart -- both
  /// connections of a repeated-operand edge list the same instance -- so the
  /// endpoint occurrences are what a materializer must rewire by. Unset for a
  /// plan built without endpoint resolution.
  ///
  /// Derived projection: the connection's `id` already folds these occurrences
  /// (see `canonicalConnectionString`), so they are deliberately not rendered
  /// again into `canonicalPlanString` -- that would only change every plan id
  /// without adding identity.
  ///
  /// The search populates these fields and plan identity covers them; their
  /// *consumption* -- rewiring by endpoint rather than by instance -- begins in
  /// the later B1/B4 work. Until then they are recorded, not acted on.
  std::optional<PortRef> producerPort;
  llvm::SmallVector<PortRef> consumerPorts;
};

/// A complete executable proposal covering every required node.
struct CoveringPlan {
  PlanId id = 0;
  uint64_t sourceBindingHash = 0;
  /// The `micro.candidate` symbol the source binding was loaded from, or empty
  /// for a binding-free search. The hash above covers the binding's *values*
  /// only, so this name is what lets a reader of the report name the exact
  /// `candidate=` a replay must pass (the id itself is reproducible from the
  /// same search point). It is provenance, not plan content:
  /// `canonicalPlanString` deliberately does not fold it, so a plan id is
  /// unchanged by it.
  std::string sourceBindingCandidate;
  llvm::SmallVector<InstanceId> instances;
  llvm::SmallVector<ConnectionId> connections;
  /// The same selections, resolved: which node each instance covers and how
  /// each connection runs. Placements are ordered by their (executor, memory,
  /// layout) binding tuple, then node / instance id (design §22.1);
  /// `connectionPlans` by connection id.
  llvm::SmallVector<PlanPlacement> placements;
  /// Explicit inline capacity: `PlanConnection` is large (it carries its route,
  /// engines, transform, and consumers), so the default inlined-element
  /// heuristic would not apply.
  llvm::SmallVector<PlanConnection, 4> connectionPlans;
  llvm::StringMap<SearchValue> globalParameters;
  Cost totalCost;
  PlanDiagnostics diagnostics;
};

/// Sorts and removes duplicates. Used to enforce the "sorted and unique"
/// contract on node, instance, and connection id sets.
template <typename T> void sortUnique(llvm::SmallVectorImpl<T> &values) {
  llvm::sort(values);
  values.erase(std::unique(values.begin(), values.end()), values.end());
}

CandidateId computeCandidateId(const MappingCandidate &candidate);
InstanceId computeInstanceId(const CandidateInstance &instance);
ConnectionId computeConnectionId(const ConnectionPlan &connection);
PlanId computePlanId(const CoveringPlan &plan);

std::string canonicalCandidateString(const MappingCandidate &candidate);
std::string canonicalInstanceString(const CandidateInstance &instance);
std::string canonicalConnectionString(const ConnectionPlan &connection);
std::string canonicalPlanString(const CoveringPlan &plan);

} // namespace mlir::llk::mapping

#endif // LLK_MAPPING_MAPPINGPLAN_H
