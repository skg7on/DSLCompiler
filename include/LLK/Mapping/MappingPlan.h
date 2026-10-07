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
#include "llvm/ADT/StringSwitch.h"

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
  /// The workload port occurrence this requirement governs, when the rule
  /// named one (`require memory output "large" kind dram`). Unset for the
  /// legacy bare form (`require memory kind sram`), which governs the whole
  /// requirement kind rather than one port. A set port is what lets the search
  /// charge a materialized output to the memory its own port selected instead
  /// of replicating the first output across every binding.
  std::optional<PortRef> port;
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

/// A compute/memory assignment for one endpoint port (design §9.6): the memory
/// node a named occurrence's value is bound to. Persisted with the selected
/// state so a materializer can attribute a storage decision to the occurrence
/// that owns it rather than to a whole instance.
struct PortMemoryBinding {
  PortRef port;
  MemoryNodeId memory;
  bool operator==(const PortMemoryBinding &) const = default;
};

/// One mapping candidate placed on concrete resources.
struct CandidateInstance {
  InstanceId id = 0;
  CandidateId candidate = 0;
  /// The candidate's opaque bundle, carried unchanged so a materializer reads
  /// it off the placed instance rather than re-looking-up the rule.
  TargetBundle bundle;
  llvm::StringMap<ExecutorId> executorBindings;
  /// Memory bindings of the legacy bare requirements, keyed by requirement
  /// kind (`require memory kind sram`). A requirement that names a port records
  /// a `PortMemoryBinding` instead, so this map stays byte-identical for the
  /// rules that do not name ports.
  llvm::StringMap<MemoryNodeId> memoryBindings;
  llvm::StringMap<std::string> computeBindings;
  llvm::StringMap<LayoutId> layoutBindings;
  /// The solved instantiation of each layout requirement, keyed by layout class
  /// (index-disambiguated when one class is required by several ports; see
  /// `SolvedLayout`). Empty when the candidate requires no layout.
  llvm::StringMap<SolvedLayout> layoutSolutions;
  /// The memory each named endpoint occurrence is bound to, one entry per
  /// requirement that named a port. Sorted by port. Unlike `memoryBindings`,
  /// this is keyed by occurrence, so two same-kind requirements can select
  /// different nodes. Part of the instance's canonical identity when non-empty
  /// (a resolved memory assignment is execution-affecting), and empty for every
  /// rule that declares no named-port requirement. An out-of-line container so
  /// adding it does not push `PlanPlacement`/`CandidateInstance` past the
  /// `SmallVector` element-size limit.
  std::vector<PortMemoryBinding> portMemoryBindings;
  ResourceUsage resourceUsage;
  Cost localCost;
};

/// A scheduling step index. Concrete storage lifetimes are expressed in these
/// (design §9.6); B1 persists them, B3 builds them.
using PlanStepId = uint64_t;

/// One concrete storage allocation the selected plan reserves: the workload
/// value it holds, the memory it lives in, its footprint, an optional aliased
/// allocation, and the step interval it is live over. All arithmetic over these
/// fields is checked by their consumers; an unknown footprint is not zero.
struct StorageAllocation {
  uint64_t id = 0;
  WorkloadValueId value = 0;
  MemoryNodeId memory;
  uint64_t bytes = 0;
  std::optional<uint64_t> aliasOf;
  PlanStepId beginStep = 0;
  PlanStepId endStep = 0; // live through this step
};

/// One synchronization decision: a step that waits for a set of connections,
/// precedes a set of endpoint occurrences, and may require a barrier.
struct SynchronizationStep {
  uint64_t id = 0;
  std::vector<ConnectionId> waitsFor;
  std::vector<PortRef> precedes;
  bool requiresBarrier = false;
};

/// What a plan step does. `finalizeStoragePlan` builds a deterministic
/// plan-step DAG: one `Compute` step per selected placement, one `Movement`
/// step per materialized connection, and one `Synchronization` step per wait a
/// movement implies.
enum class PlanStepKind { Compute, Movement, Synchronization };

llvm::StringRef stringifyPlanStepKind(PlanStepKind kind);
std::optional<PlanStepKind> symbolizePlanStepKind(llvm::StringRef text);

/// One node of the plan-step DAG (design §9.6). The `node` is set for a
/// `Compute` step and the `connection` for a `Movement` or `Synchronization`
/// step; the other is left 0. A storage allocation's `beginStep`/`endStep` and
/// every dependency edge name these stable ids.
struct PlanStep {
  PlanStepId id = 0;
  PlanStepKind kind = PlanStepKind::Compute;
  WorkloadNodeId node = 0;
  ConnectionId connection = 0;
};

/// A dependency edge in the plan-step DAG: `to` must follow `from`.
struct PlanStepEdge {
  PlanStepId from = 0;
  PlanStepId to = 0;
  bool operator==(const PlanStepEdge &) const = default;
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

/// The explicit combination a multi-producer (`Reduce`) connection performs
/// (task B6). Nothing is inferred from topology: a connection that gathers
/// several producers and declares no semantics cannot be materialized, and the
/// stable reason names that. The table is hand-written here rather than in a
/// `.td` enum because the target-neutral mapping core links no dialect, and the
/// same three words are the Micro `micro.gather` kinds.
enum class GatherSemantics { Sum, Max, Concatenate };

inline llvm::StringRef stringifyGatherSemantics(GatherSemantics semantics) {
  switch (semantics) {
  case GatherSemantics::Sum:
    return "sum";
  case GatherSemantics::Max:
    return "max";
  case GatherSemantics::Concatenate:
    return "concat";
  }
  return "";
}

inline std::optional<GatherSemantics>
symbolizeGatherSemantics(llvm::StringRef text) {
  return llvm::StringSwitch<std::optional<GatherSemantics>>(text)
      .Case("sum", GatherSemantics::Sum)
      .Case("max", GatherSemantics::Max)
      .Case("concat", GatherSemantics::Concatenate)
      .Default(std::nullopt);
}

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
  /// The explicit combination semantics of a `Reduce` (gather). Unset for every
  /// other kind and for a multi-producer connection that has not declared what
  /// its combination means -- which is exactly the connection that cannot be
  /// materialized (task B6). Nothing is inferred from the producer count.
  std::optional<GatherSemantics> gatherSemantics;
  /// The axis a `Concatenate` gather joins along. Set exactly when the
  /// semantics is `Concatenate`; a `Sum`/`Max` gather carries no axis.
  std::optional<uint64_t> concatAxis;
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
  /// Informational notes a post-search stage emits -- storage finalization's
  /// occupancy and analysis-fallback reports. Deliberately separate from
  /// `warnings`: `canonicalPlanString` does not fold this field, so a staged
  /// note never changes a plan id and finalization is idempotent. A caller that
  /// recomputes `plan.id` after `finalizeStoragePlan` gets the same id it had
  /// before.
  std::vector<std::string> storageNotes;
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
  /// The concrete capability node selected for each compute requirement the
  /// rule declared, keyed by the requirement's kind (`vector_engine`), copied
  /// from the selected instance's `computeBindings` -- never re-derived from
  /// the executor. Two attached engines of one kind are two distinct
  /// placements, and this is what lets that decision survive report, metadata
  /// and replay instead of collapsing to the machine's first engine. Empty for
  /// a rule that requires no compute capability. Part of the plan's canonical
  /// identity when non-empty; a materializer and the normalized event stream
  /// read the selected node from here (issue #129, task R1).
  llvm::StringMap<std::string> computeBindings{};
  /// The compute capability kinds this placement's rule requires
  /// (`vector_engine`), sorted and unique. It is what lets a stage reading
  /// `computeBindings` tell "the rule requires no capability" -- a copy or
  /// store placement -- from "the recorded selection is missing", which is a
  /// dropped or tampered decision and must be diagnosed rather than re-derived
  /// from the executor's first attachment.
  ///
  /// Derived from `rule`, exactly as a layout solution's parameters are: it is
  /// a projection of content the plan already carries, so it is deliberately
  /// *not* folded into `canonicalPlanString` (two placements of one rule cannot
  /// differ in it) and a decoder re-derives it from the target's rule registry.
  std::vector<std::string> computeRequirements{};

  /// The resolved values of the rule's own declared parameters (the ones its
  /// `require` constraints derive), so the selected plan states the exact
  /// assignment generation solved rather than leaving a reader to re-derive
  /// one. Verification validates *this* assignment and never substitutes a
  /// different legal one.
  llvm::StringMap<SearchValue> resolvedParameters;
  /// The memory bound to each named endpoint occurrence, copied from the
  /// instance's `portMemoryBindings`. Empty for a placement whose rule declares
  /// no named-port memory requirement; when set it lets a materializer
  /// attribute a storage decision to the occurrence that owns it. Sorted by
  /// port.
  std::vector<PortMemoryBinding> portMemoryBindings;
  /// The selected instance's measured-or-static cost (task B8), copied from the
  /// search entry so a plan's normalized event stream can charge this node the
  /// *same* estimate the search ranked it on rather than re-deriving one. A
  /// derived execution fact: deliberately excluded from `canonicalPlanString`,
  /// so adding it does not change any plan id.
  Cost cost;
  /// The node's output element count (MACs for a matrix op), copied from the
  /// extraction facts. Excluded from `canonicalPlanString`, like `cost`.
  uint64_t workItems = 0;
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
  /// The explicit combination semantics of a `Reduce` (gather) connection and
  /// the axis a `Concatenate` joins along, copied from the `ConnectionPlan`.
  /// Unset for every non-gather connection and for a multi-producer connection
  /// whose semantics were never declared -- the case that stays Partial-only
  /// with a stable reason instead of being materialized as an assumed sum.
  std::optional<GatherSemantics> gatherSemantics;
  std::optional<uint64_t> concatAxis;
  /// The producer-side value occurrences a gather combines, in the order a
  /// `Concatenate` joins them. Empty for every non-gather connection; the
  /// single `producerPort` above remains the one producer of a movement.
  llvm::SmallVector<PortRef> producerPorts;
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
  /// The storage allocations (design §9.6) this connection reads or writes,
  /// by `StorageAllocation::id`. B1 persists the ids; B3 populates them. Empty
  /// for a plan built before storage planning.
  llvm::SmallVector<uint64_t> storageIds;
  /// The connection's synthesized cost (task B8): the transfer estimate the
  /// route carried, or the shared transform estimate for a conversion. A
  /// derived execution fact, deliberately excluded from `canonicalPlanString`
  /// (the connection's own id already folds its kind, route, engines and
  /// transform, so the cost adds no identity a reader could not re-derive).
  Cost cost;
  /// The carried value's element count, for the normalized event stream.
  uint64_t workItems = 0;
  /// The carried value's type, so a normalized transform event can be charged
  /// from the shared conversion estimate exactly as the materialized kernel's
  /// is. A derived execution fact, excluded from `canonicalPlanString`.
  mlir::Type valueType;
};

/// Where a plan's final score came from (task B8). A score is only a schedule
/// when the plan's normalized events could be built; when an unknown strict
/// fact prevents that, the additive accumulation is reported *as* the
/// accumulation, never presented as a scheduled latency.
enum class PlanScoreSource { Accumulation, Schedule };

llvm::StringRef stringifyPlanScoreSource(PlanScoreSource source);

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
  /// layout, compute) binding tuple, then node / instance id (design §22.1);
  /// `connectionPlans` by connection id.
  /// Explicit inline capacity: `PlanPlacement` carries the selected bundle,
  /// bindings and now its measured cost, so the default inlined-element
  /// heuristic would not apply.
  llvm::SmallVector<PlanPlacement, 1> placements;
  /// Explicit inline capacity: `PlanConnection` is large (it carries its route,
  /// engines, transform, and consumers), so the default inlined-element
  /// heuristic would not apply.
  llvm::SmallVector<PlanConnection, 4> connectionPlans;
  llvm::StringMap<SearchValue> globalParameters;
  /// The final candidate score (task B8): the overlapped latency the *shared*
  /// resource scheduler produces from the plan's normalized events, not the
  /// additive sum of rule and connection costs. It folds into
  /// `canonicalPlanString`, so a plan's id reflects the score a reader ranks it
  /// by. Falls back to `accumulatedCost` only when the plan's events cannot be
  /// built (an unknown strict fact), which is reported.
  Cost totalCost;
  /// The additive accumulation of rule-local and connection costs over the
  /// whole plan -- the search's optimistic partial-cost model, kept separately
  /// named so it is never confused with the scheduled final score. Excluded
  /// from `canonicalPlanString`: it is a search-internal quantity, not content.
  Cost accumulatedCost;
  /// Whether `totalCost` is the shared schedule's latency or the accumulation
  /// fallback. A reader must not treat an `Accumulation` score as scheduled.
  PlanScoreSource scoreSource = PlanScoreSource::Accumulation;
  PlanDiagnostics diagnostics;

  // --- persisted selected state (schema v3, tasks B1/R1) -------------------
  //
  // These are deliberately excluded from `canonicalPlanString`: they are
  // provenance and persisted selection metadata, not fields that change a
  // plan's content id. A plan decoded from metadata keeps the id it was encoded
  // with. (Version 3 is current: it persists each placement's selected compute
  // node, and its canonical identity is the `v3|`-tagged form that omits
  // derived scores and diagnostics.)
  //
  // The metadata schema this plan was persisted under (0 when it was never
  // persisted -- an in-memory search result).
  uint64_t schemaVersion = 0;
  /// True when every execution-affecting decision was materialized. A partial
  /// plan (see `BindContract`) records `false`.
  bool materialized = true;
  /// Content hash of the canonical, pre-materialization source workload graph
  /// (see `computeSourceGraphHash`): operand occurrences, types, access maps
  /// and semantic attributes, with bookkeeping attributes omitted. Retained
  /// when a materialized graph is validated through its recorded connection
  /// provenance.
  uint64_t graphHash = 0;
  /// Content hash of the target: its name plus the machine, layout and rule
  /// library hashes. A plan is only executable against the target it was bound
  /// for.
  uint64_t targetHash = 0;
  uint64_t machineHash = 0;
  uint64_t layoutHash = 0;
  uint64_t ruleHash = 0;
  /// Concrete storage allocations the plan reserves (design §9.6). B3 builds
  /// these; B1 round-trips them.
  std::vector<StorageAllocation> allocations;
  /// Synchronization decisions the plan makes (design §9.6). B3 builds these;
  /// B1 round-trips them.
  std::vector<SynchronizationStep> synchronization;
  /// The deterministic plan-step DAG `finalizeStoragePlan` built (design §9.6):
  /// every step's kind and what it names, and the dependency edges between
  /// them. B3 builds these; B4-B6 consume them to order materialization. Like
  /// the allocations they annotate, they are deliberately excluded from
  /// `canonicalPlanString`, so storage planning never changes a plan id.
  std::vector<PlanStep> steps;
  std::vector<PlanStepEdge> stepEdges;
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

/// The executor a connection's layout conversion runs on: the first consumer
/// placement's, else the producer placement's. The *one* policy the normalized
/// plan events and the canonical materializer share, so the executor a
/// transform event names and the `micro.engine` the binder stamps cannot
/// disagree (task B8). Empty only when neither endpoint resolves to a
/// placement.
std::string transformExecutorFor(const CoveringPlan &plan,
                                 const PlanConnection &connection);

/// Length-delimited, sorted canonical rendering of a selected-compute map
/// (`<keyLen>:<key>=<valueLen>:<value>`), the identical encoding the instance
/// and plan content keys fold. Exported so a measurement key renders the same
/// decision the same way -- a plain separator-joined form would let a key or
/// node id containing the separator collide with a different selection (issue
/// #129, task R1).
std::string
canonicalComputeBindingsString(const llvm::StringMap<std::string> &bindings);

/// The first compute capability kind `placement`'s rule requires but for which
/// `placement.computeBindings` records no concrete node, or `nullopt` when the
/// placement is complete.
///
/// A recorded selection is authoritative: a rule that requires a capability but
/// whose plan records no node for it has had that decision dropped or tampered
/// with, so it is a defect -- never a licence to re-derive an engine from the
/// executor's declaration order (issue #129, task R1). Shared by the
/// normalizer, the report reader and the metadata decoder so all three reach
/// the same verdict.
std::optional<std::string>
missingComputeBinding(const PlanPlacement &placement);

std::string canonicalCandidateString(const MappingCandidate &candidate);
std::string canonicalInstanceString(const CandidateInstance &instance);
std::string canonicalConnectionString(const ConnectionPlan &connection);
std::string canonicalPlanString(const CoveringPlan &plan);

} // namespace mlir::llk::mapping

#endif // LLK_MAPPING_MAPPINGPLAN_H
