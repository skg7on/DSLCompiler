//===- MachineModel.h - Versioned machine topology ------------------------===//
//
// Part of the v2 machine model (issue #82, epic #67).
//
// A machine is a graph of stable-ID nodes and directed edges, not a flat bag
// of parameters:
//
//   executor         a place work can execute
//   memory           a place data can reside
//   compute          a capability attached to an executor
//   transfer_engine  a resource that moves data over links
//
// with edges `contains` (a parent), `dominates` (memory visibility),
// `attached_to` (a capability's executor), and `link` (a directed data path).
//
// The model answers the questions a mapping search asks -- can this owner run
// here, can this executor see this memory, what capability and transfer
// resource hang off this executor -- through normalized queries. Placement
// matches an abstract Micro owner against an executor's *kind*, never against
// a concrete executor id (design §11.5).
//
// Node `kind` values are the Micro dialect vocabulary: executor, compute, and
// transfer kinds are `micro::Owner` names, memory kinds are
// `micro::MemorySpace` names. The model stores them as strings so it carries no
// dependency on the dialect headers at its public surface; the terminology
// (and the validation) lives in the dialect enums.
//
//===----------------------------------------------------------------------===//

#ifndef LLK_MACHINE_MACHINEMODEL_H
#define LLK_MACHINE_MACHINEMODEL_H

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <vector>

namespace mlir::llk::machine {

/// The only schema major this build understands. A file naming another major
/// is rejected rather than reinterpreted (design §11.6).
inline constexpr uint32_t kSupportedSchemaMajor = 2;

/// The highest schema *minor* this build understands. Design §11.6 requires a
/// minor addition to be optional or defaulted, so a file at a *higher* minor
/// may carry keys added after this build and the loader ignores the keys it
/// does not recognize. At this minor or lower, an unrecognized key is a typo
/// (or a misdeclared schema), and the loader rejects it: silently ignoring it
/// would defeat the diagnostic §11.6 depends on.
///
/// The declared minor is unbounded and self-declared, so a file may claim a
/// large minor to opt out of unknown-key rejection (typos included). Only
/// *unknown keys* are tolerated this way; the major check and
/// `verifyMachineModel` still enforce structure, required fields, and kinds.
inline constexpr uint32_t kSupportedSchemaMinor = 0;

/// The schema string a v2 file declares.
inline constexpr llvm::StringLiteral kSchemaName = "llk.machine.v2";

/// How an executor orders and overlaps the work handed to it. This is the
/// scheduling half of design §11.3's "supported concurrency and scheduling
/// properties"; the concurrency half is `ExecutorNode::concurrency`.
///
///   InOrder      work retires in issue order, so at most one item per slot is
///                in flight and a simulator may not overlap two.
///   OutOfOrder   independent items may overlap and retire out of order.
///
/// Co-residency -- several owners sharing one executor -- is not a separate
/// field: `concurrency` already says how many slots share the executor, and
/// this describes how each slot schedules.
///
/// Representational today: the field is loaded, validated, and content-hashed,
/// but the simulator does not yet consult it (tracked follow-up: let the
/// simulator overlap work on `OutOfOrder` executors).
enum class SchedulingClass : uint8_t { InOrder, OutOfOrder };

/// The name a v2 file writes for `value` (`in_order` / `out_of_order`).
llvm::StringRef stringifySchedulingClass(SchedulingClass value);
/// The inverse of stringifySchedulingClass; nullopt for an unknown name.
std::optional<SchedulingClass> symbolizeSchedulingClass(llvm::StringRef text);

/// Whether a link is meant to carry the reverse transfer as well. `source` and
/// `destination` fix the transfer's direction; this records whether the
/// *opposite* direction is intended to be modelled by the same edge or by its
/// own link. It is the directionality half of design §11.3's "directionality
/// and concurrency class"; concurrency is `LinkEdge::concurrency`.
///
/// Representational today: the router follows `source -> destination` only and
/// does not yet consume `Bidirectional`, so declaring it does NOT by itself
/// make the reverse hop routable (tracked follow-up: make routing honour it).
/// Until then, model a reverse path with an explicit reverse link -- which is
/// what the shipped profiles do -- and do not declare `Bidirectional` on an
/// edge that already has an explicit reverse link, or a reader that later
/// honours the field would count the reverse hop twice.
enum class LinkDirectionality : uint8_t { Unidirectional, Bidirectional };

/// The name a v2 file writes for `value`
/// (`unidirectional` / `bidirectional`).
llvm::StringRef stringifyLinkDirectionality(LinkDirectionality value);
/// The inverse of stringifyLinkDirectionality; nullopt for an unknown name.
std::optional<LinkDirectionality> symbolizeLinkDirectionality(llvm::StringRef);

/// A place work can execute. `parent` is the containment edge; `refines` names
/// additional owner kinds this executor satisfies, which is how an accelerator
/// PE can accept a kernel that asked for a `worker`.
struct ExecutorNode {
  std::string id;
  std::string kind;
  std::optional<std::string> parent;
  std::vector<int64_t> coordinates;
  uint32_t concurrency = 1;
  std::vector<std::string> refines;
  /// Defaults to in-order: it is the conservative assumption and the one the
  /// model implied before the field existed, so existing profiles keep their
  /// meaning. Loaded, hashed, and verified, but not yet consulted by the
  /// simulator.
  SchedulingClass schedulingClass = SchedulingClass::InOrder;
  /// Executor ids this executor declares interchangeable (YAML
  /// `equivalent_to`). Design §15.1: "symmetric placements may be canonicalized
  /// when *the target declares* executors equivalent". A declaration is the
  /// target's assertion that two executors can be swapped without changing a
  /// binding; it is not a cost proof -- the model cannot guarantee
  /// cost-invariance, and a declared group may differ in concurrency, so a
  /// declaration deliberately collapses even executors the structural
  /// heuristic would keep apart. The symmetry reducer collapses a declared
  /// group to one representative while `PlacementOptions::reduceSymmetry`
  /// remains the on/off switch.
  ///
  /// The declaration must be *mutual*, name an *existing* executor, and join
  /// executors of the *same kind*; `verifyMachineModel` rejects a one-sided,
  /// unknown, self-, or cross-kind entry (so a loaded model always satisfies
  /// the rule, and the order of checks is deterministic). Empty means "the
  /// target declares nothing"; symmetry reduction then falls back to the
  /// structural heuristic, which compares kind, parent, logical coordinates,
  /// concurrency, scheduling class, and the attached compute and visible memory
  /// nodes -- so two executors with distinct performance characteristics are
  /// not collapsed merely because they look alike.
  std::vector<std::string> equivalentTo{};
};

/// A place data can reside. `visibleFrom` names the executor scope that can
/// address this memory; descendants of that executor inherit visibility.
struct MemoryNode {
  std::string id;
  std::string kind;
  std::string visibleFrom;
  uint64_t capacityBytes = 0;
  uint64_t alignmentBytes = 1;
  std::vector<std::string> supportedLayouts;
  std::optional<uint32_t> banks;
  /// Cost of *accessing* this memory, as opposed to moving between two of
  /// them: a link carries transfer cost, a memory carries access cost. Zero
  /// means the profile does not model it.
  double bandwidthBytesPerCycle = 0.0;
  uint64_t latencyCycles = 0;
  /// Smallest unit a single *access* moves, when the profile models one. Named
  /// for access, not transfer, to distinguish it from `LinkEdge::
  /// transactionBytes`, which is the transfer granule the router divides a
  /// copy's size by; this one describes one access into the memory itself and
  /// is not consulted by the router. It is the transaction half of design
  /// §11.3's "optional banking and transaction properties"; `banks` is the
  /// independent capacity/parallelism half. Absent means "not modelled": a
  /// consumer falls back to `alignmentBytes`, the smallest addressable granule,
  /// and a declared value must be positive. Loaded, hashed, and verified, but
  /// not yet consulted by the router or simulator.
  std::optional<uint64_t> accessGranularityBytes = std::nullopt;
};

/// A capability attached to an executor.
struct ComputeNode {
  std::string id;
  std::string kind;
  std::string attachedTo;
  std::vector<std::string> elementTypes;
  std::vector<std::string> supportedLayouts;
  std::vector<std::vector<int64_t>> shapes;
  /// Elements per instruction for a dtype, when the capability has one (a
  /// vector engine's lane count). Absent for capabilities without a per-dtype
  /// width, such as a matrix engine. Layout declarations query this as
  /// `machine.compute(<kind>).lanes(<dtype>)`.
  std::map<std::string, int64_t> lanes;
  uint64_t issueCycles = 1;
  uint64_t latencyCycles = 0;
  std::optional<double> throughputPerCycle;
  uint32_t concurrency = 1;
  /// Element types this capability accumulates into, when it distinguishes
  /// them from its inputs. Ranked by the checks that guard accumulator
  /// capacity and MMA compatibility.
  std::vector<std::string> accumulatorDTypes;
  /// Most work items that may reside on the capability at once, when the
  /// profile models a limit. This is the occupancy half of design §11.3's
  /// "concurrency and occupancy limits"; `concurrency` is the issue-slot half,
  /// and the two are independent (resident items may outnumber issue slots).
  /// Absent means "unconstrained" -- the slots are the only bound, which is
  /// what the model assumed before the field existed -- and a declared value
  /// must be positive. Loaded, hashed, and verified, but not yet consulted by
  /// the simulator.
  std::optional<uint32_t> occupancyLimit = std::nullopt;
};

/// A resource that moves data over links, attached to an executor.
struct TransferEngineNode {
  std::string id;
  std::string kind;
  std::string attachedTo;
  uint32_t count = 1;
  uint32_t maxOutstanding = 1;
  /// Fixed cost of starting a transfer, independent of its size.
  uint64_t setupCycles = 0;
};

/// A directed data path between two memories.
struct LinkEdge {
  std::string id;
  std::string source;
  std::string destination;
  double bandwidthBytesPerCycle = 0.0;
  uint64_t latencyCycles = 0;
  /// Granularity of one *transfer* over this link: the router requires a copy's
  /// size to tile into whole transactions (design §11.3 "transaction
  /// granularity"; ruling P9: granularity, not a size cap). Distinct from
  /// `MemoryNode::accessGranularityBytes`, which describes an access into a
  /// memory and which the router does not read.
  uint64_t transactionBytes = 1;
  std::vector<std::string> transferEngines;
  uint32_t concurrency = 1;
  /// Defaults to unidirectional: a transfer runs from `source` to
  /// `destination` only, the semantics the edge had before the field existed.
  /// Representational today -- the router follows `source -> destination` only
  /// and does not consume `Bidirectional`; see the enum for the follow-up and
  /// the explicit-reverse-link caveat.
  LinkDirectionality directionality = LinkDirectionality::Unidirectional;
};

/// Synchronization costs a machine charges for its barriers and waits.
struct SyncModel {
  uint64_t barrierCycles = 0;
  uint64_t waitCycles = 0;
};

struct MachineModel {
  uint32_t schemaMajor = kSupportedSchemaMajor;
  /// The declared schema minor, 0 when the file wrote a bare
  /// `llk.machine.v2`. Recorded so callers can tell which revision a profile
  /// targets; it is schema metadata, not machine content, so it does not enter
  /// the content hash.
  uint32_t schemaMinor = kSupportedSchemaMinor;
  std::string target;
  std::string description;

  /// Clock rate, absent when the profile does not model one. A cycle estimate
  /// is still meaningful without it; a *nanosecond* estimate is not, which is
  /// why this is optional rather than zero.
  std::optional<uint64_t> clockHz;
  /// Host workers available to the target. Defaults to 1: a machine that does
  /// not model thread-level parallelism still executes somewhere.
  uint32_t workerThreads = 1;
  SyncModel sync;

  std::vector<ExecutorNode> executors;
  std::vector<MemoryNode> memories;
  std::vector<ComputeNode> computes;
  std::vector<TransferEngineNode> transferEngines;
  std::vector<LinkEdge> links;

  /// Stable hash of the normalized model (see computeContentHash).
  uint64_t contentHash = 0;

  const ExecutorNode *findExecutor(llvm::StringRef id) const;
  const MemoryNode *findMemory(llvm::StringRef id) const;
  const ComputeNode *findCompute(llvm::StringRef id) const;
  const TransferEngineNode *findTransferEngine(llvm::StringRef id) const;
  const LinkEdge *findLink(llvm::StringRef id) const;

  /// True when `nodeId` is `ancestorId` or a descendant of it along the
  /// containment chain.
  bool isWithin(llvm::StringRef nodeId, llvm::StringRef ancestorId) const;

  /// True when an abstract owner matches `executorId`: the executor's kind is
  /// `ownerKind`, or it declares `ownerKind` in `refines`.
  bool ownerMatches(llvm::StringRef ownerKind,
                    llvm::StringRef executorId) const;

  /// True when `executorId` can address `memoryId` (design §11.2 `dominates`).
  bool isVisible(llvm::StringRef memoryId, llvm::StringRef executorId) const;

  /// Elements per instruction for `elementType` on the first capability of
  /// `computeKind`, or nullopt when no such capability or dtype is modelled.
  std::optional<int64_t> lanesFor(llvm::StringRef computeKind,
                                  llvm::StringRef elementType) const;

  /// The first memory of `kind`, in declaration order, or null. Profiles
  /// describe memory *spaces* (`sram`), while nodes are instances (`sram.0`).
  const MemoryNode *findMemoryOfKind(llvm::StringRef kind) const;

  /// How many executors the machine offers for an owner kind, counting each
  /// executor's own concurrency. This is what a "worker count" means once
  /// executors are concrete nodes rather than a kind with a multiplicity.
  uint32_t ownerCount(llvm::StringRef ownerKind) const;

  /// True when `ownerKind` names at least one executor.
  bool hasOwnerKind(llvm::StringRef ownerKind) const {
    return ownerCount(ownerKind) > 0;
  }

  /// The first directed link from a memory of `sourceKind` to one of
  /// `destinationKind`, or null. Callers name memory *spaces* (`dram` to
  /// `sram`); links join concrete nodes, so this resolves the pair.
  const LinkEdge *findLinkByKinds(llvm::StringRef sourceKind,
                                  llvm::StringRef destinationKind) const;

  /// Every compute capability of `kind`, in declaration order. The simulator
  /// asks for all matrix engines, then picks one; a placement asks for the
  /// ones attached to one executor.
  std::vector<const ComputeNode *> computesOfKind(llvm::StringRef kind) const;

  /// Total transfer engines the machine offers, summed over the nodes.
  uint32_t transferEngineCount() const;

  /// The first transfer engine, or null. A machine that models one engine
  /// class answers per-engine questions (setup cost, outstanding limit) from
  /// it; a machine with several is expected to be asked per node.
  const TransferEngineNode *primaryTransferEngine() const;

  /// Compute capabilities and transfer engines directly attached to
  /// `executorId`, in declaration order.
  std::vector<const ComputeNode *>
  computesFor(llvm::StringRef executorId) const;
  std::vector<const TransferEngineNode *>
  transferEnginesFor(llvm::StringRef executorId) const;
};

/// Deterministic rendering of the model, nodes sorted by id within each
/// section, so YAML key order cannot affect it.
std::string canonicalMachineString(const MachineModel &model);

/// Stable content hash of `model`; keys the latency cache (design §17.4).
uint64_t computeContentHash(const MachineModel &model);

/// Checks the invariants a loaded model must satisfy: supported schema major,
/// non-empty target, globally unique non-empty ids, declared and acyclic
/// containment, known kinds, and links/attachments that reference existing
/// nodes. Returns the first violation found, in declaration order, so
/// diagnostics are deterministic.
llvm::Error verifyMachineModel(const MachineModel &model);

} // namespace mlir::llk::machine

#endif // LLK_MACHINE_MACHINEMODEL_H
