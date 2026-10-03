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
// Node `kind` values are the Micro dialect vocabulary: executor kinds are
// `micro::Owner` names, memory kinds are `micro::MemorySpace` names. The model
// stores them as strings so it carries no dependency on the dialect headers at
// its public surface; the loader validates them against the dialect enums.
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

/// The schema string a v2 file declares.
inline constexpr llvm::StringLiteral kSchemaName = "llk.machine.v2";

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
};

/// A resource that moves data over links, attached to an executor.
struct TransferEngineNode {
  std::string id;
  std::string kind;
  std::string attachedTo;
  uint32_t count = 1;
  uint32_t maxOutstanding = 1;
};

/// A directed data path between two memories.
struct LinkEdge {
  std::string id;
  std::string source;
  std::string destination;
  double bandwidthBytesPerCycle = 0.0;
  uint64_t latencyCycles = 0;
  uint64_t transactionBytes = 1;
  std::vector<std::string> transferEngines;
  uint32_t concurrency = 1;
};

/// Synchronization costs a machine charges for its barriers and waits.
struct SyncModel {
  uint64_t barrierCycles = 0;
  uint64_t waitCycles = 0;
};

struct MachineModel {
  uint32_t schemaMajor = kSupportedSchemaMajor;
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
