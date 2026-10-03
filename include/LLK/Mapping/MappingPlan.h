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

/// A layout conversion over a value, expressed as an affine relationship so
/// equivalence and composition use MLIR's canonicalization.
struct LayoutTransform {
  std::string srcLayout;
  std::string dstLayout;
  AffineMap map;
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
};

/// One selected connection, with the route it takes.
struct PlanConnection {
  ConnectionId id = 0;
  WorkloadValueId value = 0;
  ConnectionKind kind = ConnectionKind::Direct;
  llvm::SmallVector<MemoryNodeId> route;
  llvm::SmallVector<ExecutorId> engines;
  std::optional<LayoutTransform> transform;
};

/// A complete executable proposal covering every required node.
struct CoveringPlan {
  PlanId id = 0;
  uint64_t sourceBindingHash = 0;
  llvm::SmallVector<InstanceId> instances;
  llvm::SmallVector<ConnectionId> connections;
  /// The same selections, resolved: which node each instance covers and how
  /// each connection runs. Sorted by node / connection id.
  llvm::SmallVector<PlanPlacement> placements;
  llvm::SmallVector<PlanConnection> connectionPlans;
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
