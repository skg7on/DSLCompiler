//===- WorkloadGraph.h - Target-independent work graph from Micro-IR ------===//
//
// Part of the target-independent mapping core (issue #80, epic #67 D1).
//
// The mapping engine does not pattern-match Micro-IR directly. It first
// extracts a workload graph: the operations that need a target implementation
// (compute and data-movement ops), joined by the values that cross between
// them. Logical ops (`tile_view`, `tile_partition`) are transparent -- they
// name a relationship, not work -- so a value read through them resolves to
// the same workload value as its source. Structural ops (`for`,
// `spatial_for`, `pipeline`) and scheduling ops (`alloc`, `wait`) are not
// nodes: iteration and synchronization are properties of a chosen plan, not
// units of work to place.
//
// Everything is keyed by stable ids, never by MLIR pointers, so a graph can be
// serialized, compared, and re-derived. `finalize()` canonicalizes by content,
// which is what makes a graph assembled in any insertion order hash and
// serialize identically.
//
//===----------------------------------------------------------------------===//

#ifndef LLK_MAPPING_WORKLOADGRAPH_H
#define LLK_MAPPING_WORKLOADGRAPH_H

#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/OperationSupport.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <optional>
#include <string>

namespace mlir::llk::mapping {

using WorkloadNodeId = uint32_t;
using WorkloadValueId = uint32_t;

/// One value in the workload graph: a tensor or tile an operation reads from
/// or writes to. External values have no producing node in the graph (function
/// inputs, allocations, captured constants).
struct WorkloadValue {
  WorkloadValueId id = 0;
  Type type;
  std::string name;
  bool external = false;
};

/// A value as seen from one node. `accessMap` is the affine relationship
/// between the node's iteration space and this value, when known. Extraction
/// populates it for *input* ports whose operand chain states one; output ports
/// and chains that state none are left empty, to be refined by later
/// rule/layout stages.
struct WorkloadPort {
  WorkloadValueId value = 0;
  Type type;
  std::optional<AffineMap> accessMap;
};

/// Which side of a node a port sits on.
enum class PortDirection { Input, Output };

/// One operand or result occurrence on a node: the node it belongs to, the side
/// it sits on, and its position within that side. A `PortRef` names the
/// occurrence, not the SSA value it carries -- a single value used through two
/// operand ports is two refs -- so it is what tells repeated uses apart. Refs
/// name finalized node ids, so resolve them only after `finalize()`.
struct PortRef {
  WorkloadNodeId node = 0;
  PortDirection direction = PortDirection::Input;
  uint32_t index = 0;
  bool operator==(const PortRef &) const = default;
};

/// One unit of work to implement. `coveredNodes` in a mapping candidate names
/// these ids.
///
/// `opName` is the operation's string name rather than an `OperationName`:
/// the graph is context-free and serialized, and an interned `OperationName`
/// would tie it to the MLIR context that built it. Classification and
/// comparison all work on strings anyway.
struct WorkloadNode {
  WorkloadNodeId id = 0;
  std::string opName;
  llvm::SmallVector<WorkloadPort> inputs;
  llvm::SmallVector<WorkloadPort> outputs;
  DictionaryAttr attributes;
  /// Program position, used only to break content-key ties during
  /// canonicalization; it never survives as an ordering key on its own.
  uint32_t sourceOrdinal = 0;
};

/// A target-independent view of one concrete `micro.kernel`'s work.
class WorkloadGraph {
public:
  /// Appends a value and returns its temporary id. Ids are only meaningful
  /// until `finalize()` reassigns them canonically.
  WorkloadValueId addValue(WorkloadValue value);

  /// Appends a node. Any `id` already set is ignored; `finalize()` assigns it.
  void addNode(WorkloadNode node);

  /// Sorts nodes and values by canonical content key and reassigns all ids, so
  /// the same graph built in any insertion order finalizes to identical ids
  /// and identical `canonicalString()` output. When `valueRemap` is given it
  /// records each temporary value id's final id.
  void finalize(
      llvm::DenseMap<WorkloadValueId, WorkloadValueId> *valueRemap = nullptr);

  llvm::ArrayRef<WorkloadNode> getNodes() const { return nodes; }
  llvm::ArrayRef<WorkloadValue> getValues() const { return values; }

  const WorkloadNode *findNode(WorkloadNodeId id) const;
  const WorkloadValue *findValue(WorkloadValueId id) const;

  /// Deterministic rendering of the finalized graph, used to compare graphs
  /// and to key hashes.
  std::string canonicalString() const;

private:
  llvm::SmallVector<WorkloadNode> nodes;
  llvm::SmallVector<WorkloadValue> values;
};

/// The port a reference names on a finalized graph, or null when the node does
/// not exist or `index` is out of range for that direction.
const WorkloadPort *lookupPort(const WorkloadGraph &graph, const PortRef &ref);

/// Deterministic rendering of a port reference (`node=7,input=0`). Used as the
/// endpoint identity in candidate and connection content keys.
std::string canonicalPortRefString(const PortRef &p);

/// True for ops that become workload nodes: concrete execution and movement
/// ops that a target must implement.
bool isWorkloadNodeOp(OperationName op);

/// Same test by operation name, for callers that hold only a string (a mapping
/// rule's `match` clause, for instance).
bool isWorkloadNodeOp(llvm::StringRef name);

/// True for logical ops that are folded into their consumer's port rather than
/// becoming nodes.
bool isTransparentWorkloadOp(OperationName op);

/// The correspondence between a graph and the IR it was extracted from.
///
/// A `WorkloadGraph` is deliberately context-free -- it holds stable ids, not
/// MLIR pointers -- so anything that has to act on the IR again (the plan
/// binder, for one) needs this side table. Only extraction can build it: the
/// ids are assigned and then canonicalized there, so it cannot be recovered
/// from a finished graph.
struct WorkloadGraphBinding {
  llvm::DenseMap<WorkloadNodeId, Operation *> nodeOps;
  llvm::DenseMap<WorkloadValueId, Value> values;

  /// The op a node came from, or null when the id is unknown.
  Operation *opFor(WorkloadNodeId node) const;
  /// The SSA value a workload value came from, or a null Value.
  Value valueFor(WorkloadValueId value) const;
};

/// Extracts the workload graph from a `micro.kernel`. Fails when `kernel` is
/// not a `micro.kernel` op. When `binding` is given, it is filled with the
/// node-to-op and value-to-SSA correspondence for this extraction.
llvm::Expected<WorkloadGraph>
extractWorkloadGraph(Operation *kernel,
                     WorkloadGraphBinding *binding = nullptr);

} // namespace mlir::llk::mapping

#endif // LLK_MAPPING_WORKLOADGRAPH_H
