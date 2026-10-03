//===- MachineModel.cpp - Versioned machine topology ----------------------===//

#include "LLK/Machine/MachineModel.h"

#include "MachineHash.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"

#include <algorithm>

namespace mlir::llk::machine {

namespace {

template <typename NodeT>
const NodeT *findById(const std::vector<NodeT> &nodes, llvm::StringRef id) {
  for (const NodeT &node : nodes)
    if (node.id == id)
      return &node;
  return nullptr;
}

std::string joinStrings(const std::vector<std::string> &parts,
                        llvm::StringRef separator) {
  std::string out;
  for (const std::string &part : parts) {
    if (!out.empty())
      out += separator;
    out += part;
  }
  return out;
}

std::string joinNumbers(const std::vector<int64_t> &values) {
  std::string out;
  for (int64_t value : values) {
    if (!out.empty())
      out += ',';
    out += std::to_string(value);
  }
  return out;
}

std::string optionalString(const std::optional<std::string> &value) {
  return value ? *value : "<none>";
}

template <typename T>
std::string optionalNumber(const std::optional<T> &value) {
  return value ? std::to_string(*value) : "<none>";
}

std::string renderExecutor(const ExecutorNode &node) {
  std::vector<std::string> refines(node.refines);
  llvm::sort(refines);
  std::string out = "executor|id=";
  out += node.id;
  out += "|kind=";
  out += node.kind;
  out += "|parent=";
  out += optionalString(node.parent);
  out += "|coordinates=";
  out += joinNumbers(node.coordinates);
  out += "|concurrency=";
  out += std::to_string(node.concurrency);
  out += "|refines=";
  out += joinStrings(refines, ",");
  return out;
}

std::string renderMemory(const MemoryNode &node) {
  std::vector<std::string> layouts(node.supportedLayouts);
  llvm::sort(layouts);
  std::string out = "memory|id=";
  out += node.id;
  out += "|kind=";
  out += node.kind;
  out += "|visible_from=";
  out += node.visibleFrom;
  out += "|capacity=";
  out += std::to_string(node.capacityBytes);
  out += "|alignment=";
  out += std::to_string(node.alignmentBytes);
  out += "|layouts=";
  out += joinStrings(layouts, ",");
  out += "|banks=";
  out += optionalNumber(node.banks);
  return out;
}

std::string renderCompute(const ComputeNode &node) {
  std::vector<std::string> layouts(node.supportedLayouts);
  std::vector<std::string> elementTypes(node.elementTypes);
  llvm::sort(layouts);
  llvm::sort(elementTypes);
  std::string out = "compute|id=";
  out += node.id;
  out += "|kind=";
  out += node.kind;
  out += "|attached_to=";
  out += node.attachedTo;
  out += "|element_types=";
  out += joinStrings(elementTypes, ",");
  out += "|layouts=";
  out += joinStrings(layouts, ",");
  out += "|shapes=";
  for (const std::vector<int64_t> &shape : node.shapes) {
    out += '[';
    out += joinNumbers(shape);
    out += ']';
  }
  out += "|issue=";
  out += std::to_string(node.issueCycles);
  out += "|latency=";
  out += std::to_string(node.latencyCycles);
  out += "|throughput=";
  out += node.throughputPerCycle ? formatDouble(*node.throughputPerCycle)
                                 : "<none>";
  out += "|concurrency=";
  out += std::to_string(node.concurrency);
  return out;
}

std::string renderTransferEngine(const TransferEngineNode &node) {
  std::string out = "transfer|id=";
  out += node.id;
  out += "|kind=";
  out += node.kind;
  out += "|attached_to=";
  out += node.attachedTo;
  out += "|count=";
  out += std::to_string(node.count);
  out += "|max_outstanding=";
  out += std::to_string(node.maxOutstanding);
  return out;
}

std::string renderLink(const LinkEdge &node) {
  std::vector<std::string> engines(node.transferEngines);
  llvm::sort(engines);
  std::string out = "link|id=";
  out += node.id;
  out += "|source=";
  out += node.source;
  out += "|destination=";
  out += node.destination;
  out += "|bandwidth=";
  out += formatDouble(node.bandwidthBytesPerCycle);
  out += "|latency=";
  out += std::to_string(node.latencyCycles);
  out += "|transaction=";
  out += std::to_string(node.transactionBytes);
  out += "|engines=";
  out += joinStrings(engines, ",");
  out += "|concurrency=";
  out += std::to_string(node.concurrency);
  return out;
}

/// Renders a section in id order, so declaration order in the source file does
/// not affect the canonical string.
template <typename NodeT, typename RenderFn>
void renderSection(std::string &out, const std::vector<NodeT> &nodes,
                   RenderFn render) {
  std::vector<const NodeT *> ordered;
  ordered.reserve(nodes.size());
  for (const NodeT &node : nodes)
    ordered.push_back(&node);
  llvm::sort(ordered, [](const NodeT *lhs, const NodeT *rhs) {
    return lhs->id < rhs->id;
  });
  for (const NodeT *node : ordered) {
    out += render(*node);
    out += '\n';
  }
}

} // namespace

const ExecutorNode *MachineModel::findExecutor(llvm::StringRef id) const {
  return findById(executors, id);
}

const MemoryNode *MachineModel::findMemory(llvm::StringRef id) const {
  return findById(memories, id);
}

const ComputeNode *MachineModel::findCompute(llvm::StringRef id) const {
  return findById(computes, id);
}

const TransferEngineNode *
MachineModel::findTransferEngine(llvm::StringRef id) const {
  return findById(transferEngines, id);
}

const LinkEdge *MachineModel::findLink(llvm::StringRef id) const {
  return findById(links, id);
}

bool MachineModel::isWithin(llvm::StringRef nodeId,
                            llvm::StringRef ancestorId) const {
  const ExecutorNode *current = findExecutor(nodeId);
  llvm::SmallPtrSet<const ExecutorNode *, 8> visited;
  while (current) {
    if (current->id == ancestorId)
      return true;
    if (!current->parent)
      return false;
    // A malformed in-memory model must not spin forever; verification rejects
    // cycles, but queries stay safe on unverified input.
    if (!visited.insert(current).second)
      return false;
    current = findExecutor(*current->parent);
  }
  return false;
}

bool MachineModel::ownerMatches(llvm::StringRef ownerKind,
                                llvm::StringRef executorId) const {
  const ExecutorNode *executor = findExecutor(executorId);
  if (!executor)
    return false;
  return executor->kind == ownerKind ||
         llvm::is_contained(executor->refines, ownerKind);
}

bool MachineModel::isVisible(llvm::StringRef memoryId,
                             llvm::StringRef executorId) const {
  const MemoryNode *memory = findMemory(memoryId);
  if (!memory)
    return false;
  return isWithin(executorId, memory->visibleFrom);
}

std::vector<const ComputeNode *>
MachineModel::computesFor(llvm::StringRef executorId) const {
  std::vector<const ComputeNode *> result;
  for (const ComputeNode &node : computes)
    if (node.attachedTo == executorId)
      result.push_back(&node);
  return result;
}

std::vector<const TransferEngineNode *>
MachineModel::transferEnginesFor(llvm::StringRef executorId) const {
  std::vector<const TransferEngineNode *> result;
  for (const TransferEngineNode &node : transferEngines)
    if (node.attachedTo == executorId)
      result.push_back(&node);
  return result;
}

std::string canonicalMachineString(const MachineModel &model) {
  std::string out = "schema=";
  out += std::to_string(model.schemaMajor);
  out += "\ntarget=";
  out += model.target;
  out += "\ndescription=";
  out += model.description;
  out += '\n';
  renderSection(out, model.executors, renderExecutor);
  renderSection(out, model.memories, renderMemory);
  renderSection(out, model.computes, renderCompute);
  renderSection(out, model.transferEngines, renderTransferEngine);
  renderSection(out, model.links, renderLink);
  return out;
}

uint64_t computeContentHash(const MachineModel &model) {
  return stableHash(canonicalMachineString(model));
}

} // namespace mlir::llk::machine
