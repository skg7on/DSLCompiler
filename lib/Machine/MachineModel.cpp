//===- MachineModel.cpp - Versioned machine topology ----------------------===//

#include "LLK/Machine/MachineModel.h"

#include "LLK/Dialect/Micro/MicroEnums.h"

#include "MachineHash.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/ErrorHandling.h"

namespace mlir::llk::machine {

llvm::StringRef stringifySchedulingClass(SchedulingClass value) {
  switch (value) {
  case SchedulingClass::InOrder:
    return "in_order";
  case SchedulingClass::OutOfOrder:
    return "out_of_order";
  }
  llvm_unreachable("unhandled SchedulingClass");
}

std::optional<SchedulingClass> symbolizeSchedulingClass(llvm::StringRef text) {
  if (text == "in_order")
    return SchedulingClass::InOrder;
  if (text == "out_of_order")
    return SchedulingClass::OutOfOrder;
  return std::nullopt;
}

llvm::StringRef stringifyLinkDirectionality(LinkDirectionality value) {
  switch (value) {
  case LinkDirectionality::Unidirectional:
    return "unidirectional";
  case LinkDirectionality::Bidirectional:
    return "bidirectional";
  }
  llvm_unreachable("unhandled LinkDirectionality");
}

std::optional<LinkDirectionality>
symbolizeLinkDirectionality(llvm::StringRef text) {
  if (text == "unidirectional")
    return LinkDirectionality::Unidirectional;
  if (text == "bidirectional")
    return LinkDirectionality::Bidirectional;
  return std::nullopt;
}

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
  out += "|scheduling=";
  out += stringifySchedulingClass(node.schedulingClass);
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
  out += "|bandwidth=";
  out += formatDouble(node.bandwidthBytesPerCycle);
  out += "|latency=";
  out += std::to_string(node.latencyCycles);
  out += "|transaction=";
  out += optionalNumber(node.transactionBytes);
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
  // std::map iterates in key order, so the rendering is already canonical.
  out += "|lanes=";
  for (const auto &lane : node.lanes) {
    out += lane.first;
    out += ':';
    out += std::to_string(lane.second);
    out += ',';
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
  out += "|accumulator_dtypes=";
  std::vector<std::string> accumulators(node.accumulatorDTypes);
  llvm::sort(accumulators);
  out += joinStrings(accumulators, ",");
  out += "|occupancy=";
  out += optionalNumber(node.occupancyLimit);
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
  out += "|setup=";
  out += std::to_string(node.setupCycles);
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
  out += "|directionality=";
  out += stringifyLinkDirectionality(node.directionality);
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

const LinkEdge *
MachineModel::findLinkByKinds(llvm::StringRef sourceKind,
                              llvm::StringRef destinationKind) const {
  for (const LinkEdge &link : links) {
    const MemoryNode *source = findMemory(link.source);
    const MemoryNode *destination = findMemory(link.destination);
    if (source && destination && source->kind == sourceKind &&
        destination->kind == destinationKind)
      return &link;
  }
  return nullptr;
}

std::vector<const ComputeNode *>
MachineModel::computesOfKind(llvm::StringRef kind) const {
  std::vector<const ComputeNode *> result;
  for (const ComputeNode &node : computes)
    if (node.kind == kind)
      result.push_back(&node);
  return result;
}

uint32_t MachineModel::transferEngineCount() const {
  uint32_t count = 0;
  for (const TransferEngineNode &engine : transferEngines)
    count += engine.count;
  return count;
}

const TransferEngineNode *MachineModel::primaryTransferEngine() const {
  return transferEngines.empty() ? nullptr : &transferEngines.front();
}

const MemoryNode *MachineModel::findMemoryOfKind(llvm::StringRef kind) const {
  for (const MemoryNode &node : memories)
    if (node.kind == kind)
      return &node;
  return nullptr;
}

uint32_t MachineModel::ownerCount(llvm::StringRef ownerKind) const {
  uint32_t count = 0;
  for (const ExecutorNode &executor : executors)
    if (ownerMatches(ownerKind, executor.id))
      count += std::max(1u, executor.concurrency);
  return count;
}

std::optional<int64_t>
MachineModel::lanesFor(llvm::StringRef computeKind,
                       llvm::StringRef elementType) const {
  for (const ComputeNode &node : computes) {
    if (node.kind != computeKind)
      continue;
    auto it = node.lanes.find(elementType.str());
    if (it != node.lanes.end())
      return it->second;
    // The first capability of this kind is the one queried; if it does not
    // model the dtype, neither does the kind.
    return std::nullopt;
  }
  return std::nullopt;
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
  out += "\nclock_hz=";
  out += model.clockHz ? std::to_string(*model.clockHz) : "<none>";
  out += "\nworker_threads=";
  out += std::to_string(model.workerThreads);
  out += "\nsync=barrier:";
  out += std::to_string(model.sync.barrierCycles);
  out += ",wait:";
  out += std::to_string(model.sync.waitCycles);
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

namespace {

llvm::Error invalid(const std::string &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(), message);
}

bool isKnownOwnerKind(llvm::StringRef kind) {
  return micro::symbolizeOwner(kind).has_value();
}

bool isKnownMemoryKind(llvm::StringRef kind) {
  return micro::symbolizeMemorySpace(kind).has_value();
}

/// Compute capabilities and transfer resources are `micro::Owner` vocabulary
/// (design §11.5): a kernel maps onto a `matrix_engine` or `vector_engine`
/// owner, and a `dma` owner moves its data. The subset a machine may declare is
/// written with the enum's own enumerators, so a rename in MicroEnums.h breaks
/// this code at compile time instead of letting a string list drift. Execution
/// scopes such as `core` are also owners, but they are not capabilities a
/// machine attaches, so they are excluded here.
bool isComputeOwner(micro::Owner owner) {
  return owner == micro::Owner::matrix_engine ||
         owner == micro::Owner::vector_engine;
}

bool isKnownComputeKind(llvm::StringRef kind) {
  std::optional<micro::Owner> owner = micro::symbolizeOwner(kind);
  return owner && isComputeOwner(*owner);
}

bool isKnownTransferKind(llvm::StringRef kind) {
  std::optional<micro::Owner> owner = micro::symbolizeOwner(kind);
  return owner && *owner == micro::Owner::dma;
}

} // namespace

llvm::Error verifyMachineModel(const MachineModel &model) {
  if (model.schemaMajor != kSupportedSchemaMajor)
    return invalid("schema: unsupported major " +
                   std::to_string(model.schemaMajor) + ", expected " +
                   std::to_string(kSupportedSchemaMajor));
  if (model.target.empty())
    return invalid("target: must not be empty");
  // A declared clock of zero would make every nanosecond estimate infinite;
  // an *absent* clock is fine -- cycles are still meaningful.
  if (model.clockHz && *model.clockHz == 0)
    return invalid("clock_hz: must be positive when declared");
  if (model.workerThreads == 0)
    return invalid("worker_threads: must be positive");

  // Ids share one namespace across every node kind, so a memory and an
  // executor can never collide silently.
  llvm::StringSet<> seenIds;
  auto checkId = [&](const std::string &id,
                     const std::string &path) -> llvm::Error {
    if (id.empty())
      return invalid(path + ": id must not be empty");
    if (!seenIds.insert(id).second)
      return invalid(path + ": duplicate id '" + id + "'");
    return llvm::Error::success();
  };
  for (size_t i = 0; i < model.executors.size(); ++i)
    if (auto error = checkId(model.executors[i].id,
                             "executors[" + std::to_string(i) + "]"))
      return error;
  for (size_t i = 0; i < model.memories.size(); ++i)
    if (auto error = checkId(model.memories[i].id,
                             "memories[" + std::to_string(i) + "]"))
      return error;
  for (size_t i = 0; i < model.computes.size(); ++i)
    if (auto error =
            checkId(model.computes[i].id, "compute[" + std::to_string(i) + "]"))
      return error;
  for (size_t i = 0; i < model.transferEngines.size(); ++i)
    if (auto error = checkId(model.transferEngines[i].id,
                             "transfer_engines[" + std::to_string(i) + "]"))
      return error;
  for (size_t i = 0; i < model.links.size(); ++i)
    if (auto error =
            checkId(model.links[i].id, "links[" + std::to_string(i) + "]"))
      return error;

  for (size_t i = 0; i < model.executors.size(); ++i) {
    const ExecutorNode &executor = model.executors[i];
    std::string path = "executors[" + std::to_string(i) + "]";
    if (!isKnownOwnerKind(executor.kind))
      return invalid(path + ".kind: unknown owner kind '" + executor.kind +
                     "'");
    for (const std::string &refined : executor.refines)
      if (!isKnownOwnerKind(refined))
        return invalid(path + ".refines: unknown owner kind '" + refined + "'");
    if (executor.concurrency == 0)
      return invalid(path + ".concurrency: must be positive");
    if (executor.parent && !model.findExecutor(*executor.parent))
      return invalid(path + ".parent: unknown executor '" + *executor.parent +
                     "'");
  }

  // Containment must be acyclic: walk each executor's parent chain.
  for (size_t i = 0; i < model.executors.size(); ++i) {
    const ExecutorNode *current = &model.executors[i];
    llvm::SmallPtrSet<const ExecutorNode *, 8> visited;
    while (current) {
      if (!visited.insert(current).second)
        return invalid("executors[" + std::to_string(i) +
                       "].parent: containment cycle");
      if (!current->parent)
        break;
      current = model.findExecutor(*current->parent);
    }
  }

  for (size_t i = 0; i < model.memories.size(); ++i) {
    const MemoryNode &memory = model.memories[i];
    std::string path = "memories[" + std::to_string(i) + "]";
    if (!isKnownMemoryKind(memory.kind))
      return invalid(path + ".kind: unknown memory kind '" + memory.kind + "'");
    if (!model.findExecutor(memory.visibleFrom))
      return invalid(path + ".visible_from: unknown executor '" +
                     memory.visibleFrom + "'");
    if (memory.capacityBytes == 0)
      return invalid(path + ".capacity_bytes: must be positive");
    if (memory.alignmentBytes == 0)
      return invalid(path + ".alignment_bytes: must be positive");
    // A zero-granule transaction can never move anything; absent is fine, and
    // means the profile does not model a granularity (fall back to alignment).
    if (memory.transactionBytes && *memory.transactionBytes == 0)
      return invalid(path +
                     ".transaction_bytes: must be positive when declared");
  }

  for (size_t i = 0; i < model.computes.size(); ++i) {
    const ComputeNode &compute = model.computes[i];
    std::string path = "compute[" + std::to_string(i) + "]";
    if (!isKnownComputeKind(compute.kind))
      return invalid(path + ".kind: unknown capability kind '" + compute.kind +
                     "'");
    if (!model.findExecutor(compute.attachedTo))
      return invalid(path + ".attached_to: unknown executor '" +
                     compute.attachedTo + "'");
    if (compute.concurrency == 0)
      return invalid(path + ".concurrency: must be positive");
    // An occupancy limit of zero would forbid all resident work; absent is
    // fine, and means the capability is bounded only by its slots.
    if (compute.occupancyLimit && *compute.occupancyLimit == 0)
      return invalid(path + ".occupancy_limit: must be positive when declared");
    if (compute.shapes.empty())
      return invalid(path + ".shapes: must not be empty");
    for (const std::vector<int64_t> &shape : compute.shapes) {
      if (shape.empty())
        return invalid(path + ".shapes: shape must not be empty");
      for (int64_t extent : shape)
        if (extent <= 0)
          return invalid(path + ".shapes: extent must be positive");
    }
    for (const auto &lane : compute.lanes)
      if (lane.second <= 0)
        return invalid(path + ".lanes['" + lane.first +
                       "']: lane count must be positive");
  }

  for (size_t i = 0; i < model.transferEngines.size(); ++i) {
    const TransferEngineNode &engine = model.transferEngines[i];
    std::string path = "transfer_engines[" + std::to_string(i) + "]";
    if (!isKnownTransferKind(engine.kind))
      return invalid(path + ".kind: unknown transfer kind '" + engine.kind +
                     "'");
    if (!model.findExecutor(engine.attachedTo))
      return invalid(path + ".attached_to: unknown executor '" +
                     engine.attachedTo + "'");
    if (engine.count == 0)
      return invalid(path + ".count: must be positive");
    if (engine.maxOutstanding == 0)
      return invalid(path + ".max_outstanding: must be positive");
  }

  for (size_t i = 0; i < model.links.size(); ++i) {
    const LinkEdge &link = model.links[i];
    std::string path = "links[" + std::to_string(i) + "]";
    if (!model.findMemory(link.source))
      return invalid(path + ".source: unknown memory '" + link.source + "'");
    if (!model.findMemory(link.destination))
      return invalid(path + ".destination: unknown memory '" +
                     link.destination + "'");
    if (link.bandwidthBytesPerCycle <= 0.0)
      return invalid(path + ".bandwidth_bytes_per_cycle: must be positive");
    if (link.transactionBytes == 0)
      return invalid(path + ".transaction_bytes: must be positive");
    if (link.concurrency == 0)
      return invalid(path + ".concurrency: must be positive");
    for (const std::string &engine : link.transferEngines)
      if (!model.findTransferEngine(engine))
        return invalid(path + ".transfer_engines: unknown transfer engine '" +
                       engine + "'");
  }

  return llvm::Error::success();
}

} // namespace mlir::llk::machine
