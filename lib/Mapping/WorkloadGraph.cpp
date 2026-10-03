//===- WorkloadGraph.cpp - Target-independent work graph from Micro-IR ----===//

#include "LLK/Mapping/WorkloadGraph.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"

#include <functional>
#include <numeric>

namespace mlir::llk::mapping {

namespace {

/// The ops that become workload nodes: concrete execution and movement that a
/// target must implement. Kept as data so adding an op is one line, and so the
/// set is greppable against the dialect's op list.
constexpr llvm::StringLiteral kNodeOps[] = {
    "micro.mma",        "micro.vector",          "micro.reduce",
    "micro.async_copy", "micro.tile_async_copy", "micro.store",
    "micro.tile_store"};

/// Logical ops folded into their consumer rather than becoming nodes.
constexpr llvm::StringLiteral kTransparentOps[] = {"micro.tile_view",
                                                   "micro.tile_partition"};

std::string typeString(Type type) {
  std::string text;
  llvm::raw_string_ostream stream(text);
  type.print(stream);
  return stream.str();
}

std::string attributeString(Attribute attribute) {
  std::string text;
  llvm::raw_string_ostream stream(text);
  attribute.print(stream);
  return stream.str();
}

/// Async tokens carry synchronization, not data, so they are not workload
/// ports. The generated `AsyncTokenType` class is not part of the dialect's
/// public headers, so the type is recognised by its printed form.
bool isAsyncToken(Type type) {
  return typeString(type) == "!micro.async_token";
}

std::string nameFor(Value value) {
  if (auto argument = dyn_cast<BlockArgument>(value))
    return "arg" + std::to_string(argument.getArgNumber());
  Operation *definingOp = value.getDefiningOp();
  if (!definingOp)
    return "value";
  std::string name = definingOp->getName().getStringRef().str();
  name += ".result";
  name += std::to_string(cast<OpResult>(value).getResultNumber());
  return name;
}

/// Content key used to order nodes canonically. Types and attributes are
/// content; `sourceOrdinal` only breaks ties between otherwise identical
/// nodes, so identical content never depends on insertion sequence.
std::string nodeContentKey(const WorkloadNode &node) {
  std::string key = node.opName;
  key += "|in:";
  for (const WorkloadPort &port : node.inputs) {
    key += typeString(port.type);
    key += ',';
  }
  key += "|out:";
  for (const WorkloadPort &port : node.outputs) {
    key += typeString(port.type);
    key += ',';
  }
  key += "|attr:";
  if (node.attributes)
    key += attributeString(node.attributes);
  key += "|ordinal:";
  key += std::to_string(node.sourceOrdinal);
  return key;
}

} // namespace

bool isWorkloadNodeOp(OperationName op) {
  return isWorkloadNodeOp(op.getStringRef());
}

bool isWorkloadNodeOp(llvm::StringRef name) {
  return llvm::is_contained(kNodeOps, name);
}

bool isTransparentWorkloadOp(OperationName op) {
  llvm::StringRef name = op.getStringRef();
  return llvm::is_contained(kTransparentOps, name);
}

WorkloadValueId WorkloadGraph::addValue(WorkloadValue value) {
  WorkloadValueId id = static_cast<WorkloadValueId>(values.size());
  value.id = id;
  values.push_back(std::move(value));
  return id;
}

void WorkloadGraph::addNode(WorkloadNode node) {
  nodes.push_back(std::move(node));
}

const WorkloadNode *WorkloadGraph::findNode(WorkloadNodeId id) const {
  if (id >= nodes.size())
    return nullptr;
  return &nodes[id];
}

const WorkloadValue *WorkloadGraph::findValue(WorkloadValueId id) const {
  if (id >= values.size())
    return nullptr;
  return &values[id];
}

void WorkloadGraph::finalize(
    llvm::DenseMap<WorkloadValueId, WorkloadValueId> *valueRemapOut) {
  llvm::SmallVector<std::string> nodeKeys;
  nodeKeys.reserve(nodes.size());
  for (const WorkloadNode &node : nodes)
    nodeKeys.push_back(nodeContentKey(node));

  // Which node produced each temporary value id.
  llvm::DenseMap<WorkloadValueId, std::pair<unsigned, unsigned>> producer;
  for (unsigned nodeIndex = 0; nodeIndex < nodes.size(); ++nodeIndex)
    for (unsigned resultIndex = 0;
         resultIndex < nodes[nodeIndex].outputs.size(); ++resultIndex)
      producer[nodes[nodeIndex].outputs[resultIndex].value] = {nodeIndex,
                                                               resultIndex};

  llvm::SmallVector<std::string> valueKeys;
  valueKeys.reserve(values.size());
  for (unsigned valueIndex = 0; valueIndex < values.size(); ++valueIndex) {
    auto it = producer.find(static_cast<WorkloadValueId>(valueIndex));
    if (!values[valueIndex].external && it != producer.end()) {
      std::string key = "def:";
      key += nodeKeys[it->second.first];
      key += '#';
      key += std::to_string(it->second.second);
      valueKeys.push_back(std::move(key));
    } else {
      valueKeys.push_back("ext:" + values[valueIndex].name);
    }
  }

  llvm::SmallVector<unsigned> valueOrder(values.size());
  std::iota(valueOrder.begin(), valueOrder.end(), 0u);
  llvm::stable_sort(valueOrder, [&](unsigned lhs, unsigned rhs) {
    return valueKeys[lhs] < valueKeys[rhs];
  });

  llvm::SmallVector<WorkloadValueId> valueRemap(values.size());
  llvm::SmallVector<WorkloadValue> newValues;
  newValues.reserve(values.size());
  for (unsigned newId = 0; newId < valueOrder.size(); ++newId) {
    valueRemap[valueOrder[newId]] = newId;
    WorkloadValue value = values[valueOrder[newId]];
    value.id = newId;
    newValues.push_back(std::move(value));
  }

  llvm::SmallVector<unsigned> nodeOrder(nodes.size());
  std::iota(nodeOrder.begin(), nodeOrder.end(), 0u);
  llvm::stable_sort(nodeOrder, [&](unsigned lhs, unsigned rhs) {
    return nodeKeys[lhs] < nodeKeys[rhs];
  });

  llvm::SmallVector<WorkloadNode> newNodes;
  newNodes.reserve(nodes.size());
  for (unsigned newId = 0; newId < nodeOrder.size(); ++newId) {
    WorkloadNode node = nodes[nodeOrder[newId]];
    node.id = newId;
    for (WorkloadPort &port : node.inputs)
      port.value = valueRemap[port.value];
    for (WorkloadPort &port : node.outputs)
      port.value = valueRemap[port.value];
    newNodes.push_back(std::move(node));
  }

  if (valueRemapOut)
    for (unsigned index = 0; index < valueRemap.size(); ++index)
      (*valueRemapOut)[static_cast<WorkloadValueId>(index)] = valueRemap[index];

  nodes = std::move(newNodes);
  values = std::move(newValues);
}

Operation *WorkloadGraphBinding::opFor(WorkloadNodeId node) const {
  auto it = nodeOps.find(node);
  return it == nodeOps.end() ? nullptr : it->second;
}

Value WorkloadGraphBinding::valueFor(WorkloadValueId value) const {
  auto it = values.find(value);
  return it == values.end() ? Value() : it->second;
}

std::string WorkloadGraph::canonicalString() const {
  std::string out;
  for (const WorkloadValue &value : values) {
    out += "value ";
    out += std::to_string(value.id);
    out += value.external ? " external " : " internal ";
    out += typeString(value.type);
    out += " name=";
    out += value.name;
    out += '\n';
  }
  for (const WorkloadNode &node : nodes) {
    out += "node ";
    out += std::to_string(node.id);
    out += ' ';
    out += node.opName;
    out += " in=[";
    for (const WorkloadPort &port : node.inputs) {
      out += std::to_string(port.value);
      out += ',';
    }
    out += "] out=[";
    for (const WorkloadPort &port : node.outputs) {
      out += std::to_string(port.value);
      out += ',';
    }
    out += ']';
    if (node.attributes) {
      out += " attrs=";
      out += attributeString(node.attributes);
    }
    out += '\n';
  }
  return out;
}

llvm::Expected<WorkloadGraph>
extractWorkloadGraph(Operation *kernel, WorkloadGraphBinding *binding) {
  if (!kernel || kernel->getName().getStringRef() != "micro.kernel")
    return llvm::createStringError(
        "expected a micro.kernel op to extract a workload graph from");

  WorkloadGraph graph;
  llvm::DenseMap<Value, WorkloadValueId> valueIds;
  // Filled alongside the graph: temporary value ids, and the op each node's
  // source ordinal belongs to.
  llvm::DenseMap<WorkloadValueId, Value> pendingValues;
  llvm::DenseMap<unsigned, Operation *> ordinalOps;
  unsigned ordinal = 0;

  std::function<WorkloadValueId(Value)> resolve =
      [&](Value value) -> WorkloadValueId {
    auto it = valueIds.find(value);
    if (it != valueIds.end())
      return it->second;

    if (Operation *definingOp = value.getDefiningOp()) {
      if (isTransparentWorkloadOp(definingOp->getName()) &&
          definingOp->getNumOperands() > 0) {
        WorkloadValueId source = resolve(definingOp->getOperand(0));
        valueIds[value] = source;
        return source;
      }
    }

    WorkloadValueId id = graph.addValue(
        WorkloadValue{0, value.getType(), nameFor(value), /*external=*/true});
    valueIds[value] = id;
    pendingValues[id] = value;
    return id;
  };

  kernel->walk([&](Operation *op) {
    OperationName name = op->getName();

    if (isTransparentWorkloadOp(name)) {
      if (op->getNumResults() > 0 && op->getNumOperands() > 0) {
        WorkloadValueId source = resolve(op->getOperand(0));
        for (Value result : op->getResults())
          valueIds[result] = source;
      }
      return;
    }

    if (isWorkloadNodeOp(name)) {
      WorkloadNode node;
      node.opName = name.getStringRef().str();
      node.attributes = op->getAttrDictionary();
      node.sourceOrdinal = ordinal;
      ordinalOps[ordinal] = op;
      ++ordinal;
      for (Value operand : op->getOperands()) {
        if (isAsyncToken(operand.getType()))
          continue;
        node.inputs.push_back(
            WorkloadPort{resolve(operand), operand.getType(), std::nullopt});
      }
      for (Value result : op->getResults()) {
        if (isAsyncToken(result.getType()))
          continue;
        WorkloadValueId id = graph.addValue(WorkloadValue{
            0, result.getType(), nameFor(result), /*external=*/false});
        valueIds[result] = id;
        pendingValues[id] = result;
        node.outputs.push_back(
            WorkloadPort{id, result.getType(), std::nullopt});
      }
      graph.addNode(std::move(node));
      return;
    }

    // Structural, allocation, and synchronization ops are not nodes, but their
    // results are still values a node may read (an allocation, a token-free
    // result). Register them as external so consumers can resolve them.
    for (Value result : op->getResults()) {
      if (isAsyncToken(result.getType()) || valueIds.count(result))
        continue;
      WorkloadValueId id = graph.addValue(WorkloadValue{
          0, result.getType(), nameFor(result), /*external=*/true});
      valueIds[result] = id;
      pendingValues[id] = result;
    }
  });

  llvm::DenseMap<WorkloadValueId, WorkloadValueId> remap;
  graph.finalize(&remap);
  if (binding) {
    for (const auto &entry : pendingValues)
      binding->values[remap[entry.first]] = entry.second;
    for (const WorkloadNode &node : graph.getNodes()) {
      auto it = ordinalOps.find(node.sourceOrdinal);
      if (it != ordinalOps.end())
        binding->nodeOps[node.id] = it->second;
    }
  }
  return std::move(graph);
}

} // namespace mlir::llk::mapping
