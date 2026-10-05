//===- MappingMetadata.cpp - Persisted selected-plan metadata (task B1)
//-----===//
//
// Generic, target-neutral encoding/decoding of a selected plan's
// execution-affecting state (see MappingMetadata.h). Target ids are only ever
// compared, hashed and reported here; nothing in this file branches on a target
// name or reads a target-owned field as target semantics.
//
// The source-graph identity is computed over *pre-materialization* semantics:
// binder-emitted movement operations (`micro.async_copy` carrying
// `micro.connection`) and binder-emitted `micro.transform` operations are
// resolved back to the value they read, and bookkeeping attributes are omitted.
// The same rendering therefore hashes a source kernel and the materialized
// kernel built from it identically -- which is what lets a materialized graph
// be validated against the source identity its metadata recorded.
//
//===----------------------------------------------------------------------===//

#include "LLK/Mapping/MappingMetadata.h"

#include "LLK/Mapping/StableHash.h"

#include "LLK/Machine/MachineModel.h"

#include "mlir/IR/BuiltinTypes.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <cstdint>
#include <string>
#include <vector>

namespace mlir::llk::mapping {

namespace {

constexpr llvm::StringLiteral kPlanAttr = "micro.plan";
constexpr llvm::StringLiteral kMappingAttr = "micro.mapping";
constexpr llvm::StringLiteral kRoutesAttr = "micro.routes";

/// Binder bookkeeping attributes excluded from the canonical source-graph
/// rendering: they are materialization/provenance, not source semantics.
bool isBookkeepingAttribute(llvm::StringRef name) {
  return name == "micro.mapping" || name == "micro.routes" ||
         name == "micro.plan" || name == "micro.value" ||
         name == "micro.dst_node" || name == "micro.connection" ||
         name == "micro.hop";
}

llvm::Error metadataError(std::string message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 std::move(message));
}

mlir::IntegerAttr u64Attr(mlir::MLIRContext *context, uint64_t value) {
  return mlir::IntegerAttr::get(mlir::IntegerType::get(context, 64),
                                static_cast<int64_t>(value));
}

std::string printedType(mlir::Type type) {
  std::string text;
  llvm::raw_string_ostream stream(text);
  type.print(stream);
  return stream.str();
}

std::string printedAttribute(mlir::Attribute attribute) {
  std::string text;
  llvm::raw_string_ostream stream(text);
  attribute.print(stream);
  return stream.str();
}

std::string printedMap(mlir::AffineMap map) {
  if (!map)
    return "<null>";
  std::string text;
  llvm::raw_string_ostream stream(text);
  map.print(stream);
  return stream.str();
}

/// The attributes of `node` with bookkeeping removed, rendered
/// deterministically.
std::string strippedAttributesString(const WorkloadNode &node) {
  if (!node.attributes)
    return {};
  llvm::SmallVector<mlir::NamedAttribute> kept;
  for (mlir::NamedAttribute attribute : node.attributes)
    if (!isBookkeepingAttribute(attribute.getName().getValue()))
      kept.push_back(attribute);
  if (kept.empty())
    return {};
  return printedAttribute(
      mlir::DictionaryAttr::get(node.attributes.getContext(), kept));
}

/// The structural content key of a node: operation name, port types and
/// semantic attributes. It carries no value identity, so it is stable under the
/// value rewiring materialization performs -- and is what correlates a source
/// node with its materialized counterpart.
std::string structuralNodeKey(const WorkloadNode &node) {
  std::string key = node.opName;
  key += "|in:";
  for (const WorkloadPort &port : node.inputs) {
    key += printedType(port.type);
    key += ',';
  }
  key += "|out:";
  for (const WorkloadPort &port : node.outputs) {
    key += printedType(port.type);
    key += ',';
  }
  key += "|attr:";
  key += strippedAttributesString(node);
  return key;
}

/// A binder-emitted movement: the canonical materialized copy marked with its
/// connection provenance. Such a node is not source work and is dropped from
/// the source projection.
bool isBinderMovement(const WorkloadNode &node) {
  return node.attributes && node.attributes.get("micro.connection") != nullptr;
}

/// The source projection of `graph`: the semantic nodes (binder movements
/// dropped) with their operand occurrences resolved back through materialized
/// movements, in canonical source order. `nodeOps`, when `binding` is given,
/// maps each projected (source) node id to the operation it came from.
struct SourceProjection {
  WorkloadGraph graph;
  llvm::DenseMap<WorkloadNodeId, mlir::Operation *> nodeOps;
};

/// Resolves a value id back through binder-emitted aliases (a movement's result
/// to the value it reads, a binder `micro.transform`'s result to its operand),
/// using the SSA correspondence `binding` records.
WorkloadValueId resolveAlias(WorkloadValueId value,
                             const WorkloadGraphBinding &binding) {
  // SSA value -> finalized workload value id.
  llvm::DenseMap<mlir::Value, WorkloadValueId> bySsa;
  for (const auto &entry : binding.values)
    bySsa[entry.second] = entry.first;

  for (unsigned depth = 0; depth < 64; ++depth) {
    mlir::Value ssa = binding.valueFor(value);
    if (!ssa)
      break;
    mlir::Operation *def = ssa.getDefiningOp();
    if (!def)
      break;
    bool aliases = def->hasAttr("micro.connection") ||
                   def->getName().getStringRef() == "micro.transform";
    if (!aliases || def->getNumOperands() == 0)
      break;
    auto it = bySsa.find(def->getOperand(0));
    if (it == bySsa.end() || it->second == value)
      break;
    value = it->second;
  }
  return value;
}

SourceProjection buildSourceProjection(const WorkloadGraph &graph,
                                       const WorkloadGraphBinding *binding) {
  llvm::DenseMap<WorkloadValueId, WorkloadValueId> alias;
  if (binding)
    for (const WorkloadValue &value : graph.getValues())
      alias[value.id] = resolveAlias(value.id, *binding);
  auto resolve = [&](WorkloadValueId value) {
    auto it = alias.find(value);
    return it == alias.end() ? value : it->second;
  };

  llvm::SmallVector<const WorkloadNode *, 8> semantic;
  for (const WorkloadNode &node : graph.getNodes())
    if (!isBinderMovement(node))
      semantic.push_back(&node);

  // Canonical source order: by structural key, preserving the graph's own order
  // for structurally identical nodes. The source graph's finalized order is the
  // same (key, program order) ordering, so projected ids equal source ids.
  llvm::stable_sort(semantic,
                    [](const WorkloadNode *lhs, const WorkloadNode *rhs) {
                      return structuralNodeKey(*lhs) < structuralNodeKey(*rhs);
                    });

  SourceProjection projection;
  llvm::DenseMap<WorkloadValueId, WorkloadValueId> valueIds;
  auto projectedValue = [&](WorkloadValueId value) {
    WorkloadValueId resolved = resolve(value);
    auto it = valueIds.find(resolved);
    if (it != valueIds.end())
      return it->second;
    WorkloadValue copy;
    if (const WorkloadValue *existing = graph.findValue(resolved))
      copy = *existing;
    copy.id = 0;
    WorkloadValueId id = projection.graph.addValue(std::move(copy));
    valueIds[resolved] = id;
    return id;
  };

  for (const WorkloadNode *node : semantic) {
    WorkloadNode copy = *node;
    // Drop the program ordinal so the projection's canonical order depends only
    // on structural content, not on where the source sat in the materialized
    // kernel's program order.
    copy.sourceOrdinal = 0;
    copy.inputs.clear();
    copy.outputs.clear();
    for (const WorkloadPort &port : node->inputs) {
      WorkloadPort projected = port;
      projected.value = projectedValue(port.value);
      copy.inputs.push_back(projected);
    }
    for (const WorkloadPort &port : node->outputs) {
      WorkloadPort projected = port;
      projected.value = projectedValue(port.value);
      copy.outputs.push_back(projected);
    }
    if (binding) {
      if (mlir::Operation *op = binding->opFor(node->id))
        projection.nodeOps[static_cast<WorkloadNodeId>(
            projection.graph.getNodes().size())] = op;
    }
    projection.graph.addNode(std::move(copy));
  }
  projection.graph.finalize();
  return projection;
}

/// One projected node's canonical record: its ports with unresolvable value
/// labels replaced by producer-relative labels, so the rendering does not
/// depend on absolute value ids.
std::string canonicalProjectedGraphString(const WorkloadGraph &graph) {
  llvm::DenseMap<WorkloadValueId, std::pair<WorkloadNodeId, unsigned>> producer;
  for (const WorkloadNode &node : graph.getNodes())
    for (unsigned index = 0; index < node.outputs.size(); ++index)
      producer[node.outputs[index].value] = {node.id, index};

  auto label = [&](WorkloadValueId value) -> std::string {
    const WorkloadValue *entry = graph.findValue(value);
    std::string name = entry ? entry->name : std::string();
    if (entry && entry->external)
      return "ext:" + name;
    auto it = producer.find(value);
    if (it == producer.end())
      return "ext:" + name;
    const WorkloadNode *node = graph.findNode(it->second.first);
    std::string text = "def:";
    if (node)
      text += structuralNodeKey(*node);
    text += "#";
    text += std::to_string(it->second.second);
    return text;
  };

  std::vector<std::string> records;
  records.reserve(graph.getNodes().size());
  for (const WorkloadNode &node : graph.getNodes()) {
    std::string record = node.opName;
    for (unsigned index = 0; index < node.inputs.size(); ++index) {
      const WorkloadPort &port = node.inputs[index];
      record += "|in";
      record += std::to_string(index);
      record += "=";
      record += printedType(port.type);
      record += "@";
      record += port.accessMap ? printedMap(*port.accessMap) : "-";
      record += "#";
      record += label(port.value);
    }
    for (unsigned index = 0; index < node.outputs.size(); ++index) {
      const WorkloadPort &port = node.outputs[index];
      record += "|out";
      record += std::to_string(index);
      record += "=";
      record += printedType(port.type);
      record += "#";
      record += label(port.value);
    }
    record += "|attrs=";
    record += strippedAttributesString(node);
    records.push_back(std::move(record));
  }
  llvm::sort(records);
  std::string out;
  for (const std::string &record : records) {
    out += record;
    out += '\n';
  }
  return out;
}

/// The module's single `micro.kernel`, or an error naming why it is not unique.
llvm::Expected<mlir::Operation *> findKernel(mlir::ModuleOp module) {
  llvm::SmallVector<mlir::Operation *, 2> kernels;
  module->walk([&](mlir::Operation *op) {
    if (op->getName().getStringRef() == "micro.kernel")
      kernels.push_back(op);
  });
  if (kernels.empty())
    return metadataError("the module has no micro.kernel");
  if (kernels.size() > 1)
    return metadataError("the module has " + std::to_string(kernels.size()) +
                         " micro.kernels; one kernel per module is required");
  return kernels.front();
}

/// One search value as a typed attribute, so a persisted parameter round-trips
/// as its own type rather than being flattened to text.
mlir::Attribute searchValueAttr(mlir::MLIRContext *context,
                                const SearchValue &value) {
  if (const int64_t *integer = std::get_if<int64_t>(&value))
    return u64Attr(context, static_cast<uint64_t>(*integer));
  return mlir::StringAttr::get(context, std::get<std::string>(value));
}

/// The typed value an attribute encodes, or an error for an unsupported kind.
llvm::Expected<SearchValue> searchValueOf(mlir::Attribute attribute,
                                          llvm::StringRef where) {
  if (auto integer = mlir::dyn_cast<mlir::IntegerAttr>(attribute))
    return SearchValue{integer.getInt()};
  if (auto text = mlir::dyn_cast<mlir::StringAttr>(attribute))
    return SearchValue{text.getValue().str()};
  return metadataError(where.str() + ": parameter is not an integer or string");
}

mlir::DictionaryAttr
stringMapAttr(mlir::MLIRContext *context,
              const llvm::StringMap<std::string> &entries) {
  std::vector<std::pair<std::string, std::string>> ordered;
  for (const auto &entry : entries)
    ordered.emplace_back(entry.first().str(), entry.second);
  llvm::sort(ordered);
  llvm::SmallVector<mlir::NamedAttribute> attributes;
  for (const auto &entry : ordered)
    attributes.emplace_back(mlir::StringAttr::get(context, entry.first),
                            mlir::StringAttr::get(context, entry.second));
  return mlir::DictionaryAttr::get(context, attributes);
}

mlir::ArrayAttr stringArrayAttr(mlir::MLIRContext *context,
                                llvm::ArrayRef<std::string> values) {
  llvm::SmallVector<mlir::Attribute> attributes;
  for (const std::string &value : values)
    attributes.push_back(mlir::StringAttr::get(context, value));
  return mlir::ArrayAttr::get(context, attributes);
}

/// The layout class a `layout_parameters` key names (strips an `#<index>`
/// disambiguating suffix).
llvm::StringRef baseLayoutClass(llvm::StringRef key) {
  size_t hash = key.rfind('#');
  if (hash == llvm::StringRef::npos)
    return key;
  llvm::StringRef suffix = key.substr(hash + 1);
  if (suffix.empty() ||
      !llvm::all_of(suffix, [](char c) { return c >= '0' && c <= '9'; }))
    return key;
  return key.substr(0, hash);
}

/// The solved parameterization of each layout requirement, keyed exactly as the
/// plan's `layoutSolutions` are. Keys and parameter names are sorted, so the
/// persisted form is deterministic.
mlir::DictionaryAttr
layoutParametersAttr(mlir::MLIRContext *context,
                     const llvm::StringMap<SolvedLayout> &solutions) {
  std::vector<std::string> keys;
  keys.reserve(solutions.size());
  for (const auto &entry : solutions)
    keys.push_back(entry.first().str());
  llvm::sort(keys);
  llvm::SmallVector<mlir::NamedAttribute> attributes;
  for (const std::string &key : keys) {
    const SolvedLayout &solution = solutions.lookup(key);
    std::vector<std::string> names;
    names.reserve(solution.parameters.size());
    for (const auto &entry : solution.parameters)
      names.push_back(entry.first().str());
    llvm::sort(names);
    llvm::SmallVector<mlir::NamedAttribute> parameters;
    for (const std::string &name : names)
      parameters.emplace_back(
          mlir::StringAttr::get(context, name),
          searchValueAttr(context, solution.parameters.lookup(name)));
    attributes.emplace_back(mlir::StringAttr::get(context, key),
                            mlir::DictionaryAttr::get(context, parameters));
  }
  return mlir::DictionaryAttr::get(context, attributes);
}

/// The endpoint occurrence dictionaries of a layout entry.
mlir::DictionaryAttr layoutEntryAttr(mlir::MLIRContext *context,
                                     llvm::StringRef key,
                                     const SolvedLayout &solution,
                                     const PlanPlacement &placement) {
  llvm::SmallVector<mlir::NamedAttribute> fields;
  fields.emplace_back(mlir::StringAttr::get(context, "key"),
                      mlir::StringAttr::get(context, key));
  fields.emplace_back(mlir::StringAttr::get(context, "class"),
                      mlir::StringAttr::get(context, solution.layoutClass));
  std::string family = placement.layouts.lookup(solution.layoutClass);
  fields.emplace_back(mlir::StringAttr::get(context, "family"),
                      mlir::StringAttr::get(context, family));
  if (solution.port)
    fields.emplace_back(mlir::StringAttr::get(context, "port"),
                        metadataPortRefAttr(context, *solution.port));
  if (solution.map)
    fields.emplace_back(mlir::StringAttr::get(context, "map"),
                        mlir::AffineMapAttr::get(solution.map));
  return mlir::DictionaryAttr::get(context, fields);
}

} // namespace

//===----------------------------------------------------------------------===//
// Content hashes
//===----------------------------------------------------------------------===//

uint64_t computeSourceGraphHash(const WorkloadGraph &graph) {
  SourceProjection projection =
      buildSourceProjection(graph, /*binding=*/nullptr);
  return stableHash(canonicalProjectedGraphString(projection.graph));
}

llvm::Expected<uint64_t> computeModuleSourceGraphHash(mlir::Operation *kernel) {
  if (!kernel || kernel->getName().getStringRef() != "micro.kernel")
    return metadataError("expected a micro.kernel op");
  WorkloadGraphBinding binding;
  llvm::Expected<WorkloadGraph> graph = extractWorkloadGraph(kernel, &binding);
  if (!graph)
    return graph.takeError();
  SourceProjection projection = buildSourceProjection(*graph, &binding);
  return stableHash(canonicalProjectedGraphString(projection.graph));
}

llvm::Expected<SourceGraphView> buildSourceGraphView(mlir::Operation *kernel) {
  if (!kernel || kernel->getName().getStringRef() != "micro.kernel")
    return metadataError("expected a micro.kernel op");
  WorkloadGraphBinding binding;
  llvm::Expected<WorkloadGraph> graph = extractWorkloadGraph(kernel, &binding);
  if (!graph)
    return graph.takeError();
  SourceProjection projection = buildSourceProjection(*graph, &binding);
  SourceGraphView view;
  view.graph = std::move(projection.graph);
  view.nodeOps = std::move(projection.nodeOps);
  return view;
}

uint64_t computeTargetContentHash(const MappingTarget &target) {
  uint64_t hash = stableHash(target.name());
  hash = stableHashCombine(hash, machine::computeContentHash(target.machine()));
  hash = stableHashCombine(hash, target.layouts().computeContentHash());
  hash = stableHashCombine(hash, target.rules().computeContentHash());
  return hash;
}

//===----------------------------------------------------------------------===//
// Shared typed readers
//===----------------------------------------------------------------------===//

llvm::Expected<std::string> readMetadataString(mlir::DictionaryAttr dict,
                                               llvm::StringRef name,
                                               llvm::StringRef where) {
  mlir::Attribute raw = dict.get(name);
  if (!raw)
    return metadataError((where + ": missing '" + name + "'").str());
  auto text = mlir::dyn_cast<mlir::StringAttr>(raw);
  if (!text)
    return metadataError((where + ": '" + name + "' is not a string").str());
  return text.getValue().str();
}

llvm::Expected<llvm::StringMap<std::string>>
readMetadataStringMap(mlir::Attribute raw, llvm::StringRef name,
                      llvm::StringRef where) {
  auto entries = mlir::dyn_cast<mlir::DictionaryAttr>(raw);
  if (!entries)
    return metadataError(
        (where + ": '" + name + "' is not a dictionary").str());
  llvm::StringMap<std::string> values;
  for (const mlir::NamedAttribute &entry : entries) {
    auto value = mlir::dyn_cast<mlir::StringAttr>(entry.getValue());
    if (!value)
      return metadataError((where + ": '" + name + "' entry '" +
                            entry.getName().str() + "' is not a string")
                               .str());
    values[entry.getName()] = value.getValue().str();
  }
  return values;
}

llvm::Expected<llvm::SmallVector<std::string, 4>>
readMetadataStringArray(mlir::Attribute raw, llvm::StringRef name,
                        llvm::StringRef where) {
  llvm::SmallVector<std::string, 4> values;
  auto array = mlir::dyn_cast<mlir::ArrayAttr>(raw);
  if (!array)
    return metadataError((where + ": '" + name + "' is not an array").str());
  for (mlir::Attribute element : array) {
    auto text = mlir::dyn_cast<mlir::StringAttr>(element);
    if (!text)
      return metadataError(
          (where + ": '" + name + "' has a non-string entry").str());
    values.push_back(text.getValue().str());
  }
  return values;
}

llvm::Expected<PortRef> readMetadataPortRef(mlir::Attribute raw,
                                            llvm::StringRef name,
                                            llvm::StringRef where) {
  auto dict = mlir::dyn_cast<mlir::DictionaryAttr>(raw);
  if (!dict)
    return metadataError(
        (where + ": '" + name + "' is not a dictionary").str());
  PortRef ref;
  auto node = dict.getAs<mlir::IntegerAttr>("node");
  if (!node)
    return metadataError(
        (where + ": '" + name + "' has no integer 'node'").str());
  if (node.getInt() < 0)
    return metadataError((where + ": '" + name + "' node is negative").str());
  ref.node = static_cast<WorkloadNodeId>(node.getInt());
  std::string direction;
  auto directionAttr = dict.getAs<mlir::StringAttr>("direction");
  if (!directionAttr)
    return metadataError(
        (where + ": '" + name + "' has no string 'direction'").str());
  direction = directionAttr.getValue().str();
  if (direction == "input")
    ref.direction = PortDirection::Input;
  else if (direction == "output")
    ref.direction = PortDirection::Output;
  else
    return metadataError(
        (where + ": '" + name + "' names unknown direction '" + direction + "'")
            .str());
  auto index = dict.getAs<mlir::IntegerAttr>("index");
  if (!index)
    return metadataError(
        (where + ": '" + name + "' has no integer 'index'").str());
  if (index.getInt() < 0 || index.getInt() > UINT32_MAX)
    return metadataError(
        (where + ": '" + name + "' index is out of range").str());
  ref.index = static_cast<uint32_t>(index.getInt());
  return ref;
}

mlir::Attribute metadataPortRefAttr(mlir::MLIRContext *context,
                                    const PortRef &port) {
  llvm::SmallVector<mlir::NamedAttribute> fields;
  fields.emplace_back(mlir::StringAttr::get(context, "node"),
                      u64Attr(context, port.node));
  fields.emplace_back(
      mlir::StringAttr::get(context, "direction"),
      mlir::StringAttr::get(context, port.direction == PortDirection::Input
                                         ? "input"
                                         : "output"));
  fields.emplace_back(mlir::StringAttr::get(context, "index"),
                      u64Attr(context, port.index));
  return mlir::DictionaryAttr::get(context, fields);
}

//===----------------------------------------------------------------------===//
// Encoding
//===----------------------------------------------------------------------===//

llvm::Error encodeSelectedPlan(mlir::ModuleOp module, const CoveringPlan &plan,
                               const MappingTarget &target) {
  llvm::Expected<mlir::Operation *> resolvedKernel = findKernel(module);
  if (!resolvedKernel)
    return resolvedKernel.takeError();
  mlir::Operation *kernel = *resolvedKernel;

  WorkloadGraphBinding binding;
  llvm::Expected<WorkloadGraph> graph = extractWorkloadGraph(kernel, &binding);
  if (!graph)
    return graph.takeError();

  mlir::MLIRContext *context = module.getContext();
  uint64_t graphHash = computeSourceGraphHash(*graph);
  uint64_t machineHash = machine::computeContentHash(target.machine());
  uint64_t layoutHash = target.layouts().computeContentHash();
  uint64_t ruleHash = target.rules().computeContentHash();
  uint64_t targetHash = computeTargetContentHash(target);

  llvm::SmallVector<mlir::NamedAttribute> planFields;
  planFields.emplace_back(mlir::StringAttr::get(context, "schema_version"),
                          u64Attr(context, kMappingMetadataVersion));
  planFields.emplace_back(mlir::StringAttr::get(context, "id"),
                          u64Attr(context, plan.id));
  planFields.emplace_back(mlir::StringAttr::get(context, "binding_hash"),
                          u64Attr(context, plan.sourceBindingHash));
  planFields.emplace_back(mlir::StringAttr::get(context, "materialized"),
                          mlir::BoolAttr::get(context, plan.materialized));
  planFields.emplace_back(mlir::StringAttr::get(context, "graph_hash"),
                          mlir::StringAttr::get(context, hexId(graphHash)));
  planFields.emplace_back(mlir::StringAttr::get(context, "target_hash"),
                          mlir::StringAttr::get(context, hexId(targetHash)));
  planFields.emplace_back(mlir::StringAttr::get(context, "machine_hash"),
                          mlir::StringAttr::get(context, hexId(machineHash)));
  planFields.emplace_back(mlir::StringAttr::get(context, "layout_hash"),
                          mlir::StringAttr::get(context, hexId(layoutHash)));
  planFields.emplace_back(mlir::StringAttr::get(context, "rule_hash"),
                          mlir::StringAttr::get(context, hexId(ruleHash)));
  planFields.emplace_back(
      mlir::StringAttr::get(context, "truncated"),
      mlir::BoolAttr::get(context, plan.diagnostics.searchTruncated));
  kernel->setAttr(kPlanAttr, mlir::DictionaryAttr::get(context, planFields));

  for (const PlanPlacement &placement : plan.placements) {
    mlir::Operation *op = binding.opFor(placement.node);
    if (!op)
      return metadataError("plan covers node " +
                           std::to_string(placement.node) +
                           ", which the kernel does not contain");
    const RuleDef *rule = target.rules().find(placement.rule);
    if (!rule)
      return metadataError("plan selects unknown rule '" + placement.rule +
                           "'");

    llvm::StringMap<std::string> memories;
    for (const auto &entry : placement.memories)
      memories[entry.first()] = entry.second;
    llvm::StringMap<std::string> layouts;
    for (const auto &entry : placement.layouts)
      layouts[entry.first()] = entry.second;

    llvm::SmallVector<mlir::NamedAttribute> attributes;
    attributes.emplace_back(mlir::StringAttr::get(context, "schema_version"),
                            u64Attr(context, kMappingMetadataVersion));
    attributes.emplace_back(mlir::StringAttr::get(context, "node"),
                            u64Attr(context, placement.node));
    attributes.emplace_back(mlir::StringAttr::get(context, "instance"),
                            u64Attr(context, placement.instance));
    attributes.emplace_back(mlir::StringAttr::get(context, "rule"),
                            mlir::StringAttr::get(context, placement.rule));
    attributes.emplace_back(
        mlir::StringAttr::get(context, "bundle"),
        mlir::StringAttr::get(context, placement.bundle.name));
    attributes.emplace_back(
        mlir::StringAttr::get(context, "emitter"),
        mlir::StringAttr::get(context, placement.bundle.emitterKey));
    attributes.emplace_back(mlir::StringAttr::get(context, "executor"),
                            mlir::StringAttr::get(context, placement.executor));
    attributes.emplace_back(mlir::StringAttr::get(context, "memories"),
                            stringMapAttr(context, memories));
    attributes.emplace_back(mlir::StringAttr::get(context, "layouts"),
                            stringMapAttr(context, layouts));
    if (placement.bundle.parameters)
      attributes.emplace_back(
          mlir::StringAttr::get(context, "bundle_parameters"),
          placement.bundle.parameters);

    if (rule->layoutRequirements.empty()) {
      // An explicit "no layout" marker: a v2 reader requires exactly one of the
      // layout container or this marker, so deleting the container cannot skip
      // layout verification.
      attributes.emplace_back(mlir::StringAttr::get(context, "no_layout"),
                              mlir::BoolAttr::get(context, true));
    } else {
      attributes.emplace_back(
          mlir::StringAttr::get(context, "layout_parameters"),
          layoutParametersAttr(context, placement.layoutSolutions));
      llvm::SmallVector<mlir::Attribute> entries;
      std::vector<std::string> keys;
      keys.reserve(placement.layoutSolutions.size());
      for (const auto &entry : placement.layoutSolutions)
        keys.push_back(entry.first().str());
      llvm::sort(keys);
      for (const std::string &key : keys)
        entries.push_back(layoutEntryAttr(
            context, key, placement.layoutSolutions.lookup(key), placement));
      attributes.emplace_back(mlir::StringAttr::get(context, "layout_entries"),
                              mlir::ArrayAttr::get(context, entries));
    }
    op->setAttr(kMappingAttr, mlir::DictionaryAttr::get(context, attributes));
  }

  llvm::SmallVector<mlir::Attribute> routes;
  for (const PlanConnection &connection : plan.connectionPlans) {
    llvm::SmallVector<mlir::NamedAttribute> attributes;
    attributes.emplace_back(mlir::StringAttr::get(context, "id"),
                            u64Attr(context, connection.id));
    attributes.emplace_back(mlir::StringAttr::get(context, "value"),
                            u64Attr(context, connection.value));
    attributes.emplace_back(
        mlir::StringAttr::get(context, "kind"),
        mlir::StringAttr::get(context,
                              stringifyConnectionKind(connection.kind)));
    std::vector<std::string> route;
    for (const MemoryNodeId &node : connection.route)
      route.push_back(node);
    attributes.emplace_back(mlir::StringAttr::get(context, "route"),
                            stringArrayAttr(context, route));
    std::vector<std::string> engines;
    for (const ExecutorId &engine : connection.engines)
      engines.push_back(engine);
    attributes.emplace_back(mlir::StringAttr::get(context, "engines"),
                            stringArrayAttr(context, engines));
    llvm::SmallVector<mlir::Attribute> consumers;
    for (InstanceId consumer : connection.consumers)
      consumers.push_back(u64Attr(context, consumer));
    attributes.emplace_back(mlir::StringAttr::get(context, "consumers"),
                            mlir::ArrayAttr::get(context, consumers));
    if (connection.producerPort)
      attributes.emplace_back(
          mlir::StringAttr::get(context, "producer_port"),
          metadataPortRefAttr(context, *connection.producerPort));
    llvm::SmallVector<mlir::Attribute> consumerPorts;
    for (const PortRef &port : connection.consumerPorts)
      consumerPorts.push_back(metadataPortRefAttr(context, port));
    attributes.emplace_back(mlir::StringAttr::get(context, "consumer_ports"),
                            mlir::ArrayAttr::get(context, consumerPorts));
    llvm::SmallVector<mlir::Attribute> storageIds;
    for (uint64_t id : connection.storageIds)
      storageIds.push_back(u64Attr(context, id));
    attributes.emplace_back(mlir::StringAttr::get(context, "storage_ids"),
                            mlir::ArrayAttr::get(context, storageIds));
    if (connection.transform) {
      llvm::SmallVector<mlir::NamedAttribute> transform;
      transform.emplace_back(
          mlir::StringAttr::get(context, "src"),
          mlir::StringAttr::get(context, connection.transform->srcLayout));
      transform.emplace_back(
          mlir::StringAttr::get(context, "dst"),
          mlir::StringAttr::get(context, connection.transform->dstLayout));
      if (connection.transform->srcMap)
        transform.emplace_back(
            mlir::StringAttr::get(context, "src_map"),
            mlir::AffineMapAttr::get(connection.transform->srcMap));
      if (connection.transform->dstMap)
        transform.emplace_back(
            mlir::StringAttr::get(context, "dst_map"),
            mlir::AffineMapAttr::get(connection.transform->dstMap));
      attributes.emplace_back(mlir::StringAttr::get(context, "transform"),
                              mlir::DictionaryAttr::get(context, transform));
    }
    routes.push_back(mlir::DictionaryAttr::get(context, attributes));
  }
  kernel->setAttr(kRoutesAttr, mlir::ArrayAttr::get(context, routes));
  return llvm::Error::success();
}

//===----------------------------------------------------------------------===//
// Decoding
//===----------------------------------------------------------------------===//

namespace {

/// Reads a `{param = value}` parameter dictionary into the typed variant.
llvm::Expected<llvm::StringMap<SearchValue>>
readParameters(mlir::DictionaryAttr dict, llvm::StringRef where) {
  llvm::StringMap<SearchValue> values;
  for (const mlir::NamedAttribute &entry : dict) {
    llvm::Expected<SearchValue> value = searchValueOf(entry.getValue(), where);
    if (!value)
      return value.takeError();
    values[entry.getName()] = *value;
  }
  return values;
}

/// The rule's port occurrence for `portName`, mirroring the positional wiring
/// generation uses (inputs then outputs).
std::optional<PortRef> portRefForRulePort(const RuleDef &rule,
                                          const WorkloadNode &node,
                                          llvm::StringRef portName) {
  size_t inputIndex = 0;
  size_t outputIndex = 0;
  for (const RulePort &port : rule.ports) {
    if (port.isInput) {
      if (port.name == portName && inputIndex < node.inputs.size())
        return PortRef{node.id, PortDirection::Input,
                       static_cast<uint32_t>(inputIndex)};
      ++inputIndex;
    } else {
      if (port.name == portName && outputIndex < node.outputs.size())
        return PortRef{node.id, PortDirection::Output,
                       static_cast<uint32_t>(outputIndex)};
      ++outputIndex;
    }
  }
  return std::nullopt;
}

} // namespace

llvm::Expected<CoveringPlan> decodeSelectedPlan(mlir::ModuleOp module,
                                                const MappingTarget &target) {
  llvm::Expected<mlir::Operation *> resolvedKernel = findKernel(module);
  if (!resolvedKernel)
    return resolvedKernel.takeError();
  mlir::Operation *kernel = *resolvedKernel;

  auto planAttr = kernel->getAttrOfType<mlir::DictionaryAttr>(kPlanAttr);
  if (!planAttr)
    return metadataError("the kernel is not mapped: it has no micro.plan");

  uint64_t schemaVersion = 1;
  if (auto version = planAttr.getAs<mlir::IntegerAttr>("schema_version")) {
    if (version.getInt() < 0)
      return metadataError("micro.plan schema_version is negative");
    schemaVersion = static_cast<uint64_t>(version.getInt());
  }
  if (schemaVersion > kSupportedMappingMetadataVersion)
    return metadataError("micro.plan records unsupported schema version " +
                         std::to_string(schemaVersion) +
                         "; this reader understands at most " +
                         std::to_string(kSupportedMappingMetadataVersion));
  const bool v2 = schemaVersion >= 2;

  WorkloadGraphBinding binding;
  llvm::Expected<WorkloadGraph> graph = extractWorkloadGraph(kernel, &binding);
  if (!graph)
    return graph.takeError();
  SourceProjection projection = buildSourceProjection(*graph, &binding);
  const WorkloadGraph &sourceGraph = projection.graph;

  CoveringPlan plan;
  plan.schemaVersion = schemaVersion;
  if (auto id = planAttr.getAs<mlir::IntegerAttr>("id"))
    plan.id = static_cast<PlanId>(id.getInt());
  if (auto bindingHash = planAttr.getAs<mlir::IntegerAttr>("binding_hash"))
    plan.sourceBindingHash = static_cast<uint64_t>(bindingHash.getInt());
  if (auto materialized = planAttr.getAs<mlir::BoolAttr>("materialized"))
    plan.materialized = materialized.getValue();
  if (auto truncated = planAttr.getAs<mlir::BoolAttr>("truncated"))
    plan.diagnostics.searchTruncated = truncated.getValue();

  if (v2) {
    llvm::Expected<std::string> graphHash =
        readMetadataString(planAttr, "graph_hash", "micro.plan");
    if (!graphHash)
      return graphHash.takeError();
    llvm::Expected<std::string> targetHash =
        readMetadataString(planAttr, "target_hash", "micro.plan");
    if (!targetHash)
      return targetHash.takeError();
    llvm::Expected<uint64_t> current = computeModuleSourceGraphHash(kernel);
    if (!current)
      return current.takeError();
    if (*graphHash != hexId(*current))
      return metadataError("micro.plan graph_hash does not match the kernel's "
                           "source graph");
    if (*targetHash != hexId(computeTargetContentHash(target)))
      return metadataError("micro.plan target_hash does not match the target");
    plan.graphHash = *current;
    plan.targetHash = computeTargetContentHash(target);
    plan.machineHash = machine::computeContentHash(target.machine());
    plan.layoutHash = target.layouts().computeContentHash();
    plan.ruleHash = target.rules().computeContentHash();
  } else {
    // Legacy: still expose the current hashes for a caller that re-encodes, but
    // do not validate them -- a v1 binding predates the fields.
    plan.machineHash = machine::computeContentHash(target.machine());
    plan.layoutHash = target.layouts().computeContentHash();
    plan.ruleHash = target.rules().computeContentHash();
    plan.targetHash = computeTargetContentHash(target);
  }

  // --- placements --------------------------------------------------------
  llvm::DenseMap<mlir::Operation *, WorkloadNodeId> sourceNodeForOp;
  for (const auto &entry : projection.nodeOps)
    sourceNodeForOp[entry.second] = entry.first;

  bool sawUnrecoverable = false;
  module->walk([&](mlir::Operation *op) {
    if (sawUnrecoverable)
      return;
    auto mapping = op->getAttrOfType<mlir::DictionaryAttr>(kMappingAttr);
    if (!mapping)
      return;
    const std::string where =
        "mapped op '" + op->getName().getStringRef().str() + "'";

    PlanPlacement placement;
    // The source node id: recorded for v2, recovered from the graph for v1.
    if (v2) {
      auto node = mapping.getAs<mlir::IntegerAttr>("node");
      if (!node) {
        sawUnrecoverable = true;
        return;
      }
      placement.node = static_cast<WorkloadNodeId>(node.getInt());
      // The recorded node must be the op's own projected source node, so a
      // mapping cannot be moved onto another operation.
      auto found = sourceNodeForOp.find(op);
      if (found == sourceNodeForOp.end() || found->second != placement.node) {
        sawUnrecoverable = true;
        return;
      }
      if (auto instance = mapping.getAs<mlir::IntegerAttr>("instance"))
        placement.instance = static_cast<InstanceId>(instance.getInt());
    } else {
      auto found = sourceNodeForOp.find(op);
      if (found == sourceNodeForOp.end()) {
        sawUnrecoverable = true;
        return;
      }
      placement.node = found->second;
    }

    llvm::Expected<std::string> rule =
        readMetadataString(mapping, "rule", where);
    if (!rule) {
      sawUnrecoverable = true;
      return;
    }
    placement.rule = *rule;
    llvm::Expected<std::string> executor =
        readMetadataString(mapping, "executor", where);
    if (!executor) {
      sawUnrecoverable = true;
      return;
    }
    placement.executor = *executor;
    if (auto bundle = mapping.get("bundle")) {
      if (auto text = mlir::dyn_cast<mlir::StringAttr>(bundle))
        placement.bundle.name = text.getValue().str();
    }
    if (auto emitter = mapping.get("emitter")) {
      if (auto text = mlir::dyn_cast<mlir::StringAttr>(emitter))
        placement.bundle.emitterKey = text.getValue().str();
    }
    if (auto parameters =
            mapping.getAs<mlir::DictionaryAttr>("bundle_parameters"))
      placement.bundle.parameters = parameters;
    if (mlir::Attribute memories = mapping.get("memories")) {
      llvm::Expected<llvm::StringMap<std::string>> read =
          readMetadataStringMap(memories, "memories", where);
      if (!read) {
        sawUnrecoverable = true;
        return;
      }
      for (const auto &entry : *read)
        placement.memories[entry.first()] = entry.second;
    }
    if (mlir::Attribute layouts = mapping.get("layouts")) {
      llvm::Expected<llvm::StringMap<std::string>> read =
          readMetadataStringMap(layouts, "layouts", where);
      if (!read) {
        sawUnrecoverable = true;
        return;
      }
      for (const auto &entry : *read)
        placement.layouts[entry.first()] = entry.second;
    }

    // Endpoints and concrete maps of each solved layout, when recorded.
    llvm::DenseMap<llvm::StringRef, mlir::DictionaryAttr> entries;
    if (auto array = mapping.getAs<mlir::ArrayAttr>("layout_entries"))
      for (mlir::Attribute element : array)
        if (auto entry = mlir::dyn_cast<mlir::DictionaryAttr>(element))
          if (auto key = entry.getAs<mlir::StringAttr>("key"))
            entries[key.getValue()] = entry;

    if (mlir::Attribute rawParameters = mapping.get("layout_parameters")) {
      auto byClass = mlir::dyn_cast<mlir::DictionaryAttr>(rawParameters);
      if (!byClass) {
        sawUnrecoverable = true;
        return;
      }
      for (const mlir::NamedAttribute &entry : byClass) {
        auto parameters =
            mlir::dyn_cast<mlir::DictionaryAttr>(entry.getValue());
        if (!parameters) {
          sawUnrecoverable = true;
          return;
        }
        SolvedLayout solved;
        solved.layoutClass = baseLayoutClass(entry.getName().getValue()).str();
        llvm::Expected<llvm::StringMap<SearchValue>> values =
            readParameters(parameters, where);
        if (!values) {
          sawUnrecoverable = true;
          return;
        }
        solved.parameters = std::move(*values);
        auto found = entries.find(entry.getName().getValue());
        if (found != entries.end()) {
          mlir::DictionaryAttr detail = found->second;
          if (auto family = detail.getAs<mlir::StringAttr>("family"))
            solved.layoutClass = family.getValue().str();
          if (mlir::Attribute port = detail.get("port")) {
            llvm::Expected<PortRef> ref =
                readMetadataPortRef(port, "port", where);
            if (!ref) {
              sawUnrecoverable = true;
              return;
            }
            if (!lookupPort(sourceGraph, *ref)) {
              sawUnrecoverable = true;
              return;
            }
            solved.port = *ref;
          }
          if (auto map = detail.getAs<mlir::AffineMapAttr>("map"))
            solved.map = map.getValue();
        }
        // Legacy recovery: a v1 binding records no endpoint, so recover the
        // occurrence the rule names for this class when the declaration makes
        // it unique.
        if (!v2 && !solved.port) {
          if (const RuleDef *def = target.rules().find(placement.rule)) {
            if (const WorkloadNode *node =
                    sourceGraph.findNode(placement.node)) {
              for (const RuleLayoutRequirement &requirement :
                   def->layoutRequirements)
                if (requirement.layoutId == solved.layoutClass)
                  if (std::optional<PortRef> ref =
                          portRefForRulePort(*def, *node, requirement.port)) {
                    solved.port = ref;
                    break;
                  }
            }
          }
        }
        placement.layoutSolutions[entry.getName()] = std::move(solved);
      }
    } else if (!v2) {
      // Legacy recovery: rebuild the solved assignment from the declaration so
      // an unambiguous v1 binding still replays. A missing assignment the rule
      // requires is not recoverable.
      if (const RuleDef *ruleDef = target.rules().find(placement.rule)) {
        for (const RuleLayoutRequirement &requirement :
             ruleDef->layoutRequirements) {
          const WorkloadNode *node = sourceGraph.findNode(placement.node);
          if (!node) {
            sawUnrecoverable = true;
            return;
          }
          SolvedLayout solved;
          solved.layoutClass = requirement.layoutId;
          if (std::optional<PortRef> ref =
                  portRefForRulePort(*ruleDef, *node, requirement.port))
            solved.port = ref;
          placement.layoutSolutions[requirement.layoutId] = std::move(solved);
        }
      }
    }

    plan.placements.push_back(std::move(placement));
  });
  if (sawUnrecoverable)
    return metadataError(
        "legacy or malformed micro.mapping is not uniquely recoverable for "
        "executable replay");

  // --- routes ------------------------------------------------------------
  if (auto routes = kernel->getAttrOfType<mlir::ArrayAttr>(kRoutesAttr)) {
    for (mlir::Attribute element : routes) {
      auto route = mlir::dyn_cast<mlir::DictionaryAttr>(element);
      if (!route)
        return metadataError("micro.routes entry is not a dictionary");
      const std::string where = "micro.routes";
      PlanConnection connection;
      if (auto id = route.getAs<mlir::IntegerAttr>("id"))
        connection.id = static_cast<ConnectionId>(id.getInt());
      if (auto value = route.getAs<mlir::IntegerAttr>("value"))
        connection.value = static_cast<WorkloadValueId>(value.getInt());
      llvm::Expected<std::string> kind =
          readMetadataString(route, "kind", where);
      if (!kind)
        return kind.takeError();
      std::optional<ConnectionKind> symbolized = symbolizeConnectionKind(*kind);
      if (!symbolized)
        return metadataError("micro.routes: unknown connection kind '" + *kind +
                             "'");
      connection.kind = *symbolized;
      if (mlir::Attribute rawRoute = route.get("route")) {
        llvm::Expected<llvm::SmallVector<std::string, 4>> nodes =
            readMetadataStringArray(rawRoute, "route", where);
        if (!nodes)
          return nodes.takeError();
        for (const std::string &node : *nodes)
          connection.route.push_back(node);
      }
      if (mlir::Attribute rawEngines = route.get("engines")) {
        llvm::Expected<llvm::SmallVector<std::string, 4>> engines =
            readMetadataStringArray(rawEngines, "engines", where);
        if (!engines)
          return engines.takeError();
        for (const std::string &engine : *engines)
          connection.engines.push_back(engine);
      }
      if (auto consumers = route.getAs<mlir::ArrayAttr>("consumers"))
        for (mlir::Attribute consumer : consumers)
          if (auto id = mlir::dyn_cast<mlir::IntegerAttr>(consumer))
            connection.consumers.push_back(
                static_cast<InstanceId>(id.getInt()));
      if (mlir::Attribute producerPort = route.get("producer_port")) {
        llvm::Expected<PortRef> ref =
            readMetadataPortRef(producerPort, "producer_port", where);
        if (!ref)
          return ref.takeError();
        if (!lookupPort(sourceGraph, *ref))
          return metadataError(
              "micro.routes producer_port does not resolve in the source "
              "graph");
        connection.producerPort = *ref;
      }
      if (auto consumerPorts = route.getAs<mlir::ArrayAttr>("consumer_ports"))
        for (mlir::Attribute port : consumerPorts) {
          llvm::Expected<PortRef> ref =
              readMetadataPortRef(port, "consumer_ports", where);
          if (!ref)
            return ref.takeError();
          if (!lookupPort(sourceGraph, *ref))
            return metadataError(
                "micro.routes consumer_ports entry does not resolve in the "
                "source graph");
          connection.consumerPorts.push_back(*ref);
        }
      if (auto storageIds = route.getAs<mlir::ArrayAttr>("storage_ids"))
        for (mlir::Attribute id : storageIds)
          if (auto integer = mlir::dyn_cast<mlir::IntegerAttr>(id))
            connection.storageIds.push_back(
                static_cast<uint64_t>(integer.getInt()));
      if (auto transform = route.getAs<mlir::DictionaryAttr>("transform")) {
        LayoutTransform layoutTransform;
        if (auto src = transform.getAs<mlir::StringAttr>("src"))
          layoutTransform.srcLayout = src.getValue().str();
        if (auto dst = transform.getAs<mlir::StringAttr>("dst"))
          layoutTransform.dstLayout = dst.getValue().str();
        if (auto srcMap = transform.getAs<mlir::AffineMapAttr>("src_map"))
          layoutTransform.srcMap = srcMap.getValue();
        if (auto dstMap = transform.getAs<mlir::AffineMapAttr>("dst_map"))
          layoutTransform.dstMap = dstMap.getValue();
        connection.transform = layoutTransform;
      }

      // Legacy recovery: a v1 connection records no consumer endpoints, and
      // which operand occurrence a recorded consumer instance reads is not
      // uniquely recoverable from the instance projection -- two operand uses
      // of one value list the same instance. A v1 connection that names
      // consumers is therefore ambiguous executable replay and is rejected. A
      // consumer-free connection has nothing to recover; its producer endpoint
      // is recovered when the value has exactly one producing occurrence.
      if (!v2 && !connection.consumers.empty())
        return metadataError(
            "legacy micro.routes has no recoverable consumer endpoints; "
            "ambiguous executable replay is rejected");
      if (!v2 && connection.consumerPorts.empty()) {
        const WorkloadNode *producerNode = nullptr;
        unsigned producerIndex = 0;
        unsigned producers = 0;
        for (const WorkloadNode &node : sourceGraph.getNodes())
          for (unsigned index = 0; index < node.outputs.size(); ++index)
            if (node.outputs[index].value == connection.value) {
              producerNode = &node;
              producerIndex = index;
              ++producers;
            }
        if (producers == 1 && producerNode)
          connection.producerPort =
              PortRef{producerNode->id, PortDirection::Output, producerIndex};
      }

      plan.connectionPlans.push_back(std::move(connection));
    }
  }

  // Reconstruct the compatibility projections the plan carries.
  for (const PlanPlacement &placement : plan.placements)
    plan.instances.push_back(placement.instance);
  llvm::sort(plan.instances);
  plan.instances.erase(
      std::unique(plan.instances.begin(), plan.instances.end()),
      plan.instances.end());
  for (const PlanConnection &connection : plan.connectionPlans) {
    plan.connections.push_back(connection.id);
  }
  llvm::sort(plan.connections);

  llvm::sort(plan.placements,
             [](const PlanPlacement &lhs, const PlanPlacement &rhs) {
               if (lhs.executor != rhs.executor)
                 return lhs.executor < rhs.executor;
               auto sorted = [](const llvm::StringMap<std::string> &map) {
                 std::vector<std::string> parts;
                 for (const auto &entry : map) {
                   std::string text = entry.first().str();
                   text += '=';
                   text += entry.second;
                   parts.push_back(std::move(text));
                 }
                 llvm::sort(parts);
                 return parts;
               };
               std::vector<std::string> lhsMemories = sorted(lhs.memories);
               std::vector<std::string> rhsMemories = sorted(rhs.memories);
               if (lhsMemories != rhsMemories)
                 return lhsMemories < rhsMemories;
               std::vector<std::string> lhsLayouts = sorted(lhs.layouts);
               std::vector<std::string> rhsLayouts = sorted(rhs.layouts);
               if (lhsLayouts != rhsLayouts)
                 return lhsLayouts < rhsLayouts;
               if (lhs.node != rhs.node)
                 return lhs.node < rhs.node;
               return lhs.instance < rhs.instance;
             });
  llvm::sort(plan.connectionPlans,
             [](const PlanConnection &lhs, const PlanConnection &rhs) {
               return lhs.id < rhs.id;
             });
  return plan;
}

} // namespace mlir::llk::mapping
