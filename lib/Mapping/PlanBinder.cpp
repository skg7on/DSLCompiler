//===- PlanBinder.cpp - Materialize a selected plan (D7/#50) -------------===//

#include "LLK/Mapping/PlanBinder.h"

#include "LLK/Mapping/Diagnostics.h"
#include "LLK/Mapping/WorkloadGraph.h"

#include "mlir/AsmParser/AsmParser.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Verifier.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Error.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace mlir::llk::mapping {

namespace {

using machine::MachineModel;

/// The generic, target-neutral metadata containers (design §18.1).
constexpr llvm::StringLiteral kPlanAttr = "micro.plan";
constexpr llvm::StringLiteral kMappingAttr = "micro.mapping";
constexpr llvm::StringLiteral kRoutesAttr = "micro.routes";

/// Stable, greppable reasons for a connection the binder cannot materialize
/// (design §18.2). They are part of the report contract, so they are named
/// constants rather than free-form prose.
constexpr llvm::StringLiteral kReduceReason = "reduce_not_materialized";
constexpr llvm::StringLiteral kHoplessRouteReason =
    "route_has_no_hop_to_materialize";

/// The reason a connection with no Micro operation form cannot be
/// materialized. Every kind the binder emits inline -- `Direct`, the
/// movements, replication, and the layout transform -- must never reach here.
llvm::StringRef unmaterializedReason(ConnectionKind kind) {
  switch (kind) {
  case ConnectionKind::Reduce:
    return kReduceReason;
  case ConnectionKind::Direct:
  case ConnectionKind::Transfer:
  case ConnectionKind::TransferAndTransform:
  case ConnectionKind::Replicate:
  case ConnectionKind::LayoutTransform:
    llvm_unreachable("this connection kind is materialized inline");
  }
  llvm_unreachable("all connection kinds handled");
}

llvm::Error bindError(const std::string &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(), message);
}

/// The identity of a connection's layout transform: the two families and the
/// two solved maps. Two connections that move one value to one memory under
/// *different* transforms are different work, so they must not share a copy
/// chain -- the chain key includes this.
std::string transformIdentity(const PlanConnection &connection) {
  if (!connection.transform)
    return {};
  auto mapText = [](mlir::AffineMap map) {
    if (!map)
      return std::string("<null>");
    std::string printed;
    llvm::raw_string_ostream stream(printed);
    map.print(stream);
    return stream.str();
  };
  std::string text = connection.transform->srcLayout;
  text += "->";
  text += connection.transform->dstLayout;
  text += ":src=";
  text += mapText(connection.transform->srcMap);
  text += ":dst=";
  text += mapText(connection.transform->dstMap);
  return text;
}

/// A phase-2 (machine-aware) verification failure carrying its stable §22.3
/// code. The code is the interface a caller may group or switch on; the message
/// is human detail that may change between releases (see Diagnostics.h), so the
/// two are joined as the code's stable string, a colon, and the message. Which
/// code fits a violation is decided where the violation is detected -- the
/// strings themselves come from `stringifyDiagnosticCode`.
llvm::Error verifyError(DiagnosticCode code, const std::string &message) {
  return bindError((stringifyDiagnosticCode(code) + ": " + message).str());
}

mlir::IntegerAttr u64Attr(mlir::MLIRContext *context, uint64_t value) {
  return mlir::IntegerAttr::get(mlir::IntegerType::get(context, 64),
                                static_cast<int64_t>(value));
}

/// A workload node rebuilt with the binder's own bookkeeping attributes
/// removed, so a rule predicate reads the operation's declared attributes
/// (`op = "add"`) rather than the metadata the binder stamped onto it.
WorkloadNode strippedWorkloadNode(const WorkloadNode &node) {
  if (!node.attributes)
    return node;
  llvm::SmallVector<mlir::NamedAttribute> kept;
  bool stripped = false;
  for (mlir::NamedAttribute attribute : node.attributes) {
    llvm::StringRef name = attribute.getName().getValue();
    if (name == kMappingAttr || name == "micro.value" ||
        name == "micro.dst_node") {
      stripped = true;
      continue;
    }
    kept.push_back(attribute);
  }
  if (!stripped)
    return node;
  WorkloadNode copy = node;
  copy.attributes =
      mlir::DictionaryAttr::get(node.attributes.getContext(), kept);
  return copy;
}

/// The `micro.kernel` an operation sits in, or null when it is outside one.
mlir::Operation *enclosingKernel(mlir::Operation *op) {
  for (mlir::Operation *parent = op->getParentOp(); parent;
       parent = parent->getParentOp())
    if (parent->getName().getStringRef() == "micro.kernel")
      return parent;
  return nullptr;
}

/// A `micro.kernel`'s symbol name for diagnostics, or a placeholder when it has
/// none.
std::string kernelLabel(mlir::Operation *kernel) {
  if (auto symbol = kernel->getAttrOfType<mlir::StringAttr>("sym_name"))
    return ("'" + symbol.getValue() + "'").str();
  return "<unnamed>";
}

/// Reads a required string field from a metadata dictionary with a checked
/// cast. Generic mapping metadata is untrusted (design §25.1): an absent field
/// and a wrongly-typed one are distinct, stable diagnostics rather than a cast
/// that aborts the process.
llvm::Expected<std::string> requiredString(mlir::DictionaryAttr dict,
                                           llvm::StringRef name,
                                           llvm::StringRef where) {
  mlir::Attribute raw = dict.get(name);
  if (!raw)
    return bindError((where + ": missing '" + name + "'").str());
  auto text = mlir::dyn_cast<mlir::StringAttr>(raw);
  if (!text)
    return bindError((where + ": '" + name + "' is not a string").str());
  return text.getValue().str();
}

/// Reads a `key = "value"` string-map attribute (`memories` or `layouts`),
/// type-checking the container and every entry.
llvm::Expected<llvm::StringMap<std::string>>
readStringMap(mlir::Attribute raw, llvm::StringRef name,
              llvm::StringRef where) {
  auto entries = mlir::dyn_cast<mlir::DictionaryAttr>(raw);
  if (!entries)
    return bindError((where + ": '" + name + "' is not a dictionary").str());
  llvm::StringMap<std::string> values;
  for (const mlir::NamedAttribute &entry : entries) {
    auto value = mlir::dyn_cast<mlir::StringAttr>(entry.getValue());
    if (!value)
      return bindError((where + ": '" + name + "' entry '" +
                        entry.getName().str() + "' is not a string")
                           .str());
    values[entry.getName()] = value.getValue().str();
  }
  return values;
}

/// Reads an array-of-strings attribute, type-checking the container and every
/// element.
llvm::Expected<llvm::SmallVector<std::string, 4>>
readStringArray(mlir::Attribute raw, llvm::StringRef name,
                llvm::StringRef where) {
  llvm::SmallVector<std::string, 4> values;
  auto array = mlir::dyn_cast<mlir::ArrayAttr>(raw);
  if (!array)
    return bindError((where + ": '" + name + "' is not an array").str());
  for (mlir::Attribute element : array) {
    auto text = mlir::dyn_cast<mlir::StringAttr>(element);
    if (!text)
      return bindError(
          (where + ": '" + name + "' has a non-string entry").str());
    values.push_back(text.getValue().str());
  }
  return values;
}

/// One search value -- an integer or a string -- as a typed attribute, so a
/// persisted parameter round-trips as its own type rather than being flattened
/// to text.
mlir::Attribute searchValueAttr(mlir::MLIRContext *context,
                                const SearchValue &value) {
  if (const int64_t *integer = std::get_if<int64_t>(&value))
    return mlir::IntegerAttr::get(mlir::IntegerType::get(context, 64),
                                  *integer);
  return mlir::StringAttr::get(context, std::get<std::string>(value));
}

/// The solved parameterization of each layout requirement, keyed exactly as the
/// plan's `layoutSolutions` are (the layout class, index-disambiguated when one
/// class is required by several ports). Each value is a dictionary of parameter
/// name -> typed value, so the bound IR states the concrete instantiation the
/// plan chose (`VW = 8`), not merely the layout family. Keys and parameter
/// names are sorted, so the persisted form is deterministic.
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

mlir::DictionaryAttr
stringMapAttr(mlir::MLIRContext *context,
              const llvm::StringMap<std::string> &entries) {
  llvm::SmallVector<mlir::NamedAttribute> attributes;
  std::vector<std::pair<std::string, std::string>> ordered;
  for (const auto &entry : entries)
    ordered.emplace_back(entry.first().str(), entry.second);
  llvm::sort(ordered);
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

/// The module's single `micro.kernel`. Fails when there is none (nothing to
/// bind) or more than one (no selector exists, so binding "the first" would
/// silently ignore the rest).
llvm::Expected<mlir::Operation *> findKernel(mlir::ModuleOp module) {
  llvm::SmallVector<mlir::Operation *, 2> kernels;
  module->walk([&](mlir::Operation *op) {
    if (op->getName().getStringRef() == "micro.kernel")
      kernels.push_back(op);
  });
  if (kernels.empty())
    return bindError("bindPlan: the source module has no micro.kernel");
  if (kernels.size() > 1)
    return bindError("bindPlan: the source module has " +
                     std::to_string(kernels.size()) +
                     " micro.kernels; one kernel per module is required");
  return kernels.front();
}

} // namespace

llvm::Expected<BoundPlan> bindPlan(mlir::ModuleOp source,
                                   const CoveringPlan &plan,
                                   const MappingTarget &target,
                                   BindContract contract) {
  mlir::OwningOpRef<mlir::ModuleOp> module(
      mlir::cast<mlir::ModuleOp>(source->clone()));
  llvm::Expected<mlir::Operation *> resolvedKernel = findKernel(*module);
  if (!resolvedKernel)
    return resolvedKernel.takeError();
  mlir::Operation *kernel = *resolvedKernel;

  WorkloadGraphBinding binding;
  llvm::Expected<WorkloadGraph> graph = extractWorkloadGraph(kernel, &binding);
  if (!graph)
    return graph.takeError();

  mlir::MLIRContext *context = module->getContext();

  // --- kernel metadata -------------------------------------------------
  llvm::SmallVector<mlir::NamedAttribute> planAttributes;
  planAttributes.emplace_back(mlir::StringAttr::get(context, "id"),
                              u64Attr(context, plan.id));
  planAttributes.emplace_back(mlir::StringAttr::get(context, "binding_hash"),
                              u64Attr(context, plan.sourceBindingHash));
  planAttributes.emplace_back(
      mlir::StringAttr::get(context, "truncated"),
      mlir::BoolAttr::get(context, plan.diagnostics.searchTruncated));
  kernel->setAttr(kPlanAttr,
                  mlir::DictionaryAttr::get(context, planAttributes));

  // --- per-placement metadata -----------------------------------------
  for (const PlanPlacement &placement : plan.placements) {
    mlir::Operation *op = binding.opFor(placement.node);
    if (!op)
      return bindError("bindPlan: the plan covers node " +
                       std::to_string(placement.node) +
                       ", which the kernel does not contain");
    const RuleDef *rule = target.rules().find(placement.rule);
    if (!rule)
      return bindError("bindPlan: the plan selects unknown rule '" +
                       placement.rule + "'");

    llvm::StringMap<std::string> memories;
    for (const auto &entry : placement.memories)
      memories[entry.first()] = entry.second;
    llvm::StringMap<std::string> layouts;
    for (const auto &entry : placement.layouts)
      layouts[entry.first()] = entry.second;

    llvm::SmallVector<mlir::NamedAttribute> attributes;
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
    // The opaque bundle's typed parameters, carried unchanged so a target
    // emitter sees the resolved bundle the plan selected rather than only its
    // name. Absent when the rule declares none (a null attribute has no impl to
    // walk, so it is not written).
    if (placement.bundle.parameters)
      attributes.emplace_back(
          mlir::StringAttr::get(context, "bundle_parameters"),
          placement.bundle.parameters);
    // The solved parameterization of each layout requirement, so a materializer
    // (or a report reader) can state the concrete instantiation the plan chose.
    if (!placement.layoutSolutions.empty())
      attributes.emplace_back(
          mlir::StringAttr::get(context, "layout_parameters"),
          layoutParametersAttr(context, placement.layoutSolutions));
    op->setAttr(kMappingAttr, mlir::DictionaryAttr::get(context, attributes));
  }

  // --- route metadata --------------------------------------------------
  llvm::SmallVector<mlir::Attribute> routes;
  for (const PlanConnection &connection : plan.connectionPlans) {
    llvm::SmallVector<mlir::NamedAttribute> attributes;
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
    // Which placed instances this connection serves. A reader of the route
    // metadata can then see whose dataflow the connection carries, rather than
    // having to re-derive it from the placements: without it the connection is
    // "a movement of value N", not "the movement value N's consumer C reads".
    llvm::SmallVector<mlir::Attribute> consumers;
    for (InstanceId consumer : connection.consumers)
      consumers.push_back(u64Attr(context, consumer));
    attributes.emplace_back(mlir::StringAttr::get(context, "consumers"),
                            mlir::ArrayAttr::get(context, consumers));
    if (connection.transform) {
      llvm::SmallVector<mlir::NamedAttribute> transform;
      transform.emplace_back(
          mlir::StringAttr::get(context, "src"),
          mlir::StringAttr::get(context, connection.transform->srcLayout));
      transform.emplace_back(
          mlir::StringAttr::get(context, "dst"),
          mlir::StringAttr::get(context, connection.transform->dstLayout));
      attributes.emplace_back(mlir::StringAttr::get(context, "transform"),
                              mlir::DictionaryAttr::get(context, transform));
    }
    routes.push_back(mlir::DictionaryAttr::get(context, attributes));
  }
  kernel->setAttr(kRoutesAttr, mlir::ArrayAttr::get(context, routes));

  BoundPlan bound;
  // Named `machineModel` on purpose: a local called `machine` would shadow the
  // `machine` namespace this file needs for `machine::MemoryNode`.
  const MachineModel &machineModel = target.machine();

  // --- materialize the movement (design §18.2) -------------------------
  //
  // One copy chain per (value, route). A value that fans out along one route is
  // served by a single chain; a value whose consumers need *different* routes
  // gets one chain per route, because one chain cannot land the value in two
  // memories. Each chain rewires only the consumers its connection selected, so
  // a copy that lands in one memory never redirects a reader the plan placed
  // elsewhere.
  //
  // Chains are recorded as they are emitted and rewired in a second pass: a
  // later connection that shares a chain's route can still add its consumers to
  // it, and the rewire must see the final set.
  mlir::OpBuilder builder(context);

  // Which operation each placed instance is, so a connection's consumer
  // instances resolve to the operations that must read its result.
  llvm::DenseMap<InstanceId, mlir::Operation *> instanceOps;
  for (const PlanPlacement &placement : plan.placements)
    if (mlir::Operation *op = binding.opFor(placement.node))
      instanceOps[placement.instance] = op;

  struct MaterializedChain {
    WorkloadValueId value = 0;
    llvm::SmallVector<MemoryNodeId> route;
    /// The chain's layout transform identity, so two connections moving one
    /// value to one memory under *different* transforms do not share a chain.
    /// Empty for a connection with no transform.
    std::string transform;
    /// What a consumer of this chain reads: the last copy's result, or the
    /// transform's result when the connection carries one.
    mlir::Value result;
    llvm::SmallVector<mlir::Operation *, 4> consumers;
  };
  std::vector<MaterializedChain> chains;

  // The covered node that writes `value`, or null. A transform-only connection
  // needs it just as a movement does.
  auto producerOperationFor = [&](WorkloadValueId value) -> mlir::Operation * {
    for (const WorkloadNode &node : graph->getNodes()) {
      bool writes = llvm::any_of(node.outputs, [&](const WorkloadPort &port) {
        return port.value == value;
      });
      if (writes)
        return binding.opFor(node.id);
    }
    return nullptr;
  };

  // Emits the target-neutral representation change: the value read through
  // `transform.srcMap` is written through `transform.dstMap`. The operation
  // names the layouts by their index relation, never by a target-owned id, so
  // it stays generic (design §13.4).
  auto emitTransform = [&](mlir::Operation *anchor, mlir::Value input,
                           const LayoutTransform &transform) -> mlir::Value {
    builder.setInsertionPointAfter(anchor);
    mlir::OperationState state(anchor->getLoc(), "micro.transform");
    state.addOperands(input);
    state.addTypes({input.getType()});
    if (transform.srcMap)
      state.addAttribute("src_map", mlir::AffineMapAttr::get(transform.srcMap));
    if (transform.dstMap)
      state.addAttribute("dst_map", mlir::AffineMapAttr::get(transform.dstMap));
    return builder.create(state)->getResult(0);
  };

  for (const PlanConnection &connection : plan.connectionPlans) {
    // A `Direct` connection is materialized by construction: the producer wrote
    // the value to the memory the consumer reads, in a layout the consumer
    // addresses, so there is nothing to emit and nothing to report.
    if (connection.kind == ConnectionKind::Direct)
      continue;

    // The kinds a Micro operation can express: the movements and replication (a
    // fan-out copy, served by the same copy chain), plus a layout conversion,
    // which becomes a target-neutral `micro.transform`. `Reduce` still has no
    // Micro operation form, so it is reported rather than dropped (§18.2).
    if (connection.kind != ConnectionKind::Transfer &&
        connection.kind != ConnectionKind::TransferAndTransform &&
        connection.kind != ConnectionKind::Replicate &&
        connection.kind != ConnectionKind::LayoutTransform) {
      bound.unmaterialized.push_back(
          "value " + std::to_string(connection.value) + ": " +
          unmaterializedReason(connection.kind).str());
      continue;
    }

    // The operations this connection serves. Empty for a plan built without
    // consumer associations: nothing is rewired, but the chain is emitted.
    llvm::SmallVector<mlir::Operation *, 4> consumers;
    for (InstanceId id : connection.consumers)
      if (mlir::Operation *op = instanceOps.lookup(id))
        if (!llvm::is_contained(consumers, op))
          consumers.push_back(op);

    const std::string transform = transformIdentity(connection);

    // An already-emitted chain for this (value, route, transform) serves this
    // connection too -- it lands the value in the same memory under the same
    // layout, so the same result is read.
    auto existing = llvm::find_if(chains, [&](const MaterializedChain &chain) {
      return chain.value == connection.value &&
             chain.route == connection.route && chain.transform == transform;
    });
    if (existing != chains.end()) {
      for (mlir::Operation *op : consumers)
        if (!llvm::is_contained(existing->consumers, op))
          existing->consumers.push_back(op);
      continue;
    }

    // A transform-only connection moves nothing: it converts the value in
    // place, where the producer already wrote it, so there is no route to walk
    // and no memory to check. It is materialized as one `micro.transform`.
    if (connection.kind == ConnectionKind::LayoutTransform) {
      if (!connection.transform) {
        bound.unmaterialized.push_back(
            "value " + std::to_string(connection.value) +
            ": layout transform carries no maps to emit");
        continue;
      }
      mlir::Value input = binding.valueFor(connection.value);
      mlir::Operation *producer = producerOperationFor(connection.value);
      if (!producer || !input) {
        bound.unmaterialized.push_back(
            "value " + std::to_string(connection.value) +
            ": no producing operation in the kernel");
        continue;
      }
      if (!mlir::isa<mlir::ShapedType>(input.getType())) {
        bound.unmaterialized.push_back(
            "value " + std::to_string(connection.value) +
            ": 'micro.transform' needs a shaped or tile type, which the binder "
            "cannot construct generically for this value");
        continue;
      }
      MaterializedChain chain;
      chain.value = connection.value;
      chain.route = connection.route;
      chain.transform = transform;
      chain.result = emitTransform(producer, input, *connection.transform);
      chain.consumers = std::move(consumers);
      chains.push_back(std::move(chain));
      continue;
    }

    // A movement needs at least one hop between two memories; a shorter route
    // has nothing to emit. This is unreachable for the current placement code,
    // but a selected connection must never vanish silently. A transform-only
    // connection moves nothing -- its route is a single memory -- so the check
    // does not apply to it.
    const bool moves =
        connection.kind == ConnectionKind::Transfer ||
        connection.kind == ConnectionKind::TransferAndTransform ||
        connection.kind == ConnectionKind::Replicate;
    if (moves && connection.route.size() < 2) {
      bound.unmaterialized.push_back("value " +
                                     std::to_string(connection.value) + ": " +
                                     kHoplessRouteReason.str());
      continue;
    }

    // The producer is the covered node that writes this value; the consumer is
    // any covered node that reads it.
    mlir::Value value = binding.valueFor(connection.value);
    mlir::Operation *producer = producerOperationFor(connection.value);
    if (!producer || !value) {
      bound.unmaterialized.push_back("value " +
                                     std::to_string(connection.value) +
                                     ": no producing operation in the kernel");
      continue;
    }
    if (!mlir::isa<mlir::ShapedType>(value.getType())) {
      bound.unmaterialized.push_back(
          "value " + std::to_string(connection.value) +
          ": 'micro.tile_async_copy' needs a destination-memory tile type, "
          "which the binder cannot construct generically");
      continue;
    }

    const machine::MemoryNode *source =
        machineModel.findMemory(connection.route.front());
    const machine::MemoryNode *destination =
        machineModel.findMemory(connection.route.back());
    if (!source || !destination) {
      bound.unmaterialized.push_back("value " +
                                     std::to_string(connection.value) +
                                     ": route names an unknown memory");
      continue;
    }
    mlir::Attribute srcMemory =
        mlir::parseAttribute(("#micro.memory<" + source->kind + ">"), context);
    mlir::Attribute dstMemory = mlir::parseAttribute(
        ("#micro.memory<" + destination->kind + ">"), context);
    if (!srcMemory || !dstMemory || source->kind == destination->kind) {
      // A route whose endpoints are the same memory space has nothing to move
      // between, and `micro.async_copy` insists its two spaces differ.
      bound.unmaterialized.push_back(
          "value " + std::to_string(connection.value) +
          ": route endpoints share a micro memory kind");
      continue;
    }
    mlir::Type tokenType = mlir::parseType("!micro.async_token", context);
    if (!tokenType) {
      bound.unmaterialized.push_back("value " +
                                     std::to_string(connection.value) +
                                     ": async token type is not registered");
      continue;
    }

    // One copy per hop, chained: the value moves to the first staging memory,
    // then on to the next, until it reaches the consumer's. Each hop is
    // followed by its own wait, and the intermediate value *is* the staging
    // storage -- `micro.async_copy` produces a value in the destination
    // memory, so there is no separate buffer to allocate for it.
    builder.setInsertionPointAfter(producer);
    mlir::Value current = value;
    mlir::Operation *lastCopy = nullptr;
    bool materialized = true;
    for (size_t hop = 1; hop < connection.route.size(); ++hop) {
      const machine::MemoryNode *from =
          machineModel.findMemory(connection.route[hop - 1]);
      const machine::MemoryNode *to =
          machineModel.findMemory(connection.route[hop]);
      if (!from || !to || from->kind == to->kind) {
        materialized = false;
        break;
      }
      mlir::Attribute hopSrc =
          mlir::parseAttribute(("#micro.memory<" + from->kind + ">"), context);
      mlir::Attribute hopDst =
          mlir::parseAttribute(("#micro.memory<" + to->kind + ">"), context);
      if (!hopSrc || !hopDst) {
        materialized = false;
        break;
      }

      llvm::SmallVector<mlir::Type> copyResultTypes{value.getType(), tokenType};
      mlir::OperationState copyState(producer->getLoc(), "micro.async_copy");
      copyState.addOperands(current);
      copyState.addTypes(copyResultTypes);
      copyState.addAttribute("src_memory", hopSrc);
      copyState.addAttribute("dst_memory", hopDst);
      // Target-neutral identity for the performance model (design
      // §12.4/§23.3): which connection this hop belongs to, and the concrete
      // memory node it lands in. Without it a movement is indistinguishable
      // from the other movements that share its endpoint memory *kinds*, and
      // is charged the wrong link.
      copyState.addAttribute("micro.value", u64Attr(context, connection.value));
      copyState.addAttribute("micro.dst_node",
                             mlir::StringAttr::get(context, to->id));
      mlir::Operation *copy = builder.create(copyState);

      mlir::OperationState waitState(producer->getLoc(), "micro.wait");
      waitState.addOperands(copy->getResult(1));
      builder.create(waitState);

      lastCopy = copy;
      current = copy->getResult(0);
    }

    if (!materialized || !lastCopy) {
      bound.unmaterialized.push_back(
          "value " + std::to_string(connection.value) +
          ": a route hop names memories the machine cannot carry data between");
      continue;
    }

    // A transfer-and-transform moved the value; the conversion itself is the
    // same target-neutral `micro.transform`, emitted after the movement so the
    // consumer reads the layout it requires.
    mlir::Value produced = lastCopy->getResult(0);
    if (connection.kind == ConnectionKind::TransferAndTransform) {
      if (!connection.transform) {
        bound.unmaterialized.push_back(
            "value " + std::to_string(connection.value) +
            ": transfer-and-transform carries no maps to emit");
        continue;
      }
      produced = emitTransform(lastCopy, produced, *connection.transform);
    }

    // Record the chain; its consumers are rewired once every chain is emitted,
    // so a later connection sharing its route can still be added to it.
    MaterializedChain chain;
    chain.value = connection.value;
    chain.route = connection.route;
    chain.transform = transform;
    chain.result = produced;
    chain.consumers = std::move(consumers);
    chains.push_back(std::move(chain));
  }

  // Rewire each chain's own consumers to what it produced. Only these
  // operations are touched: a reader the plan placed on another route keeps
  // reading what that route produced.
  for (const MaterializedChain &chain : chains) {
    mlir::Value original = binding.valueFor(chain.value);
    if (!original || !chain.result)
      continue;
    for (mlir::Operation *consumer : chain.consumers)
      for (mlir::OpOperand &use : consumer->getOpOperands())
        if (use.get() == original)
          use.set(chain.result);
  }

  // An executable contract refuses a plan it could not fully materialize: the
  // caller asked for code a backend may run, and a partial binding would omit
  // part of the selected plan without saying so in a form the backend checks.
  // The partial contract keeps the report, which is the analysis contract.
  if (contract == BindContract::Executable && !bound.unmaterialized.empty()) {
    std::string message =
        "bindPlan: the plan is not fully executable; " +
        std::to_string(bound.unmaterialized.size()) +
        " execution-affecting decision(s) could not be materialized:";
    for (const std::string &reason : bound.unmaterialized)
      message += "\n  " + reason;
    return bindError(message);
  }

  bound.planId = plan.id;
  bound.kernel = kernel;
  bound.module = std::move(module);
  return std::move(bound);
}

llvm::Error verifyMappedMicroIR(mlir::ModuleOp module,
                                const MappingTarget &target) {
  // 1. structural
  if (mlir::failed(mlir::verify(module)))
    return bindError("verifyMappedMicroIR: structural verification failed");

  const MachineModel &machine = target.machine();
  llvm::Error failure = llvm::Error::success();
  /// Records the first violation. Callers return immediately afterwards, so a
  /// later walk (guarded by `if (failure) return;`) never overwrites it.
  auto fail = [&](DiagnosticCode code, std::string message) {
    failure = verifyError(code, std::move(message));
  };
  /// Reports malformed generic metadata with the §22.3 code that names it
  /// (design §25.1). The reader already produced the detail.
  auto failMetadata = [&](llvm::Error error) {
    fail(DiagnosticCode::InvalidMappingMetadata,
         llvm::toString(std::move(error)));
  };

  // --- 2a. kernel completeness -------------------------------------------
  //
  // Phase 2 verifies the *mapped* form of a kernel. A kernel without
  // `micro.plan` was never bound: it carries no selection to resolve, so
  // accepting it would report success for exactly the input the pass exists to
  // reject. Every kernel is checked, so a module that maps one of two kernels
  // fails on the second rather than passing on the first. Inside a mapped
  // kernel, every workload operation that is not a binder-emitted movement must
  // carry its own `micro.mapping`, or the kernel is only partly mapped.
  module->walk([&](mlir::Operation *kernel) {
    if (failure)
      return;
    if (kernel->getName().getStringRef() != "micro.kernel")
      return;
    const std::string where = "kernel " + kernelLabel(kernel);

    auto plan = kernel->getAttrOfType<mlir::DictionaryAttr>(kPlanAttr);
    if (!plan) {
      fail(DiagnosticCode::NoMatchingRule,
           where + " is not mapped: it has no micro.plan");
      return;
    }
    // The plan container is generic metadata; validate its shape before read.
    for (llvm::StringRef name : {"id", "binding_hash"}) {
      mlir::Attribute raw = plan.get(name);
      if (!raw) {
        failMetadata(
            bindError(where + ": micro.plan is missing '" + name.str() + "'"));
        return;
      }
      if (!mlir::isa<mlir::IntegerAttr>(raw)) {
        failMetadata(bindError(where + ": micro.plan '" + name.str() +
                               "' is not an integer"));
        return;
      }
    }
    if (mlir::Attribute truncated = plan.get("truncated"))
      if (!mlir::isa<mlir::BoolAttr>(truncated)) {
        failMetadata(
            bindError(where + ": micro.plan 'truncated' is not a bool"));
        return;
      }

    kernel->walk([&](mlir::Operation *op) {
      if (failure)
        return;
      if (!isWorkloadNodeOp(op->getName()))
        return;
      // A binder-emitted movement is a materialized connection, not a workload
      // operation that needed a rule; it is identified by the connection
      // bookkeeping the binder stamps on the copy.
      if (op->hasAttr("micro.value"))
        return;
      if (!op->getAttr(kMappingAttr))
        fail(DiagnosticCode::NoMatchingRule,
             where + ": op '" + op->getName().getStringRef().str() +
                 "' carries no micro.mapping");
    });
  });
  if (failure)
    return failure;

  // --- 2b. per-operation metadata ----------------------------------------
  //
  // Rule selection is re-checked against the *original* workload endpoints:
  // each kernel's graph is re-extracted once and a recorded operation's node
  // looked up by its own operation, so a predicate reads the boundary ports
  // generation classified rather than the metadata the binder stamped.
  struct KernelWorkload {
    WorkloadGraph graph;
    llvm::DenseMap<mlir::Operation *, const WorkloadNode *> byOp;
  };
  llvm::DenseMap<mlir::Operation *, std::unique_ptr<KernelWorkload>> workloads;
  auto workloadNodeFor = [&](mlir::Operation *op) -> const WorkloadNode * {
    mlir::Operation *kernel = enclosingKernel(op);
    if (!kernel)
      return nullptr;
    auto entry = workloads.find(kernel);
    if (entry == workloads.end()) {
      WorkloadGraphBinding binding;
      llvm::Expected<WorkloadGraph> graph =
          extractWorkloadGraph(kernel, &binding);
      auto fresh = std::make_unique<KernelWorkload>();
      if (graph) {
        fresh->graph = std::move(*graph);
        for (const auto &pair : binding.nodeOps)
          fresh->byOp[pair.second] = fresh->graph.findNode(pair.first);
      } else {
        llvm::consumeError(graph.takeError());
      }
      entry = workloads.insert({kernel, std::move(fresh)}).first;
    }
    auto found = entry->second->byOp.find(op);
    return found == entry->second->byOp.end() ? nullptr : found->second;
  };

  module->walk([&](mlir::Operation *op) {
    if (failure)
      return;
    mlir::Attribute rawMapping = op->getAttr(kMappingAttr);
    if (!rawMapping)
      return;
    const std::string where =
        "mapped op '" + op->getName().getStringRef().str() + "'";
    auto mapping = mlir::dyn_cast<mlir::DictionaryAttr>(rawMapping);
    if (!mapping) {
      failMetadata(bindError(where + ": micro.mapping is not a dictionary"));
      return;
    }

    llvm::Expected<std::string> ruleId = requiredString(mapping, "rule", where);
    if (!ruleId) {
      failMetadata(ruleId.takeError());
      return;
    }
    const RuleDef *rule = target.rules().find(*ruleId);
    if (!rule) {
      fail(DiagnosticCode::NoMatchingRule,
           where + ": unknown rule '" + *ruleId + "'");
      return;
    }
    // The recorded rule must implement *this* operation, not merely exist: a
    // vector op labelled with an MMA rule is not a mapping.
    if (rule->matchOp != op->getName().getStringRef()) {
      fail(DiagnosticCode::NoMatchingRule,
           where + ": rule '" + *ruleId + "' implements '" + rule->matchOp +
               "', not '" + op->getName().getStringRef().str() + "'");
      return;
    }

    llvm::Expected<std::string> executor =
        requiredString(mapping, "executor", where);
    if (!executor) {
      failMetadata(executor.takeError());
      return;
    }
    if (!machine.findExecutor(*executor)) {
      fail(DiagnosticCode::NoLegalExecutor,
           where + ": unknown executor '" + *executor + "'");
      return;
    }

    // Memory role bindings: every bound id must resolve and be visible, and a
    // rule that requires a memory kind must have bound one.
    llvm::StringMap<std::string> memories;
    if (mlir::Attribute rawMemories = mapping.get("memories")) {
      llvm::Expected<llvm::StringMap<std::string>> read =
          readStringMap(rawMemories, "memories", where);
      if (!read) {
        failMetadata(read.takeError());
        return;
      }
      memories = std::move(*read);
      for (const auto &entry : memories) {
        if (!machine.findMemory(entry.second)) {
          fail(DiagnosticCode::NoMemoryRoute,
               where + ": unknown memory '" + entry.second + "'");
          return;
        }
        if (!machine.isVisible(entry.second, *executor)) {
          fail(DiagnosticCode::NoMemoryRoute,
               where + ": executor '" + *executor + "' cannot see memory '" +
                   entry.second + "'");
          return;
        }
      }
    }
    for (const KindRequirement &requirement : rule->kindRequirements) {
      if (requirement.role != "memory")
        continue;
      if (!memories.count(requirement.kind)) {
        fail(DiagnosticCode::NoLegalExecutor,
             where + ": rule '" + *ruleId + "' requires a '" +
                 requirement.kind +
                 "' memory, which the mapping does not bind");
        return;
      }
    }

    // The recorded rule must still be fully legal for this operation under
    // generation's own predicates, constraints and machine capabilities: a
    // matching mnemonic and existing ids are not enough. The node is the
    // original workload endpoint, with the binder's bookkeeping attributes
    // removed, so a predicate reads the operation rather than the metadata.
    if (const WorkloadNode *node = workloadNodeFor(op)) {
      WorkloadNode endpoint = strippedWorkloadNode(*node);
      RecordedRuleSelection selection;
      selection.executor = *executor;
      for (const auto &entry : memories)
        selection.memories[entry.first()] = entry.second;
      // Resolved rule parameters are not persisted on this schema yet, so the
      // selection carries none and verification falls back to the same
      // existential requirement check generation applied. Schema v2 records
      // them and this call validates the recorded assignment exactly.
      if (llvm::Error error =
              verifyRuleSelection(*rule, endpoint, machine, selection, where)) {
        failure = std::move(error);
        return;
      }
    }

    // Layout role bindings: every bound id must resolve, and a rule that
    // requires a layout on a port must have bound it.
    llvm::StringMap<std::string> layouts;
    if (mlir::Attribute rawLayouts = mapping.get("layouts")) {
      llvm::Expected<llvm::StringMap<std::string>> read =
          readStringMap(rawLayouts, "layouts", where);
      if (!read) {
        failMetadata(read.takeError());
        return;
      }
      layouts = std::move(*read);
      for (const auto &entry : layouts) {
        if (!target.layouts().find(entry.second)) {
          fail(DiagnosticCode::NoLegalLayout,
               where + ": unknown layout '" + entry.second + "'");
          return;
        }
      }
    }
    for (const RuleLayoutRequirement &requirement : rule->layoutRequirements) {
      if (!layouts.count(requirement.layoutId)) {
        fail(DiagnosticCode::NoLegalLayout,
             where + ": rule '" + *ruleId + "' requires layout '" +
                 requirement.layoutId + "' on port '" + requirement.port +
                 "', which the mapping does not bind");
        return;
      }
    }

    // 3. target: the recorded bundle and emitter must be the selected rule's,
    // the emitter must be one the target declares, and the target's own
    // emitter must accept the bundle before lowering (design §18.3, phase 3).
    llvm::Expected<std::string> bundleName =
        requiredString(mapping, "bundle", where);
    if (!bundleName) {
      failMetadata(bundleName.takeError());
      return;
    }
    if (*bundleName != rule->bundle) {
      fail(DiagnosticCode::TargetBundleInvalid,
           where + ": rule '" + *ruleId + "' selects bundle '" + rule->bundle +
               "', but the mapping records '" + *bundleName + "'");
      return;
    }
    llvm::Expected<std::string> emitterKey =
        requiredString(mapping, "emitter", where);
    if (!emitterKey) {
      failMetadata(emitterKey.takeError());
      return;
    }
    // Existence first, then rule compatibility: a key the target does not
    // declare is "unknown" even when it also mismatches the rule, so the
    // diagnostic names the more fundamental problem.
    if (!target.isKnownEmitter(*emitterKey)) {
      fail(DiagnosticCode::TargetBundleInvalid,
           where + ": unknown emitter '" + *emitterKey + "'");
      return;
    }
    if (*emitterKey != rule->emitter) {
      fail(DiagnosticCode::TargetBundleInvalid,
           where + ": rule '" + *ruleId + "' selects emitter '" +
               rule->emitter + "', but the mapping records '" + *emitterKey +
               "'");
      return;
    }
    // The persisted bundle parameters and solved layout parameters are generic
    // metadata too, so their shape is validated with checked casts before they
    // are handed to the plugin or a reader.
    if (mlir::Attribute rawBundleParameters =
            mapping.get("bundle_parameters")) {
      auto parameters =
          mlir::dyn_cast<mlir::DictionaryAttr>(rawBundleParameters);
      if (!parameters) {
        failMetadata(
            bindError(where + ": 'bundle_parameters' is not a dictionary"));
        return;
      }
      for (const mlir::NamedAttribute &entry : parameters)
        if (!mlir::isa<mlir::IntegerAttr>(entry.getValue()) &&
            !mlir::isa<mlir::StringAttr>(entry.getValue())) {
          failMetadata(bindError(where + ": 'bundle_parameters' entry '" +
                                 entry.getName().str() +
                                 "' is not an integer or string"));
          return;
        }
    }
    if (mlir::Attribute rawLayoutParameters =
            mapping.get("layout_parameters")) {
      auto byClass = mlir::dyn_cast<mlir::DictionaryAttr>(rawLayoutParameters);
      if (!byClass) {
        failMetadata(
            bindError(where + ": 'layout_parameters' is not a dictionary"));
        return;
      }
      for (const mlir::NamedAttribute &entry : byClass) {
        auto parameters =
            mlir::dyn_cast<mlir::DictionaryAttr>(entry.getValue());
        if (!parameters) {
          failMetadata(bindError(where + ": 'layout_parameters' entry '" +
                                 entry.getName().str() +
                                 "' is not a dictionary"));
          return;
        }
        for (const mlir::NamedAttribute &parameter : parameters)
          if (!mlir::isa<mlir::IntegerAttr>(parameter.getValue()) &&
              !mlir::isa<mlir::StringAttr>(parameter.getValue())) {
            failMetadata(bindError(where + ": 'layout_parameters' entry '" +
                                   entry.getName().str() + "' parameter '" +
                                   parameter.getName().str() +
                                   "' is not an integer or string"));
            return;
          }
      }
    }

    if (std::unique_ptr<TargetEmitter> emitter =
            target.createEmitter(*emitterKey)) {
      TargetBundle bundle;
      bundle.name = *bundleName;
      bundle.emitterKey = *emitterKey;
      if (mlir::Attribute rawBundleParameters =
              mapping.get("bundle_parameters"))
        bundle.parameters =
            mlir::cast<mlir::DictionaryAttr>(rawBundleParameters);
      // The target's own emitter decides whether the resolved bundle is
      // complete enough to lower (design §18.3, phase 3).
      if (llvm::Error error = emitter->verify(bundle)) {
        fail(DiagnosticCode::TargetBundleInvalid,
             where + ": " + llvm::toString(std::move(error)));
        return;
      }
    }
  });
  if (failure)
    return failure;

  // --- 2c. routes --------------------------------------------------------
  // Every route entry is a dictionary, every node resolves, consecutive nodes
  // are joined by a link, and every named engine is one the machine declares.
  module->walk([&](mlir::Operation *op) {
    if (failure)
      return;
    mlir::Attribute rawRoutes = op->getAttr(kRoutesAttr);
    if (!rawRoutes)
      return;
    const std::string where =
        "micro.routes on '" + op->getName().getStringRef().str() + "'";
    auto routes = mlir::dyn_cast<mlir::ArrayAttr>(rawRoutes);
    if (!routes) {
      failMetadata(bindError(where + " is not an array"));
      return;
    }
    for (mlir::Attribute element : routes) {
      if (failure)
        return;
      auto route = mlir::dyn_cast<mlir::DictionaryAttr>(element);
      if (!route) {
        failMetadata(bindError(where + ": a route entry is not a dictionary"));
        return;
      }
      llvm::Expected<std::string> kind = requiredString(route, "kind", where);
      if (!kind) {
        failMetadata(kind.takeError());
        return;
      }
      if (!symbolizeConnectionKind(*kind)) {
        failMetadata(
            bindError(where + ": unknown connection kind '" + *kind + "'"));
        return;
      }
      mlir::Attribute rawRoute = route.get("route");
      if (!rawRoute) {
        failMetadata(bindError(where + ": missing 'route'"));
        return;
      }
      llvm::Expected<llvm::SmallVector<std::string, 4>> nodes =
          readStringArray(rawRoute, "route", where);
      if (!nodes) {
        failMetadata(nodes.takeError());
        return;
      }
      for (const std::string &id : *nodes) {
        if (!machine.findMemory(id)) {
          fail(DiagnosticCode::NoMemoryRoute,
               "route names unknown memory '" + id + "'");
          return;
        }
      }
      for (size_t index = 1; index < nodes->size(); ++index) {
        const std::string &from = (*nodes)[index - 1];
        const std::string &to = (*nodes)[index];
        bool linked =
            llvm::any_of(machine.links, [&](const machine::LinkEdge &l) {
              return l.source == from && l.destination == to;
            });
        if (!linked) {
          fail(DiagnosticCode::NoMemoryRoute,
               "route hop '" + from + "' -> '" + to + "' has no link");
          return;
        }
      }
      if (mlir::Attribute rawEngines = route.get("engines")) {
        llvm::Expected<llvm::SmallVector<std::string, 4>> engines =
            readStringArray(rawEngines, "engines", where);
        if (!engines) {
          failMetadata(engines.takeError());
          return;
        }
        for (const std::string &engine : *engines) {
          if (!machine.findTransferEngine(engine)) {
            fail(DiagnosticCode::NoMemoryRoute,
                 "route names unknown transfer engine '" + engine + "'");
            return;
          }
        }
      }
      // The consumer association is generic metadata too: an array of instance
      // ids, validated with checked casts like every other container.
      if (mlir::Attribute rawConsumers = route.get("consumers")) {
        auto consumers = mlir::dyn_cast<mlir::ArrayAttr>(rawConsumers);
        if (!consumers) {
          failMetadata(bindError(where + ": 'consumers' is not an array"));
          return;
        }
        for (mlir::Attribute consumer : consumers)
          if (!mlir::isa<mlir::IntegerAttr>(consumer)) {
            failMetadata(
                bindError(where + ": 'consumers' has a non-integer entry"));
            return;
          }
      }
    }
  });
  return failure;
}

} // namespace mlir::llk::mapping
