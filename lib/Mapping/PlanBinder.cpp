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
constexpr llvm::StringLiteral kLayoutTransformReason =
    "layout_transform_requires_dialect_op";
constexpr llvm::StringLiteral kTransferAndTransformReason =
    "transfer_and_transform_layout_not_applied";
constexpr llvm::StringLiteral kReplicateReason = "replicate_not_materialized";
constexpr llvm::StringLiteral kReduceReason = "reduce_not_materialized";
constexpr llvm::StringLiteral kHoplessRouteReason =
    "route_has_no_hop_to_materialize";
constexpr llvm::StringLiteral kDuplicateRouteReason =
    "duplicate_route_for_value";

/// The reason a non-movement connection cannot be materialized. `Direct` and
/// the movements are handled inline and must never reach here.
llvm::StringRef unmaterializedReason(ConnectionKind kind) {
  switch (kind) {
  case ConnectionKind::LayoutTransform:
    return kLayoutTransformReason;
  case ConnectionKind::Replicate:
    return kReplicateReason;
  case ConnectionKind::Reduce:
    return kReduceReason;
  case ConnectionKind::Direct:
  case ConnectionKind::Transfer:
  case ConnectionKind::TransferAndTransform:
    llvm_unreachable("a direct or movement connection is materialized inline");
  }
  llvm_unreachable("all connection kinds handled");
}

llvm::Error bindError(const std::string &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(), message);
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

mlir::Operation *findKernel(mlir::ModuleOp module) {
  mlir::Operation *kernel = nullptr;
  module->walk([&](mlir::Operation *op) {
    if (!kernel && op->getName().getStringRef() == "micro.kernel")
      kernel = op;
  });
  return kernel;
}

} // namespace

llvm::Expected<BoundPlan> bindPlan(mlir::ModuleOp source,
                                   const CoveringPlan &plan,
                                   const MappingTarget &target) {
  mlir::OwningOpRef<mlir::ModuleOp> module(
      mlir::cast<mlir::ModuleOp>(source->clone()));
  mlir::Operation *kernel = findKernel(*module);
  if (!kernel)
    return bindError("bindPlan: the source module has no micro.kernel");

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
  // One copy chain per moved value: the value is produced in one memory and
  // read in another, so the copies are placed right after the producer and
  // every other use is rewired to the last of them. Emitting one chain per
  // *connection* would duplicate it when a value fans out.
  //
  // A value reached by several connections (a producer feeding several
  // consumers) is handled once. When every such connection takes the same
  // route, one chain serves them all, so a duplicate is merged silently. When
  // a duplicate takes a *different* route, that single chain cannot serve it --
  // the second consumer would silently read the first route's memory -- and the
  // binder cannot yet emit a second chain and rewire only that consumer, so the
  // duplicate is reported instead of dropped.
  llvm::DenseMap<WorkloadValueId, llvm::SmallVector<MemoryNodeId>> movedRoutes;
  mlir::OpBuilder builder(context);
  for (const PlanConnection &connection : plan.connectionPlans) {
    // A `Direct` connection is materialized by construction: the producer wrote
    // the value to the memory the consumer reads, in a layout the consumer
    // addresses, so there is nothing to emit and nothing to report.
    if (connection.kind == ConnectionKind::Direct)
      continue;

    // Anything else that is not a plain movement cannot be emitted as Micro ops
    // today. Report it with a stable reason rather than dropping it (design
    // §18.2).
    if (connection.kind != ConnectionKind::Transfer &&
        connection.kind != ConnectionKind::TransferAndTransform) {
      bound.unmaterialized.push_back(
          "value " + std::to_string(connection.value) + ": " +
          unmaterializedReason(connection.kind).str());
      continue;
    }

    // A movement needs at least one hop between two memories; a shorter route
    // has nothing to emit. This is unreachable for the current placement code,
    // but a selected connection must never vanish silently.
    if (connection.route.size() < 2) {
      bound.unmaterialized.push_back("value " +
                                     std::to_string(connection.value) + ": " +
                                     kHoplessRouteReason.str());
      continue;
    }
    auto handled = movedRoutes.find(connection.value);
    if (handled != movedRoutes.end()) {
      if (handled->second == connection.route)
        continue; // the same movement: one chain serves both connections
      bound.unmaterialized.push_back("value " +
                                     std::to_string(connection.value) + ": " +
                                     kDuplicateRouteReason.str());
      continue;
    }
    movedRoutes.try_emplace(connection.value, connection.route);

    // The producer is the covered node that writes this value; the consumer is
    // any covered node that reads it.
    mlir::Value value = binding.valueFor(connection.value);
    mlir::Operation *producer = nullptr;
    for (const WorkloadNode &node : graph->getNodes()) {
      bool writes = llvm::any_of(node.outputs, [&](const WorkloadPort &port) {
        return port.value == connection.value;
      });
      if (writes) {
        producer = binding.opFor(node.id);
        break;
      }
    }
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
    mlir::Operation *firstCopy = nullptr;
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

      if (!firstCopy)
        firstCopy = copy;
      lastCopy = copy;
      current = copy->getResult(0);
    }

    if (!materialized || !lastCopy) {
      bound.unmaterialized.push_back(
          "value " + std::to_string(connection.value) +
          ": a route hop names memories the machine cannot carry data between");
      continue;
    }

    // A transfer-and-transform moved the value, but the selected layout
    // transform itself has no Micro operation form (design §13.4). The
    // movement is real, so the connection is not dropped -- but the transform
    // must still be reported, or it vanishes without a trace. The token differs
    // from the transform-only case so "nothing materialized" stays
    // distinguishable from "movement done, transform dropped".
    if (connection.kind == ConnectionKind::TransferAndTransform)
      bound.unmaterialized.push_back("value " +
                                     std::to_string(connection.value) + ": " +
                                     kTransferAndTransformReason.str());

    // Rewire every other reader to the last hop's value; the copies themselves
    // already read the previous one.
    llvm::SmallVector<mlir::OpOperand *> uses;
    for (mlir::OpOperand &use : value.getUses())
      if (use.getOwner() != firstCopy)
        uses.push_back(&use);
    for (mlir::OpOperand *use : uses)
      use->set(lastCopy->getResult(0));
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
    if (std::unique_ptr<TargetEmitter> emitter =
            target.createEmitter(*emitterKey)) {
      TargetBundle bundle;
      bundle.name = *bundleName;
      bundle.emitterKey = *emitterKey;
      // Bundle parameters are not persisted by the binder yet (design §18.1
      // open item); the plugin's shape check therefore sees the name and key
      // only.
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
    }
  });
  return failure;
}

} // namespace mlir::llk::mapping
