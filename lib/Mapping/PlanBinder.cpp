//===- PlanBinder.cpp - Materialize a selected plan (D7/#50) -------------===//

#include "LLK/Mapping/PlanBinder.h"

#include "LLK/Mapping/WorkloadGraph.h"

#include "mlir/AsmParser/AsmParser.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Verifier.h"

#include "llvm/ADT/DenseSet.h"
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

llvm::Error bindError(const std::string &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(), message);
}

mlir::IntegerAttr u64Attr(mlir::MLIRContext *context, uint64_t value) {
  return mlir::IntegerAttr::get(mlir::IntegerType::get(context, 64),
                                static_cast<int64_t>(value));
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
    attributes.emplace_back(mlir::StringAttr::get(context, "bundle"),
                            mlir::StringAttr::get(context, placement.bundle));
    attributes.emplace_back(mlir::StringAttr::get(context, "emitter"),
                            mlir::StringAttr::get(context, rule->emitter));
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
  llvm::DenseSet<WorkloadValueId> moved;
  mlir::OpBuilder builder(context);
  for (const PlanConnection &connection : plan.connectionPlans) {
    if (connection.kind != ConnectionKind::Transfer &&
        connection.kind != ConnectionKind::TransferAndTransform)
      continue;
    if (connection.route.size() < 2)
      continue;
    if (!moved.insert(connection.value).second)
      continue;

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
      bound.unmaterialized.push_back("async token type is not registered");
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

  // 2. machine-aware, and 3. target
  module->walk([&](mlir::Operation *op) {
    if (failure)
      return;
    auto mapping = op->getAttrOfType<mlir::DictionaryAttr>(kMappingAttr);
    if (!mapping)
      return;
    std::string where =
        "mapped op '" + op->getName().getStringRef().str() + "'";

    auto stringValue = [&](llvm::StringRef name) -> std::string {
      if (auto attribute = mapping.getAs<mlir::StringAttr>(name))
        return attribute.getValue().str();
      return {};
    };

    std::string ruleId = stringValue("rule");
    const RuleDef *rule = target.rules().find(ruleId);
    if (!rule) {
      failure = bindError(where + ": unknown rule '" + ruleId + "'");
      return;
    }

    std::string executor = stringValue("executor");
    if (!machine.findExecutor(executor)) {
      failure = bindError(where + ": unknown executor '" + executor + "'");
      return;
    }

    if (auto memories = mapping.getAs<mlir::DictionaryAttr>("memories")) {
      for (const mlir::NamedAttribute &entry : memories) {
        std::string memoryId =
            mlir::cast<mlir::StringAttr>(entry.getValue()).getValue().str();
        const machine::MemoryNode *memory = machine.findMemory(memoryId);
        if (!memory) {
          failure = bindError(where + ": unknown memory '" + memoryId + "'");
          return;
        }
        if (!machine.isVisible(memoryId, executor)) {
          failure = bindError(where + ": executor '" + executor +
                              "' cannot see memory '" + memoryId + "'");
          return;
        }
      }
    }

    if (auto layouts = mapping.getAs<mlir::DictionaryAttr>("layouts")) {
      for (const mlir::NamedAttribute &entry : layouts) {
        std::string layoutId =
            mlir::cast<mlir::StringAttr>(entry.getValue()).getValue().str();
        if (!target.layouts().find(layoutId)) {
          failure = bindError(where + ": unknown layout '" + layoutId + "'");
          return;
        }
      }
    }

    // 3. target: the emitter the rule selected must be one the target declares.
    std::string emitter = stringValue("emitter");
    if (!target.isKnownEmitter(emitter)) {
      failure = bindError(where + ": unknown emitter '" + emitter + "'");
      return;
    }
  });
  if (failure)
    return failure;

  // Routes: every node resolves, and consecutive nodes are joined by a link.
  llvm::Error routeFailure = llvm::Error::success();
  module->walk([&](mlir::Operation *op) {
    if (routeFailure)
      return;
    auto routes = op->getAttrOfType<mlir::ArrayAttr>(kRoutesAttr);
    if (!routes)
      return;
    for (mlir::Attribute entry : routes) {
      auto route = mlir::cast<mlir::DictionaryAttr>(entry);
      auto nodes = route.getAs<mlir::ArrayAttr>("route");
      if (!nodes)
        continue;
      for (mlir::Attribute node : nodes) {
        std::string id = mlir::cast<mlir::StringAttr>(node).getValue().str();
        if (!machine.findMemory(id)) {
          routeFailure = bindError("route names unknown memory '" + id + "'");
          return;
        }
      }
      for (size_t index = 1; index < nodes.size(); ++index) {
        std::string from =
            mlir::cast<mlir::StringAttr>(nodes[index - 1]).getValue().str();
        std::string to =
            mlir::cast<mlir::StringAttr>(nodes[index]).getValue().str();
        bool linked =
            llvm::any_of(machine.links, [&](const machine::LinkEdge &l) {
              return l.source == from && l.destination == to;
            });
        if (!linked) {
          routeFailure =
              bindError("route hop '" + from + "' -> '" + to + "' has no link");
          return;
        }
      }
    }
  });
  return routeFailure;
}

} // namespace mlir::llk::mapping
