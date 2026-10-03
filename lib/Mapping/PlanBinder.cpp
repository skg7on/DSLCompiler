//===- PlanBinder.cpp - Materialize a selected plan (D7/#50) -------------===//

#include "LLK/Mapping/PlanBinder.h"

#include "LLK/Mapping/WorkloadGraph.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Verifier.h"

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
