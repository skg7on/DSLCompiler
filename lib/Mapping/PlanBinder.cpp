//===- PlanBinder.cpp - Materialize a selected plan (D7/#50) -------------===//
//
// `bindPlan` clones the source kernel, persists the selected plan as schema-v2
// metadata (`encodeSelectedPlan`, see MappingMetadata.h), then materializes the
// selected connections. Verification of already-mapped Micro-IR lives in
// PlanVerification.cpp.

#include "LLK/Mapping/PlanBinder.h"

#include "LLK/Mapping/Diagnostics.h"
#include "LLK/Mapping/MappingMetadata.h"
#include "LLK/Mapping/TileFacts.h"
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

constexpr llvm::StringLiteral kPlanAttr = "micro.plan";

/// The binder's target-neutral bookkeeping on a materialized movement (design
/// §12.4/§23.3). A copy belongs to a selected connection and to one hop of that
/// connection's route; the performance model charges it by those ids, and the
/// completeness verifier resolves the same stamps back to the route so that
/// `micro.value` alone cannot exempt an arbitrary operation.
constexpr llvm::StringLiteral kValueAttr = "micro.value";
constexpr llvm::StringLiteral kDstNodeAttr = "micro.dst_node";
constexpr llvm::StringLiteral kConnectionAttr = "micro.connection";
constexpr llvm::StringLiteral kHopAttr = "micro.hop";

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

mlir::IntegerAttr u64Attr(mlir::MLIRContext *context, uint64_t value) {
  return mlir::IntegerAttr::get(mlir::IntegerType::get(context, 64),
                                static_cast<int64_t>(value));
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

  // --- persist the selected state (schema v2) ---------------------------
  // Encoded before any movement is emitted, so the recorded source-graph hash
  // is the pre-materialization identity a later reader recovers through the
  // recorded connection provenance.
  if (llvm::Error error = encodeSelectedPlan(*module, plan, target))
    return std::move(error);

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
      // is charged the wrong link. The same stamps are what the completeness
      // verifier resolves (A5): `micro.value` alone must not exempt an op.
      copyState.addAttribute(kValueAttr, u64Attr(context, connection.value));
      copyState.addAttribute(kDstNodeAttr,
                             mlir::StringAttr::get(context, to->id));
      copyState.addAttribute(kConnectionAttr, u64Attr(context, connection.id));
      copyState.addAttribute(kHopAttr, u64Attr(context, hop));
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
    // consumer reads the layout it requires. It carries the connection stamp
    // too, so the completeness verifier and the source-graph projection can
    // recognise it as binder-emitted.
    mlir::Value produced = lastCopy->getResult(0);
    if (connection.kind == ConnectionKind::TransferAndTransform) {
      if (!connection.transform) {
        bound.unmaterialized.push_back(
            "value " + std::to_string(connection.value) +
            ": transfer-and-transform carries no maps to emit");
        continue;
      }
      produced = emitTransform(lastCopy, produced, *connection.transform);
      if (mlir::Operation *transformOp = produced.getDefiningOp()) {
        transformOp->setAttr(kValueAttr, u64Attr(context, connection.value));
        transformOp->setAttr(kConnectionAttr, u64Attr(context, connection.id));
      }
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

  // Record materialization completeness in the persisted selection, so a reader
  // can tell a fully materialized plan from a partial one.
  if (!bound.unmaterialized.empty()) {
    if (auto planAttr =
            kernel->getAttrOfType<mlir::DictionaryAttr>(kPlanAttr)) {
      mlir::NamedAttrList updated(planAttr);
      updated.set("materialized", mlir::BoolAttr::get(context, false));
      kernel->setAttr(kPlanAttr, updated.getDictionary(context));
    }
  }

  bound.planId = plan.id;
  bound.kernel = kernel;
  bound.module = std::move(module);
  return std::move(bound);
}

} // namespace mlir::llk::mapping
