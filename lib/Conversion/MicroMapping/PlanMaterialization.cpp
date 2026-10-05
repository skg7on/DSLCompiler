//===- PlanMaterialization.cpp - Construct a selected plan's Micro ops ----===//
//
// The canonical `PlanMaterializer` (design §18.2, "Materialization versus
// target readiness"). `bindPlan` persists a selected plan as target-neutral
// metadata; this component is the dialect-aware half that turns the selected
// connections into Micro operations and rewires the recorded consumer
// endpoints. It lives beside the Micro mapping passes -- never inside
// `lib/Mapping` -- so the target-neutral mapping library links no dialect.
//
// What it emits, per connection kind:
//   * `Transfer` / `Replicate`: one asynchronously awaited copy per route hop,
//     landing the value in each hop's destination memory. A tile value becomes
//     `micro.tile_async_copy` (its result retyped into the destination memory
//     via the canonical `materializedTileType`); a shaped value keeps the
//     generic `micro.async_copy`.
//   * `LayoutTransform`: one `micro.transform` naming the two layouts by their
//     solved affine maps.
//   * `TransferAndTransform`: the movement copies followed by that transform.
//   * `Reduce`: reported (`reduce_not_materialized`) -- no Micro operation
//   form.
//
// Rewiring is scoped to the connection's recorded `consumerPorts`: the operand
// occurrence each `PortRef` names is redirected to the movement's result, and
// nothing else. A consumer that read the value through a transparent
// `micro.tile_view` gets a fresh view over the movement result, so the recorded
// endpoint genuinely reads what the connection produced rather than the
// original value.

#include "LLK/Mapping/PlanBinder.h"

#include "LLK/Dialect/Micro/MicroDialect.h"
#include "LLK/Dialect/Micro/MicroEnums.h"
#include "LLK/Dialect/Micro/MicroHelpers.h"

#include "LLK/Mapping/WorkloadGraph.h"

#include "mlir/AsmParser/AsmParser.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Error.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

#define GET_ATTRDEF_CLASSES
#include "LLK/Dialect/Micro/MicroAttributes.h.inc"
#define GET_TYPEDEF_CLASSES
#include "LLK/Dialect/Micro/MicroTypes.h.inc"

#include "MicroMappingCommon.h"

namespace mlir::llk::micro_mapping_detail {

using namespace mlir;
using namespace mlir::llk::mapping;

namespace {

constexpr llvm::StringLiteral kValueAttr = "micro.value";
constexpr llvm::StringLiteral kSrcNodeAttr = "micro.src_node";
constexpr llvm::StringLiteral kDstNodeAttr = "micro.dst_node";
constexpr llvm::StringLiteral kConnectionAttr = "micro.connection";
constexpr llvm::StringLiteral kHopAttr = "micro.hop";

/// Stable, greppable reasons a connection is not materialized (design §18.2).
/// They are part of the report contract, so they are named constants rather
/// than free-form prose.
constexpr llvm::StringLiteral kReduceReason = "reduce_not_materialized";
constexpr llvm::StringLiteral kFeedUnresolvedReason =
    "gather_feed_endpoint_unresolved";
constexpr llvm::StringLiteral kGatherFeedMismatchReason =
    "gather_feed_types_incompatible";
constexpr llvm::StringLiteral kHoplessRouteReason =
    "route_has_no_hop_to_materialize";
constexpr llvm::StringLiteral kSameNodeReason =
    "route_endpoints_name_the_same_memory_node";
constexpr llvm::StringLiteral kNoProducerReason =
    "no_producing_operation_in_the_kernel";
constexpr llvm::StringLiteral kUnknownMemoryReason =
    "route_names_an_unknown_memory";
constexpr llvm::StringLiteral kMissingFactReason = "missing_static_fact";
constexpr llvm::StringLiteral kNoConsumerReason =
    "consumer_endpoint_unresolved";

llvm::Error materializeError(const std::string &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(), message);
}

mlir::IntegerAttr u64Attr(mlir::MLIRContext *context, uint64_t value) {
  return mlir::IntegerAttr::get(mlir::IntegerType::get(context, 64),
                                static_cast<int64_t>(value));
}

/// The `#micro.memory<kind>` attribute a machine memory kind names, or null
/// when the kind is not a canonical Micro abstract space. An unknown target
/// memory kind has no canonical abstract-space mapping, so the caller rejects
/// it rather than inventing a dialect enum: a machine kind either spells a real
/// `#micro.memory<...>` space, or an explicit mapping must be added.
mlir::Attribute memoryAttrFor(mlir::MLIRContext *context,
                              llvm::StringRef kind) {
  return mlir::parseAttribute(("#micro.memory<" + kind + ">").str(), context);
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

/// The canonical reason a connection kind has no Micro operation form. Every
/// kind the materializer emits -- Direct, the movements, replication and the
/// layout transform -- must never reach here.
llvm::StringRef unmaterializedKindReason(ConnectionKind kind) {
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

/// True when `type` is a `!micro.tile`.
bool isTileType(mlir::Type type) { return llvm::isa<micro::TileType>(type); }

/// True for the zero-cost logical ops a consumer may read a value through.
bool isTransparentOp(mlir::Operation *op) {
  llvm::StringRef name = op->getName().getStringRef();
  return name == "micro.tile_view" || name == "micro.tile_partition";
}

/// Resolves `value` through the transparent logical ops to the value it
/// ultimately reads.
mlir::Value resolveThroughTransparent(mlir::Value value) {
  while (mlir::Operation *defining = value.getDefiningOp()) {
    if (!isTransparentOp(defining) || defining->getNumOperands() == 0)
      break;
    value = defining->getOperand(0);
  }
  return value;
}

/// The operation whose `micro.mapping` records source node `node`, or null.
/// Endpoints name source-graph node ids; the persisted stamp is what ties a
/// materialized op back to the node it covers, so resolving through it keeps
/// rewiring correct even if re-extraction of the encoded clone would relabel
/// nodes (the bookkeeping attributes are themselves node content).
mlir::Operation *materializedOpForNode(mlir::Operation *kernel, uint64_t node) {
  mlir::Operation *found = nullptr;
  kernel->walk([&](mlir::Operation *op) {
    if (found)
      return;
    auto mapping = op->getAttrOfType<mlir::DictionaryAttr>("micro.mapping");
    if (!mapping)
      return;
    if (auto recorded = mapping.getAs<mlir::IntegerAttr>("node"))
      if (recorded.getValue().getZExtValue() == node)
        found = op;
  });
  return found;
}

/// The canonical materializer. It resolves the producer and consumer
/// *endpoints* the plan records (through the persisted `micro.mapping.node`
/// stamps) and only falls back to re-extracting the workload graph for a plan
/// built without endpoints, then emits and rewires.
class CanonicalPlanMaterializer final : public PlanMaterializer {
public:
  llvm::Error materialize(mlir::ModuleOp module, const CoveringPlan &plan,
                          const MappingTarget &target,
                          BoundPlan &bound) override;

private:
  /// A copy chain emitted for one (value, route, transform): what a consumer of
  /// the chain reads, and the endpoint occurrences that must read it.
  struct Chain {
    WorkloadValueId value = 0;
    llvm::SmallVector<MemoryNodeId> route;
    std::string transform;
    /// The value the movement read: what a recorded consumer must currently
    /// read (directly or through a transparent view) for the rewire to apply.
    mlir::Value source;
    /// Additional values a recorded consumer may currently read instead of
    /// `source`. A gather combines several feeds, so its consumers may read any
    /// one of them before being rewired to the gathered result.
    llvm::SmallVector<mlir::Value, 2> altSources;
    mlir::Value result;
    mlir::Attribute dstMemory;
    llvm::SmallVector<PortRef> consumers;
  };
};

/// The operand a consumer endpoint names, or null when the index is out of
/// range. Extraction records one input port per non-token operand, in order, so
/// the port index is the operand index for every node op the plan covers.
mlir::OpOperand *operandForPort(mlir::Operation *consumer, uint32_t index) {
  if (index >= consumer->getNumOperands())
    return nullptr;
  return &consumer->getOpOperand(index);
}

/// Emits the target-neutral re-representation change: the value read through
/// `transform.srcMap` is written through `transform.dstMap`. The operation
/// names the layouts by their index relation, never by a target-owned id
/// (design §13.4).
mlir::Value emitTransform(mlir::OpBuilder &builder, mlir::Operation *anchor,
                          mlir::Value input, const LayoutTransform &transform) {
  builder.setInsertionPointAfter(anchor);
  mlir::OperationState state(anchor->getLoc(), "micro.transform");
  state.addOperands(input);
  state.addTypes({input.getType()});
  if (transform.srcMap)
    state.addAttribute("src_map", mlir::AffineMapAttr::get(transform.srcMap));
  if (transform.dstMap)
    state.addAttribute("dst_map", mlir::AffineMapAttr::get(transform.dstMap));
  return builder.create(state)->getResult(0);
}

/// Returns `produced` retyped so it can be read where `consumerOperand` sits.
///
/// A consumer that reads the value directly is rewired to the movement's
/// result. A consumer that read it through a transparent `micro.tile_view` --
/// so its operand is a tile while the movement produced a shaped value -- gets
/// a fresh view over the movement result in the destination memory, which is
/// the transparent indirection A5 accepts. Returns null when no legal retyping
/// exists; the caller reports the endpoint as unresolved rather than forcing an
/// ill-typed operand.
mlir::Value retypeForConsumer(mlir::OpBuilder &builder, mlir::Value produced,
                              mlir::Value consumerOperand,
                              mlir::Attribute dstMemory) {
  mlir::Type consumed = consumerOperand.getType();
  mlir::Type producedType = produced.getType();
  if (consumed == producedType)
    return produced;
  if (isTileType(consumed) && isTileType(producedType))
    return produced; // same tile, different memory: the destination does not
                     // change the operand's shape or element type.
  if (isTileType(consumed) && llvm::isa<mlir::ShapedType>(producedType)) {
    mlir::Type dstTile = micro::materializedTileType(consumed, dstMemory);
    auto consumedTile = llvm::dyn_cast<micro::TileType>(consumed);
    if (!dstTile || !consumedTile)
      return {};
    builder.setInsertionPointAfter(produced.getDefiningOp());
    mlir::OperationState state(produced.getDefiningOp()->getLoc(),
                               "micro.tile_view");
    state.addOperands(produced);
    state.addTypes({dstTile});
    state.addAttribute("shape",
                       mlir::DenseI64ArrayAttr::get(builder.getContext(),
                                                    consumedTile.getShape()));
    return builder.create(state)->getResult(0);
  }
  return {};
}

/// The result type of a `micro.gather` over `feeds`: for `Concatenate` the
/// shapes joined along `axis` (every other extent must already agree), for
/// Sum/Max the feed type unchanged. A tile result is retyped into `memory`;
/// null when the inputs do not state a statically known, consistent image (the
/// caller then reports the missing fact rather than emitting a mistyped op).
mlir::Type gatherResultType(llvm::ArrayRef<mlir::Value> feeds,
                            GatherSemantics semantics,
                            std::optional<uint64_t> axis,
                            mlir::Attribute memory) {
  if (feeds.empty())
    return {};
  using ShapeAndElement = std::pair<llvm::SmallVector<int64_t, 4>, mlir::Type>;
  auto shapeAndElement = [](mlir::Type type) -> std::optional<ShapeAndElement> {
    if (auto shaped = llvm::dyn_cast<mlir::ShapedType>(type))
      return std::make_pair(
          llvm::SmallVector<int64_t, 4>(shaped.getShape().begin(),
                                        shaped.getShape().end()),
          shaped.getElementType());
    if (auto tile = llvm::dyn_cast<micro::TileType>(type))
      return std::make_pair(llvm::SmallVector<int64_t, 4>(
                                tile.getShape().begin(), tile.getShape().end()),
                            tile.getElementType());
    return std::nullopt;
  };

  llvm::SmallVector<int64_t, 4> shape;
  mlir::Type elementType;
  if (semantics == GatherSemantics::Concatenate) {
    if (!axis)
      return {};
    for (size_t i = 0; i < feeds.size(); ++i) {
      std::optional<ShapeAndElement> read = shapeAndElement(feeds[i].getType());
      if (!read)
        return {};
      if (*axis >= read->first.size())
        return {};
      for (int64_t extent : read->first)
        if (ShapedType::isDynamic(extent))
          return {};
      if (i == 0) {
        shape = read->first;
        elementType = read->second;
        shape[*axis] = 0;
      } else if (read->second != elementType ||
                 read->first.size() != shape.size()) {
        return {};
      } else {
        for (unsigned dim = 0; dim < shape.size(); ++dim)
          if (dim != *axis && read->first[dim] != shape[dim])
            return {};
      }
      shape[*axis] += read->first[*axis];
    }
  } else {
    std::optional<ShapeAndElement> read =
        shapeAndElement(feeds.front().getType());
    if (!read)
      return {};
    shape = read->first;
    elementType = read->second;
    // Sum/Max combine like-for-like operands: every feed must state the same
    // shape and element type as the first, or the emitted gather would violate
    // its own verifier. Refused here rather than emitted and caught later.
    for (size_t i = 1; i < feeds.size(); ++i) {
      std::optional<ShapeAndElement> other =
          shapeAndElement(feeds[i].getType());
      if (!other || other->second != elementType || other->first != shape)
        return {};
    }
  }

  mlir::MLIRContext *context = feeds.front().getContext();
  if (auto firstTile =
          llvm::dyn_cast<micro::TileType>(feeds.front().getType())) {
    auto memoryAttr = llvm::dyn_cast_or_null<micro::MemorySpaceAttr>(memory);
    return micro::TileType::get(
        context, shape, elementType, firstTile.getLayout(),
        memoryAttr ? memoryAttr : firstTile.getMemory(), firstTile.getOwner());
  }
  // A shaped value names no memory in its type: the staging memory is carried
  // by the copies that produced its operands, not by the gather's result type.
  return mlir::RankedTensorType::get(shape, elementType);
}

llvm::Error CanonicalPlanMaterializer::materialize(mlir::ModuleOp module,
                                                   const CoveringPlan &plan,
                                                   const MappingTarget &target,
                                                   BoundPlan &bound) {
  mlir::Operation *kernel = nullptr;
  module->walk([&](mlir::Operation *op) {
    if (!kernel && op->getName().getStringRef() == "micro.kernel")
      kernel = op;
  });
  if (!kernel)
    return materializeError(
        "materialize: the cloned module has no micro.kernel");

  WorkloadGraphBinding binding;
  llvm::Expected<WorkloadGraph> graph = extractWorkloadGraph(kernel, &binding);
  if (!graph)
    return graph.takeError();

  mlir::MLIRContext *context = module->getContext();
  mlir::OpBuilder builder(context);
  const machine::MachineModel &machineModel = target.machine();

  mlir::Type tokenType = mlir::parseType("!micro.async_token", context);
  if (!tokenType) {
    // The token type is registered by the loaded dialect; without it no copy
    // can be emitted at all.
    for (const PlanConnection &connection : plan.connectionPlans)
      if (connection.kind != ConnectionKind::Direct)
        bound.unmaterialized.push_back(
            "value " + std::to_string(connection.value) + ": " +
            kMissingFactReason.str() + ": async token type is not registered");
    return llvm::Error::success();
  }

  // The covered node that writes `connection`'s value: the op whose persisted
  // `micro.mapping.node` matches the connection's recorded producer endpoint,
  // falling back to the extracted graph for a plan built without endpoints.
  auto producerOperationFor =
      [&](const PlanConnection &connection) -> mlir::Operation * {
    if (connection.producerPort)
      if (mlir::Operation *op =
              materializedOpForNode(kernel, connection.producerPort->node))
        return op;
    for (const WorkloadNode &node : graph->getNodes()) {
      bool writes = llvm::any_of(node.outputs, [&](const WorkloadPort &port) {
        return port.value == connection.value;
      });
      if (writes)
        return binding.opFor(node.id);
    }
    return nullptr;
  };

  // The SSA value a connection moves or transforms: the producer endpoint's
  // result when recorded, else the extracted value.
  auto moveSourceFor = [&](const PlanConnection &connection) -> mlir::Value {
    if (connection.producerPort)
      if (mlir::Operation *op =
              materializedOpForNode(kernel, connection.producerPort->node))
        if (connection.producerPort->index < op->getNumResults())
          return op->getResult(connection.producerPort->index);
    return binding.valueFor(connection.value);
  };

  auto report = [&](const PlanConnection &connection, llvm::StringRef reason) {
    bound.unmaterialized.push_back("value " + std::to_string(connection.value) +
                                   ": " + reason.str());
  };

  // Barrier requirements the selected plan recorded, by connection id (task
  // B6). A barrier is emitted only where the plan's synchronization asks for
  // one, so two independent movements that share no dependency carry none
  // between them.
  llvm::DenseMap<uint64_t, char> barrierFor;
  for (const SynchronizationStep &step : plan.synchronization) {
    if (!step.requiresBarrier)
      continue;
    for (ConnectionId waited : step.waitsFor)
      barrierFor[waited] = 1;
  }

  auto emitBarrier = [&](mlir::Operation *anchor,
                         llvm::ArrayRef<mlir::Value> tokens,
                         ConnectionId connectionId) {
    if (!anchor || tokens.empty())
      return;
    builder.setInsertionPointAfter(anchor);
    mlir::OperationState state(anchor->getLoc(), "micro.barrier");
    state.addOperands(tokens);
    state.addAttribute("scope",
                       mlir::StringAttr::get(
                           context, micro::stringifyBarrierScope(
                                        micro::BarrierScope::executor_group)));
    // Which planned synchronization this barrier represents, so verification
    // can prove the required synchronization is present rather than any
    // barrier.
    state.addAttribute(kConnectionAttr, u64Attr(context, connectionId));
    builder.create(state);
  };

  // Emits one awaited copy per route hop, landing `value` in the route's final
  // memory. Returns the final result, or null (with `reason` set) when a hop
  // names memories the machine cannot carry data between. Every copy's token is
  // appended to `tokens`, so a barrier can cover exactly this movement.
  auto emitChain = [&](mlir::Value value, mlir::Operation *producer,
                       const llvm::SmallVector<MemoryNodeId> &route,
                       WorkloadValueId connectionValue,
                       ConnectionId connectionId,
                       llvm::SmallVectorImpl<mlir::Value> &tokens,
                       std::string &reason) -> mlir::Value {
    if (!producer || !value) {
      reason = kNoProducerReason.str();
      return {};
    }
    if (route.size() < 2) {
      reason = kHoplessRouteReason.str();
      return {};
    }
    builder.setInsertionPointAfter(producer);
    mlir::Value current = value;
    mlir::Operation *lastCopy = nullptr;
    for (size_t hop = 1; hop < route.size(); ++hop) {
      const machine::MemoryNode *from = machineModel.findMemory(route[hop - 1]);
      const machine::MemoryNode *to = machineModel.findMemory(route[hop]);
      if (!from || !to || from->id == to->id) {
        reason = "a route hop names memories the machine cannot carry data "
                 "between";
        return {};
      }
      // A same-kind hop is carried only when the machine declares a link
      // between the two concrete nodes. Whether that link is legal -- engine,
      // transaction granule, alignment -- is the machine verifier's check; the
      // materializer refuses to emit a movement no link backs.
      const bool linked =
          llvm::any_of(machineModel.links, [&](const machine::LinkEdge &edge) {
            return edge.source == from->id && edge.destination == to->id;
          });
      if (!linked) {
        reason = "a route hop names memories the machine cannot carry data "
                 "between";
        return {};
      }
      mlir::Attribute hopSrc = memoryAttrFor(context, from->kind);
      mlir::Attribute hopDst = memoryAttrFor(context, to->kind);
      if (!hopSrc || !hopDst) {
        reason = "a route hop names memories the machine cannot carry data "
                 "between";
        return {};
      }

      mlir::Type resultType = current.getType();
      const bool tileHop = isTileType(current.getType());
      if (tileHop) {
        // A tile carries its memory in the type, so the copy's result is the
        // tile retyped into the hop's destination memory.
        mlir::Type dstTile =
            micro::materializedTileType(current.getType(), hopDst);
        if (!dstTile) {
          reason = (kMissingFactReason.str() +
                    ": cannot construct a destination tile for the moved "
                    "value");
          return {};
        }
        resultType = dstTile;
      }
      mlir::OperationState copyState(producer->getLoc(),
                                     tileHop ? "micro.tile_async_copy"
                                             : "micro.async_copy");
      if (tileHop) {
        copyState.addOperands(current);
        copyState.addTypes({resultType, tokenType});
        copyState.addAttribute("dst_memory", hopDst);
      } else {
        // A shaped value keeps the generic copy: its type does not name a
        // memory, so the source and destination are carried by attributes.
        copyState.addOperands(current);
        copyState.addTypes({resultType, tokenType});
        copyState.addAttribute("src_memory", hopSrc);
        copyState.addAttribute("dst_memory", hopDst);
      }
      // Target-neutral identity for the performance model (design
      // §12.4/§23.3) and the completeness verifier (A5): which connection this
      // hop belongs to, and the concrete memory node it lands in.
      copyState.addAttribute(kValueAttr, u64Attr(context, connectionValue));
      // The concrete node identity the hop crosses. Kind equality is not node
      // identity, so both endpoints are recorded: a same-kind hop is real work
      // exactly when `src_node` and `dst_node` name two distinct memories.
      copyState.addAttribute(kSrcNodeAttr,
                             mlir::StringAttr::get(context, from->id));
      copyState.addAttribute(kDstNodeAttr,
                             mlir::StringAttr::get(context, to->id));
      copyState.addAttribute(kConnectionAttr, u64Attr(context, connectionId));
      copyState.addAttribute(kHopAttr, u64Attr(context, hop));
      mlir::Operation *copy = builder.create(copyState);

      mlir::OperationState waitState(producer->getLoc(), "micro.wait");
      waitState.addOperands(copy->getResult(1));
      builder.create(waitState);

      tokens.push_back(copy->getResult(1));
      lastCopy = copy;
      current = copy->getResult(0);
    }
    if (!lastCopy) {
      reason = kHoplessRouteReason.str();
      return {};
    }
    return lastCopy->getResult(0);
  };

  std::vector<Chain> chains;

  for (const PlanConnection &connection : plan.connectionPlans) {
    // A `Direct` connection is materialized by construction.
    if (connection.kind == ConnectionKind::Direct)
      continue;

    // The endpoint occurrences this connection serves, copied onto the chain.
    llvm::SmallVector<PortRef> consumers(connection.consumerPorts.begin(),
                                         connection.consumerPorts.end());
    const std::string transform = transformIdentity(connection);

    // --- gather (Reduce): explicit semantics, feeds, staging, tokens --------
    //
    // Nothing is inferred from the producer count: a `Reduce` without declared
    // semantics stays Partial-only under the same `reduce_not_materialized`
    // reason it always had. With semantics, each recorded feed occurrence is
    // staged (when a route moves it) and combined by one `micro.gather`.
    if (connection.kind == ConnectionKind::Reduce) {
      if (!connection.gatherSemantics) {
        report(connection, kReduceReason);
        continue;
      }
      const bool concat =
          *connection.gatherSemantics == GatherSemantics::Concatenate;
      if (concat && !connection.concatAxis) {
        report(connection, "concat gather has no axis");
        continue;
      }
      if (!concat && connection.concatAxis) {
        report(connection, "sum and max gather must not carry an axis");
        continue;
      }

      llvm::SmallVector<mlir::Value> feeds;
      llvm::SmallVector<mlir::Operation *> feedProducers;
      bool resolved = true;
      for (const PortRef &ref : connection.producerPorts) {
        mlir::Operation *op = materializedOpForNode(kernel, ref.node);
        if (!op)
          op = binding.opFor(ref.node);
        if (!op || ref.index >= op->getNumResults()) {
          resolved = false;
          break;
        }
        feeds.push_back(op->getResult(ref.index));
        feedProducers.push_back(op);
      }
      if (!resolved || feeds.size() < 2) {
        report(connection, kFeedUnresolvedReason);
        continue;
      }

      // The staging memory is the route's destination. A feed already there
      // needs no copy; a feed elsewhere is staged along the recorded route,
      // each hop followed by its wait.
      const bool staged = connection.route.size() >= 2;
      llvm::SmallVector<mlir::Value> stagedFeeds;
      llvm::SmallVector<mlir::Value> tokens;
      std::string reason;
      bool ok = true;
      for (size_t i = 0; i < feeds.size(); ++i) {
        if (!staged) {
          stagedFeeds.push_back(feeds[i]);
          continue;
        }
        mlir::Value moved =
            emitChain(feeds[i], feedProducers[i], connection.route,
                      connection.value, connection.id, tokens, reason);
        if (!moved) {
          ok = false;
          break;
        }
        stagedFeeds.push_back(moved);
      }
      if (!ok) {
        report(connection, reason);
        continue;
      }

      mlir::Attribute stageMemory;
      if (staged)
        if (const machine::MemoryNode *dst =
                machineModel.findMemory(connection.route.back()))
          stageMemory = memoryAttrFor(context, dst->kind);
      mlir::Type resultType =
          gatherResultType(stagedFeeds, *connection.gatherSemantics,
                           connection.concatAxis, stageMemory);
      if (!resultType) {
        report(connection, concat
                               ? (kMissingFactReason.str() +
                                  ": cannot construct the gathered result type")
                               : kGatherFeedMismatchReason.str());
        continue;
      }

      // The gather and its barrier are anchored after the last feed producer.
      // Every feed is in that producer's block, so a region boundary is never
      // crossed silently: a feed from another region has no SSA path here and
      // its copy would not dominate the gather.
      mlir::Operation *anchor = feedProducers.back();
      builder.setInsertionPointAfter(anchor);
      mlir::OperationState gatherState(anchor->getLoc(), "micro.gather");
      gatherState.addOperands(stagedFeeds);
      gatherState.addTypes({resultType});
      gatherState.addAttribute(
          "kind",
          mlir::StringAttr::get(
              context, stringifyGatherSemantics(*connection.gatherSemantics)));
      if (connection.concatAxis)
        gatherState.addAttribute("axis",
                                 u64Attr(context, *connection.concatAxis));
      gatherState.addAttribute(kValueAttr, u64Attr(context, connection.value));
      gatherState.addAttribute(kConnectionAttr,
                               u64Attr(context, connection.id));
      mlir::Operation *gather = builder.create(gatherState);

      if (barrierFor.count(connection.id))
        emitBarrier(gather, tokens, connection.id);

      Chain chain;
      chain.value = connection.value;
      chain.route = connection.route;
      chain.transform = transform;
      chain.source = feeds.front();
      for (size_t i = 1; i < feeds.size(); ++i)
        chain.altSources.push_back(feeds[i]);
      chain.result = gather->getResult(0);
      chain.dstMemory = stageMemory;
      chain.consumers = std::move(consumers);
      chains.push_back(std::move(chain));
      continue;
    }

    if (connection.kind != ConnectionKind::Transfer &&
        connection.kind != ConnectionKind::TransferAndTransform &&
        connection.kind != ConnectionKind::Replicate &&
        connection.kind != ConnectionKind::LayoutTransform) {
      report(connection, unmaterializedKindReason(connection.kind));
      continue;
    }

    // An already-emitted chain for this (value, route, transform) serves this
    // connection too -- it lands the value in the same memory under the same
    // layout, so the same result is read. Merge the endpoint sets.
    auto existing = llvm::find_if(chains, [&](const Chain &chain) {
      return chain.value == connection.value &&
             chain.route == connection.route && chain.transform == transform;
    });
    if (existing != chains.end()) {
      for (const PortRef &ref : consumers)
        if (!llvm::is_contained(existing->consumers, ref))
          existing->consumers.push_back(ref);
      continue;
    }

    // A transform-only connection moves nothing: it converts the value in
    // place, where the producer already wrote it, so there is no route to walk
    // and no memory to check. It is materialized as one `micro.transform`.
    if (connection.kind == ConnectionKind::LayoutTransform) {
      if (!connection.transform) {
        report(connection, "layout transform carries no maps to emit");
        continue;
      }
      mlir::Value input = moveSourceFor(connection);
      mlir::Operation *producer = producerOperationFor(connection);
      if (!producer || !input) {
        report(connection, kNoProducerReason);
        continue;
      }
      mlir::Attribute dstMemory;
      if (!connection.route.empty()) {
        if (const machine::MemoryNode *dst =
                machineModel.findMemory(connection.route.back()))
          dstMemory = memoryAttrFor(context, dst->kind);
      }
      Chain chain;
      chain.value = connection.value;
      chain.route = connection.route;
      chain.transform = transform;
      chain.source = input;
      chain.result =
          emitTransform(builder, producer, input, *connection.transform);
      chain.dstMemory = dstMemory;
      chain.consumers = std::move(consumers);
      chains.push_back(std::move(chain));
      continue;
    }

    // A movement needs at least one hop between two memories.
    if (connection.route.size() < 2) {
      report(connection, kHoplessRouteReason);
      continue;
    }

    mlir::Value value = moveSourceFor(connection);
    mlir::Operation *producer = producerOperationFor(connection);
    if (!producer || !value) {
      report(connection, kNoProducerReason);
      continue;
    }
    const bool tileValue = isTileType(value.getType());
    if (!tileValue && !llvm::isa<mlir::ShapedType>(value.getType())) {
      report(connection,
             "value has neither a tile nor a shaped type to materialize");
      continue;
    }

    const machine::MemoryNode *source =
        machineModel.findMemory(connection.route.front());
    const machine::MemoryNode *destination =
        machineModel.findMemory(connection.route.back());
    if (!source || !destination) {
      report(connection, kUnknownMemoryReason);
      continue;
    }
    if (source->id == destination->id) {
      // The two endpoints are one concrete memory: whatever their kind, a route
      // that lands where it started is not work. Two *distinct* nodes of one
      // abstract kind are the case B5 makes legal; only node identity, never
      // kind equality, decides.
      report(connection, kSameNodeReason);
      continue;
    }
    mlir::Attribute dstMemory = memoryAttrFor(context, destination->kind);
    if (!dstMemory) {
      report(connection, "destination memory has no micro memory kind");
      continue;
    }

    // One copy per hop, chained: the value moves to the first staging memory,
    // then on, until it reaches the consumer's. Each hop is followed by its own
    // wait; the intermediate value *is* the staging storage.
    llvm::SmallVector<mlir::Value> tokens;
    std::string reason;
    mlir::Value moved =
        emitChain(value, producer, connection.route, connection.value,
                  connection.id, tokens, reason);
    if (!moved) {
      report(connection, reason);
      continue;
    }
    mlir::Operation *lastCopy = moved.getDefiningOp();

    // A transfer-and-transform moved the value; the conversion itself is the
    // same target-neutral `micro.transform`, emitted after the movement so the
    // consumer reads the layout it requires.
    mlir::Value produced = moved;
    if (connection.kind == ConnectionKind::TransferAndTransform) {
      if (!connection.transform) {
        report(connection, "transfer-and-transform carries no maps to emit");
        continue;
      }
      produced =
          emitTransform(builder, lastCopy, produced, *connection.transform);
      if (mlir::Operation *transformOp = produced.getDefiningOp()) {
        transformOp->setAttr(kValueAttr, u64Attr(context, connection.value));
        transformOp->setAttr(kConnectionAttr, u64Attr(context, connection.id));
      }
    }

    // A movement the plan's synchronization marks as needing a barrier gets one
    // over the tokens it produced; a movement that does not gets none.
    if (barrierFor.count(connection.id))
      emitBarrier(produced.getDefiningOp(), tokens, connection.id);

    Chain chain;
    chain.value = connection.value;
    chain.route = connection.route;
    chain.transform = transform;
    chain.source = value;
    chain.result = produced;
    chain.dstMemory = dstMemory;
    chain.consumers = std::move(consumers);
    chains.push_back(std::move(chain));
  }

  // Rewire exactly each chain's recorded consumer endpoints. A recorded
  // endpoint that cannot be resolved to an operand reading the value is
  // reported rather than forced -- a spoofed or stale association must not
  // silently redirect the wrong operand, and must not pass as complete.
  for (const Chain &chain : chains) {
    if (!chain.result)
      continue;
    for (const PortRef &ref : chain.consumers) {
      mlir::Operation *consumer = materializedOpForNode(kernel, ref.node);
      if (!consumer)
        consumer = binding.opFor(ref.node);
      mlir::OpOperand *operand =
          consumer ? operandForPort(consumer, ref.index) : nullptr;
      // A recorded endpoint must currently read this connection's source value
      // (directly or through a transparent view), or the association is stale
      // and redirecting it would corrupt unrelated data. A second endpoint
      // served by the same transparent view is already rewired to the movement
      // result, which is equally valid.
      if (operand) {
        mlir::Value current = resolveThroughTransparent(operand->get());
        if (current == chain.result)
          continue;
        // A gather combines several feeds, so its consumer may currently read
        // any one of them; a movement has the single source.
        bool readsSource = current == chain.source;
        for (const mlir::Value &alt : chain.altSources)
          readsSource |= current == alt;
        if (!readsSource) {
          bound.unmaterialized.push_back(
              "value " + std::to_string(chain.value) + ": " +
              kNoConsumerReason.str() + " (node " + std::to_string(ref.node) +
              ", input " + std::to_string(ref.index) + ")");
          continue;
        }
      }
      if (!operand) {
        bound.unmaterialized.push_back(
            "value " + std::to_string(chain.value) + ": " +
            kNoConsumerReason.str() + " (node " + std::to_string(ref.node) +
            ", input " + std::to_string(ref.index) + ")");
        continue;
      }
      // A consumer reading through a transparent logical op keeps its own
      // (source-typed) operand: feed the movement's result into the view,
      // preserving the consumer's logical type so the source identity is
      // unchanged. Otherwise rewire the operand to the movement result.
      if (mlir::Operation *view = operand->get().getDefiningOp();
          view && isTransparentOp(view)) {
        view->setOperand(0, chain.result);
        continue;
      }
      mlir::Value replacement = retypeForConsumer(
          builder, chain.result, operand->get(), chain.dstMemory);
      if (!replacement) {
        bound.unmaterialized.push_back(
            "value " + std::to_string(chain.value) + ": " +
            kNoConsumerReason.str() + " (node " + std::to_string(ref.node) +
            ", input " + std::to_string(ref.index) +
            "): cannot be retyped to read the movement");
        continue;
      }
      operand->set(replacement);
    }
  }

  return llvm::Error::success();
}

} // namespace

std::unique_ptr<mapping::PlanMaterializer> createCanonicalPlanMaterializer() {
  return std::make_unique<CanonicalPlanMaterializer>();
}

} // namespace mlir::llk::micro_mapping_detail
