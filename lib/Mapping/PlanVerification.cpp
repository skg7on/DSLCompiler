//===- PlanVerification.cpp - Layered verification of mapped Micro-IR -----===//
//
// Split out of PlanBinder.cpp (task B1) so the binder stays a materializer and
// semantic checking lives on its own. The public entry point
// `verifyMappedMicroIR` is declared in PlanBinder.h and keeps its contract.
//
// Besides the phase-2 machine checks the binder already relied on, this file
// closes two stage-A findings against the schema-v2 metadata:
//
//   * A4 -- the solved-layout container is now *required* for a v2 binding (or
//     an explicit `no_layout` marker), so deleting the container no longer
//     skips layout validation;
//   * A5 -- a materialized-movement exemption now proves the recorded consumer
//     endpoints actually read a movement of the connection's value, instead of
//     only checking that the recorded consumer list is well-shaped.

#include "LLK/Mapping/PlanBinder.h"

#include "LLK/Mapping/Diagnostics.h"
#include "LLK/Mapping/MappingHelpers.h"
#include "LLK/Mapping/MappingMetadata.h"
#include "LLK/Mapping/StableHash.h"
#include "LLK/Mapping/TileFacts.h"
#include "LLK/Mapping/WorkloadGraph.h"

#include "mlir/AsmParser/AsmParser.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Verifier.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Support/Error.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace mlir::llk::mapping {

namespace {

using machine::MachineModel;

constexpr llvm::StringLiteral kPlanAttr = "micro.plan";
constexpr llvm::StringLiteral kMappingAttr = "micro.mapping";
constexpr llvm::StringLiteral kRoutesAttr = "micro.routes";

constexpr llvm::StringLiteral kValueAttr = "micro.value";
constexpr llvm::StringLiteral kSrcNodeAttr = "micro.src_node";
constexpr llvm::StringLiteral kDstNodeAttr = "micro.dst_node";
constexpr llvm::StringLiteral kConnectionAttr = "micro.connection";
constexpr llvm::StringLiteral kHopAttr = "micro.hop";

/// The Micro operations the canonical materializer emits to materialize a
/// connection's movement: the generic `micro.async_copy` for a shaped value and
/// `micro.tile_async_copy` for a tile. A completeness exemption is granted to
/// exactly these operations, and only when their connection provenance
/// resolves; no other op can claim it by carrying a movement stamp.
llvm::StringRef materializedMovementOpName(llvm::StringRef name) {
  if (name == "micro.async_copy" || name == "micro.tile_async_copy")
    return name;
  return {};
}

llvm::Error bindError(const std::string &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(), message);
}

/// A phase-2 (machine-aware) verification failure carrying its stable §22.3
/// code.
llvm::Error verifyError(DiagnosticCode code, const std::string &message) {
  return bindError((stringifyDiagnosticCode(code) + ": " + message).str());
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
    if (name == kMappingAttr || name == kValueAttr || name == kSrcNodeAttr ||
        name == kDstNodeAttr || name == kConnectionAttr || name == kHopAttr) {
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

/// The operation in `kernel` whose `micro.mapping` records source node `node`,
/// or null. Consumer endpoints name *source* node ids, and a mapped workload
/// operation records the source node it covers, so this is the materialized
/// consumer an endpoint resolves to.
mlir::Operation *mappedOpForNode(mlir::Operation *kernel, uint64_t node) {
  mlir::Operation *found = nullptr;
  kernel->walk([&](mlir::Operation *op) {
    if (found)
      return;
    auto mapping = op->getAttrOfType<mlir::DictionaryAttr>(kMappingAttr);
    if (!mapping)
      return;
    auto recorded = mapping.getAs<mlir::IntegerAttr>("node");
    if (recorded && recorded.getValue().getZExtValue() == node)
      found = op;
  });
  return found;
}

/// The printed form of a type, for the `element_type` a `LayoutContext` reads.
std::string printedTypeOf(mlir::Type type) {
  std::string text;
  llvm::raw_string_ostream stream(text);
  type.print(stream);
  return stream.str();
}

/// The layout class a `layout_parameters` key names. Placement keys a class
/// required by several ports as `class#<index>` so each occurrence keeps its
/// own assignment, so the disambiguating suffix is stripped before the
/// declaration is resolved.
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

/// A recorded parameter value in the variant the solver evaluates.
LayoutValue layoutValueOf(mlir::Attribute attribute) {
  if (auto integer = mlir::dyn_cast<mlir::IntegerAttr>(attribute))
    return integer.getInt();
  return mlir::cast<mlir::StringAttr>(attribute).getValue().str();
}

/// The rank and element type a layout requirement is validated against: the
/// operand the rule names, resolved positionally against the node's ports.
LayoutContext layoutContextForRule(const RuleDef &rule,
                                   const WorkloadNode &node,
                                   llvm::StringRef layoutId,
                                   const LayoutContext &fallback) {
  LayoutContext context = fallback;
  size_t inputIndex = 0;
  size_t outputIndex = 0;
  for (const RulePort &port : rule.ports) {
    const WorkloadPort *nodePort = nullptr;
    if (port.isInput) {
      if (inputIndex < node.inputs.size())
        nodePort = &node.inputs[inputIndex];
      ++inputIndex;
    } else {
      if (outputIndex < node.outputs.size())
        nodePort = &node.outputs[outputIndex];
      ++outputIndex;
    }
    if (!nodePort)
      continue;
    bool required = false;
    for (const RuleLayoutRequirement &requirement : rule.layoutRequirements)
      if (requirement.layoutId == layoutId && requirement.port == port.name) {
        required = true;
        break;
      }
    if (!required)
      continue;
    if (mlir::Type element = elementTypeOf(nodePort->type))
      context.elementType = printedTypeOf(element);
    if (std::optional<llvm::SmallVector<int64_t, 4>> shape =
            staticShapeOf(nodePort->type))
      context.rank = static_cast<int64_t>(shape->size());
    break;
  }
  return context;
}

/// Reads a required integer bookkeeping attribute from a materialized movement
/// with a checked cast.
llvm::Expected<uint64_t> movementUintAttr(mlir::Operation *op,
                                          llvm::StringRef name,
                                          const std::string &where) {
  std::string prefix =
      where + ": op '" + op->getName().getStringRef().str() + "'";
  mlir::Attribute raw = op->getAttr(name);
  if (!raw)
    return verifyError(DiagnosticCode::InvalidMappingMetadata,
                       prefix + ": '" + name.str() + "' is missing");
  auto integer = mlir::dyn_cast<mlir::IntegerAttr>(raw);
  if (!integer)
    return verifyError(DiagnosticCode::InvalidMappingMetadata,
                       prefix + ": '" + name.str() + "' is not an integer");
  return integer.getValue().getZExtValue();
}

llvm::Expected<std::string> movementStringAttr(mlir::Operation *op,
                                               llvm::StringRef name,
                                               const std::string &where) {
  std::string prefix =
      where + ": op '" + op->getName().getStringRef().str() + "'";
  mlir::Attribute raw = op->getAttr(name);
  if (!raw)
    return verifyError(DiagnosticCode::InvalidMappingMetadata,
                       prefix + ": '" + name.str() + "' is missing");
  auto text = mlir::dyn_cast<mlir::StringAttr>(raw);
  if (!text)
    return verifyError(DiagnosticCode::InvalidMappingMetadata,
                       prefix + ": '" + name.str() + "' is not a string");
  return text.getValue().str();
}

/// Validates that `op` is a binder-emitted materialized movement, so the
/// completeness walk may exempt it from carrying its own `micro.mapping`.
///
/// The exemption requires *all* of: the canonical movement operation; typed
/// stamps; a connection in `micro.routes` the id resolves to; a hop inside that
/// connection's route whose source/destination the copy agrees with, carried by
/// a link and transfer engine the machine declares; a type-preserving copy
/// reading kernel work; a well-formed consumer set; and -- new in schema v2 --
/// that every recorded consumer endpoint actually reads a materialized movement
/// of this connection's value (stage-A A5), rather than merely being an
/// integer-shaped list.
llvm::Error verifyMaterializedMovement(mlir::Operation *op,
                                       mlir::Operation *kernel,
                                       const MachineModel &machine,
                                       const std::string &where) {
  llvm::StringRef name = op->getName().getStringRef();
  const bool claimsMovement =
      op->hasAttr(kValueAttr) || op->hasAttr(kConnectionAttr) ||
      op->hasAttr(kHopAttr) || op->hasAttr(kDstNodeAttr);
  if (materializedMovementOpName(name).empty() || !claimsMovement)
    return verifyError(DiagnosticCode::NoMatchingRule,
                       where + ": op '" + name.str() +
                           "' carries no micro.mapping");

  auto metadataError = [&](llvm::Error error) {
    return verifyError(DiagnosticCode::InvalidMappingMetadata,
                       llvm::toString(std::move(error)));
  };

  llvm::Expected<uint64_t> connectionId =
      movementUintAttr(op, kConnectionAttr, where);
  if (!connectionId)
    return connectionId.takeError();
  llvm::Expected<uint64_t> hop = movementUintAttr(op, kHopAttr, where);
  if (!hop)
    return hop.takeError();
  llvm::Expected<uint64_t> value = movementUintAttr(op, kValueAttr, where);
  if (!value)
    return value.takeError();
  llvm::Expected<std::string> dstNode =
      movementStringAttr(op, kDstNodeAttr, where);
  if (!dstNode)
    return dstNode.takeError();

  auto routes = kernel->getAttrOfType<mlir::ArrayAttr>(kRoutesAttr);
  if (!routes)
    return verifyError(DiagnosticCode::InvalidMappingMetadata,
                       where + ": op '" + name.str() + "' names connection " +
                           std::to_string(*connectionId) +
                           ", but the kernel has no micro.routes");
  mlir::DictionaryAttr route;
  for (mlir::Attribute element : routes) {
    auto entry = mlir::dyn_cast<mlir::DictionaryAttr>(element);
    if (!entry)
      return verifyError(DiagnosticCode::InvalidMappingMetadata,
                         where + ": micro.routes entry is not a dictionary");
    auto id = entry.getAs<mlir::IntegerAttr>("id");
    if (id && id.getValue().getZExtValue() == *connectionId) {
      route = entry;
      break;
    }
  }
  if (!route)
    return verifyError(DiagnosticCode::InvalidMappingMetadata,
                       where + ": op '" + name.str() + "' names connection " +
                           std::to_string(*connectionId) +
                           ", which micro.routes does not declare");

  llvm::Expected<std::string> kind = readMetadataString(route, "kind", where);
  if (!kind)
    return metadataError(kind.takeError());
  std::optional<ConnectionKind> symbolized = symbolizeConnectionKind(*kind);
  if (!symbolized)
    return metadataError(
        bindError(where + ": unknown connection kind '" + *kind + "'"));
  switch (*symbolized) {
  case ConnectionKind::Transfer:
  case ConnectionKind::TransferAndTransform:
  case ConnectionKind::Replicate:
    break;
  case ConnectionKind::Direct:
  case ConnectionKind::LayoutTransform:
  case ConnectionKind::Reduce:
    return verifyError(DiagnosticCode::InvalidMappingMetadata,
                       where + ": op '" + name.str() +
                           "' resolves to connection " +
                           std::to_string(*connectionId) +
                           ", which is not materialized as a movement");
  }

  auto routeValue =
      mlir::dyn_cast_or_null<mlir::IntegerAttr>(route.get("value"));
  if (!routeValue)
    return verifyError(DiagnosticCode::InvalidMappingMetadata,
                       where + ": connection " + std::to_string(*connectionId) +
                           " records no integer value");
  if (routeValue.getValue().getZExtValue() != *value)
    return verifyError(
        DiagnosticCode::InvalidMappingMetadata,
        where + ": op '" + name.str() + "' records value " +
            std::to_string(*value) + ", but connection " +
            std::to_string(*connectionId) + " carries value " +
            std::to_string(routeValue.getValue().getZExtValue()));

  mlir::Attribute rawRoute = route.get("route");
  if (!rawRoute)
    return verifyError(DiagnosticCode::InvalidMappingMetadata,
                       where + ": connection " + std::to_string(*connectionId) +
                           " has no 'route'");
  llvm::Expected<llvm::SmallVector<std::string, 4>> nodes =
      readMetadataStringArray(rawRoute, "route", where);
  if (!nodes)
    return metadataError(nodes.takeError());
  if (*hop < 1 || *hop >= nodes->size())
    return verifyError(DiagnosticCode::InvalidMappingMetadata,
                       where + ": op '" + name.str() + "' names hop " +
                           std::to_string(*hop) +
                           ", which is outside connection " +
                           std::to_string(*connectionId) + "'s route");
  const std::string &fromNode = (*nodes)[*hop - 1];
  const std::string &toNode = (*nodes)[*hop];
  if (toNode != *dstNode)
    return verifyError(
        DiagnosticCode::InvalidMappingMetadata,
        where + ": op '" + name.str() + "' records micro.dst_node '" +
            *dstNode + "', but connection " + std::to_string(*connectionId) +
            " hop " + std::to_string(*hop) + " lands in '" + toNode + "'");

  // The source node identity is optional -- a movement written before B5, or a
  // hand-written one, may omit it -- but when recorded it must be this hop's
  // actual source. A same-kind movement always carries it: the structural
  // verifier rejects one that does not.
  if (mlir::Attribute rawSrcNode = op->getAttr(kSrcNodeAttr)) {
    auto srcNode = mlir::dyn_cast<mlir::StringAttr>(rawSrcNode);
    if (!srcNode)
      return verifyError(DiagnosticCode::InvalidMappingMetadata,
                         where + ": op '" + name.str() +
                             "' has a non-string micro.src_node");
    if (srcNode.getValue() != fromNode)
      return verifyError(
          DiagnosticCode::InvalidMappingMetadata,
          where + ": op '" + name.str() + "' records micro.src_node '" +
              srcNode.getValue().str() + "', but connection " +
              std::to_string(*connectionId) + " hop " + std::to_string(*hop) +
              " starts in '" + fromNode + "'");
  }

  const machine::MemoryNode *fromMemory = machine.findMemory(fromNode);
  const machine::MemoryNode *toMemory = machine.findMemory(toNode);
  if (!fromMemory || !toMemory)
    return verifyError(DiagnosticCode::InvalidMappingMetadata,
                       where + ": connection " + std::to_string(*connectionId) +
                           " hop " + std::to_string(*hop) +
                           " names a memory the machine does not declare");
  mlir::MLIRContext *context = op->getContext();
  mlir::Attribute expectedSrc = memoryAttrFor(context, fromMemory->kind);
  mlir::Attribute expectedDst = memoryAttrFor(context, toMemory->kind);
  if (!expectedSrc || !expectedDst)
    return verifyError(DiagnosticCode::InvalidMappingMetadata,
                       where + ": connection " + std::to_string(*connectionId) +
                           " hop " + std::to_string(*hop) +
                           " names a memory kind Micro cannot represent");
  // The generic copy carries both spaces as attributes. `micro.tile_async_copy`
  // names only the destination, because a tile's source and result memories are
  // part of their types; the dialect verifier already enforced that the result
  // tile's memory equals `dst_memory`, so checking the destination here (plus
  // the dialect verifier) fixes the landing memory without re-reading tiles.
  const bool tileCopy = name == "micro.tile_async_copy";
  const bool memoriesMatch =
      op->getAttr("dst_memory") == expectedDst &&
      (tileCopy || op->getAttr("src_memory") == expectedSrc);
  if (!memoriesMatch)
    return verifyError(DiagnosticCode::InvalidMappingMetadata,
                       where + ": op '" + name.str() +
                           "' declares memories that do not match "
                           "connection " +
                           std::to_string(*connectionId) + " hop " +
                           std::to_string(*hop) + " ('" + fromNode + "' -> '" +
                           toNode + "')");

  const machine::LinkEdge *link = nullptr;
  for (const machine::LinkEdge &edge : machine.links)
    if (edge.source == fromNode && edge.destination == toNode) {
      link = &edge;
      break;
    }
  if (!link)
    return verifyError(DiagnosticCode::InvalidMappingMetadata,
                       where + ": op '" + name.str() + "' hop '" + fromNode +
                           "' -> '" + toNode + "' has no link");
  llvm::SmallVector<std::string, 4> engines;
  if (mlir::Attribute rawEngines = route.get("engines")) {
    llvm::Expected<llvm::SmallVector<std::string, 4>> read =
        readMetadataStringArray(rawEngines, "engines", where);
    if (!read)
      return metadataError(read.takeError());
    engines = std::move(*read);
  }
  if (!link->transferEngines.empty()) {
    if (engines.empty())
      return verifyError(DiagnosticCode::InvalidMappingMetadata,
                         where + ": op '" + name.str() + "' hop '" + fromNode +
                             "' -> '" + toNode +
                             "' names no transfer engine, but its link "
                             "requires one");
    bool served = false;
    for (const std::string &engine : engines) {
      if (!machine.findTransferEngine(engine))
        return verifyError(DiagnosticCode::InvalidMappingMetadata,
                           where + ": op '" + name.str() +
                               "' route names unsupported transfer engine '" +
                               engine + "'");
      if (llvm::is_contained(link->transferEngines, engine))
        served = true;
    }
    if (!served)
      return verifyError(DiagnosticCode::InvalidMappingMetadata,
                         where + ": op '" + name.str() + "' hop '" + fromNode +
                             "' -> '" + toNode +
                             "' names no transfer engine its link offers");
  }

  // Source/destination kind equality is not node identity. A same-kind hop is
  // legal only between two *distinct* concrete nodes, and only when the
  // machine's own facts back the node identity: the link's transaction granule
  // and the destination's alignment must admit the moving value. Kind equality
  // alone grants nothing.
  if (fromMemory->kind == toMemory->kind) {
    if (fromNode == toNode)
      return verifyError(
          DiagnosticCode::InvalidMappingMetadata,
          where + ": op '" + name.str() + "' hop '" + fromNode + "' -> '" +
              toNode +
              "' names one memory node: equal memory kinds are not node "
              "identity");
    TileFacts facts = tileFactsFor(op->getOperand(0).getType());
    if (facts.known) {
      if (link->transactionBytes == 0 ||
          facts.bytes % link->transactionBytes != 0)
        return verifyError(
            DiagnosticCode::InvalidMappingMetadata,
            where + ": op '" + name.str() + "' hop '" + fromNode + "' -> '" +
                toNode + "' moves " + std::to_string(facts.bytes) +
                " bytes, which do not tile into whole " +
                std::to_string(link->transactionBytes) + "-byte transactions");
      if (facts.alignment > 0 &&
          toMemory->alignmentBytes % facts.alignment != 0)
        return verifyError(
            DiagnosticCode::InvalidMappingMetadata,
            where + ": op '" + name.str() + "' hop '" + fromNode + "' -> '" +
                toNode + "' needs " + std::to_string(facts.alignment) +
                "-byte alignment, which memory '" + toNode + "' (" +
                std::to_string(toMemory->alignmentBytes) +
                " bytes) does not support");
    }
  }

  if (op->getNumOperands() < 1 || op->getNumResults() < 1)
    return verifyError(DiagnosticCode::InvalidMappingMetadata,
                       where + ": op '" + name.str() +
                           "' is not a well-formed movement");
  // A generic copy must be type-preserving. A tile copy's result is the tile
  // retyped into the destination memory, so its operand and result differ by
  // that memory -- the dialect verifier already proved the shapes, element
  // types and destination memory agree, so this only distinguishes the two
  // operation forms.
  if (!tileCopy && op->getOperand(0).getType() != op->getResult(0).getType())
    return verifyError(DiagnosticCode::InvalidMappingMetadata,
                       where + ": op '" + name.str() +
                           "' copies a value whose type differs from its "
                           "result");
  mlir::Operation *producer = op->getOperand(0).getDefiningOp();
  if (!producer || enclosingKernel(producer) != kernel)
    return verifyError(
        DiagnosticCode::InvalidMappingMetadata,
        where + ": op '" + name.str() +
            "' reads a value no operation in its kernel defines");
  if (!producer->getAttr(kMappingAttr) && !producer->hasAttr(kConnectionAttr))
    return verifyError(DiagnosticCode::InvalidMappingMetadata,
                       where + ": op '" + name.str() +
                           "' reads a value that no mapped or materialized "
                           "operation produces");

  // Selected consumers: the connection's consumer set is generic metadata too,
  // so its shape is validated here with the same checked reads as every other
  // container.
  if (mlir::Attribute rawConsumers = route.get("consumers")) {
    auto consumers = mlir::dyn_cast<mlir::ArrayAttr>(rawConsumers);
    if (!consumers)
      return metadataError(bindError(where + ": 'consumers' is not an array"));
    for (mlir::Attribute consumer : consumers)
      if (!mlir::isa<mlir::IntegerAttr>(consumer))
        return metadataError(
            bindError(where + ": 'consumers' has a non-integer entry"));
  }

  // Stage-A A5: prove the recorded consumer endpoints actually read a
  // materialized movement of this connection's value. A consumer_ports array
  // present but pointing at an operand that is not a movement of the value is
  // spoofed provenance; a consumer list recorded with no endpoint at all is
  // ambiguous and cannot be replayed. Both are reported for a schema-v2
  // binding; v1 metadata (no consumer_ports) is left to historical analysis.
  if (mlir::Attribute rawConsumerPorts = route.get("consumer_ports")) {
    auto consumerPorts = mlir::dyn_cast<mlir::ArrayAttr>(rawConsumerPorts);
    if (!consumerPorts)
      return metadataError(
          bindError(where + ": 'consumer_ports' is not an array"));
    if (!consumerPorts.empty()) {
      // A recorded endpoint names a *source* node id, so it resolves against
      // the source projection, not the materialized graph whose ids shifted.
      llvm::Expected<SourceGraphView> view = buildSourceGraphView(kernel);
      if (!view)
        return metadataError(view.takeError());
      // The occurrence the movement produces from; its value is what every
      // recorded consumer endpoint must genuinely read. Comparing the
      // endpoint's *value* (not its absolute id) keeps this independent of how
      // the projection relabels values.
      mlir::Attribute producerPort = route.get("producer_port");
      if (!producerPort)
        return verifyError(
            DiagnosticCode::InvalidMappingMetadata,
            where + ": connection " + std::to_string(*connectionId) +
                " records consumer endpoints but no producer_port");
      llvm::Expected<PortRef> producer =
          readMetadataPortRef(producerPort, "producer_port", where);
      if (!producer)
        return metadataError(producer.takeError());
      const WorkloadPort *produced = lookupPort(view->graph, *producer);
      if (!produced)
        return verifyError(DiagnosticCode::InvalidMappingMetadata,
                           where + ": producer_port does not resolve in the "
                                   "source graph");
      for (mlir::Attribute element : consumerPorts) {
        llvm::Expected<PortRef> ref =
            readMetadataPortRef(element, "consumer_ports", where);
        if (!ref)
          return metadataError(ref.takeError());
        if (ref->direction != PortDirection::Input)
          return verifyError(DiagnosticCode::InvalidMappingMetadata,
                             where + ": consumer_ports entry is not an input "
                                     "occurrence");
        const WorkloadPort *consumed = lookupPort(view->graph, *ref);
        if (!consumed)
          return verifyError(DiagnosticCode::InvalidMappingMetadata,
                             where + ": consumer_ports entry does not resolve "
                                     "in the source graph");
        if (consumed->value != produced->value)
          return verifyError(
              DiagnosticCode::InvalidMappingMetadata,
              where +
                  ": consumer_ports entry does not read the value "
                  "connection " +
                  std::to_string(*connectionId) + " carries");

        // B4 strengthening of A5: the graph-level check above only proves the
        // recorded endpoint reads the same *value*; it does not prove the
        // consumer reads the emitted movement. Resolve the endpoint to its
        // materialized consumer operation and require its operand -- through
        // the transparent `micro.tile_view` the materializer inserts when a
        // shaped movement feeds a tile consumer -- to be this movement's
        // result. A consumer still reading the original value is spoofed
        // provenance, not a complete binding.
        mlir::Operation *consumer = mappedOpForNode(kernel, ref->node);
        if (!consumer)
          return verifyError(
              DiagnosticCode::InvalidMappingMetadata,
              where + ": consumer_ports entry names node " +
                  std::to_string(ref->node) +
                  ", but no mapped operation in the kernel records it");
        if (ref->index >= consumer->getNumOperands())
          return verifyError(
              DiagnosticCode::InvalidMappingMetadata,
              where + ": consumer_ports entry names input " +
                  std::to_string(ref->index) + " of node " +
                  std::to_string(ref->node) + ", which has only " +
                  std::to_string(consumer->getNumOperands()) + " operand(s)");
        mlir::Value read =
            resolveThroughTransparentOps(consumer->getOperand(ref->index));
        // The consumer must read a value this connection materialized: either
        // this movement op itself, or -- for a transfer-and-transform, or a
        // later route hop -- another operation stamped with the same
        // `micro.connection` id. Anything else is the original value, so the
        // consumer does not read the movement.
        mlir::Operation *defining = read.getDefiningOp();
        bool readsMovement = defining == op;
        if (!readsMovement && defining) {
          auto stamped =
              defining->getAttrOfType<mlir::IntegerAttr>(kConnectionAttr);
          readsMovement =
              stamped && stamped.getValue().getZExtValue() == *connectionId;
        }
        if (!readsMovement)
          return verifyError(
              DiagnosticCode::InvalidMappingMetadata,
              where + ": consumer_ports entry (node " +
                  std::to_string(ref->node) + ", input " +
                  std::to_string(ref->index) +
                  ") does not read the materialized movement of connection " +
                  std::to_string(*connectionId));
      }
    }
  }
  return llvm::Error::success();
}

/// True when the kernel's `micro.plan` records a schema-v2 (or newer) binding.
/// Delegates to `planMetadataIsSchemaV2`, so a deleted `schema_version` cannot
/// downgrade a binding that still records v2-only fields.
bool kernelUsesSchemaV2(mlir::Operation *op) {
  mlir::Operation *kernel = enclosingKernel(op);
  if (!kernel)
    return false;
  return kernelMetadataIsSchemaV2(kernel);
}

} // namespace

llvm::Error verifyMappedMicroIR(mlir::ModuleOp module,
                                const MappingTarget &target) {
  // 1. structural
  if (mlir::failed(mlir::verify(module)))
    return bindError("verifyMappedMicroIR: structural verification failed");

  const MachineModel &machine = target.machine();
  llvm::Error failure = llvm::Error::success();
  auto fail = [&](DiagnosticCode code, std::string message) {
    failure = verifyError(code, std::move(message));
  };
  auto failMetadata = [&](llvm::Error error) {
    fail(DiagnosticCode::InvalidMappingMetadata,
         llvm::toString(std::move(error)));
  };

  // --- 2a. kernel completeness -------------------------------------------
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

    // A schema version newer than this reader understands must not be replayed
    // under these semantics.
    if (auto version = plan.getAs<mlir::IntegerAttr>("schema_version")) {
      if (version.getInt() < 0 || static_cast<uint64_t>(version.getInt()) >
                                      kSupportedMappingMetadataVersion) {
        failMetadata(
            bindError(where + ": micro.plan schema_version is not supported"));
        return;
      }
    }

    // Task B6: a fully materialized plan that records a synchronization
    // requiring a barrier must have that barrier represented in the IR. A
    // required barrier the kernel does not express is synchronization the
    // binding silently dropped, so it is rejected rather than passed. An
    // unmaterialized (partial) plan is exempt: it is honestly reporting what it
    // could not build.
    const bool fullyMaterialized = [&] {
      if (auto materialized = plan.getAs<mlir::BoolAttr>("materialized"))
        return materialized.getValue();
      return true;
    }();
    if (fullyMaterialized) {
      if (mlir::Attribute rawSync = plan.get("synchronization")) {
        auto sync = mlir::dyn_cast<mlir::ArrayAttr>(rawSync);
        if (!sync) {
          failMetadata(bindError(
              where + ": micro.plan 'synchronization' is not an array"));
          return;
        }
        for (mlir::Attribute element : sync) {
          auto entry = mlir::dyn_cast<mlir::DictionaryAttr>(element);
          if (!entry) {
            failMetadata(
                bindError(where + ": micro.plan synchronization entry is not a "
                                  "dictionary"));
            return;
          }
          auto barrier = entry.getAs<mlir::BoolAttr>("requires_barrier");
          if (!barrier || !barrier.getValue())
            continue;
          llvm::SmallVector<uint64_t, 2> waited;
          if (auto waits = entry.getAs<mlir::ArrayAttr>("waits_for"))
            for (mlir::Attribute id : waits)
              if (auto integer = mlir::dyn_cast<mlir::IntegerAttr>(id))
                waited.push_back(integer.getValue().getZExtValue());
          bool represented = false;
          kernel->walk([&](mlir::Operation *op) {
            if (represented || op->getName().getStringRef() != "micro.barrier")
              return;
            if (waited.empty()) {
              represented = true;
              return;
            }
            if (auto stamped =
                    op->getAttrOfType<mlir::IntegerAttr>(kConnectionAttr))
              represented =
                  llvm::is_contained(waited, stamped.getValue().getZExtValue());
          });
          if (!represented) {
            fail(DiagnosticCode::InvalidMappingMetadata,
                 where +
                     ": the plan requires synchronization the kernel does not "
                     "express (no micro.barrier for the waited connection)");
            return;
          }
        }
      }
    }

    kernel->walk([&](mlir::Operation *op) {
      if (failure)
        return;
      if (!isWorkloadNodeOp(op->getName()))
        return;
      if (op->getAttr(kMappingAttr))
        return;
      if (llvm::Error error =
              verifyMaterializedMovement(op, kernel, machine, where))
        failure = std::move(error);
    });
  });
  if (failure)
    return failure;

  // --- 2b. per-operation metadata ----------------------------------------
  struct KernelWorkload {
    WorkloadGraph graph;
    llvm::DenseMap<mlir::Operation *, const WorkloadNode *> byOp;
  };
  struct NodeLookup {
    const WorkloadNode *node = nullptr;
    std::string error;
  };
  llvm::DenseMap<mlir::Operation *, std::unique_ptr<KernelWorkload>> workloads;
  auto workloadNodeFor = [&](mlir::Operation *op) -> NodeLookup {
    mlir::Operation *kernel = enclosingKernel(op);
    if (!kernel)
      return {nullptr, "it is not inside a micro.kernel"};
    auto entry = workloads.find(kernel);
    if (entry == workloads.end()) {
      WorkloadGraphBinding binding;
      llvm::Expected<WorkloadGraph> graph =
          extractWorkloadGraph(kernel, &binding);
      if (!graph)
        return {nullptr,
                "its kernel's workload graph could not be extracted: " +
                    llvm::toString(graph.takeError())};
      auto fresh = std::make_unique<KernelWorkload>();
      fresh->graph = std::move(*graph);
      for (const auto &pair : binding.nodeOps)
        fresh->byOp[pair.second] = fresh->graph.findNode(pair.first);
      entry = workloads.insert({kernel, std::move(fresh)}).first;
    }
    auto found = entry->second->byOp.find(op);
    if (found == entry->second->byOp.end())
      return {nullptr,
              "it is not a workload node in its kernel's workload graph"};
    return {found->second, {}};
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
    const bool schemaV2 = kernelUsesSchemaV2(op);

    llvm::Expected<std::string> ruleId =
        readMetadataString(mapping, "rule", where);
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
    NodeLookup lookup = workloadNodeFor(op);
    if (!lookup.node) {
      fail(DiagnosticCode::InvalidMappingMetadata,
           where + ": cannot verify the recorded rule '" + *ruleId +
               "': " + lookup.error);
      return;
    }
    if (rule->matchOp != op->getName().getStringRef()) {
      fail(DiagnosticCode::NoMatchingRule,
           where + ": rule '" + *ruleId + "' implements '" + rule->matchOp +
               "', not '" + op->getName().getStringRef().str() + "'");
      return;
    }

    llvm::Expected<std::string> executor =
        readMetadataString(mapping, "executor", where);
    if (!executor) {
      failMetadata(executor.takeError());
      return;
    }
    if (!machine.findExecutor(*executor)) {
      fail(DiagnosticCode::NoLegalExecutor,
           where + ": unknown executor '" + *executor + "'");
      return;
    }

    llvm::StringMap<std::string> memories;
    if (mlir::Attribute rawMemories = mapping.get("memories")) {
      llvm::Expected<llvm::StringMap<std::string>> read =
          readMetadataStringMap(rawMemories, "memories", where);
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
    // Named-port memory associations (task B2). Each recorded occurrence must
    // resolve in the source graph and name a memory the executor can see; the
    // association, not the kind-keyed map, is what a named requirement is
    // re-checked against.
    std::vector<PortMemoryBinding> portMemories;
    if (mlir::Attribute rawPortMemories = mapping.get("port_memories")) {
      auto array = mlir::dyn_cast<mlir::ArrayAttr>(rawPortMemories);
      if (!array) {
        failMetadata(bindError(where + ": 'port_memories' is not an array"));
        return;
      }
      for (mlir::Attribute element : array) {
        auto entry = mlir::dyn_cast<mlir::DictionaryAttr>(element);
        if (!entry) {
          failMetadata(
              bindError(where + ": 'port_memories' entry is not a dictionary"));
          return;
        }
        llvm::Expected<PortRef> ref =
            readMetadataPortRef(entry.get("port"), "port", where);
        if (!ref) {
          failMetadata(ref.takeError());
          return;
        }
        // The recorded occurrence names *this* op's source node (the id in the
        // mapping's `node`), and must resolve in the source graph -- the same
        // two checks a recorded layout endpoint gets. The materialized graph's
        // ids shift once the binder inserts movement ops, so the association is
        // re-keyed to the mapped node's materialized id for the replay check.
        auto recordedNode = mapping.getAs<mlir::IntegerAttr>("node");
        if (!recordedNode || ref->node != recordedNode.getInt()) {
          fail(DiagnosticCode::NoMemoryRoute,
               where + ": port_memories port does not name this operation's "
                       "node");
          return;
        }
        llvm::ArrayRef<WorkloadPort> ports =
            ref->direction == PortDirection::Input ? lookup.node->inputs
                                                   : lookup.node->outputs;
        if (ref->index >= ports.size()) {
          fail(DiagnosticCode::NoMemoryRoute,
               where + ": port_memories port does not resolve on the mapped "
                       "operation");
          return;
        }
        llvm::Expected<SourceGraphView> sourceView =
            buildSourceGraphView(enclosingKernel(op));
        if (!sourceView) {
          failMetadata(sourceView.takeError());
          return;
        }
        if (!lookupPort(sourceView->graph, *ref)) {
          fail(DiagnosticCode::NoMemoryRoute,
               where + ": port_memories port does not resolve in the source "
                       "graph");
          return;
        }
        llvm::Expected<std::string> memory =
            readMetadataString(entry, "memory", where);
        if (!memory) {
          failMetadata(memory.takeError());
          return;
        }
        if (!machine.findMemory(*memory)) {
          fail(DiagnosticCode::NoMemoryRoute,
               where + ": unknown memory '" + *memory + "'");
          return;
        }
        if (!machine.isVisible(*memory, *executor)) {
          fail(DiagnosticCode::NoMemoryRoute,
               where + ": executor '" + *executor + "' cannot see memory '" +
                   *memory + "'");
          return;
        }
        portMemories.push_back(PortMemoryBinding{
            PortRef{lookup.node->id, ref->direction, ref->index}, *memory});
      }
    }
    for (const KindRequirement &requirement : rule->kindRequirements) {
      if (requirement.role != "memory")
        continue;
      if (requirement.port) {
        std::optional<PortRef> ref =
            portRefForRulePort(*rule, *lookup.node, *requirement.port);
        if (!ref) {
          fail(DiagnosticCode::NoMemoryRoute,
               where + ": rule '" + *ruleId + "' requires a '" +
                   requirement.kind + "' memory on port '" +
                   requirement.port->name +
                   "', which the operation does not expose");
          return;
        }
        bool bound = false;
        for (const PortMemoryBinding &entry : portMemories)
          bound |= entry.port == *ref;
        if (!bound) {
          // Memory-specific code: this is a missing memory association, not an
          // executor problem.
          fail(DiagnosticCode::NoMemoryRoute,
               where + ": rule '" + *ruleId + "' requires a '" +
                   requirement.kind + "' memory on port '" +
                   requirement.port->name +
                   "', which the mapping does not bind");
          return;
        }
      } else if (!memories.count(requirement.kind)) {
        fail(DiagnosticCode::NoMemoryRoute,
             where + ": rule '" + *ruleId + "' requires a '" +
                 requirement.kind +
                 "' memory, which the mapping does not bind");
        return;
      }
    }

    WorkloadNode endpoint = strippedWorkloadNode(*lookup.node);
    RecordedRuleSelection selection;
    selection.executor = *executor;
    for (const auto &entry : memories)
      selection.memories[entry.first()] = entry.second;
    selection.portMemories = std::move(portMemories);
    // The resolved rule parameters (task B1, Critical 1). A v2 binding must
    // record the container; when it does, verification validates *that*
    // assignment via `verifyRuleSelection`'s recorded-assignment branch instead
    // of generation's existential fallback, so a tampered parameter is rejected
    // rather than substituted.
    if (mlir::Attribute rawRuleParameters = mapping.get("rule_parameters")) {
      auto parameters = mlir::dyn_cast<mlir::DictionaryAttr>(rawRuleParameters);
      if (!parameters) {
        failMetadata(
            bindError(where + ": 'rule_parameters' is not a dictionary"));
        return;
      }
      for (const mlir::NamedAttribute &parameter : parameters) {
        llvm::Expected<SearchValue> value =
            readMetadataSearchValue(parameter.getValue(), where);
        if (!value) {
          failMetadata(value.takeError());
          return;
        }
        selection.parameters[parameter.getName()] = *value;
      }
    } else if (schemaV2) {
      failMetadata(bindError(
          where + ": micro.mapping records no 'rule_parameters' for a v2 "
                  "binding"));
      return;
    }
    // An *empty* recorded assignment must not downgrade a parametrized rule to
    // generation's existential fallback: for a v2 binding whose rule derives a
    // parameter, the derivation is required, so an empty map is a missing
    // assignment rather than a rule with nothing to record. (A rule whose
    // constraints reference no declared parameter may still record an empty
    // map.)
    if (schemaV2 && selection.parameters.empty() &&
        ruleDerivesParameters(*rule)) {
      fail(DiagnosticCode::NoMatchingRule,
           where + ": rule '" + *ruleId +
               "' requires a recorded parameter assignment, but the mapping "
               "records none");
      return;
    }
    if (llvm::Error error =
            verifyRuleSelection(*rule, endpoint, machine, selection, where)) {
      failure = std::move(error);
      return;
    }

    llvm::StringMap<std::string> layouts;
    if (mlir::Attribute rawLayouts = mapping.get("layouts")) {
      llvm::Expected<llvm::StringMap<std::string>> read =
          readMetadataStringMap(rawLayouts, "layouts", where);
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
        bool required =
            llvm::any_of(rule->layoutRequirements,
                         [&](const RuleLayoutRequirement &requirement) {
                           return requirement.layoutId == entry.first();
                         });
        if (!required) {
          fail(DiagnosticCode::NoLegalLayout,
               where + ": rule '" + *ruleId + "' does not require layout '" +
                   entry.first().str() + "', which the mapping records");
          return;
        }
        if (entry.second != entry.first()) {
          fail(DiagnosticCode::NoLegalLayout,
               where + ": layout class '" + entry.first().str() +
                   "' records family '" + entry.second +
                   "', which is not its own declaration");
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

    // Stage-A A4: for schema v2 a mapped operation must record *exactly one* of
    // the solved-layout container or an explicit `no_layout` marker, matching
    // whether its rule requires a layout. Deleting the container can therefore
    // no longer skip solved-layout verification; a v1 binding keeps its
    // historical shape for analysis.
    if (schemaV2) {
      const bool hasContainer = mapping.get("layout_parameters") != nullptr;
      const bool noLayoutMarker = [&] {
        auto marker = mapping.getAs<mlir::BoolAttr>("no_layout");
        return marker && marker.getValue();
      }();
      const bool requiresLayout = !rule->layoutRequirements.empty();
      if (requiresLayout && noLayoutMarker) {
        fail(DiagnosticCode::NoLegalLayout,
             where + ": rule '" + *ruleId +
                 "' requires a layout, but the mapping records no_layout");
        return;
      }
      if (!requiresLayout && hasContainer) {
        fail(DiagnosticCode::NoLegalLayout,
             where + ": rule '" + *ruleId +
                 "' requires no layout, but the mapping records "
                 "layout_parameters");
        return;
      }
      if (requiresLayout && !hasContainer) {
        fail(DiagnosticCode::NoLegalLayout,
             where + ": rule '" + *ruleId +
                 "' requires a layout, but the mapping records no "
                 "layout_parameters");
        return;
      }
      if (!requiresLayout && !noLayoutMarker) {
        failMetadata(bindError(where + ": rule '" + *ruleId +
                               "' requires no layout, but the mapping records "
                               "neither layout_parameters nor no_layout"));
        return;
      }
    }

    llvm::Expected<std::string> bundleName =
        readMetadataString(mapping, "bundle", where);
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
        readMetadataString(mapping, "emitter", where);
    if (!emitterKey) {
      failMetadata(emitterKey.takeError());
      return;
    }
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
      // Schema v2 also records a per-solution endpoint, family and concrete
      // map in `layout_entries`. Collect and type-check them; they are
      // validated per solution below, so an endpoint or map that was tampered
      // with cannot ride along unreviewed.
      llvm::StringMap<mlir::DictionaryAttr> layoutEntries;
      if (mlir::Attribute rawEntries = mapping.get("layout_entries")) {
        auto array = mlir::dyn_cast<mlir::ArrayAttr>(rawEntries);
        if (!array) {
          failMetadata(bindError(where + ": 'layout_entries' is not an array"));
          return;
        }
        for (mlir::Attribute element : array) {
          auto dict = mlir::dyn_cast<mlir::DictionaryAttr>(element);
          if (!dict) {
            failMetadata(bindError(where + ": 'layout_entries' entry is not a "
                                           "dictionary"));
            return;
          }
          auto key = dict.getAs<mlir::StringAttr>("key");
          if (!key) {
            failMetadata(bindError(
                where + ": 'layout_entries' entry has no string 'key'"));
            return;
          }
          layoutEntries[key.getValue().str()] = dict;
        }
      }
      if (schemaV2 && layoutEntries.size() != byClass.size()) {
        fail(DiagnosticCode::NoLegalLayout,
             where + ": 'layout_entries' does not describe every recorded "
                     "layout assignment");
        return;
      }
      for (const RuleLayoutRequirement &requirement :
           rule->layoutRequirements) {
        bool recorded =
            llvm::any_of(byClass, [&](const mlir::NamedAttribute &entry) {
              return baseLayoutClass(entry.getName().getValue()) ==
                     requirement.layoutId;
            });
        if (!recorded) {
          fail(DiagnosticCode::NoLegalLayout,
               where + ": rule '" + *ruleId + "' requires layout '" +
                   requirement.layoutId +
                   "', which the mapping records no assignment for");
          return;
        }
      }
      for (const mlir::NamedAttribute &entry : byClass) {
        if (failure)
          return;
        llvm::StringRef layoutClass =
            baseLayoutClass(entry.getName().getValue());
        bool required =
            llvm::any_of(rule->layoutRequirements,
                         [&](const RuleLayoutRequirement &requirement) {
                           return requirement.layoutId == layoutClass;
                         });
        if (!required) {
          fail(DiagnosticCode::NoLegalLayout,
               where + ": rule '" + *ruleId + "' does not require layout '" +
                   layoutClass.str() + "', which the mapping records");
          return;
        }
        const LayoutDef *def = target.layouts().find(layoutClass);
        if (!def) {
          fail(DiagnosticCode::NoLegalLayout,
               where + ": unknown layout '" + layoutClass.str() + "'");
          return;
        }
        llvm::StringMap<LayoutValue> recordedValues;
        for (const mlir::NamedAttribute &parameter :
             mlir::cast<mlir::DictionaryAttr>(entry.getValue()))
          recordedValues[parameter.getName()] =
              layoutValueOf(parameter.getValue());
        // Validate the v2 layout entry before re-solving: its family must be
        // the class's own declaration, its endpoint (when recorded) must name
        // this operation's own source node and resolve, and its concrete map is
        // the one the recorded parameters must rebuild.
        mlir::AffineMap recordedMap;
        if (schemaV2) {
          auto found = layoutEntries.find(entry.getName().getValue());
          if (found == layoutEntries.end()) {
            fail(DiagnosticCode::NoLegalLayout,
                 where + ": no 'layout_entries' entry for layout '" +
                     layoutClass.str() + "'");
            return;
          }
          mlir::DictionaryAttr detail = found->second;
          auto family = detail.getAs<mlir::StringAttr>("family");
          if (!family || family.getValue() != layoutClass) {
            fail(
                DiagnosticCode::NoLegalLayout,
                where + ": layout class '" + layoutClass.str() +
                    "' records family '" +
                    (family ? family.getValue().str() : std::string("<none>")) +
                    "', which is not its own declaration");
            return;
          }
          if (mlir::Attribute rawPort = detail.get("port")) {
            llvm::Expected<PortRef> ref =
                readMetadataPortRef(rawPort, "port", where);
            if (!ref) {
              failMetadata(ref.takeError());
              return;
            }
            auto recordedNode = mapping.getAs<mlir::IntegerAttr>("node");
            if (!recordedNode || ref->node != recordedNode.getInt()) {
              fail(DiagnosticCode::NoLegalLayout,
                   where + ": layout endpoint does not name this operation's "
                           "node");
              return;
            }
            llvm::Expected<SourceGraphView> view =
                buildSourceGraphView(enclosingKernel(op));
            if (!view) {
              failMetadata(view.takeError());
              return;
            }
            if (!lookupPort(view->graph, *ref)) {
              fail(DiagnosticCode::NoLegalLayout,
                   where + ": layout endpoint does not resolve in the source "
                           "graph");
              return;
            }
          }
          if (auto mapAttr = detail.getAs<mlir::AffineMapAttr>("map"))
            recordedMap = mapAttr.getValue();
        }
        LayoutContext portContext = layoutContextForRule(
            *rule, *lookup.node, layoutClass, LayoutContext{});
        if (llvm::Error error = verifySolvedLayout(
                *def, machine, *module.getContext(), portContext,
                recordedValues, recordedMap, {}, where)) {
          failure = std::move(error);
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
      if (mlir::Attribute rawId = route.get("id"))
        if (!mlir::isa<mlir::IntegerAttr>(rawId)) {
          failMetadata(bindError(where + ": route 'id' is not an integer"));
          return;
        }
      llvm::Expected<std::string> kind =
          readMetadataString(route, "kind", where);
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
          readMetadataStringArray(rawRoute, "route", where);
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
            readMetadataStringArray(rawEngines, "engines", where);
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
  if (failure)
    return failure;

  // --- 2d. frozen v2 identity -------------------------------------------
  // A frozen schema-v2 plan cannot bypass current graph/target verification:
  // its recorded source-graph and target hashes must still match. This runs
  // last so a semantic defect keeps its specific §22.3 code rather than being
  // masked by the coarser "identity changed" diagnostic.
  module->walk([&](mlir::Operation *kernel) {
    if (failure)
      return;
    if (kernel->getName().getStringRef() != "micro.kernel")
      return;
    auto plan = kernel->getAttrOfType<mlir::DictionaryAttr>(kPlanAttr);
    if (!plan)
      return;
    // v2 is inferred from any v2-only field too, so deleting `schema_version`
    // cannot disable these content-hash checks.
    if (!kernelMetadataIsSchemaV2(kernel))
      return;
    const std::string where = "kernel " + kernelLabel(kernel);
    llvm::Expected<std::string> graphHash =
        readMetadataString(plan, "graph_hash", where);
    if (!graphHash) {
      failMetadata(graphHash.takeError());
      return;
    }
    llvm::Expected<std::string> targetHash =
        readMetadataString(plan, "target_hash", where);
    if (!targetHash) {
      failMetadata(targetHash.takeError());
      return;
    }
    llvm::Expected<uint64_t> current = computeModuleSourceGraphHash(kernel);
    if (!current) {
      failMetadata(current.takeError());
      return;
    }
    if (*graphHash != hexId(*current)) {
      failMetadata(bindError(
          where + ": micro.plan graph_hash does not match the kernel's source "
                  "graph"));
      return;
    }
    if (*targetHash != hexId(computeTargetContentHash(target))) {
      failMetadata(bindError(where +
                             ": micro.plan target_hash does not match the "
                             "target"));
      return;
    }
  });
  return failure;
}

} // namespace mlir::llk::mapping
