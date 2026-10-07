//===- MappingPlan.cpp - Mapping candidates, instances, connections, plans ===//

#include "LLK/Mapping/MappingPlan.h"

#include "LLK/Mapping/StableHash.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/raw_ostream.h"

#include <array>

namespace mlir::llk::mapping {

namespace {

std::string renderValue(const std::string &value) { return value; }
std::string renderValue(uint64_t value) { return std::to_string(value); }

/// `key=value` entries of a string-keyed map, sorted by the rendered string so
/// iteration order of the map never leaks into an id.
template <typename MapT>
std::vector<std::string> sortedEntries(const MapT &map) {
  std::vector<std::string> entries;
  entries.reserve(map.size());
  for (const auto &entry : map) {
    std::string text = entry.first().str();
    text += '=';
    text += renderValue(entry.second);
    entries.push_back(std::move(text));
  }
  llvm::sort(entries);
  return entries;
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

/// A port-to-memory assignment list rendered `port=memory`, sorted so the order
/// entries were appended in never reaches an id. Empty for an instance or
/// placement whose rule declares no named-port memory requirement, so a
/// canonical string for such a value is unchanged.
std::string
portMemoryBindingsString(const std::vector<PortMemoryBinding> &bindings) {
  std::vector<std::string> entries;
  entries.reserve(bindings.size());
  for (const PortMemoryBinding &binding : bindings) {
    std::string text = canonicalPortRefString(binding.port);
    text += '=';
    text += binding.memory;
    entries.push_back(std::move(text));
  }
  llvm::sort(entries);
  return joinStrings(entries, ",");
}

/// Renders numeric ids as a sorted, comma-separated list, so id order in the
/// source vector never leaks into a canonical string.
template <typename Container> std::string joinNumbers(const Container &input) {
  llvm::SmallVector<typename Container::value_type, 8> values(input.begin(),
                                                              input.end());
  llvm::sort(values);
  std::string out;
  for (auto value : values) {
    if (!out.empty())
      out += ',';
    out += std::to_string(value);
  }
  return out;
}

std::string mapString(const AffineMap &map) {
  if (!map)
    return "<null>";
  std::string text;
  llvm::raw_string_ostream stream(text);
  map.print(stream);
  return stream.str();
}

std::string transformString(const LayoutTransform &transform) {
  std::string out = transform.srcLayout;
  out += "->";
  out += transform.dstLayout;
  out += ":src=";
  out += mapString(transform.srcMap);
  out += ":dst=";
  out += mapString(transform.dstMap);
  if (!transform.memoryNode.empty())
    out += ":memory=" + transform.memoryNode;
  if (!transform.computeResource.empty())
    out += ":compute=" + transform.computeResource;
  return out;
}

/// The solved layout assignments of an instance or placement: one
/// `class{params}` group per layout class, sorted so map iteration order never
/// leaks into an id. The parameters render through `canonicalSearchValueString`
/// (sorted and type-tagged, so `8` and `"8"` differ). The affine map is
/// deliberately omitted -- see `SolvedLayout`: it is a function of the class id
/// and the values rendered here, so hashing its printed form would add a
/// dependency on MLIR's printer for no distinguishing power.
std::string
solvedLayoutsString(const llvm::StringMap<SolvedLayout> &solutions) {
  std::vector<std::string> groups;
  groups.reserve(solutions.size());
  for (const auto &entry : solutions) {
    std::string text = entry.first().str();
    text += '{';
    text += canonicalSearchValueString(entry.second.parameters);
    text += '}';
    groups.push_back(std::move(text));
  }
  llvm::sort(groups);
  return joinStrings(groups, ";");
}

/// One bundle parameter as `key:type:value`. The type tag keeps a string `8`
/// and an integer `8` from colliding, so two typed parameters that differ only
/// in type still get different ids.
std::string bundleParameterString(mlir::NamedAttribute entry) {
  std::string text = entry.getName().str();
  text += ':';
  mlir::Attribute value = entry.getValue();
  if (auto integer = mlir::dyn_cast<mlir::IntegerAttr>(value)) {
    text += "int:";
    text += std::to_string(integer.getInt());
    return text;
  }
  if (auto string = mlir::dyn_cast<mlir::StringAttr>(value)) {
    text += "str:";
    text += string.getValue().str();
    return text;
  }
  text += "attr:";
  std::string printed;
  llvm::raw_string_ostream stream(printed);
  value.print(stream);
  text += stream.str();
  return text;
}

/// The opaque bundle's canonical content: name, emitter key, then its typed
/// parameters sorted by the rendered `key:type:value` so declaration and
/// dictionary iteration order never leak into an id. Generic code does not
/// interpret any field; it only makes the bundle content-addressed.
std::string bundleString(const TargetBundle &bundle) {
  std::vector<std::string> parameters;
  if (bundle.parameters) {
    parameters.reserve(bundle.parameters.size());
    for (mlir::NamedAttribute entry : bundle.parameters)
      parameters.push_back(bundleParameterString(entry));
    llvm::sort(parameters);
  }
  std::string out = bundle.name;
  out += "|emit=";
  out += bundle.emitterKey;
  out += "|parameters=";
  out += joinStrings(parameters, ",");
  return out;
}

} // namespace

std::string
canonicalComputeBindingsString(const llvm::StringMap<std::string> &bindings) {
  std::vector<std::string> entries;
  entries.reserve(bindings.size());
  for (const auto &entry : bindings) {
    std::string text = std::to_string(entry.first().size());
    text += ':';
    text += entry.first().str();
    text += '=';
    text += std::to_string(entry.second.size());
    text += ':';
    text += entry.second;
    entries.push_back(std::move(text));
  }
  llvm::sort(entries);
  return joinStrings(entries, ",");
}

std::optional<std::string>
missingComputeBinding(const PlanPlacement &placement) {
  for (const std::string &kind : placement.computeRequirements)
    if (!placement.computeBindings.count(kind))
      return kind;
  return std::nullopt;
}

llvm::StringRef stringifyConnectionKind(ConnectionKind kind) {
  switch (kind) {
  case ConnectionKind::Direct:
    return "direct";
  case ConnectionKind::Transfer:
    return "transfer";
  case ConnectionKind::LayoutTransform:
    return "layout_transform";
  case ConnectionKind::TransferAndTransform:
    return "transfer_and_transform";
  case ConnectionKind::Replicate:
    return "replicate";
  case ConnectionKind::Reduce:
    return "reduce";
  }
  return "";
}

std::optional<ConnectionKind> symbolizeConnectionKind(llvm::StringRef text) {
  constexpr std::array<std::pair<llvm::StringLiteral, ConnectionKind>, 6>
      kKinds{{
          {"direct", ConnectionKind::Direct},
          {"transfer", ConnectionKind::Transfer},
          {"layout_transform", ConnectionKind::LayoutTransform},
          {"transfer_and_transform", ConnectionKind::TransferAndTransform},
          {"replicate", ConnectionKind::Replicate},
          {"reduce", ConnectionKind::Reduce},
      }};
  for (const auto &entry : kKinds)
    if (entry.first == text)
      return entry.second;
  return std::nullopt;
}

llvm::StringRef stringifyPlanStepKind(PlanStepKind kind) {
  switch (kind) {
  case PlanStepKind::Compute:
    return "compute";
  case PlanStepKind::Movement:
    return "movement";
  case PlanStepKind::Synchronization:
    return "synchronization";
  }
  return "";
}

std::optional<PlanStepKind> symbolizePlanStepKind(llvm::StringRef text) {
  constexpr std::array<std::pair<llvm::StringLiteral, PlanStepKind>, 3> kKinds{{
      {"compute", PlanStepKind::Compute},
      {"movement", PlanStepKind::Movement},
      {"synchronization", PlanStepKind::Synchronization},
  }};
  for (const auto &entry : kKinds)
    if (entry.first == text)
      return entry.second;
  return std::nullopt;
}

std::string canonicalCandidateString(const MappingCandidate &candidate) {
  std::vector<std::string> ports;
  ports.reserve(candidate.ports.size());
  for (const PortSpec &port : candidate.ports) {
    std::string text = port.name;
    text += ':';
    text += std::to_string(port.value);
    text += port.isInput ? ":in" : ":out";
    // The occurrence, when resolved, is part of the port's identity: two
    // operand uses of one value must not collapse into one spec. Braces keep
    // the reference's commas from reading as the port-list separator.
    if (port.port) {
      text += ":{";
      text += canonicalPortRefString(*port.port);
      text += '}';
    }
    ports.push_back(std::move(text));
  }
  llvm::sort(ports);

  std::vector<std::string> executors;
  executors.reserve(candidate.executorRequirements.size());
  for (const ExecutorRequirement &requirement :
       candidate.executorRequirements) {
    std::string text = requirement.capability;
    text += '{';
    text += joinStrings(sortedEntries(requirement.attrs), ",");
    text += '}';
    executors.push_back(std::move(text));
  }
  llvm::sort(executors);

  std::vector<std::string> memories;
  memories.reserve(candidate.memoryRequirements.size());
  for (const MemoryRequirement &requirement : candidate.memoryRequirements) {
    std::string text = requirement.kind;
    text += ':';
    text += std::to_string(requirement.minBytes);
    memories.push_back(std::move(text));
  }
  llvm::sort(memories);

  std::vector<std::string> layouts;
  layouts.reserve(candidate.layoutRequirements.size());
  for (const LayoutRequirement &requirement : candidate.layoutRequirements)
    layouts.push_back(requirement.layoutClass);
  llvm::sort(layouts);

  std::vector<std::string> computes;
  computes.reserve(candidate.computeRequirements.size());
  for (const ComputeRequirement &requirement : candidate.computeRequirements)
    computes.push_back(requirement.kind);
  llvm::sort(computes);

  llvm::SmallVector<WorkloadNodeId> covered(candidate.coveredNodes);
  sortUnique(covered);

  std::string out = "rule=";
  out += candidate.rule;
  out += "|bundle=";
  out += bundleString(candidate.bundle);
  out += "|covered=";
  out += joinNumbers(covered);
  out += "|ports=";
  out += joinStrings(ports, ",");
  out += "|exec=";
  out += joinStrings(executors, ";");
  out += "|mem=";
  out += joinStrings(memories, ";");
  out += "|layout=";
  out += joinStrings(layouts, ";");
  out += "|compute=";
  out += joinStrings(computes, ";");
  out += "|params=";
  out += canonicalSearchValueString(candidate.resolvedParameters);
  out += "|lower=";
  out += canonicalCostString(candidate.lowerBound);
  return out;
}

std::string canonicalInstanceString(const CandidateInstance &instance) {
  // The `v3` tag makes the identity explicitly versioned (issue #129, task R1):
  // a change to what the string folds -- here, dropping the derived
  // `localCost` and folding the selected compute nodes length-delimited -- must
  // not silently hash to the same id as the older content.
  std::string out = "v3|candidate=";
  out += std::to_string(instance.candidate);
  out += "|bundle=";
  out += bundleString(instance.bundle);
  out += "|exec=";
  out += joinStrings(sortedEntries(instance.executorBindings), ",");
  out += "|mem=";
  out += joinStrings(sortedEntries(instance.memoryBindings), ",");
  // A named-port assignment is execution-affecting (which node holds which
  // occurrence), so it joins the id -- but only when present, so a rule that
  // declares no named-port requirement keeps its pre-existing id.
  if (!instance.portMemoryBindings.empty()) {
    out += "|portmem=";
    out += portMemoryBindingsString(instance.portMemoryBindings);
  }
  out += "|layout=";
  out += joinStrings(sortedEntries(instance.layoutBindings), ",");
  out += "|solvedlayout=";
  out += solvedLayoutsString(instance.layoutSolutions);
  out += "|compute=";
  out += canonicalComputeBindingsString(instance.computeBindings);
  out += "|slots=";
  out += std::to_string(instance.resourceUsage.executorSlots);
  out += "|membytes=";
  out += joinStrings(sortedEntries(instance.resourceUsage.memoryBytes), ",");
  // `localCost` is deliberately absent (issue #129, task R1): it is a derived
  // ranking score, not a decision, and folding it in forced a plan to carry its
  // final score before it could acquire an id.
  return out;
}

std::string canonicalConnectionString(const ConnectionPlan &connection) {
  llvm::SmallVector<InstanceId> consumers(connection.consumers);
  llvm::SmallVector<InstanceId> producers(connection.producers);
  llvm::SmallVector<MemoryNodeId> route(connection.memoryRoute);
  llvm::SmallVector<ExecutorId> engines(connection.transferEngines);
  std::vector<std::string> consumerMaps;
  consumerMaps.reserve(connection.consumerMaps.size());
  for (const AffineMap &map : connection.consumerMaps)
    consumerMaps.push_back(mapString(map));
  llvm::sort(consumerMaps);

  std::string out = "producer=";
  out += std::to_string(connection.producer);
  out += "|consumers=";
  out += joinNumbers(consumers);
  out += "|producers=";
  out += joinNumbers(producers);
  out += "|value=";
  out += std::to_string(connection.value);
  out += "|kind=";
  out += stringifyConnectionKind(connection.kind).str();
  // Endpoint occurrences join the identity once resolved. They are omitted
  // entirely while unset, so a connection built without endpoints keeps the id
  // it had before endpoint resolution migrated in.
  if (connection.producerPort) {
    out += "|producerPort={";
    out += canonicalPortRefString(*connection.producerPort);
    out += '}';
  }
  if (!connection.consumerPorts.empty()) {
    std::vector<std::string> consumerPorts;
    consumerPorts.reserve(connection.consumerPorts.size());
    for (const PortRef &port : connection.consumerPorts) {
      std::string text = "{";
      text += canonicalPortRefString(port);
      text += '}';
      consumerPorts.push_back(std::move(text));
    }
    llvm::sort(consumerPorts);
    out += "|consumerPorts=";
    out += joinStrings(consumerPorts, ",");
  }
  out += "|route=";
  llvm::sort(route);
  out += joinStrings(std::vector<std::string>(route.begin(), route.end()), ",");
  out += "|engines=";
  llvm::sort(engines);
  out += joinStrings(std::vector<std::string>(engines.begin(), engines.end()),
                     ",");
  out += "|producerMap=";
  out += connection.producerMap ? mapString(*connection.producerMap) : "<null>";
  out += "|consumerMaps=";
  out += joinStrings(consumerMaps, ",");
  out += "|transform=";
  out +=
      connection.transform ? transformString(*connection.transform) : "<null>";
  // A gather's declared semantics and axis are execution-affecting content, so
  // they join the id -- but only when present, so a connection that carries no
  // gather semantics keeps the byte-identical id it had before (task B6).
  if (connection.gatherSemantics) {
    out += "|gather=";
    out += stringifyGatherSemantics(*connection.gatherSemantics);
    if (connection.concatAxis) {
      out += ":axis=";
      out += std::to_string(*connection.concatAxis);
    }
  }
  out += "|cost=";
  out += canonicalCostString(connection.cost);
  return out;
}

std::string canonicalPlanString(const CoveringPlan &plan) {
  std::vector<std::string> placements;
  placements.reserve(plan.placements.size());
  for (const PlanPlacement &placement : plan.placements) {
    std::string text = std::to_string(placement.node);
    text += '=';
    text += std::to_string(placement.instance);
    text += ':';
    text += placement.rule;
    text += ':';
    text += bundleString(placement.bundle);
    text += ':';
    text += placement.executor;
    text += ":mem=";
    text += joinStrings(sortedEntries(placement.memories), ",");
    if (!placement.portMemoryBindings.empty()) {
      text += ":portmem=";
      text += portMemoryBindingsString(placement.portMemoryBindings);
    }
    text += ":layout=";
    text += joinStrings(sortedEntries(placement.layouts), ",");
    text += ":solvedlayout=";
    text += solvedLayoutsString(placement.layoutSolutions);
    // The concrete compute node each requirement selected joins the identity
    // (issue #129, task R1), so two placements that differ only in which
    // attached engine of a kind ran are distinct plans rather than colliding on
    // one id and collapsing to the machine's first engine. Length-delimited via
    // `canonicalComputeBindingsString`, and empty for a rule that requires no
    // compute capability.
    text += ":compute=";
    text += canonicalComputeBindingsString(placement.computeBindings);
    placements.push_back(std::move(text));
  }
  llvm::sort(placements);

  std::vector<std::string> connectionPlans;
  connectionPlans.reserve(plan.connectionPlans.size());
  for (const PlanConnection &connection : plan.connectionPlans) {
    std::string text = std::to_string(connection.id);
    text += ':';
    text += stringifyConnectionKind(connection.kind).str();
    text += ":route=";
    llvm::SmallVector<MemoryNodeId> route(connection.route);
    llvm::sort(route);
    text +=
        joinStrings(std::vector<std::string>(route.begin(), route.end()), ",");
    text += ":engines=";
    llvm::SmallVector<ExecutorId> engines(connection.engines);
    llvm::sort(engines);
    text += joinStrings(
        std::vector<std::string>(engines.begin(), engines.end()), ",");
    // A gather's semantics/axis and its (order-bearing) producer occurrences
    // are execution-affecting, so they join the plan id too -- gated on
    // presence, so a plan with no gather connection keeps its id unchanged
    // (task B6).
    if (connection.gatherSemantics) {
      text += ":gather=";
      text += stringifyGatherSemantics(*connection.gatherSemantics);
      if (connection.concatAxis)
        text += ":axis=" + std::to_string(*connection.concatAxis);
    }
    if (!connection.producerPorts.empty()) {
      std::vector<std::string> producerPorts;
      producerPorts.reserve(connection.producerPorts.size());
      for (const PortRef &port : connection.producerPorts)
        producerPorts.push_back(canonicalPortRefString(port));
      text += ":producerPorts=" + joinStrings(producerPorts, ",");
    }
    // The physical movement hops join the identity (issue #129, task R4). The
    // engine a hop runs on and the storage slot it reads and writes are
    // *decisions*, so two plans that differ only in which buffer a hop uses are
    // distinct plans rather than colliding on one id. Gated on presence, so a
    // plan built without storage planning keeps the id it had. Order-bearing:
    // the hop sequence is the route, so the hops are rendered in route order,
    // never sorted. The steps that order the hops are derived timing and stay
    // out, as do the allocation intervals and every diagnostic.
    if (!connection.hops.empty()) {
      std::vector<std::string> hopTexts;
      hopTexts.reserve(connection.hops.size());
      for (const PlanMovementHop &hop : connection.hops) {
        // Length-delimited strings, so a memory or engine id containing the
        // punctuation below cannot be confused with a different hop.
        auto field = [](llvm::StringRef value) {
          return std::to_string(value.size()) + ":" + value.str();
        };
        std::string hopText = std::to_string(hop.index);
        hopText += ':';
        hopText += field(hop.srcMemory);
        hopText += '>';
        hopText += field(hop.dstMemory);
        hopText += ':';
        hopText += field(hop.engine);
        hopText += ':';
        hopText += std::to_string(hop.sourceStorageId);
        hopText += '>';
        hopText += std::to_string(hop.destinationStorageId);
        hopTexts.push_back(std::move(hopText));
      }
      text += ":hops=" + joinStrings(hopTexts, ",");
    }
    connectionPlans.push_back(std::move(text));
  }
  llvm::sort(connectionPlans);

  llvm::SmallVector<InstanceId> instances(plan.instances);
  llvm::SmallVector<ConnectionId> connections(plan.connections);

  // The `v3` tag makes the identity explicitly versioned (issue #129, task R1).
  // The derived ranking score (`totalCost`/`accumulatedCost`) and the
  // diagnostic/truncation fields are deliberately absent: they are search
  // outcomes and provenance, not decisions, and folding them in made a plan
  // unable to acquire an id before it was scored. A plan's id therefore depends
  // only on what it decided -- binding, selections, placements, routes,
  // parameters and, once storage planning has run, the physical movement hops
  // (their memories, engines and storage slots) that R4 added. The derived
  // timing around those hops -- the plan-step DAG, the allocation intervals --
  // stays out, so `finalizeStoragePlan` assigns the id exactly once, from the
  // decisions it built, and re-finalizing reproduces it.
  std::string out = "v3|binding=";
  out += hexId(plan.sourceBindingHash);
  out += "|instances=";
  out += joinNumbers(instances);
  out += "|connections=";
  out += joinNumbers(connections);
  out += "|placements=";
  out += joinStrings(placements, ";");
  out += "|routes=";
  out += joinStrings(connectionPlans, ";");
  out += "|params=";
  out += canonicalSearchValueString(plan.globalParameters);
  return out;
}

CandidateId computeCandidateId(const MappingCandidate &candidate) {
  return stableHash(canonicalCandidateString(candidate));
}

InstanceId computeInstanceId(const CandidateInstance &instance) {
  return stableHash(canonicalInstanceString(instance));
}

ConnectionId computeConnectionId(const ConnectionPlan &connection) {
  return stableHash(canonicalConnectionString(connection));
}

PlanId computePlanId(const CoveringPlan &plan) {
  return stableHash(canonicalPlanString(plan));
}

llvm::StringRef stringifyPlanScoreSource(PlanScoreSource source) {
  switch (source) {
  case PlanScoreSource::Accumulation:
    return "accumulation";
  case PlanScoreSource::Schedule:
    return "schedule";
  }
  return "accumulation";
}

std::string transformExecutorFor(const CoveringPlan &plan,
                                 const PlanConnection &connection) {
  for (const PlanPlacement &placement : plan.placements)
    for (InstanceId consumer : connection.consumers)
      if (placement.instance == consumer)
        return placement.executor;
  if (connection.producerPort)
    for (const PlanPlacement &placement : plan.placements)
      if (placement.node == connection.producerPort->node)
        return placement.executor;
  return {};
}

} // namespace mlir::llk::mapping
