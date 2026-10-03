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
  out += ':';
  out += mapString(transform.map);
  return out;
}

} // namespace

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

std::string canonicalCandidateString(const MappingCandidate &candidate) {
  std::vector<std::string> ports;
  ports.reserve(candidate.ports.size());
  for (const PortSpec &port : candidate.ports) {
    std::string text = port.name;
    text += ':';
    text += std::to_string(port.value);
    text += port.isInput ? ":in" : ":out";
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
  out += candidate.targetBundle;
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
  std::string out = "candidate=";
  out += std::to_string(instance.candidate);
  out += "|exec=";
  out += joinStrings(sortedEntries(instance.executorBindings), ",");
  out += "|mem=";
  out += joinStrings(sortedEntries(instance.memoryBindings), ",");
  out += "|layout=";
  out += joinStrings(sortedEntries(instance.layoutBindings), ",");
  out += "|compute=";
  out += joinStrings(sortedEntries(instance.computeBindings), ",");
  out += "|slots=";
  out += std::to_string(instance.resourceUsage.executorSlots);
  out += "|membytes=";
  out += joinStrings(sortedEntries(instance.resourceUsage.memoryBytes), ",");
  out += "|cost=";
  out += canonicalCostString(instance.localCost);
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
  out += "|cost=";
  out += canonicalCostString(connection.cost);
  return out;
}

std::string canonicalPlanString(const CoveringPlan &plan) {
  llvm::SmallVector<InstanceId> instances(plan.instances);
  llvm::SmallVector<ConnectionId> connections(plan.connections);
  std::vector<std::string> errors(plan.diagnostics.errors);
  std::vector<std::string> warnings(plan.diagnostics.warnings);
  llvm::sort(errors);
  llvm::sort(warnings);

  std::string out = "binding=";
  out += hexId(plan.sourceBindingHash);
  out += "|instances=";
  out += joinNumbers(instances);
  out += "|connections=";
  out += joinNumbers(connections);
  out += "|params=";
  out += canonicalSearchValueString(plan.globalParameters);
  out += "|cost=";
  out += canonicalCostString(plan.totalCost);
  out += "|errors=";
  out += joinStrings(errors, ";");
  out += "|warnings=";
  out += joinStrings(warnings, ";");
  out += "|truncated=";
  out += plan.diagnostics.searchTruncated ? "1" : "0";
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

} // namespace mlir::llk::mapping
