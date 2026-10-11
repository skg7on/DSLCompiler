//===- LatencyProvider.cpp - Optional measured latencies ------------------===//

#include "LLK/Mapping/LatencyProvider.h"

#include <string>

namespace mlir::llk::mapping {

namespace {

/// Appends one field length-delimited. `out` already holds a `name` label, so
/// the caller names the field first and this writes `<len>:<bytes>`. A field
/// whose bytes contain the separator, a `|`, or a newline therefore cannot be
/// confused with a field boundary: the reader counts bytes, never scans for a
/// delimiter (task B8).
void appendField(std::string &out, llvm::StringRef name,
                 llvm::StringRef value) {
  out += name.str();
  out += '=';
  out += std::to_string(value.size());
  out += ':';
  out.append(value.begin(), value.end());
  out += ';';
}

} // namespace

std::string OperationSignature::canonicalString() const {
  std::string out;
  appendField(out, "operation", operation);
  appendField(out, "operand_types", operandTypes);
  appendField(out, "result_types", resultTypes);
  appendField(out, "attributes", attributes);
  appendField(out, "rule", rule);
  appendField(out, "rule_version", std::to_string(ruleVersion));
  appendField(out, "bundle", bundle);
  appendField(out, "bundle_parameters", bundleParameters);
  appendField(out, "layout", layout);
  appendField(out, "placement_class", placementClass);
  appendField(out, "placement", placement);
  appendField(out, "compute", compute);
  appendField(out, "route", routeClass);
  appendField(out, "cost_model", std::to_string(costModelVersion));
  return out;
}

std::string ConnectionSignature::canonicalString() const {
  std::string out;
  appendField(out, "kind", kind);
  appendField(out, "value_type", valueType);
  appendField(out, "producer", producerEndpoint);
  appendField(out, "consumers", consumerEndpoints);
  appendField(out, "route", route);
  appendField(out, "links", links);
  appendField(out, "engines", engines);
  appendField(out, "maps", maps);
  appendField(out, "parameters", parameters);
  appendField(out, "storage", storage);
  appendField(out, "key_version", std::to_string(keyVersion));
  return out;
}

} // namespace mlir::llk::mapping
