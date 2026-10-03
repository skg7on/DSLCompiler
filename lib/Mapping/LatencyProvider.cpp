//===- LatencyProvider.cpp - Optional measured latencies ------------------===//

#include "LLK/Mapping/LatencyProvider.h"

namespace mlir::llk::mapping {

std::string OperationSignature::canonicalString() const {
  std::string out = "operation=";
  out += operation;
  out += "|rule=";
  out += rule;
  out += "|rule_version=";
  out += std::to_string(ruleVersion);
  out += "|bundle=";
  out += bundle;
  out += "|layout=";
  out += layout;
  out += "|placement=";
  out += placementClass;
  out += "|route=";
  out += routeClass;
  out += "|cost_model=";
  out += std::to_string(costModelVersion);
  return out;
}

} // namespace mlir::llk::mapping
