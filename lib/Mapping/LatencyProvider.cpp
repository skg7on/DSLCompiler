//===- LatencyProvider.cpp - Optional measured latencies ------------------===//

#include "LLK/Mapping/LatencyProvider.h"

namespace mlir::llk::mapping {

std::string OperationSignature::canonicalString() const {
  std::string out = "operation=";
  out += operation;
  out += "|operand_types=";
  out += operandTypes;
  out += "|result_types=";
  out += resultTypes;
  out += "|attributes=";
  out += attributes;
  out += "|rule=";
  out += rule;
  out += "|rule_version=";
  out += std::to_string(ruleVersion);
  out += "|bundle=";
  out += bundle;
  out += "|bundle_parameters=";
  out += bundleParameters;
  out += "|layout=";
  out += layout;
  out += "|placement_class=";
  out += placementClass;
  out += "|placement=";
  out += placement;
  out += "|route=";
  out += routeClass;
  out += "|cost_model=";
  out += std::to_string(costModelVersion);
  return out;
}

} // namespace mlir::llk::mapping
