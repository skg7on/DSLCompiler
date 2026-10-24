//===- CodegenRequirements.cpp - Host codegen requirements ----------------===//
//
// Part of the LLK Compiler Project, under the Apache License v2.0 with
// LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "LLK/Mapping/CodegenRequirements.h"

#include <string>

namespace mlir::llk::mapping {

namespace {

llvm::Error requirementError(const std::string &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(), message);
}

} // namespace

llvm::Error checkHostRequirements(const TargetCodegenRequirements &requirements,
                                  llvm::StringRef hostArchitecture,
                                  const llvm::StringMap<bool> &hostFeatures,
                                  std::optional<int64_t> hostVectorLengthBits) {
  if (!requirements.architecture.empty() &&
      requirements.architecture != hostArchitecture)
    return requirementError(
        "selected target requires architecture '" + requirements.architecture +
        "' but host architecture is '" + hostArchitecture.str() + "'");
  for (const std::string &feature : requirements.requiredFeatures) {
    auto available = hostFeatures.find(feature);
    if (available == hostFeatures.end() || !available->second)
      return requirementError("selected target requires host feature '" +
                              feature + "'");
  }
  if (requirements.vectorLengthBits) {
    if (!hostVectorLengthBits)
      return requirementError("host vector length is unknown but '" +
                              requirements.architecture + "' requires " +
                              std::to_string(*requirements.vectorLengthBits) +
                              " bits");
    if (*hostVectorLengthBits != *requirements.vectorLengthBits)
      return requirementError(
          "host vector length is " + std::to_string(*hostVectorLengthBits) +
          " bits but the selected target requires " +
          std::to_string(*requirements.vectorLengthBits) + " bits");
  }
  return llvm::Error::success();
}

} // namespace mlir::llk::mapping
