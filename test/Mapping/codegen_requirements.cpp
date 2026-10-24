//===- codegen_requirements.cpp - Host codegen requirement tests ----------===//
//
// Part of the LLK Compiler Project, under the Apache License v2.0 with
// LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "LLK/Mapping/CodegenRequirements.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/Error.h"
#include <gtest/gtest.h>

using mlir::llk::mapping::checkHostRequirements;
using mlir::llk::mapping::TargetCodegenRequirements;

TEST(CodegenRequirements, AcceptsMatchingVectorLength) {
  TargetCodegenRequirements requirements{"aarch64", "", {"sve"}, 512};
  llvm::StringMap<bool> features{{"sve", true}};
  EXPECT_FALSE(static_cast<bool>(
      checkHostRequirements(requirements, "aarch64", features, 512)));
}

TEST(CodegenRequirements, RejectsMissingOrDifferentVectorLength) {
  TargetCodegenRequirements requirements{"aarch64", "", {"sve"}, 512};
  llvm::StringMap<bool> features{{"sve", true}};
  EXPECT_TRUE(static_cast<bool>(
      checkHostRequirements(requirements, "aarch64", features, std::nullopt)));
  EXPECT_TRUE(static_cast<bool>(
      checkHostRequirements(requirements, "aarch64", features, 256)));
  features["sve"] = false;
  EXPECT_TRUE(static_cast<bool>(
      checkHostRequirements(requirements, "aarch64", features, 512)));
}
