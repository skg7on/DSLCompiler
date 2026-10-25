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

#include <string>

using mlir::llk::mapping::checkHostRequirements;
using mlir::llk::mapping::TargetCodegenRequirements;
using mlir::llk::mapping::vectorLengthBitsFromPrctl;

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

TEST(CodegenRequirements, IgnoresHostVectorLengthWithoutRequirement) {
  TargetCodegenRequirements requirements{"aarch64", "", {"sve"}};
  llvm::StringMap<bool> features{{"sve", true}};
  // A target that names no vector length makes no claim about the host's.
  EXPECT_FALSE(static_cast<bool>(
      checkHostRequirements(requirements, "aarch64", features, 256)));
  EXPECT_FALSE(static_cast<bool>(
      checkHostRequirements(requirements, "aarch64", features, std::nullopt)));
}

TEST(CodegenRequirements, ReportsVectorLengthRefusals) {
  TargetCodegenRequirements requirements{"aarch64", "", {"sve"}, 512};
  llvm::StringMap<bool> features{{"sve", true}};

  llvm::Error unknown =
      checkHostRequirements(requirements, "aarch64", features, std::nullopt);
  ASSERT_TRUE(static_cast<bool>(unknown));
  std::string unknownText = llvm::toString(std::move(unknown));
  EXPECT_NE(unknownText.find("host vector length is unknown"),
            std::string::npos)
      << unknownText;
  EXPECT_NE(unknownText.find("requires 512 bits"), std::string::npos)
      << unknownText;

  llvm::Error mismatch =
      checkHostRequirements(requirements, "aarch64", features, 256);
  ASSERT_TRUE(static_cast<bool>(mismatch));
  std::string mismatchText = llvm::toString(std::move(mismatch));
  EXPECT_NE(mismatchText.find("host vector length is 256 bits"),
            std::string::npos)
      << mismatchText;
  EXPECT_NE(mismatchText.find("requires 512 bits"), std::string::npos)
      << mismatchText;
}

TEST(CodegenRequirements, ChecksStreamingVectorLengthIndependently) {
  // SME has no non-streaming vector-length requirement; its SVL is separate.
  TargetCodegenRequirements requirements{
      "aarch64", "", {"sme"}, std::nullopt, 512};
  llvm::StringMap<bool> features{{"sme", true}};
  EXPECT_FALSE(static_cast<bool>(checkHostRequirements(
      requirements, "aarch64", features, std::nullopt, 512)));
  // A non-streaming length of 128 is irrelevant to an SME-only requirement.
  EXPECT_FALSE(static_cast<bool>(
      checkHostRequirements(requirements, "aarch64", features, 128, 512)));
  EXPECT_TRUE(static_cast<bool>(checkHostRequirements(
      requirements, "aarch64", features, std::nullopt, std::nullopt)));
  EXPECT_TRUE(static_cast<bool>(checkHostRequirements(
      requirements, "aarch64", features, std::nullopt, 256)));
}

TEST(CodegenRequirements, ReportsStreamingVectorLengthRefusals) {
  TargetCodegenRequirements requirements{
      "aarch64", "", {"sme"}, std::nullopt, 512};
  llvm::StringMap<bool> features{{"sme", true}};

  llvm::Error unknown = checkHostRequirements(requirements, "aarch64", features,
                                              std::nullopt, std::nullopt);
  ASSERT_TRUE(static_cast<bool>(unknown));
  std::string unknownText = llvm::toString(std::move(unknown));
  EXPECT_NE(unknownText.find("host streaming vector length is unknown"),
            std::string::npos)
      << unknownText;
  EXPECT_NE(unknownText.find("requires 512 bits"), std::string::npos)
      << unknownText;

  llvm::Error mismatch = checkHostRequirements(requirements, "aarch64",
                                               features, std::nullopt, 256);
  ASSERT_TRUE(static_cast<bool>(mismatch));
  std::string mismatchText = llvm::toString(std::move(mismatch));
  EXPECT_NE(mismatchText.find("host streaming vector length is 256 bits"),
            std::string::npos)
      << mismatchText;
}

TEST(CodegenRequirements, KeepsInvocationAttributesAsTargetPolicy) {
  TargetCodegenRequirements requirements{
      "aarch64", "", {"sme"}, std::nullopt, 512, {"aarch64_pstate_sm_enabled"}};
  EXPECT_EQ(requirements.streamingVectorLengthBits, 512);
  EXPECT_EQ(requirements.invocationAttributes,
            std::vector<std::string>{"aarch64_pstate_sm_enabled"});
}

TEST(CodegenRequirements, ConvertsSvePrctlLengthFromBytesToBits) {
  // PR_SVE_GET_VL reports bytes: 64 bytes is the modeled 512-bit length and
  // 32 bytes a 256-bit one.
  EXPECT_EQ(vectorLengthBitsFromPrctl(64), 512);
  EXPECT_EQ(vectorLengthBitsFromPrctl(32), 256);
  // Only the low 16 bits carry the length; higher bits are status flags.
  EXPECT_EQ(vectorLengthBitsFromPrctl((512L << 16) | 64), 512);
}
