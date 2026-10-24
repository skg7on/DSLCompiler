//===- mapped_codegen_requirements.cpp - Host codegen requirement tests ---===//
//
// Part of the LLK Compiler Project, under the Apache License v2.0 with
// LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "LLK/Mapping/CodegenRequirements.h"
#include "LLK/Runtime/MappedObject.h"

#include "llvm/ADT/StringMap.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/Error.h"

#include "gtest/gtest.h"

#include <string>

using mlir::llk::mapping::TargetCodegenRequirements;

namespace {
TargetCodegenRequirements avx2Requirements() {
  return {"x86_64", "haswell", {"avx2"}};
}
} // namespace

TEST(MappedCodegenRequirements, RejectsWrongArchitecture) {
  llvm::StringMap<bool> features;
  features["avx2"] = true;
  llvm::Error error =
      checkHostRequirements(avx2Requirements(), "aarch64", features);
  ASSERT_TRUE(static_cast<bool>(error));
  EXPECT_NE(llvm::toString(std::move(error)).find("x86_64"), std::string::npos);
}

TEST(MappedCodegenRequirements, RejectsMissingRequiredAVX2) {
  llvm::StringMap<bool> features;
  features["avx2"] = false;
  llvm::Error error =
      checkHostRequirements(avx2Requirements(), "x86_64", features);
  ASSERT_TRUE(static_cast<bool>(error));
  EXPECT_NE(llvm::toString(std::move(error)).find("avx2"), std::string::npos);
}

TEST(MappedCodegenRequirements, AcceptsRequiredAVX2) {
  llvm::StringMap<bool> features;
  features["avx2"] = true;
  EXPECT_FALSE(static_cast<bool>(
      checkHostRequirements(avx2Requirements(), "x86_64", features)));
}

TEST(MappedCodegenRequirements, ChecksOptionalFmaOnlyWhenRequired) {
  llvm::StringMap<bool> features;
  features["avx2"] = true;
  TargetCodegenRequirements avx2AndFma = {"x86_64", "haswell", {"avx2", "fma"}};
  llvm::Error rejected = checkHostRequirements(avx2AndFma, "x86_64", features);
  ASSERT_TRUE(static_cast<bool>(rejected));
  llvm::consumeError(std::move(rejected));

  features["fma"] = true;
  EXPECT_FALSE(
      static_cast<bool>(checkHostRequirements(avx2AndFma, "x86_64", features)));
}

TEST(MappedCodegenRequirements, AppliesInvocationAttributesToTheWrapper) {
  llvm::LLVMContext context;
  llvm::Module module("m", context);
  auto *wrapper = llvm::Function::Create(
      llvm::FunctionType::get(llvm::Type::getVoidTy(context), false),
      llvm::GlobalValue::ExternalLinkage, "_mlir_ciface_k", module);
  ASSERT_FALSE(wrapper->hasFnAttribute("aarch64_pstate_sm_enabled"));

  llvm::Error error = llk::applyInvocationAttributes(
      module, "_mlir_ciface_k", {"aarch64_pstate_sm_enabled"});
  ASSERT_FALSE(static_cast<bool>(error));
  EXPECT_TRUE(wrapper->hasFnAttribute("aarch64_pstate_sm_enabled"));

  // An empty attribute list is a no-op even when the wrapper is absent.
  EXPECT_FALSE(static_cast<bool>(
      llk::applyInvocationAttributes(module, "_mlir_ciface_missing", {})));

  // Naming a missing wrapper with attributes is an error, not a silent skip.
  llvm::Error missing = llk::applyInvocationAttributes(
      module, "_mlir_ciface_missing", {"aarch64_pstate_sm_enabled"});
  ASSERT_TRUE(static_cast<bool>(missing));
  EXPECT_NE(llvm::toString(std::move(missing)).find("_mlir_ciface_missing"),
            std::string::npos);
}
