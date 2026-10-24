//===- arm_targets.cpp - Static Arm target packages -----------------------===//
//
// Part of the LLK Compiler Project, under the Apache License v2.0 with
// LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "arm_target_fixture.h"

#include "LLK/Mapping/MappingLowering.h"
#include "LLK/Mapping/MappingTarget.h"
#include "LLK/Target/AArch64/Mapping/SMEMappingTarget.h"
#include "LLK/Target/AArch64/Mapping/SVEMappingTarget.h"
#include "LLK/Target/MappingTargets.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/PassManager.h"

#include "llvm/Support/FileUtilities.h"
#include "llvm/Support/MemoryBuffer.h"

#include <gtest/gtest.h>

#include <set>
#include <string>

using namespace mlir::llk::mapping;

#ifndef LLK_SOURCE_DIR
#error "LLK_SOURCE_DIR must name the repository root"
#endif

#ifndef LLK_TEST_BINARY_DIR
#error "LLK_TEST_BINARY_DIR must name this test category's build directory"
#endif

namespace {

struct ArmConfiguration {
  llvm::StringRef name;
  llvm::StringRef prefix;
  llvm::StringRef accumulator;
  uint64_t capacity;
};

const ArmConfiguration kConfigurations[] = {
    {"arm-sve", "sve", "acc.0", 2048},
    {"arm-sme", "sme", "za.0", 4096},
};

llvm::Expected<std::unique_ptr<MappingTarget>>
loadRegistered(const ArmConfiguration &configuration,
               llvm::StringRef root = LLK_SOURCE_DIR,
               llvm::StringRef machine = {}) {
  mlir::llk::target::registerAllMappingTargets();
  return createRegisteredMappingTarget(configuration.name, root, machine);
}

} // namespace

// Catches a missing registration, swapped factories, or incorrect file paths.
TEST(ArmTargets, RegistryLoadsBoth) {
  for (const ArmConfiguration &configuration : kConfigurations) {
    SCOPED_TRACE(configuration.name.str());
    auto target = loadRegistered(configuration);
    ASSERT_TRUE(bool(target)) << llvm::toString(target.takeError());
    EXPECT_EQ((*target)->name(), configuration.name);
    EXPECT_EQ((*target)->machine().target, configuration.name);
    const auto *accumulator =
        (*target)->machine().findMemory(configuration.accumulator);
    ASSERT_NE(accumulator, nullptr);
    EXPECT_EQ(accumulator->capacityBytes, configuration.capacity);
    EXPECT_NE(
        (*target)->layouts().find((configuration.prefix + ".blocked_2d").str()),
        nullptr);
    EXPECT_NE(
        (*target)->rules().find((configuration.prefix + ".mma_f32").str()),
        nullptr);
  }
}

// Catches a factory ignoring its override and silently using the default model.
TEST(ArmTargets, RegistryUsesMachineOverride) {
  for (const ArmConfiguration &configuration : kConfigurations) {
    SCOPED_TRACE(configuration.name.str());
    auto original = loadRegistered(configuration);
    ASSERT_TRUE(bool(original)) << llvm::toString(original.takeError());
    auto buffer = llvm::MemoryBuffer::getFile(
        (LLK_SOURCE_DIR "/machines/" + configuration.name + "-v2.yaml").str());
    ASSERT_TRUE(bool(buffer)) << buffer.getError().message();
    std::string text = (*buffer)->getBuffer().str();
    std::string capacity =
        "capacity_bytes: " + std::to_string(configuration.capacity);
    size_t offset = text.find(capacity);
    ASSERT_NE(offset, std::string::npos);
    text.replace(offset, capacity.size(),
                 "capacity_bytes: " +
                     std::to_string(configuration.capacity / 2));
    auto path = arm_test::writeMachineVariant(text);
    ASSERT_TRUE(bool(path)) << llvm::toString(path.takeError());
    llvm::FileRemover cleanup(*path);
    auto overridden = loadRegistered(configuration, LLK_SOURCE_DIR, *path);
    ASSERT_TRUE(bool(overridden)) << llvm::toString(overridden.takeError());
    const auto *accumulator =
        (*overridden)->machine().findMemory(configuration.accumulator);
    ASSERT_NE(accumulator, nullptr);
    EXPECT_EQ(accumulator->capacityBytes, configuration.capacity / 2);
    EXPECT_NE((*overridden)->machine().contentHash,
              (*original)->machine().contentHash);
  }
}

// Catches fallback to the default model after an override read error.
TEST(ArmTargets, MissingMachineOverrideReturnsError) {
  for (const ArmConfiguration &configuration : kConfigurations) {
    auto target =
        loadRegistered(configuration, LLK_SOURCE_DIR,
                       LLK_TEST_BINARY_DIR "/missing-arm-machine.yaml");
    ASSERT_FALSE(bool(target));
    EXPECT_NE(
        llvm::toString(target.takeError()).find("missing-arm-machine.yaml"),
        std::string::npos);
  }
}

// Catches bypassing configurationRoot or hiding a failed policy/profile load.
TEST(ArmTargets, MissingConfigurationReturnsError) {
  for (const ArmConfiguration &configuration : kConfigurations) {
    auto target = loadRegistered(configuration,
                                 LLK_TEST_BINARY_DIR "/missing-arm-config");
    ASSERT_FALSE(bool(target));
    EXPECT_NE(llvm::toString(target.takeError()).find("missing-arm-config"),
              std::string::npos);
  }
}

// Catches lost rule emitter keys and failure to accept parameterless LLKMap
// bundles (their parameters are a null attribute rather than an empty dict).
TEST(ArmTargets, EveryRuleBundleHasAVerifyingEmitter) {
  mlir::MLIRContext context;
  for (const ArmConfiguration &configuration : kConfigurations) {
    auto target = loadRegistered(configuration);
    ASSERT_TRUE(bool(target)) << llvm::toString(target.takeError());
    for (const RuleDef &rule : (*target)->rules().all()) {
      SCOPED_TRACE(rule.id);
      auto emitter = (*target)->createEmitter(rule.emitter);
      ASSERT_NE(emitter, nullptr);
      EXPECT_EQ(emitter->key(), rule.emitter);
      TargetBundle bundle{rule.bundle, {}, rule.emitter};
      EXPECT_EQ(llvm::toString(emitter->verify(bundle)), "");
      bundle.parameters = mlir::DictionaryAttr::get(
          &context,
          {mlir::NamedAttribute(mlir::StringAttr::get(&context, "width"),
                                mlir::IntegerAttr::get(
                                    mlir::IntegerType::get(&context, 64), 16)),
           mlir::NamedAttribute(mlir::StringAttr::get(&context, "variant"),
                                mlir::StringAttr::get(&context, "modeled"))});
      EXPECT_EQ(llvm::toString(emitter->verify(bundle)), "");
    }
  }
}

TEST(ArmTargets, DeclaredKeysMatchPolicyAndHaveEmitters) {
  const std::set<std::string> expectedSve = {
      "sve_vector_add", "sve_vector_convert", "sve_vector_silu",
      "sve_vector_mul", "sve_reduce",         "sve_copy",
      "sve_tile_copy",  "sve_tile_store",     "sve_fused_convert_silu_mul",
      "sve_mma"};
  const std::set<std::string> expectedSme = {
      "sme_vector_add", "sme_vector_convert", "sme_vector_silu",
      "sme_vector_mul", "sme_reduce",         "sme_copy",
      "sme_tile_copy",  "sme_tile_store",     "sme_fused_convert_silu_mul",
      "sme_mma_za"};
  for (const ArmConfiguration &configuration : kConfigurations) {
    auto target = loadRegistered(configuration);
    ASSERT_TRUE(bool(target)) << llvm::toString(target.takeError());
    auto keys = configuration.name == "arm-sve"
                    ? mlir::llk::target::sve::emitterKeys()
                    : mlir::llk::target::sme::emitterKeys();
    const auto &expected =
        configuration.name == "arm-sve" ? expectedSve : expectedSme;
    std::set<std::string> declared;
    for (llvm::StringRef key : keys) {
      declared.insert(key.str());
      EXPECT_TRUE((*target)->isKnownEmitter(key));
      EXPECT_NE((*target)->createEmitter(key), nullptr);
    }
    EXPECT_EQ(keys.size(), expected.size());
    EXPECT_EQ(declared, expected);
    std::set<std::string> ruleKeys;
    for (const RuleDef &rule : (*target)->rules().all())
      ruleKeys.insert(rule.emitter);
    EXPECT_EQ(ruleKeys, expected);
  }
}

// Catches accepting an unknown or different emitter key, or malformed attrs.
TEST(ArmTargets, RejectsUnknownCrossKeyAndMalformedBundles) {
  mlir::MLIRContext context;
  for (const ArmConfiguration &configuration : kConfigurations) {
    auto target = loadRegistered(configuration);
    ASSERT_TRUE(bool(target)) << llvm::toString(target.takeError());
    EXPECT_EQ((*target)->createEmitter("unknown_arm_emitter"), nullptr);
    EXPECT_EQ((*target)->createEmitter("avx2_vector_add"), nullptr);
    auto emitter =
        (*target)->createEmitter((configuration.prefix + "_vector_add").str());
    ASSERT_NE(emitter, nullptr);
    TargetBundle bundle{"opaque", {}, "unknown_arm_emitter"};
    EXPECT_NE(llvm::toString(emitter->verify(bundle)).find("does not declare"),
              std::string::npos);
    bundle.emitterKey = (configuration.prefix + "_vector_mul").str();
    EXPECT_NE(llvm::toString(emitter->verify(bundle)).find("handles"),
              std::string::npos);
    bundle.emitterKey = emitter->key().str();
    bundle.parameters = mlir::DictionaryAttr::get(
        &context,
        {mlir::NamedAttribute(mlir::StringAttr::get(&context, "width"),
                              mlir::UnitAttr::get(&context))});
    EXPECT_NE(llvm::toString(emitter->verify(bundle)).find("width"),
              std::string::npos);
  }
}

// Catches a package inadvertently gaining native lowering or ISA requirements.
TEST(ArmTargets, NoExecutableBackend) {
  mlir::MLIRContext context;
  mlir::IRRewriter rewriter(&context);
  mlir::OpPassManager pipeline;
  for (const ArmConfiguration &configuration : kConfigurations) {
    auto target = loadRegistered(configuration);
    ASSERT_TRUE(bool(target)) << llvm::toString(target.takeError());
    EXPECT_FALSE((*target)->codegenRequirements().has_value());
    EXPECT_NE(llvm::toString((*target)->buildSelectedBackendPipeline(pipeline))
                  .find("unsupported"),
              std::string::npos);
    TargetLoweringContext lowering{(*target)->machine(), {}, {}};
    for (const RuleDef &rule : (*target)->rules().all()) {
      auto emitter = (*target)->createEmitter(rule.emitter);
      ASSERT_NE(emitter, nullptr);
      EXPECT_FALSE(emitter->hasLowering());
      TargetBundle bundle{rule.bundle, {}, rule.emitter};
      EXPECT_NE(llvm::toString(emitter->lower({}, bundle, lowering, rewriter))
                    .find("target_lowering_unsupported"),
                std::string::npos);
    }
  }
}
