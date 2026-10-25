//===- SMEMappingTarget.cpp - Static SME mapping target -------------------===//
//
// Part of the LLK Compiler Project, under the Apache License v2.0 with
// LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "LLK/Target/AArch64/Mapping/SMEMappingTarget.h"

#include <string>
#include <utility>
#include <vector>

namespace mlir::llk::target::sme {

namespace {

/// Wraps the configuration target with the executable f32 contraction
/// subset, the streaming ABI contract, and both lowering hooks. The
/// configuration loader knows a bundle's shape but nothing about SME code.
class SMETarget : public mapping::MappingTarget {
public:
  explicit SMETarget(std::unique_ptr<mapping::MappingTarget> configuration)
      : configuration_(std::move(configuration)) {}

  llvm::StringRef name() const override { return configuration_->name(); }
  const machine::MachineModel &machine() const override {
    return configuration_->machine();
  }
  const mapping::LayoutRegistry &layouts() const override {
    return configuration_->layouts();
  }
  const mapping::RuleRegistry &rules() const override {
    return configuration_->rules();
  }
  const mapping::LatencyProvider *latencyProvider() const override {
    return configuration_->latencyProvider();
  }
  bool isKnownEmitter(llvm::StringRef key) const override {
    return configuration_->isKnownEmitter(key);
  }
  std::optional<mapping::TargetCodegenRequirements>
  codegenRequirements() const override {
    return mapping::TargetCodegenRequirements{
        "aarch64",
        "",
        {"sme"},
        /*vectorLengthBits=*/std::nullopt,
        /*streamingVectorLengthBits=*/512,
        /*invocationAttributes=*/{"aarch64_pstate_sm_enabled"}};
  }
  llvm::Error
  buildSelectedBackendPipeline(mlir::OpPassManager &pm) const override {
    buildSMESelectedBackendPipeline(pm);
    return llvm::Error::success();
  }
  llvm::Error buildPostBufferizationBackendPipeline(
      mlir::OpPassManager &pm) const override {
    buildSMEPostBufferizationPipeline(pm);
    return llvm::Error::success();
  }
  llvm::Error buildLateBackendPipeline(mlir::OpPassManager &pm) const override {
    buildSMELateBackendPipeline(pm);
    return llvm::Error::success();
  }
  std::unique_ptr<mapping::TargetEmitter>
  createEmitter(llvm::StringRef key) const override {
    if (!configuration_->isKnownEmitter(key))
      return nullptr;
    return createSMEEmitter(key);
  }
  std::unique_ptr<mapping::TargetEmitter> createEmitter() const override {
    llvm::ArrayRef<llvm::StringRef> keys = emitterKeys();
    return keys.empty() ? nullptr : createEmitter(keys.front());
  }

private:
  std::unique_ptr<mapping::MappingTarget> configuration_;
};

} // namespace

llvm::ArrayRef<llvm::StringRef> emitterKeys() {
  static const llvm::StringRef keys[] = {
      "sme_vector_add", "sme_vector_convert", "sme_vector_silu",
      "sme_vector_mul", "sme_reduce",         "sme_copy",
      "sme_tile_copy",  "sme_tile_store",     "sme_fused_convert_silu_mul",
      "sme_mma_za"};
  return keys;
}

llvm::Expected<std::unique_ptr<mapping::MappingTarget>>
createMappingTarget(llvm::StringRef configurationRoot,
                    llvm::StringRef machinePath) {
  std::string root = configurationRoot.str();
  std::string machine = machinePath.empty() ? root + "/machines/arm-sme-v2.yaml"
                                            : machinePath.str();
  std::vector<std::string> keys;
  for (llvm::StringRef key : emitterKeys())
    keys.push_back(key.str());
  llvm::Expected<std::unique_ptr<mapping::MappingTarget>> configuration =
      mapping::loadMappingTarget(
          "arm-sme", machine, root + "/mapping/arm-sme/layouts.llkmap",
          root + "/mapping/arm-sme/rules.llkmap", std::move(keys));
  if (!configuration)
    return configuration.takeError();
  return std::make_unique<SMETarget>(std::move(*configuration));
}

} // namespace mlir::llk::target::sme
