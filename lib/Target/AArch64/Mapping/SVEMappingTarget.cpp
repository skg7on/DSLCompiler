//===- SVEMappingTarget.cpp - Static SVE mapping target -------------------===//
//
// Part of the LLK Compiler Project, under the Apache License v2.0 with
// LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "LLK/Target/AArch64/Mapping/SVEMappingTarget.h"

#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace mlir::llk::target::sve {

namespace {

/// The SVE target wraps the loaded configuration and supplies the plugin's own
/// emitters plus the executable subset's code-generation contract.
///
/// The configuration loader knows a bundle's *shape* but nothing about SVE
/// code, so its default emitters reject lowering. Overriding `createEmitter`
/// here is what makes the arithmetic subset executable: the two emitters it
/// hands out for `sve_vector_add`/`sve_vector_mul` lower their bundles, and
/// every other key keeps the configuration-only rejection. No generic code
/// learns which keys those are.
class SVETarget : public mapping::MappingTarget {
public:
  explicit SVETarget(std::unique_ptr<mapping::MappingTarget> configuration)
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
  bool isKnownEmitter(llvm::StringRef key) const override {
    return configuration_->isKnownEmitter(key);
  }
  std::optional<mapping::TargetCodegenRequirements>
  codegenRequirements() const override {
    return mapping::TargetCodegenRequirements{"aarch64", "", {"sve"}};
  }
  llvm::Error
  buildSelectedBackendPipeline(mlir::OpPassManager &pm) const override {
    const machine::ComputeNode *engine = machine().findCompute("svp");
    if (!engine)
      return llvm::createStringError(
          llvm::inconvertibleErrorCode(),
          "arm-sve selected backend has no vector engine 'svp'");
    auto lanes = engine->lanes.find("f32");
    if (lanes == engine->lanes.end())
      return llvm::createStringError(
          llvm::inconvertibleErrorCode(),
          "arm-sve selected backend has no f32 lane width");
    buildSVESelectedBackendPipeline(pm, lanes->second);
    return llvm::Error::success();
  }
  std::unique_ptr<mapping::TargetEmitter>
  createEmitter(llvm::StringRef key) const override {
    if (!configuration_->isKnownEmitter(key))
      return nullptr;
    return createSVEEmitter(key);
  }
  std::unique_ptr<mapping::TargetEmitter> createEmitter() const override {
    llvm::ArrayRef<llvm::StringRef> keys = emitterKeys();
    if (keys.empty())
      return nullptr;
    return createEmitter(keys.front());
  }

private:
  std::unique_ptr<mapping::MappingTarget> configuration_;
};

} // namespace

llvm::ArrayRef<llvm::StringRef> emitterKeys() {
  static const llvm::StringRef keys[] = {
      "sve_vector_add", "sve_vector_convert", "sve_vector_silu",
      "sve_vector_mul", "sve_reduce",         "sve_copy",
      "sve_tile_copy",  "sve_tile_store",     "sve_fused_convert_silu_mul",
      "sve_mma"};
  return keys;
}

llvm::Expected<std::unique_ptr<mapping::MappingTarget>>
createMappingTarget(llvm::StringRef configurationRoot,
                    llvm::StringRef machinePath) {
  std::string root = configurationRoot.str();
  std::string machine = machinePath.empty() ? root + "/machines/arm-sve-v2.yaml"
                                            : machinePath.str();
  std::vector<std::string> keys;
  for (llvm::StringRef key : emitterKeys())
    keys.push_back(key.str());
  llvm::Expected<std::unique_ptr<mapping::MappingTarget>> configuration =
      mapping::loadMappingTarget(
          "arm-sve", machine, root + "/mapping/arm-sve/layouts.llkmap",
          root + "/mapping/arm-sve/rules.llkmap", std::move(keys));
  if (!configuration)
    return configuration.takeError();
  return std::make_unique<SVETarget>(std::move(*configuration));
}

} // namespace mlir::llk::target::sve
