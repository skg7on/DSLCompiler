//===- AVX2MappingTarget.cpp - AVX2 mapping package (D7) ------------------===//

#include "LLK/Target/X86/Mapping/AVX2MappingTarget.h"

#include <string>
#include <utility>
#include <vector>

namespace mlir::llk::target::avx2 {

namespace {

/// The AVX2 target wraps the loaded configuration and supplies the plugin's
/// own emitters.
///
/// The configuration loader knows a bundle's *shape* but nothing about AVX2
/// code, so its default emitters reject lowering. Overriding `createEmitter`
/// here is what makes this target executable: the emitters it hands out lower
/// AVX2 bundles, and no generic code learns that they exist.
class AVX2Target : public mapping::MappingTarget {
public:
  explicit AVX2Target(std::unique_ptr<mapping::MappingTarget> configuration)
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
  const mapping::LatencyProvider *latencyProvider() const override {
    return configuration_->latencyProvider();
  }

  std::unique_ptr<mapping::TargetEmitter>
  createEmitter(llvm::StringRef key) const override {
    if (!configuration_->isKnownEmitter(key))
      return nullptr;
    return createAVX2Emitter(key);
  }

  std::unique_ptr<mapping::TargetEmitter> createEmitter() const override {
    llvm::ArrayRef<llvm::StringLiteral> keys = emitterKeys();
    if (keys.empty())
      return nullptr;
    return createEmitter(keys.front());
  }

private:
  std::unique_ptr<mapping::MappingTarget> configuration_;
};

} // namespace

llvm::ArrayRef<llvm::StringLiteral> emitterKeys() {
  static const llvm::StringLiteral kKeys[] = {
      "avx2_vector_add",  "avx2_vector_convert",
      "avx2_vector_silu", "avx2_vector_mul",
      "avx2_mma",         "avx2_reduce",
      "avx2_copy",        "avx2_tile_copy",
      "avx2_tile_store",  "avx2_fused_convert_silu_mul"};
  return kKeys;
}

llvm::Expected<std::unique_ptr<mapping::MappingTarget>>
createMappingTarget(llvm::StringRef configurationRoot) {
  std::string root = configurationRoot.str();
  std::vector<std::string> keys;
  for (llvm::StringLiteral key : emitterKeys())
    keys.push_back(key.str());
  llvm::Expected<std::unique_ptr<mapping::MappingTarget>> configuration =
      mapping::loadMappingTarget(
          "x86-avx2", root + "/machines/x86-avx2-v2.yaml",
          root + "/mapping/x86-avx2/layouts.llkmap",
          root + "/mapping/x86-avx2/rules.llkmap", std::move(keys));
  if (!configuration)
    return configuration.takeError();
  return std::make_unique<AVX2Target>(std::move(*configuration));
}

} // namespace mlir::llk::target::avx2
