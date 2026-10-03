//===- GenericAcceleratorMappingTarget.cpp - second target (D7) -----------===//

#include "LLK/Target/GenericAccelerator/Mapping/GenericAcceleratorMappingTarget.h"

#include <string>
#include <vector>

namespace mlir::llk::target::generic_accel {

llvm::ArrayRef<llvm::StringRef> emitterKeys() {
  static const llvm::StringRef kKeys[] = {"accel_vector_add", "accel_mxu",
                                          "accel_copy"};
  return kKeys;
}

llvm::Expected<std::unique_ptr<mapping::MappingTarget>>
createMappingTarget(llvm::StringRef configurationRoot) {
  std::string root = configurationRoot.str();
  std::vector<std::string> keys;
  for (llvm::StringRef key : emitterKeys())
    keys.push_back(key.str());
  return mapping::loadMappingTarget(
      "generic-ai-accel", root + "/machines/generic-ai-accel-v2.yaml",
      root + "/mapping/generic-ai-accel/layouts.llkmap",
      root + "/mapping/generic-ai-accel/rules.llkmap", std::move(keys));
}

} // namespace mlir::llk::target::generic_accel
