//===- AVX2MappingTarget.cpp - AVX2 mapping package (D7) ------------------===//

#include "LLK/Target/X86/Mapping/AVX2MappingTarget.h"

#include <string>
#include <vector>

namespace mlir::llk::target::avx2 {

llvm::ArrayRef<llvm::StringLiteral> emitterKeys() {
  static const llvm::StringLiteral kKeys[] = {
      "avx2_vector_add", "avx2_mma",       "avx2_reduce",
      "avx2_copy",       "avx2_tile_copy", "avx2_tile_store"};
  return kKeys;
}

llvm::Expected<std::unique_ptr<mapping::MappingTarget>>
createMappingTarget(llvm::StringRef configurationRoot) {
  std::string root = configurationRoot.str();
  std::vector<std::string> keys;
  for (llvm::StringLiteral key : emitterKeys())
    keys.push_back(key.str());
  return mapping::loadMappingTarget(
      "x86-avx2", root + "/machines/x86-avx2-v2.yaml",
      root + "/mapping/x86-avx2/layouts.llkmap",
      root + "/mapping/x86-avx2/rules.llkmap", std::move(keys));
}

} // namespace mlir::llk::target::avx2
