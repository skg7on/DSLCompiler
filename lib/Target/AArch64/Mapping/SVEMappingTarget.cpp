//===- SVEMappingTarget.cpp - Static SVE mapping target -------------------===//
//
// Part of the LLK Compiler Project, under the Apache License v2.0 with
// LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "LLK/Target/AArch64/Mapping/SVEMappingTarget.h"

#include <string>
#include <utility>
#include <vector>

namespace mlir::llk::target::sve {

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
  return mapping::loadMappingTarget(
      "arm-sve", machine, root + "/mapping/arm-sve/layouts.llkmap",
      root + "/mapping/arm-sve/rules.llkmap", std::move(keys));
}

} // namespace mlir::llk::target::sve
