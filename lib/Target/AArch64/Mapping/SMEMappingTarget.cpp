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
  return mapping::loadMappingTarget(
      "arm-sme", machine, root + "/mapping/arm-sme/layouts.llkmap",
      root + "/mapping/arm-sme/rules.llkmap", std::move(keys));
}

} // namespace mlir::llk::target::sme
