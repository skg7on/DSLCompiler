//===- SVEMappingTarget.h - Static SVE mapping target -----------*- C++ -*-===//
//
// Part of the LLK Compiler Project, under the Apache License v2.0 with
// LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLK_TARGET_AARCH64_MAPPING_SVEMAPPINGTARGET_H
#define LLK_TARGET_AARCH64_MAPPING_SVEMAPPINGTARGET_H

#include "LLK/Mapping/MappingTarget.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <memory>

namespace mlir::llk::target::sve {

/// Declared static-mapping emitter keys; these emitters have no lowering.
llvm::ArrayRef<llvm::StringRef> emitterKeys();

/// Load machines/arm-sve-v2.yaml and mapping/arm-sve/{layouts,rules}.llkmap
/// below configurationRoot. A nonempty machinePath overrides the profile.
/// Returns the file-backed target without a native code-generation contract.
llvm::Expected<std::unique_ptr<mapping::MappingTarget>>
createMappingTarget(llvm::StringRef configurationRoot,
                    llvm::StringRef machinePath = {});

} // namespace mlir::llk::target::sve

#endif // LLK_TARGET_AARCH64_MAPPING_SVEMAPPINGTARGET_H
