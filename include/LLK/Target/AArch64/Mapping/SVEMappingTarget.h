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

#include "mlir/Pass/PassManager.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <memory>

namespace mlir::llk::target::sve {

/// Declared mapping emitter keys. Only `sve_vector_add` and `sve_vector_mul`
/// carry a lowering; every other key keeps its explicit rejection.
llvm::ArrayRef<llvm::StringRef> emitterKeys();

/// An emitter that lowers the f32 vector add/multiply subset handed to it and
/// keeps every other key's explicit rejection.
std::unique_ptr<mapping::TargetEmitter> createSVEEmitter(llvm::StringRef key);

/// Adds the SVE scalable-vector lowering after generic Micro to Linalg
/// conversion. `f32VectorWidth` is the machine profile's f32 lane count; the
/// pass emits `vscale * 4` f32 lanes so the IR is valid at any SVE vector
/// length while the modeled profile is the 512-bit, 16-lane configuration.
void buildSVESelectedBackendPipeline(mlir::OpPassManager &pm,
                                     int64_t f32VectorWidth);

/// Load machines/arm-sve-v2.yaml and mapping/arm-sve/{layouts,rules}.llkmap
/// below configurationRoot. A nonempty machinePath overrides the profile.
/// Returns the target with its executable arithmetic emitters and the
/// AArch64/SVE code-generation contract; the unimplemented keys still reject.
llvm::Expected<std::unique_ptr<mapping::MappingTarget>>
createMappingTarget(llvm::StringRef configurationRoot,
                    llvm::StringRef machinePath = {});

} // namespace mlir::llk::target::sve

#endif // LLK_TARGET_AARCH64_MAPPING_SVEMAPPINGTARGET_H
