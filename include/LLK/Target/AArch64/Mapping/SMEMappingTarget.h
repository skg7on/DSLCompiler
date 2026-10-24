//===- SMEMappingTarget.h - Static SME mapping target -----------*- C++ -*-===//
//
// Part of the LLK Compiler Project, under the Apache License v2.0 with
// LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLK_TARGET_AARCH64_MAPPING_SMEMAPPINGTARGET_H
#define LLK_TARGET_AARCH64_MAPPING_SMEMAPPINGTARGET_H

#include "LLK/Mapping/MappingTarget.h"

#include "mlir/Pass/PassManager.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <memory>

namespace mlir::llk::target::sme {

/// Declared mapping emitter keys. `sme_mma_za` and the three movement keys
/// carry a lowering; every other key keeps its explicit rejection.
llvm::ArrayRef<llvm::StringRef> emitterKeys();

/// An emitter for one declared key. Only `sme_mma_za`, `sme_copy`,
/// `sme_tile_copy` and `sme_tile_store` lower their covered groups.
std::unique_ptr<mapping::TargetEmitter> createSMEEmitter(llvm::StringRef key);

/// Adds the pre-bufferization scalable-vector lowering for the selected
/// contraction. Defined in SMEVectorLowering.cpp.
void buildSMESelectedBackendPipeline(mlir::OpPassManager &pm);

/// Adds the post-bufferization ArmSME conversion and realization accounting.
/// Defined in SMEArmLowering.cpp.
void buildSMEPostBufferizationPipeline(mlir::OpPassManager &pm);

/// Loads machines/arm-sme-v2.yaml and mapping/arm-sme/{layouts,rules}.llkmap
/// below configurationRoot. Returns the target with the f32 contraction
/// executable subset and the AArch64/SME streaming contract.
llvm::Expected<std::unique_ptr<mapping::MappingTarget>>
createMappingTarget(llvm::StringRef configurationRoot,
                    llvm::StringRef machinePath = {});

} // namespace mlir::llk::target::sme

#endif // LLK_TARGET_AARCH64_MAPPING_SMEMAPPINGTARGET_H
