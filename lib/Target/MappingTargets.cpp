//===- MappingTargets.cpp - Register the mapping target packages ----------===//
//
// Part of the LLK Compiler Project, under the Apache License v2.0 with
// LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// The composition root for target packages: the one translation unit that names
// them, so nothing else has to. Each name here is the label its package already
// writes into its own configuration, and it is what `--target=<name>` resolves.

#include "LLK/Target/MappingTargets.h"

#include "LLK/Target/AArch64/Mapping/SMEMappingTarget.h"
#include "LLK/Target/AArch64/Mapping/SVEMappingTarget.h"
#include "LLK/Target/GenericAccelerator/Mapping/GenericAcceleratorMappingTarget.h"
#include "LLK/Target/X86/Mapping/AVX2MappingTarget.h"

namespace mlir::llk::target {

void registerAllMappingTargets() {
  mapping::registerMappingTarget("x86-avx2", &avx2::createMappingTarget);
  mapping::registerMappingTarget("generic-ai-accel",
                                 &generic_accel::createMappingTarget);
  mapping::registerMappingTarget("arm-sve", &sve::createMappingTarget);
  mapping::registerMappingTarget("arm-sme", &sme::createMappingTarget);
}

} // namespace mlir::llk::target
