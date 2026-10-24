//===- SMEVectorLowering.cpp - Selected SME pre-bufferization lowering ----===//
//
// Part of the LLK Compiler Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Task 4 stub: the selective executable target declares the hook so the
// target contract is complete. Task 5 fills the pass sequence.

#include "LLK/Target/AArch64/Mapping/SMEMappingTarget.h"

namespace mlir::llk::target::sme {

void buildSMESelectedBackendPipeline(mlir::OpPassManager &) {}

} // namespace mlir::llk::target::sme
