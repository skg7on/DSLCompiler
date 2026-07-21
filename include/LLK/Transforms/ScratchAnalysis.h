//===- ScratchAnalysis.h - Audit memref allocs for intermediates -*- C++
//-*-===//
//
// Part of the LLK Compiler Project, under the Apache License v2.0.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Post-bufferization pass that walks all memref.alloc ops and warns on
// allocations larger than 1 MB, flagging potential full-size intermediates
// that could be fused away.
//
//===----------------------------------------------------------------------===//

#ifndef LLK_TRANSFORMS_SCRATCHANALYSIS_H
#define LLK_TRANSFORMS_SCRATCHANALYSIS_H

#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir {
namespace llk {
std::unique_ptr<Pass> createScratchAnalysisPass();
} // namespace llk
} // namespace mlir

#endif
