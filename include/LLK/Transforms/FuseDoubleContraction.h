//===- FuseDoubleContraction.h - Fuse 2 matmuls + SiLU consumer -*- C++ -*-===//
//
// Part of the LLK Compiler Project, under the Apache License v2.0.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Fuses two linalg.matmul ops sharing the same X operand with their
// elementwise SiLU consumer into a single linalg.generic with 2 reduction
// outputs and an epilogue generic.
//
//===----------------------------------------------------------------------===//

#ifndef LLK_TRANSFORMS_FUSEDOUBLECONTRACTION_H
#define LLK_TRANSFORMS_FUSEDOUBLECONTRACTION_H

#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir {
namespace llk {
std::unique_ptr<Pass> createFuseDoubleContractionPass();
} // namespace llk
} // namespace mlir

#endif
