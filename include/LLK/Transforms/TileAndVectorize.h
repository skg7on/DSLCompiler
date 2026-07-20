//===- TileAndVectorize.h - Tiling + Vectorization Pass -----------------*- C++ -*-===//
//
// Part of the LLK Compiler Project, under the Apache License v2.0.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Applies a Transform dialect schedule to tile, fuse, and vectorize LLK
// operations that have been lowered to Linalg. Reads schedule from a
// .mlir transform file or uses built-in defaults keyed by target ISA.
//
//===----------------------------------------------------------------------===//

#ifndef LLK_TRANSFORMS_TILEANDVECTORIZE_H
#define LLK_TRANSFORMS_TILEANDVECTORIZE_H

#include "mlir/Pass/Pass.h"
#include <memory>
#include <string>

namespace mlir {
namespace llk {

struct TileAndVectorizeOptions {
  std::string scheduleFile;
  std::string targetIsa = "avx2";
};

std::unique_ptr<Pass> createTileAndVectorizePass();
std::unique_ptr<Pass> createTileAndVectorizePass(const TileAndVectorizeOptions &opts);

} // namespace llk
} // namespace mlir

#endif // LLK_TRANSFORMS_TILEANDVECTORIZE_H
