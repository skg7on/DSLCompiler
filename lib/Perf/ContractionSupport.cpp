//===- ContractionSupport.cpp - Contraction dtype compatibility -----------===//
//
// Part of the LLK Compiler Project, under the Apache License v2.0 with
// LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "LLK/Perf/ContractionSupport.h"

#include "llvm/ADT/STLExtras.h"

namespace mlir::llk::perf {

bool supportsContractionTypes(const machine::ComputeNode &engine,
                              llvm::StringRef inputDtype,
                              llvm::StringRef accumulatorDtype) {
  // Only compute capabilities execute contractions. A node of another kind is
  // rejected even if it happens to carry matching dtype lists.
  if (engine.kind != "vector_engine" && engine.kind != "matrix_engine")
    return false;
  return llvm::is_contained(engine.elementTypes, inputDtype) &&
         llvm::is_contained(engine.accumulatorDTypes, accumulatorDtype);
}

} // namespace mlir::llk::perf
