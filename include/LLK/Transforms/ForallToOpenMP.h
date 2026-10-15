//===- ForallToOpenMP.h - Lower scf.forall to OpenMP ------------*- C++ -*-===//
//
// Part of the LLK Compiler Project, under the Apache License v2.0 with
// LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Prototype parallel lowering path: scf.forall -> scf.parallel -> OpenMP.
// Uses MLIR's built-in conversion chain. For production use, prefer
// the LLRT thread pool path (ForallToLLRT).

#ifndef LLK_TRANSFORMS_FORALLTOOPENMP_H
#define LLK_TRANSFORMS_FORALLTOOPENMP_H

#include <memory>

namespace mlir {
class Pass;
} // namespace mlir

namespace mlir {
namespace llk {

/// Create a pass that lowers scf.forall to OpenMP via the chain:
///   scf.forall -> scf.parallel -> omp.parallel + omp.wsloop
///
/// This is an alternative to ForallToLLRT for environments where
/// OpenMP runtime integration is preferred over a custom thread pool.
std::unique_ptr<mlir::Pass> createForallToOpenMPPass();

} // namespace llk
} // namespace mlir

#endif // LLK_TRANSFORMS_FORALLTOOPENMP_H
