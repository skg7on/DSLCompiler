//===- BoundaryDerivation.h - Boundary facts of covered operations --------===//
//
// Part of the LLK Compiler Project, under the Apache License v2.0 with
// LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Dynamic-shape design §5/§6. The target-neutral mapping core carries a typed
// `BoundaryRequirements`; the Micro dialect is the only place that can read an
// operation's boundary operands, so it lives here, above the core and above
// every target.
//
// The derivation is deliberately structural and operator-generic: it asks what
// the covered operations need, never what workload they belong to. A bounded
// load in a direct elementwise kernel and one in a contraction produce the same
// requirement, which is what keeps the contract from becoming a contraction
// feature.

#ifndef LLK_CONVERSION_MICROMAPPING_BOUNDARYDERIVATION_H
#define LLK_CONVERSION_MICROMAPPING_BOUNDARYDERIVATION_H

#include "LLK/Mapping/BoundaryRequirements.h"

#include "llvm/ADT/ArrayRef.h"

namespace mlir {
class Operation;
} // namespace mlir

namespace mlir::llk::mapping {

/// The boundary behaviour `coveredOps` require.
///
/// A `micro.tile_load` needs bounded-load support unless every valid extent is
/// a constant that fills the result tile; a `micro.tile_store` with valid
/// extents needs bounded-store support under the same proof; a `micro.mma` with
/// a valid-K bound needs partial-K support unless that bound provably covers
/// the static K capacity. A runtime extent is never assumed full: it is exactly
/// the `PredicateVerdict::RuntimeGuard` case the bounded implementation exists
/// for. Operations outside that family require nothing, which is what keeps
/// every pre-existing full-tile plan on its unchanged path.
BoundaryRequirements
boundaryRequirementsFor(llvm::ArrayRef<mlir::Operation *> coveredOps);

} // namespace mlir::llk::mapping

#endif // LLK_CONVERSION_MICROMAPPING_BOUNDARYDERIVATION_H
