//===- TargetCoverageValidation.h - Validate covered operations ----------===//
//
// Part of the LLK Compiler Project, under the Apache License v2.0 with
// LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Framework C3. Bundle-key validation proves a bundle is well formed; it does
// not prove the target implements the *operation* the bundle was selected for.
// This helper groups the bound IR by the selected instance metadata, asks each
// plan bundle's emitter to validate its covered operations, and never rewrites
// the IR. It runs before complete-plan analysis/scoring, so an unsupported
// variant or dtype is a candidate rejection rather than a late lowering error.
//
//===----------------------------------------------------------------------===//

#ifndef LLK_CONVERSION_MICROMAPPING_TARGETCOVERAGEVALIDATION_H
#define LLK_CONVERSION_MICROMAPPING_TARGETCOVERAGEVALIDATION_H

#include "LLK/Mapping/MappingPlan.h"
#include "LLK/Mapping/MappingTarget.h"

#include "mlir/IR/BuiltinOps.h"

#include "llvm/Support/Error.h"

namespace mlir::llk::mapping {

/// Verifies every selected instance in `bound` against the emitter its plan
/// bundle names.
///
/// Operations are grouped by the `micro.mapping` instance metadata the binder
/// recorded; each plan placement must have exactly one mapped operation, a
/// group's bundle/rule must agree with its placement, and the emitter's
/// `verifyOperations` must accept the group. Returns the first violation in
/// deterministic (placement/instance) order. The IR is not modified.
llvm::Error verifyTargetOperationCoverage(mlir::ModuleOp bound,
                                          const CoveringPlan &plan,
                                          const MappingTarget &target);

} // namespace mlir::llk::mapping

#endif // LLK_CONVERSION_MICROMAPPING_TARGETCOVERAGEVALIDATION_H
