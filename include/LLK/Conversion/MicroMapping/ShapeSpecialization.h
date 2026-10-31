//===- ShapeSpecialization.h - Build an analysis-only shape sample --------===//
//
// Part of the LLK Compiler Project, under the Apache License v2.0 with
// LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Dynamic analysis ranks a compiled kernel on explicit concrete shape samples.
// The executable artifact keeps its runtime dimensions; scoring one sample uses
// an *analysis-only clone* in which the port dimension queries and the guards
// built from them are resolved to constants. Loop bounds then fold to concrete
// trip counts, while boundary extents that still depend on an induction
// variable remain ordinary arithmetic for the shared shape evaluator.
//
// `materializeShapeSample` performs exactly that substitution. It clones the
// module first, so the caller's symbolic source is never mutated. It resolves
// only shape computations -- `tensor.dim`/`memref.dim` of a kernel argument
// bound to a contract port -- and never rewrites tensor types, tile shapes or
// computational operations. A value it cannot relate to a contract symbol is an
// error rather than a guessed constant.
//
//===----------------------------------------------------------------------===//

#ifndef LLK_CONVERSION_MICROMAPPING_SHAPESPECIALIZATION_H
#define LLK_CONVERSION_MICROMAPPING_SHAPESPECIALIZATION_H

#include "LLK/Conversion/Workload/WorkloadContract.h"
#include "LLK/Shape/ShapeContract.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"

#include "llvm/Support/Error.h"

namespace mlir::llk::mapping {

/// The plain shape contract a versioned workload contract describes. Ports keep
/// their declared order and directions; the symbol table is copied verbatim.
shape::ShapeContract
toShapeContract(const workload::WorkloadContract &contract);

/// Decodes the `micro.workload_contract` a kernel carries into its plain shape
/// contract. Fails when the kernel carries none: a sample cannot be bound to
/// symbols the artifact never declared.
llvm::Expected<shape::ShapeContract> shapeContractOf(mlir::Operation *kernel);

/// Clones `source` and resolves the shape computations of every `micro.kernel`
/// it contains against `values`.
///
/// * Each kernel argument is related to a contract port through its
///   `argument_bindings` attribute; a kernel without that attribute is treated
///   as signature-ordered (arguments bind contract ports in order).
/// * A `tensor.dim`/`memref.dim` of such an argument is replaced by the
///   contract's constant or by the sampled symbol value.
/// * `values` must bind every declared symbol exactly once, be nonnegative and
///   satisfy the contract's symbol constraints. A missing, unknown or
///   inconsistent value fails before any IR is changed.
///
/// The clone is folded afterwards so guards and loop bounds become constants.
/// The original `source` and its runtime dimensions are untouched.
llvm::Expected<mlir::OwningOpRef<mlir::ModuleOp>>
materializeShapeSample(mlir::ModuleOp source,
                       const shape::ShapeContract &contract,
                       const shape::ShapeValues &values);

} // namespace mlir::llk::mapping

#endif // LLK_CONVERSION_MICROMAPPING_SHAPESPECIALIZATION_H
