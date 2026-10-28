//===- BoundaryDerivation.cpp - Boundary facts of covered operations ------===//
//
// Part of the LLK Compiler Project, under the Apache License v2.0 with
// LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "BoundaryDerivation.h"

#include "LLK/Shape/ShapeContract.h"

// MicroDialect/MicroEnums/MicroHelpers are prerequisites for the generated op
// declarations below, not direct uses here.
#include "LLK/Dialect/Micro/MicroDialect.h" // IWYU pragma: keep
#include "LLK/Dialect/Micro/MicroEnums.h"   // IWYU pragma: keep
#include "LLK/Dialect/Micro/MicroHelpers.h" // IWYU pragma: keep

#include "mlir/Bytecode/BytecodeOpInterface.h" // IWYU pragma: keep
#include "mlir/Dialect/Arith/IR/Arith.h"       // IWYU pragma: keep
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/Builders.h"         // IWYU pragma: keep
#include "mlir/IR/BuiltinTypes.h"     // IWYU pragma: keep
#include "mlir/IR/OpImplementation.h" // IWYU pragma: keep
#include "mlir/IR/Operation.h"
#include "mlir/Interfaces/CallInterfaces.h" // IWYU pragma: keep

#include <optional>

// Micro op declarations.
#define GET_ATTRDEF_CLASSES
#include "LLK/Dialect/Micro/MicroAttributes.h.inc"
#define GET_TYPEDEF_CLASSES
#include "LLK/Dialect/Micro/MicroTypes.h.inc"
#define GET_OP_CLASSES
#include "LLK/Dialect/Micro/MicroOps.h.inc"

namespace mlir::llk::mapping {

namespace {

/// Classifies one boundary extent against the tile capacity it must fill.
///
/// A constant is decided exactly: equal to the capacity is a proven full tile,
/// anything smaller is a proven partial one. A runtime value is a guard: it may
/// be full at one invocation and partial at the next, so the plan needs the
/// bounded implementation and the executable form keeps the guard.
shape::PredicateVerdict classifyExtent(Value extent, int64_t capacity) {
  std::optional<int64_t> constant = getConstantIntValue(extent);
  if (!constant)
    return shape::PredicateVerdict::RuntimeGuard;
  if (capacity <= 0)
    return shape::PredicateVerdict::RuntimeGuard;
  return *constant == capacity ? shape::PredicateVerdict::Proven
                               : shape::PredicateVerdict::Disproven;
}

/// True when any extent in `extents` is not a proven full tile. `capacities` is
/// one capacity per extent, in the same order.
bool needsBounded(mlir::ValueRange extents,
                  llvm::ArrayRef<int64_t> capacities) {
  if (extents.size() != capacities.size())
    return true; // an unverifiable boundary is not a full-tile one
  for (size_t index = 0; index < extents.size(); ++index)
    if (classifyExtent(extents[index], capacities[index]) !=
        shape::PredicateVerdict::Proven)
      return true;
  return false;
}

} // namespace

BoundaryRequirements
boundaryRequirementsFor(llvm::ArrayRef<mlir::Operation *> coveredOps) {
  BoundaryRequirements requirements;
  for (mlir::Operation *op : coveredOps) {
    if (!op)
      continue;

    if (auto load = dyn_cast<micro::TileLoadOp>(op)) {
      // The valid rectangle is copied into the origin of the static result
      // tile, so the capacity it must fill is the result tile's own shape.
      auto resultType = dyn_cast<micro::TileType>(load.getResult().getType());
      if (!resultType) {
        requirements.boundedLoad = true;
        continue;
      }
      requirements.boundedLoad |=
          needsBounded(load.getValidSizes(), resultType.getShape());
      continue;
    }

    if (auto store = dyn_cast<micro::TileStoreOp>(op)) {
      // An absent `valid[...]` is the legacy full-tile write; only a present
      // pair of extents can be a clipped store, and the capacity they must fill
      // is the source tile's shape (the rectangle lives in the source).
      if (store.getValidSizes().empty())
        continue;
      auto sourceType = dyn_cast<micro::TileType>(store.getSource().getType());
      if (!sourceType) {
        requirements.boundedStore = true;
        continue;
      }
      requirements.boundedStore |=
          needsBounded(store.getValidSizes(), sourceType.getShape());
      continue;
    }

    if (auto mma = dyn_cast<micro::MmaOp>(op)) {
      Value validK = mma.getValidK();
      if (!validK)
        continue; // omission is the full-K contract every emitter already has
      llvm::ArrayRef<int64_t> shape = mma.getShape();
      // `shape` is [M, N, K]; the reduction capacity is K.
      int64_t capacity = shape.size() == 3 ? shape[2] : 0;
      requirements.partialK |=
          classifyExtent(validK, capacity) != shape::PredicateVerdict::Proven;
      continue;
    }
  }
  return requirements;
}

} // namespace mlir::llk::mapping
