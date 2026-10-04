//===- TileFacts.h - Bytes and alignment of a moving value ----------------===//
//
// Part of the target-independent mapping core (epic #67, phase 3).
//
// A connection, a capacity charge, and a staging entry all need the size of the
// tile actually moving, not one assumed number. `tileFactsFor` derives it from
// the value's type.
//
// The core deliberately does not link the Micro dialect (see WorkloadGraph.h),
// so a `!micro.tile<shape x element, ...>` is read through its printed form and
// re-parsed as the `tensor<shape x element>` its head spells. That unwrapping
// lives here, once, so matching and placement cannot drift into a second and
// third answer for the same concept.
//
//===----------------------------------------------------------------------===//

#ifndef LLK_MAPPING_TILE_FACTS_H
#define LLK_MAPPING_TILE_FACTS_H

#include "mlir/IR/Types.h"

#include "llvm/ADT/SmallVector.h"

#include <cstdint>
#include <optional>

namespace mlir::llk::mapping {

/// The byte size and natural alignment of the tile a value carries, or
/// `known == false` when its type does not state a static, width-bearing size.
struct TileFacts {
  uint64_t bytes = 0;
  uint64_t alignment = 0;
  /// True only when `bytes` and `alignment` were derived from a static shape
  /// and a width-bearing element type. A caller must not read the two numbers
  /// as a size when this is false -- it sizes the value by an explicit,
  /// *reported* fallback instead.
  bool known = false;
};

/// The `tensor` a `!micro.tile<shape x element, ...>` denotes, read through its
/// printed form. Any other type -- including a `tensor` already -- yields a
/// null type.
mlir::Type tileAsTensor(mlir::Type type);

/// The element type a type exposes: a shaped type's element (a `!micro.tile`
/// unwrapped first), or a bare float, integer, or index type itself. A null
/// type when neither applies.
mlir::Type elementTypeOf(mlir::Type type);

/// The static shape a type exposes (a `!micro.tile` unwrapped first), or
/// `std::nullopt` for a dynamic, unranked, or opaque type.
std::optional<llvm::SmallVector<int64_t, 4>> staticShapeOf(mlir::Type type);

/// Bytes and alignment of the tile `valueType` carries:
/// `bytes = numElements * elementByteWidth`, `alignment = elementByteWidth`.
/// A dynamic or unranked shape, a widthless element type (an index), or an
/// unrecognised type returns `known == false` with both numbers zero.
TileFacts tileFactsFor(mlir::Type valueType);

} // namespace mlir::llk::mapping

#endif // LLK_MAPPING_TILE_FACTS_H
