//===- MicroHelpers.h - Micro dialect shared helpers -------------*- C++
//-*-===//
//
// Small shared helpers for the Micro dialect, used by both the dialect
// implementation (MicroDialect.cpp) and the op verifiers (MicroOps.cpp).
//
//===----------------------------------------------------------------------===//

#ifndef LLK_DIALECT_MICRO_MICROHELPERS_H
#define LLK_DIALECT_MICRO_MICROHELPERS_H

#include "LLK/Dialect/Micro/MicroEnums.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinTypes.h"

#include <optional>

namespace mlir::micro {

/// Returns the Micro dtype corresponding to an MLIR element type, or nullopt
/// if the type is not a supported tile element type (f32, f16, bf16, i32, i8).
inline std::optional<DType> dtypeOfElementType(Type elementType) {
  if (elementType.isBF16())
    return DType::bf16;
  if (elementType.isF16())
    return DType::f16;
  if (elementType.isF32())
    return DType::f32;
  if (elementType.isInteger(32))
    return DType::i32;
  if (elementType.isInteger(8))
    return DType::i8;
  return std::nullopt;
}

/// Builds a materialized `!micro.tile` for `source` in the abstract memory
/// space `memory` (a `#micro.memory<kind>` attribute), preserving the logical
/// shape and element type.
///
/// `source` may be an existing tile -- whose layout and owner are carried over
/// -- or a shaped value (tensor/vector/memref), which is given the destination
/// memory with no layout or owner. This is the canonical adapter a materializer
/// uses to retype a value into the memory a selected transfer lands it in.
///
/// Returns a null `Type` when `source` has no statically known shape, carries
/// no supported Micro element type, or is not a shaped value or tile; or when
/// `memory` is not a memory-space attribute. Declared here but defined in
/// MicroOps.cpp, which has the generated tile and attribute classes.
Type materializedTileType(Type source, Attribute memory);

} // namespace mlir::micro

#endif // LLK_DIALECT_MICRO_MICROHELPERS_H
