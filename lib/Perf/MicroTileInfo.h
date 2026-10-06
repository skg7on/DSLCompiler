//===- MicroTileInfo.h - Tile metadata read off an MLIR type -------------===//
//
// Internal to the perf library. The cost model must read shape, dtype, layout,
// memory space, and owner out of whatever a `micro` op produced or consumed,
// and those live on two different carriers: `!micro.tile` for execution values,
// and a plain shaped type for the tensors a kernel reads from and writes back
// to external memory. describeType() flattens both into one struct so no cost
// computation has to care which one it got.
//
//===----------------------------------------------------------------------===//

#ifndef LLK_PERF_MICROTILEINFO_H
#define LLK_PERF_MICROTILEINFO_H

#include "LLK/Dialect/Micro/MicroDialect.h"
#include "LLK/Dialect/Micro/MicroEnums.h"
#include "LLK/Dialect/Micro/MicroHelpers.h"

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSwitch.h"

#define GET_ATTRDEF_CLASSES
#include "LLK/Dialect/Micro/MicroAttributes.h.inc"

#define GET_TYPEDEF_CLASSES
#include "LLK/Dialect/Micro/MicroTypes.h.inc"

#define GET_OP_CLASSES
#include "LLK/Dialect/Micro/MicroOps.h.inc"

#include <string>

namespace mlir::llk::perf {

struct TileInfo {
  bool isTile = false;
  llvm::SmallVector<int64_t, 4> shape;
  std::string dtype;
  std::string layout;
  std::string memory;
  std::string owner;

  /// Bytes of one element of `dtype`, or 0 for a dtype the micro dialect does
  /// not name.
  static unsigned dtypeBytes(llvm::StringRef dtype) {
    return llvm::StringSwitch<unsigned>(dtype)
        .Case("f32", 4)
        .Case("i32", 4)
        .Case("f16", 2)
        .Case("bf16", 2)
        .Case("i8", 1)
        .Default(0);
  }

  /// A dynamic extent makes the element count, and therefore the byte count,
  /// unknown. Concrete micro kernels are expected to have static tiles.
  bool hasDynamicShape() const {
    for (int64_t dim : shape)
      if (ShapedType::isDynamic(dim))
        return true;
    return false;
  }

  uint64_t elements() const {
    if (shape.empty())
      return 0;
    uint64_t count = 1;
    for (int64_t dim : shape) {
      if (ShapedType::isDynamic(dim) || dim <= 0)
        return 0;
      count *= static_cast<uint64_t>(dim);
    }
    return count;
  }

  uint64_t bytes() const {
    unsigned width = dtypeBytes(dtype);
    return width == 0 ? 0 : elements() * width;
  }
};

inline TileInfo describeType(mlir::Type type) {
  TileInfo info;
  if (auto tile = mlir::dyn_cast<micro::TileType>(type)) {
    info.isTile = true;
    info.shape.assign(tile.getShape().begin(), tile.getShape().end());
    if (auto dtype = micro::dtypeOfElementType(tile.getElementType()))
      info.dtype = micro::stringifyDType(*dtype).str();
    if (auto layout = tile.getLayout())
      info.layout = micro::stringifyLayoutKind(
                        static_cast<micro::LayoutKind>(layout.getKind()))
                        .str();
    if (auto memory = tile.getMemory())
      info.memory = micro::stringifyMemorySpace(memory.getValue()).str();
    if (auto owner = tile.getOwner())
      // The owner is an open symbol now: the spelling the IR carries is exactly
      // the string a target resolves, so it is recorded verbatim.
      info.owner = owner.getSymbol().str();
    return info;
  }
  if (auto shaped = mlir::dyn_cast<ShapedType>(type)) {
    info.shape.assign(shaped.getShape().begin(), shaped.getShape().end());
    if (auto dtype = micro::dtypeOfElementType(shaped.getElementType()))
      info.dtype = micro::stringifyDType(*dtype).str();
  }
  return info;
}

/// `16x16x32`, with `?` for a dynamic extent.
inline std::string shapeString(llvm::ArrayRef<int64_t> shape) {
  std::string text;
  for (size_t i = 0; i < shape.size(); ++i) {
    if (i != 0)
      text += "x";
    if (ShapedType::isDynamic(shape[i]))
      text += "?";
    else
      text += std::to_string(shape[i]);
  }
  return text;
}

} // namespace mlir::llk::perf

#endif // LLK_PERF_MICROTILEINFO_H
