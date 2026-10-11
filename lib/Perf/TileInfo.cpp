#include "MicroTileInfo.h"

namespace mlir::llk::perf {
TileInfo describeType(mlir::Type type) {
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
std::string shapeString(llvm::ArrayRef<int64_t> shape) {
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
