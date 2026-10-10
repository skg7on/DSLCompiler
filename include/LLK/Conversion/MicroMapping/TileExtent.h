//===- TileExtent.h - Shared bounded-tile utilities -------------*- C++ -*-===//
//
// This file is part of the LLK project.
//
//===----------------------------------------------------------------------===//

#ifndef LLK_CONVERSION_MICROMAPPING_TILEEXTENT_H
#define LLK_CONVERSION_MICROMAPPING_TILEEXTENT_H

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"

#include <limits>
#include <optional>

namespace mlir::llk::micro_mapping {

/// Returns the smallest multiple of `tile` that covers `logical`, or no value
/// when the extent is invalid or the rounded value would overflow.
inline std::optional<int64_t> roundUpExtent(int64_t logical, int64_t tile) {
  if (logical < 0 || tile <= 0)
    return std::nullopt;
  int64_t blocks = logical / tile + (logical % tile != 0);
  if (blocks > std::numeric_limits<int64_t>::max() / tile)
    return std::nullopt;
  return blocks * tile;
}

/// Materializes a statically shaped, zero-filled pad for a ranked tensor.
inline Value padTensorWithZeros(OpBuilder &builder, Location loc, Value source,
                                ArrayRef<int64_t> paddedShape) {
  auto sourceType = dyn_cast<RankedTensorType>(source.getType());
  if (!sourceType ||
      static_cast<size_t>(sourceType.getRank()) != paddedShape.size())
    return {};
  if (sourceType.getShape() == paddedShape)
    return source;

  SmallVector<int64_t> low(paddedShape.size(), 0);
  SmallVector<int64_t> high(paddedShape.size(), 0);
  for (size_t i = 0; i < paddedShape.size(); ++i) {
    if (paddedShape[i] < sourceType.getDimSize(i))
      return {};
    high[i] = paddedShape[i] - sourceType.getDimSize(i);
  }
  auto resultType =
      RankedTensorType::get(paddedShape, sourceType.getElementType());
  tensor::PadOp padded = tensor::PadOp::create(
      builder, loc, resultType, source, ValueRange{}, ValueRange{}, low, high,
      /*nofold=*/false);
  Block *body = new Block();
  padded.getRegion().push_back(body);
  for (size_t i = 0; i < paddedShape.size(); ++i)
    body->addArgument(IndexType::get(builder.getContext()), loc);
  builder.setInsertionPointToEnd(body);
  Value zero = arith::ConstantOp::create(
      builder, loc, builder.getZeroAttr(sourceType.getElementType()));
  tensor::YieldOp::create(builder, loc, zero);
  builder.setInsertionPointAfter(padded);
  return padded.getResult();
}

} // namespace mlir::llk::micro_mapping

#endif // LLK_CONVERSION_MICROMAPPING_TILEEXTENT_H
