//===- MaskGeneration.cpp - Masked load/store for vector tails ------------===//
//
// Utilities for generating vector.create_mask operations to handle tail
// elements after tiling when the tile size doesn't divide the vector width.
//
//===----------------------------------------------------------------------===//

#include "LLK/Transforms/Common/MaskGeneration.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Builders.h"

using namespace mlir;

namespace mlir {
namespace llk {

Value createVectorMask(OpBuilder &b, Location loc, int64_t numElements,
                       int64_t vectorWidth) {
  Value numElems = b.create<arith::ConstantIndexOp>(loc, numElements);
  Value dim = b.create<arith::ConstantIndexOp>(loc, vectorWidth);
  return b.create<vector::CreateMaskOp>(
      loc, VectorType::get({vectorWidth}, b.getI1Type()),
      ValueRange{numElems, dim});
}

} // namespace llk
} // namespace mlir
