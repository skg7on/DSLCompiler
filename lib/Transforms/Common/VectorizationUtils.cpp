//===- VectorizationUtils.cpp - Generic vectorization utilities -----------===//
//
// Convenience builders for vector.transfer_read, vector.transfer_write,
// boundary masks, and zero-index helpers.  Extracts common patterns that
// were previously duplicated across lowering passes.
//
//===----------------------------------------------------------------------===//

#include "LLK/Transforms/Common/VectorizationUtils.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Attributes.h"

using namespace mlir;

namespace mlir {
namespace llk {

// ---------------------------------------------------------------------------
// buildZeroIndices
// ---------------------------------------------------------------------------

SmallVector<Value> buildZeroIndices(OpBuilder &b, Location loc, unsigned rank) {
  SmallVector<Value> zeros;
  for (unsigned i = 0; i < rank; ++i)
    zeros.push_back(arith::ConstantIndexOp::create(b, loc, 0));
  return zeros;
}

// ---------------------------------------------------------------------------
// getAllInBounds
// ---------------------------------------------------------------------------

SmallVector<bool> getAllInBounds(unsigned rank) {
  return SmallVector<bool>(rank, true);
}

// ---------------------------------------------------------------------------
// createTransferRead
// ---------------------------------------------------------------------------

Value createTransferRead(OpBuilder &b, Location loc, Value memref,
                         ValueRange indices, VectorType vectorType, Value pad,
                         Value mask, ArrayRef<bool> inBounds) {
  auto ctx = b.getContext();
  unsigned rank = indices.size();

  // Identity permutation map.
  auto permMap =
      AffineMapAttr::get(AffineMap::getMultiDimIdentityMap(rank, ctx));

  // Default: all elements are in-bounds (best-effort from subview).
  SmallVector<bool> bounds;
  if (inBounds.empty())
    bounds = getAllInBounds(rank);
  else
    bounds.assign(inBounds.begin(), inBounds.end());
  auto boundsAttr = b.getBoolArrayAttr(bounds);

  // Default: zero padding.
  if (!pad) {
    auto elemType = vectorType.getElementType();
    pad = arith::ConstantOp::create(b, loc, elemType, b.getZeroAttr(elemType));
  }

  return vector::TransferReadOp::create(b, loc, vectorType, memref, indices,
                                        permMap, pad, mask, boundsAttr)
      .getResult();
}

// ---------------------------------------------------------------------------
// createTransferWrite
// ---------------------------------------------------------------------------

void createTransferWrite(OpBuilder &b, Location loc, Value vector, Value memref,
                         ValueRange indices, Value mask,
                         ArrayRef<bool> inBounds) {
  auto ctx = b.getContext();
  unsigned rank = indices.size();

  auto permMap =
      AffineMapAttr::get(AffineMap::getMultiDimIdentityMap(rank, ctx));

  SmallVector<bool> bounds;
  if (inBounds.empty())
    bounds = getAllInBounds(rank);
  else
    bounds.assign(inBounds.begin(), inBounds.end());
  auto boundsAttr = b.getBoolArrayAttr(bounds);

  vector::TransferWriteOp::create(b, loc, vector, memref, indices, permMap,
                                  mask, boundsAttr);
}

// ---------------------------------------------------------------------------
// createBoundaryMask
// ---------------------------------------------------------------------------

Value createBoundaryMask(OpBuilder &b, Location loc, VectorType maskType,
                         ArrayRef<Value> globalDims,
                         ArrayRef<Value> baseOffsets) {
  unsigned rank = globalDims.size();
  SmallVector<Value> remaining;
  for (unsigned i = 0; i < rank; ++i) {
    remaining.push_back(
        arith::SubIOp::create(b, loc, globalDims[i], baseOffsets[i]));
  }
  return vector::CreateMaskOp::create(b, loc, maskType, remaining);
}

} // namespace llk
} // namespace mlir
