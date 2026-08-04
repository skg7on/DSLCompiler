//===- VectorizationUtils.h - Generic vectorization utilities ---*- C++ -*-===//
//
// Common patterns for vectorizing linalg operations:
//   transfer_read → vector contract / arith → transfer_write
//
//===----------------------------------------------------------------------===//

#ifndef LLK_TRANSFORMS_COMMON_VECTORIZATIONUTILS_H
#define LLK_TRANSFORMS_COMMON_VECTORIZATIONUTILS_H

#include "mlir/IR/Builders.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir {
namespace llk {

/// Vector width constants for common ISAs.
struct VectorWidth {
  static constexpr int AVX2_F32 = 8;    ///< 256-bit / 32-bit = 8 lanes
  static constexpr int AVX512_F32 = 16; ///< 512-bit / 32-bit = 16 lanes
  static constexpr int NEON_F32 = 4;    ///< 128-bit / 32-bit = 4 lanes
  static constexpr int SVE_F32_MIN = 4; ///< Minimum SVE vector width
};

//===----------------------------------------------------------------------===//
// Convenience builders for vector.transfer_read / vector.transfer_write
//===----------------------------------------------------------------------===//

/// Build zero-valued index constants for a given rank.
/// Returns a SmallVector of arith::ConstantOp results (index 0 x N).
SmallVector<Value> buildZeroIndices(OpBuilder &b, Location loc, unsigned rank);

/// Build an all-true in_bounds array for a transfer op (rank copies of true).
SmallVector<bool> getAllInBounds(unsigned rank);

/// Create a vector.transfer_read with identity permutation map, zero padding,
/// and optional boundary mask.  The caller provides the memref, zero-based
/// indices, target vector type, padding value (default: zero), and mask.
Value createTransferRead(OpBuilder &b, Location loc, Value memref,
                         ValueRange indices, VectorType vectorType,
                         Value pad = Value(), Value mask = Value(),
                         ArrayRef<bool> inBounds = {});

/// Create a vector.transfer_write with identity permutation map and optional
/// boundary mask.  The caller provides the vector, memref, zero-based indices,
/// and mask.
void createTransferWrite(OpBuilder &b, Location loc, Value vector, Value memref,
                         ValueRange indices, Value mask = Value(),
                         ArrayRef<bool> inBounds = {});

/// Create a vector.create_mask for boundary masking: for each dimension i,
/// mask[i] = (index < globalDim[i] - baseOffset[i]).  This is the canonical
/// tail-tile masking pattern used by block-pointer lowering and tiling passes.
Value createBoundaryMask(OpBuilder &b, Location loc, VectorType maskType,
                         ArrayRef<Value> globalDims,
                         ArrayRef<Value> baseOffsets);

} // namespace llk
} // namespace mlir

#endif // LLK_TRANSFORMS_COMMON_VECTORIZATIONUTILS_H
