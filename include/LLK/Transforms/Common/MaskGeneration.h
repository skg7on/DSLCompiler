//===- MaskGeneration.h - Masked load/store for vector tails ---*- C++ -*-===//
//
// Utilities for generating vector.create_mask and vector.mask operations
// to handle non-aligned trailing elements after tiling.
//
//===----------------------------------------------------------------------===//

#ifndef LLK_TRANSFORMS_COMMON_MASKGENERATION_H
#define LLK_TRANSFORMS_COMMON_MASKGENERATION_H

#include <cstdint>

namespace mlir {
class OpBuilder;
class Location;
class Value;
namespace llk {

/// Create a vector mask for `numElements` lanes with the given vector width.
/// Returns a vector.create_mask op.
/// - numElements: number of valid elements (may be less than vectorWidth)
/// - vectorWidth: total vector width (e.g., 8 for AVX2 f32)
Value createVectorMask(OpBuilder &b, Location loc, int64_t numElements,
                       int64_t vectorWidth);

/// Wrap an operation in a vector.mask region for predicated execution.
/// This is a helper for generating masked load/store patterns.
///
/// Usage: the caller provides a callback that emits the masked body.
// void maskOperation(OpBuilder &b, Location loc, Value mask,
//                    llvm::function_ref<void(OpBuilder &, Location)> bodyFn);

} // namespace llk
} // namespace mlir

#endif // LLK_TRANSFORMS_COMMON_MASKGENERATION_H
