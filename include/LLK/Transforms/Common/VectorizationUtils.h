//===- VectorizationUtils.h - Generic vectorization utilities ---*- C++ -*-===//
//
// Common patterns for vectorizing linalg operations:
//   transfer_read → vector contract / arith → transfer_write
//
//===----------------------------------------------------------------------===//

#ifndef LLK_TRANSFORMS_COMMON_VECTORIZATIONUTILS_H
#define LLK_TRANSFORMS_COMMON_VECTORIZATIONUTILS_H

namespace mlir {
namespace llk {

/// Vector width constants for common ISAs.
struct VectorWidth {
  static constexpr int AVX2_F32 = 8;    ///< 256-bit / 32-bit = 8 lanes
  static constexpr int AVX512_F32 = 16; ///< 512-bit / 32-bit = 16 lanes
  static constexpr int NEON_F32 = 4;    ///< 128-bit / 32-bit = 4 lanes
  static constexpr int SVE_F32_MIN = 4; ///< Minimum SVE vector width
};

} // namespace llk
} // namespace mlir

#endif // LLK_TRANSFORMS_COMMON_VECTORIZATIONUTILS_H
