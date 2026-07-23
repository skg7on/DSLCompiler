//===- TilingUtils.h - Generic tile-size computation -------------*- C++
//-*-===//
//
// Common utilities for computing tile sizes and constructing loop nests
// from schedule parameters. Used by all three kernel types.
//
//===----------------------------------------------------------------------===//

#ifndef LLK_TRANSFORMS_COMMON_TILINGUTILS_H
#define LLK_TRANSFORMS_COMMON_TILINGUTILS_H

#include <cstdint>

namespace mlir {
namespace llk {

/// Tile size parameters derived from schedule selection.
struct TileParams {
  int64_t BM{0}; ///< Tile size for M / batch dimension
  int64_t BN{0}; ///< Tile size for N / column dimension
  int64_t BK{0}; ///< Tile size for K / reduction dimension
};

/// Compute the number of tiles along a dimension.
/// Handles boundary cases where the dimension is not evenly divisible.
inline int64_t numTiles(int64_t dimSize, int64_t tileSize) {
  if (tileSize == 0)
    return 1;
  return (dimSize + tileSize - 1) / tileSize;
}

/// Return the size of tile `tileIdx` along a dimension of size `dimSize`
/// tiled with size `tileSize`. The last tile may be smaller.
inline int64_t tileSize(int64_t dimSize, int64_t tileSize, int64_t tileIdx) {
  int64_t start = tileIdx * tileSize;
  if (start >= dimSize)
    return 0;
  int64_t remaining = dimSize - start;
  return remaining < tileSize ? remaining : tileSize;
}

} // namespace llk
} // namespace mlir

#endif // LLK_TRANSFORMS_COMMON_TILINGUTILS_H
