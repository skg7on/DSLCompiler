#include "LLK/Runtime/PackedWeights.h"

#include <algorithm>

struct Tensor2D {
  void *data;
  int64_t dim0, dim1;
  int64_t stride0, stride1;
};

PackedWeights packWeightMatrix(const Tensor2D &W, int64_t BK) {
  PackedWeights pw;
  pw.N = W.dim1;
  pw.BK = BK;
  pw.K_blocks = (W.dim0 + BK - 1) / BK;
  pw.dtype = DType::BF16;

  int64_t elemSize = 2; // BF16
  pw.block_size = BK * pw.N * elemSize;
  size_t totalSize = pw.K_blocks * pw.block_size;
  pw.data = malloc(totalSize);
  memset(pw.data, 0, totalSize);

  // Copy in block-major order
  uint8_t *dst = static_cast<uint8_t *>(pw.data);
  const uint8_t *src = static_cast<const uint8_t *>(W.data);
  int64_t srcRowStride = W.stride0 * elemSize;

  for (int64_t kb = 0; kb < pw.K_blocks; kb++) {
    int64_t kStart = kb * BK;
    int64_t rowsInBlock = std::min(BK, W.dim0 - kStart);
    for (int64_t ki = 0; ki < rowsInBlock; ki++) {
      int64_t k = kStart + ki;
      memcpy(dst + kb * pw.block_size + ki * pw.N * elemSize,
             src + k * srcRowStride, pw.N * elemSize);
    }
  }
  return pw;
}

void repack(PackedWeights &pw, const Tensor2D &W) {
  pw = packWeightMatrix(W, pw.BK);
}
