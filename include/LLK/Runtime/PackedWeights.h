#ifndef LLK_RUNTIME_PACKEDWEIGHTS_H
#define LLK_RUNTIME_PACKEDWEIGHTS_H

#include "LLK/Runtime/JitCache.h"

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>

enum class DType : uint8_t { BF16 = 0, FP16 = 1, FP32 = 2 };

struct PackedWeights {
  void *data = nullptr;
  int64_t K_blocks = 0;
  int64_t N = 0;
  int64_t block_size = 0; // BK * N * element_size
  int64_t BK = 0;
  DType dtype = DType::BF16;

  PackedWeights() = default;
  ~PackedWeights() { free(data); }
  PackedWeights(const PackedWeights &) = delete;
  PackedWeights &operator=(const PackedWeights &) = delete;
  PackedWeights(PackedWeights &&other) noexcept
      : data(other.data), K_blocks(other.K_blocks), N(other.N),
        block_size(other.block_size), BK(other.BK), dtype(other.dtype) {
    other.data = nullptr;
  }
  PackedWeights &operator=(PackedWeights &&other) noexcept {
    if (this != &other) {
      free(data);
      data = other.data;
      K_blocks = other.K_blocks;
      N = other.N;
      block_size = other.block_size;
      BK = other.BK;
      dtype = other.dtype;
      other.data = nullptr;
    }
    return *this;
  }
};

// Pack a single weight matrix from row-major to block-major layout
PackedWeights packWeightMatrix(const Tensor2D &W, int64_t BK);

// Repack when weights change
void repack(PackedWeights &pw, const Tensor2D &W);

#endif
