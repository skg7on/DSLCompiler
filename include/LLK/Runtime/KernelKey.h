//===- KernelKey.h - Deterministic cache key for compiled kernels ---------===//
//
// 13-field KernelKey uniquely identifies a compiled kernel variant.
// Also defines WeightKey for the packed-weight cache (L4).
//
//===----------------------------------------------------------------------===//

#ifndef LLK_RUNTIME_KERNELKEY_H
#define LLK_RUNTIME_KERNELKEY_H

#include <cstdint>
#include <cstring>
#include <functional>
#include <sstream>
#include <string>

namespace llk {

// ---------------------------------------------------------------------------
// Runtime enums (separate from MLIR dialect enums to avoid MLIR dependency)
// ---------------------------------------------------------------------------

enum class DType : uint8_t { BF16 = 0, FP16 = 1, FP32 = 2 };

enum class CpuIsa : uint8_t {
  AVX2 = 0,
  AVX512_BF16 = 1,
  AVX512_VNNI = 2,
  AMX_BF16 = 3,
  NEON = 4,
  SVE = 5,
};

enum class OperationKind : uint8_t { FusedSwiGLU = 0, RoPE = 1, Attention = 2 };

enum class Layout : uint8_t { RowMajor = 0, PackedKN = 1 };

enum class MathMode : uint8_t { Strict = 0, BoundedFast = 1, UnsafeFast = 2 };

// ---------------------------------------------------------------------------
// KernelKey: 13-field deterministic identifier for a compiled kernel variant.
// ---------------------------------------------------------------------------

struct KernelKey {
  OperationKind operation{OperationKind::FusedSwiGLU};
  DType input_type{DType::BF16};
  DType output_type{DType::BF16};
  int64_t M_bucket{0}; // 0-4, or -1 for generic
  int64_t N{0};        // exact
  int64_t K{0};        // exact
  Layout weight_layout{Layout::RowMajor};
  CpuIsa isa{CpuIsa::AVX2};
  MathMode math_mode{MathMode::BoundedFast};
  uint16_t thread_count{1};
  uint64_t compiler_version{0};
  uint64_t schedule_version{0};

  /// 64-bit hash combining all fields via FNV-1a-like mixing.
  uint64_t hash() const {
    // Mix each field using splitmix64-style finalizer to get good avalanche.
    auto mix = [](uint64_t x) -> uint64_t {
      x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
      x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
      x = x ^ (x >> 31);
      return x;
    };

    uint64_t h = 0x9ae16a3b2f90404fULL; // random seed
    auto feed = [&](uint64_t v) { h = mix(h ^ v); };

    feed(static_cast<uint8_t>(operation));
    feed(static_cast<uint8_t>(input_type));
    feed(static_cast<uint8_t>(output_type));
    feed(static_cast<uint64_t>(M_bucket));
    feed(static_cast<uint64_t>(N));
    feed(static_cast<uint64_t>(K));
    feed(static_cast<uint8_t>(weight_layout));
    feed(static_cast<uint8_t>(isa));
    feed(static_cast<uint8_t>(math_mode));
    feed(static_cast<uint64_t>(thread_count));
    feed(compiler_version);
    feed(schedule_version);

    return h;
  }

  /// Human-readable representation for logging/debugging.
  std::string toString() const {
    std::ostringstream oss;
    oss << "KernelKey{op=" << static_cast<int>(operation)
        << " dtype_in=" << static_cast<int>(input_type)
        << " dtype_out=" << static_cast<int>(output_type)
        << " M_bucket=" << M_bucket << " N=" << N << " K=" << K
        << " layout=" << static_cast<int>(weight_layout)
        << " isa=" << static_cast<int>(isa)
        << " math=" << static_cast<int>(math_mode)
        << " threads=" << thread_count << " compiler=" << compiler_version
        << " schedule=" << schedule_version << "}";
    return oss.str();
  }

  bool operator==(const KernelKey &other) const {
    return memcmp(this, &other, sizeof(KernelKey)) == 0;
  }

  bool operator!=(const KernelKey &other) const { return !(*this == other); }
};

// ---------------------------------------------------------------------------
// WeightKey: identifies a packed-weight buffer (L4 cache).
// ---------------------------------------------------------------------------

struct WeightKey {
  const void *weight_ptr{nullptr};
  uint64_t shape_hash{0};
  int64_t BK{0};
  int64_t BN{0};

  bool operator==(const WeightKey &other) const {
    return weight_ptr == other.weight_ptr && shape_hash == other.shape_hash &&
           BK == other.BK && BN == other.BN;
  }
};

} // namespace llk

// ---------------------------------------------------------------------------
// std::hash specializations for unordered_map keys
// ---------------------------------------------------------------------------

template <> struct std::hash<llk::KernelKey> {
  size_t operator()(const llk::KernelKey &k) const { return k.hash(); }
};

template <> struct std::hash<llk::WeightKey> {
  size_t operator()(const llk::WeightKey &w) const {
    return reinterpret_cast<size_t>(w.weight_ptr) ^
           static_cast<size_t>(w.shape_hash) ^ static_cast<size_t>(w.BK) ^
           (static_cast<size_t>(w.BN) << 16);
  }
};

#endif // LLK_RUNTIME_KERNELKEY_H
