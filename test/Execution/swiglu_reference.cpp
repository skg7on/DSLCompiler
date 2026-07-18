//===- swiglu_reference.cpp - Naive C++ SwiGLU reference implementation ---===//
//
// Standalone reference: FP32 naive triple-loop matmul + SiLU, plus FP64
// ground truth for error measurement. No MLIR dependencies.
//
//===----------------------------------------------------------------------===//

#include <gtest/gtest.h>

#include <cmath>
#include <cstring>
#include <random>
#include <vector>

// ---------------------------------------------------------------------------
// BF16 conversion helpers (simulated with float)
// ---------------------------------------------------------------------------

static float bf16_to_f32(uint16_t bf16) {
  uint32_t bits = static_cast<uint32_t>(bf16) << 16;
  float f;
  std::memcpy(&f, &bits, sizeof(f));
  return f;
}

static uint16_t f32_to_bf16(float f) {
  uint32_t bits;
  std::memcpy(&bits, &f, sizeof(bits));
  // Round to nearest even
  bits += 0x8000 + ((bits >> 16) & 1);
  return static_cast<uint16_t>(bits >> 16);
}

// ---------------------------------------------------------------------------
// Naive SwiGLU: Y = SiLU(X @ Wg) * (X @ Wu)   (FP32 arithmetic)
// ---------------------------------------------------------------------------

static void swiglu_reference(const float* x, const float* wg, const float* wu,
                             float* y, int64_t M, int64_t N, int64_t K) {
  std::vector<float> gate(M * N, 0.0f);
  std::vector<float> up(M * N, 0.0f);

  // gate = X @ Wg
  for (int64_t m = 0; m < M; m++) {
    for (int64_t n = 0; n < N; n++) {
      float sum = 0.0f;
      for (int64_t k = 0; k < K; k++) {
        sum += x[m * K + k] * wg[k * N + n];
      }
      gate[m * N + n] = sum;
    }
  }

  // up = X @ Wu
  for (int64_t m = 0; m < M; m++) {
    for (int64_t n = 0; n < N; n++) {
      float sum = 0.0f;
      for (int64_t k = 0; k < K; k++) {
        sum += x[m * K + k] * wu[k * N + n];
      }
      up[m * N + n] = sum;
    }
  }

  // Y = silu(gate) * up
  for (int64_t i = 0; i < M * N; i++) {
    float g = gate[i];
    float silu = g / (1.0f + std::exp(-g));
    y[i] = silu * up[i];
  }
}

// ---------------------------------------------------------------------------
// FP64 reference for error measurement
// ---------------------------------------------------------------------------

void swiglu_reference_fp64(const double* x, const double* wg, const double* wu,
                           double* y, int64_t M, int64_t N, int64_t K) {
  std::vector<double> gate(M * N, 0.0);
  std::vector<double> up(M * N, 0.0);

  for (int64_t m = 0; m < M; m++) {
    for (int64_t n = 0; n < N; n++) {
      double sum_g = 0.0, sum_u = 0.0;
      for (int64_t k = 0; k < K; k++) {
        sum_g += x[m * K + k] * wg[k * N + n];
        sum_u += x[m * K + k] * wu[k * N + n];
      }
      gate[m * N + n] = sum_g;
      up[m * N + n] = sum_u;
    }
  }

  for (int64_t i = 0; i < M * N; i++) {
    double g = gate[i];
    double silu = g / (1.0 + std::exp(-g));
    y[i] = silu * up[i];
  }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

TEST(SwiGLURef, SmallShapes) {
  struct Shape {
    int64_t M, N, K;
  };
  std::vector<Shape> shapes = {
      {1, 4, 2},
      {2, 3, 4},
      {4, 8, 16},
  };

  for (auto [M, N, K] : shapes) {
    std::vector<float> x(M * K), wg(K * N), wu(K * N), y(M * N);
    std::vector<double> xd(M * K), wgd(K * N), wud(K * N), yd(M * N);

    std::mt19937 rng(42);
    std::normal_distribution<float> dist(0.0f, 1.0f);

    for (size_t i = 0; i < x.size(); i++) {
      x[i] = dist(rng);
      xd[i] = x[i];
    }
    for (size_t i = 0; i < wg.size(); i++) {
      wg[i] = dist(rng);
      wgd[i] = wg[i];
      wu[i] = dist(rng);
      wud[i] = wu[i];
    }

    swiglu_reference(x.data(), wg.data(), wu.data(), y.data(), M, N, K);
    swiglu_reference_fp64(xd.data(), wgd.data(), wud.data(), yd.data(), M, N,
                          K);

    float max_err = 0.0f;
    for (int64_t i = 0; i < M * N; i++) {
      max_err = std::max(max_err, std::abs(y[i] - static_cast<float>(yd[i])));
    }
    EXPECT_LT(max_err, 1e-2f) << "M=" << M << " N=" << N << " K=" << K;
  }
}

TEST(SwiGLURef, BF16RoundTrip) {
  // Verify BF16 conversion helpers are consistent
  for (float v : {-3.14f, -1.0f, 0.0f, 1.0f, 3.14f, 42.0f}) {
    uint16_t bf16 = f32_to_bf16(v);
    float recovered = bf16_to_f32(bf16);
    // BF16 has ~3 decimal digits of precision
    EXPECT_NEAR(recovered, v, 1e-2f * std::max(1.0f, std::abs(v)))
        << "BF16 round-trip failed for " << v;
  }
}

TEST(SwiGLURef, KnownValues) {
  // Simple known-value test: X=[1], Wg=[2], Wu=[3], M=1,N=1,K=1
  // gate = 1*2 = 2, up = 1*3 = 3
  // SiLU(2) = 2/(1+exp(-2)) = 2/(1+0.1353) = 2/1.1353 ≈ 1.7616
  // result = 1.7616 * 3 ≈ 5.2847
  float x[] = {1.0f};
  float wg[] = {2.0f};
  float wu[] = {3.0f};
  float y = 0.0f;
  swiglu_reference(x, wg, wu, &y, 1, 1, 1);

  double xd[] = {1.0}, wgd[] = {2.0}, wud[] = {3.0}, yd = 0.0;
  swiglu_reference_fp64(xd, wgd, wud, &yd, 1, 1, 1);

  double expected = (2.0 / (1.0 + std::exp(-2.0))) * 3.0;
  EXPECT_NEAR(yd, expected, 1e-12)
      << "FP64 reference should match known values exactly";
  EXPECT_LT(std::abs(y - static_cast<float>(yd)), 1e-4f)
      << "FP32 should be within tolerance of FP64";
}
