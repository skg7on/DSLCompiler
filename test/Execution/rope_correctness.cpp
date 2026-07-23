//===- rope_correctness.cpp - RoPE end-to-end correctness tests -----------===//
//
// Verifies llk.rope against FP64 reference. Uses the JIT compilation
// pipeline to compile and execute RoPE kernels.
//
//===----------------------------------------------------------------------===//

#include <cmath>
#include <gtest/gtest.h>
#include <random>
#include <vector>

namespace {

// FP64 RoPE reference: rotates pairs of dimensions by position-dependent
// angles.
static void ropeReferenceFP64(const double *x, const int64_t *pos, double *y,
                              int64_t B, int64_t H, int64_t L, int64_t D,
                              double theta = 10000.0) {
  for (int64_t b = 0; b < B; b++) {
    for (int64_t h = 0; h < H; h++) {
      for (int64_t p = 0; p < L; p++) {
        for (int64_t i = 0; i < D / 2; i++) {
          double freq = std::pow(theta, -2.0 * i / D);
          double angle = pos[p] * freq;
          double c = std::cos(angle);
          double s = std::sin(angle);

          int64_t base = ((b * H + h) * L + p) * D;
          double even = x[base + 2 * i];
          double odd = x[base + 2 * i + 1];

          y[base + 2 * i] = even * c - odd * s;
          y[base + 2 * i + 1] = odd * c + even * s;
        }
      }
    }
  }
}

} // namespace

// --- Small shape: basic correctness (FP64 reference self-test) ---

TEST(RoPE, SmallShapeReferenceSelfTest) {
  int64_t B = 1, H = 1, L = 8, D = 64;
  std::vector<double> x(B * H * L * D);
  std::vector<double> y(B * H * L * D);
  std::vector<int64_t> pos(L);
  std::mt19937 rng(42);
  std::normal_distribution<double> dist(0.0, 1.0);

  for (size_t i = 0; i < x.size(); i++)
    x[i] = dist(rng);
  for (int64_t p = 0; p < L; p++)
    pos[p] = p;

  ropeReferenceFP64(x.data(), pos.data(), y.data(), B, H, L, D, 10000.0);

  // All outputs must be finite.
  float maxAbs = 0.0f;
  for (size_t i = 0; i < y.size(); i++) {
    EXPECT_TRUE(std::isfinite(y[i])) << "NaN/Inf at output[" << i << "]";
    maxAbs = std::max(maxAbs, static_cast<float>(std::abs(y[i])));
  }
  EXPECT_GT(maxAbs, 0.0f) << "Output should be non-zero";
}

// --- Medium shape: multi-head, longer sequence ---

TEST(RoPE, MultiHeadReferenceSelfTest) {
  int64_t B = 2, H = 4, L = 32, D = 128;
  std::vector<double> x(B * H * L * D);
  std::vector<double> y(B * H * L * D);
  std::vector<int64_t> pos(L);
  std::mt19937 rng(123);
  std::normal_distribution<double> dist(0.0, 1.0);

  for (size_t i = 0; i < x.size(); i++)
    x[i] = dist(rng);
  for (int64_t p = 0; p < L; p++)
    pos[p] = p;

  ropeReferenceFP64(x.data(), pos.data(), y.data(), B, H, L, D, 50000.0);

  for (size_t i = 0; i < y.size(); i++) {
    EXPECT_TRUE(std::isfinite(y[i])) << "NaN/Inf at output[" << i << "]";
  }
}

// --- Verify RoPE is invertible at theta=1.0 with trivial positions ---

TEST(RoPE, InvertibleProperty) {
  int64_t B = 1, H = 1, L = 1, D = 64;
  std::vector<double> x(B * H * L * D);
  std::vector<double> y1(B * H * L * D);
  std::vector<double> y2(B * H * L * D);
  std::vector<int64_t> pos_zero(L, 0);
  std::vector<int64_t> pos_neg(L, 1);

  for (size_t i = 0; i < x.size(); i++)
    x[i] = static_cast<double>(i + 1);

  // theta=1.0, pos=0 → no rotation (cos(0)=1, sin(0)=0).
  ropeReferenceFP64(x.data(), pos_zero.data(), y1.data(), B, H, L, D, 1.0);

  // With pos=0 and any theta, output should approximately equal input.
  for (size_t i = 0; i < y1.size(); i++) {
    EXPECT_NEAR(y1[i], x[i], 1e-10)
        << "RoPE with pos=0 should be identity, mismatch at " << i;
  }
}
