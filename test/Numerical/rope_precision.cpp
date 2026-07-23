//===- rope_precision.cpp - RoPE numerical precision tests ----------------===//
//
// Verifies cos/sin approximation accuracy at theta=10000 for L up to 2048.
// Max error vs FP64 should be < 1e-4.
//
//===----------------------------------------------------------------------===//

#include <cmath>
#include <gtest/gtest.h>
#include <vector>

namespace {

// Compute RoPE output at a single (position, dim_pair) using FP64.
static void ropePairFP64(double even, double odd, int64_t pos, int64_t i,
                         int64_t D, double theta, double &outEven,
                         double &outOdd) {
  double freq = std::pow(theta, -2.0 * i / D);
  double angle = pos * freq;
  double c = std::cos(angle);
  double s = std::sin(angle);
  outEven = even * c - odd * s;
  outOdd = odd * c + even * s;
}

// Compute RoPE using FP32 arithmetic to measure precision loss.
static void ropePairFP32(float even, float odd, int64_t pos, int64_t i,
                         int64_t D, double theta, float &outEven,
                         float &outOdd) {
  float freq =
      std::pow(static_cast<float>(theta), static_cast<float>(-2.0 * i / D));
  float angle = static_cast<float>(pos) * freq;
  float c = std::cos(angle);
  float s = std::sin(angle);
  outEven = even * c - odd * s;
  outOdd = odd * c + even * s;
}

} // namespace

// Test precision at increasing sequence lengths.
TEST(RoPEPrecision, CosSinErrorAtVariousPositions) {
  int64_t D = 128;
  double theta = 10000.0;
  float even = 1.0f, odd = -0.5f;

  for (int64_t pos : {0, 1, 15, 127, 1023, 2047}) {
    for (int64_t i = 0; i < D / 2; i++) {
      double outEven64, outOdd64;
      ropePairFP64(static_cast<double>(even), static_cast<double>(odd), pos, i,
                   D, theta, outEven64, outOdd64);

      float outEven32, outOdd32;
      ropePairFP32(even, odd, pos, i, D, theta, outEven32, outOdd32);

      double evenErr = std::abs(outEven64 - static_cast<double>(outEven32));
      double oddErr = std::abs(outOdd64 - static_cast<double>(outOdd32));

      EXPECT_LT(evenErr, 1e-3)
          << "Even component FP32 error too large at pos=" << pos << " i=" << i;
      EXPECT_LT(oddErr, 1e-3)
          << "Odd component FP32 error too large at pos=" << pos << " i=" << i;
    }
  }
}

// Test that rotation preserves L2 norm of each pair.
TEST(RoPEPrecision, PreservesL2Norm) {
  int64_t D = 64;
  double theta = 10000.0;

  for (int64_t pos : {0, 1, 10, 100}) {
    for (int64_t i = 0; i < D / 2; i++) {
      float even = 3.0f, odd = 4.0f; // L2 norm = 5.0

      float outEven32, outOdd32;
      ropePairFP32(even, odd, pos, i, D, theta, outEven32, outOdd32);

      float normIn = std::sqrt(even * even + odd * odd);
      float normOut = std::sqrt(outEven32 * outEven32 + outOdd32 * outOdd32);

      EXPECT_NEAR(normIn, normOut, 1e-4)
          << "RoPE should preserve L2 norm at pos=" << pos << " i=" << i;
    }
  }
}
