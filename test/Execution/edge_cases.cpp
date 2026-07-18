//===- edge_cases.cpp - Edge case tests for SwiGLU computation ------------===//
//
// Tests edge cases: unit dimensions, K=1 (no reduction), zero dimensions.
// Standalone — no MLIR dependencies, tests reference algorithm directly.
//
//===----------------------------------------------------------------------===//

#include <gtest/gtest.h>

#include <cmath>
#include <cstring>
#include <vector>

// ---------------------------------------------------------------------------
// FP64 reference — local copy for edge case testing
// ---------------------------------------------------------------------------

static void swiglu_reference_fp64(const double* x, const double* wg,
                                  const double* wu, double* y, int64_t M,
                                  int64_t N, int64_t K) {
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

TEST(EdgeCases, UnitDimensions) {
  // M=1, N=1, K=1 — minimal work, single element
  std::vector<double> x = {2.0}, wg = {0.5}, wu = {0.3}, y(1);
  swiglu_reference_fp64(x.data(), wg.data(), wu.data(), y.data(), 1, 1, 1);

  EXPECT_TRUE(std::isfinite(y[0])) << "Unit output should be finite";
  // x=[2.0], wg=[0.5], wu=[0.3]
  // gate = 2.0*0.5 = 1.0, SiLU(1.0) = 1.0/(1+exp(-1)) ≈ 0.7310586
  // up = 2.0*0.3 = 0.6
  // result = 0.7310586 * 0.6 ≈ 0.438635
  double expected =
      (1.0 / (1.0 + std::exp(-1.0))) * (2.0 * 0.3);
  EXPECT_NEAR(y[0], expected, 1e-12);
}

TEST(EdgeCases, K_equals_1) {
  // No reduction to accumulate over — single multiply per output element
  int64_t M = 3, N = 2, K = 1;
  std::vector<double> x(M * K, 1.0), wg(K * N, 0.5), wu(K * N, 0.3),
      y(M * N);
  swiglu_reference_fp64(x.data(), wg.data(), wu.data(), y.data(), M, N, K);

  // Each output element: gate = 1.0*0.5 = 0.5, up = 1.0*0.3 = 0.3
  // SiLU(0.5) = 0.5/(1+exp(-0.5)) ≈ 0.5/1.6065 ≈ 0.3112
  // result = 0.3112 * 0.3 ≈ 0.09337
  for (int64_t i = 0; i < M * N; i++) {
    EXPECT_TRUE(std::isfinite(y[i]))
        << "Output[" << i << "] should be finite";
  }

  // All outputs should be identical since inputs are broadcasted
  for (int64_t i = 1; i < M * N; i++) {
    EXPECT_DOUBLE_EQ(y[i], y[0])
        << "All outputs should be identical for K=1 with uniform inputs";
  }
}

TEST(EdgeCases, ZeroDim_ExpectedError) {
  // M=0 would produce an empty output and no work.
  // The LLK dialect verifier should reject M=0 (or any zero-dim tensor).
  // This test validates that the reference algorithm does not crash on
  // zero-element output, but the dialect-level enforcement is tested
  // separately via FileCheck in ops_invalid.mlir.

  // Reference: 0 elements is valid — produces empty output
  std::vector<double> x, wg(4 * 8, 0.5), wu(4 * 8, 0.3), y;
  swiglu_reference_fp64(x.data(), wg.data(), wu.data(), y.data(), 0, 8, 4);
  EXPECT_EQ(y.size(), 0u) << "M=0 should produce empty output";

  SUCCEED();
}

TEST(EdgeCases, NegativeInputs) {
  // SiLU is well-defined for negative inputs (asymptotically 0 for large
  // negatives, linear for large positives)
  int64_t M = 1, N = 4, K = 2;
  std::vector<double> x = {-2.0, 3.0};          // M*K = 2
  std::vector<double> wg = {0.5, -1.0, 0.0, 2.0, -3.0, 0.5, 1.0, 0.0};  // K*N = 8
  std::vector<double> wu = {1.0, 0.5, -0.5, 0.0, 2.0, -1.0, 0.0, -2.0};
  std::vector<double> y(N);

  swiglu_reference_fp64(x.data(), wg.data(), wu.data(), y.data(), M, N, K);

  for (int64_t i = 0; i < N; i++) {
    EXPECT_TRUE(std::isfinite(y[i]))
        << "Output[" << i << "] should be finite with negative inputs";
  }
}

TEST(EdgeCases, LargeValues) {
  // Large positive inputs: SiLU(g) ≈ g for large g
  // Large negative inputs: SiLU(g) ≈ 0 for large negative g
  int64_t M = 1, N = 2, K = 2;
  std::vector<double> x = {10.0, -10.0};
  std::vector<double> wg = {1.0, 0.0, 0.0, 1.0};  // Identity-like
  std::vector<double> wu = {1.0, 0.0, 0.0, 1.0};
  std::vector<double> y(N);

  swiglu_reference_fp64(x.data(), wg.data(), wu.data(), y.data(), M, N, K);

  // gate[0] = 10*1 + (-10)*0 = 10, SiLU(10) ≈ 10/(1+exp(-10)) ≈ 9.9995
  // up[0] = 10*1 + (-10)*0 = 10, result ≈ 99.995
  EXPECT_GT(y[0], 90.0) << "Large positive gate should produce large output";
  EXPECT_TRUE(std::isfinite(y[0]));

  // gate[1] = 10*0 + (-10)*1 = -10, SiLU(-10) ≈ -10/(1+exp(10)) ≈ -0.00045
  // up[1] = 10*0 + (-10)*1 = -10, result ≈ 0.0045
  EXPECT_LT(std::abs(y[1]), 1.0)
      << "Large negative gate should produce near-zero output";
  EXPECT_TRUE(std::isfinite(y[1]));
}

TEST(EdgeCases, AllZeros) {
  // Zeros everywhere: SiLU(0) = 0, all outputs = 0
  int64_t M = 2, N = 3, K = 4;
  std::vector<double> x(M * K, 0.0), wg(K * N, 0.0), wu(K * N, 0.0),
      y(M * N);

  swiglu_reference_fp64(x.data(), wg.data(), wu.data(), y.data(), M, N, K);

  for (int64_t i = 0; i < M * N; i++) {
    EXPECT_DOUBLE_EQ(y[i], 0.0) << "All-zero inputs should produce zero output";
  }
}
