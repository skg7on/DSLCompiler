//===- schedule_regression.cpp - Performance regression baseline ---------===//
//
// Records baseline GFLOPS per M-bucket for schedule regression detection.
// CI should fail if GFLOPS drops >5% below baseline for any bucket.
//
//===----------------------------------------------------------------------===//

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <vector>

// Baseline GFLOPS thresholds per M-bucket (placeholder values;
// actual baselines should be recorded from the tuning harness).
struct Baseline {
  int64_t M_bucket;
  double min_gflops;
};

// These thresholds are intentionally low for initial check-in.
// They should be tightened once llk-bench produces real numbers.
static const Baseline kBaselines[] = {
    {0, 0.5},  // M=1:    at least 0.5 GFLOPS (GEMV-like)
    {1, 1.0},  // M=2-4:  at least 1.0 GFLOPS
    {2, 2.0},  // M=5-16: at least 2.0 GFLOPS
    {3, 5.0},  // M=17-64: at least 5.0 GFLOPS
    {4, 10.0}, // M>=65:   at least 10.0 GFLOPS (GEMM-like)
};

TEST(ScheduleRegression, BaselineThresholds_Exist) {
  // Verify we have thresholds for all 5 buckets.
  EXPECT_EQ(sizeof(kBaselines) / sizeof(kBaselines[0]), 5);
}

TEST(ScheduleRegression, Thresholds_AreMonotonicallyIncreasing) {
  double prev = 0.0;
  for (auto &b : kBaselines) {
    EXPECT_GE(b.min_gflops, prev) << "M_bucket " << b.M_bucket << " threshold "
                                  << b.min_gflops << " < previous " << prev;
    prev = b.min_gflops;
  }
}

TEST(ScheduleRegression, Thresholds_ArePositive) {
  for (auto &b : kBaselines) {
    EXPECT_GT(b.min_gflops, 0.0)
        << "M_bucket " << b.M_bucket << " has non-positive threshold";
  }
}
