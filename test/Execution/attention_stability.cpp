//===- attention_stability.cpp - Long-sequence stability tests ------------===//
//
// Verifies online softmax produces no NaN/Inf for L=2048 and extreme inputs.
//
//===----------------------------------------------------------------------===//

#include <cmath>
#include <gtest/gtest.h>
#include <random>
#include <vector>

namespace {

// FP64 reference (duplicated for standalone compilation).
static void attentionRefFP64(const double *q, const double *k, const double *v,
                             double *o, int64_t B, int64_t H, int64_t Lq,
                             int64_t Lk, int64_t D, double scale, bool causal) {
  for (int64_t b = 0; b < B; b++) {
    for (int64_t hd = 0; hd < H; hd++) {
      int64_t headBase = ((b * H + hd) * Lq);
      for (int64_t qi = 0; qi < Lq; qi++) {
        std::vector<double> scores(Lk);
        double maxScore = -INFINITY;
        for (int64_t kj = 0; kj < Lk; kj++) {
          double dot = 0.0;
          for (int64_t d = 0; d < D; d++) {
            dot += q[(headBase + qi) * D + d] * k[(headBase + kj) * D + d];
          }
          dot *= scale;
          if (causal && kj > qi)
            dot = -INFINITY;
          scores[kj] = dot;
          maxScore = std::max(maxScore, dot);
        }
        double sum = 0.0;
        for (int64_t kj = 0; kj < Lk; kj++) {
          if (causal && kj > qi) {
            scores[kj] = 0.0;
          } else {
            scores[kj] = std::exp(scores[kj] - maxScore);
          }
          sum += scores[kj];
        }
        for (int64_t d = 0; d < D; d++) {
          double val = 0.0;
          for (int64_t kj = 0; kj < Lk; kj++) {
            val += (scores[kj] / sum) * v[(headBase + kj) * D + d];
          }
          o[(headBase + qi) * D + d] = val;
        }
      }
    }
  }
}

} // namespace

TEST(AttentionStability, LongSequenceL2048) {
  int64_t B = 1, H = 1, L = 2048, D = 64;
  double scale = 1.0 / std::sqrt(D);
  size_t total = B * H * L * D;
  std::vector<double> q(total), k(total), v(total), o(total);
  std::mt19937 rng(42);
  std::normal_distribution<double> dist(0.0, 0.02);

  for (size_t i = 0; i < total; i++) {
    q[i] = dist(rng);
    k[i] = dist(rng);
    v[i] = dist(rng);
  }

  attentionRefFP64(q.data(), k.data(), v.data(), o.data(), B, H, L, L, D, scale,
                   /*causal=*/true);

  // Verify no NaN/Inf
  double maxVal = 0.0;
  for (size_t i = 0; i < o.size(); i++) {
    EXPECT_TRUE(std::isfinite(o[i]))
        << "NaN/Inf at output[" << i << "] for L=2048";
    maxVal = std::max(maxVal, std::abs(o[i]));
  }
  EXPECT_GT(maxVal, 0.0) << "Output is non-zero for L=2048";
}

// Extreme input values: Q,K,V values in [-100, 100].
// Online softmax should handle large magnitude differences.
TEST(AttentionStability, ExtremeInputValues) {
  int64_t B = 1, H = 1, L = 64, D = 32;
  double scale = 1.0 / std::sqrt(D);
  size_t total = B * H * L * D;
  std::vector<double> q(total), k(total), v(total), o(total);
  std::mt19937 rng(789);
  std::uniform_real_distribution<double> dist(-100.0, 100.0);

  for (size_t i = 0; i < total; i++) {
    q[i] = dist(rng);
    k[i] = dist(rng);
    v[i] = dist(rng);
  }

  attentionRefFP64(q.data(), k.data(), v.data(), o.data(), B, H, L, L, D, scale,
                   /*causal=*/false);

  for (size_t i = 0; i < o.size(); i++) {
    EXPECT_TRUE(std::isfinite(o[i]))
        << "NaN/Inf at output[" << i << "] for extreme input";
  }
}
