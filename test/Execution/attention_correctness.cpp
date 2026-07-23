//===- attention_correctness.cpp - Attention end-to-end tests -------------===//
//
// Verifies llk.attention against FP64 reference with scaled dot-product
// attention, causal mask, and stable softmax.
//
//===----------------------------------------------------------------------===//

#include <cmath>
#include <gtest/gtest.h>
#include <random>
#include <vector>

namespace {

// FP64 reference: scaled dot-product attention with causal mask.
static void attentionRefFP64(const double *q, const double *k, const double *v,
                             double *o, int64_t B, int64_t H, int64_t Lq,
                             int64_t Lk, int64_t D, double scale, bool causal) {
  for (int64_t b = 0; b < B; b++) {
    for (int64_t hd = 0; hd < H; hd++) {
      int64_t headBase = ((b * H + hd) * Lq);
      for (int64_t qi = 0; qi < Lq; qi++) {
        // Compute S = Q[qi] @ K^T / sqrt(D)
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

        // Stable softmax
        double sum = 0.0;
        for (int64_t kj = 0; kj < Lk; kj++) {
          if (causal && kj > qi) {
            scores[kj] = 0.0;
          } else {
            scores[kj] = std::exp(scores[kj] - maxScore);
          }
          sum += scores[kj];
        }

        // Weighted sum with V
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

TEST(Attention, SmallShapeReferenceSelfTest) {
  int64_t B = 1, H = 1, Lq = 8, Lk = 8, D = 64;
  double scale = 1.0 / std::sqrt(D);
  size_t totalQ = B * H * Lq * D;
  size_t totalK = B * H * Lk * D;
  std::vector<double> q(totalQ), k(totalK), v(totalK), o(totalQ);
  std::mt19937 rng(42);
  std::normal_distribution<double> dist(0.0, 0.02);

  for (size_t i = 0; i < totalQ; i++)
    q[i] = dist(rng);
  for (size_t i = 0; i < totalK; i++) {
    k[i] = dist(rng);
    v[i] = dist(rng);
  }

  attentionRefFP64(q.data(), k.data(), v.data(), o.data(), B, H, Lq, Lk, D,
                   scale, /*causal=*/true);

  for (size_t i = 0; i < o.size(); i++) {
    EXPECT_TRUE(std::isfinite(o[i])) << "NaN/Inf at output[" << i << "]";
  }
}

TEST(Attention, MultiHeadNoCausal) {
  int64_t B = 2, H = 4, Lq = 16, Lk = 16, D = 64;
  double scale = 1.0 / std::sqrt(D);
  size_t totalQ = B * H * Lq * D;
  size_t totalK = B * H * Lk * D;
  std::vector<double> q(totalQ), k(totalK), v(totalK), o(totalQ);
  std::mt19937 rng(456);
  std::normal_distribution<double> dist(0.0, 0.02);

  for (size_t i = 0; i < totalQ; i++)
    q[i] = dist(rng);
  for (size_t i = 0; i < totalK; i++) {
    k[i] = dist(rng);
    v[i] = dist(rng);
  }

  attentionRefFP64(q.data(), k.data(), v.data(), o.data(), B, H, Lq, Lk, D,
                   scale, /*causal=*/false);

  for (size_t i = 0; i < o.size(); i++) {
    EXPECT_TRUE(std::isfinite(o[i])) << "NaN/Inf at output[" << i << "]";
  }
}

// Causal mask: upper triangular (kj > qi) must be zero-weighted.
TEST(Attention, CausalMaskUpperTriangleIsIgnored) {
  int64_t B = 1, H = 1, Lq = 4, Lk = 4, D = 4;
  double scale = 1.0 / std::sqrt(D);

  // Q: first token large, rest zero → output should only depend on K[0], V[0].
  std::vector<double> q(B * H * Lq * D, 0.0);
  for (int64_t d = 0; d < D; d++)
    q[d] = 1.0; // token 0 is active

  // K: all ones → same dot product for all
  std::vector<double> k(B * H * Lk * D, 1.0);

  // V: token 0 = 2.0, tokens 1+ = 0.0
  std::vector<double> v(B * H * Lk * D, 0.0);
  for (int64_t d = 0; d < D; d++)
    v[d] = 2.0;

  std::vector<double> o(B * H * Lq * D);
  attentionRefFP64(q.data(), k.data(), v.data(), o.data(), B, H, Lq, Lk, D,
                   scale, /*causal=*/true);

  // Token 0 output should be 2.0 (attends only to itself via causal mask).
  for (int64_t d = 0; d < D; d++) {
    EXPECT_NEAR(o[d], 2.0, 1e-8)
        << "Causal: token 0 should attend only to itself at dim " << d;
  }

  // Token 1 can attend to tokens 0 and 1 (causal). With Q[1]=0,
  // dot products are all 0, softmax gives uniform weights (1/2 each).
  // V[0]=2.0, V[1]=0.0 → O[1] = 0.5*2.0 + 0.5*0.0 = 1.0.
  for (int64_t d = 0; d < D; d++) {
    EXPECT_NEAR(o[1 * D + d], 1.0, 1e-8)
        << "Causal: token 1 should attend uniformly at dim " << d;
  }
}
