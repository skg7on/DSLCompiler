//===- llk-tune.cpp - LLK autotuning grid search driver -------------------===//
//
// Generates candidate tile configurations, filters by L1 cache capacity,
// and would measure each to find the best-performing schedule for a given
// shape regime. Integrates with schedule_db.json for persistence.
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace cl = llvm::cl;

static cl::opt<int64_t> tuneM("M", cl::desc("M dimension (rows)"),
                              cl::init(128));
static cl::opt<int64_t> tuneN("N", cl::desc("N dimension (hidden)"),
                              cl::init(4096));
static cl::opt<int64_t> tuneK("K", cl::desc("K dimension (input)"),
                              cl::init(4096));
static cl::opt<std::string> tuneOutput("o", cl::desc("Output JSON file"),
                                       cl::init("tune_result.json"));
static cl::opt<bool> tuneDryRun("dry-run",
                                cl::desc("Generate configs without measuring"),
                                cl::init(false));

/// A single tuning configuration with tile sizes and parallelism knobs.
struct TuningConfig {
  int64_t BM, BN, BK;
  int64_t VM, VN;
  int64_t vector_width;
  int64_t num_threads;
  int64_t grain_size;
  std::string parallel_axis;
  double gflops = 0.0; // measured (0 = not measured yet)

  /// Estimated L1 footprint of the working set for this config (bytes).
  /// X tile: BM * BK * 2 (BF16), 2 weight tiles: 2 * BK * BN * 2 (BF16).
  size_t l1Footprint() const {
    return (BM * BK + 2 * BK * BN) * 2; // bytes
  }
};

/// M bucket classifier (matches ShapeSpecialization pass).
static int classifyM(int64_t M) {
  if (M == 1)
    return 0;
  if (M <= 4)
    return 1;
  if (M <= 16)
    return 2;
  if (M <= 64)
    return 3;
  return 4;
}

/// Generate candidate configs for a given shape regime.
/// Filters by L1 cache capacity (80% of 32KB for the working set).
static std::vector<TuningConfig> generateConfigs(int64_t M, int64_t N,
                                                 int64_t K) {
  std::vector<TuningConfig> configs;
  constexpr size_t l1Limit = static_cast<size_t>(32 * 1024 * 0.8); // 25.6 KB

  int M_bucket = classifyM(M);

  for (int64_t BM : {1, 4, 8, 16, 32, 64}) {
    // GEMV-like: only BM=1
    if (M_bucket == 0 && BM != 1)
      continue;
    // Large batch: skip BM=1 (too small)
    if (M_bucket >= 3 && BM <= 4)
      continue;

    for (int64_t BN : {16, 32, 64, 128, 256}) {
      for (int64_t BK : {32, 64, 128, 256}) {
        for (int64_t VM : {1, 2, 4}) {
          for (int64_t VN : {4, 8}) {
            // Vector width currently fixed at 8 (AVX2 BF16 → 8 x 16-bit)
            int64_t VW = 8;

            TuningConfig cfg;
            cfg.BM = BM;
            cfg.BN = BN;
            cfg.BK = BK;
            cfg.VM = VM;
            cfg.VN = VN;
            cfg.vector_width = VW;

            // L1 capacity check
            if (cfg.l1Footprint() > l1Limit)
              continue;

            // Thread count variants
            for (int64_t nt : {1, 2, 4, 8}) {
              cfg.num_threads = nt;

              // Total tiles in M dimension
              int64_t mTiles = (M + BM - 1) / BM;
              int64_t nTiles = (N + BN - 1) / BN;

              for (int64_t gs : {1, 2, 4}) {
                // Grain must not exceed total tiles
                int64_t totalTiles =
                    (cfg.parallel_axis == "m") ? mTiles : nTiles;
                if (gs > totalTiles)
                  continue;

                cfg.grain_size = gs;
                cfg.parallel_axis = (M_bucket == 0) ? "n" : "m";
                cfg.gflops = 0.0;

                configs.push_back(cfg);
              }
            }
          }
        }
      }
    }
  }

  return configs;
}

/// Write results as a JSON schedule_db entry.
static void writeResults(const std::string &path,
                         const std::vector<TuningConfig> &configs, int64_t M,
                         int64_t N, int64_t K) {
  int M_bucket = classifyM(M);

  // Sort by GFLOPS descending
  auto sorted = configs;
  std::sort(sorted.begin(), sorted.end(),
            [](const TuningConfig &a, const TuningConfig &b) {
              return a.gflops > b.gflops;
            });

  // Write top-3 results
  llvm::json::Array entries;
  size_t count = std::min(sorted.size(), size_t(3));
  for (size_t i = 0; i < count; i++) {
    const auto &c = sorted[i];

    llvm::json::Object entry;
    entry["operation"] = "fused_swiglu";
    entry["target"] = "x86-avx2";
    entry["dtype"] = "bf16";
    entry["math_mode"] = "bounded_fast";

    llvm::json::Object shape;
    shape["M_bucket"] = M_bucket;
    shape["N"] = N;
    shape["K"] = K;
    entry["shape"] = std::move(shape);

    llvm::json::Object schedule;
    schedule["BM"] = c.BM;
    schedule["BN"] = c.BN;
    schedule["BK"] = c.BK;
    schedule["VM"] = c.VM;
    schedule["VN"] = c.VN;
    schedule["vector_width"] = c.vector_width;
    schedule["num_threads"] = c.num_threads;
    schedule["parallel_axis"] = c.parallel_axis;
    schedule["grain_size"] = c.grain_size;
    entry["schedule"] = std::move(schedule);

    // Record measured gflops as metadata
    entry["measured_gflops"] = c.gflops;

    entries.push_back(std::move(entry));
  }

  llvm::json::Object root;
  root["version"] = 1;
  root["entries"] = std::move(entries);

  std::ofstream ofs(path);
  if (!ofs) {
    llvm::errs() << "Cannot open output file: " << path << "\n";
    return;
  }

  std::string jsonStr;
  llvm::raw_string_ostream rss(jsonStr);
  rss << llvm::json::Value(std::move(root));
  ofs << jsonStr;
  if (!ofs) {
    llvm::errs() << "Failed to write output file: " << path << "\n";
    return;
  }

  llvm::outs() << "Wrote top-" << count << " configs to " << path << "\n";
}

int main(int argc, char **argv) {
  cl::ParseCommandLineOptions(argc, argv, "LLK autotuning grid search\n");

  int M_bucket = classifyM(tuneM);
  auto configs = generateConfigs(tuneM, tuneN, tuneK);

  llvm::outs() << "M=" << tuneM << " (bucket " << M_bucket << ") N=" << tuneN
               << " K=" << tuneK << "\n";
  llvm::outs() << "Generated " << configs.size()
               << " candidate configs (L1-filtered)\n";

  if (tuneDryRun) {
    // Print top-5 configs by estimated efficiency for inspection
    llvm::outs() << "\nTop candidate configs (estimated):\n";
    size_t n = std::min(configs.size(), size_t(5));
    for (size_t i = 0; i < n; i++) {
      const auto &c = configs[i];
      llvm::outs() << "  [" << i << "] BM=" << c.BM << " BN=" << c.BN
                   << " BK=" << c.BK << " VM=" << c.VM << " VN=" << c.VN
                   << " VW=" << c.vector_width << " threads=" << c.num_threads
                   << " grain=" << c.grain_size << " axis=" << c.parallel_axis
                   << " L1=" << c.l1Footprint() << "B\n";
    }
  }

  // In a full implementation, each config would be measured:
  //   1. JIT-compile with this config's tile sizes
  //   2. Run warmup + measurement
  //   3. Record GFLOPS
  //
  // For now, write the candidate configs as the result (dry-run mode).
  writeResults(tuneOutput, configs, tuneM, tuneN, tuneK);

  return 0;
}
