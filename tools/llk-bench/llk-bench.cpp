//===- llk-bench.cpp - LLK kernel benchmarking harness --------------------===//
//
// Measures GFLOPS and execution time for LLK kernels across different
// shape regimes (M bucket, N, K) and thread counts.
//
//===----------------------------------------------------------------------===//

#include "LLK/Runtime/ThreadPool.h"

#include "llvm/Support/CommandLine.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <vector>

namespace cl = llvm::cl;

static cl::opt<int64_t> benchM("M", cl::desc("M dimension (rows)"),
                               cl::init(128));
static cl::opt<int64_t> benchN("N", cl::desc("N dimension (hidden)"),
                               cl::init(4096));
static cl::opt<int64_t> benchK("K", cl::desc("K dimension (input)"),
                               cl::init(4096));
static cl::opt<int> benchThreads("threads", cl::desc("Thread count"),
                                 cl::init(8));
static cl::opt<int> benchWarmup("warmup-ms", cl::desc("Warmup milliseconds"),
                                cl::init(500));
static cl::opt<int> benchMeasure("measure-ms",
                                 cl::desc("Measurement milliseconds"),
                                 cl::init(2000));
static cl::opt<int> benchReps("reps", cl::desc("Measurement repetitions"),
                              cl::init(5));

/// Dummy kernel work: simple elementwise computation over M*N elements.
/// Used as a stand-in until the full JIT pipeline is integrated.
static void dummyKernel(float *output, const float *input, int64_t M, int64_t N,
                        int64_t K, int num_threads) {
  llk::ThreadPool pool(num_threads);

  pool.parallelFor(M, 1, [&](int64_t tile_id, int worker_id) {
    int64_t m = tile_id;
    for (int64_t n = 0; n < N; n++) {
      float sum = 0.0f;
      for (int64_t k = 0; k < K; k++) {
        sum += input[m * K + k] * 0.5f;
      }
      output[m * N + n] = sum;
    }
  });
}

int main(int argc, char **argv) {
  cl::ParseCommandLineOptions(argc, argv, "LLK kernel benchmark\n");

  std::cout << "LLK Bench — M=" << benchM << " N=" << benchN << " K=" << benchK
            << " threads=" << benchThreads << "\n";

  size_t inputSize = benchM * benchK;
  size_t outputSize = benchM * benchN;
  std::vector<float> input(inputSize, 1.0f);
  std::vector<float> output(outputSize, 0.0f);

  // Warmup
  {
    auto warmupStart = std::chrono::high_resolution_clock::now();
    while (true) {
      dummyKernel(output.data(), input.data(), benchM, benchN, benchK,
                  benchThreads);
      auto now = std::chrono::high_resolution_clock::now();
      auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                         now - warmupStart)
                         .count();
      if (elapsed >= benchWarmup)
        break;
    }
  }

  // Measurement
  double bestGflops = 0.0;
  double bestMs = 0.0;

  for (int rep = 0; rep < benchReps; rep++) {
    auto start = std::chrono::high_resolution_clock::now();

    // Run until measurement window elapses
    int64_t iters = 0;
    while (true) {
      dummyKernel(output.data(), input.data(), benchM, benchN, benchK,
                  benchThreads);
      iters++;
      auto now = std::chrono::high_resolution_clock::now();
      auto elapsed =
          std::chrono::duration_cast<std::chrono::milliseconds>(now - start)
              .count();
      if (elapsed >= benchMeasure)
        break;
    }

    auto end = std::chrono::high_resolution_clock::now();
    double totalMs =
        std::chrono::duration<double, std::milli>(end - start).count();
    double msPerIter = totalMs / iters;

    // FLOPs for dummy kernel: 2*K FLOPs per output element (1 mul + 1 add)
    double totalFlops = static_cast<double>(benchM) * benchN * (2 * benchK);
    double gflops = (totalFlops / (msPerIter * 1e-3)) / 1e9;

    if (gflops > bestGflops) {
      bestGflops = gflops;
      bestMs = msPerIter;
    }
  }

  std::cout << "Best: " << bestGflops << " GFLOPS (" << bestMs << " ms/iter, "
            << benchReps << " reps)\n";

  return 0;
}
