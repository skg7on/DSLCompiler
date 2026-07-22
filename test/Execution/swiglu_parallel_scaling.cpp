#include "LLK/Runtime/ThreadPool.h"

#include <atomic>
#include <chrono>
#include <cmath>
#include <gtest/gtest.h>
#include <thread>
#include <vector>

// Simulated tile work: compute-intensive loop representing
// the fused matmul + activation inner kernel.
static void simulatedTileWork(int64_t /*tile_id*/, int /*worker_id*/) {
  volatile double x = 0.0;
  for (int i = 0; i < 10000; i++) {
    x += std::sin(static_cast<double>(i)) * std::cos(static_cast<double>(i));
  }
  (void)x;
}

TEST(SwiGLUParallel, Scaling_1_to_N_threads) {
  int maxThreads = static_cast<int>(std::thread::hardware_concurrency());
  if (maxThreads < 2)
    maxThreads = 2;

  std::vector<int> threadCounts;
  for (int t = 1; t <= maxThreads; t *= 2)
    threadCounts.push_back(t);
  if (threadCounts.back() != maxThreads)
    threadCounts.push_back(maxThreads);

  std::vector<double> times;

  for (int nThreads : threadCounts) {
    llk::ThreadPool pool(nThreads);

    auto start = std::chrono::high_resolution_clock::now();
    pool.parallelFor(100, 4, simulatedTileWork);
    auto end = std::chrono::high_resolution_clock::now();

    double ms = std::chrono::duration<double, std::milli>(end - start).count();
    times.push_back(ms);

    std::cout << "  " << nThreads << " thread(s): " << ms << " ms" << std::endl;
  }

  // Verify monotonic decrease (within 30% noise tolerance for small systems)
  for (size_t i = 1; i < times.size(); i++) {
    EXPECT_LE(times[i], times[0] * 1.3)
        << "Thread count " << threadCounts[i]
        << " was >30% slower than single-thread (baseline: " << times[0]
        << "ms)";
  }
}

TEST(SwiGLUParallel, ResultIsDeterministic) {
  // Verify that running the same work multiple times with different
  // thread counts produces consistent per-tile execution counts.
  for (int nThreads : {1, 2, 4}) {
    llk::ThreadPool pool(nThreads);
    std::vector<std::atomic<int>> counters(64);
    for (auto &c : counters)
      c = 0;

    pool.parallelFor(64, 1,
                     [&](int64_t tid, int wid) { counters[tid].fetch_add(1); });

    for (int i = 0; i < 64; i++) {
      EXPECT_EQ(counters[i].load(), 1)
          << "With " << nThreads << " threads: tile " << i << " executed "
          << counters[i].load() << " times";
    }
  }
}

TEST(SwiGLUParallel, LargeTileCount) {
  llk::ThreadPool pool(std::thread::hardware_concurrency());
  std::atomic<int64_t> totalOps{0};
  int64_t numTiles = 1000;

  pool.parallelFor(numTiles, 1, [&](int64_t tid, int wid) {
    // Simulate a small amount of work per tile
    totalOps.fetch_add(1);
  });

  EXPECT_EQ(totalOps.load(), numTiles);
}
