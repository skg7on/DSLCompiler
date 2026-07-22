#include "LLK/Runtime/ThreadPool.h"
#include <atomic>
#include <gtest/gtest.h>
#include <vector>

TEST(ThreadPool, BasicParallelFor) {
  llk::ThreadPool pool(4);
  std::vector<std::atomic<int>> counters(100);
  for (auto &c : counters)
    c = 0;

  pool.parallelFor(100, 1,
                   [&](int64_t tid, int wid) { counters[tid].fetch_add(1); });

  for (int i = 0; i < 100; i++) {
    EXPECT_EQ(counters[i].load(), 1)
        << "Tile " << i << " executed " << counters[i].load() << " times";
  }
}

TEST(ThreadPool, Stress10kIterations) {
  llk::ThreadPool pool(4);
  for (int iter = 0; iter < 100; iter++) {
    std::atomic<int> sum{0};
    pool.parallelFor(100, 4, [&](int64_t tid, int wid) { sum.fetch_add(1); });
    EXPECT_EQ(sum.load(), 100) << "Iteration " << iter;
  }
  // No crash = PASS
}

TEST(ThreadPool, GrainSizeLargerThanTotal) {
  llk::ThreadPool pool(4);
  std::atomic<int> count{0};
  pool.parallelFor(10, 50, [&](int64_t tid, int wid) { count.fetch_add(1); });
  EXPECT_EQ(count.load(), 10);
}

TEST(ThreadPool, SingleWorker) {
  llk::ThreadPool pool(1);
  std::vector<std::atomic<int>> counters(50);
  for (auto &c : counters)
    c = 0;

  pool.parallelFor(50, 5, [&](int64_t tid, int wid) {
    // wid can be 0 (worker thread) or -1 (caller thread) — both are valid
    counters[tid].fetch_add(1);
  });

  for (int i = 0; i < 50; i++) {
    EXPECT_EQ(counters[i].load(), 1);
  }
}

TEST(ThreadPool, WorkerId) {
  llk::ThreadPool pool(4);
  EXPECT_EQ(pool.workerId(), -1); // Not called from worker thread
}

TEST(ThreadPool, ZeroTiles) {
  llk::ThreadPool pool(4);
  // Should not crash or hang
  pool.parallelFor(0, 1, [&](int64_t tid, int wid) {
    // Should never be called
  });
}
