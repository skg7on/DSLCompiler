#include "LLK/Runtime/ThreadPool.h"

#include <atomic>
#include <gtest/gtest.h>
#include <thread>

TEST(SwiGLUSmallShape, FewTilesExecutedCorrectly) {
  // Small tile count: single-threaded execution produces correct results
  llk::ThreadPool pool(1);
  std::vector<std::atomic<int>> counters(3);
  for (auto &c : counters)
    c = 0;

  pool.parallelFor(3, 1,
                   [&](int64_t tid, int wid) { counters[tid].fetch_add(1); });

  for (int i = 0; i < 3; i++) {
    EXPECT_EQ(counters[i].load(), 1)
        << "Small shape: tile " << i << " executed " << counters[i].load()
        << " times";
  }
}

TEST(SwiGLUSmallShape, SingleTile) {
  llk::ThreadPool pool(std::thread::hardware_concurrency());
  std::atomic<int> count{0};

  pool.parallelFor(1, 1, [&](int64_t tid, int wid) { count.fetch_add(1); });

  EXPECT_EQ(count.load(), 1);
}
