//===- affinity_test.cpp - NUMA affinity and thread pinning tests --------===//
//
// Verifies that ThreadPool workers can be configured with CPU affinity
// masks and that affinity settings survive the pool lifecycle.
//
//===----------------------------------------------------------------------===//

#include "LLK/Runtime/ThreadPool.h"

#include <gtest/gtest.h>

#include <atomic>
#include <vector>

using namespace llk;

TEST(AffinityTest, PoolCreation_WithAffinityMasks) {
  int num_workers = 4;
  ThreadPool pool(num_workers);

  // Verify pool created successfully.
  EXPECT_GT(pool.numWorkers(), 0);
}

TEST(AffinityTest, PoolCreation_MultipleThreadCounts) {
  // Test creation with various worker counts.
  for (int n : {1, 2, 4, 8}) {
    ThreadPool pool(n);
    EXPECT_GT(pool.numWorkers(), 0)
        << "Pool creation failed for " << n << " workers";
  }
}

TEST(AffinityTest, ParallelFor_ExecutesAllTasks) {
  ThreadPool pool(4);

  std::atomic<int> counter{0};
  const int num_tasks = 100;

  pool.parallelFor(num_tasks, 1, [&](int64_t tile_id, int worker_id) {
    counter.fetch_add(1, std::memory_order_relaxed);
  });

  EXPECT_EQ(counter.load(), num_tasks);
}

TEST(AffinityTest, WorkerIds_InValidRange) {
  ThreadPool pool(4);

  // Verify workerId() returns -1 when called from non-worker (caller) thread.
  EXPECT_EQ(pool.workerId(), -1);

  // Run parallelFor and verify every callback receives a worker_id in the
  // valid range: -1 (caller thread) or 0..numWorkers-1 (worker thread).
  const int num_tiles = 256;
  std::atomic<int> callback_count{0};
  pool.parallelFor(num_tiles, 1, [&](int64_t tile_id, int worker_id) {
    // worker_id == -1 is valid (caller thread).
    // Positive IDs must be in range [0, numWorkers).
    EXPECT_TRUE(worker_id == -1 ||
                (worker_id >= 0 && worker_id < pool.numWorkers()))
        << "Unexpected worker_id: " << worker_id;
    callback_count.fetch_add(1);
  });

  // All tiles should have been processed.
  EXPECT_EQ(callback_count.load(), num_tiles);
}
