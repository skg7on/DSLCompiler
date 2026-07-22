#ifndef LLK_RUNTIME_THREADPOOL_H
#define LLK_RUNTIME_THREADPOOL_H

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

namespace llk {

class ThreadPool {
public:
  explicit ThreadPool(int num_threads = std::thread::hardware_concurrency());
  ~ThreadPool();

  void
  parallelFor(int64_t total_tiles, int64_t grain_size,
              std::function<void(int64_t tile_id, int worker_id)> callback);

  void barrier(int worker_id);

  void *getScratch(int worker_id);
  void setScratch(int worker_id, void *ptr, size_t size);

  int numWorkers() const { return workers_.size(); }
  int workerId() const;

  void setAffinity(int worker_id, int cpu_core);

private:
  struct Worker {
    int id;
    std::thread thread;
    void *scratch = nullptr;
    size_t scratch_size = 0;
  };

  void workerLoop(int worker_id);

  std::vector<Worker> workers_;
  // Use unique_ptr<T[]> instead of vector<T> because std::atomic<bool>
  // is not Cpp17MoveInsertable on all STL implementations (libc++).
  std::unique_ptr<std::atomic<bool>[]> running_;
  std::unique_ptr<std::atomic<bool>[]> worker_done_;
  int num_workers_{0};
  std::mutex mutex_;
  std::condition_variable cv_work_;
  std::condition_variable cv_done_;
  std::atomic<int64_t> next_tile_{0};
  int64_t total_tiles_{0};
  int64_t grain_size_{0};
  std::function<void(int64_t, int)> *callback_{nullptr};
  std::atomic<int> barrier_count_{0};
  std::atomic<int> barrier_epoch_{0};
  int barrier_target_{0};
  static thread_local int current_worker_id_;
};

} // namespace llk

#endif
