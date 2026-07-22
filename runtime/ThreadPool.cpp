#include "LLK/Runtime/ThreadPool.h"
#include <cstdlib>

namespace llk {

thread_local int ThreadPool::current_worker_id_ = -1;

ThreadPool::ThreadPool(int num_threads) {
  if (num_threads < 1)
    num_threads = 1;
  num_workers_ = num_threads;
  barrier_target_ = num_threads;
  workers_.reserve(num_threads);
  running_.reset(new std::atomic<bool>[num_threads]);
  worker_done_.reset(new std::atomic<bool>[num_threads]);
  for (int i = 0; i < num_threads; i++) {
    running_[i] = true;
    worker_done_[i] = false;
    Worker w;
    w.id = i;
    w.thread = std::thread(&ThreadPool::workerLoop, this, i);
    workers_.push_back(std::move(w));
  }
}

ThreadPool::~ThreadPool() {
  for (int i = 0; i < num_workers_; i++)
    running_[i] = false;
  cv_work_.notify_all();
  for (auto &w : workers_) {
    if (w.thread.joinable())
      w.thread.join();
  }
}

void ThreadPool::workerLoop(int worker_id) {
  current_worker_id_ = worker_id;
  while (running_[worker_id]) {
    std::unique_lock<std::mutex> lock(mutex_);
    cv_work_.wait(lock, [&] {
      return !running_[worker_id] ||
             (callback_ != nullptr && !worker_done_[worker_id]);
    });
    if (!running_[worker_id])
      return;

    lock.unlock();
    while (true) {
      int64_t start = next_tile_.fetch_add(grain_size_);
      if (start >= total_tiles_)
        break;
      int64_t end = std::min(start + grain_size_, total_tiles_);
      for (int64_t t = start; t < end; t++) {
        (*callback_)(t, worker_id);
      }
    }

    // Signal done under mutex to prevent lost wakeup (finding #7)
    lock.lock();
    worker_done_[worker_id] = true;
    cv_done_.notify_one();
  }
}

void ThreadPool::parallelFor(int64_t total_tiles, int64_t grain_size,
                             std::function<void(int64_t, int)> callback) {
  if (total_tiles <= 0)
    return;

  total_tiles_ = total_tiles;
  grain_size_ = std::max(int64_t(1), grain_size);
  next_tile_ = 0;

  // Reset done flags and set callback under mutex to prevent races
  // (findings #1, #8)
  {
    std::unique_lock<std::mutex> lock(mutex_);
    for (int i = 0; i < num_workers_; i++)
      worker_done_[i] = false;
    callback_ = &callback;
  }
  cv_work_.notify_all();

  // Caller thread also participates
  while (true) {
    int64_t start = next_tile_.fetch_add(grain_size_);
    if (start >= total_tiles_)
      break;
    int64_t end = std::min(start + grain_size_, total_tiles_);
    for (int64_t t = start; t < end; t++) {
      callback(t, -1); // -1 = caller thread
    }
  }

  // Wait for all worker threads to finish
  std::unique_lock<std::mutex> lock(mutex_);
  cv_done_.wait(lock, [&] {
    for (int i = 0; i < num_workers_; i++)
      if (!worker_done_[i])
        return false;
    return true;
  });
  callback_ = nullptr;
}

void ThreadPool::barrier(int worker_id) {
  int epoch = barrier_epoch_.load();
  int count = barrier_count_.fetch_add(1) + 1;
  if (count == barrier_target_) {
    barrier_epoch_++;
    barrier_count_ = 0;
  } else {
    while (barrier_epoch_.load() == epoch) {
      std::this_thread::yield();
    }
  }
}

void *ThreadPool::getScratch(int worker_id) {
  if (worker_id < 0 || worker_id >= static_cast<int>(workers_.size()))
    return nullptr;
  return workers_[worker_id].scratch;
}

void ThreadPool::setScratch(int worker_id, void *ptr, size_t size) {
  if (worker_id < 0 || worker_id >= static_cast<int>(workers_.size()))
    return;
  // NOTE: The old scratch pointer is freed with free().
  // Callers MUST allocate scratch with malloc/calloc/realloc or pass nullptr.
  // Passing a pointer from new[], mmap, or any other allocator is UB.
  // free(nullptr) is safe per the C standard.
  free(workers_[worker_id].scratch);
  workers_[worker_id].scratch = ptr;
  workers_[worker_id].scratch_size = size;
}

int ThreadPool::workerId() const { return current_worker_id_; }

void ThreadPool::setAffinity(int worker_id, int cpu_core) {
  // Stub: platform-specific implementation (pthread_setaffinity_np on Linux)
}

} // namespace llk
