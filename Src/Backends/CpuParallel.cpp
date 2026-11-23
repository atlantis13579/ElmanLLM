#include "Backends/CpuParallel.h"
#include "Core/LocalFunction.h"

#include "Backends/CpuMath.h"

#include <algorithm>
#include <atomic>
#include <charconv>
#include <condition_variable>
#include <cstdlib>
#include <exception>
#include <future>
#include <memory>
#include <mutex>
#include <queue>
#include <stdexcept>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

namespace elman {
namespace {

constexpr int maximum_threads = 256;
std::atomic<int> thread_override{0};
thread_local bool inside_parallel_region = false;

class ParallelRegion {
 public:
  ParallelRegion() : previous_(inside_parallel_region) { inside_parallel_region = true; }
  ~ParallelRegion() { inside_parallel_region = previous_; }
 private:
  bool previous_;
};

class ThreadPool {
 public:
  explicit ThreadPool(int workers) {
    threads_.reserve(workers);
    try {
      for (int i = 0; i < workers; ++i) threads_.emplace_back([this] { work(); });
    } catch (...) {
      stop();
      throw;
    }
  }

  ~ThreadPool() { stop(); }

  std::future<void> submit(std::packaged_task<void()> task) {
    std::future<void> result = task.get_future();
    {
      std::lock_guard<std::mutex> lock(mutex_);
      tasks_.push(std::move(task));
    }
    ready_.notify_one();
    return result;
  }

 private:
  void stop() {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      stopped_ = true;
    }
    ready_.notify_all();
    for (std::thread& thread : threads_) thread.join();
  }

  void work() {
    ParallelRegion region;
    for (;;) {
      std::packaged_task<void()> task;
      {
        std::unique_lock<std::mutex> lock(mutex_);
        ready_.wait(lock, [&] { return stopped_ || !tasks_.empty(); });
        if (tasks_.empty()) return;
        task = std::move(tasks_.front());
        tasks_.pop();
      }
      task();
    }
  }

  std::mutex mutex_;
  std::condition_variable ready_;
  std::queue<std::packaged_task<void()>> tasks_;
  std::vector<std::thread> threads_;
  bool stopped_ = false;
};

std::shared_ptr<ThreadPool> pool_for(int workers) {
  static std::mutex mutex;
  static std::shared_ptr<ThreadPool> pool;
  static int worker_count = 0;
  std::lock_guard<std::mutex> lock(mutex);
  if (workers > worker_count) {
    pool = std::make_shared<ThreadPool>(workers);
    worker_count = workers;
  }
  return pool;
}

int environment_threads() {
  const char* value = std::getenv("ELMAN_CPU_THREADS");
  if (!value) return 0;
  const std::string_view text(value);
  const char* const end = text.data() + text.size();
  int threads = 0;
  const std::from_chars_result parsed = std::from_chars(text.data(), end, threads);
  if (parsed.ec != std::errc{} || parsed.ptr != end || threads < 1 || threads > maximum_threads)
    throw std::invalid_argument("ELMAN_CPU_THREADS must be in [1,256]");
  return threads;
}

}  // namespace

void set_cpu_thread_count(int threads) {
  if (threads < 0 || threads > maximum_threads)
    throw std::invalid_argument("CPU thread count must be in [0,256]");
  thread_override.store(threads);
}

int configured_cpu_thread_count() {
  const int threads = thread_override.load();
  if (threads) return threads;
  static const int environment = environment_threads();
  return environment;
}

void parallel_for_cpu(int64_t count, int64_t grain, const std::function<void(int64_t, int64_t)>& task,
                      int thread_limit) {
  if (count <= 0) return;
  if (grain <= 0) throw std::invalid_argument("CPU parallel grain must be positive");
  if (inside_parallel_region || count <= grain) {
    task(0, count);
    return;
  }
  int threads = std::min(cpu_math_thread_count(), maximum_threads);
  if (thread_limit > 0) threads = std::min(threads, thread_limit);
  const int partitions = static_cast<int>(std::min<int64_t>(1 + (count - 1) / grain, threads));
  if (partitions == 1) {
    task(0, count);
    return;
  }

  // Keep one partition on the caller. Observe all submitted tasks before
  // propagating errors so tensor references stay alive until workers finish.
  std::shared_ptr<ThreadPool> pool = pool_for(partitions - 1);
  LocalFunction range{[&](int part) {
    const int64_t segment = count / partitions;
    const int64_t remainder = count % partitions;
    const int64_t begin = segment * part + std::min<int64_t>(part, remainder);
    const int64_t length = segment + (part < remainder ? 1 : 0);
    task(begin, begin + length);
  }};
  std::vector<std::future<void>> jobs;
  jobs.reserve(partitions - 1);
  std::exception_ptr failure;
  try {
    for (int part = 1; part < partitions; ++part) {
      jobs.push_back(pool->submit(std::packaged_task<void()>([&, part] { range(part); })));
    }
    ParallelRegion region;
    range(0);
  } catch (...) {
    failure = std::current_exception();
  }
  for (std::future<void>& job : jobs) {
    try {
      job.get();
    } catch (...) {
      if (!failure) failure = std::current_exception();
    }
  }
  if (failure) std::rethrow_exception(failure);
}

}  // namespace elman
