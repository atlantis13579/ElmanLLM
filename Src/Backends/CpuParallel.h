#pragma once

#include <cstdint>
#include <functional>

namespace elman {

// Zero restores the environment or backend default. A positive value includes
// the caller and caps the number of threads used by one kernal.
void set_cpu_thread_count(int threads);
int configured_cpu_thread_count();

// Tasks receive disjoint half-open ranges and complete before this call returns.
void parallel_for_cpu(int64_t count, int64_t grain, const std::function<void(int64_t, int64_t)>& task,
                      int thread_limit = 0);

}  // namespace elman
