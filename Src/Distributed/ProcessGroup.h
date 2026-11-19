#pragma once

#include <cstddef>
#include <memory>
#include <string>

namespace elman {

struct ProcessGroupOptions {
  int rank = 0;
  int size = 1;
  int local_rank = 0;
  std::string address = "127.0.0.1";
  int port = 29500;
  int timeout_seconds = 60;
};

ProcessGroupOptions process_group_environment();

// All ranks call collectives in the same order. Each group has one caller thread.
class ProcessGroup {
 public:
  explicit ProcessGroup(const ProcessGroupOptions& options);
  ~ProcessGroup();
  ProcessGroup(const ProcessGroup&) = delete;
  ProcessGroup& operator=(const ProcessGroup&) = delete;

  int rank() const;
  int size() const;
  void broadcast(void* values, size_t bytes);
  void all_reduce_sum(float* values, size_t count);
  void all_reduce_sum(double* values, size_t count);
  void barrier();

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace elman
