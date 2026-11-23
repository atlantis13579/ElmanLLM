#include "IO/AtomicFile.h"

#include <atomic>
#include <chrono>
#include <cstdint>
#include <stdexcept>
#include <string>
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <Windows.h>
#else
#include <unistd.h>
#endif

namespace elman {

void write_file_atomically(const std::filesystem::path& destination,
                           const std::function<void(const std::filesystem::path&)>& write) {
  static std::atomic<uint64_t> sequence{0};
  const std::chrono::steady_clock::rep now = std::chrono::steady_clock::now().time_since_epoch().count();
#ifdef _WIN32
  const DWORD process = GetCurrentProcessId();
#else
  const pid_t process = getpid();
#endif
  if (!destination.parent_path().empty()) std::filesystem::create_directories(destination.parent_path());
  std::filesystem::path temporary = destination;
  temporary += ".tmp." + std::to_string(process) + "." + std::to_string(now) + "." +
               std::to_string(sequence.fetch_add(1, std::memory_order_relaxed));
  try {
    write(temporary);
#ifdef _WIN32
    if (!MoveFileExW(temporary.c_str(), destination.c_str(), MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH)) {
      const DWORD error = GetLastError();
      throw std::runtime_error("Cannot replace output file (Windows error " + std::to_string(error) + ")");
    }
#else
    std::filesystem::rename(temporary, destination);
#endif
  } catch (...) {
    std::error_code ignored;
    std::filesystem::remove(temporary, ignored);
    throw;
  }
}

}  // namespace elman
