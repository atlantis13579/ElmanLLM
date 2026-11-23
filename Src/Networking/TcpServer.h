#pragma once
#include <atomic>
#include <chrono>
#include <cstdint>
#include <functional>
#include <stdexcept>
#include <string>

namespace elman {
struct TcpOptions {
  std::string address = "127.0.0.1";
  int port = 8080, max_connections = 32;
  int read_timeout_ms = 5000, write_timeout_ms = 5000;
};
class ConnectionClosed : public std::runtime_error {
 public:
  ConnectionClosed() : std::runtime_error("Connection closed or timed out") {
  }
};
class ReadTimeout : public std::runtime_error {
 public:
  ReadTimeout() : std::runtime_error("Socket read deadline exceeded") {
  }
};
struct TcpStream {
  std::function<size_t(char*, size_t, std::chrono::steady_clock::time_point)>
      read;
  std::function<void(const std::string&)> send;
  std::function<bool()> closed;
};
struct ConnectionCounts {
  std::atomic<uint64_t> accepted{0}, rejected{0};
};
void listen_tcp(const TcpOptions& options, const std::string& scheme,
                const std::string& busy_reply,
                const std::function<bool()>& stop,
                const std::function<void(const TcpStream&)>& handle,
                ConnectionCounts& counts,
                const std::function<void(int)>& ready = {});
}  // namespace elman
