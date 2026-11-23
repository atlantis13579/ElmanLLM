#include "Distributed/ProcessGroup.h"

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <winsock2.h>
#include <ws2tcpip.h>
#else
#include <arpa/inet.h>
#include <fcntl.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <sys/socket.h>
#include <unistd.h>
#endif

#include <algorithm>
#include <array>
#include <cerrno>
#include <charconv>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <stdexcept>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

namespace elman {
namespace {

using Clock = std::chrono::steady_clock;
#ifdef _WIN32
using SocketHandle = SOCKET;
using SocketLength = int;
using SocketReadSize = int;
constexpr SocketHandle invalid_socket = INVALID_SOCKET;
int socket_error() { return WSAGetLastError(); }
bool interrupted() { return socket_error() == WSAEINTR; }
bool connect_pending() { return socket_error() == WSAEWOULDBLOCK; }
void close_socket(SocketHandle handle) { closesocket(handle); }
#else
using SocketHandle = int;
using SocketLength = socklen_t;
using SocketReadSize = ssize_t;
constexpr SocketHandle invalid_socket = -1;
int socket_error() { return errno; }
bool interrupted() { return socket_error() == EINTR; }
bool connect_pending() { return socket_error() == EINPROGRESS || interrupted(); }
void close_socket(SocketHandle handle) { close(handle); }
#endif

[[noreturn]] void fail(const char* operation) {
  throw std::runtime_error(std::string("Process group ") + operation +
                           " failed (socket error " + std::to_string(socket_error()) + ")");
}

class Network {
 public:
  explicit Network(bool enabled) : enabled_(enabled) {
#ifdef _WIN32
    WSADATA data;
    if (enabled_ && WSAStartup(MAKEWORD(2, 2), &data) != 0)
      throw std::runtime_error("Cannot initialize Winsock for the process group");
#endif
  }
  ~Network() {
#ifdef _WIN32
    if (enabled_) WSACleanup();
#endif
  }
 private:
  bool enabled_;
};

class Socket {
 public:
  explicit Socket(SocketHandle handle = invalid_socket) : handle_(handle) {}
  Socket(Socket&& other) noexcept : handle_(std::exchange(other.handle_, invalid_socket)) {}
  Socket& operator=(Socket&& other) noexcept {
    if (handle_ != invalid_socket) close_socket(handle_);
    handle_ = std::exchange(other.handle_, invalid_socket);
    return *this;
  }
  ~Socket() { if (handle_ != invalid_socket) close_socket(handle_); }
  explicit operator bool() const { return handle_ != invalid_socket; }
  SocketHandle get() const { return handle_; }
 private:
  SocketHandle handle_;
};

Socket tcp_socket() {
  Socket socket_handle(socket(AF_INET, SOCK_STREAM, IPPROTO_TCP));
  if (!socket_handle) fail("socket creation");
  return socket_handle;
}

void set_option(const Socket& socket_handle, int level, int option, const void* value, SocketLength size) {
  if (setsockopt(socket_handle.get(), level, option, static_cast<const char*>(value), size) != 0)
    fail("socket configuration");
}

void configure(const Socket& socket_handle, int seconds) {
#ifdef _WIN32
  const DWORD timeout = static_cast<DWORD>(seconds) * 1000;
#else
  const timeval timeout{seconds, 0};
#endif
  set_option(socket_handle, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout));
  set_option(socket_handle, SOL_SOCKET, SO_SNDTIMEO, &timeout, sizeof(timeout));
  const int enabled = 1;
  set_option(socket_handle, IPPROTO_TCP, TCP_NODELAY, &enabled, sizeof(enabled));
#ifdef SO_NOSIGPIPE
  set_option(socket_handle, SOL_SOCKET, SO_NOSIGPIPE, &enabled, sizeof(enabled));
#endif
}

void nonblocking(const Socket& socket_handle, bool enabled) {
#ifdef _WIN32
  u_long mode = enabled ? 1 : 0;
  if (ioctlsocket(socket_handle.get(), FIONBIO, &mode) != 0) fail("socket mode change");
#else
  const int flags = fcntl(socket_handle.get(), F_GETFL);
  if (flags < 0 || fcntl(socket_handle.get(), F_SETFL, enabled ? flags | O_NONBLOCK : flags & ~O_NONBLOCK) < 0)
    fail("socket mode change");
#endif
}

bool ready(const Socket& socket_handle, bool writing, Clock::time_point deadline) {
  for (;;) {
    const std::chrono::milliseconds::rep remaining = std::chrono::duration_cast<std::chrono::milliseconds>(deadline - Clock::now()).count();
    if (remaining <= 0) return false;
#ifdef _WIN32
    fd_set sockets, errors;
    FD_ZERO(&sockets);
    FD_SET(socket_handle.get(), &sockets);
    FD_ZERO(&errors);
    FD_SET(socket_handle.get(), &errors);
    timeval timeout{static_cast<long>(remaining / 1000), static_cast<long>((remaining % 1000) * 1000)};
    const int result = select(0, writing ? nullptr : &sockets, writing ? &sockets : nullptr, &errors, &timeout);
#else
    pollfd descriptor{socket_handle.get(), static_cast<short>(writing ? POLLOUT : POLLIN), 0};
    const int result = poll(&descriptor, 1, static_cast<int>(remaining));
#endif
    if (result >= 0) return result > 0;
    if (!interrupted()) fail("socket wait");
  }
}

void send_all(const Socket& socket_handle, const void* data, size_t bytes) {
  const char* next = static_cast<const char*>(data);
  while (bytes > 0) {
    const int chunk = static_cast<int>(std::min<size_t>(bytes, 1 << 20));
#ifdef MSG_NOSIGNAL
    const ssize_t sent = send(socket_handle.get(), next, chunk, MSG_NOSIGNAL);
#else
    const int sent = send(socket_handle.get(), next, chunk, 0);
#endif
    if (sent < 0 && interrupted()) continue;
    if (sent <= 0) fail("send");
    next += sent;
    bytes -= static_cast<size_t>(sent);
  }
}

void receive_all(const Socket& socket_handle, void* data, size_t bytes) {
  char* next = static_cast<char*>(data);
  while (bytes > 0) {
    const int chunk = static_cast<int>(std::min<size_t>(bytes, 1 << 20));
    const SocketReadSize received = recv(socket_handle.get(), next, chunk, 0);
    if (received < 0 && interrupted()) continue;
    if (received < 0) fail("receive");
    if (received == 0) throw std::runtime_error("A process group peer disconnected");
    next += received;
    bytes -= static_cast<size_t>(received);
  }
}

enum class Collective : uint64_t { Broadcast = 1, SumFloat, SumDouble, Barrier };
constexpr uint64_t protocol = 0x454c4d414e000001ull;

int environment_int(const char* key, int fallback) {
  const char* value = std::getenv(key);
  if (!value) return fallback;
  std::string_view text(value);
  int result = 0;
  const char* const end = text.data() + text.size();
  const std::from_chars_result parsed = std::from_chars(text.data(), end, result);
  if (parsed.ec != std::errc{} || parsed.ptr != end)
    throw std::invalid_argument(std::string("Invalid distributed option ") + key);
  return result;
}

}  // namespace

class ProcessGroup::Impl {
 public:
  explicit Impl(const ProcessGroupOptions& options)
      : options_(options), network_(options.size > 1) {
    if (options.size <= 0 || options.rank < 0 || options.rank >= options.size || options.local_rank < 0 ||
        options.port < 1 || options.port > 65535 || options.timeout_seconds < 1 || options.timeout_seconds > 3600)
      throw std::invalid_argument("Invalid process group parameters");
    if (options.size == 1) return;

    sockaddr_in address{};
    address.sin_family = AF_INET;
    address.sin_port = htons(static_cast<uint16_t>(options.port));
    if (inet_pton(AF_INET, options.address.c_str(), &address.sin_addr) != 1)
      throw std::invalid_argument("Process group master address must be a numeric IPv4 address");
    const std::chrono::steady_clock::time_point deadline = Clock::now() + std::chrono::seconds(options.timeout_seconds);
    peers_.resize(options.size);
    if (options.rank == 0) {
      listen_for_peers(address, deadline);
    } else {
      connect_to_master(address, deadline);
    }
  }

  int rank() const { return options_.rank; }
  int size() const { return options_.size; }

  void broadcast(void* data, size_t bytes) {
    if (size() == 1) return;
    run(Collective::Broadcast, bytes, [&] {
      if (rank() == 0) {
        for (int peer = 1; peer < size(); ++peer) send_all(peers_[peer], data, bytes);
      } else {
        receive_all(peers_[0], data, bytes);
      }
    });
  }

  template <class T>
  void sum(Collective operation, T* values, size_t count) {
    if (size() == 1) return;
    if (count > SIZE_MAX / sizeof(T)) throw std::overflow_error("Collective payload is too large");
    run(operation, count, [&] {
      if (rank() != 0) {
        send_all(peers_[0], values, count * sizeof(T));
        receive_all(peers_[0], values, count * sizeof(T));
        return;
      }
      std::array<T, 4096> scratch;
      for (int peer = 1; peer < size(); ++peer) {
        for (size_t offset = 0; offset < count;) {
          const size_t chunk = std::min(scratch.size(), count - offset);
          receive_all(peers_[peer], scratch.data(), chunk * sizeof(T));
          for (size_t i = 0; i < chunk; ++i) values[offset + i] += scratch[i];
          offset += chunk;
        }
      }
      for (int peer = 1; peer < size(); ++peer) send_all(peers_[peer], values, count * sizeof(T));
    });
  }

  void barrier() {
    if (size() > 1) run(Collective::Barrier, 0, [] {});
  }

 private:
  void listen_for_peers(const sockaddr_in& address, Clock::time_point deadline) {
    Socket listener = tcp_socket();
    const int enabled = 1;
#ifdef _WIN32
    set_option(listener, SOL_SOCKET, SO_EXCLUSIVEADDRUSE, &enabled, sizeof(enabled));
#else
    set_option(listener, SOL_SOCKET, SO_REUSEADDR, &enabled, sizeof(enabled));
#endif
    if (bind(listener.get(), reinterpret_cast<const sockaddr*>(&address), sizeof(address)) != 0) fail("bind");
    if (listen(listener.get(), SOMAXCONN) != 0) fail("listen");
    for (int count = 1; count < size(); ++count) {
      if (!ready(listener, false, deadline)) throw std::runtime_error("Timed out waiting for process group ranks");
      Socket peer(accept(listener.get(), nullptr, nullptr));
      if (!peer) fail("accept");
      configure(peer, options_.timeout_seconds);
      std::array<uint64_t, 3> greeting{};
      receive_all(peer, greeting.data(), sizeof(greeting));
      if (greeting[0] != protocol || greeting[1] != uint64_t(size()) ||
          greeting[2] == 0 || greeting[2] >= uint64_t(size()) || peers_[greeting[2]])
        throw std::runtime_error("Unexpected process group rank handshake");
      peers_[greeting[2]] = std::move(peer);
    }
    for (int peer = 1; peer < size(); ++peer) send_all(peers_[peer], &protocol, sizeof(protocol));
  }

  void connect_to_master(const sockaddr_in& address, Clock::time_point deadline) {
    while (Clock::now() < deadline) {
      Socket peer = tcp_socket();
      nonblocking(peer, true);
      bool connected = connect(peer.get(), reinterpret_cast<const sockaddr*>(&address), sizeof(address)) == 0;
      if (!connected && connect_pending() && ready(peer, true, deadline)) {
        int error = 0;
        SocketLength length = sizeof(error);
        connected = getsockopt(peer.get(), SOL_SOCKET, SO_ERROR, reinterpret_cast<char*>(&error), &length) == 0 && error == 0;
      }
      if (!connected) {
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        continue;
      }
      nonblocking(peer, false);
      configure(peer, options_.timeout_seconds);
      const std::array<uint64_t, 3> greeting{protocol, uint64_t(size()), uint64_t(rank())};
      send_all(peer, greeting.data(), sizeof(greeting));
      uint64_t response = 0;
      receive_all(peer, &response, sizeof(response));
      if (response != protocol) throw std::runtime_error("Invalid process group master handshake");
      peers_[0] = std::move(peer);
      return;
    }
    throw std::runtime_error("Timed out connecting to the process group master");
  }

  template <class Work>
  void run(Collective operation, size_t count, Work work) {
    if (failed_) throw std::runtime_error("Process group is closed after a collective failure");
    try {
      const std::array<uint64_t, 3> header{sequence_++, uint64_t(operation), count};
      const uint8_t accepted = 1;
      if (rank() == 0) {
        for (int peer = 1; peer < size(); ++peer) {
          std::array<uint64_t, 3> remote{};
          receive_all(peers_[peer], remote.data(), sizeof(remote));
          if (remote != header) throw std::runtime_error("Process group collective order or size mismatch");
        }
        for (int peer = 1; peer < size(); ++peer) send_all(peers_[peer], &accepted, sizeof(accepted));
      } else {
        send_all(peers_[0], header.data(), sizeof(header));
        uint8_t response = 0;
        receive_all(peers_[0], &response, sizeof(response));
        if (response != accepted) throw std::runtime_error("Process group collective rejected");
      }
      work();
    } catch (...) {
      failed_ = true;
      peers_.clear();
      throw;
    }
  }

  ProcessGroupOptions options_;
  Network network_;
  std::vector<Socket> peers_;
  uint64_t sequence_ = 0;
  bool failed_ = false;
};

ProcessGroup::ProcessGroup(const ProcessGroupOptions& options) : impl_(std::make_unique<Impl>(options)) {}
ProcessGroup::~ProcessGroup() = default;
int ProcessGroup::rank() const { return impl_->rank(); }
int ProcessGroup::size() const { return impl_->size(); }
void ProcessGroup::broadcast(void* values, size_t bytes) { impl_->broadcast(values, bytes); }
void ProcessGroup::all_reduce_sum(float* values, size_t count) { impl_->sum(Collective::SumFloat, values, count); }
void ProcessGroup::all_reduce_sum(double* values, size_t count) { impl_->sum(Collective::SumDouble, values, count); }
void ProcessGroup::barrier() { impl_->barrier(); }

ProcessGroupOptions process_group_environment() {
  ProcessGroupOptions options;
  options.rank = environment_int("ELMAN_RANK", options.rank);
  options.size = environment_int("ELMAN_WORLD_SIZE", options.size);
  options.local_rank = environment_int("ELMAN_LOCAL_RANK", options.rank);
  options.port = environment_int("ELMAN_MASTER_PORT", options.port);
  options.timeout_seconds = environment_int("ELMAN_DISTRIBUTED_TIMEOUT", options.timeout_seconds);
  if (const char* address = std::getenv("ELMAN_MASTER_ADDR")) options.address = address;
  return options;
}

}  // namespace elman
