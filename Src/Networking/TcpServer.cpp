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
#include <sys/select.h>
#include <sys/socket.h>
#include <unistd.h>

#include <cerrno>
#endif

#include <algorithm>
#include <atomic>
#include <chrono>
#include <iostream>
#include <memory>
#include <thread>
#include <vector>

#include "Networking/TcpServer.h"
#include "Core/LocalFunction.h"

namespace elman {
namespace {
using Clock = std::chrono::steady_clock;
#ifdef _WIN32
using NativeSocket = SOCKET;
constexpr NativeSocket invalid_socket = INVALID_SOCKET;
int socket_error() {
  return WSAGetLastError();
}
bool would_block(int code) {
  return code == WSAEWOULDBLOCK || code == WSAEINTR;
}
void close_socket(NativeSocket value) {
  closesocket(value);
}
using SocketLength = int;
struct SocketRuntime {
  SocketRuntime() {
    WSADATA data{};
    if (WSAStartup(MAKEWORD(2, 2), &data)) {
      throw std::runtime_error("WSAStartup failed");
    }
  }
  ~SocketRuntime() {
    WSACleanup();
  }
};
#else
using NativeSocket = int;
constexpr NativeSocket invalid_socket = -1;
int socket_error() {
  return errno;
}
bool would_block(int code) {
  return code == EWOULDBLOCK || code == EAGAIN || code == EINTR;
}
void close_socket(NativeSocket value) {
  close(value);
}
using SocketLength = socklen_t;
struct SocketRuntime {};
#endif
class Socket {
 public:
  explicit Socket(NativeSocket value = invalid_socket) : value_(value) {
#ifdef SO_NOSIGPIPE
    int enabled = 1;
    if (value_ != invalid_socket &&
        setsockopt(value_, SOL_SOCKET, SO_NOSIGPIPE, &enabled,
                   sizeof(enabled)) != 0) {
      close_socket(value_);
      throw std::runtime_error("Cannot disable SIGPIPE on socket");
    }
#endif
  }
  ~Socket() {
    if (value_ != invalid_socket) {
      close_socket(value_);
    }
  }
  Socket(Socket&& other) noexcept : value_(other.value_) {
    other.value_ = invalid_socket;
  }
  Socket(const Socket&) = delete;
  NativeSocket get() const {
    return value_;
  }

 private:
  NativeSocket value_;
};
void nonblocking(NativeSocket value) {
#ifdef _WIN32
  u_long mode = 1;
  if (ioctlsocket(value, FIONBIO, &mode)) {
    throw std::runtime_error("Cannot set nonblocking socket");
  }
#else
  int flags = fcntl(value, F_GETFL, 0);
  if (flags < 0 || fcntl(value, F_SETFL, flags | O_NONBLOCK) < 0) {
    throw std::runtime_error("Cannot set nonblocking socket");
  }
  if (fcntl(value, F_SETFD, FD_CLOEXEC) < 0) {
    throw std::runtime_error("Cannot set close-on-exec on socket");
  }
#endif
}
bool poll_socket(NativeSocket value, bool write, int milliseconds) {
#ifdef _WIN32
  fd_set set;
  FD_ZERO(&set);
  FD_SET(value, &set);
  timeval timeout{milliseconds / 1000, (milliseconds % 1000) * 1000};
  int result = select(int(value + 1), write ? nullptr : &set,
                      write ? &set : nullptr, nullptr, &timeout);
#else
  pollfd descriptor{value, short(write ? POLLOUT : POLLIN), 0};
  int result = poll(&descriptor, 1, milliseconds);
#endif
  if (result < 0 && !would_block(socket_error())) {
    throw ConnectionClosed();
  }
  return result > 0;
}
bool disconnected(NativeSocket value) {
  if (!poll_socket(value, false, 0)) {
    return false;
  }
  char byte;
  int count = recv(value, &byte, 1, MSG_PEEK);
  return count == 0 || (count < 0 && !would_block(socket_error()));
}
void finish_rejection(NativeSocket value) {
  // Deliver the rejection before closing a socket with unread request bytes.
  // The accept thread has a small, fixed drain budget, independent of clients.
#ifdef _WIN32
  shutdown(value, SD_SEND);
#else
  shutdown(value, SHUT_WR);
#endif
  const std::chrono::steady_clock::time_point deadline = Clock::now() + std::chrono::milliseconds(25);
  size_t remaining = 64 * 1024;
  while (remaining && Clock::now() < deadline) {
    if (!poll_socket(value, false, 5)) {
      continue;
    }
    char bytes[4096];
    int count = recv(value, bytes, int(std::min(remaining, sizeof(bytes))), 0);
    if (count < 0 && would_block(socket_error())) {
      continue;
    }
    if (count <= 0) {
      break;
    }
    remaining -= size_t(count);
  }
}
void send_all(NativeSocket value, const std::string& text, int timeout_ms,
              const std::atomic<bool>& stopping) {
  size_t offset = 0;
  const std::chrono::steady_clock::time_point deadline = Clock::now() + std::chrono::milliseconds(timeout_ms);
  while (offset < text.size()) {
    if (stopping.load() || Clock::now() >= deadline) {
      throw ConnectionClosed();
    }
    if (!poll_socket(value, true, 25)) {
      continue;
    }
#ifdef MSG_NOSIGNAL
    constexpr int flags = MSG_NOSIGNAL;
#else
    constexpr int flags = 0;
#endif
    int sent =
        send(value, text.data() + offset,
             int(std::min(size_t(1024 * 1024), text.size() - offset)), flags);
    if (sent < 0 && would_block(socket_error())) {
      continue;
    }
    if (sent <= 0) {
      throw ConnectionClosed();
    }
    offset += size_t(sent);
  }
}
}  // namespace
void listen_tcp(const TcpOptions& options, const std::string& scheme,
                const std::string& busy_reply,
                const std::function<bool()>& stop,
                const std::function<void(const TcpStream&)>& handle,
                ConnectionCounts& counts,
                const std::function<void(int)>& ready) {
  if (options.port < 0 || options.port > 65535 ||
      options.max_connections <= 0 || options.read_timeout_ms <= 0 ||
      options.write_timeout_ms <= 0) {
    throw std::invalid_argument("Invalid TCP limits");
  }
  SocketRuntime runtime;
  Socket listener(::socket(AF_INET, SOCK_STREAM, IPPROTO_TCP));
  if (listener.get() == invalid_socket) {
    throw std::runtime_error("Cannot create listener");
  }
  int enabled = 1;
#ifdef _WIN32
  setsockopt(listener.get(), SOL_SOCKET, SO_EXCLUSIVEADDRUSE,
             reinterpret_cast<const char*>(&enabled), sizeof(enabled));
#else
  setsockopt(listener.get(), SOL_SOCKET, SO_REUSEADDR, &enabled,
             sizeof(enabled));
#endif
  sockaddr_in address{};
  address.sin_family = AF_INET;
  address.sin_port = htons(static_cast<uint16_t>(options.port));
  if (inet_pton(AF_INET, options.address.c_str(), &address.sin_addr) != 1) {
    throw std::invalid_argument("Bind address must be an IPv4 literal");
  }
  if (bind(listener.get(), reinterpret_cast<const sockaddr*>(&address),
           sizeof(address)) ||
      listen(listener.get(), options.max_connections)) {
    throw std::runtime_error("Cannot bind or listen on requested TCP address");
  }
  nonblocking(listener.get());
  SocketLength length = sizeof(address);
  if (getsockname(listener.get(), reinterpret_cast<sockaddr*>(&address), &length))
    throw std::runtime_error("Cannot determine the bound TCP address");
  const int port = ntohs(address.sin_port);
  std::cout << "Serving " << scheme << "://" << options.address << ':'
            << port << '\n' << std::flush;
  struct Client {
    std::thread thread;
    std::shared_ptr<std::atomic<bool>> done;
  };
  std::vector<Client> clients;
  clients.reserve(size_t(options.max_connections));
  std::atomic<bool> stopping{false};
  LocalFunction join{[&] {
    stopping.store(true);
    for (Client& client : clients) {
      if (client.thread.joinable()) {
        client.thread.join();
      }
    }
  }};
  try {
    if (ready && !stop()) ready(port);
    while (!stop()) {
      for (size_t i = 0; i < clients.size();) {
        if (!clients[i].done->load()) {
          ++i;
          continue;
        }
        clients[i].thread.join();
        clients.erase(clients.begin() + i);
      }
      if (!poll_socket(listener.get(), false, 25)) {
        continue;
      }
      Socket accepted(accept(listener.get(), nullptr, nullptr));
      if (accepted.get() == invalid_socket) {
        if (would_block(socket_error())) {
          continue;
        }
        throw std::runtime_error("TCP accept failed");
      }
      nonblocking(accepted.get());
      ++counts.accepted;
      if (clients.size() >= size_t(options.max_connections)) {
        ++counts.rejected;
        try {
          send_all(accepted.get(), busy_reply, 25, stopping);
          finish_rejection(accepted.get());
        } catch (...) {
        }
        continue;
      }
      int send_buffer = 16 * 1024;
      setsockopt(accepted.get(), SOL_SOCKET, SO_SNDBUF,
                 reinterpret_cast<const char*>(&send_buffer),
                 sizeof(send_buffer));
      setsockopt(accepted.get(), IPPROTO_TCP, TCP_NODELAY,
                 reinterpret_cast<const char*>(&enabled), sizeof(enabled));
      std::shared_ptr<std::atomic<bool>> done = std::make_shared<std::atomic<bool>>(false);
      std::thread client([socket = std::move(accepted), &handle, &options,
                          &stopping, done]() mutable {
        TcpStream channel{
            [&](char* destination, size_t capacity,
                Clock::time_point deadline) -> size_t {
              for (;;) {
                if (stopping.load()) {
                  throw ConnectionClosed();
                }
                if (Clock::now() >= deadline) {
                  throw ReadTimeout();
                }
                if (!poll_socket(socket.get(), false, 25)) {
                  continue;
                }
                int n = recv(socket.get(), destination,
                             int(std::min(capacity, size_t(65536))), 0);
                if (n < 0 && would_block(socket_error())) {
                  continue;
                }
                if (n <= 0) {
                  throw ConnectionClosed();
                }
                return static_cast<size_t>(n);
              }
            },
            [&](const std::string& data) {
              send_all(socket.get(), data, options.write_timeout_ms, stopping);
            },
            [&] { return stopping.load() || disconnected(socket.get()); }};
        try {
          handle(channel);
        } catch (const ConnectionClosed&) {
        } catch (const std::exception& e) {
          std::cerr << "TCP connection: " << e.what() << '\n';
        }
        done->store(true);
      });
      clients.push_back(Client{std::move(client), std::move(done)});
    }
  } catch (...) {
    join();
    throw;
  }
  join();
}

}  // namespace elman
