#pragma once

#include "ChatService.h"
#include "ImageService.h"

#include <cstddef>
#include <functional>
#include <string>
#include <variant>

namespace elman {
using HostedModel = std::variant<ChatService, ImageService>;

struct HttpOptions {
    std::string address = "127.0.0.1";
    int port = 8080, max_connections = 32;
    size_t header_bytes = 16 * 1024, body_bytes = 1024 * 1024;
    int read_timeout_ms = 5000, write_timeout_ms = 5000, request_timeout_ms = 120000;
};
// One HTTP/1.1 request per connection. Socket I/O is nonblocking with bounded
// read/write/request deadlines. The stop predicate belongs to the accept thread.
// Returns only after connection threads and their cancellation signals finish.
void serve_http(HostedModel& model, const HttpOptions& options, const std::function<bool()>& stop);
} // namespace elman
