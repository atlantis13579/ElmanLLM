#pragma once

#include "ChatService.h"
#include "ImageService.h"

#include <filesystem>
#include <functional>
#include <variant>

namespace elman {
using HostedModel = std::variant<ChatService, ImageService>;

// One HTTP/1.1 request per connection. Socket I/O is nonblocking with bounded
// read/write/request deadlines. The stop predicate belongs to the accept thread.
// Returns only after connection threads and their cancellation signals finish.
void serve_http(HostedModel& model, const HttpOptions& options, const std::filesystem::path& web_root,
                const std::function<bool()>& stop, const std::function<void(int)>& ready);
} // namespace elman
