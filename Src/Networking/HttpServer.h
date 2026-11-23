#pragma once
#include <filesystem>
#include <functional>
#include <map>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "Networking/TcpServer.h"

namespace elman {
struct HttpOptions : TcpOptions {
  size_t header_bytes = 16 * 1024, body_bytes = 1024 * 1024;
  int request_timeout_ms = 120000;
  std::vector<std::string> cors_origins;
};
struct HttpRequest {
  std::string method, target, body;
  std::map<std::string, std::string> headers;
};
class HttpError : public std::runtime_error {
 public:
  HttpError(int status, std::string code, const std::string& message)
      : std::runtime_error(message), status(status), code(std::move(code)) {
  }
  int status;
  std::string code;
};

struct HttpConnection {
  std::function<HttpRequest()> receive;
  std::function<void(const std::string&)> send;
  std::function<bool()> closed;
};
std::string http_response(int status, std::string_view mime,
                          std::string_view body, std::string_view headers = {});
std::string http_response_header(int status, std::string_view mime,
                                 uintmax_t length, std::string_view headers = {});
// The root is a canonical directory. Missing files are left to the caller.
bool send_http_file(const HttpConnection& connection, const std::filesystem::path& root,
                    std::string_view path, bool head);
void listen_http(const HttpOptions& options, const std::function<bool()>& stop,
                 const std::function<void(const HttpConnection&)>& handle,
                 ConnectionCounts& counts,
                 const std::function<void(int)>& ready);
}  // namespace elman
