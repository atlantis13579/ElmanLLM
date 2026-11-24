#include "Networking/HttpServer.h"
#include "Core/LocalFunction.h"

#include <algorithm>

namespace elman {
namespace {
using Clock = std::chrono::steady_clock;
std::string lower(std::string text) {
  for (char& c : text) {
    if (c >= 'A' && c <= 'Z') {
      c += 'a' - 'A';
    }
  }
  return text;
}
bool token_char(unsigned char c) {
  return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
         (c >= '0' && c <= '9') ||
         std::string("!#$%&'*+-.^_`|~").find(char(c)) != std::string::npos;
}
std::string trim(const std::string& value) {
  size_t begin = value.find_first_not_of(" \t");
  if (begin == std::string::npos) {
    return {};
  }
  return value.substr(begin, value.find_last_not_of(" \t") - begin + 1);
}
HttpRequest receive(const TcpStream& stream, const HttpOptions& options) {
  std::string buffer;
  std::chrono::steady_clock::time_point deadline =
      Clock::now() + std::chrono::milliseconds(options.read_timeout_ms);
  LocalFunction read_more{[&](size_t cap) {
    char bytes[4096];
    try {
      buffer.append(bytes,
                    stream.read(bytes, std::min(cap, sizeof(bytes)), deadline));
    } catch (const ReadTimeout&) {
      throw HttpError(408, "read_timeout", "Request read deadline exceeded");
    }
  }};
  size_t end;
  while ((end = buffer.find("\r\n\r\n")) == std::string::npos) {
    if (buffer.size() >= options.header_bytes) {
      throw HttpError(431, "headers_too_large",
                      "Request headers exceed the limit");
    }
    read_more(options.header_bytes - buffer.size());
  }
  if (end + 4 > options.header_bytes) {
    throw HttpError(431, "headers_too_large",
                    "Request headers exceed the limit");
  }
  HttpRequest request;
  size_t first = buffer.find("\r\n");
  std::string line = buffer.substr(0, first);
  size_t space = line.find(' '), last = line.rfind(' ');
  if (space == std::string::npos || space == last ||
      line.substr(last + 1) != "HTTP/1.1" ||
      line.find(' ', space + 1) != last) {
    throw HttpError(400, "invalid_http", "Expected an HTTP/1.1 request line");
  }
  request.method = line.substr(0, space);
  request.target = line.substr(space + 1, last - space - 1);
  if (request.method.empty() ||
      !std::all_of(request.method.begin(), request.method.end(), token_char) ||
      request.target.empty() || request.target[0] != '/' ||
      !std::all_of(request.target.begin(), request.target.end(),
                   [](unsigned char c) { return c > 32 && c < 127; })) {
    throw HttpError(400, "invalid_http",
                    "Invalid HTTP method or origin target");
  }
  for (size_t position = first + 2; position < end;) {
    size_t next = buffer.find("\r\n", position);
    std::string field = buffer.substr(position, next - position);
    position = next + 2;
    size_t colon = field.find(':');
    if (colon == std::string::npos || !colon ||
        !std::all_of(field.begin(), field.begin() + colon, token_char)) {
      throw HttpError(400, "invalid_http",
                      "Invalid header name or obsolete line folding");
    }
    std::string name = lower(field.substr(0, colon)),
         data = trim(field.substr(colon + 1));
    if (!std::all_of(data.begin(), data.end(), [](unsigned char c) {
          return c == '\t' || (c >= 32 && c != 127);
        })) {
      throw HttpError(400, "invalid_http", "Invalid header value");
    }
    if (!request.headers.emplace(name, data).second) {
      throw HttpError(400, "duplicate_header", "Duplicate HTTP header");
    }
  }
  if (!request.headers.count("host") || request.headers.at("host").empty()) {
    throw HttpError(400, "invalid_http", "Host header is required");
  }
  if (request.headers.count("transfer-encoding")) {
    throw HttpError(400, "unsupported_framing",
                    "Transfer-Encoding is not supported; send Content-Length");
  }
  if (request.headers.count("expect")) {
    throw HttpError(417, "unsupported_expectation", "Expect is not supported");
  }
  if (request.headers.count("content-encoding") &&
      lower(request.headers.at("content-encoding")) != "identity") {
    throw HttpError(415, "unsupported_encoding",
                    "Compressed request bodies are not supported");
  }
  size_t length = 0;
  if (request.headers.count("content-length")) {
    std::string data = request.headers.at("content-length");
    if (data.empty() || !std::all_of(data.begin(), data.end(), [](char c) {
          return c >= '0' && c <= '9';
        })) {
      throw HttpError(400, "invalid_length", "Invalid Content-Length");
    }
    for (char digit : data) {
      if (length > options.body_bytes / 10 ||
          (length == options.body_bytes / 10 &&
           size_t(digit - '0') > options.body_bytes % 10)) {
        throw HttpError(413, "body_too_large",
                        "Request body exceeds the limit");
      }
      length = length * 10 + size_t(digit - '0');
    }
  } else if (request.method == "POST") {
    throw HttpError(411, "length_required", "Content-Length is required");
  }
  buffer.erase(0, end + 4);
  if (buffer.size() > length) {
    throw HttpError(400, "pipelining_unsupported",
                    "Only one request is supported per connection");
  }
  while (buffer.size() < length) {
    read_more(length - buffer.size());
  }
  request.body = std::move(buffer);
  return request;
}
const char* status_text(int status) {
  switch (status) {
    case 200:
      return "OK";
    case 400:
      return "Bad Request";
    case 403:
      return "Forbidden";
    case 404:
      return "Not Found";
    case 405:
      return "Method Not Allowed";
    case 408:
      return "Request Timeout";
    case 409:
      return "Conflict";
    case 411:
      return "Length Required";
    case 413:
      return "Payload Too Large";
    case 415:
      return "Unsupported Media Type";
    case 417:
      return "Expectation Failed";
    case 429:
      return "Too Many Requests";
    case 431:
      return "Request Header Fields Too Large";
    case 503:
      return "Service Unavailable";
    default:
      return "Internal Server Error";
  }
}
}  // namespace
std::string http_response(int status, std::string_view mime,
                          std::string_view body, std::string_view headers) {
  std::string message = http_response_header(status, mime, body.size(), headers);
  message += body;
  return message;
}
std::string http_response_header(int status, std::string_view mime,
                                 uintmax_t length, std::string_view headers) {
  std::string message =
      "HTTP/1.1 " + std::to_string(status) + " " + status_text(status) +
      "\r\nContent-Type: " + std::string(mime) +
      "\r\nContent-Length: " + std::to_string(length) +
      "\r\nCache-Control: no-store\r\nX-Content-Type-Options: nosniff\r\n"
      "Connection: close\r\n";
  message.append(headers);
  message += "\r\n";
  return message;
}
void listen_http(const HttpOptions& options, const std::function<bool()>& stop,
                 const std::function<void(const HttpConnection&)>& handle,
                 ConnectionCounts& counts,
                 const std::function<void(int)>& ready) {
  if (options.header_bytes < 64 || !options.body_bytes ||
      options.request_timeout_ms <= 0) {
    throw std::invalid_argument("Invalid HTTP limits");
  }
  const std::string busy = http_response(
      429, "application/json",
      R"({"error":{"message":"Connection limit reached","code":"connections_full","type":"rate_limit_error"}})");
  listen_tcp(
      options, "http", busy, stop,
      [&](const TcpStream& stream) {
        handle({[&] { return receive(stream, options); }, stream.send,
                stream.closed});
      },
      counts, ready);
}
}  // namespace elman
