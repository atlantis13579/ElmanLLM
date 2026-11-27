#include "Networking/HttpServer.h"

#include <algorithm>
#include <charconv>
#include <fstream>
#include <system_error>

namespace elman {
namespace {

std::filesystem::path asset_path(std::string_view path) {
  if (path.empty() || path.front() != '/') return {};
  if (path == "/") return "Index.html";
  std::string name;
  for (size_t i = 1; i < path.size(); ++i) {
    unsigned char byte = static_cast<unsigned char>(path[i]);
    if (byte == '%') {
      if (path.size() - i < 3) return {};
      const char* const end = path.data() + i + 3;
      const std::from_chars_result result = std::from_chars(path.data() + i + 1, end, byte, 16);
      if (result.ec != std::errc{} || result.ptr != end) return {};
      i += 2;
    }
    if (byte < 32 || byte == 127 || byte == '\\' || byte == ':') return {};
    name.push_back(static_cast<char>(byte));
  }
  std::filesystem::path relative;
  try { relative = std::filesystem::u8path(name); }
  catch (const std::system_error&) { return {}; }
  if (relative.has_root_path()) return {};
  for (const std::filesystem::path& part : relative) {
    const std::filesystem::path::string_type& component = part.native();
    if (component.empty() || component.front() == '.' ||
        component.back() == '.' || component.back() == ' ') return {};
  }
  return relative;
}

std::string_view asset_mime(const std::filesystem::path& file) {
  std::string extension = file.extension().u8string();
  for (char& ch : extension) if (ch >= 'A' && ch <= 'Z') ch += 'a' - 'A';
  static constexpr std::pair<std::string_view, std::string_view> types[] = {
      {".html", "text/html; charset=utf-8"}, {".css", "text/css; charset=utf-8"},
      {".js", "text/javascript; charset=utf-8"}, {".mjs", "text/javascript; charset=utf-8"},
      {".json", "application/json"}, {".map", "application/json"},
      {".webmanifest", "application/manifest+json"}, {".wasm", "application/wasm"},
      {".svg", "image/svg+xml"}, {".png", "image/png"}, {".jpg", "image/jpeg"},
      {".jpeg", "image/jpeg"}, {".gif", "image/gif"}, {".webp", "image/webp"},
      {".avif", "image/avif"}, {".ico", "image/x-icon"},
      {".woff", "font/woff"}, {".woff2", "font/woff2"}, {".ttf", "font/ttf"},
      {".otf", "font/otf"}, {".txt", "text/plain; charset=utf-8"}
  };
  for (const std::pair<std::string_view, std::string_view>& entry : types) {
    const std::string_view& suffix = entry.first;
    const std::string_view& type = entry.second;
    if (extension == suffix) return type;
  }
  return "application/octet-stream";
}

}  // namespace

bool send_http_file(const HttpConnection& connection, const std::filesystem::path& root,
                    std::string_view path, bool head) {
  const std::filesystem::path relative = asset_path(path);
  if (relative.empty()) return false;
  std::error_code error;
  const std::filesystem::path file = std::filesystem::canonical(root / relative, error);
  if (error) return false;
  const std::filesystem::path contained = file.lexically_relative(root);
  if (contained.empty() || *contained.begin() == ".." ||
      !std::filesystem::is_regular_file(file, error)) return false;

  std::ifstream stream(file, std::ios::binary | std::ios::ate);
  const std::streampos length = stream.tellg();
  if (length < 0 || !stream.seekg(0)) return false;
  uintmax_t remaining = static_cast<uintmax_t>(length);
  connection.send(http_response_header(200, asset_mime(relative), remaining));
  if (head) return true;
  constexpr size_t chunk_bytes = 64 * 1024;
  std::string chunk;
  while (remaining) {
    chunk.resize(static_cast<size_t>(std::min<uintmax_t>(remaining, chunk_bytes)));
    if (!stream.read(chunk.data(), static_cast<std::streamsize>(chunk.size()))) throw ConnectionClosed();
    connection.send(chunk);
    remaining -= chunk.size();
  }
  return true;
}

}  // namespace elman
