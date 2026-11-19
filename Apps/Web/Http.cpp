#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <winsock2.h>
#include <ws2tcpip.h>
#else
#include <arpa/inet.h>
#include <cerrno>
#include <fcntl.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <sys/select.h>
#include <sys/socket.h>
#include <unistd.h>
#endif

#include "Http.h"
#include "Assets.h"
#include "ChatResponse.h"
#include "GpuMonitor.h"
#include "ServerLog.h"
#include <algorithm>
#include <atomic>
#include <charconv>
#include <chrono>
#include <iostream>
#include <thread>

namespace elman {
namespace {
using Clock = std::chrono::steady_clock;
#ifdef _WIN32
using NativeSocket = SOCKET;
constexpr NativeSocket invalid_socket = INVALID_SOCKET;
int socket_error() { return WSAGetLastError(); }
bool would_block(int code) { return code == WSAEWOULDBLOCK || code == WSAEINTR; }
void close_socket(NativeSocket value) { closesocket(value); }
using SocketLength = int;
struct SocketRuntime {
    SocketRuntime() { WSADATA data{}; if (WSAStartup(MAKEWORD(2, 2), &data)) throw std::runtime_error("WSAStartup failed"); }
    ~SocketRuntime() { WSACleanup(); }
};
#else
using NativeSocket = int;
constexpr NativeSocket invalid_socket = -1;
int socket_error() { return errno; }
bool would_block(int code) { return code == EWOULDBLOCK || code == EAGAIN || code == EINTR; }
void close_socket(NativeSocket value) { close(value); }
using SocketLength = socklen_t;
struct SocketRuntime {};
#endif
class Socket {
public:
    explicit Socket(NativeSocket value = invalid_socket) : value_(value) {
#ifdef SO_NOSIGPIPE
        int enabled = 1;
        if (value_ != invalid_socket && setsockopt(value_, SOL_SOCKET, SO_NOSIGPIPE, &enabled, sizeof(enabled)) != 0) {
            close_socket(value_);
            throw std::runtime_error("Cannot disable SIGPIPE on HTTP socket");
        }
#endif
    }
    ~Socket() { if (value_ != invalid_socket) close_socket(value_); }
    Socket(Socket&& other) noexcept : value_(other.value_) { other.value_ = invalid_socket; }
    Socket(const Socket&) = delete;
    NativeSocket get() const { return value_; }
private:
    NativeSocket value_;
};
class ConnectionClosed : public std::runtime_error {
public: ConnectionClosed() : std::runtime_error("Connection closed or timed out") {}
};
void nonblocking(NativeSocket value) {
#ifdef _WIN32
    u_long mode = 1; if (ioctlsocket(value, FIONBIO, &mode)) throw std::runtime_error("Cannot set nonblocking socket");
#else
    int flags = fcntl(value, F_GETFL, 0);
    if (flags < 0 || fcntl(value, F_SETFL, flags | O_NONBLOCK) < 0) throw std::runtime_error("Cannot set nonblocking socket");
#endif
}
bool poll_socket(NativeSocket value, bool write, int milliseconds) {
#ifdef _WIN32
    fd_set set; FD_ZERO(&set); FD_SET(value, &set);
    timeval timeout{milliseconds / 1000, (milliseconds % 1000) * 1000};
    int result = select(int(value + 1), write ? nullptr : &set, write ? &set : nullptr, nullptr, &timeout);
#else
    pollfd descriptor{value, short(write ? POLLOUT : POLLIN), 0};
    int result = poll(&descriptor, 1, milliseconds);
#endif
    if (result < 0 && !would_block(socket_error())) throw ConnectionClosed();
    return result > 0;
}
bool disconnected(NativeSocket value) {
    if (!poll_socket(value, false, 0)) return false;
    char byte; int count = recv(value, &byte, 1, MSG_PEEK);
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
    const auto deadline = Clock::now() + std::chrono::milliseconds(25);
    size_t remaining = 64 * 1024;
    while (remaining && Clock::now() < deadline) {
        if (!poll_socket(value, false, 5)) continue;
        char bytes[4096]; int count = recv(value, bytes, int(std::min(remaining, sizeof(bytes))), 0);
        if (count < 0 && would_block(socket_error())) continue;
        if (count <= 0) break;
        remaining -= size_t(count);
    }
}
void send_all(NativeSocket value, const std::string& text, int timeout_ms, const std::atomic<bool>& stopping) {
    size_t offset = 0; const auto deadline = Clock::now() + std::chrono::milliseconds(timeout_ms);
    while (offset < text.size()) {
        if (stopping.load() || Clock::now() >= deadline) throw ConnectionClosed();
        if (!poll_socket(value, true, 25)) continue;
#ifdef MSG_NOSIGNAL
        constexpr int flags = MSG_NOSIGNAL;
#else
        constexpr int flags = 0;
#endif
        int sent = send(value, text.data() + offset, int(std::min(size_t(1024 * 1024), text.size() - offset)), flags);
        if (sent < 0 && would_block(socket_error())) continue;
        if (sent <= 0) throw ConnectionClosed();
        offset += size_t(sent);
    }
}
std::string lower(std::string text) {
    for (char& c : text) if (c >= 'A' && c <= 'Z') c += 'a' - 'A'; return text;
}
bool token_char(unsigned char c) {
    return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || std::string("!#$%&'*+-.^_`|~").find(char(c)) != std::string::npos;
}
std::string trim(const std::string& value) {
    size_t begin = value.find_first_not_of(" \t"); if (begin == std::string::npos) return {};
    return value.substr(begin, value.find_last_not_of(" \t") - begin + 1);
}
struct HttpRequest { std::string method, target, body; std::map<std::string, std::string> headers; };
HttpRequest receive(NativeSocket value, const HttpOptions& options, const std::atomic<bool>& stopping) {
    std::string buffer;
    auto deadline = Clock::now() + std::chrono::milliseconds(options.read_timeout_ms);
    auto read_more = [&](size_t cap) {
        for (;;) {
            if (stopping.load()) throw ConnectionClosed();
            if (Clock::now() >= deadline) throw ChatError(408, "read_timeout", "Request read deadline exceeded");
            if (!poll_socket(value, false, 25)) continue;
            char bytes[4096]; int count = recv(value, bytes, int(std::min(cap, sizeof(bytes))), 0);
            if (count < 0 && would_block(socket_error())) continue;
            if (count <= 0) throw ConnectionClosed();
            buffer.append(bytes, size_t(count)); return;
        }
    };
    size_t end;
    while ((end = buffer.find("\r\n\r\n")) == std::string::npos) {
        if (buffer.size() >= options.header_bytes) throw ChatError(431, "headers_too_large", "Request headers exceed the limit");
        read_more(options.header_bytes - buffer.size());
    }
    if (end + 4 > options.header_bytes) throw ChatError(431, "headers_too_large", "Request headers exceed the limit");
    HttpRequest request;
    auto first = buffer.find("\r\n"); auto line = buffer.substr(0, first);
    auto space = line.find(' '), last = line.rfind(' ');
    if (space == std::string::npos || space == last || line.substr(last + 1) != "HTTP/1.1" ||
        line.find(' ', space + 1) != last) throw ChatError(400, "invalid_http", "Expected an HTTP/1.1 request line");
    request.method = line.substr(0, space); request.target = line.substr(space + 1, last - space - 1);
    if (request.method.empty() || !std::all_of(request.method.begin(), request.method.end(), token_char) || request.target.empty() || request.target[0] != '/' ||
        !std::all_of(request.target.begin(), request.target.end(), [](unsigned char c) { return c > 32 && c < 127; }))
        throw ChatError(400, "invalid_http", "Invalid HTTP method or origin target");
    for (size_t position = first + 2; position < end;) {
        size_t next = buffer.find("\r\n", position); auto field = buffer.substr(position, next - position); position = next + 2;
        auto colon = field.find(':');
        if (colon == std::string::npos || !colon || !std::all_of(field.begin(), field.begin() + colon, token_char))
            throw ChatError(400, "invalid_http", "Invalid header name or obsolete line folding");
        auto name = lower(field.substr(0, colon)), data = trim(field.substr(colon + 1));
        if (!std::all_of(data.begin(), data.end(), [](unsigned char c) { return c == '\t' || (c >= 32 && c != 127); }))
            throw ChatError(400, "invalid_http", "Invalid header value");
        if (!request.headers.emplace(name, data).second) throw ChatError(400, "duplicate_header", "Duplicate HTTP header");
    }
    if (!request.headers.count("host") || request.headers.at("host").empty()) throw ChatError(400, "invalid_http", "Host header is required");
    if (request.headers.count("transfer-encoding")) throw ChatError(400, "unsupported_framing", "Transfer-Encoding is not supported; send Content-Length");
    if (request.headers.count("expect")) throw ChatError(417, "unsupported_expectation", "Expect is not supported");
    if (request.headers.count("content-encoding") && lower(request.headers.at("content-encoding")) != "identity")
        throw ChatError(415, "unsupported_encoding", "Compressed request bodies are not supported");
    size_t length = 0;
    if (request.headers.count("content-length")) {
        auto data = request.headers.at("content-length");
        if (data.empty() || !std::all_of(data.begin(), data.end(), [](char c) { return c >= '0' && c <= '9'; }))
            throw ChatError(400, "invalid_length", "Invalid Content-Length");
        for (char digit : data) {
            if (length > options.body_bytes / 10 || (length == options.body_bytes / 10 && size_t(digit - '0') > options.body_bytes % 10))
                throw ChatError(413, "body_too_large", "Request body exceeds the limit");
            length = length * 10 + size_t(digit - '0');
        }
    } else if (request.method == "POST") throw ChatError(411, "length_required", "Content-Length is required");
    buffer.erase(0, end + 4);
    if (buffer.size() > length) throw ChatError(400, "pipelining_unsupported", "Only one request is supported per connection");
    while (buffer.size() < length) read_more(length - buffer.size());
    request.body = std::move(buffer); return request;
}
const char* status_text(int status) {
    switch (status) {
    case 200: return "OK"; case 400: return "Bad Request"; case 404: return "Not Found";
    case 405: return "Method Not Allowed"; case 408: return "Request Timeout"; case 409: return "Conflict"; case 411: return "Length Required";
    case 413: return "Payload Too Large"; case 415: return "Unsupported Media Type"; case 417: return "Expectation Failed";
    case 429: return "Too Many Requests"; case 431: return "Request Header Fields Too Large";
    case 503: return "Service Unavailable"; default: return "Internal Server Error";
    }
}
std::string response(int status, std::string_view mime, std::string_view body) {
    auto message = "HTTP/1.1 " + std::to_string(status) + " " + status_text(status) + "\r\nContent-Type: ";
    message.append(mime);
    message += "\r\nContent-Length: " + std::to_string(body.size()) +
        "\r\nCache-Control: no-store\r\nX-Content-Type-Options: nosniff\r\n"
        "Content-Security-Policy: default-src 'self'; img-src 'self' blob:; base-uri 'none'; frame-ancestors 'none'\r\n"
        "Connection: close\r\n\r\n";
    message.append(body);
    return message;
}
std::string response(int status, const Json& data) {
    return response(status, "application/json; charset=utf-8", data.dump(0));
}
std::string event(const Json& data) { return "data: " + data.dump(0) + "\n\n"; }
constexpr char event_stream_header[] =
    "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream; charset=utf-8\r\nCache-Control: no-cache\r\n"
    "Connection: close\r\nX-Accel-Buffering: no\r\n\r\n";

std::string_view progress_parameter(std::string_view query, std::string_view key) {
    while (!query.empty()) {
        const auto end = query.find('&');
        const auto field = query.substr(0, end);
        if (field.size() > key.size() && field[key.size()] == '=' && field.substr(0, key.size()) == key)
            return field.substr(key.size() + 1);
        if (end == query.npos) break;
        query.remove_prefix(end + 1);
    }
    return {};
}

struct HttpStats {
    std::atomic<uint64_t> accepted{0}, completed{0}, errors{0}, disconnected{0}, rejected{0};
    std::mutex mutex;
    double wall_ms = 0, first_text_ms = 0;
    uint64_t first_text_count = 0;
};
Json metrics(const ChatService* service, Backend backend, HttpStats& http) {
    const auto s = service ? service->stats() : SchedulerStats{};
    std::lock_guard<std::mutex> lock(http.mutex);
    Json::Object result{{"http_connections", double(http.accepted.load())}, {"http_completions", double(http.completed.load())}, {"http_errors", double(http.errors.load())},
        {"http_disconnects", double(http.disconnected.load())}, {"http_rejected_connections", double(http.rejected.load())},
        {"completion_wall_ms_sum", http.wall_ms},
        {"tracked_host_bytes", double(memory_stats(Backend::CPU).live_bytes())}, {"tracked_device_bytes", double(memory_stats(backend).live_bytes())},
        {"tracked_host_peak_bytes", double(memory_stats(Backend::CPU).peak_bytes)}, {"tracked_device_peak_bytes", double(memory_stats(backend).peak_bytes)}};
    if (service) result.merge(Json::Object{{"queued_requests", double(s.queued)}, {"active_requests", double(s.active)}, {"reserved_cache_blocks", double(s.reserved_blocks)},
        {"live_cache_blocks", double(s.live_blocks)}, {"capacity_cache_blocks", double(s.capacity_blocks)}, {"peak_cache_blocks", double(s.peak_live_blocks)},
        {"submitted_requests", double(s.submitted)}, {"completed_requests", double(s.completed)}, {"cancelled_requests", double(s.cancelled)},
        {"failed_requests", double(s.failed)}, {"model_steps", double(s.model_steps)}, {"input_tokens", double(s.input_tokens)}, {"memory_retries", double(s.memory_retries)},
        {"cuda_graph_captures", double(s.graph_captures)}, {"cuda_graph_replays", double(s.graph_replays)},
        {"prefix_cache_entries", double(s.prefix.entries)}, {"prefix_cache_retained_blocks", double(s.prefix.retained_blocks)},
        {"prefix_cache_hits", double(s.prefix.hits)}, {"prefix_cache_misses", double(s.prefix.misses)},
        {"prefix_cache_reused_tokens", double(s.prefix.reused_tokens)}, {"prefix_cache_publications", double(s.prefix.publications)},
        {"prefix_cache_evictions", double(s.prefix.evictions)}, {"prefix_cache_allocation_failures", double(s.prefix.allocation_failures)},
        {"first_text_ms_sum", http.first_text_ms}, {"first_text_samples", double(http.first_text_count)}});
    return result;
}
void connection(Socket socket, HostedModel& hosted, const HttpOptions& options, const std::atomic<bool>& stopping, HttpStats& stats) {
    auto* chat_service = std::get_if<ChatService>(&hosted);
    auto* image_service = std::get_if<ImageService>(&hosted);
    const auto& model = chat_service ? chat_service->options().model_name : image_service->name();
    ChatRequest chat;
    bool streaming = false, image_request = false;
    auto send = [&](const std::string& text) { send_all(socket.get(), text, options.write_timeout_ms, stopping); };
    auto fail = [&](int status, const std::string& code, const std::string& message) {
        ++stats.errors;
        try {
            if (image_request) {
                Json data = Json::Object{{"error", code}, {"detail", message}};
                send(streaming ? event(data) : response(status, data));
            }
            else send(streaming ? event(chat_error(message, code, status)) + "data: [DONE]\n\n" : response(status, chat_error(message, code, status)));
        } catch (...) {}
    };
    try {
        MemoryReservation body_memory(Backend::CPU, checked_memory_add(checked_memory_mul(options.body_bytes, 8), checked_memory_mul(options.header_bytes, 2)));
        auto request = receive(socket.get(), options, stopping);
        if (request.method == "GET") {
            if (request.target == "/") { send(response(200, "text/html; charset=utf-8", assets::index)); return; }
            if (request.target == "/style.css") { send(response(200, "text/css; charset=utf-8", assets::style)); return; }
            if (request.target == "/app.js") { send(response(200, "text/javascript; charset=utf-8", assets::script)); return; }
            if (request.target == "/health") {
                Json::Object health{{"status", "ok"}, {"model", model},
                    {"model_type", chat_service ? "llm" : "diffusion"},
                    {"chat_text", chat_service && !chat_service->options().token_ids_only},
                    {"image_generation", image_service != nullptr}};
                if (image_service) health.emplace("image_sampler", image_service->sampler());
                send(response(200, std::move(health))); return;
            }
            if (request.target == "/metrics") {
                auto backend = chat_service ? chat_service->options().backend : image_service->backend();
                send(response(200, metrics(chat_service, backend, stats))); return;
            }
            if (chat_service && request.target == "/v1/models") {
                Json entry = Json::Object{{"id", model}, {"object", "model"}, {"created", 0}, {"owned_by", "elman"}};
                send(response(200, Json::Object{{"object", "list"}, {"data", Json::Array{entry}}})); return;
            }
            const std::string_view target = request.target;
            const auto separator = target.find('?');
            if (image_service && target.substr(0, separator) == "/sdapi/v1/progress") {
                const auto query = separator == target.npos ? std::string_view{} : target.substr(separator + 1);
                const auto id = progress_parameter(query, "task_id");
                const auto cursor = progress_parameter(query, "after_preview_step");
                int after_preview_step = 0;
                if (!cursor.empty()) {
                    const auto end = cursor.data() + cursor.size();
                    const auto result = std::from_chars(cursor.data(), end, after_preview_step);
                    if (result.ec != std::errc{} || result.ptr != end || after_preview_step < 0)
                        throw ChatError(400, "invalid_preview_step", "after_preview_step must be a nonnegative integer");
                }
                const auto skip = progress_parameter(query, "skip_current_image");
                send(response(200, image_service->progress(id, after_preview_step, skip == "true" || skip == "1")));
                return;
            }
        }
        image_request = request.target == "/sdapi/v1/txt2img";
        const bool chat_request = request.target == "/v1/chat/completions";
        if (!(image_request && image_service) && !(chat_request && chat_service))
            throw ChatError(404, "not_found", "This endpoint is unavailable for the loaded model");
        if (request.method != "POST") throw ChatError(405, "method_not_allowed", "This endpoint requires POST");
        auto content_type = request.headers.find("content-type");
        if (content_type == request.headers.end() || trim(lower(content_type->second.substr(0, content_type->second.find(';')))) != "application/json")
            throw ChatError(415, "unsupported_media_type", "Content-Type must be application/json");
        Json body;
        try { utf16(request.body); body = Json::parse(request.body); }
        catch (const std::exception& error) { throw ChatError(400, "invalid_json", error.what()); }
        auto started = Clock::now();
        auto deadline = started + std::chrono::milliseconds(options.request_timeout_ms);
        auto check = [&] {
            if (stopping.load() || disconnected(socket.get())) throw ConnectionClosed();
            if (Clock::now() >= deadline) throw ChatError(408, "generation_timeout", "Request deadline exceeded");
        };
        auto complete = [&] {
            ++stats.completed;
            std::lock_guard<std::mutex> lock(stats.mutex);
            stats.wall_ms += std::chrono::duration<double, std::milli>(Clock::now() - started).count();
        };
        if (image_request) {
            const auto accept = request.headers.find("accept");
            if (accept != request.headers.end() && lower(accept->second) == "text/event-stream") {
                auto emit = [&](const Json& data) {
                    if (!streaming) {
                        send(event_stream_header);
                        streaming = true;
                    }
                    send(event(data));
                };
                auto result = image_service->generate(body, check, emit);
                emit(Json::Object{{"type", "result"}, {"result", std::move(result)}});
            } else {
                send(response(200, image_service->generate(body, check)));
            }
            complete();
            return;
        }
        check();
        chat = chat_service->submit(body);
        if (chat.stream) {
            send(event_stream_header);
            streaming = true; send(event(chat_chunk(chat, model, Json::Object{{"role", "assistant"}, {"content", ""}})));
        }
        bool first_text = false;
        auto emit = [&](const std::string& piece) {
            if (piece.empty()) return;
            if (!first_text) {
                std::lock_guard<std::mutex> lock(stats.mutex); ++stats.first_text_count;
                stats.first_text_ms += std::chrono::duration<double, std::milli>(Clock::now() - started).count(); first_text = true;
            }
            send(event(chat_chunk(chat, model, Json::Object{{"content", piece}})));
        };
        while (true) {
            check();
            if (chat_service->stopped()) throw ConnectionClosed();
            if (chat.stream) emit(chat.output->pop(std::chrono::milliseconds(10)));
            else chat.generation.wait_for(std::chrono::milliseconds(10));
            if (!chat.generation.ready()) continue;
            if (chat.generation.error()) std::rethrow_exception(chat.generation.error());
            chat_finish_reason(chat.generation.result());
            std::string tail; auto result = chat.output->finish(chat.generation.result(), tail);
            if (chat.stream) {
                for (;;) { auto piece = chat.output->pop(std::chrono::milliseconds(0)); if (piece.empty()) break; emit(piece); }
                emit(tail);
                send(event(chat_chunk(chat, model, Json::Object{}, chat_finish_reason(result))));
                if (chat.include_usage) send(event(chat_chunk(chat, model, Json::Object{}, {}, chat_usage(chat.prompt_tokens, result.ids.size()), true)));
                send("data: [DONE]\n\n");
            } else send(response(200, chat_completion(chat, model, result)));
            complete();
            break;
        }
    } catch (const ConnectionClosed&) { ++stats.disconnected; }
      catch (const ChatError& error) { fail(error.status, error.code, error.what()); }
      catch (const RequestQueueFull&) { fail(429, "queue_full", "Waiting request queue is full"); }
      catch (const SchedulerClosed&) { fail(503, "service_closed", "Service is shutting down"); }
      catch (const MemoryLimitExceeded&) { fail(503, "memory_capacity", "Insufficient admitted memory for this request"); }
      catch (const std::exception& error) {
          server_status("http", std::string("Request failed: ") + error.what());
          fail(500, "generation_failed", "Generation failed. See the server log for details.");
      }
    if (chat.generation) { chat.generation.cancel(); chat.output->abandon(); }
}
}

void serve_http(HostedModel& model, const HttpOptions& options, const std::function<bool()>& stop) {
    if (options.port < 0 || options.port > 65535 || options.max_connections <= 0 || options.header_bytes < 64 || !options.body_bytes ||
        options.read_timeout_ms <= 0 || options.write_timeout_ms <= 0 || options.request_timeout_ms <= 0)
        throw std::invalid_argument("Invalid HTTP limits");
    SocketRuntime runtime;
    Socket listener(::socket(AF_INET, SOCK_STREAM, IPPROTO_TCP));
    if (listener.get() == invalid_socket) throw std::runtime_error("Cannot create listener");
    int enabled = 1;
#ifdef _WIN32
    setsockopt(listener.get(), SOL_SOCKET, SO_EXCLUSIVEADDRUSE, reinterpret_cast<const char*>(&enabled), sizeof(enabled));
#else
    setsockopt(listener.get(), SOL_SOCKET, SO_REUSEADDR, &enabled, sizeof(enabled));
#endif
    sockaddr_in address{}; address.sin_family = AF_INET; address.sin_port = htons(static_cast<uint16_t>(options.port));
    if (inet_pton(AF_INET, options.address.c_str(), &address.sin_addr) != 1) throw std::invalid_argument("Bind address must be an IPv4 literal");
    if (bind(listener.get(), reinterpret_cast<const sockaddr*>(&address), sizeof(address)) || listen(listener.get(), options.max_connections))
        throw std::runtime_error("Cannot bind or listen on requested HTTP address");
    nonblocking(listener.get()); SocketLength length = sizeof(address); getsockname(listener.get(), reinterpret_cast<sockaddr*>(&address), &length);
    std::cout << "Serving http://" << options.address << ':' << ntohs(address.sin_port) << '\n' << std::flush;
    struct Client { std::thread thread; std::shared_ptr<std::atomic<bool>> done; };
    std::vector<Client> clients; clients.reserve(size_t(options.max_connections));
    std::atomic<bool> stopping{false}; HttpStats stats;
    auto* chat = std::get_if<ChatService>(&model);
    auto* image = std::get_if<ImageService>(&model);
    GpuMonitor gpu_monitor(chat ? chat->options().backend : image->backend(), [chat, image] {
        return chat ? chat->stats().active != 0 : image->has_active_job();
    });
    auto join = [&] {
        stopping.store(true);
        for (auto& client : clients) if (client.thread.joinable()) client.thread.join();
    };
    auto finish = [&] {
        if (chat) {
            chat->shutdown();
            chat->wait();
        }
    };
    try {
        while (!stop() && !(chat && chat->stopped())) {
            for (size_t i = 0; i < clients.size();) {
                if (!clients[i].done->load()) { ++i; continue; }
                clients[i].thread.join(); clients.erase(clients.begin() + i);
            }
            if (!poll_socket(listener.get(), false, 25)) continue;
            Socket accepted(accept(listener.get(), nullptr, nullptr));
            if (accepted.get() == invalid_socket) { if (would_block(socket_error())) continue; throw std::runtime_error("HTTP accept failed"); }
            nonblocking(accepted.get()); ++stats.accepted;
            if (clients.size() >= size_t(options.max_connections)) {
                ++stats.rejected;
                try {
                    send_all(accepted.get(), response(429, chat_error("Connection limit reached", "connections_full", 429)), 25, stopping);
                    finish_rejection(accepted.get());
                } catch (...) {}
                continue;
            }
            int send_buffer = 16 * 1024;
            setsockopt(accepted.get(), SOL_SOCKET, SO_SNDBUF, reinterpret_cast<const char*>(&send_buffer), sizeof(send_buffer));
            setsockopt(accepted.get(), IPPROTO_TCP, TCP_NODELAY, reinterpret_cast<const char*>(&enabled), sizeof(enabled));
            auto done = std::make_shared<std::atomic<bool>>(false);
            std::thread client([socket = std::move(accepted), &model, &options, &stopping, &stats, done]() mutable {
                connection(std::move(socket), model, options, stopping, stats); done->store(true);
            });
            clients.push_back(Client{std::move(client), std::move(done)});
        }
    } catch (...) { join(); try { finish(); } catch (...) {} throw; }
    join(); finish();
}

} // namespace elman
