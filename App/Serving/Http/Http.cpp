#include "Http.h"
#include "Core/LocalFunction.h"
#include "ChatResponse.h"
#include "Serving/GpuMonitor.h"
#include "Serving/ServerLog.h"
#include <algorithm>
#include <atomic>
#include <charconv>
#include <chrono>
#include <iostream>
#include <thread>

namespace elman {
namespace {
using Clock = std::chrono::steady_clock;
std::string media_type(std::string value) {
    value = value.substr(0, value.find(';'));
    size_t first = value.find_first_not_of(" \t"), last = value.find_last_not_of(" \t");
    if (first == value.npos) return {};
    value = value.substr(first, last - first + 1);
    for (char& c : value) if (c >= 'A' && c <= 'Z') c += 'a' - 'A';
    return value;
}
std::string event(const Json& data) { return "data: " + data.dump(0) + "\n\n"; }
constexpr char event_stream_header[] =
    "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream; charset=utf-8\r\nCache-Control: no-cache\r\n"
    "Connection: close\r\nX-Accel-Buffering: no\r\n";

std::string_view progress_parameter(std::string_view query, std::string_view key) {
    while (!query.empty()) {
        const size_t end = query.find('&');
        const std::string_view field = query.substr(0, end);
        if (field.size() > key.size() && field[key.size()] == '=' && field.substr(0, key.size()) == key)
            return field.substr(key.size() + 1);
        if (end == query.npos) break;
        query.remove_prefix(end + 1);
    }
    return {};
}

struct HttpStats : ConnectionCounts {
    std::atomic<uint64_t> completed{0}, errors{0}, disconnected{0};
    std::mutex mutex;
    double wall_ms = 0, first_text_ms = 0;
    uint64_t first_text_count = 0;
};
Json metrics(const ChatService* service, Backend backend, HttpStats& http) {
    const SchedulerStats s = service ? service->stats() : SchedulerStats{};
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
void connection(const HttpConnection& channel, HostedModel& hosted, const HttpOptions& options,
                const std::filesystem::path& web_root, HttpStats& stats) {
    ChatService* chat_service = std::get_if<ChatService>(&hosted);
    ImageService* image_service = std::get_if<ImageService>(&hosted);
    const std::string& model = chat_service ? chat_service->options().model_name : image_service->name();
    ChatRequest chat;
    bool streaming = false, image_request = false, head = false;
    std::string cors = options.cors_origins.empty() ? "" : "Vary: Origin\r\n";
    const std::function<void (const std::string&)>& send = channel.send;
    LocalFunction send_json{[&](int status, const Json& data) {
        constexpr const char* mime = "application/json; charset=utf-8";
        const std::string body = data.dump(0);
        send(head ? http_response_header(status, mime, body.size(), cors) : http_response(status, mime, body, cors));
    }};
    LocalFunction start_stream{[&] {
        send(event_stream_header + cors + "\r\n");
        streaming = true;
    }};
    LocalFunction fail{[&](int status, const std::string& code, const std::string& message) {
        ++stats.errors;
        try {
            if (image_request) {
                Json data = Json::Object{{"error", code}, {"detail", message}};
                if (streaming) send(event(data));
                else send_json(status, data);
            } else {
                const Json data = chat_error(message, code, status);
                if (streaming) send(event(data) + "data: [DONE]\n\n");
                else send_json(status, data);
            }
        } catch (...) {}
    }};
    try {
        MemoryReservation body_memory(Backend::CPU, checked_memory_add(checked_memory_mul(options.body_bytes, 8), checked_memory_mul(options.header_bytes, 2)));
        HttpRequest request = channel.receive();
        head = request.method == "HEAD";
        const std::string_view target = request.target;
        const size_t separator = target.find('?');
        const std::string_view path = target.substr(0, separator);
        const auto origin = request.headers.find("origin");
        const bool allow_origin = origin != request.headers.end() &&
            std::find(options.cors_origins.begin(), options.cors_origins.end(), origin->second) != options.cors_origins.end();
        if (allow_origin) cors += "Access-Control-Allow-Origin: " + origin->second + "\r\n";
        if (request.method == "OPTIONS" && request.headers.count("access-control-request-method")) {
            if (!allow_origin) throw ChatError(403, "origin_not_allowed", "This browser origin is not allowed");
            send("HTTP/1.1 204 No Content\r\nConnection: close\r\n" + cors +
                "Access-Control-Allow-Methods: GET, POST\r\nAccess-Control-Allow-Headers: Content-Type\r\n\r\n");
            return;
        }
        if (request.method == "GET") {
            if (request.target == "/health") {
                Json::Object health{{"status", "ok"}, {"model", model},
                    {"model_type", chat_service ? "llm" : "diffusion"},
                    {"chat_text", chat_service && !chat_service->options().token_ids_only},
                    {"image_generation", image_service != nullptr}};
                if (image_service) health.emplace("image_sampler", image_service->sampler());
                send_json(200, std::move(health)); return;
            }
            if (request.target == "/metrics") {
                Backend backend = chat_service ? chat_service->options().backend : image_service->backend();
                send_json(200, metrics(chat_service, backend, stats)); return;
            }
            if (chat_service && request.target == "/v1/models") {
                Json entry = Json::Object{{"id", model}, {"object", "model"}, {"created", 0}, {"owned_by", "elman"}};
                send_json(200, Json::Object{{"object", "list"}, {"data", Json::Array{entry}}}); return;
            }
            if (image_service && path == "/sdapi/v1/progress") {
                const std::string_view query = separator == target.npos ? std::string_view{} : target.substr(separator + 1);
                const std::string_view id = progress_parameter(query, "task_id");
                const std::string_view cursor = progress_parameter(query, "after_preview_step");
                int after_preview_step = 0;
                if (!cursor.empty()) {
                    const char* const end = cursor.data() + cursor.size();
                    const std::from_chars_result result = std::from_chars(cursor.data(), end, after_preview_step);
                    if (result.ec != std::errc{} || result.ptr != end || after_preview_step < 0)
                        throw ChatError(400, "invalid_preview_step", "after_preview_step must be a nonnegative integer");
                }
                const std::string_view skip = progress_parameter(query, "skip_current_image");
                send_json(200, image_service->progress(id, after_preview_step, skip == "true" || skip == "1"));
                return;
            }
        }
        if (!web_root.empty() && (request.method == "GET" || head) &&
            path != "/health" && path != "/metrics" && path.substr(0, 4) != "/v1/" && path.substr(0, 7) != "/sdapi/" &&
            send_http_file(channel, web_root, path, head)) return;
        image_request = request.target == "/sdapi/v1/txt2img";
        const bool chat_request = request.target == "/v1/chat/completions";
        if (!(image_request && image_service) && !(chat_request && chat_service))
            throw ChatError(404, "not_found", "This endpoint is unavailable for the loaded model");
        if (request.method != "POST") throw ChatError(405, "method_not_allowed", "This endpoint requires POST");
        auto content_type = request.headers.find("content-type");
        if (content_type == request.headers.end() || media_type(content_type->second) != "application/json")
            throw ChatError(415, "unsupported_media_type", "Content-Type must be application/json");
        Json body;
        try { utf16(request.body); body = Json::parse(request.body); }
        catch (const std::exception& error) { throw ChatError(400, "invalid_json", error.what()); }
        std::chrono::steady_clock::time_point started = Clock::now();
        std::chrono::steady_clock::time_point deadline = started + std::chrono::milliseconds(options.request_timeout_ms);
        LocalFunction check{[&] {
            if (channel.closed()) throw ConnectionClosed();
            if (Clock::now() >= deadline) throw ChatError(408, "generation_timeout", "Request deadline exceeded");
        }};
        LocalFunction complete{[&] {
            ++stats.completed;
            std::lock_guard<std::mutex> lock(stats.mutex);
            stats.wall_ms += std::chrono::duration<double, std::milli>(Clock::now() - started).count();
        }};
        if (image_request) {
            const auto accept = request.headers.find("accept");
            if (accept != request.headers.end() && media_type(accept->second) == "text/event-stream") {
                LocalFunction emit{[&](const Json& data) {
                    if (!streaming) start_stream();
                    send(event(data));
                }};
                Json result = image_service->generate(body, check, emit);
                emit(Json::Object{{"type", "result"}, {"result", std::move(result)}});
            } else {
                send_json(200, image_service->generate(body, check));
            }
            complete();
            return;
        }
        check();
        chat = chat_service->submit(body);
        if (chat.stream) {
            start_stream();
            send(event(chat_chunk(chat, model, Json::Object{{"role", "assistant"}, {"content", ""}})));
        }
        bool first_text = false;
        LocalFunction emit{[&](const std::string& piece) {
            if (piece.empty()) return;
            if (!first_text) {
                std::lock_guard<std::mutex> lock(stats.mutex); ++stats.first_text_count;
                stats.first_text_ms += std::chrono::duration<double, std::milli>(Clock::now() - started).count(); first_text = true;
            }
            send(event(chat_chunk(chat, model, Json::Object{{"content", piece}})));
        }};
        while (true) {
            check();
            if (chat_service->stopped()) throw ConnectionClosed();
            if (chat.stream) emit(chat.output->pop(std::chrono::milliseconds(10)));
            else chat.generation.wait_for(std::chrono::milliseconds(10));
            if (!chat.generation.ready()) continue;
            if (chat.generation.error()) std::rethrow_exception(chat.generation.error());
            chat_finish_reason(chat.generation.result());
            std::string tail; GenerationResult result = chat.output->finish(chat.generation.result(), tail);
            if (chat.stream) {
                for (;;) { std::string piece = chat.output->pop(std::chrono::milliseconds(0)); if (piece.empty()) break; emit(piece); }
                emit(tail);
                send(event(chat_chunk(chat, model, Json::Object{}, chat_finish_reason(result))));
                if (chat.include_usage) send(event(chat_chunk(chat, model, Json::Object{}, {}, chat_usage(chat.prompt_tokens, result.ids.size()), true)));
                send("data: [DONE]\n\n");
            } else send_json(200, chat_completion(chat, model, result));
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

void serve_http(HostedModel& model, const HttpOptions& options, const std::filesystem::path& web_root,
                const std::function<bool()>& stop, const std::function<void(int)>& ready) {
    HttpStats stats;
    ChatService* chat = std::get_if<ChatService>(&model);
    ImageService* image = std::get_if<ImageService>(&model);
    GpuMonitor gpu_monitor(chat ? chat->options().backend : image->backend(), [chat, image] {
        return chat ? chat->stats().active != 0 : image->has_active_job();
    });
    LocalFunction finish{[&] {
        if (chat) {
            chat->shutdown();
            chat->wait();
        }
    }};
    try {
        listen_http(options, [&] { return stop() || (chat && chat->stopped()); },
                    [&](const HttpConnection& channel) { connection(channel, model, options, web_root, stats); }, stats, ready);
    } catch (...) { try { finish(); } catch (...) {} throw; }
    finish();
}
} // namespace elman
