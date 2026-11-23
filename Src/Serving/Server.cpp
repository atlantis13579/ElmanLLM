#include "Server.h"
#include "Core/LocalFunction.h"

#include "Http/Http.h"
#include "Reinforcement/PolicyValueServer.h"
#include "ServerLog.h"

#include <algorithm>
#include <csignal>
#include <functional>
#include <initializer_list>
#include <stdexcept>
#include <string>
#include <string_view>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <shellapi.h>
#else
#include <cerrno>
#include <spawn.h>
#include <sys/wait.h>
#include <thread>
extern char** environ;
#endif

namespace elman {
namespace {

void check_fields(const Json& section, std::initializer_list<std::string_view> allowed) {
    for (const std::pair<const std::string, Json>& entry : section.object()) {
        const std::string& key = entry.first;
        const Json& value = entry.second;
        if (std::find(allowed.begin(), allowed.end(), key) == allowed.end())
            throw std::invalid_argument("Unknown server configuration field: " + key);
    }
}

size_t size_or(const Json& section, const char* key, size_t fallback) {
    if (!section.contains(key)) return fallback;
    const int n = section.at(key).integer();
    if (n < 0) throw std::invalid_argument(std::string(key) + " must be nonnegative");
    return static_cast<size_t>(n);
}

std::filesystem::path resolve(const std::filesystem::path& base, const std::string& path) {
    return path.empty() ? std::filesystem::path{} : (base / std::filesystem::u8path(path)).lexically_normal();
}

HttpOptions http_options(const Json& config, bool image) {
    check_fields(config, {"host", "port", "max_connections", "max_header_bytes", "max_body_bytes",
        "read_timeout_ms", "write_timeout_ms", "request_timeout_ms", "cors_origins", "web_root"});
    HttpOptions options;
    options.address = config.string_or("host", options.address);
    options.port = config.int_or("port", options.port);
    options.max_connections = config.int_or("max_connections", options.max_connections);
    options.header_bytes = size_or(config, "max_header_bytes", options.header_bytes);
    options.body_bytes = size_or(config, "max_body_bytes", options.body_bytes);
    options.read_timeout_ms = config.int_or("read_timeout_ms", options.read_timeout_ms);
    options.write_timeout_ms = config.int_or("write_timeout_ms", options.write_timeout_ms);
    options.request_timeout_ms = config.int_or("request_timeout_ms", image ? 1800000 : options.request_timeout_ms);
    if (config.contains("cors_origins")) {
        for (const Json& origin : config.at("cors_origins").array())
            options.cors_origins.push_back(origin.string());
    }
    return options;
}

void open_browser(std::string url) {
#ifdef _WIN32
    const std::wstring wide_url(url.begin(), url.end());
    if (reinterpret_cast<INT_PTR>(ShellExecuteW(nullptr, L"open", wide_url.c_str(), nullptr, nullptr, SW_SHOWNORMAL)) > 32)
        return;
#else
#ifdef __APPLE__
    char launcher[] = "open";
#else
    char launcher[] = "xdg-open";
#endif
    char* args[] = {launcher, url.data(), nullptr};
    pid_t child;
    if (posix_spawnp(&child, launcher, nullptr, nullptr, args, environ) == 0) {
        std::thread([child, url = std::move(url)] {
            int status = 0;
            pid_t result;
            do { result = waitpid(child, &status, 0); } while (result < 0 && errno == EINTR);
            if (result == child && (!WIFEXITED(status) || WEXITSTATUS(status) != 0))
                server_status("web", "Open " + url + " in your browser");
        }).detach();
        return;
    }
#endif
    server_status("web", "Open " + url + " in your browser");
}

ChatServiceOptions chat_options(const Json& config, const std::filesystem::path& base) {
    check_fields(config, {"max_context", "max_new_tokens", "cache_blocks", "prefix_cache_blocks", "prefix_cache_entries",
        "max_queued_requests", "max_active_requests", "max_batch_tokens", "prefill_chunk_size", "memory_limit_mib",
        "memory_reserve_mib", "output_bytes", "stream_bytes", "stream_chunks", "trace", "asynchronous",
        "pinned_staging", "reuse_memory", "cuda_graph", "direct_paged_attention", "token_ids_only"});
    ChatServiceOptions options;
    SchedulerOptions& scheduler = options.scheduler;
    options.max_new_tokens = config.int_or("max_new_tokens", options.max_new_tokens);
    scheduler.max_context = config.int_or("max_context", scheduler.max_context);
    scheduler.cache_blocks = config.int_or("cache_blocks", scheduler.cache_blocks);
    scheduler.prefix_cache_blocks = config.int_or("prefix_cache_blocks", scheduler.prefix_cache_blocks);
    scheduler.prefix_cache_entries = config.int_or("prefix_cache_entries", scheduler.prefix_cache_entries);
    scheduler.max_queued_requests = config.int_or("max_queued_requests", scheduler.max_queued_requests);
    scheduler.max_active_requests = config.int_or("max_active_requests", scheduler.max_active_requests);
    scheduler.max_batch_tokens = config.int_or("max_batch_tokens", scheduler.max_batch_tokens);
    scheduler.prefill_chunk_size = config.int_or("prefill_chunk_size", scheduler.prefill_chunk_size);
    scheduler.memory.limit_bytes = checked_memory_mul(size_or(config, "memory_limit_mib", 0), 1024 * 1024);
    scheduler.memory.reserve_bytes = checked_memory_mul(size_or(config, "memory_reserve_mib", 0), 1024 * 1024);
    options.output_bytes = size_or(config, "output_bytes", options.output_bytes);
    options.stream_bytes = size_or(config, "stream_bytes", options.stream_bytes);
    options.stream_chunks = size_or(config, "stream_chunks", options.stream_chunks);
    options.trace_path = resolve(base, config.string_or("trace", "")).u8string();
    scheduler.asynchronous = config.bool_or("asynchronous", scheduler.asynchronous);
    scheduler.pinned_staging = config.bool_or("pinned_staging", scheduler.pinned_staging);
    scheduler.reuse_memory = config.bool_or("reuse_memory", scheduler.reuse_memory);
    scheduler.cuda_graph = config.bool_or("cuda_graph", scheduler.cuda_graph);
    options.direct_paged_attention = config.bool_or("direct_paged_attention", options.direct_paged_attention);
    options.token_ids_only = config.bool_or("token_ids_only", options.token_ids_only);
    return options;
}

volatile std::sig_atomic_t interrupted = 0;
void interrupt_handler(int) { interrupted = 1; }

class ServerSignals {
public:
    ServerSignals() {
        interrupted = 0;
        interrupt_ = std::signal(SIGINT, interrupt_handler);
        terminate_ = std::signal(SIGTERM, interrupt_handler);
#ifdef _WIN32
        stop_break_ = std::signal(SIGBREAK, interrupt_handler);
#endif
    }
    ~ServerSignals() {
        std::signal(SIGINT, interrupt_);
        std::signal(SIGTERM, terminate_);
#ifdef _WIN32
        std::signal(SIGBREAK, stop_break_);
#endif
    }

private:
    using Handler = void(*)(int);
    Handler interrupt_, terminate_;
#ifdef _WIN32
    Handler stop_break_;
#endif
};

} // namespace

void run_server(const std::filesystem::path& config_path, bool open_web) {
    const Json config = Json::read(config_path);
    check_fields(config, {"model", "http", "tcp", "chat", "training", "stop_file"});
    const Json& model = config.at("model");
    const std::string type = model.at("type").string();
    const std::filesystem::path base = std::filesystem::absolute(config_path).parent_path();
    const std::filesystem::path stop_file = resolve(base, config.string_or("stop_file", ""));
    if (!stop_file.empty() && std::filesystem::exists(stop_file)) throw std::invalid_argument("Stop file already exists");
    LocalFunction stop{[&] { return interrupted || (!stop_file.empty() && std::filesystem::exists(stop_file)); }};
    ServerSignals signals;
    if (type == "policy_value" || type == "alpha") {
        if (open_web) throw std::invalid_argument("--open requires an HTTP model with http.web_root");
        check_fields(model, {"type", "path", "checkpoint", "backend"});
        if (config.contains("chat")) throw std::invalid_argument("chat settings require model.type llm");
        if (config.contains("training")) {
            const Json& training = config.at("training");
            check_fields(training, {"directory", "learning_rate", "weight_decay", "clip_grad_norm", "seed"});
            if (training.at("directory").string().empty()) throw std::invalid_argument("Expected a training directory");
        }
        const Backend backend = parse_backend(model.string_or("backend", "cpu"));
        if (!backend_available(backend)) throw std::invalid_argument("Policy-value backend unavailable");
        if (config.contains("http")) throw std::invalid_argument("Policy-value serving requires the tcp configuration");
        if (config.contains("tcp"))
            check_fields(config.at("tcp"), {"host", "port", "max_connections", "max_message_bytes",
                                           "read_timeout_ms", "write_timeout_ms"});
        serve_policy_value(config, base, backend, stop);
        return;
    }
    if (config.contains("tcp")) throw std::invalid_argument("tcp settings require model.type policy_value");
    if (config.contains("training")) throw std::invalid_argument("training settings require model.type policy_value");
    check_fields(model, {"type", "path", "name", "backend", "dtype"});
    const bool flux2 = type == "flux2";
    const bool image = type == "diffusion" || flux2;
    if (type != "llm" && !image) throw std::invalid_argument("model.type must be llm, diffusion, flux2 or policy_value");
    if (image && config.contains("chat")) throw std::invalid_argument("chat settings require model.type llm");
    std::filesystem::path directory = resolve(base, model.at("path").string());
    if (directory.empty() || !std::filesystem::is_directory(directory))
        throw std::invalid_argument("Model directory does not exist: " + directory.u8string());
    if (!directory.has_filename()) directory = directory.parent_path();
    const std::string name = model.string_or("name", directory.filename().u8string());
    if (name.empty()) throw std::invalid_argument("Model name must be nonempty");
    const Backend backend = parse_backend(model.string_or("backend", "cpu"));
    const DType dtype = parse_dtype(model.string_or("dtype",
        backend == Backend::CUDA ? (image && !flux2 ? "float16" : "bfloat16") : "float32"));
    if (!backend_available(backend) || !is_compute_float(dtype))
        throw std::invalid_argument("Unavailable server backend or compute dtype");
    const Json defaults = Json::Object{};
    const Json& http_config = config.contains("http") ? config.at("http") : defaults;
    const HttpOptions http = http_options(http_config, image);
    std::filesystem::path web_root = resolve(base, http_config.string_or("web_root", ""));
    if (!web_root.empty()) {
        if (!std::filesystem::is_directory(web_root))
            throw std::invalid_argument("Web root is not a directory: " + web_root.u8string());
        web_root = std::filesystem::canonical(web_root);
    }
    if (open_web && web_root.empty()) throw std::invalid_argument("--open requires http.web_root");
    std::function<void(int)> ready;
    if (open_web) {
        ready = [&](int port) {
            const std::string host = http.address == "0.0.0.0" ? "127.0.0.1" : http.address;
            open_browser("http://" + host + ':' + std::to_string(port) + '/');
        };
    }
    server_status("server", "Loading " + type + " model " + name + " (" + backend_name(backend) + ", " + dtype_name(dtype) + ")");
    if (image) {
        HostedModel hosted(std::in_place_type<ImageService>, directory, name, backend, dtype, flux2);
        serve_http(hosted, http, web_root, stop, ready);
    } else {
        ChatServiceOptions options = chat_options(config.contains("chat") ? config.at("chat") : defaults, base);
        options.model_directory = directory.u8string();
        options.model_name = name;
        options.backend = backend;
        options.dtype = dtype;
        HostedModel hosted(std::in_place_type<ChatService>, std::move(options));
        serve_http(hosted, http, web_root, stop, ready);
    }
    server_status("server", "Stopped");
}

} // namespace elman
