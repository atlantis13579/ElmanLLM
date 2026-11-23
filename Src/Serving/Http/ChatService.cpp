#include "ChatService.h"
#include "Serving/ServerLog.h"
#include "Models/LLM/DecoderFactory.h"
#include <algorithm>
#include <atomic>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <future>
#include <limits>
#include <set>
#include <thread>

namespace elman {
namespace {
void require(bool value, const char* message) {
    if (!value) throw ChatError(400, "invalid_request", message);
}
void fields(const Json& object, std::initializer_list<const char*> allowed) {
    for (const std::pair<const std::string, Json>& pair : object.object()) require(std::find(allowed.begin(), allowed.end(), pair.first) != allowed.end(),
        ("Unsupported field: " + pair.first).c_str());
}
Json ids_json(const std::vector<int32_t>& ids) { Json::Array result; for (int id : ids) result.emplace_back(id); return result; }
std::string trace_error(const std::exception_ptr& error) {
    if (!error) return {};
    try { std::rethrow_exception(error); }
    catch (const ChatError& failure) { return failure.code; }
    catch (...) { return "generation_failed"; }
}
uint64_t seed(const Json& value) {
    // String form retains all UInt64 bits; numeric JSON accepts exact integers.
    try {
        const std::string text = value.string();
        require(!text.empty() && text.size() <= 20 && std::all_of(text.begin(), text.end(), [](char c) { return c >= '0' && c <= '9'; }), "Invalid seed string");
        size_t used = 0; unsigned long long result = std::stoull(text, &used); require(used == text.size(), "Invalid seed string"); return result;
    } catch (const ChatError&) { throw; }
      catch (const std::out_of_range&) { throw ChatError(400, "invalid_seed", "Seed exceeds UInt64"); }
      catch (const std::runtime_error&) {}
    double number = value.number();
    require(number >= 0 && number <= 9007199254740991. && std::floor(number) == number, "Numeric seed must be an exact nonnegative integer; use a decimal string for UInt64");
    return uint64_t(number);
}
ChatInput parse_input(const Json& raw_body, const HuggingFaceTokenizer* tokenizer, const DecoderConfig& config,
                      const GenerationOptions& defaults, const ChatServiceOptions& settings) {
    fields(raw_body, {"model", "messages", "input_ids", "stream", "stream_options", "max_tokens", "max_completion_tokens", "temperature", "top_p", "top_k",
        "repetition_penalty", "seed", "stop", "n", "user", "frequency_penalty", "presence_penalty", "response_format"});
    Json::Object normalized = raw_body.object();
    for (auto it = normalized.begin(); it != normalized.end();) {
        if (it->second.is_null() && it->first != "model" && it->first != "messages") it = normalized.erase(it);
        else ++it;
    }
    Json body(std::move(normalized));
    if (body.at("model").string() != settings.model_name) throw ChatError(404, "model_not_found", "Unknown model");
    require(body.int_or("n", 1) == 1, "Only n=1 is supported");
    require(body.number_or("frequency_penalty", 0) == 0 && body.number_or("presence_penalty", 0) == 0, "Frequency/presence penalties are not supported");
    if (body.contains("user")) body.at("user").string();
    if (body.contains("response_format")) {
        fields(body.at("response_format"), {"type"}); require(body.at("response_format").at("type").string() == "text", "Only text response format is supported");
    }
    ChatInput result; result.generation = defaults;
    result.stream = body.bool_or("stream", false);
    if (body.contains("stream_options") && !body.at("stream_options").is_null()) {
        require(result.stream, "stream_options requires stream=true");
        const Json& options = body.at("stream_options"); fields(options, {"include_usage", "include_obfuscation"});
        result.include_usage = options.contains("include_usage") && !options.at("include_usage").is_null() && options.at("include_usage").boolean();
        require(!options.contains("include_obfuscation") || options.at("include_obfuscation").is_null() || !options.at("include_obfuscation").boolean(),
            "Stream obfuscation is not supported");
    }
    require(!(body.contains("max_tokens") && body.contains("max_completion_tokens")), "Use one output token limit");
    result.generation.max_new_tokens = body.int_or("max_completion_tokens", body.int_or("max_tokens", defaults.max_new_tokens));
    result.generation.sampling.temperature = body.number_or("temperature", defaults.sampling.temperature);
    require(result.generation.sampling.temperature >= 0 && result.generation.sampling.temperature <= 2, "temperature must be in [0,2]");
    result.generation.do_sample = result.generation.sampling.temperature != 0;
    if (!result.generation.do_sample) result.generation.sampling.temperature = 1;
    result.generation.sampling.top_p = body.number_or("top_p", defaults.sampling.top_p);
    result.generation.sampling.top_k = body.int_or("top_k", defaults.sampling.top_k);
    result.generation.sampling.repetition_penalty = body.number_or("repetition_penalty", defaults.sampling.repetition_penalty);
    validate_sampling(result.generation.sampling);
    if (body.contains("seed")) result.generation.seed = seed(body.at("seed"));
    if (body.contains("stop") && !body.at("stop").is_null()) {
        const Json& stop = body.at("stop");
        if (stop.is_array()) for (const Json& item : stop.array()) result.stops.push_back(item.string());
        else result.stops.push_back(stop.string());
        require(result.stops.size() <= 4, "At most four stop strings are supported");
        for (const std::string& item : result.stops) require(item.size() <= 4096, "Stop string exceeds 4096 UTF-8 bytes");
        TextStopFilter validate(result.stops);
    }
    result.token_ids = body.contains("input_ids");
    require(result.token_ids != body.contains("messages"), "Provide exactly one of messages and input_ids");
    if (result.token_ids) {
        require(!result.stream && result.stops.empty(), "Token ID requests require non-streaming output without text stops");
        for (const Json& id : body.at("input_ids").array()) result.ids.push_back(id.integer());
        validate_generation_request(config, result.ids, result.generation);
        return result;
    }
    require(tokenizer != nullptr, "This service accepts input_ids requests");
    Json::Array messages;
    for (const Json& message : body.at("messages").array()) {
        fields(message, {"role", "content"});
        std::string role = message.at("role").string();
        require(role == "system" || role == "user" || role == "assistant", "Only system, user and assistant text messages are supported");
        std::string text; const Json& content = message.at("content");
        if (content.is_array()) for (const Json& part : content.array()) {
            fields(part, {"type", "text"}); require(part.at("type").string() == "text", "Only text content parts are supported"); text += part.at("text").string();
        } else text = content.string();
        utf16(text); messages.emplace_back(Json::Object{{"role", role}, {"content", text}});
    }
    result.ids = tokenizer->encode_chat(messages);
    validate_generation_request(config, result.ids, result.generation);
    return result;
}
}

ChatOutput::ChatOutput(const HuggingFaceTokenizer& tokenizer, const ChatInput& input, const ChatServiceOptions& options)
    : ChatOutput(&tokenizer, input, options) {}
ChatOutput::ChatOutput(const HuggingFaceTokenizer* tokenizer, const ChatInput& input, const ChatServiceOptions& options)
    : memory_(Backend::CPU, checked_memory_add(checked_memory_mul(options.output_bytes, 3),
        checked_memory_add(checked_memory_add(options.stream_bytes, checked_memory_mul(options.stream_chunks, 256)), 64 * 1024))),
      filter_(input.stops), stream_(input.stream), output_limit_(options.output_bytes),
      queue_limit_(options.stream_bytes), chunk_limit_(options.stream_chunks) {
    require(output_limit_ > 0 && queue_limit_ > 0 && chunk_limit_ > 0, "Output limits must be positive");
    if (!input.token_ids) {
        require(tokenizer != nullptr, "Text output requires a tokenizer");
        decoder_.emplace(*tokenizer, true);
    }
}
void ChatOutput::append(const std::string& text, bool enqueue) {
    if (text.empty()) return;
    std::lock_guard<std::mutex> lock(mutex_);
    if (abandoned_) throw ChatError(499, "client_closed", "Client disconnected");
    if (text.size() > output_limit_ - text_.size()) throw ChatError(503, "output_limit", "Generated text exceeded the configured byte limit");
    if (stream_ && enqueue) {
        if (pieces_.size() >= chunk_limit_ || text.size() > queue_limit_ - queued_bytes_)
            throw ChatError(503, "output_backpressure", "Client output queue is full");
        pieces_.push_back(text); queued_bytes_ += text.size();
    }
    text_ += text;
    changed_.notify_one();
}
bool ChatOutput::accept(int32_t id) {
    if (finished_) throw std::logic_error("Text output is already finalized");
    if (decoder_) append(filter_.push(decoder_->push(id)), true);
    return !filter_.stopped();
}
std::string ChatOutput::pop(std::chrono::milliseconds timeout) {
    std::unique_lock<std::mutex> lock(mutex_);
    changed_.wait_for(lock, timeout, [&] { return abandoned_ || !pieces_.empty(); });
    if (pieces_.empty()) return {};
    std::string text = std::move(pieces_.front()); pieces_.pop_front(); queued_bytes_ -= text.size(); return text;
}
GenerationResult ChatOutput::finish(const GenerationResult& result, std::string& tail) {
    if (finished_) throw std::logic_error("Text output was finalized twice");
    finished_ = true;
    if (!decoder_) { tail.clear(); return result; }
    tail = filter_.push(decoder_->finish()); tail += filter_.finish(); append(tail, false);
    GenerationResult final = result; final.text = text_;
    if (filter_.stopped()) { final.finish_reason = "stop"; final.matched_stop = filter_.matched_stop(); }
    return final;
}
void ChatOutput::abandon() {
    std::lock_guard<std::mutex> lock(mutex_); abandoned_ = true; pieces_.clear(); queued_bytes_ = 0; changed_.notify_all();
}

struct ChatService::Impl {
    struct Observation {
        GenerationRequest request;
        size_t prompt_tokens;
        size_t prefilled = 0;
        int decode_steps = 0;
        std::chrono::steady_clock::time_point last_decode_log;
    };
    ChatServiceOptions options;
    std::unique_ptr<HuggingFaceTokenizer> tokenizer;
    DecoderConfig config;
    GenerationOptions defaults;
    mutable std::mutex mutex, encoding;
    GenerationScheduler* scheduler = nullptr;
    SchedulerStats final_stats;
    std::atomic<bool> stopping{false};
    std::exception_ptr failure;
    std::thread worker;
    std::ofstream trace;
    std::map<uint64_t, Observation> observed;
    std::string prefix;
    Impl(ChatServiceOptions options_) : options(std::move(options_)),
        config(read_decoder_config(options.model_directory)) {
        if (!options.token_ids_only) tokenizer = std::make_unique<HuggingFaceTokenizer>(options.model_directory);
        if (options.model_name.empty()) {
            std::filesystem::path directory = std::filesystem::path(options.model_directory).lexically_normal();
            if (!directory.has_filename()) directory = directory.parent_path();
            options.model_name = directory.filename().string();
        }
        require(!options.model_name.empty(), "Model name is empty");
        require(options.max_new_tokens >= 0, "Default output token limit must be nonnegative");
        defaults.max_context = options.scheduler.max_context; defaults.max_new_tokens = options.max_new_tokens;
        defaults.memory = options.scheduler.memory; defaults.prefill_chunk_size = options.scheduler.prefill_chunk_size;
        if (tokenizer && tokenizer->eos_id() >= 0) defaults.eos_ids.push_back(tokenizer->eos_id());
        std::filesystem::path path = std::filesystem::path(options.model_directory) / "generation_config.json";
        if (std::filesystem::exists(path)) {
            Json generation = Json::read(path.string());
            defaults.sampling.temperature = generation.number_or("temperature", 1);
            defaults.sampling.top_k = generation.int_or("top_k", 0);
            defaults.sampling.top_p = generation.number_or("top_p", 1);
            defaults.sampling.repetition_penalty = generation.number_or("repetition_penalty", 1);
            if (generation.contains("eos_token_id") && !generation.at("eos_token_id").is_null()) {
                defaults.eos_ids.clear(); const Json& eos = generation.at("eos_token_id");
                if (eos.is_array()) for (const Json& id : eos.array()) defaults.eos_ids.push_back(id.integer());
                else defaults.eos_ids.push_back(eos.integer());
            }
        }
        validate_sampling(defaults.sampling);
        require(options.output_bytes > 0 && options.stream_bytes > 0 && options.stream_chunks > 0, "Output limits must be positive");
        prefix = "chatcmpl-" + std::to_string(std::chrono::high_resolution_clock::now().time_since_epoch().count()) + "-";
        if (!options.trace_path.empty()) {
            trace.exceptions(std::ios::badbit | std::ios::failbit);
            trace.open(options.trace_path, std::ios::binary);
        }
        std::promise<void> ready; std::future<void> future = ready.get_future();
        worker = std::thread([this, ready = std::move(ready)]() mutable {
            bool published = false;
            std::unique_ptr<DecoderModel> model;
            std::unique_ptr<GenerationScheduler> runner;
            try {
                {
                    DecoderMemoryPlan plan = plan_decoder_weight_load(options.model_directory, options.backend, options.dtype);
                    MemoryAdmission admission(options.backend, options.scheduler.memory, plan.additional_bytes, plan.host_additional_bytes);
                    DecoderLoadOptions load;
                    load.direct_paged_attention = options.direct_paged_attention;
                    model = load_decoder(options.model_directory, options.backend, options.dtype, load);
                }
                runner = std::make_unique<GenerationScheduler>(*model, options.scheduler);
                { std::lock_guard<std::mutex> lock(mutex); scheduler = runner.get(); }
                ready.set_value(); published = true;
                while (true) {
                    if (stopping.load()) runner->request_shutdown();
                    SchedulerStats current = runner->stats();
                    if (!(stopping.load() && !current.queued && !current.active)) runner->wait_for_work(std::chrono::milliseconds(50));
                    SchedulerStep step = runner->step();
                    std::lock_guard<std::mutex> lock(mutex);
                    if (trace.is_open()) {
                        Json::Array members, consumed, prefixes;
                        for (uint64_t id : step.requests) members.emplace_back(std::to_string(id));
                        for (int count : step.batch.consumed) consumed.emplace_back(count);
                        for (const PrefixCacheEvent& prefix : step.prefixes) prefixes.emplace_back(Json::Object{
                            {"request", std::to_string(prefix.request)}, {"entry", std::to_string(prefix.entry)},
                            {"tokens", prefix.tokens}, {"publication", prefix.publication}});
                        if (!step.requests.empty() || !prefixes.empty()) write_trace(Json::Object{
                            {"type", "step"}, {"members", members}, {"consumed", consumed}, {"prefixes", prefixes},
                            {"cuda_graph_captures", double(step.batch.graph_captures)}, {"cuda_graph_replays", double(step.batch.graph_replays)},
                            {"query", step.batch.query_tokens}, {"prefill", step.prefill}, {"inputs", step.batch.input_tokens}});
                    }
                    observe(step);
                    current = runner->stats();
                    if (stopping.load() && !current.queued && !current.active) {
                        if (trace.is_open()) write_trace(Json::Object{{"type", "shutdown"}, {"live_cache_blocks", double(current.live_blocks)},
                            {"reserved_cache_blocks", double(current.reserved_blocks)}, {"prefix_cache_entries", double(current.prefix.entries)}});
                        break;
                    }
                }
                { std::lock_guard<std::mutex> lock(mutex); final_stats = runner->stats(); scheduler = nullptr; }
            } catch (...) {
                stopping.store(true);
                { std::lock_guard<std::mutex> lock(mutex); scheduler = nullptr; failure = std::current_exception(); }
                if (!published) ready.set_exception(failure);
            }
            // Detach the public pointer before worker-owned session destruction.
            runner.reset(); model.reset();
        });
        try { future.get(); } catch (...) { worker.join(); throw; }
    }
    void write_trace(const Json& event) { trace << event.dump(0) << '\n'; trace.flush(); }

    void complete(const GenerationRequest& request) {
        const GenerationResult& result = request.result();
        const std::exception_ptr error = request.error();
        const std::string id = std::to_string(request.id());
        server_status("llm " + id, (error ? "Failed" : result.finish_reason == "cancelled" ? "Cancelled" : "Completed") +
            std::string(" (reason=") + result.finish_reason + ", prefill=" + std::to_string(result.prefill_tokens) +
            ", cached=" + std::to_string(result.cached_prompt_tokens) + ", generated=" + std::to_string(result.ids.size()) + " tokens)");
        if (trace.is_open()) write_trace(Json::Object{{"type", "complete"}, {"id", id}, {"ids", ids_json(result.ids)},
            {"reason", result.finish_reason}, {"prefill", result.prefill_tokens}, {"error", bool(error)},
            {"cached_prompt_tokens", result.cached_prompt_tokens}, {"prefix_entry", std::to_string(result.prefix_entry)},
            {"error_code", trace_error(error)}});
    }

    void observe(const SchedulerStep& step) {
        for (const PrefixCacheEvent& prefix : step.prefixes) {
            auto found = observed.find(prefix.request);
            if (!prefix.publication && found != observed.end()) found->second.prefilled = static_cast<size_t>(prefix.tokens);
        }
        for (size_t i = 0; i < std::min(step.requests.size(), step.batch.consumed.size()); ++i) {
            const int consumed = step.batch.consumed[i];
            auto found = observed.find(step.requests[i]);
            if (consumed <= 0 || found == observed.end()) continue;
            ChatService::Impl::Observation& row = found->second;
            if (step.prefill) {
                row.prefilled += static_cast<size_t>(consumed);
                server_status("llm " + std::to_string(found->first), "Prefill " + std::to_string(row.prefilled) +
                    "/" + std::to_string(row.prompt_tokens) + " input tokens");
            } else {
                const std::chrono::steady_clock::time_point now = std::chrono::steady_clock::now();
                ++row.decode_steps;
                if (row.decode_steps == 1 || row.decode_steps % 16 == 0 || now - row.last_decode_log >= std::chrono::seconds(2)) {
                    server_status("llm " + std::to_string(found->first), "Decode step " + std::to_string(row.decode_steps));
                    row.last_decode_log = now;
                }
            }
        }
        for (auto it = observed.begin(); it != observed.end();) {
            if (!it->second.request.ready()) { ++it; continue; }
            complete(it->second.request);
            it = observed.erase(it);
        }
    }
};
ChatService::ChatService(ChatServiceOptions options) : impl_(std::make_unique<Impl>(std::move(options))) {}
ChatService::~ChatService() { shutdown(); if (impl_->worker.joinable()) impl_->worker.join(); }
ChatRequest ChatService::submit(const Json& body) {
    ChatService::Impl& s = *impl_; ChatInput input;
    try { std::lock_guard<std::mutex> lock(s.encoding); input = parse_input(body, s.tokenizer.get(), s.config, s.defaults, s.options); }
    catch (const ChatError&) { throw; }
    catch (const std::exception& error) { throw ChatError(400, "invalid_request", error.what()); }
    std::shared_ptr<ChatOutput> output = std::make_shared<ChatOutput>(s.tokenizer.get(), input, s.options);
    std::lock_guard<std::mutex> lock(s.mutex);
    if (!s.scheduler || s.stopping.load()) throw SchedulerClosed();
    GenerationRequest handle = s.scheduler->submit(input.ids, input.generation, [output](int32_t id) { return output->accept(id); });
    try {
        server_status("llm " + std::to_string(handle.id()), "Queued (input " + std::to_string(input.ids.size()) +
            " tokens, requested output limit " + std::to_string(input.generation.max_new_tokens) + ")");
        if (s.trace.is_open()) {
            Json::Array stops; for (const std::string& stop : input.stops) stops.emplace_back(stop);
            const GenerationOptions& g = input.generation;
            s.write_trace(Json::Object{{"type", "request"}, {"id", std::to_string(handle.id())}, {"body", body}, {"input_ids", ids_json(input.ids)},
                {"sample", g.do_sample}, {"seed", std::to_string(g.seed)}, {"max_new_tokens", g.max_new_tokens}, {"max_context", g.max_context},
                {"eos_ids", ids_json(g.eos_ids)}, {"temperature", g.sampling.temperature}, {"top_p", g.sampling.top_p}, {"top_k", g.sampling.top_k},
                {"repetition_penalty", g.sampling.repetition_penalty}, {"stops", stops}});
        }
        if (input.generation.max_new_tokens == 0) s.complete(handle);
        else s.observed.emplace(handle.id(), Impl::Observation{handle, input.ids.size(), 0, 0, {}});
        std::chrono::seconds::rep created = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::system_clock::now().time_since_epoch()).count();
        return ChatRequest{handle, output, input.ids.size(), input.stream, input.include_usage,
                           s.prefix + std::to_string(handle.id()), double(created), input.token_ids};
    } catch (...) {
        handle.cancel(); output->abandon(); throw;
    }
}
void ChatService::shutdown() {
    ChatService::Impl& s = *impl_; s.stopping.store(true); std::lock_guard<std::mutex> lock(s.mutex); if (s.scheduler) s.scheduler->request_shutdown();
}
void ChatService::wait() {
    if (impl_->worker.joinable()) impl_->worker.join();
    std::lock_guard<std::mutex> lock(impl_->mutex); if (impl_->failure) std::rethrow_exception(impl_->failure);
}
bool ChatService::stopped() const { return impl_->stopping.load(); }
SchedulerStats ChatService::stats() const {
    std::lock_guard<std::mutex> lock(impl_->mutex); return impl_->scheduler ? impl_->scheduler->stats() : impl_->final_stats;
}
const ChatServiceOptions& ChatService::options() const { return impl_->options; }
} // namespace elman
