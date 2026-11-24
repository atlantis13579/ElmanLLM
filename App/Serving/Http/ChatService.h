#pragma once

#include "Serving/GenerationScheduler.h"
#include "Text/TokenStream.h"
#include "Networking/HttpServer.h"
#include <condition_variable>
#include <deque>
#include <mutex>
#include <optional>

namespace elman {
using ChatError = HttpError;
struct ChatServiceOptions {
    std::string model_directory, model_name, trace_path;
    Backend backend = Backend::CPU;
    DType dtype = DType::Float32;
    SchedulerOptions scheduler;
    bool direct_paged_attention = false;
    bool token_ids_only = false;
    int max_new_tokens = 64;
    size_t output_bytes = 1024 * 1024, stream_bytes = 64 * 1024, stream_chunks = 256;
};
struct ChatInput {
    GenerationOptions generation;
    std::vector<int32_t> ids;
    std::vector<std::string> stops;
    bool stream = false, include_usage = false;
    bool token_ids = false;
};

// One producer (the model worker) and one consumer (the connection thread).
// Completion/finalization happens only after the generation handle is ready.
// The producer never waits for network consumption. Full queues fail that row.
class ChatOutput {
public:
    ChatOutput(const HuggingFaceTokenizer& tokenizer, const ChatInput& input, const ChatServiceOptions& options);
    ChatOutput(const HuggingFaceTokenizer* tokenizer, const ChatInput& input, const ChatServiceOptions& options);
    bool accept(int32_t id);
    std::string pop(std::chrono::milliseconds timeout);
    // Drains no queue entries. tail contains final decoder/stop-filter bytes;
    // send queued pieces first, then tail, then the completion envelope.
    GenerationResult finish(const GenerationResult& result, std::string& tail);
    void abandon();
private:
    void append(const std::string& text, bool enqueue);
    MemoryReservation memory_;
    std::optional<TokenizerStream> decoder_;
    TextStopFilter filter_;
    bool stream_, finished_ = false;
    size_t output_limit_, queue_limit_, chunk_limit_, queued_bytes_ = 0;
    std::string text_;
    std::mutex mutex_;
    std::condition_variable changed_;
    std::deque<std::string> pieces_;
    bool abandoned_ = false;
};
struct ChatRequest {
    GenerationRequest generation;
    std::shared_ptr<ChatOutput> output;
    size_t prompt_tokens = 0;
    bool stream = false, include_usage = false;
    std::string id;
    double created = 0;
    bool token_ids = false;
};

// Owns one model worker. The tokenizer is immutable; all encoding is serialized
// separately from GPU execution. Destruction requires quiescent public callers.
class ChatService {
public:
    explicit ChatService(ChatServiceOptions options);
    ~ChatService();
    ChatService(const ChatService&) = delete;
    ChatService& operator=(const ChatService&) = delete;
    ChatRequest submit(const Json& body);
    void shutdown();
    // Join after shutting down and quiescing client calls; rethrow worker failure.
    void wait();
    bool stopped() const;
    SchedulerStats stats() const;
    const ChatServiceOptions& options() const;
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
} // namespace elman
