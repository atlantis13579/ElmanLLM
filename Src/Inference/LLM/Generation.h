#pragma once

#include "Models/LLM/Qwen2.h"
#include "Inference/LLM/Sampling.h"
#include "Inference/LLM/MemoryPlan.h"
#include "Inference/LLM/PrefixCache.h"
#include "Core/AllocationPool.h"
#include <functional>
#include <exception>

namespace elman {
class HuggingFaceTokenizer;
class PagedKVPool;
class Qwen2PagedGraph;

struct GreedyOptions {
    int max_new_tokens = 64;
    int max_context = 4096;
    std::vector<int32_t> eos_ids;
    MemoryBudget memory;
    bool reuse_memory = true;
    bool asynchronous = true;
    bool pinned_staging = true;
    bool paged_cache = false;
    bool full_logits = false;
    bool cuda_graph = false;
    int prefill_chunk_size = 0; // Zero consumes the entire prompt in one step.
    // Runs after each committed prompt chunk, before token selection. False
    // cancels without generating a token; exceptions release request resources.
    std::function<bool(int consumed, int total)> on_prefill;
};

struct GenerationOptions : GreedyOptions {
    bool do_sample = false;
    SamplingOptions sampling;
    uint64_t seed = 0;
};

struct GenerationResult {
    std::vector<int32_t> ids;
    std::string finish_reason = "length";
    std::string text, matched_stop;
    double prefill_ms = 0, decode_ms = 0;
    int prefill_tokens = 0, prefill_chunks = 0;
    int cached_prompt_tokens = 0;
    uint64_t prefix_entry = 0;
    MemoryReport memory;
    std::vector<DeviceMemoryReport> device_memory;
    AllocationPoolStats allocations;
    size_t graph_commands = 0, graph_nodes = 0, graph_replays = 0, graph_retained_bytes = 0;
    double graph_capture_ms = 0;
};

enum class GenerationState { Prefill, Decode, Finished, Failed };

// Pure request validation, safe before dispatch to a model worker. Returns the
// output limit after clipping to the requested total context length.
int validate_generation_request(const DecoderConfig& config, const std::vector<int32_t>& input,
                                const GenerationOptions& options);

struct GenerationBatchOptions {
    int token_budget = 128; // Total model input tokens across active rows.
    MemoryBudget memory;
    bool reuse_memory = true;
    bool asynchronous = true;
    bool pinned_staging = true;
    std::unique_ptr<Qwen2PagedGraph>* decode_graph = nullptr;
    int graph_context_limit = 0;
};
struct GenerationBatchResult {
    int query_tokens = 0, input_tokens = 0;
    std::vector<int> consumed; // Caller order; zero for rows excluded before the forward.
    std::vector<std::exception_ptr> errors; // Row-local selection/callback failures.
    std::vector<PrefixPublication> publications; // Caller order; inserted committed prompt prefixes.
    MemoryReport memory;
    AllocationPoolStats allocations;
    double elapsed_ms = 0;
    size_t graph_captures = 0, graph_replays = 0, graph_commands = 0, graph_nodes = 0, graph_retained_bytes = 0;
    double graph_capture_ms = 0;
};

// A resumable, unpadded request. The model must outlive the session. A shared
// pool selects paged execution and retains its physical storage across requests.
// step() consumes at most one prompt chunk or one decode input; token_budget
// bounds model input tokens, not output tokens. The final prompt chunk selects
// the first output ID. No logits or sampling state are shared between requests.
// All operations except request_cancel() belong to the constructing thread.
// Cancellation is observed before/after the next forward and releases KV leases.
// Any execution/admission/callback exception terminates this request and is
// rethrown; other requests and the shared pool remain usable.
class GenerationSession {
public:
    GenerationSession(const DecoderModel& model, const std::vector<int32_t>& input,
                      const GenerationOptions& options,
                      const std::function<bool(int32_t)>& on_token = {},
                      std::shared_ptr<PagedKVPool> shared_pool = {},
                      std::shared_ptr<PrefixCache> prefix_store = {}, PrefixMatch initial_prefix = {});
    ~GenerationSession();
    GenerationSession(const GenerationSession&) = delete;
    GenerationSession& operator=(const GenerationSession&) = delete;
    int step(int token_budget);
    int pending_input_tokens() const;
    int cache_length() const;
    void cancel();
    void request_cancel() noexcept;
    GenerationState state() const;
    bool finished() const;
    const GenerationResult& result() const;
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    friend GenerationResult generate(const DecoderModel&, const std::vector<int32_t>&,
                                     const GenerationOptions&, const std::function<bool(int32_t)>&);
    friend GenerationBatchResult step_generation_batch(const std::vector<GenerationSession*>&, const GenerationBatchOptions&);
};

// One model forward for distinct sessions on this worker, using the same model
// and physical pool handle. Terminal/cancelled rows are removed before choosing
// a common query length. The total token budget must fit at least one token per
// active row. Batch options control allocation/submission; all request memory
// ceilings still apply. Admission or pre-commit model failure leaves active
// requests retryable. Post-commit selection/callback failures terminate only
// their row and are returned in errors; other rows finish their step normally.
// Batch wall time/allocations belong to this result. Each row's timing includes
// shared forward time, so row times must not be added to estimate throughput.
// Sessions must outlive the call, including every participant's callbacks.
GenerationBatchResult step_generation_batch(const std::vector<GenerationSession*>& sessions,
                                             const GenerationBatchOptions& options = {});

// One unpadded request. Each call owns its cache; no state leaks between requests.
// The generated EOS is retained in IDs. Context counts prompt plus output IDs.
// on_token observes each selected ID; false stops before the next model call.
// Callback stopping takes priority over EOS and includes the current ID.
GenerationResult generate_greedy(const DecoderModel& model, const std::vector<int32_t>& input,
                                 const GreedyOptions& options,
                                 const std::function<bool(int32_t)>& on_token = {});
GenerationResult generate(const DecoderModel& model, const std::vector<int32_t>& input,
                          const GenerationOptions& options,
                          const std::function<bool(int32_t)>& on_token = {});
GenerationResult generate_text(const DecoderModel& model, const HuggingFaceTokenizer& tokenizer,
                               const std::vector<int32_t>& input, const GenerationOptions& options,
                               const std::vector<std::string>& stops = {},
                               const std::function<void(const std::string&)>& on_text = {});
GenerationResult generate_text(const DecoderModel& model, const HuggingFaceTokenizer& tokenizer,
                               const std::vector<int32_t>& input, const GreedyOptions& options,
                               const std::vector<std::string>& stops = {},
                               const std::function<void(const std::string&)>& on_text = {});
int generate_command(int argc, char** argv);

} // namespace elman
