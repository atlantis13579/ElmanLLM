#pragma once

#include "Inference/LLM/Generation.h"
#include <chrono>
#include <stdexcept>

namespace elman {
class RequestQueueFull : public std::runtime_error {
public:
    RequestQueueFull() : std::runtime_error("Scheduler: waiting queue is full") {}
};
class SchedulerClosed : public std::runtime_error {
public:
    SchedulerClosed() : std::runtime_error("Scheduler: submissions are closed") {}
};

struct SchedulerOptions {
    int max_queued_requests = 64, max_active_requests = 4;
    int max_batch_tokens = 128, prefill_chunk_size = 128;
    int max_context = 4096, cache_blocks = 512;
    int prefix_cache_blocks = 0, prefix_cache_entries = 64;
    MemoryBudget memory;
    bool reuse_memory = true, asynchronous = true;
    bool pinned_staging = true;
    bool cuda_graph = false;
};

struct GenerationRequestState;
// Copies share one request. Cancellation, completion reads and waits are safe
// on any thread. Results are immutable after ready(), and survive the scheduler.
// User callbacks execute on the worker and must not wait for their own result.
class GenerationRequest {
public:
    GenerationRequest() = default;
    uint64_t id() const;
    bool ready() const;
    bool wait_for(std::chrono::milliseconds timeout) const;
    void cancel() const;
    const GenerationResult& result() const;
    std::exception_ptr error() const;
    explicit operator bool() const { return bool(state_); }
private:
    explicit GenerationRequest(std::shared_ptr<GenerationRequestState> state) : state_(std::move(state)) {}
    std::shared_ptr<GenerationRequestState> state_;
    friend class GenerationScheduler;
};

struct SchedulerStats {
    // Page observations are refreshed at worker step boundaries; queue and
    // lifecycle counters publish as requests enter and leave the scheduler.
    size_t queued = 0, active = 0, reserved_blocks = 0, live_blocks = 0, capacity_blocks = 0;
    size_t peak_queued = 0, peak_active = 0, peak_reserved_blocks = 0, peak_live_blocks = 0;
    uint64_t submitted = 0, admitted = 0, completed = 0, cancelled = 0, failed = 0;
    uint64_t model_steps = 0, input_tokens = 0, memory_retries = 0;
    uint64_t graph_captures = 0, graph_replays = 0;
    bool closed = false;
    PrefixCacheStats prefix;
};
struct PrefixCacheEvent {
    uint64_t request = 0, entry = 0;
    int tokens = 0;
    bool publication = false;
};
struct SchedulerStep {
    std::vector<uint64_t> requests; // Order passed to the final batch attempt.
    GenerationBatchResult batch;
    bool prefill = false;
    size_t admitted = 0, completed = 0, attempts = 0;
    std::vector<PrefixCacheEvent> prefixes;
};

// The constructing thread/device owns step(), wait_for_work() and destruction.
// submit(), stats() and request_shutdown() are thread-safe. Destruction requires
// quiescent callers and completes unfinished handles as cancelled. The model
// must outlive this object. Callbacks may submit/cancel/read stats, but may not
// reenter step() or destroy the scheduler.
//
// Admission is FIFO and reserves each request's maximum future page demand.
// With prefix retention enabled, shared resident pages count once and only
// each active request's unallocated suffix contributes additional future pages.
// Physical pages are allocated lazily within one fixed private pool. Queue
// entries own no KV. Prefill/decode phases alternate, with round-robin selection
// inside each phase. Memory admission can reduce query length and then batch
// width before any committed forward; a failing one-row, one-token request is
// isolated. Page quotas guarantee cache capacity, not future free host/device RAM.
// A decoder without paged cache support runs one contiguous session at a time.
class GenerationScheduler {
public:
    GenerationScheduler(const DecoderModel& model, const SchedulerOptions& options = {});
    ~GenerationScheduler();
    GenerationScheduler(const GenerationScheduler&) = delete;
    GenerationScheduler& operator=(const GenerationScheduler&) = delete;
    GenerationRequest submit(const std::vector<int32_t>& input, const GenerationOptions& options,
                             const std::function<bool(int32_t)>& on_token = {});
    SchedulerStep step();
    bool wait_for_work(std::chrono::milliseconds timeout);
    void request_shutdown();
    SchedulerStats stats() const;
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
} // namespace elman
