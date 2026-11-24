#include "Serving/GenerationScheduler.h"
#include "Core/LocalFunction.h"
#include "Inference/LLM/PagedKVCache.h"
#include "Inference/LLM/Qwen2PagedGraph.h"
#include "Core/Execution.h"
#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <limits>
#include <mutex>
#include <thread>

namespace elman {
namespace {
void require(bool condition, const char* message) {
    if (!condition) throw std::invalid_argument(std::string("Scheduler: ") + message);
}
}

struct GenerationRequestState {
    uint64_t id;
    MemoryReservation host;
    mutable std::mutex mutex;
    std::condition_variable changed;
    std::atomic<bool> done{false}, cancelled{false};
    GenerationSession* active = nullptr; // Protected until detached before destruction.
    GenerationResult result;
    std::exception_ptr error;

    GenerationRequestState(uint64_t id_, size_t input, int count, size_t eos)
        : id(id_), host(Backend::CPU, checked_memory_add(64 * 1024,
            checked_memory_mul(checked_memory_add(checked_memory_add(input, size_t(count)), eos), 32))) {
        result.ids.reserve(size_t(count));
    }
    void cancel() {
        std::lock_guard<std::mutex> lock(mutex);
        if (done.load()) return;
        cancelled.store(true);
        if (active) active->request_cancel();
    }
    void publish(const GenerationResult& value, std::exception_ptr failure = {}) {
        {
            std::lock_guard<std::mutex> lock(mutex);
            result = value;
            error = failure;
            active = nullptr;
            done.store(true, std::memory_order_release);
        }
        changed.notify_all();
    }
};

uint64_t GenerationRequest::id() const { require(bool(state_), "empty request handle"); return state_->id; }
bool GenerationRequest::ready() const {
    require(bool(state_), "empty request handle");
    return state_->done.load(std::memory_order_acquire);
}
bool GenerationRequest::wait_for(std::chrono::milliseconds timeout) const {
    require(bool(state_), "empty request handle");
    std::unique_lock<std::mutex> lock(state_->mutex);
    return state_->changed.wait_for(lock, timeout, [&] { return state_->done.load(); });
}
void GenerationRequest::cancel() const { require(bool(state_), "empty request handle"); state_->cancel(); }
const GenerationResult& GenerationRequest::result() const {
    require(ready(), "request result is not ready");
    return state_->result;
}
std::exception_ptr GenerationRequest::error() const {
    require(ready(), "request error is not ready");
    return state_->error;
}

struct GenerationScheduler::Impl {
    struct Waiting {
        std::shared_ptr<GenerationRequestState> state;
        std::vector<int32_t> input;
        GenerationOptions options;
        std::function<bool(int32_t)> callback;
        size_t blocks;
        int count;
    };
    struct Active {
        std::shared_ptr<GenerationRequestState> state;
        std::unique_ptr<GenerationSession> session;
        size_t blocks;
        std::exception_ptr error;
    };
    const DecoderModel& model;
    const SchedulerOptions options;
    const std::thread::id owner = std::this_thread::get_id();
    const int device;
    MemoryReservation metadata;
    std::shared_ptr<PagedKVPool> pool;
    std::shared_ptr<PrefixCache> prefixes;
    std::unique_ptr<Qwen2PagedGraph> graph;
    mutable std::mutex mutex;
    std::condition_variable work;
    std::vector<Waiting> waiting;
    std::vector<std::shared_ptr<GenerationRequestState>> live;
    std::vector<Active> active; // Worker-only; published counts use mutex.
    SchedulerStats counters;
    uint64_t next_id = 1, prefill_cursor = 0, decode_cursor = 0;
    bool stepping = false, next_prefill = true;

    Impl(const DecoderModel& model_, const SchedulerOptions& options_)
        : model(model_), options(options_), device(detail::allocation_device(model.backend())) {
        require(options.max_queued_requests > 0 && options.max_active_requests > 0, "request limits must be positive");
        require(options.max_batch_tokens > 0 && options.prefill_chunk_size > 0, "token budgets must be positive");
        require(options.max_context > 0 && options.max_context <= model.config().max_positions, "invalid context limit");
        require(!options.cuda_graph || (model.backend() == Backend::CUDA && model.supports_paged_cache() &&
                model.tiled_attention_enabled() && dynamic_cast<const Qwen2*>(&model)),
                "CUDA Graph serving requires the single-device paged Qwen decoder with tiled attention");
        size_t descriptors = checked_memory_mul(checked_memory_add(size_t(options.max_queued_requests),
            size_t(options.max_active_requests)), 1024);
        if (!model.supports_paged_cache()) {
            require(options.prefix_cache_blocks == 0, "prefix sharing requires a decoder with paged cache support");
            MemoryAdmission admission(model.backend(), options.memory, model.backend() == Backend::CPU ? descriptors : 0, descriptors);
            metadata = MemoryReservation(Backend::CPU, descriptors);
            reserve_requests();
            return;
        }
        require(options.cache_blocks > 0, "cache capacity must be positive");
        require(options.prefix_cache_blocks >= 0 && options.prefix_cache_blocks <= options.cache_blocks &&
                options.prefix_cache_entries >= 0 && (!options.prefix_cache_blocks || options.prefix_cache_entries > 0),
                "invalid prefix cache bounds");
        const DecoderConfig& c = model.config();
        PagedKVShape shape{c.layers, c.kv_heads, c.head_dim(), 16, options.cache_blocks};
        QwenBatchMemoryPlan minimum = plan_qwen_batch(c, model.backend(), model.dtype(), 1, 1, size_t(options.max_context),
            Qwen2Logits::Last, true, model.tiled_attention_enabled(), 16, model.quantization_workspace_bytes());
        size_t host = checked_memory_add(descriptors, PagedKVPool::metadata_bytes(shape));
        if (options.prefix_cache_blocks) {
            host = checked_memory_add(host, PrefixCache::metadata_bytes(shape,size_t(options.prefix_cache_blocks),size_t(options.prefix_cache_entries)));
            size_t publication = checked_memory_mul(size_t(options.prefix_cache_blocks),2*sizeof(CacheBlock));
            minimum.host_additional_bytes = checked_memory_add(minimum.host_additional_bytes,publication);
            if (model.backend()==Backend::CPU) minimum.additional_bytes=checked_memory_add(minimum.additional_bytes,publication);
        }
        size_t payload = PagedKVPool::payload_bytes(shape, model.backend(), model.dtype());
        if (model.backend()==Backend::CUDA && options.asynchronous && options.pinned_staging)
            minimum.host_additional_bytes=checked_memory_add(minimum.host_additional_bytes,host_staging_memory_bound());
        MemoryAdmission admission(model.backend(), options.memory,
            checked_memory_add(checked_memory_add(payload, minimum.additional_bytes), model.backend() == Backend::CPU ? host : 0),
            checked_memory_add(host, minimum.host_additional_bytes));
        metadata = MemoryReservation(Backend::CPU, descriptors);
        reserve_requests();
        pool = std::make_shared<PagedKVPool>(shape, model.backend(), model.dtype());
        if (options.prefix_cache_blocks) prefixes=std::make_shared<PrefixCache>(pool,size_t(options.prefix_cache_blocks),size_t(options.prefix_cache_entries));
        counters.capacity_blocks = size_t(options.cache_blocks);
    }
    void reserve_requests() {
        waiting.reserve(size_t(options.max_queued_requests));
        active.reserve(size_t(options.max_active_requests));
        live.reserve(checked_memory_add(size_t(options.max_queued_requests), size_t(options.max_active_requests)));
    }
    void check_owner() const {
        require(owner == std::this_thread::get_id(), "worker belongs to another thread");
        require(device == detail::allocation_device(model.backend()), "worker belongs to another device");
    }
    void record_completion_locked(const std::shared_ptr<GenerationRequestState>& state) {
        ++counters.completed;
        if (state->result.finish_reason == "cancelled") ++counters.cancelled;
        if (state->error) ++counters.failed;
        live.erase(std::remove(live.begin(), live.end(), state), live.end());
    }
    void refresh_locked() {
        if (!pool) return;
        CacheBlockStats pages=pool->stats();size_t reserved=pages.live;
        for (GenerationScheduler::Impl::Active& row:active) if (!row.session->finished()) {
            size_t allocated=(size_t(row.session->cache_length())+15)/16;
            if (allocated>row.blocks) throw std::logic_error("Scheduler: request exceeded its future quota");
            reserved=checked_memory_add(reserved,row.blocks-allocated);
        }
        if (reserved>counters.capacity_blocks) throw std::logic_error("Scheduler: shared pages exceed future quota");
        counters.live_blocks=pages.live;counters.peak_live_blocks=pages.peak;
        counters.reserved_blocks=reserved;
        counters.peak_reserved_blocks=std::max(counters.peak_reserved_blocks,reserved);
        if (prefixes) counters.prefix=prefixes->stats();
    }
    void complete_active(size_t index, std::exception_ptr failure = {}) {
        GenerationScheduler::Impl::Active& row = active[index];
        GenerationResult output = row.session->result();
        if (failure && !row.session->finished()) output.finish_reason = "error";
        row.state->publish(output, failure ? failure : row.error);
        // publish() detaches the cancellation bridge before releasing the session.
        row.session.reset();
        {
            std::lock_guard<std::mutex> lock(mutex);
            --counters.active;
            record_completion_locked(row.state);
            active.erase(active.begin() + index);
            refresh_locked();
        }
    }
    void reap() {
        for (size_t index = 0; index < active.size();) {
            GenerationScheduler::Impl::Active& row = active[index];
            if (!row.session->finished() && row.state->cancelled.load()) row.session->cancel();
            if (row.session->finished()) complete_active(index);
            else ++index;
        }
        // Queued cancellations cannot be held behind a head waiting for quota.
        std::lock_guard<std::mutex> lock(mutex);
        for (size_t index = 0; index < waiting.size();) {
            GenerationScheduler::Impl::Waiting& row = waiting[index];
            if (!row.state->cancelled.load()) { ++index; continue; }
            GenerationResult output; output.finish_reason = "cancelled";
            row.state->publish(output);
            record_completion_locked(row.state);
            waiting.erase(waiting.begin() + index);
            --counters.queued;
        }
        if (counters.closed && active.empty() && prefixes) prefixes->clear();
        refresh_locked();
    }
    void admit(std::vector<PrefixCacheEvent>& events) {
        // Queue state and the active cancellation bridge publish together. No
        // model dispatch or user callbacks occur under this lock.
        std::lock_guard<std::mutex> lock(mutex);
        // Contiguous cache sessions use the model's single-request memory plan.
        const size_t active_limit = pool ? size_t(options.max_active_requests) : 1;
        while (!counters.closed && !waiting.empty() && active.size() < active_limit) {
            GenerationScheduler::Impl::Waiting& row = waiting.front();
            LocalFunction fits{[&](int cached) { return row.blocks-size_t(cached/16)<=counters.capacity_blocks-counters.reserved_blocks; }};
            int matched=prefixes ? prefixes->matched_tokens(row.input) : 0;
            while (!fits(matched) && prefixes && prefixes->reclaim_one()) {
                refresh_locked();matched=prefixes->matched_tokens(row.input);
            }
            if (!fits(matched)) break;
            std::unique_ptr<GenerationSession> session;
            uint64_t entry = 0;
            int reused = 0;
            try {
                PrefixMatch prefix=prefixes ? prefixes->lookup_if_possible(row.input,int(row.input.size())+row.count) : PrefixMatch{};
                while (!fits(prefix.tokens) && prefixes && prefixes->reclaim_one()) refresh_locked();
                if (!fits(prefix.tokens)) break;
                entry = prefix.entry; reused = prefix.tokens;
                session = std::make_unique<GenerationSession>(model, row.input, row.options, row.callback, pool, prefixes, std::move(prefix));
            } catch (...) {
                GenerationResult output; output.finish_reason = "error";
                row.state->publish(output, std::current_exception());
                record_completion_locked(row.state);
            }
            if (session) {
                {
                    std::lock_guard<std::mutex> request_lock(row.state->mutex);
                    row.state->active = session.get();
                    if (row.state->cancelled.load()) session->request_cancel();
                }
                // Both vectors have capacity reserved before any row mutation.
                active.push_back(Active{row.state, std::move(session), row.blocks, {}});
                ++counters.active; ++counters.admitted;
                counters.peak_active = std::max(counters.peak_active, counters.active);
                if (reused) events.push_back({row.state->id, entry, reused, false});
            }
            waiting.erase(waiting.begin());
            --counters.queued;
            refresh_locked();
        }
    }
    std::vector<Active*> select(bool& prefill) {
        bool has_prefill = false, has_decode = false;
        for (GenerationScheduler::Impl::Active& row : active) {
            if (row.session->finished() || row.state->cancelled.load()) continue;
            if (row.session->state() == GenerationState::Prefill) has_prefill = true;
            else has_decode = true;
        }
        prefill = has_prefill && (!has_decode || next_prefill);
        if (has_prefill && has_decode) next_prefill = !prefill;
        std::vector<Active*> rows;
        for (GenerationScheduler::Impl::Active& row : active) if (!row.session->finished() && !row.state->cancelled.load() &&
            (row.session->state() == GenerationState::Prefill) == prefill) rows.push_back(&row);
        uint64_t cursor = prefill ? prefill_cursor : decode_cursor;
        auto start = std::find_if(rows.begin(), rows.end(), [&](Active* row) { return row->state->id > cursor; });
        std::rotate(rows.begin(), start, rows.end());
        if (rows.size() > size_t(options.max_batch_tokens)) rows.resize(size_t(options.max_batch_tokens));
        return rows;
    }
    void update_pages() {
        std::lock_guard<std::mutex> lock(mutex);
        refresh_locked();
    }
    void record_model_step(int consumed) {
        if (!consumed) return;
        std::lock_guard<std::mutex> lock(mutex);
        ++counters.model_steps;
        counters.input_tokens += uint64_t(consumed);
    }
};

GenerationScheduler::GenerationScheduler(const DecoderModel& model, const SchedulerOptions& options)
    : impl_(std::make_unique<Impl>(model, options)) {}
GenerationScheduler::~GenerationScheduler() {
    // Callers are quiescent. Avoid user callbacks, but wake every waiting handle.
    request_shutdown();
    impl_->reap();
}
GenerationRequest GenerationScheduler::submit(const std::vector<int32_t>& input, const GenerationOptions& options,
                                              const std::function<bool(int32_t)>& on_token) {
    GenerationScheduler::Impl& s = *impl_;
    int count = validate_generation_request(s.model.config(), input, options);
    require(options.max_context <= s.options.max_context, "request exceeds worker context limit");
    size_t blocks = s.pool ? (input.size() + size_t(count) + 15) / 16 : 0;
    require(!s.pool || !count || blocks <= size_t(s.options.cache_blocks), "request future pages exceed the entire pool");
    std::unique_lock<std::mutex> lock(s.mutex);
    if (s.counters.closed) throw SchedulerClosed();
    if (count && s.waiting.size() == size_t(s.options.max_queued_requests)) throw RequestQueueFull();
    require(s.next_id != std::numeric_limits<uint64_t>::max(), "request ID space exhausted");
    std::shared_ptr<GenerationRequestState> state = std::make_shared<GenerationRequestState>(s.next_id, input.size(), count, options.eos_ids.size());
    if (count) {
        GenerationOptions request_options = options;
        request_options.paged_cache = bool(s.pool);
        if (!s.pool) {
            request_options.memory = s.options.memory;
            request_options.reuse_memory = s.options.reuse_memory;
            request_options.asynchronous = s.options.asynchronous;
            request_options.pinned_staging = s.options.pinned_staging;
        }
        request_options.prefill_chunk_size = std::min(options.prefill_chunk_size ? options.prefill_chunk_size :
            s.options.prefill_chunk_size, s.options.prefill_chunk_size);
        s.waiting.push_back(Impl::Waiting{state, input, std::move(request_options), on_token, blocks,count});
        s.live.push_back(state);
        ++s.counters.queued;
        s.counters.peak_queued = std::max(s.counters.peak_queued, s.counters.queued);
    } else {
        state->publish(GenerationResult{});
        ++s.counters.completed;
    }
    ++s.next_id; ++s.counters.submitted;
    lock.unlock();
    s.work.notify_one();
    return GenerationRequest(std::move(state));
}

SchedulerStep GenerationScheduler::step() {
    GenerationScheduler::Impl& s = *impl_; s.check_owner();
    require(!s.stepping, "cannot reenter the worker");
    s.stepping = true;
    struct Reset { bool& flag; ~Reset() { flag = false; } } reset{s.stepping};
    SchedulerStats before = stats();
    SchedulerStep result;
    result.prefixes.reserve(size_t(s.options.max_active_requests)*2);
    s.reap(); s.admit(result.prefixes); s.reap();
    std::vector<Impl::Active*> selected = s.select(result.prefill);
    if (!selected.empty() && !s.pool) {
        GenerationScheduler::Impl::Active& row = *selected.front();
        result.requests.push_back(row.state->id);
        result.attempts = 1;
        result.batch.consumed.resize(1);
        result.batch.errors.resize(1);
        result.batch.publications.resize(1);
        try {
            int consumed = row.session->step(s.options.max_batch_tokens);
            result.batch.consumed.front() = consumed;
            result.batch.query_tokens = result.batch.input_tokens = consumed;
            s.record_model_step(consumed);
        } catch (...) {
            row.error = result.batch.errors.front() = std::current_exception();
        }
    } else if (!selected.empty()) {
        GenerationBatchOptions batch_options;
        batch_options.token_budget = s.options.max_batch_tokens;
        batch_options.memory = s.options.memory;
        batch_options.reuse_memory = s.options.reuse_memory;
        batch_options.asynchronous = s.options.asynchronous;
        batch_options.pinned_staging = s.options.pinned_staging;
        if (s.options.cuda_graph) {
            batch_options.decode_graph = &s.graph;
            batch_options.graph_context_limit = s.options.max_context;
        }
        for (;;) {
            std::vector<GenerationSession*> sessions;
            result.requests.clear();
            for (GenerationScheduler::Impl::Active* row : selected) { sessions.push_back(row->session.get()); result.requests.push_back(row->state->id); }
            ++result.attempts;
            try {
                result.batch = step_generation_batch(sessions, batch_options);
                for (size_t i = 0; i < selected.size(); ++i) selected[i]->error = result.batch.errors[i];
                for (size_t i = 0; i < selected.size(); ++i) {
                    const PrefixPublication& publication=result.batch.publications[i];
                    if (publication.inserted) result.prefixes.push_back({selected[i]->state->id,publication.entry,publication.tokens,true});
                }
                s.record_model_step(result.batch.input_tokens);
                if (result.batch.graph_captures || result.batch.graph_replays) {
                    std::lock_guard<std::mutex> lock(s.mutex);
                    s.counters.graph_captures += result.batch.graph_captures;
                    s.counters.graph_replays += result.batch.graph_replays;
                }
                break;
            } catch (const MemoryLimitExceeded&) {
                std::exception_ptr error = std::current_exception();
                // A post-commit failure is terminal in the batch executor. Never
                // replay it, and never retry a shape changed by cancellation.
                bool terminal = false;
                for (GenerationScheduler::Impl::Active* row : selected) if (row->session->finished()) {
                    terminal = true;
                    if (row->session->state() == GenerationState::Failed) row->error = error;
                }
                if (terminal || std::any_of(selected.begin(), selected.end(), [](Impl::Active* row) { return row->state->cancelled.load(); })) break;
                int query = batch_options.token_budget / int(selected.size());
                for (GenerationScheduler::Impl::Active* row : selected) query = std::min(query, row->session->pending_input_tokens());
                if (query > 1) batch_options.token_budget = int(selected.size()) * std::max(1, query / 2);
                else if (selected.size() > 1) { selected.resize(std::max(size_t(1), selected.size() / 2)); batch_options.token_budget = int(selected.size()); }
                else {
                    size_t index = size_t(selected.front() - s.active.data());
                    (result.prefill ? s.prefill_cursor : s.decode_cursor) = selected.front()->state->id;
                    s.complete_active(index, error);
                    selected.clear();
                    break;
                }
                std::lock_guard<std::mutex> lock(s.mutex); ++s.counters.memory_retries;
            } catch (...) {
                std::exception_ptr error = std::current_exception();
                // Preserve row completion reasons if the executor already
                // published them; only unfinished rows become worker failures.
                for (GenerationScheduler::Impl::Active* row : selected) row->error = error;
                std::vector<uint64_t> failed = result.requests;
                for (uint64_t id : failed) {
                    auto found = std::find_if(s.active.begin(), s.active.end(), [&](const Impl::Active& row) { return row.state->id == id; });
                    if (found != s.active.end()) s.complete_active(size_t(found - s.active.begin()), error);
                }
                selected.clear();
                break;
            }
        }
        if (!selected.empty()) (result.prefill ? s.prefill_cursor : s.decode_cursor) = selected.back()->state->id;
    }
    s.reap(); s.update_pages();
    SchedulerStats after = stats();
    result.admitted = size_t(after.admitted - before.admitted);
    result.completed = size_t(after.completed - before.completed);
    return result;
}

bool GenerationScheduler::wait_for_work(std::chrono::milliseconds timeout) {
    GenerationScheduler::Impl& s = *impl_; s.check_owner();
    require(!s.stepping, "cannot wait inside a worker callback");
    std::unique_lock<std::mutex> lock(s.mutex);
    return s.work.wait_for(lock, timeout, [&] { return s.counters.closed || s.counters.queued || s.counters.active; });
}
void GenerationScheduler::request_shutdown() {
    GenerationScheduler::Impl& s = *impl_;
    {
        std::lock_guard<std::mutex> lock(s.mutex);
        s.counters.closed = true;
        for (std::shared_ptr<GenerationRequestState>& state : s.live) state->cancel();
    }
    s.work.notify_all();
}
SchedulerStats GenerationScheduler::stats() const {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    return impl_->counters;
}
} // namespace elman
