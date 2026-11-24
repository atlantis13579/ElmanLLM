#include "Inference/LLM/Generation.h"
#include "Core/LocalFunction.h"
#include "Inference/LLM/PagedKVCache.h"
#include "Inference/LLM/Qwen2Graph.h"
#include "Inference/LLM/Qwen2PagedGraph.h"
#include "Core/Execution.h"
#include "Inference/LLM/GenerationMemory.h"
#include "Models/LLM/DecoderFactory.h"
#include "Text/HuggingFaceTokenizer.h"
#include "Text/TokenStream.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <limits>
#include <map>
#include <stdexcept>
#include <thread>

namespace elman {
namespace {
using Clock = std::chrono::steady_clock;
ExecutionStream inference_stream(Backend backend) {
    // A worker reuses its queue and vendor context across sequential requests.
    thread_local std::map<std::pair<Backend, int>, ExecutionStream> streams;
    std::pair<Backend, int> key = std::make_pair(backend, detail::allocation_device(backend));
    auto found = streams.find(key);
    if (found == streams.end()) found = streams.emplace(key, ExecutionStream(backend)).first;
    return found->second;
}
size_t staging_allowance(Backend backend, bool asynchronous, bool enabled) {
    return backend == Backend::CUDA && asynchronous && enabled ? host_staging_memory_bound() : 0;
}
struct FailedTransferGuard {
    const ExecutionStream& stream;
    const DecoderModel* model = nullptr;
    int exceptions = std::uncaught_exceptions();
    ~FailedTransferGuard() {
        if (std::uncaught_exceptions() <= exceptions) return;
        try { if (model) model->synchronize(); stream.synchronize(); stream.trim_host_staging(); } catch (...) {}
    }
};
double elapsed(Clock::time_point start) {
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}
void require(bool condition, const std::string& message) {
    if (!condition) throw std::invalid_argument("Generation: " + message);
}
void merge_memory(MemoryReport& memory, const MemoryReport& report) {
    memory.budget_bytes = std::max(memory.budget_bytes, report.budget_bytes);
    memory.host_budget_bytes = std::max(memory.host_budget_bytes, report.host_budget_bytes);
    memory.estimated_peak_bytes = std::max(memory.estimated_peak_bytes, report.estimated_peak_bytes);
    memory.measured_peak_bytes = std::max(memory.measured_peak_bytes, report.measured_peak_bytes);
    memory.host_estimated_peak_bytes = std::max(memory.host_estimated_peak_bytes, report.host_estimated_peak_bytes);
    memory.host_measured_peak_bytes = std::max(memory.host_measured_peak_bytes, report.host_measured_peak_bytes);
}
int validate_request(const DecoderConfig& config, const std::vector<int32_t>& input, const GenerationOptions& options) {
    if (options.do_sample) validate_sampling(options.sampling);
    require(options.max_new_tokens >= 0, "max_new_tokens must be nonnegative");
    require(options.prefill_chunk_size >= 0, "prefill_chunk_size must be nonnegative");
    require(options.max_context > 0 && options.max_context <= config.max_positions,
            "max_context must be inside the model's position limit");
    require(!input.empty() && input.size() <= size_t(options.max_context), "empty input or prompt exceeds max_context");
    for (int32_t id : input) require(id >= 0 && id < config.vocabulary, "input ID outside the model vocabulary");
    for (int32_t id : options.eos_ids) require(id >= 0 && id < config.vocabulary, "EOS ID outside the model vocabulary");
    return std::min(options.max_new_tokens, options.max_context - static_cast<int>(input.size()));
}
}

HostStagingStats generation_host_staging_stats(Backend backend) {
    return inference_stream(backend).host_staging_stats();
}

int validate_generation_request(const DecoderConfig& config, const std::vector<int32_t>& input,
                                const GenerationOptions& options) {
    return validate_request(config, input, options);
}

struct GenerationSession::Impl {
    const DecoderModel& model;
    GenerationOptions options;
    std::function<bool(int32_t)> on_token;
    std::shared_ptr<PagedKVPool> cache_pool;
    std::unique_ptr<PagedKVCache> cache;
    std::unique_ptr<Qwen2Graph> graph;
    std::unique_ptr<Qwen2PagedGraph> paged_graph;
    std::shared_ptr<PrefixCache> prefixes;
    PrefixPublication publication;
    int count, device;
    MemoryReservation host_request;
    std::vector<int32_t> input, history;
    SamplingRng rng;
    Qwen2Output output;
    GenerationResult result;
    GenerationState state = GenerationState::Prefill;
    std::atomic<bool> cancelled{false};
    std::thread::id owner = std::this_thread::get_id();
    bool initialized = false, busy = false;

    Impl(const DecoderModel& model_, const std::vector<int32_t>& input_, const GenerationOptions& options_,
         const std::function<bool(int32_t)>& callback, std::shared_ptr<PagedKVPool> shared_pool,
         std::shared_ptr<PrefixCache> prefix_store, PrefixMatch initial_prefix)
        : model(model_), options(options_), on_token(callback), cache_pool(std::move(shared_pool)),
          cache(std::move(initial_prefix.cache)), prefixes(std::move(prefix_store)),
          count(validate_request(model.config(), input_, options)), device(model.input_device()),
          host_request(Backend::CPU, checked_memory_add(checked_memory_mul(input_.size() + size_t(count), 32), 64 * 1024)),
          input(input_), history(input_), rng(options.seed) {
        require(model.supports_paged_cache() || (!options.paged_cache && !cache_pool && !prefixes),
                "this decoder placement requires per-layer contiguous caches");
        require(!options.cuda_graph || (dynamic_cast<const Qwen2*>(&model) && model.backend() == Backend::CUDA &&
                model.execution_stages() == 1 && model.tensor_parallel_size() == 1 && model.tiled_attention_enabled()),
                "CUDA Graph requires single-device Qwen with tiled attention");
        if (cache_pool) {
            PagedKVShape shape = cache_pool->shape();
            const DecoderConfig& c = model.config();
            require(cache_pool->backend() == model.backend() && cache_pool->dtype() == model.dtype() &&
                    shape.layers == c.layers && shape.heads == c.kv_heads && shape.dimension == c.head_dim(),
                    "shared cache pool does not match model");
            // The generation memory planner currently uses sixteen-token pages.
            require(shape.block_size == 16, "generation requires sixteen-token cache pages");
            options.paged_cache = true;
        }
        require(!prefixes || (cache_pool && prefixes->belongs_to(*cache_pool)), "prefix index belongs to another pool");
        require(bool(cache) == (initial_prefix.tokens > 0), "prefix handle and length differ");
        if (cache) {
            require(count > 0 && prefixes && initial_prefix.entry > 0 && cache_pool && cache->belongs_to(*cache_pool) && cache->batch() == 1 &&
                    cache->length() == initial_prefix.tokens && initial_prefix.tokens % 16 == 0 &&
                    size_t(initial_prefix.tokens) < input.size() && cache->max_length() >= int(input.size()) + count,
                    "invalid initial prompt prefix");
            result.prefill_tokens = result.cached_prompt_tokens = initial_prefix.tokens;
            result.prefix_entry = initial_prefix.entry;
        }
        if (!count) finish("length");
    }
    bool finished() const { return state == GenerationState::Finished || state == GenerationState::Failed; }
    void check_owner() const {
        require(owner == std::this_thread::get_id(), "session belongs to a different worker thread");
    }
    void finish(const char* reason, bool failed = false) {
        result.finish_reason = reason;
        state = failed ? GenerationState::Failed : GenerationState::Finished;
        output = {};
        graph.reset();
        paged_graph.reset();
        cache.reset();
        cache_pool.reset();
        prefixes.reset();
        // The result remains readable until destruction; its host allowance is
        // retained, while all KV leases and computation tensors are released.
        std::vector<int32_t>().swap(input);
        std::vector<int32_t>().swap(history);
    }
    void initialize() {
        if (initialized) return;
        if (options.paged_cache) {
            const DecoderConfig& c = model.config();
            int total = int(input.size()) + count;
            if (!cache_pool) cache_pool = std::make_shared<PagedKVPool>(
                PagedKVShape{c.layers, c.kv_heads, c.head_dim(), 16, int((size_t(total) + 15) / 16)},
                model.backend(), model.dtype());
            if (!cache) cache = std::make_unique<PagedKVCache>(*cache_pool, 1, total);
        }
        initialized = true;
    }
    int32_t select() {
        int vocabulary = model.config().vocabulary;
        Tensor last = contiguous_view(output.logits, {1, vocabulary}, output.logits.numel() - vocabulary);
        if (options.do_sample) return sample_tokens(last,
            Tensor::from_int_data({1, int(history.size())}, history, model.backend()), options.sampling, rng).front();
        return greedy_token(last);
    }
    int pending_input_tokens() const {
        if (finished() || cancelled.load()) return 0;
        if (state==GenerationState::Decode) return 1;
        int remaining=int(input.size())-result.prefill_tokens;
        return options.prefill_chunk_size ? std::min(remaining,options.prefill_chunk_size) : remaining;
    }
    void accept_output(bool prefill, int consumed) {
        publication = {};
        if (prefill) {
            result.prefill_tokens += consumed;
            ++result.prefill_chunks;
            if (prefixes) publication = prefixes->publish_if_possible(input, *cache);
            if (options.on_prefill && !options.on_prefill(result.prefill_tokens,int(input.size()))) cancelled.store(true);
            if (cancelled.load()) finish("cancelled");
            else if (result.prefill_tokens==int(input.size())) state=GenerationState::Decode;
        }
        if (!finished() && cancelled.load()) finish("cancelled");
        if (!finished() && state==GenerationState::Decode) {
            int32_t next=model.synchronize_token(select());
            result.ids.push_back(next);
            history.push_back(next);
            if (on_token && !on_token(next)) finish("stop");
            else if (cancelled.load()) finish("cancelled");
            else if (std::find(options.eos_ids.begin(),options.eos_ids.end(),next)!=options.eos_ids.end()) finish("eos");
            else if (int(result.ids.size())==count) finish("length");
        }
    }
    int advance(int budget) {
        if (finished()) return 0;
        if (cancelled.load()) { finish("cancelled"); return 0; }
        initialize();
        bool prefill = state == GenerationState::Prefill;
        std::chrono::steady_clock::time_point start = Clock::now();
        int consumed = 1;
        try {
            const DecoderLogits logits = options.full_logits ? Qwen2Logits::All : Qwen2Logits::Last;
            if (prefill) {
                int offset = result.prefill_tokens, total = int(input.size());
                consumed = std::min(budget, total - offset);
                if (options.prefill_chunk_size) consumed = std::min(consumed, options.prefill_chunk_size);
                int end = offset + consumed;
                Tensor ids = Tensor::from_int_data({1, consumed},
                    std::vector<int32_t>(input.begin() + offset, input.begin() + end));
                DecoderLogits selection = end == total ? logits : Qwen2Logits::None;
                output = cache ? model.forward_paged(ids, *cache, {}, {}, false, selection) :
                    model.forward(ids, {}, {}, offset ? &output.cache : nullptr, false, selection);
            } else if (options.cuda_graph && cache) {
                const int capacity = Qwen2PagedGraph::context_bucket(cache->length() + 1, options.max_context);
                if (paged_graph && !paged_graph->matches(1, capacity)) paged_graph.reset();
                PagedKVBatch batch({cache.get()});
                const std::vector<int32_t> tokens{result.ids.back()};
                if (!paged_graph) {
                    const std::chrono::steady_clock::time_point capture_start = Clock::now();
                    paged_graph = std::make_unique<Qwen2PagedGraph>(dynamic_cast<const Qwen2&>(model), batch, capacity, tokens);
                    result.graph_capture_ms += elapsed(capture_start);
                    result.graph_commands = paged_graph->commands();
                    result.graph_nodes = paged_graph->nodes();
                    result.graph_retained_bytes = paged_graph->retained_bytes();
                    output.logits = paged_graph->logits();
                } else {
                    output.logits = paged_graph->decode(batch, tokens);
                    ++result.graph_replays;
                }
            } else if (options.cuda_graph) {
                if (!graph) {
                    const std::chrono::steady_clock::time_point capture_start = Clock::now();
                    graph = std::make_unique<Qwen2Graph>(dynamic_cast<const Qwen2&>(model), options.max_context, output.cache);
                    result.graph_capture_ms = elapsed(capture_start);
                    result.graph_commands = graph->commands(); result.graph_nodes = graph->nodes();
                    result.graph_retained_bytes = graph->retained_bytes();
                    output = {};
                }
                output.logits = graph->decode(result.ids.back());
                ++result.graph_replays;
            } else {
                Tensor ids = Tensor::from_int_data({1, 1}, {result.ids.back()});
                output = cache ? model.forward_paged(ids, *cache, {}, {}, false, logits) :
                    model.forward(ids, {}, {}, &output.cache, false, logits);
            }
            accept_output(prefill,consumed);
        } catch (...) {
            (prefill ? result.prefill_ms : result.decode_ms) += elapsed(start);
            throw;
        }
        (prefill ? result.prefill_ms : result.decode_ms) += elapsed(start);
        return consumed;
    }
};

GenerationSession::GenerationSession(const DecoderModel& model, const std::vector<int32_t>& input,
                                     const GenerationOptions& options, const std::function<bool(int32_t)>& callback,
                                     std::shared_ptr<PagedKVPool> shared_pool, std::shared_ptr<PrefixCache> prefix_store, PrefixMatch initial_prefix)
    : impl_(std::make_unique<Impl>(model, input, options, callback, std::move(shared_pool), std::move(prefix_store), std::move(initial_prefix))) {}
GenerationSession::~GenerationSession() = default;
GenerationState GenerationSession::state() const { return impl_->state; }
bool GenerationSession::finished() const { return impl_->finished(); }
int GenerationSession::pending_input_tokens() const { impl_->check_owner(); return impl_->pending_input_tokens(); }
int GenerationSession::cache_length() const {
    impl_->check_owner();
    return impl_->cache ? impl_->cache->length() : impl_->graph ? impl_->graph->length() : impl_->output.cache.length;
}
const GenerationResult& GenerationSession::result() const { return impl_->result; }
void GenerationSession::request_cancel() noexcept { impl_->cancelled.store(true); }
void GenerationSession::cancel() {
    impl_->check_owner();
    request_cancel();
    if (!impl_->busy && !finished()) impl_->finish("cancelled");
}
int GenerationSession::step(int token_budget) {
    GenerationSession::Impl& s = *impl_;
    DeviceScope owner(s.model.backend(), s.device);
    s.check_owner();
    require(token_budget > 0, "step token budget must be positive");
    require(!s.busy, "cannot reenter the same session");
    if (finished()) return 0;
    if (s.cancelled.load()) { s.finish("cancelled"); return 0; }
    s.busy = true;
    try {
        // Paused sessions retain payloads, not process-wide allocation ceilings.
        // Re-admit against actual resident bytes on every step. The full-request
        // bound deliberately overcounts resident cache storage; it cannot promise
        // that a later step will fit if other requests consume the remaining RAM.
        int chunk = s.state == GenerationState::Decode ? 1 :
            std::min(token_budget, s.options.prefill_chunk_size ? s.options.prefill_chunk_size : token_budget);
        DecoderMemoryPlan plan = plan_decoder_generation(s.model.config(), s.model.backend(), s.model.dtype(), s.input.size(), size_t(s.count),
            s.options.do_sample, s.model.tiled_attention_enabled(), s.model.packed_projections(), s.options.paged_cache,
            chunk, s.options.full_logits, s.model.quantization_workspace_bytes());
        if (s.paged_graph && s.state == GenerationState::Decode &&
            !s.paged_graph->matches(1, Qwen2PagedGraph::context_bucket(s.cache->length() + 1, s.options.max_context)))
            s.paged_graph.reset();
        if (!s.graph && !s.paged_graph) detail::add_graph_memory(plan, s.model.config(), s.model.dtype(), s.options, s.model.quantization_workspace_bytes());
        if (s.prefixes) {
            size_t bytes=s.prefixes->publication_bytes();
            plan.host_additional_bytes=checked_memory_add(plan.host_additional_bytes,bytes);
            if (s.model.backend()==Backend::CPU) plan.additional_bytes=checked_memory_add(plan.additional_bytes,bytes);
        }
        plan.host_additional_bytes = checked_memory_add(plan.host_additional_bytes,
            staging_allowance(s.model.backend(), s.options.asynchronous, s.options.pinned_staging));
        DecoderMemoryAdmission admission(s.model, s.options.memory, plan);
        ExecutionStream stream = inference_stream(s.model.backend());
        FailedTransferGuard transfers{stream, &s.model};
        StreamScope execution(stream, s.options.asynchronous, s.options.pinned_staging);
        admission.enable_pools(s.options.reuse_memory);
        int consumed = s.advance(token_budget);
        s.model.synchronize();
        stream.synchronize();
        MemoryReport report = admission.report();
        merge_memory(s.result.memory, report);
        const std::vector<DeviceMemoryReport> reports = admission.device_reports();
        if (s.result.device_memory.empty()) s.result.device_memory = reports;
        else for (size_t i = 0; i < reports.size(); ++i) merge_memory(s.result.device_memory[i].memory, reports[i].memory);
        AllocationPoolStats stats = admission.allocation_stats();
        AllocationPoolStats& allocations = s.result.allocations;
        allocations.requests += stats.requests;
        allocations.allocations += stats.allocations;
        allocations.reuses += stats.reuses;
        allocations.evictions += stats.evictions;
        allocations.allocated_bytes += stats.allocated_bytes;
        allocations.peak_reserved_bytes = std::max(allocations.peak_reserved_bytes, stats.peak_reserved_bytes);
        allocations.peak_cached_bytes = std::max(allocations.peak_cached_bytes, stats.peak_cached_bytes);
        s.busy = false;
        return consumed;
    } catch (...) {
        s.finish("error", true);
        s.busy = false;
        throw;
    }
}

GenerationBatchResult step_generation_batch(const std::vector<GenerationSession*>& sessions, const GenerationBatchOptions& options) {
    const std::chrono::steady_clock::time_point started=Clock::now();
    require(options.token_budget>0, "batch token budget must be positive");
    require(sessions.size()<=size_t(std::numeric_limits<int>::max()), "request batch exceeds Int32");
    MemoryReservation descriptors(Backend::CPU,checked_memory_mul(sessions.size(),256));
    GenerationBatchResult result;
    result.consumed.resize(sessions.size()); result.errors.resize(sessions.size()); result.publications.resize(sessions.size());
    std::vector<GenerationSession::Impl*> all, active;
    std::vector<size_t> indices;
    for (GenerationSession* session : sessions) {
        require(session!=nullptr, "null generation session");
        GenerationSession::Impl* s=session->impl_.get(); s->check_owner();
        require(!s->busy, "cannot reenter a batched session");
        require(std::find(all.begin(),all.end(),s)==all.end(), "duplicate generation session");
        all.push_back(s);
        if (s->finished() || s->cancelled.load()) continue;
        require(s->cache_pool!=nullptr, "batched sessions require a resident paged pool");
        if (!active.empty()) require(&s->model==&active.front()->model && s->cache_pool==active.front()->cache_pool,
                                     "batched sessions must share one model and pool handle");
        indices.push_back(all.size()-1); active.push_back(s);
    }
    for (GenerationSession::Impl* s : all) if (!s->finished() && s->cancelled.load()) s->finish("cancelled");
    if (active.empty()) { result.elapsed_ms=elapsed(started); return result; }
    require(size_t(options.token_budget)>=active.size(), "batch token budget cannot fit every active row");
    int query=options.token_budget/int(active.size());
    MemoryBudget budget=options.memory;
    for (GenerationSession::Impl* s : active) {
        query=std::min(query,s->pending_input_tokens());
        if (s->options.memory.limit_bytes && (!budget.limit_bytes || s->options.memory.limit_bytes<budget.limit_bytes))
            budget.limit_bytes=s->options.memory.limit_bytes;
        budget.reserve_bytes=std::max(budget.reserve_bytes,s->options.memory.reserve_bytes);
    }
    // A concurrent cancellation after membership selection is applied at the
    // committed step boundary; it must not change the shape of the group.
    if (!query) {
        for (GenerationSession::Impl* s : active) if (s->cancelled.load()) s->finish("cancelled");
        result.elapsed_ms=elapsed(started); return result;
    }
    const DecoderModel& model=active.front()->model;
    size_t previous=0, leases=0;
    bool sampled=false, pinned_staging=options.pinned_staging;
    bool decode_only = true;
    Qwen2Logits selection=Qwen2Logits::None;
    for (GenerationSession::Impl* s : active) {
        decode_only &= s->state == GenerationState::Decode;
        pinned_staging &= s->options.pinned_staging;
        size_t length=s->cache ? size_t(s->cache->length()) : 0;
        previous=std::max(previous,length);
        bool ready=s->state==GenerationState::Decode || s->result.prefill_tokens+query==int(s->input.size());
        if (ready) {
            sampled |= s->options.do_sample;
            if (s->options.full_logits) selection=Qwen2Logits::All;
            else if (selection==Qwen2Logits::None) selection=Qwen2Logits::Last;
        }
        if (!s->cache) leases=checked_memory_add(leases,checked_memory_mul(
            std::min(size_t(s->cache_pool->shape().blocks),(s->input.size()+size_t(s->count)+15)/16),sizeof(CacheBlock)));
        if (s->prefixes) leases=checked_memory_add(leases,s->prefixes->publication_bytes());
    }
    for (GenerationSession::Impl* s : active) s->busy=true;
    bool published=false;
    try {
        QwenBatchMemoryPlan plan=plan_qwen_batch(model.config(),model.backend(),model.dtype(),active.size(),size_t(query),previous+size_t(query),
                                  selection,sampled,model.tiled_attention_enabled(),16,model.quantization_workspace_bytes(selection));
        const Qwen2* graph_model = nullptr;
        int graph_capacity = 0;
        if (options.decode_graph && decode_only) {
            graph_model = dynamic_cast<const Qwen2*>(&model);
            require(graph_model && query == 1 && model.backend() == Backend::CUDA && model.supports_paged_cache(),
                    "Batch decode graph requires the CUDA Qwen paged backend");
            const int limit = options.graph_context_limit > 0 ? options.graph_context_limit : model.config().max_positions;
            graph_capacity = Qwen2PagedGraph::context_bucket(static_cast<int>(previous) + 1, limit);
            std::unique_ptr<Qwen2PagedGraph>& graph = *options.decode_graph;
            if (graph && !graph->matches(int(active.size()), graph_capacity)) graph.reset();
            if (!graph) plan.additional_bytes = checked_memory_add(plan.additional_bytes,
                Qwen2PagedGraph::memory_bound(*graph_model, int(active.size()), graph_capacity));
            plan.host_additional_bytes = checked_memory_add(plan.host_additional_bytes,
                checked_memory_mul(checked_memory_mul(active.size(), size_t(graph_capacity)), 16));
        }
        plan.host_additional_bytes=checked_memory_add(plan.host_additional_bytes,leases);
        if (model.backend()==Backend::CPU) plan.additional_bytes=checked_memory_add(plan.additional_bytes,leases);
        plan.host_additional_bytes=checked_memory_add(plan.host_additional_bytes,
            staging_allowance(model.backend(),options.asynchronous,pinned_staging));
        MemoryAdmission admission(model.backend(),budget,plan.additional_bytes,plan.host_additional_bytes);
        ExecutionStream stream=inference_stream(model.backend());
        FailedTransferGuard transfers{stream};
        StreamScope execution(stream,options.asynchronous,pinned_staging);
        AllocationPool allocations(model.backend(),admission.report().estimated_peak_bytes,options.reuse_memory);
        std::vector<int32_t> tokens; tokens.reserve(active.size()*query);
        std::vector<PagedKVCache*> caches;
        for (GenerationSession::Impl* s : active) {
            s->initialize(); caches.push_back(s->cache.get());
            if (s->state==GenerationState::Decode) tokens.push_back(s->result.ids.back());
            else tokens.insert(tokens.end(),s->input.begin()+s->result.prefill_tokens,s->input.begin()+s->result.prefill_tokens+query);
        }
        const std::chrono::steady_clock::time_point forward_started=Clock::now();
        Qwen2Output output;
        {
            PagedKVBatch batch(caches);
            if (graph_model) {
                std::unique_ptr<Qwen2PagedGraph>& graph = *options.decode_graph;
                try {
                    if (!graph) {
                        std::chrono::steady_clock::time_point started_graph = Clock::now();
                        graph = std::make_unique<Qwen2PagedGraph>(*graph_model, batch, graph_capacity, tokens);
                        result.graph_captures = 1;
                        result.graph_capture_ms = elapsed(started_graph);
                        output.logits = graph->logits();
                    } else {
                        output.logits = graph->decode(batch, tokens);
                        result.graph_replays = 1;
                    }
                    result.graph_commands = graph->commands();
                    result.graph_nodes = graph->nodes();
                    result.graph_retained_bytes = graph->retained_bytes();
                } catch (...) {
                    graph.reset();
                    throw;
                }
            } else {
                output=model.forward_paged_batch(Tensor::from_int_data({int(active.size()),query},tokens),batch,{}, {},false,selection);
            }
        }
        published=true;
        const double forward_ms=elapsed(forward_started);
        result.query_tokens=query; result.input_tokens=int(active.size())*query;
        for (size_t row=0; row<active.size(); ++row) {
            GenerationSession::Impl* s=active[row]; bool prefill=s->state==GenerationState::Prefill;
            s->publication={};
            std::chrono::steady_clock::time_point processing=Clock::now(); result.consumed[indices[row]]=query;
            try {
                s->output={};
                bool ready=!prefill || s->result.prefill_tokens+query==int(s->input.size());
                if (ready && !s->cancelled.load()) {
                    Shape shape = output.logits.shape();
                    const size_t row_elements = output.logits.numel() / shape[0];
                    shape[0] = 1;
                    s->output.logits = contiguous_view(output.logits, shape, row * row_elements);
                }
                s->accept_output(prefill,query);
            } catch (...) {
                result.errors[indices[row]]=std::current_exception();
                s->finish("error",true);
            }
            result.publications[indices[row]]=s->publication;
            s->output={};
            (prefill ? s->result.prefill_ms : s->result.decode_ms) += forward_ms+elapsed(processing);
        }
        stream.synchronize();
        result.memory=admission.report(); result.allocations=allocations.stats();
        for (GenerationSession::Impl* s : active) {
            if (graph_model) {
                s->result.graph_commands = result.graph_commands;
                s->result.graph_nodes = result.graph_nodes;
                s->result.graph_retained_bytes = result.graph_retained_bytes;
                s->result.graph_replays += result.graph_replays;
                s->result.graph_capture_ms += result.graph_capture_ms;
            }
            MemoryReport& memory=s->result.memory; const MemoryReport& report=result.memory;
            memory.budget_bytes=std::max(memory.budget_bytes,report.budget_bytes);
            memory.host_budget_bytes=std::max(memory.host_budget_bytes,report.host_budget_bytes);
            memory.estimated_peak_bytes=std::max(memory.estimated_peak_bytes,report.estimated_peak_bytes);
            memory.measured_peak_bytes=std::max(memory.measured_peak_bytes,report.measured_peak_bytes);
            memory.host_estimated_peak_bytes=std::max(memory.host_estimated_peak_bytes,report.host_estimated_peak_bytes);
            memory.host_measured_peak_bytes=std::max(memory.host_measured_peak_bytes,report.host_measured_peak_bytes);
            s->busy=false;
        }
    } catch (...) {
        for (GenerationSession::Impl* s : active) { if (published && !s->finished()) s->finish("error",true); s->busy=false; }
        throw;
    }
    result.elapsed_ms=elapsed(started);
    return result;
}

GenerationResult generate_greedy(const DecoderModel& model, const std::vector<int32_t>& input,
                                 const GreedyOptions& options, const std::function<bool(int32_t)>& on_token) {
    GenerationOptions all;
    static_cast<GreedyOptions&>(all) = options;
    return generate(model, input, all, on_token);
}

GenerationResult generate(const DecoderModel& model, const std::vector<int32_t>& input,
                          const GenerationOptions& options, const std::function<bool(int32_t)>& on_token) {
    DeviceScope owner(model.backend(), model.input_device());
    int count = validate_request(model.config(), input, options);
    GenerationResult result;
    if (count == 0) return result;
    DecoderMemoryPlan plan = plan_decoder_generation(model.config(), model.backend(), model.dtype(), input.size(), size_t(count),
                                     options.do_sample, model.tiled_attention_enabled(), model.packed_projections(), options.paged_cache,
                                     options.prefill_chunk_size, options.full_logits, model.quantization_workspace_bytes());
    detail::add_graph_memory(plan, model.config(), model.dtype(), options, model.quantization_workspace_bytes());
    plan.host_additional_bytes = checked_memory_add(plan.host_additional_bytes,
        staging_allowance(model.backend(), options.asynchronous, options.pinned_staging));
    DecoderMemoryAdmission admission(model, options.memory, plan);
    ExecutionStream stream = inference_stream(model.backend());
    FailedTransferGuard transfers{stream, &model};
    StreamScope execution(stream, options.asynchronous, options.pinned_staging);
    admission.enable_pools(options.reuse_memory);
    GenerationSession session(model, input, options, on_token);
    while (!session.finished()) session.impl_->advance(std::numeric_limits<int>::max());
    model.synchronize();
    stream.synchronize();
    result = session.result();
    result.memory = admission.report();
    result.device_memory = admission.device_reports();
    result.allocations = admission.allocation_stats();
    return result;
}

GenerationResult generate_text(const DecoderModel& model, const HuggingFaceTokenizer& tokenizer,
                               const std::vector<int32_t>& input, const GreedyOptions& options,
                               const std::vector<std::string>& stops,
                               const std::function<void(const std::string&)>& on_text) {
    GenerationOptions all;
    static_cast<GreedyOptions&>(all) = options;
    return generate_text(model, tokenizer, input, all, stops, on_text);
}

GenerationResult generate_text(const DecoderModel& model, const HuggingFaceTokenizer& tokenizer,
                               const std::vector<int32_t>& input, const GenerationOptions& options,
                               const std::vector<std::string>& stops,
                               const std::function<void(const std::string&)>& on_text) {
    TokenizerStream decoder(tokenizer, true);
    TextStopFilter filter(stops);
    std::string text;
    LocalFunction emit{[&](const std::string& piece) {
        text += piece;
        if (on_text && !piece.empty()) on_text(piece);
    }};
    GenerationResult result = generate(model, input, options, [&](int32_t id) {
        emit(filter.push(decoder.push(id)));
        return !filter.stopped();
    });
    emit(filter.push(decoder.finish()));
    emit(filter.finish());
    result.text = std::move(text);
    if (filter.stopped()) {
        result.finish_reason = "stop";
        result.matched_stop = filter.matched_stop();
    }
    return result;
}

} // namespace elman
