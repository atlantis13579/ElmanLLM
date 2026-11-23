#include "Inference/LLM/Generation.h"
#include "Core/LocalFunction.h"
#include "Inference/LLM/PagedKVCache.h"
#include "Inference/LLM/Qwen2Graph.h"
#include "Inference/LLM/Qwen2PagedGraph.h"
#include "Core/Execution.h"
#include "Distributed/ProcessGroup.h"
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
Json integers(const std::vector<int32_t>& values) {
    Json::Array result;
    for (int value : values) result.emplace_back(value);
    return result;
}
int integer(const std::string& value) {
    size_t used = 0;
    int result = std::stoi(value, &used);
    require(used == value.size(), "invalid integer: " + value);
    return result;
}
double number(const std::string& value) {
    size_t used = 0;
    double result = std::stod(value, &used);
    require(used == value.size() && std::isfinite(result), "invalid number: " + value);
    return result;
}
uint64_t seed_number(const std::string& value) {
    require(!value.empty() && std::all_of(value.begin(), value.end(), [](char c) { return c >= '0' && c <= '9'; }),
            "seed must be an unsigned decimal integer");
    size_t used = 0;
    unsigned long long result = std::stoull(value, &used);
    require(used == value.size(), "invalid seed");
    return result;
}
size_t memory_mib(const std::string& value) {
    require(!value.empty() && std::all_of(value.begin(), value.end(), [](char c) { return c >= '0' && c <= '9'; }),
            "memory MiB must be an unsigned decimal integer");
    unsigned long long result = std::stoull(value);
    require(result <= std::numeric_limits<size_t>::max(), "memory value exceeds size_t");
    return checked_memory_mul(size_t(result), 1024 * 1024);
}
Json memory_json(const MemoryReport& report) {
    return Json::Object{{"budget_bytes", double(report.budget_bytes)},
        {"estimated_peak_bytes", double(report.estimated_peak_bytes)}, {"measured_peak_bytes", double(report.measured_peak_bytes)},
        {"host_budget_bytes", double(report.host_budget_bytes)}, {"host_estimated_peak_bytes", double(report.host_estimated_peak_bytes)},
        {"host_measured_peak_bytes", double(report.host_measured_peak_bytes)}};
}
Json device_memory_json(const std::vector<DeviceMemoryReport>& reports) {
    Json::Array result;
    for (const DeviceMemoryReport& item : reports) result.emplace_back(Json::Object{{"ordinal", item.ordinal}, {"memory", memory_json(item.memory)}});
    return result;
}
void merge_memory(MemoryReport& memory, const MemoryReport& report) {
    memory.budget_bytes = std::max(memory.budget_bytes, report.budget_bytes);
    memory.host_budget_bytes = std::max(memory.host_budget_bytes, report.host_budget_bytes);
    memory.estimated_peak_bytes = std::max(memory.estimated_peak_bytes, report.estimated_peak_bytes);
    memory.measured_peak_bytes = std::max(memory.measured_peak_bytes, report.measured_peak_bytes);
    memory.host_estimated_peak_bytes = std::max(memory.host_estimated_peak_bytes, report.host_estimated_peak_bytes);
    memory.host_measured_peak_bytes = std::max(memory.host_measured_peak_bytes, report.host_measured_peak_bytes);
}
void add_graph_memory(DecoderMemoryPlan& plan, const DecoderConfig& config, DType dtype,
                      const GenerationOptions& options, size_t weight_workspace) {
    if (!options.cuda_graph) return;
    // Single-device eager and recorded forwards request the same fixed-size
    // quantization workspace. Its lease is released at forward exit and reused
    // by the graph, with cross-stream dependencies retained by the pool.
    const size_t shared_workspace = std::min({plan.additional_bytes, plan.weight_workspace_bytes, weight_workspace});
    plan.additional_bytes = checked_memory_add(plan.additional_bytes,
        Qwen2Graph::memory_bound(config, dtype, options.max_context, weight_workspace) - shared_workspace);
    plan.host_additional_bytes = checked_memory_add(plan.host_additional_bytes, 16 * 1024 * 1024);
}
Json allocation_json(const AllocationPoolStats& stats) {
    return Json::Object{{"requests", double(stats.requests)}, {"allocations", double(stats.allocations)},
        {"reuses", double(stats.reuses)}, {"evictions", double(stats.evictions)},
        {"allocated_bytes", double(stats.allocated_bytes)}, {"peak_reserved_bytes", double(stats.peak_reserved_bytes)},
        {"peak_cached_bytes", double(stats.peak_cached_bytes)}};
}
Json staging_json(const HostStagingStats& stats) {
    return Json::Object{{"allocated_bytes", double(stats.allocated_bytes)}, {"pending_bytes", double(stats.pending_bytes)},
        {"cached_bytes", double(stats.cached_bytes)}, {"peak_bytes", double(stats.peak_bytes)},
        {"metadata_bytes", double(stats.metadata_bytes)}, {"buffers", double(stats.buffers)},
        {"allocations", double(stats.allocations)}, {"reuses", double(stats.reuses)}, {"waits", double(stats.waits)},
        {"fallbacks", double(stats.fallbacks)}, {"uploads", double(stats.uploads)}, {"downloads", double(stats.downloads)},
        {"uploaded_bytes", double(stats.uploaded_bytes)}, {"downloaded_bytes", double(stats.downloaded_bytes)}};
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
        if (!s.graph && !s.paged_graph) add_graph_memory(plan, s.model.config(), s.model.dtype(), s.options, s.model.quantization_workspace_bytes());
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
    add_graph_memory(plan, model.config(), model.dtype(), options, model.quantization_workspace_bytes());
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

int generate_command(int argc, char** argv) {
    require(argc >= 3, "generate requires a model directory or model.safetensors path");
    std::string model_path = argv[2], input_path, output_path, backend_name_, dtype_name_;
    GenerationOptions options;
    bool stream = false, fused_pointwise = true, fused_normalization = true, fused_attention_softmax = true, tiled_attention = true, packed_projections = true;
    bool direct_paged_attention = false, absorbed_attention = true;
    bool temperature_set = false, top_k_set = false, top_p_set = false, penalty_set = false, sampling_set = false;
    std::vector<std::string> stops;
    std::vector<int> device_ordinals;
    int simulated_devices = 0;
    bool tensor_parallel = false;
    for (int i = 3; i < argc; ++i) {
        std::string key = argv[i];
        if (key == "--stream") { stream = true; continue; }
        if (key == "--cuda-graph") { options.cuda_graph = true; continue; }
        if (key == "--tensor-parallel") { tensor_parallel = true; continue; }
        if (key == "--mla-expanded") { absorbed_attention = false; continue; }
        if (key == "--sample") { options.do_sample = true; continue; }
        if (key == "--no-memory-reuse") { options.reuse_memory = false; continue; }
        if (key == "--synchronous") { options.asynchronous = false; continue; }
        if (key == "--no-pinned-staging") { options.pinned_staging = false; continue; }
        if (key == "--paged-cache") { options.paged_cache = true; continue; }
        if (key == "--direct-paged-attention") { direct_paged_attention = options.paged_cache = true; continue; }
        if (key == "--full-logits") { options.full_logits = true; continue; }
        if (key == "--no-fusion") { fused_pointwise = fused_normalization = fused_attention_softmax = tiled_attention = packed_projections = false; continue; }
        if (key == "--no-normalization-fusion") { fused_normalization = false; continue; }
        if (key == "--no-attention-fusion") { fused_attention_softmax = tiled_attention = false; continue; }
        if (key == "--no-tiled-attention") { tiled_attention = false; continue; }
        if (key == "--no-packed-projections") { packed_projections = false; continue; }
        require(i + 1 < argc, "missing value for " + key);
        std::string value = argv[++i];
        if (key == "--input") input_path = value;
        else if (key == "--output") output_path = value;
        else if (key == "--backend") backend_name_ = value;
        else if (key == "--dtype") dtype_name_ = value;
        else if (key == "--simulate-devices") simulated_devices = integer(value);
        else if (key == "--devices") {
            require(device_ordinals.empty() && !value.empty(), "--devices requires one nonempty comma-separated list");
            size_t begin = 0;
            do {
                const size_t end = value.find(',', begin);
                const std::string token = value.substr(begin, end == std::string::npos ? end : end - begin);
                require(!token.empty() && std::all_of(token.begin(), token.end(), [](char c) { return c >= '0' && c <= '9'; }),
                    "device ordinals must be nonnegative decimal integers");
                device_ordinals.push_back(integer(token));
                if (end == std::string::npos) break;
                begin = end + 1;
            } while (true);
        }
        else if (key == "--max-new-tokens") options.max_new_tokens = integer(value);
        else if (key == "--max-context") options.max_context = integer(value);
        else if (key == "--prefill-chunk-size") options.prefill_chunk_size = integer(value);
        else if (key == "--memory-limit-mib") options.memory.limit_bytes = memory_mib(value);
        else if (key == "--memory-reserve-mib") options.memory.reserve_bytes = memory_mib(value);
        else if (key == "--stop") stops.push_back(value);
        else if (key == "--temperature") { options.sampling.temperature = number(value); temperature_set = sampling_set = true; }
        else if (key == "--top-k") { options.sampling.top_k = integer(value); top_k_set = sampling_set = true; }
        else if (key == "--top-p") { options.sampling.top_p = number(value); top_p_set = sampling_set = true; }
        else if (key == "--repetition-penalty") { options.sampling.repetition_penalty = number(value); penalty_set = sampling_set = true; }
        else if (key == "--seed") { options.seed = seed_number(value); sampling_set = true; }
        else throw std::invalid_argument("Unknown generation option: " + key);
    }
    require(!input_path.empty(), "generate requires --input JSON");
    const std::filesystem::path supplied_path = std::filesystem::absolute(model_path).lexically_normal();
    require(std::filesystem::exists(supplied_path), "Model path does not exist: " + model_path);
    if (std::filesystem::is_regular_file(supplied_path)) {
        require(supplied_path.filename() == "model.safetensors" || supplied_path.filename() == "model.safetensors.index.json",
            "Use the model directory, model.safetensors or model.safetensors.index.json for a complete checkpoint");
        model_path = supplied_path.parent_path().string();
    }
    require(std::filesystem::is_directory(model_path), "Model path must be a directory or supported checkpoint file");
    require(std::filesystem::exists(std::filesystem::path(model_path) / "config.json"),
        "The model directory must contain config.json alongside its weights and tokenizer files");
    if (backend_name_.empty()) backend_name_ = "cpu";
    require(options.max_new_tokens >= 0 && options.max_context > 0, "invalid output or context limit");
    require(options.prefill_chunk_size >= 0, "prefill_chunk_size must be nonnegative");
    require(options.do_sample || !sampling_set, "sampling options require --sample");
    Backend backend = parse_backend(backend_name_);
    require(backend_available(backend), "backend is unavailable");
    require(device_ordinals.empty() || backend == Backend::CUDA, "--devices requires CUDA");
    require(simulated_devices == 0 || (simulated_devices >= 2 && backend == Backend::CUDA && device_ordinals.size() <= 1),
        "--simulate-devices requires at least two stages on a single CUDA device");
    require(!simulated_devices || (!options.paged_cache && !options.cuda_graph), "simulated device stages require eager contiguous execution");
    require(device_ordinals.size() <= 1 || !options.paged_cache, "layer partitioning requires contiguous caches");
    int ordinal = device_ordinals.empty() ? current_device(backend) : device_ordinals.front();
    std::shared_ptr<ProcessGroup> parallel;
    if (tensor_parallel) {
        require(device_ordinals.empty() && !simulated_devices && !options.paged_cache && !options.cuda_graph,
                "Tensor parallel generation requires eager contiguous caches with one device per rank");
        const ProcessGroupOptions ranks = process_group_environment();
        require(ranks.size > 1, "Launch tensor parallel generation with two or more ranks");
        parallel = std::make_shared<ProcessGroup>(ranks);
        if (backend == Backend::CUDA) ordinal = ranks.local_rank;
    }
    const bool primary = !parallel || parallel->rank() == 0;
    DeviceScope input_device(backend, ordinal);
    DType dtype = parse_dtype(dtype_name_.empty() ? (backend == Backend::CUDA ? "bfloat16" : "float32") : dtype_name_);
    require(dtype == DType::Float32 || dtype == DType::Float16 || dtype == DType::BFloat16, "dtype must be float32, float16 or bfloat16");
    Json requests = Json::read(input_path);
    Json::Array items = requests.is_array() ? requests.array() : Json::Array{requests};
    require(!items.empty(), "request array is empty");
    std::unique_ptr<HuggingFaceTokenizer> tokenizer;
    if (std::any_of(items.begin(), items.end(), [](const Json& item) { return !item.contains("input_ids"); }))
        tokenizer = std::make_unique<HuggingFaceTokenizer>(model_path);
    std::filesystem::path generation_path = std::filesystem::path(model_path) / "generation_config.json";
    if (std::filesystem::exists(generation_path)) {
        Json config = Json::read(generation_path.string());
        if (options.do_sample) {
            if (!temperature_set) options.sampling.temperature = config.number_or("temperature", 1);
            if (!top_k_set) options.sampling.top_k = config.int_or("top_k", 0);
            if (!top_p_set) options.sampling.top_p = config.number_or("top_p", 1);
            if (!penalty_set) options.sampling.repetition_penalty = config.number_or("repetition_penalty", 1);
        }
        if (config.contains("eos_token_id") && !config.at("eos_token_id").is_null()) {
            const Json& eos = config.at("eos_token_id");
            if (eos.is_array()) for (const Json& value : eos.array()) options.eos_ids.push_back(value.integer());
            else options.eos_ids.push_back(eos.integer());
        }
    } else if (tokenizer && tokenizer->eos_id() >= 0) options.eos_ids.push_back(tokenizer->eos_id());
    if (options.do_sample) validate_sampling(options.sampling);
    std::vector<std::vector<int32_t>> inputs;
    std::vector<std::vector<std::string>> request_stops;
    for (const Json& item : items) {
        require(int(item.contains("prompt")) + int(item.contains("messages")) + int(item.contains("input_ids")) == 1,
            "each request needs exactly one of prompt, messages or input_ids");
        if (item.contains("input_ids")) {
            std::vector<int32_t> ids;
            for (const Json& token : item.at("input_ids").array()) ids.push_back(token.integer());
            inputs.push_back(std::move(ids));
            require(!stream && stops.empty() && !item.contains("stop"), "input_ids requests return IDs and do not support text streaming/stops");
        } else inputs.push_back(item.contains("prompt") ? tokenizer->encode(item.at("prompt").string()) :
                                tokenizer->encode_chat(item.at("messages").array()));
        std::vector<std::string> strings = stops;
        if (item.contains("stop")) {
            const Json& value = item.at("stop");
            if (value.is_array()) for (const Json& stop : value.array()) strings.push_back(stop.string());
            else strings.push_back(value.string());
        }
        TextStopFilter validate_stops(strings);
        request_stops.push_back(std::move(strings));
    }
    DecoderConfig model_config = read_decoder_config(model_path);
    require(simulated_devices <= model_config.layers, "simulated stages exceed the number of layers");
    require(!model_config.latent_cache_rank || !options.paged_cache, "DeepSeek generation uses its compressed contiguous cache");
    if (model_config.latent_cache_rank) packed_projections = false;
    options.max_context = std::min(options.max_context, model_config.max_positions);
    require(!options.cuda_graph || (backend == Backend::CUDA && !model_config.latent_cache_rank &&
        device_ordinals.size() <= 1 && tiled_attention && options.max_context >= 2),
        "--cuda-graph requires single-device Qwen, tiled attention and context >= 2");
    std::vector<DeviceLoadPlan> load_plans = parallel
        ? std::vector<DeviceLoadPlan>{plan_qwen_tensor_parallel_load(model_path, backend, dtype, packed_projections, parallel->rank(), parallel->size())}
        : plan_decoder_load(model_path, backend, dtype, packed_projections, device_ordinals);
    const DecoderMemoryPlan& load_plan = load_plans.front().memory;
    // Reject invalid or unaffordable requests before allocating model weights.
    for (const std::vector<int>& input : inputs) {
        int count = validate_request(model_config, input, options);
        DecoderMemoryPlan plan = plan_decoder_generation(model_config, backend, dtype, input.size(), size_t(count), options.do_sample, tiled_attention, packed_projections,
            options.paged_cache, options.prefill_chunk_size, options.full_logits, load_plan.weight_workspace_bytes);
        add_graph_memory(plan, model_config, dtype, options, load_plan.weight_workspace_bytes);
        plan.host_additional_bytes = checked_memory_add(plan.host_additional_bytes,
            staging_allowance(backend, options.asynchronous, options.pinned_staging));
        std::vector<DeviceLoadPlan> preflight_plans = load_plans;
        for (DeviceLoadPlan& local : preflight_plans) {
            local.memory.parameter_bytes = checked_memory_add(local.memory.parameter_bytes,
                checked_memory_mul(plan.additional_bytes, size_t(std::max(1, simulated_devices))));
            const size_t stages = simulated_devices ? size_t(simulated_devices) : device_ordinals.size();
            local.memory.host_additional_bytes = checked_memory_add(plan.host_additional_bytes,
                stages > 1 ? checked_memory_mul(stages, host_staging_memory_bound()) : 0);
        }
        DecoderMemoryAdmission preflight(backend, options.memory, preflight_plans, false);
    }
    std::chrono::steady_clock::time_point start = Clock::now();
    MemoryReport load_memory;
    std::vector<DeviceMemoryReport> load_device_memory;
    std::unique_ptr<DecoderModel> owned_model = [&] {
        DecoderMemoryAdmission admission(backend, options.memory, load_plans, true);
        DecoderLoadOptions settings;
        settings.fused_pointwise = fused_pointwise;
        settings.fused_normalization = fused_normalization;
        settings.fused_attention_softmax = fused_attention_softmax;
        settings.tiled_attention = tiled_attention;
        settings.packed_projections = packed_projections;
        settings.direct_paged_attention = direct_paged_attention;
        settings.absorbed_attention = absorbed_attention;
        settings.devices = device_ordinals;
        settings.simulated_devices = simulated_devices;
        settings.tensor_parallel = parallel;
        std::unique_ptr<DecoderModel> loaded = load_decoder(model_path, backend, dtype, settings);
        load_memory = admission.report();
        load_device_memory = admission.device_reports();
        return loaded;
    }();
    const DecoderModel& model = *owned_model;
    if (parallel) parallel->barrier();
    double load_ms = elapsed(start);
    Json::Array results;
    LocalFunction record_result{[&](const std::string& name, const std::vector<int32_t>& input, const GenerationResult& result) {
        if (!primary || output_path.empty()) return;
        results.emplace_back(Json::Object{{"name", name},
            {"input_ids", integers(input)}, {"generated_ids", integers(result.ids)}, {"text", result.text},
            {"matched_stop", result.matched_stop},
            {"finish_reason", result.finish_reason}, {"prompt_tokens", static_cast<int>(input.size())},
            {"generated_tokens", static_cast<int>(result.ids.size())},
            {"prefill_ms", result.prefill_ms}, {"decode_ms", result.decode_ms}, {"memory", memory_json(result.memory)},
            {"device_memory", device_memory_json(result.device_memory)},
            {"cuda_graph", Json::Object{{"commands", double(result.graph_commands)}, {"nodes", double(result.graph_nodes)},
                {"replays", double(result.graph_replays)}, {"capture_ms", result.graph_capture_ms},
                {"retained_bytes", double(result.graph_retained_bytes)}}},
            {"prefill_tokens", result.prefill_tokens}, {"prefill_chunks", result.prefill_chunks},
            {"allocations", allocation_json(result.allocations)}});
    }};
    for (size_t i = 0; i < items.size(); ++i) {
        std::function<void(const std::string&)> on_text;
        if (stream && primary) on_text = [](const std::string& piece) { std::cout << piece << std::flush; };
        const bool token_input = items[i].contains("input_ids");
        GenerationResult result = token_input ? generate(model, inputs[i], options) :
            generate_text(model, *tokenizer, inputs[i], options, request_stops[i], on_text);
        record_result(items[i].string_or("name", std::to_string(i)), inputs[i], result);
        if (primary) {
            if (!stream) std::cout << (token_input ? integers(result.ids).dump(0) : result.text);
            std::cout << '\n' << std::flush;
        }
    }
    if (primary && !output_path.empty()) Json(Json::Object{{"backend", backend_name(backend)}, {"dtype", dtype_name(dtype)},
        {"strategy", options.do_sample ? "sample" : "greedy"}, {"load_ms", load_ms}, {"max_context", options.max_context},
        {"load_memory", memory_json(load_memory)},
        {"load_device_memory", device_memory_json(load_device_memory)},
        {"execution_stages", model.execution_stages()}, {"simulated_device_placement", model.simulated_device_placement()},
        {"tensor_parallel_size", model.tensor_parallel_size()},
        {"activation_transfers", double(model.activation_transfers())},
        {"memory_reuse", options.reuse_memory},
        {"paged_cache", options.paged_cache},
        {"cuda_graph", options.cuda_graph},
        {"full_logits", options.full_logits}, {"prefill_chunk_size", options.prefill_chunk_size},
        {"asynchronous", backend == Backend::CUDA && options.asynchronous},
        {"pinned_staging", backend == Backend::CUDA && options.asynchronous && options.pinned_staging},
        // These counters cover the reused worker stream across all CLI requests.
        {"stream_host_staging", staging_json(inference_stream(backend).host_staging_stats())},
        {"fused_pointwise", model.fused_pointwise()},
        {"fused_normalization", model.fused_normalization()},
        {"fused_attention_softmax", model.fused_attention_softmax()},
        {"tiled_attention", model.tiled_attention_enabled()},
        {"packed_projections", model.packed_projections()},
        {"direct_paged_attention", model.direct_paged_attention()},
        {"attention_tiling", model.attention_tiling_name()},
        {"sampling", Json::Object{{"temperature", options.sampling.temperature}, {"top_k", options.sampling.top_k},
            {"top_p", options.sampling.top_p}, {"repetition_penalty", options.sampling.repetition_penalty},
            {"seed", std::to_string(options.seed)}, {"rng", "splitmix64-high24"}, {"sorting", "stable"}}},
        {"results", results}}).write(output_path);
    return 0;
}
} // namespace elman
