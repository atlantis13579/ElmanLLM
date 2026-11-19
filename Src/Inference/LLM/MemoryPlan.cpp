#include "Inference/LLM/MemoryPlan.h"
#include "Inference/LLM/PagedKVCache.h"
#include "IO/Safetensors.h"
#include "IO/InferenceWeights.h"
#include "Models/LLM/DecoderFactory.h"
#include "Core/TensorDispatch.h"
#include <algorithm>
#include <filesystem>
#include <initializer_list>
#include <limits>

namespace elman {
namespace {
size_t mul(std::initializer_list<size_t> values) {
    size_t result = 1;
    for (auto value : values) result = checked_memory_mul(result, value);
    return result;
}
size_t sum(std::initializer_list<size_t> values) {
    size_t result = 0;
    for (auto value : values) result = checked_memory_add(result, value);
    return result;
}
void require(bool valid, const char* message) {
    if (!valid) throw std::invalid_argument(std::string("Qwen memory plan: ") + message);
}
void validate(const DecoderConfig& c) {
    require(c.vocabulary > 0 && c.hidden > 0 && c.intermediate > 0 && c.layers > 0 &&
            c.heads > 0 && c.kv_heads > 0 && c.max_positions > 0, "dimensions must be positive");
    if (c.latent_cache_rank) {
        require(c.latent_cache_rank > 0 && c.latent_rotary_dimension > 0 && c.latent_rotary_dimension % 2 == 0 &&
                c.attention_query_dimension > 0 && c.attention_value_dimension > 0 && c.execution_intermediate > 0 &&
                c.routed_expert_count >= 0 && c.active_experts >= 0, "invalid latent attention dimensions");
        return;
    }
    require(c.hidden % c.heads == 0 && c.heads % c.kv_heads == 0 && c.head_dim() % 2 == 0,
            "invalid attention dimensions");
}
size_t elements(std::initializer_list<size_t> values) {
    auto count = mul(values);
    require(count <= size_t(std::numeric_limits<int>::max()), "tensor exceeds the Int32 element limit");
    return count;
}
size_t parameter_bytes(const DecoderConfig& c, Backend backend, DType dtype, bool packed) {
    require(!c.latent_cache_rank, "latent model parameters require a checkpoint catalog");
    size_t h = c.hidden, k = size_t(c.kv_heads) * c.head_dim(), i = c.intermediate;
    auto bytes = [&](std::initializer_list<size_t> shape) {
        return memory_allocation_bytes(backend, dtype_storage_bytes(dtype, elements(shape)));
    };
    auto layer = sum({mul({2, bytes({h, h})}), mul({2, bytes({k, h})}), mul({3, bytes({h, i})}),
                      mul({3, bytes({h})}), mul({2, bytes({k})})});
    if (packed) layer = sum({bytes({h, h}), bytes({h + 2*k, h}), bytes({2*i, h}), bytes({h, i}),
                             mul({2, bytes({h})}), bytes({h + 2*k})});
    return sum({mul({c.tied_embeddings ? size_t(1) : size_t(2), bytes({size_t(c.vocabulary), h})}),
                bytes({h}), mul({size_t(c.layers), layer})});
}
}

size_t qwen_host_workspace(const DecoderConfig& c, size_t batch, size_t sequence, size_t total, bool full_logits, bool tiled_attention, bool head_tiling) {
    validate(c);
    require(batch && sequence && sequence <= total && total <= size_t(c.max_positions), "invalid sequence dimensions");
    size_t dim = c.latent_cache_rank ? c.latent_rotary_dimension : c.head_dim();
    if (tiled_attention && head_tiling) elements({sequence, total});
    else elements({batch, size_t(c.heads), tiled_attention ? std::min(sequence, size_t(kQwenAttentionQueryTile)) : sequence, total});
    elements({batch, total, size_t(c.hidden)});
    elements({batch, sequence, size_t(c.intermediate)});
    elements({batch, full_logits ? sequence : size_t(1), size_t(c.vocabulary)});
    // IDs, masks, positions, causal/rotary arrays, and their upload/download
    // staging can coexist. The reservation includes conservative copy bounds.
    return sum({mul({8, batch, tiled_attention ? size_t(1) : sequence, total}), mul({16, batch, sequence, dim}),
                mul({32, batch, sequence}), mul({16, batch, total})});
}

DecoderMemoryPlan plan_decoder_generation(
    const DecoderConfig& c, Backend backend, DType dtype, size_t prompt, size_t count,
    bool sampled, bool tiled_attention, bool packed_projections, bool paged_cache,
    int prefill_chunk_size, bool full_logits, size_t weight_workspace_bytes) {
    validate(c);
    require(is_compute_float(dtype), "unsupported compute dtype");
    require(prefill_chunk_size >= 0, "prefill chunk size must be nonnegative");
    require(prompt && prompt <= size_t(c.max_positions) && count <= size_t(c.max_positions) - prompt,
            "request exceeds context length");
    DecoderMemoryPlan result;
    if (c.latent_cache_rank) {
        require(!paged_cache, "latent cache cannot use the Qwen paged layout");
        result.weight_workspace_bytes = weight_workspace_bytes;
        if (!count) return result;
        const size_t total = checked_memory_add(prompt, count), d = dtype_storage_bytes(dtype, 1);
        const size_t rows = prefill_chunk_size ? std::min(prompt, size_t(prefill_chunk_size)) : prompt;
        const size_t heads = c.heads, hidden = c.hidden, vocabulary = c.vocabulary;
        const size_t rank = c.latent_cache_rank, rotary = c.latent_rotary_dimension;
        const size_t query_dim = c.attention_query_dimension, value_dim = c.attention_value_dimension;
        result.kv_cache_bytes = mul({size_t(c.layers), total, rank + rotary, d});
        const size_t score = mul({heads, std::min(rows, size_t(128)), total, 12 * d + 8});
        const size_t projections = sum({mul({12 * d + 8, rows, hidden}),
            mul({8 * d, rows, heads, query_dim + value_dim + rank}),
            mul({4 * d, heads, query_dim + value_dim, rank}),
            mul({4 * d, heads, total, query_dim + value_dim})});
        const size_t experts = sum({mul({8 * d + 8, rows, size_t(c.execution_intermediate)}),
            mul({d + 12, rows, size_t(std::max(1, c.active_experts)), hidden}),
            mul({32, rows, size_t(c.routed_expert_count)})});
        const size_t logits = mul({d + 4, full_logits ? rows : size_t(1), vocabulary});
        const size_t selection = sampled ? sum({mul({backend == Backend::CUDA ? size_t(96) : size_t(48), vocabulary}),
            size_t(16 * 1024 * 1024)}) : mul({8, vocabulary});
        result.host_additional_bytes = sum({mul({32, total}), mul({32 + 16 * rotary, rows}),
            mul({24, rows, size_t(std::max(1, c.active_experts))}), mul({64, size_t(c.routed_expert_count)}),
            mul({8, vocabulary}), size_t(1024 * 1024)});
        result.prefill_bytes = sum({mul({2, result.kv_cache_bytes}), score, projections, experts, logits,
            weight_workspace_bytes, size_t(2 * 1024 * 1024)});
        result.decode_bytes = count > 1 ? result.prefill_bytes : 0;
        result.selection_bytes = sum({result.kv_cache_bytes, logits, selection});
        result.additional_bytes = std::max(result.prefill_bytes, result.selection_bytes);
        if (backend == Backend::CPU) result.additional_bytes = checked_memory_add(result.additional_bytes, result.host_additional_bytes);
        return result;
    }
    result.parameter_bytes = parameter_bytes(c, backend, dtype, packed_projections);
    result.weight_workspace_bytes = weight_workspace_bytes;
    if (!count) return result;
    size_t total = checked_memory_add(prompt, count), d = dtype_storage_bytes(dtype, 1);
    size_t chunk = prefill_chunk_size ? std::min(prompt, size_t(prefill_chunk_size)) : prompt;
    size_t logit_rows = full_logits ? chunk : 1;
    size_t h = c.hidden, v = c.vocabulary, heads = c.heads, dim = c.head_dim();
    auto cache = [&](size_t t) { return mul({2, size_t(c.layers), t, size_t(c.kv_heads), dim, d}); };
    auto forward = [&](size_t s, size_t t) {
        qwen_host_workspace(c, 1, s, t, full_logits, tiled_attention, backend == Backend::CPU);
        // Include full-expression temporaries, GQA materialization, residuals,
        // Float32 RMS/softmax paths, MLP activations and full-sequence logits.
        // Padding/metadata-sized native tensors are covered by the fixed margin.
        size_t attention = tiled_attention ? sum({backend == Backend::CPU ? mul({d, s, t}) : mul({d, heads, std::min(s, size_t(kQwenAttentionQueryTile)), t}),
            mul({4, t}), mul({3, heads, sizeof(void*)})}) : sum({mul({3 * d + 8, heads, s, t}), mul({d, t, h})});
        return sum({mul({4, s}), mul({2, s, dim, d}), mul({tiled_attention ? size_t(1) : s, t, d}),
            attention, mul({12 * d + 8, s, h}),
            mul({4, d, s, size_t(c.intermediate)}),
            mul({d + 4, full_logits ? s : size_t(1), v}), size_t(64 * 1024)});
    };
    result.kv_cache_bytes = cache(total);
    // An immutable previous cache coexists with the next chunk's result.
    result.prefill_bytes = sum({mul({chunk < prompt ? size_t(2) : size_t(1), cache(prompt)}), forward(chunk, prompt)});
    // Assignment keeps the previous output alive throughout the next forward.
    result.decode_bytes = count > 1 ? sum({mul({2, cache(total)}), mul({4, logit_rows, v}), forward(1, total)}) : 0;
    size_t sampling = sampled ? sum({mul({backend == Backend::CUDA ? size_t(96) : size_t(48), v}),
        backend == Backend::CUDA ? size_t(16 * 1024 * 1024) : size_t(1024 * 1024)}) : mul({4, v});
    result.selection_bytes = sum({cache(total), mul({4, logit_rows, v}), sampling, mul({8, total})});
    result.host_additional_bytes = std::max({qwen_host_workspace(c, 1, chunk, prompt, full_logits, tiled_attention, backend == Backend::CPU),
        qwen_host_workspace(c, 1, 1, total, full_logits, tiled_attention, backend == Backend::CPU), mul({8, v})});
    // Request history/results and input/selection transfer staging.
    result.host_additional_bytes = sum({result.host_additional_bytes, mul({32, total}), size_t(64 * 1024)});
    result.additional_bytes = std::max({result.prefill_bytes, result.decode_bytes, result.selection_bytes});
    if (paged_cache) {
        constexpr int page = 16;
        PagedKVShape shape{c.layers, c.kv_heads, c.head_dim(), page, int((total + page - 1) / page)};
        // Retain the contiguous bound as a conservative staging allowance and
        // add the resident physical pool, per-layer transposes and slot maps.
        size_t staging = sum({mul({4, total, size_t(c.kv_heads), dim, d}), mul({16, total}), size_t(64 * 1024)});
        result.additional_bytes = sum({result.additional_bytes, PagedKVPool::payload_bytes(shape, backend, dtype), staging});
        result.host_additional_bytes = sum({result.host_additional_bytes, PagedKVPool::metadata_bytes(shape),
            mul({size_t(shape.blocks), sizeof(CacheBlock)}), mul({32, total}), size_t(c.layers)});
    }
    result.additional_bytes = checked_memory_add(result.additional_bytes, weight_workspace_bytes);
    if (backend == Backend::CPU) result.additional_bytes = sum({result.additional_bytes, result.host_additional_bytes});
    else result.additional_bytes = sum({result.additional_bytes, size_t(1024 * 1024)}); // Lazy cuBLAS workspace.
    return result;
}

QwenBatchMemoryPlan plan_qwen_batch(const DecoderConfig& c, Backend backend, DType dtype,
                                    size_t batch, size_t query, size_t total, Qwen2Logits logits,
                                    bool sampled, bool tiled_attention, int cache_block_size, size_t weight_workspace_bytes) {
    validate(c);
    require(!c.latent_cache_rank, "latent decoder does not use the Qwen paged batch layout");
    require(is_compute_float(dtype), "unsupported compute dtype");
    require(cache_block_size > 0, "cache block size must be positive");
    require(logits==Qwen2Logits::All || logits==Qwen2Logits::Last || logits==Qwen2Logits::None, "invalid logits selection");
    const bool full=logits==Qwen2Logits::All, selected=logits!=Qwen2Logits::None;
    const size_t host_model=qwen_host_workspace(c,batch,query,total,full,tiled_attention,backend==Backend::CPU);
    size_t d=dtype_storage_bytes(dtype,1), h=c.hidden, v=c.vocabulary, heads=c.heads, dim=c.head_dim();
    size_t rows=mul({batch,query}), logit_rows=selected ? (full ? query : 1) : 0;
    size_t attention=tiled_attention ? sum({backend==Backend::CPU ? mul({d,query,total}) :
        mul({d,batch,heads,std::min(query,size_t(kQwenAttentionQueryTile)),total}),
        mul({4,batch,total}),mul({3,batch,heads,sizeof(void*)})}) :
        sum({mul({3*d+8,batch,heads,query,total}),mul({d,batch,total,h})});
    // Gather/transposes and zero-select temporaries coexist with the current
    // layer's K/V. No full-model diagnostic cache is retained by this API.
    size_t paging=sum({mul({8,d,batch,total,size_t(c.kv_heads),dim}),mul({16,batch,total+query}),
                       mul({batch,total}),size_t(64*1024)});
    const size_t tail_tokens=mul({batch,std::min(total-query,size_t(cache_block_size-1))});
    // Two persistent slot maps and one layer's indexed-copy temporary. K and V
    // are copied sequentially; no full-model copy is retained by an append.
    size_t tail_payload=tail_tokens ? sum({mul({2,memory_allocation_bytes(backend,mul({4,tail_tokens}))}),
        memory_allocation_bytes(backend,mul({d,tail_tokens,size_t(c.kv_heads),dim}))}) : 0;
    // Retired leases, host vectors and validation/transfer staging coexist.
    size_t tail_host=tail_tokens ? sum({mul({128,batch}),mul({64,tail_tokens})}) : 0;
    QwenBatchMemoryPlan result;
    result.forward_bytes=sum({weight_workspace_bytes,mul({4,rows}),mul({2,rows,dim,d}),
        mul({batch,tiled_attention ? size_t(1) : query,total,d}),attention,
        mul({12*d+8,rows,h}),mul({4,d,rows,size_t(c.intermediate)}),
        mul({d+4,batch,logit_rows,v}),paging,tail_payload,size_t(64*1024)});
    size_t sampling=sampled ? sum({mul({backend==Backend::CUDA ? size_t(96) : size_t(48),v}),
        backend==Backend::CUDA ? size_t(16*1024*1024) : size_t(1024*1024)}) : mul({4,v});
    result.selection_bytes=selected ? sum({mul({4,batch,logit_rows,v}),mul({4,logit_rows+1,v}),sampling,mul({16,total})}) : 0;
    // Group descriptors, host masks/maps, Int64 conversion, input staging and
    // one row's greedy download can overlap the model workspace reservation.
    result.host_additional_bytes=sum({host_model,mul({48,batch,total+query}),mul({256,batch}),
        size_t(c.layers),selected ? mul({8,v}) : 0,tail_host,size_t(64*1024)});
    result.additional_bytes=std::max(result.forward_bytes,result.selection_bytes);
    result.additional_bytes=sum({result.additional_bytes,backend==Backend::CPU ? result.host_additional_bytes : size_t(1024*1024)});
    return result;
}

DecoderMemoryPlan plan_decoder_weight_load(const std::string& directory, Backend backend, DType dtype,
                                           bool packed_projections) {
    require(is_compute_float(dtype), "unsupported compute dtype");
    InferenceWeights catalog(directory);
    const auto decoder = read_decoder_config(directory);
    if (decoder.latent_cache_rank) packed_projections = false;
    auto routing_weight = [&](const std::string& name) {
        return decoder.latent_cache_rank && (name.find(".mlp.gate.weight") != std::string::npos ||
            name.find(".mlp.gate.e_score_correction_bias") != std::string::npos);
    };
    if (catalog.has_quantization()) {
        DecoderMemoryPlan result;
        result.parameter_bytes = catalog.resident_bytes(backend, dtype);
        const auto& c = decoder;
        for (const auto& entry : catalog.tensors()) if (routing_weight(entry.first)) {
            const auto n = shape_element_count(entry.second.shape);
            const auto narrow = memory_allocation_bytes(backend, dtype_storage_bytes(dtype, n));
            const auto wide = memory_allocation_bytes(backend, dtype_storage_bytes(DType::Float32, n));
            if (wide > narrow) result.parameter_bytes = checked_memory_add(result.parameter_bytes, wide - narrow);
        }
        size_t workspace = catalog.largest_quantized_weight();
        if (packed_projections) {
            workspace = std::max(workspace, elements({2, size_t(c.intermediate), size_t(c.hidden)}));
            // Account for changed allocation alignment after row concatenation.
            result.parameter_bytes = checked_memory_add(result.parameter_bytes, 1024 * 1024);
        }
        result.weight_workspace_bytes = memory_allocation_bytes(backend, dtype_storage_bytes(dtype, workspace));
        // Loading validates metadata and may hold host payload copies. Packed
        // projections coexist with source partitions and intermediate concats.
        const auto largest = memory_allocation_bytes(backend, catalog.largest_payload());
        result.host_additional_bytes = mul({4, catalog.largest_payload()});
        result.additional_bytes = sum({result.parameter_bytes, mul({4, largest})});
        if (backend == Backend::CPU) result.additional_bytes = checked_memory_add(result.additional_bytes, result.host_additional_bytes);
        return result;
    }
    SafetensorsWeights weights(directory);
    DecoderMemoryPlan result;
    size_t conversion = 0;
    for (const auto& entry : weights.tensors()) {
        const auto& info = entry.second;
        require(is_compute_float(info.dtype), "unsupported checkpoint dtype");
        const auto target_dtype = routing_weight(entry.first) ? DType::Float32 : dtype;
        auto bytes = memory_allocation_bytes(backend, dtype_storage_bytes(target_dtype, shape_element_count(info.shape)));
        result.parameter_bytes = sum({result.parameter_bytes, bytes});
        result.host_additional_bytes = std::max(result.host_additional_bytes, size_t(info.bytes));
        if (info.dtype != target_dtype) conversion = std::max(conversion, memory_allocation_bytes(backend, size_t(info.bytes)));
    }
    size_t packing = 0;
    if (packed_projections) {
        const auto c = Qwen2Config::from_json(Json::read((std::filesystem::path(directory) / "config.json").string()));
        size_t h = c.hidden, k = size_t(c.kv_heads) * c.head_dim(), i = c.intermediate;
        auto bytes = [&](std::initializer_list<size_t> shape) {
            return memory_allocation_bytes(backend, dtype_storage_bytes(dtype, elements(shape)));
        };
        // Source partitions and concat's QK intermediate can coexist with the
        // final packed tensor. Retain a separate bound for dtype conversion.
        packing = std::max({sum({bytes({h, h}), mul({2, bytes({k, h})}), bytes({h + k, h}), bytes({h}), mul({2, bytes({k})})}),
                            sum({bytes({h}), mul({2, bytes({k})}), bytes({h + k})}), mul({2, bytes({i, h})})});
        result.parameter_bytes = parameter_bytes(c, backend, dtype, true);
    }
    result.additional_bytes = sum({result.parameter_bytes, conversion, packing});
    if (backend == Backend::CPU) result.additional_bytes = sum({result.additional_bytes, result.host_additional_bytes});
    return result;
}

DeviceLoadPlan plan_qwen_tensor_parallel_load(const std::string& directory, Backend backend, DType dtype,
    bool packed_projections, int rank, int ranks) {
    require(is_compute_float(dtype), "unsupported tensor parallel compute dtype");
    const auto model = Qwen2Config::from_json(Json::read((std::filesystem::path(directory) / "config.json").string()));
    const auto heads = partition_attention_heads(model.heads, model.kv_heads, rank, ranks);
    require(model.intermediate % ranks == 0, "tensor parallel ranks must divide the MLP width");
    InferenceWeights weights(directory);
    require(!weights.has_quantization(), "tensor parallel Qwen2 requires dense weights");
    const int query = heads.query_heads * model.head_dim(), kv = heads.kv_heads * model.head_dim();
    const int intermediate = model.intermediate / ranks;
    DecoderMemoryPlan result;
    size_t conversion = 0;
    for (const auto& [name, info] : weights.tensors()) {
        require(is_compute_float(info.dtype), "unsupported tensor parallel weight dtype");
        auto shape = info.shape;
        const auto has = [&](const char* component) { return name.find(component) != std::string::npos; };
        if (has(".q_proj.")) shape.at(0) = query;
        else if (has(".k_proj.") || has(".v_proj.")) shape.at(0) = kv;
        else if (has(".o_proj.")) shape.at(1) = query;
        else if (has(".gate_proj.") || has(".up_proj.")) shape.at(0) = intermediate;
        else if (has(".down_proj.")) shape.at(1) = intermediate;
        const size_t count = shape_element_count(shape);
        const size_t bytes = dtype_storage_bytes(info.dtype, count);
        result.parameter_bytes = sum({result.parameter_bytes, memory_allocation_bytes(backend, dtype_storage_bytes(dtype, count))});
        result.host_additional_bytes = std::max(result.host_additional_bytes, bytes);
        if (info.dtype != dtype) conversion = std::max(conversion, memory_allocation_bytes(backend, bytes));
    }
    size_t packing = 0;
    if (packed_projections) {
        auto bytes = [&](size_t rows, size_t cols) {
            return memory_allocation_bytes(backend, dtype_storage_bytes(dtype, elements({rows, cols})));
        };
        const size_t width = model.hidden, qkv = size_t(query) + 2 * kv;
        packing = std::max(sum({bytes(qkv, width), bytes(qkv, width), bytes(qkv, 1), bytes(qkv, 1)}),
                           bytes(2 * size_t(intermediate), width));
    }
    result.additional_bytes = sum({result.parameter_bytes, conversion, packing});
    if (backend == Backend::CPU) result.additional_bytes = sum({result.additional_bytes, result.host_additional_bytes});
    return {current_device(backend), result};
}

std::vector<DeviceLoadPlan> plan_decoder_load(const std::string& directory, Backend backend, DType dtype,
    bool packed_projections, const std::vector<int>& devices) {
    const auto config = read_decoder_config(directory);
    DecoderPlacement placement(backend, config.layers, devices);
    const auto total = plan_decoder_weight_load(directory, backend, dtype, packed_projections);
    if (!placement.distributed()) return {{placement.input_device(), total}};
    InferenceWeights catalog(directory);
    std::map<int, size_t> residents;
    for (const auto& item : catalog.tensors()) {
        int ordinal = placement.output_device();
        if (item.first == "model.embed_tokens.weight") ordinal = placement.input_device();
        else if (item.first.compare(0, 13, "model.layers.") == 0) {
            const auto end = item.first.find('.', 13);
            require(end != std::string::npos && end > 13, "invalid layer weight name");
            const auto index = item.first.substr(13, end - 13);
            require(std::all_of(index.begin(), index.end(), [](char c) { return c >= '0' && c <= '9'; }), "invalid layer index");
            ordinal = placement.layer_device(std::stoi(index));
        }
        const bool router = config.latent_cache_rank && (item.first.find(".mlp.gate.weight") != std::string::npos ||
            item.first.find(".mlp.gate.e_score_correction_bias") != std::string::npos);
        residents[ordinal] = checked_memory_add(residents[ordinal], catalog.resident_bytes(item.first, backend, router ? DType::Float32 : dtype));
    }
    if (config.tied_embeddings)
        residents[placement.output_device()] = checked_memory_add(residents[placement.output_device()],
            catalog.resident_bytes("model.embed_tokens.weight", backend, dtype));
    std::vector<DeviceLoadPlan> result;
    for (int ordinal : placement.ordinals()) {
        auto local = total;
        // Packing changes allocation alignment; retain a conservative allowance.
        local.parameter_bytes = checked_memory_add(residents[ordinal], packed_projections ? 1024 * 1024 : 0);
        local.additional_bytes = checked_memory_add(local.parameter_bytes, total.additional_bytes - total.parameter_bytes);
        result.push_back({ordinal, local});
    }
    return result;
}

DecoderMemoryAdmission::DecoderMemoryAdmission(Backend backend, const MemoryBudget& budget,
    const std::vector<DeviceLoadPlan>& plans, bool loading) : backend_(backend) {
    require(!plans.empty(), "empty device memory plan");
    for (const auto& plan : plans) {
        DeviceScope owner(backend, plan.ordinal);
        ordinals_.push_back(plan.ordinal);
        admissions_.push_back(std::make_unique<MemoryAdmission>(backend, budget,
            loading ? plan.memory.additional_bytes : plan.memory.parameter_bytes,
            plan.memory.host_additional_bytes));
    }
}
DecoderMemoryAdmission::DecoderMemoryAdmission(const DecoderModel& model, const MemoryBudget& budget,
    const DecoderMemoryPlan& forward) : backend_(model.backend()), ordinals_(model.device_ordinals()) {
    const size_t staging = backend_ == Backend::CUDA && model.execution_stages() > 1 ?
        checked_memory_mul(size_t(model.execution_stages()), host_staging_memory_bound()) : 0;
    // Simulated queues share one physical pool. Each queue can retain its
    // submitted forward scratch until cross-queue completion events retire.
    const size_t additional = checked_memory_mul(forward.additional_bytes,
        model.simulated_device_placement() ? size_t(model.execution_stages()) : size_t(1));
    for (int ordinal : ordinals_) {
        DeviceScope owner(backend_, ordinal);
        admissions_.push_back(std::make_unique<MemoryAdmission>(backend_, budget, additional,
            checked_memory_add(forward.host_additional_bytes, staging)));
    }
}
MemoryReport DecoderMemoryAdmission::report() const { return admissions_.front()->report(); }
std::vector<DeviceMemoryReport> DecoderMemoryAdmission::device_reports() const {
    std::vector<DeviceMemoryReport> result;
    for (size_t i = 0; i < ordinals_.size(); ++i) result.push_back({ordinals_[i], admissions_[i]->report()});
    return result;
}
void DecoderMemoryAdmission::enable_pools(bool reuse) {
    require(pools_.empty(), "device allocation pools already enabled");
    for (size_t i = 0; i < ordinals_.size(); ++i) {
        DeviceScope owner(backend_, ordinals_[i]);
        pools_.push_back(std::make_unique<AllocationPool>(backend_, admissions_[i]->report().estimated_peak_bytes, reuse));
    }
}
AllocationPoolStats DecoderMemoryAdmission::allocation_stats() const {
    AllocationPoolStats result;
    for (const auto& pool : pools_) {
        const auto item = pool->stats();
        result.requests += item.requests; result.allocations += item.allocations;
        result.reuses += item.reuses; result.evictions += item.evictions;
        result.allocated_bytes += item.allocated_bytes; result.reserved_bytes += item.reserved_bytes;
        result.cached_bytes += item.cached_bytes; result.peak_reserved_bytes += item.peak_reserved_bytes;
        result.peak_cached_bytes += item.peak_cached_bytes;
    }
    return result;
}

MemoryAdmission::MemoryAdmission(Backend backend, const MemoryBudget& budget, size_t additional, size_t host_additional)
    : backend_(backend) {
    report_.budget_bytes = memory_available_limit(backend, budget.limit_bytes, budget.reserve_bytes);
    auto live = memory_stats(backend).live_bytes();
    report_.estimated_peak_bytes = checked_memory_add(live, additional);
    if (report_.estimated_peak_bytes > report_.budget_bytes)
        throw MemoryLimitExceeded(backend, additional, live, report_.budget_bytes);
    if (backend == Backend::CUDA) {
        report_.host_budget_bytes = memory_available_limit(Backend::CPU, 0, budget.reserve_bytes);
        auto host_live = memory_stats(Backend::CPU).live_bytes();
        report_.host_estimated_peak_bytes = checked_memory_add(host_live, host_additional);
        if (report_.host_estimated_peak_bytes > report_.host_budget_bytes)
            throw MemoryLimitExceeded(Backend::CPU, host_additional, host_live, report_.host_budget_bytes);
        host_limit_ = std::make_unique<MemoryLimit>(Backend::CPU, report_.host_budget_bytes);
    }
    limit_ = std::make_unique<MemoryLimit>(backend, report_.budget_bytes);
}
MemoryReport MemoryAdmission::report() const {
    auto result = report_;
    result.measured_peak_bytes = limit_->peak_bytes();
    if (host_limit_) result.host_measured_peak_bytes = host_limit_->peak_bytes();
    else {
        result.host_budget_bytes = result.budget_bytes;
        result.host_estimated_peak_bytes = result.estimated_peak_bytes;
        result.host_measured_peak_bytes = result.measured_peak_bytes;
    }
    return result;
}
} // namespace elman
