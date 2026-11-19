#include "Models/LLM/Qwen2.h"
#include "Inference/LLM/MemoryPlan.h"
#include "Inference/LLM/PagedKVCache.h"
#include "Core/Graph.h"
#include "Backends/CpuMath.h"
#include "Core/TensorDispatch.h"
#include "IO/InferenceWeights.h"
#include <algorithm>

#include <cmath>
#include <cstring>
#include <filesystem>
#include <limits>
#include <map>
#include <stdexcept>

namespace elman {
namespace {
void require(bool condition, const std::string& message) {
    if (!condition) throw std::invalid_argument("Qwen2: " + message);
}
std::vector<int32_t> integers(const Tensor& tensor, bool allow_bool = false) {
    if (tensor.dtype() == DType::Int32 || (allow_bool && tensor.dtype() == DType::Bool)) return tensor.int_data();
    require(tensor.dtype() == DType::Int64, "indices must have Int32 or Int64 dtype");
    auto bytes = tensor.raw_data();
    std::vector<int32_t> values(tensor.numel());
    for (size_t i = 0; i < values.size(); ++i) {
        int64_t value;
        std::memcpy(&value, bytes.data() + i * sizeof(value), sizeof(value));
        require(value >= std::numeric_limits<int32_t>::min() && value <= std::numeric_limits<int32_t>::max(),
                "index exceeds Int32 range");
        values[i] = static_cast<int32_t>(value);
    }
    return values;
}
Tensor normalize(const Tensor& x, const Tensor& weight, float epsilon, bool fused,
                 const std::function<void(const std::string&, const Tensor&)>& trace = {}) {
    if (fused && !trace) return rms_norm(x, weight, epsilon, RmsNormRounding::Normalized);
    auto wide = cast(x, DType::Float32);
    auto variance = mean(mul(wide, wide), -1, true);
    auto inverse = rsqrt(add(variance, epsilon));
    if (trace) { trace("variance", variance); trace("inverse", inverse); trace("normalized", mul(wide, inverse)); }
    if (fused) return rms_norm(x, weight, epsilon, RmsNormRounding::Normalized);
    // Qwen2 rounds the normalized activation before applying the affine weight.
    return mul(weight, cast(mul(wide, inverse), x.dtype()));
}
Tensor rotary(const Tensor& x, const Tensor& cosine, const Tensor& sine) {
    int dim = x.shape().back();
    auto rotated = concat({mul(slice(x, -1, dim / 2, dim), -1.0f), slice(x, -1, 0, dim / 2)}, -1);
    return add(mul(x, cosine), mul(rotated, sine));
}
Tensor repeat_kv(const Tensor& x, int heads) {
    const auto& s = x.shape();
    if (s[1] == heads) return x;
    return reshape(expand(unsqueeze(x, 2), {s[0], s[1], heads / s[1], s[2], s[3]}),
                   {s[0], heads, s[2], s[3]});
}
}

Qwen2Config Qwen2Config::from_json(const Json& value) {
    require(value.at("model_type").string() == "qwen2", "expected model_type qwen2");
    require(value.string_or("hidden_act", "silu") == "silu", "only SiLU activation is supported");
    require(!value.bool_or("use_sliding_window", false), "sliding-window attention is not implemented");
    require(!value.bool_or("use_mrope", false), "multi-axis rotary embeddings are not implemented");
    require(!value.contains("rope_scaling") || value.at("rope_scaling").is_null(), "scaled rotary embeddings are not implemented");
    Qwen2Config result;
    result.vocabulary = value.at("vocab_size").integer();
    result.hidden = value.at("hidden_size").integer();
    result.intermediate = value.at("intermediate_size").integer();
    result.layers = value.at("num_hidden_layers").integer();
    result.heads = value.at("num_attention_heads").integer();
    result.kv_heads = value.int_or("num_key_value_heads", result.heads);
    result.max_positions = value.at("max_position_embeddings").integer();
    result.rms_epsilon = static_cast<float>(value.number_or("rms_norm_eps", 1e-6));
    result.rope_theta = static_cast<float>(value.number_or("rope_theta", 10000));
    result.tied_embeddings = value.bool_or("tie_word_embeddings", false);
    require(result.vocabulary > 0 && result.hidden > 0 && result.intermediate > 0 && result.layers > 0 &&
            result.heads > 0 && result.kv_heads > 0 && result.max_positions > 0, "dimensions must be positive");
    require(result.hidden % result.heads == 0 && result.heads % result.kv_heads == 0,
            "hidden/head dimensions are not divisible");
    require(result.head_dim() % 2 == 0, "rotary head dimension must be even");
    require(std::isfinite(result.rms_epsilon) && result.rms_epsilon > 0 &&
            std::isfinite(result.rope_theta) && result.rope_theta > 0, "invalid normalization or rotary parameters");
    return result;
}

size_t Qwen2::quantization_workspace_bytes(Qwen2Logits logits) const {
    const auto count = std::max(quantized_layer_count_, logits == Qwen2Logits::None ? size_t(0) : quantized_head_count_);
    return count ? memory_allocation_bytes(backend_, dtype_storage_bytes(dtype_, count)) : 0;
}
int32_t Qwen2::synchronize_token(int32_t id) const {
    if (parallel_) parallel_->broadcast(&id, sizeof(id));
    return id;
}

Qwen2::Qwen2(const std::string& directory, Backend backend, DType dtype, bool fused_pointwise, bool fused_normalization,
             bool fused_attention_softmax, bool tiled_attention, bool packed_projections, bool direct_paged_attention,
             const std::vector<int>& devices, int simulated_devices, std::shared_ptr<ProcessGroup> tensor_parallel)
    : config_(Qwen2Config::from_json(Json::read((std::filesystem::path(directory) / "config.json").string()))),
      backend_(backend), dtype_(dtype), placement_(backend, config_.layers, devices, simulated_devices),
      parallel_(std::move(tensor_parallel)),
      parallel_heads_(partition_attention_heads(config_.heads, config_.kv_heads, parallel_ ? parallel_->rank() : 0, tensor_parallel_size())),
      local_intermediate_(config_.intermediate / tensor_parallel_size()),
      fused_pointwise_(fused_pointwise), fused_normalization_(fused_normalization),
      fused_attention_softmax_(fused_attention_softmax), tiled_attention_(tiled_attention && fused_attention_softmax),
      packed_projections_(packed_projections), direct_paged_attention_(direct_paged_attention) {
    require(is_compute_float(dtype), "inference requires Float32, Float16 or BFloat16");
    require(backend_available(backend), "requested backend is unavailable");
    const int ranks = tensor_parallel_size(), rank = parallel_ ? parallel_->rank() : 0;
    require(config_.intermediate % ranks == 0, "tensor parallel ranks must divide the MLP width");
    require(ranks == 1 || (!placement_.distributed() && !direct_paged_attention),
            "tensor parallel ranks use one device and a contiguous cache each");
    require(!placement_.distributed() || !direct_paged_attention, "layer-partitioned models use per-layer contiguous caches");
    auto input_device = placement_.enter_input();
    InferenceWeights weights(directory);
    quantized_weights_ = weights.has_quantization();
    require(ranks == 1 || !quantized_weights_, "tensor parallel Qwen2 requires dense weights");
    const auto& c = config_;
    require(weights.tensors().size() == size_t(c.layers) * 12 + 2 + (c.tied_embeddings ? 0 : 1),
            "checkpoint tensor count differs from the configuration");
    int kv_width = c.kv_heads * c.head_dim();
    std::map<std::string, Shape> expected = {
        {"model.embed_tokens.weight", {c.vocabulary, c.hidden}}, {"model.norm.weight", {c.hidden}}
    };
    if (!c.tied_embeddings) expected.emplace("lm_head.weight", Shape{c.vocabulary, c.hidden});
    for (int i = 0; i < c.layers; ++i) {
        std::string prefix = "model.layers." + std::to_string(i) + ".";
        for (auto norm : {"input_layernorm.weight", "post_attention_layernorm.weight"})
            expected.emplace(prefix + norm, Shape{c.hidden});
        for (auto projection : {"q_proj", "k_proj", "v_proj"}) {
            int width = std::string(projection) == "q_proj" ? c.hidden : kv_width;
            expected.emplace(prefix + "self_attn." + projection + ".weight", Shape{width, c.hidden});
            expected.emplace(prefix + "self_attn." + projection + ".bias", Shape{width});
        }
        expected.emplace(prefix + "self_attn.o_proj.weight", Shape{c.hidden, c.hidden});
        expected.emplace(prefix + "mlp.gate_proj.weight", Shape{c.intermediate, c.hidden});
        expected.emplace(prefix + "mlp.up_proj.weight", Shape{c.intermediate, c.hidden});
        expected.emplace(prefix + "mlp.down_proj.weight", Shape{c.hidden, c.intermediate});
    }
    require(weights.tensors().size() == expected.size(), "checkpoint tensor count differs from the configuration");
    for (const auto& item : expected) {
        const auto& info = weights.info(item.first);
        require(info.shape == item.second, "checkpoint shape differs: " + item.first);
        require(is_compute_float(info.dtype) || (weights.quantized(item.first) && info.shape.size() == 2),
                "checkpoint uses unsupported weight dtype: " + item.first);
        parameter_count_ += shape_element_count(item.second);
    }
    auto load = [&](const std::string& name) { return cast(weights.load(name, backend), dtype); };
    auto load_weight = [&](const std::string& name) {
        return weights.quantized(name) ? Weight(weights.load_quantized(name, backend)) : Weight(load(name));
    };
    auto shard = [&](const std::string& name, int axis, int offset, int size) {
        return cast(weights.load_slice(name, axis, offset, size, backend), dtype);
    };
    embedding_ = load_weight("model.embed_tokens.weight");
    {
        auto output_device = placement_.enter_output();
        norm_ = load("model.norm.weight");
        lm_head_ = c.tied_embeddings ? embedding_.to_device(placement_.output_device()) : load_weight("lm_head.weight");
    }
    for (int i = 0; i < c.layers; ++i) {
        auto layer_device = placement_.enter_layer(i);
        std::string p = "model.layers." + std::to_string(i) + ".";
        Layer layer;
        layer.input_norm = load(p + "input_layernorm.weight");
        layer.post_norm = load(p + "post_attention_layernorm.weight");
        auto load_sharded = [&](const std::string& name, int axis, int offset, int size) {
            return ranks == 1 ? load_weight(name) : Weight(shard(name, axis, offset, size));
        };
        auto attention_projection = [&](const char* name, int offset, int heads, Weight& weight, Tensor& bias) {
            const std::string prefix = p + "self_attn." + name;
            const int dim = c.head_dim(), width = heads * dim;
            weight = load_sharded(prefix + ".weight", 0, offset * dim, width);
            bias = ranks == 1 ? load(prefix + ".bias") : shard(prefix + ".bias", 0, offset * dim, width);
        };
        attention_projection("q_proj", parallel_heads_.query_offset, parallel_heads_.query_heads, layer.query, layer.query_bias);
        attention_projection("k_proj", parallel_heads_.kv_offset, parallel_heads_.kv_heads, layer.key, layer.key_bias);
        attention_projection("v_proj", parallel_heads_.kv_offset, parallel_heads_.kv_heads, layer.value, layer.value_bias);
        if (packed_projections_) {
            layer.qkv = Weight::concat_rows({layer.query, layer.key, layer.value});
            layer.query = layer.key = layer.value = {};
            layer.qkv_bias = concat({layer.query_bias, layer.key_bias, layer.value_bias}, 0);
            layer.query_bias = layer.key_bias = layer.value_bias = {};
        }
        layer.output = load_sharded(p + "self_attn.o_proj.weight", 1, parallel_heads_.query_offset * c.head_dim(),
                                    parallel_heads_.query_heads * c.head_dim());
        const int intermediate_offset = rank * local_intermediate_;
        layer.gate = load_sharded(p + "mlp.gate_proj.weight", 0, intermediate_offset, local_intermediate_);
        layer.up = load_sharded(p + "mlp.up_proj.weight", 0, intermediate_offset, local_intermediate_);
        if (packed_projections_) {
            layer.gate_up = Weight::concat_rows({layer.gate, layer.up});
            layer.gate = layer.up = {};
        }
        layer.down = load_sharded(p + "mlp.down_proj.weight", 1, intermediate_offset, local_intermediate_);
        for (const auto* weight : {&layer.query, &layer.key, &layer.value, &layer.output,
                                    &layer.gate, &layer.up, &layer.down, &layer.qkv, &layer.gate_up})
            quantized_layer_count_ = std::max(quantized_layer_count_, weight->workspace_count());
        layers_.push_back(std::move(layer));
    }
    quantized_head_count_ = lm_head_.workspace_count();
    // The pinned Transformers model constructs the original RoPE cache on CPU
    // in Float32, before model.to(device, dtype). Preserve that order here.
    for (int i = 0; i < c.head_dim(); i += 2)
        inverse_frequencies_.push_back(1.0f / std::pow(c.rope_theta, float(i) / c.head_dim()));
}

Qwen2Output Qwen2::forward(const Tensor& input_ids, const Tensor& attention_mask,
                          const Tensor& position_ids, const Qwen2Cache* past,
                          bool capture_hidden, Qwen2Logits logits,
                          const std::function<void(const std::string&, const Tensor&)>& trace) const {
    return forward_impl(input_ids, attention_mask, position_ids, past, nullptr, capture_hidden, logits, trace);
}

Qwen2Output Qwen2::forward_paged(const Tensor& input_ids, PagedKVCache& cache,
                                const Tensor& attention_mask, const Tensor& position_ids,
                                bool capture_hidden, Qwen2Logits logits,
                                const std::function<void(const std::string&, const Tensor&)>& trace) const {
    require(supports_paged_cache(), "layer-partitioned models require per-layer contiguous caches");
    require(input_ids.defined() && input_ids.shape().size() == 2 && input_ids.shape()[0] == cache.batch(), "paged cache batch differs");
    auto shape = cache.shape();
    require(shape.layers == config_.layers && shape.heads == config_.kv_heads && shape.dimension == config_.head_dim() &&
            cache.backend() == backend_ && cache.dtype() == dtype_ && cache.max_length() <= config_.max_positions,
            "paged cache geometry, backend, dtype or context differs");
    auto append = cache.append(input_ids.shape()[1]);
    auto result = forward_impl(input_ids, attention_mask, position_ids, nullptr, append.get(), capture_hidden, logits, trace);
    append->commit();
    return result;
}

Qwen2Output Qwen2::forward_paged_batch(const Tensor& input_ids, PagedKVBatch& cache,
                                      const Tensor& attention_mask, const Tensor& position_ids,
                                      bool capture_hidden, Qwen2Logits logits,
                                      const std::function<void(const std::string&, const Tensor&)>& trace) const {
    require(supports_paged_cache(), "layer-partitioned models require per-layer contiguous caches");
    require(input_ids.defined() && input_ids.shape().size() == 2 && input_ids.shape()[0] == cache.batch(), "paged request batch differs");
    auto shape = cache.shape();
    require(shape.layers == config_.layers && shape.heads == config_.kv_heads && shape.dimension == config_.head_dim() &&
            cache.backend() == backend_ && cache.dtype() == dtype_ && cache.max_length() <= config_.max_positions,
            "paged request geometry, backend, dtype or context differs");
    auto append = cache.append(input_ids.shape()[1]);
    const auto& structural = append->padding_mask();
    // Account for the copied mask and Int64 conversion/download temporaries.
    MemoryReservation batch_mask(Backend::CPU, checked_memory_mul(structural.size(), 16));
    auto mask = structural;
    if (attention_mask.defined()) {
        require(attention_mask.shape() == Shape{cache.batch(), append->total()}, "batched attention mask shape differs");
        mask = integers(attention_mask, true);
        for (size_t i = 0; i < mask.size(); ++i)
            require((mask[i] == 0 || mask[i] == 1) && (structural[i] || !mask[i]), "batch mask exposes padding or contains an invalid value");
    }
    auto explicit_mask = Tensor::from_int_data({cache.batch(), append->total()}, mask);
    auto result = forward_impl(input_ids, explicit_mask, position_ids, nullptr, append.get(), capture_hidden, logits, trace);
    append->commit();
    return result;
}

Qwen2Output Qwen2::forward_impl(const Tensor& input_ids, const Tensor& attention_mask,
                               const Tensor& position_ids, const Qwen2Cache* past, PagedKVAppend* append,
                               bool capture_hidden, Qwen2Logits logits,
                               const std::function<void(const std::string&, const Tensor&)>& trace, GraphInputs* graph_inputs,
                               const PagedGraphInputs* graph_paged) const {
    auto input_device = placement_.enter_input();
    graph::RecordingGuard guard;
    const auto& c = config_;
    require(logits == Qwen2Logits::All || logits == Qwen2Logits::Last || logits == Qwen2Logits::None, "invalid logits selection");
    require(input_ids.shape().size() == 2, "input IDs must have rank two");
    int batch = input_ids.shape()[0], sequence = input_ids.shape()[1], previous = append ? append->previous() : past ? past->length : 0;
    if (graph_paged) {
        require(!append && !past && sequence == 1 && batch == graph_paged->map.batch() && !capture_hidden && !trace,
                "invalid fixed paged decode graph arguments");
        previous = graph_paged->map.length() - 1;
    }
    require(batch > 0 && sequence > 0, "empty input batch or sequence");
    require(previous >= 0 && previous <= c.max_positions && sequence <= c.max_positions - previous,
            "sequence exceeds configured context length");
    int total = previous + sequence, dim = c.head_dim();
    if (past) {
        require(past->layers.size() == layers_.size(), "cache layer count differs");
        for (const auto& layer : past->layers) {
            for (const auto* tensor : {&layer.key, &layer.value})
                require(tensor->defined() && tensor->shape() == Shape{batch, parallel_heads_.kv_heads, previous, dim} &&
                        tensor->backend() == backend_ && tensor->dtype() == dtype_, "cache shape, backend or dtype differs");
        }
    }
    size_t host_bytes = qwen_host_workspace(c, size_t(batch), size_t(sequence), size_t(total), logits == Qwen2Logits::All,
        tiled_attention_ && !trace, backend_ == Backend::CPU);
    if (tiled_attention_ && trace) host_bytes = checked_memory_add(host_bytes, checked_memory_mul(size_t(batch) * total, 8));
    MemoryReservation host_workspace(Backend::CPU, host_bytes);
    auto ids = integers(input_ids);
    for (int id : ids) require(id >= 0 && id < c.vocabulary, "token ID is outside the vocabulary");
    std::vector<int32_t> mask(size_t(batch) * total, 1);
    if (attention_mask.defined()) {
        require(attention_mask.shape() == Shape{batch, total}, "attention mask shape differs");
        mask = integers(attention_mask, true);
        for (int value : mask) require(value == 0 || value == 1, "attention mask values must be zero or one");
    }
    std::vector<int32_t> positions(ids.size());
    if (position_ids.defined()) {
        require(position_ids.shape() == Shape{batch, sequence}, "position IDs shape differs");
        positions = integers(position_ids);
    } else {
        for (int b = 0; b < batch; ++b) {
            int position = 0;
            for (int t = 0; t < total; ++t) {
                if (mask[size_t(b) * total + t]) ++position;
                if (t >= previous) positions[size_t(b) * sequence + t - previous] = mask[size_t(b) * total + t] ? position - 1 : 0;
            }
        }
    }
    for (int position : positions) require(position >= 0 && position < c.max_positions, "position ID exceeds the configured range");
    float minimum = dtype_ == DType::Float16 ? -65504.0f :
                    dtype_ == DType::BFloat16 ? -3.3895313892515355e38f : std::numeric_limits<float>::lowest();
    const bool direct = graph_paged || (append && direct_paged_attention_);
    std::unique_ptr<PagedAttentionMap> paged_map;
    if (append && direct) paged_map = std::make_unique<PagedAttentionMap>(append->attention_map(mask));
    Tensor causal_mask, padding_mask;
    if ((!tiled_attention_ && !direct) || trace) {
        Shape mask_shape{batch, 1, sequence, total};
        std::vector<float> causal(shape_element_count(mask_shape), 0);
        for (int b = 0; b < batch; ++b)
            for (int q = 0; q < sequence; ++q)
                for (int k = 0; k < total; ++k)
                    if (k > previous + q || mask[size_t(b) * total + k] == 0)
                        causal[(size_t(b) * sequence + q) * total + k] = minimum;
        causal_mask = Tensor::from_data(mask_shape, causal, backend_, false, dtype_);
    }
    if (tiled_attention_ && !direct) {
        std::vector<float> padding(mask.size());
        for (size_t i = 0; i < mask.size(); ++i) padding[i] = mask[i] ? 0 : minimum;
        padding_mask = Tensor::from_data({batch, 1, 1, total}, padding, backend_, false, dtype_);
    }
    auto tables = rotary_values(positions);
    auto cosine = Tensor::from_data({batch, 1, sequence, dim}, tables.first, backend_, false, dtype_);
    auto sine = Tensor::from_data({batch, 1, sequence, dim}, tables.second, backend_, false, dtype_);
    if (trace) { trace("rope.cos", cosine); trace("rope.sin", sine); }
    auto indices = Tensor::from_int_data({batch * sequence}, ids, backend_);
    if (graph_inputs) *graph_inputs = {indices, cosine, sine, padding_mask, causal_mask};
    auto hidden = reshape(embedding_.embedding(indices, dtype_), {batch, sequence, c.hidden});
    DeviceTensorCopies cosine_copies(cosine), sine_copies(sine), padding_copies(padding_mask), causal_copies(causal_mask);
    std::map<int, Tensor> weight_workspaces;
    const size_t workspace_count = std::max(quantized_layer_count_, logits == Qwen2Logits::None ? size_t(0) : quantized_head_count_);
    auto workspace_for = [&](int device, size_t count) {
        auto& workspace = weight_workspaces[device];
        if (count && (!workspace.defined() || workspace.numel() < count)) {
            require(count <= size_t(std::numeric_limits<int>::max()), "weight workspace exceeds indexing limits");
            workspace = Tensor::zeros({int(count)}, backend_, false, dtype_);
        }
        return workspace;
    };
    Qwen2Output result;
    result.cache.length = total;
    for (size_t i = 0; i < layers_.size(); ++i) {
        const int ordinal = placement_.layer_device(int(i));
        auto layer_device = placement_.enter_layer(int(i));
        hidden = placement_.transfer_to_layer(hidden, int(i));
        const auto& cosine = cosine_copies.get(ordinal);
        const auto& sine = sine_copies.get(ordinal);
        const auto& padding_mask = padding_copies.get(ordinal);
        const auto& causal_mask = causal_copies.get(ordinal);
        auto weight_workspace = workspace_for(ordinal, placement_.distributed() ? quantized_layer_count_ : workspace_count);
        if (past) require(past->layers[i].key.device() == ordinal && past->layers[i].value.device() == ordinal,
                          "cache layer belongs to a different device");
        const auto& layer = layers_[i];
        auto emit = [&](const std::string& name, const Tensor& tensor) {
            if (trace) trace("layer." + std::to_string(i) + "." + name, tensor);
        };
        if (capture_hidden) result.hidden_states.push_back(hidden);
        auto normalized = normalize(hidden, layer.input_norm, c.rms_epsilon, fused_normalization_);
        emit("input_norm", normalized);
        auto project = [&](const Tensor& projected, int heads) {
            if (sequence == 1) return reshape(projected, {batch, heads, 1, dim});
            return transpose(reshape(projected, {batch, sequence, heads, dim}), 1, 2);
        };
        Tensor query, key, value;
        const int query_heads = parallel_heads_.query_heads, kv_heads = parallel_heads_.kv_heads;
        const int query_width = query_heads * dim;
        if (packed_projections_) {
            int kv_width = kv_heads * dim;
            size_t rows = size_t(batch) * sequence;
            auto projections = layer.qkv.packed_linear(normalized, {query_width, kv_width, kv_width}, layer.qkv_bias, weight_workspace);
            query = project(contiguous_view(projections, {batch, sequence, query_width}), query_heads);
            key = project(contiguous_view(projections, {batch, sequence, kv_width}, rows * query_width), kv_heads);
            value = project(contiguous_view(projections, {batch, sequence, kv_width}, rows * (query_width + kv_width)), kv_heads);
        } else {
            query = project(layer.query.linear(normalized, layer.query_bias, weight_workspace), query_heads);
            key = project(layer.key.linear(normalized, layer.key_bias, weight_workspace), kv_heads);
        }
        emit("query", query); emit("key", key);
        query = fused_pointwise_ ? rotary_embedding(query, cosine, sine) : rotary(query, cosine, sine);
        key = fused_pointwise_ ? rotary_embedding(key, cosine, sine) : rotary(key, cosine, sine);
        if (!packed_projections_) value = project(layer.value.linear(normalized, layer.value_bias, weight_workspace), kv_heads);
        emit("rotary_query", query); emit("rotary_key", key); emit("value", value);
        if (graph_paged) {
            auto physical = graph_paged->storage.at(i);
            const int width = kv_heads * dim;
            physical.key.assign_indexed(0, graph_paged->destinations, reshape(transpose(key, 1, 2), {batch, width}));
            physical.value.assign_indexed(0, graph_paged->destinations, reshape(transpose(value, 1, 2), {batch, width}));
            key = std::move(physical.key);
            value = std::move(physical.value);
        } else if (append) {
            auto cached = append->write(int(i), key, value, !direct);
            key = std::move(cached.key);
            value = std::move(cached.value);
        } else if (previous) {
            key = concat({past->layers[i].key, key}, 2);
            value = concat({past->layers[i].value, value}, 2);
        }
        PagedKVLayer logical{key, value};
        if (direct && (capture_hidden || trace)) logical = append->read_written(int(i));
        if ((!append && !graph_paged) || capture_hidden) result.cache.layers.push_back({logical.key, logical.value});
        emit("cached_key", logical.key); emit("cached_value", logical.value);
        float divisor = std::sqrt(float(dim));
        Tensor attended;
        if (direct) {
            const auto& map = graph_paged ? graph_paged->map : *paged_map;
            attended = elman::paged_attention(query, key, value, map, kv_heads, divisor,
                backend_ == Backend::CPU ? std::min(sequence, 1024) : kQwenAttentionQueryTile, !graph_paged, graph_paged ? 0 : previous);
        }
        else if (tiled_attention_)
            attended = elman::tiled_attention(query, key, value, padding_mask, divisor, kQwenAttentionQueryTile, true, previous,
                backend_ == Backend::CPU ? AttentionTiling::Heads : AttentionTiling::Queries);
        if ((!tiled_attention_ && !direct) || trace) {
            // Scores/probabilities remain eager trace diagnostics. The attended
            // output still comes from the selected native tiled path.
            auto unscaled = matmul_transposed(query, repeat_kv(logical.key, query_heads));
            Tensor probability;
            if (fused_attention_softmax_) {
                if (trace) emit("scores", div(unscaled, divisor));
                probability = attention_softmax(unscaled, causal_mask, divisor);
            } else {
                auto scores = div(unscaled, divisor);
                emit("scores", scores);
                probability = cast(softmax(cast(add(scores, causal_mask), DType::Float32), -1), dtype_);
            }
            emit("probability", probability);
            if (!tiled_attention_ && !direct) attended = matmul(probability, repeat_kv(logical.value, query_heads));
        }
        emit("attended", attended);
        auto merged = reshape(sequence == 1 ? attended : transpose(attended, 1, 2), {batch, sequence, query_width});
        {
            auto projected = layer.output.linear(merged, {}, weight_workspace);
            if (parallel_) projected = reduce_tensor_parallel(projected, *parallel_);
            hidden = add(hidden, projected);
        }
        emit("attention_residual", hidden);
        std::function<void(const std::string&, const Tensor&)> norm_trace;
        if (trace) norm_trace = [&](const std::string& name, const Tensor& tensor) { emit("post_norm." + name, tensor); };
        normalized = normalize(hidden, layer.post_norm, c.rms_epsilon, fused_normalization_, norm_trace);
        emit("post_norm", normalized);
        Tensor gate, up;
        if (packed_projections_) {
            auto projections = layer.gate_up.packed_linear(normalized, {local_intermediate_, local_intermediate_}, {}, weight_workspace);
            gate = contiguous_view(projections, {batch, sequence, local_intermediate_});
            up = contiguous_view(projections, {batch, sequence, local_intermediate_}, size_t(batch) * sequence * local_intermediate_);
        } else {
            gate = layer.gate.linear(normalized, {}, weight_workspace);
            up = layer.up.linear(normalized, {}, weight_workspace);
        }
        emit("gate", gate); emit("up", up);
        auto activated = fused_pointwise_ ? silu_mul(gate, up) : mul(silu(gate), up);
        emit("activated", activated);
        auto down = layer.down.linear(activated, {}, weight_workspace);
        if (parallel_) down = reduce_tensor_parallel(down, *parallel_);
        emit("down", down);
        hidden = add(hidden, down);
        emit("output", hidden);
    }
    auto output_device = placement_.enter_output();
    hidden = hidden.to_device(placement_.output_device());
    auto weight_workspace = workspace_for(placement_.output_device(), logits == Qwen2Logits::None ? 0 : quantized_head_count_);
    if (capture_hidden || logits != Qwen2Logits::None)
        hidden = normalize(hidden, norm_, c.rms_epsilon, fused_normalization_);
    if (capture_hidden) result.hidden_states.push_back(hidden);
    if (logits == Qwen2Logits::Last && sequence > 1) hidden = slice(hidden, 1, sequence - 1, sequence);
    if (logits != Qwen2Logits::None) {
        result.logits = cast(lm_head_.linear(hidden, {}, weight_workspace), DType::Float32);
        if (trace) trace("logits", result.logits);
        result.logits = result.logits.to_device(placement_.input_device());
    }
    return result;
}

std::pair<std::vector<float>, std::vector<float>> Qwen2::rotary_values(const std::vector<int32_t>& positions) const {
    const int dim = config_.head_dim();
    std::vector<float> cosines(shape_element_count({int(positions.size()), dim})), sines(cosines.size());
    for (size_t i = 0; i < positions.size(); ++i) for (int j = 0; j < dim / 2; ++j) {
        const float angle = float(positions[i]) * inverse_frequencies_[j];
        cosines[i * dim + j] = cosines[i * dim + j + dim / 2] = angle;
        sines[i * dim + j] = sines[i * dim + j + dim / 2] = angle;
    }
    // Build FP32 tables with the same CPU math path for eager and graph decode.
    if (!try_unary_math_cpu(cosines.data(), cosines.data(), int(cosines.size()), CpuMathOperation::Cos))
        for (auto& value : cosines) value = std::cos(value);
    if (!try_unary_math_cpu(sines.data(), sines.data(), int(sines.size()), CpuMathOperation::Sin))
        for (auto& value : sines) value = std::sin(value);
    return {std::move(cosines), std::move(sines)};
}
} // namespace elman
