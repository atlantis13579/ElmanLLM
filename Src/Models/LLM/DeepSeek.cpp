#include "Models/LLM/DeepSeek.h"
#include "Core/LocalFunction.h"
#include "Inference/LLM/LatentAttention.h"
#include "IO/InferenceWeights.h"
#include "Core/Graph.h"
#include "Core/Memory.h"
#include "Core/TensorDispatch.h"
#include "Backends/CpuMath.h"
#include <algorithm>
#include <climits>
#include <cmath>
#include <filesystem>
#include <limits>
#include <set>
#include <stdexcept>

namespace elman {
namespace {
void require(bool valid, const std::string& message) {
    if (!valid) throw std::invalid_argument("DeepSeek: " + message);
}
int optional_integer(const Json& value, const std::string& name, int fallback) {
    return !value.contains(name) || value.at(name).is_null() ? fallback : value.at(name).integer();
}
double magnitude(double factor, double coefficient) {
    return factor <= 1 ? 1 : 1 + .1 * coefficient * std::log(factor);
}
std::vector<int32_t> integers(const Tensor& value, bool mask = false) {
    require(value.defined(), "undefined integer input");
    if (value.dtype() == DType::Int32 || (mask && value.dtype() == DType::Bool)) return value.int_data();
    require(value.dtype() == DType::Int64, "indices require Int32 or Int64");
    std::vector<int64_t> source = value.int64_data();
    std::vector<int32_t> result;
    result.reserve(source.size());
    for (int64_t number : source) {
        require(number >= INT_MIN && number <= INT_MAX, "integer input exceeds Int32 indexing");
        result.push_back(int32_t(number));
    }
    return result;
}
Tensor rotate(const Tensor& input, const Tensor& cosine, const Tensor& sine) {
    const Shape& s = input.shape();
    Tensor paired = reshape(input, {s[0], s[1], s[2], s[3] / 2, 2});
    Tensor split = reshape(transpose(paired, 3, 4), s);
    return rotary_embedding(split, cosine, sine);
}
}

static std::pair<Tensor, Tensor> split_latent_projection(const Tensor& weight, const DeepSeekConfig& c) {
    const int width = c.plain_dimension + c.value_dimension;
    Tensor projection = reshape(weight, {c.heads, width, c.latent_cache_rank});
    return {slice(projection, 1, 0, c.plain_dimension), slice(projection, 1, c.plain_dimension, width)};
}

DeepSeekConfig DeepSeekConfig::from_json(const Json& value) {
    DeepSeekConfig c;
    c.model_type = value.at("model_type").string();
    require(c.model_type == "deepseek_v2" || c.model_type == "deepseek_v3", "unsupported model type");
    require(value.string_or("hidden_act", "silu") == "silu", "only SiLU feed-forward blocks are supported");
    c.vocabulary = value.at("vocab_size").integer();
    c.hidden = value.at("hidden_size").integer();
    c.intermediate = value.at("intermediate_size").integer();
    c.layers = value.at("num_hidden_layers").integer();
    c.heads = value.at("num_attention_heads").integer();
    c.kv_heads = 1;
    c.max_positions = value.at("max_position_embeddings").integer();
    c.query_rank = optional_integer(value, "q_lora_rank", 0);
    c.latent_cache_rank = value.at("kv_lora_rank").integer();
    c.plain_dimension = value.at("qk_nope_head_dim").integer();
    c.rotary_dimension = c.latent_rotary_dimension = value.at("qk_rope_head_dim").integer();
    c.value_dimension = value.at("v_head_dim").integer();
    c.rms_epsilon = float(value.number_or("rms_norm_eps", 1e-6));
    c.rope_theta = float(value.number_or("rope_theta", 10000));
    c.tied_embeddings = value.bool_or("tie_word_embeddings", false);
    c.attention_bias = value.bool_or("attention_bias", false);
    c.routing.experts = optional_integer(value, "n_routed_experts", 0);
    c.shared_experts = optional_integer(value, "n_shared_experts", 0);
    c.expert_intermediate = optional_integer(value, "moe_intermediate_size", c.intermediate);
    c.first_dense_layers = value.int_or("first_k_dense_replace", 0);
    c.expert_frequency = value.int_or("moe_layer_freq", 1);
    require(c.vocabulary > 0 && c.hidden > 0 && c.intermediate > 0 && c.layers > 0 && c.heads > 0 &&
            c.max_positions > 0 && c.query_rank >= 0 && c.latent_cache_rank > 0 && c.plain_dimension > 0 &&
            c.rotary_dimension > 0 && c.rotary_dimension % 2 == 0 && c.value_dimension > 0, "invalid decoder dimensions");
    require(std::isfinite(c.rms_epsilon) && c.rms_epsilon > 0 && std::isfinite(c.rope_theta) && c.rope_theta > 1,
            "invalid normalization or rotary parameters");
    require(c.routing.experts >= 0 && c.shared_experts >= 0 && c.expert_intermediate > 0 && c.first_dense_layers >= 0 &&
            c.first_dense_layers <= c.layers && c.expert_frequency > 0, "invalid expert configuration");
    require(int64_t(c.heads) * (int64_t(c.plain_dimension) + c.rotary_dimension) <= INT_MAX &&
            int64_t(c.heads) * (int64_t(c.plain_dimension) + c.value_dimension) <= INT_MAX &&
            int64_t(c.expert_intermediate) * c.shared_experts <= INT_MAX, "projection dimensions exceed indexing limits");
    if (c.routing.experts) {
        c.routing.top_k = value.at("num_experts_per_tok").integer();
        c.routing.groups = value.int_or("n_group", 1);
        c.routing.selected_groups = value.int_or("topk_group", c.routing.groups);
        c.routing.scale = float(value.number_or("routed_scaling_factor", 1));
        const std::string score = value.string_or("scoring_func", "softmax");
        require(score == "softmax" || score == "sigmoid", "unsupported expert scoring function");
        c.routing.score = score == "softmax" ? ExpertScore::Softmax : ExpertScore::Sigmoid;
        const std::string method = value.string_or("topk_method", "greedy");
        require(method == "greedy" || method == "group_limited_greedy" || method == "noaux_tc", "unsupported expert selection method");
        if (method == "greedy") c.routing.groups = c.routing.selected_groups = 1;
        c.routing.group_score = method == "noaux_tc" ? ExpertGroupScore::TopTwo : ExpertGroupScore::Maximum;
        c.routing.zero_masked_groups = method == "group_limited_greedy";
        c.routing.normalize = value.bool_or("norm_topk_prob", false) && c.routing.top_k > 1;
        c.routing.scale_normalized = c.model_type == "deepseek_v3";
        c.routing.validate();
    }
    if (value.contains("rope_scaling") && !value.at("rope_scaling").is_null()) {
        const Json& rope = value.at("rope_scaling");
        c.rope_type = rope.string_or("rope_type", rope.string_or("type", "default"));
        require(c.rope_type == "default" || c.rope_type == "linear" || c.rope_type == "yarn", "unsupported rotary scaling type");
        c.rope_factor = rope.number_or("factor", 1);
        c.rope_original_positions = rope.int_or("original_max_position_embeddings", 4096);
        c.rope_beta_fast = rope.number_or("beta_fast", 32);
        c.rope_beta_slow = rope.number_or("beta_slow", 1);
        c.rope_mscale = rope.number_or("mscale", 1);
        c.rope_mscale_all = rope.number_or("mscale_all_dim", 0);
        require(std::isfinite(c.rope_factor) && c.rope_factor >= 1 && c.rope_original_positions > 0 &&
                std::isfinite(c.rope_beta_fast) && std::isfinite(c.rope_beta_slow) &&
                c.rope_beta_fast >= c.rope_beta_slow && c.rope_beta_slow > 0 &&
                std::isfinite(c.rope_mscale) && c.rope_mscale >= 0 && std::isfinite(c.rope_mscale_all) && c.rope_mscale_all >= 0,
                "invalid rotary scaling parameters");
    }
    c.attention_query_dimension = c.plain_dimension + c.rotary_dimension;
    c.attention_value_dimension = c.value_dimension;
    c.routed_expert_count = c.routing.experts;
    c.active_experts = c.routing.top_k;
    c.execution_intermediate = std::max({c.intermediate, c.expert_intermediate,
                                        c.expert_intermediate * c.shared_experts});
    return c;
}
double DeepSeekConfig::attention_scale() const {
    const double gain = rope_type == "yarn" ? magnitude(rope_factor, rope_mscale_all) : 1.;
    return gain * gain / std::sqrt(double(plain_dimension) + rotary_dimension);
}

DeepSeek::DeepSeek(const std::string& directory, Backend backend, DType dtype, bool absorbed_attention, const std::vector<int>& devices, int simulated_devices)
    : config_(DeepSeekConfig::from_json(Json::read((std::filesystem::path(directory) / "config.json").string()))),
      backend_(backend), dtype_(dtype), placement_(backend, config_.layers, devices, simulated_devices), absorbed_attention_(absorbed_attention) {
    require(backend_available(backend) && is_compute_float(dtype), "inference backend or dtype is unavailable");
    DeviceStage input_device = placement_.enter_input();
    InferenceWeights weights(directory);
    quantized_weights_ = weights.has_quantization();
    const DeepSeekConfig& c = config_;
    std::set<std::string> loaded;
    LocalFunction check{[&](const std::string& name, const Shape& shape) {
        const SafetensorInfo& info = weights.info(name);
        require(info.shape == shape && (is_compute_float(info.dtype) || weights.quantized(name)), "weight geometry or dtype differs: " + name);
        require(loaded.insert(name).second, "weight loaded twice: " + name);
        parameter_count_ = checked_memory_add(parameter_count_, shape_element_count(shape));
    }};
    LocalFunction dense{[&](const std::string& name, const Shape& shape, DType type) {
        check(name, shape);
        require(!weights.quantized(name), "normalization, bias and routing weights must be dense: " + name);
        return cast(weights.load(name, backend), type);
    }};
    LocalFunction matrix{[&](const std::string& name, const Shape& shape) {
        check(name, shape);
        ProjectionWeight result = weights.quantized(name) ? ProjectionWeight(weights.load_quantized(name, backend)) :
            ProjectionWeight(cast(weights.load(name, backend), dtype));
        layer_workspace_count_ = std::max(layer_workspace_count_, result.workspace_count());
        return result;
    }};
    LocalFunction expert{[&](const std::string& prefix, int width) {
        Expert result;
        result.gate = matrix(prefix + "gate_proj.weight", {width, c.hidden});
        result.up = matrix(prefix + "up_proj.weight", {width, c.hidden});
        result.down = matrix(prefix + "down_proj.weight", {c.hidden, width});
        return result;
    }};
    embedding_ = matrix("model.embed_tokens.weight", {c.vocabulary, c.hidden});
    {
        DeviceStage output_device = placement_.enter_output();
        norm_ = dense("model.norm.weight", {c.hidden}, dtype);
        head_ = c.tied_embeddings ? embedding_.to_device(placement_.output_device()) : matrix("lm_head.weight", {c.vocabulary, c.hidden});
    }
    head_workspace_count_ = head_.workspace_count();
    layer_workspace_count_ = 0;
    const int query_width = c.heads * (c.plain_dimension + c.rotary_dimension);
    for (int index = 0; index < c.layers; ++index) {
        DeviceStage layer_device = placement_.enter_layer(index);
        const std::string p = "model.layers." + std::to_string(index) + ".";
        Layer layer;
        layer.input_norm = dense(p + "input_layernorm.weight", {c.hidden}, dtype);
        layer.post_norm = dense(p + "post_attention_layernorm.weight", {c.hidden}, dtype);
        if (c.query_rank) {
            layer.query_a = matrix(p + "self_attn.q_a_proj.weight", {c.query_rank, c.hidden});
            layer.query_b = matrix(p + "self_attn.q_b_proj.weight", {query_width, c.query_rank});
            layer.query_norm = dense(p + "self_attn.q_a_layernorm.weight", {c.query_rank}, dtype);
            if (c.attention_bias) layer.query_bias = dense(p + "self_attn.q_a_proj.bias", {c.query_rank}, dtype);
        } else layer.query = matrix(p + "self_attn.q_proj.weight", {query_width, c.hidden});
        layer.latent_a = matrix(p + "self_attn.kv_a_proj_with_mqa.weight", {c.latent_cache_rank + c.rotary_dimension, c.hidden});
        layer.latent_b = matrix(p + "self_attn.kv_b_proj.weight", {c.heads * (c.plain_dimension + c.value_dimension), c.latent_cache_rank});
        if (absorbed_attention_ && layer.latent_b.dense.defined()) {
            layer.absorbed_weights = split_latent_projection(layer.latent_b.dense, c);
            layer.latent_b = {};
        }
        layer.latent_norm = dense(p + "self_attn.kv_a_layernorm.weight", {c.latent_cache_rank}, dtype);
        layer.output = matrix(p + "self_attn.o_proj.weight", {c.hidden, c.heads * c.value_dimension});
        if (c.attention_bias) {
            layer.latent_bias = dense(p + "self_attn.kv_a_proj_with_mqa.bias", {c.latent_cache_rank + c.rotary_dimension}, dtype);
            layer.output_bias = dense(p + "self_attn.o_proj.bias", {c.hidden}, dtype);
        }
        if (c.sparse_layer(index)) {
            layer.router = dense(p + "mlp.gate.weight", {c.routing.experts, c.hidden}, DType::Float32);
            if (c.routing.group_score == ExpertGroupScore::TopTwo)
                layer.correction = dense(p + "mlp.gate.e_score_correction_bias", {c.routing.experts}, DType::Float32);
            for (int id = 0; id < c.routing.experts; ++id)
                layer.experts.push_back(expert(p + "mlp.experts." + std::to_string(id) + ".", c.expert_intermediate));
            if (c.shared_experts) layer.shared = expert(p + "mlp.shared_experts.", c.shared_experts * c.expert_intermediate);
        } else layer.dense = expert(p + "mlp.", c.intermediate);
        layers_.push_back(std::move(layer));
    }
    require(loaded.size() == weights.tensors().size(), "checkpoint contains tensors outside the configured decoder");
    const int dim = c.rotary_dimension;
    double lower = 0, upper = 1;
    if (c.rope_type == "yarn") {
        LocalFunction correction{[&](double rotations) { return dim * std::log(c.rope_original_positions / (rotations * 2 * 3.14159265358979323846)) / (2 * std::log(c.rope_theta)); }};
        lower = std::max(0., std::floor(correction(c.rope_beta_fast)));
        upper = std::min(double(dim - 1), std::ceil(correction(c.rope_beta_slow)));
        if (lower == upper) upper += .001;
        rotary_magnitude_ = float(magnitude(c.rope_factor, c.rope_mscale) / magnitude(c.rope_factor, c.rope_mscale_all));
    }
    for (int i = 0; i < dim / 2; ++i) {
        const float base = 1.f / std::pow(c.rope_theta, float(2 * i) / dim);
        float frequency = base;
        if (c.rope_type == "linear") frequency = float(base / c.rope_factor);
        if (c.rope_type == "yarn") {
            const float ramp = float(std::clamp((i - lower) / (upper - lower), 0., 1.));
            frequency = float(base / c.rope_factor) * ramp + base * (1.f - ramp);
        }
        inverse_frequencies_.push_back(frequency);
    }
}

size_t DeepSeek::quantization_workspace_bytes(DecoderLogits logits) const {
    const uint64_t count = std::max(layer_workspace_count_, logits == DecoderLogits::None ? size_t(0) : head_workspace_count_);
    return count ? memory_allocation_bytes(backend_, dtype_storage_bytes(dtype_, count)) : 0;
}

DecoderOutput DeepSeek::forward(const Tensor& input_ids, const Tensor& attention_mask, const Tensor& position_ids,
    const DecoderCache* past, bool capture_hidden, DecoderLogits logits, const DecoderTrace& trace) const {
    DeviceStage input_device = placement_.enter_input();
    const DeepSeekConfig& c = config_;
    require(logits == DecoderLogits::All || logits == DecoderLogits::Last || logits == DecoderLogits::None, "invalid logits selection");
    require(input_ids.defined() && input_ids.shape().size() == 2 && input_ids.shape()[0] > 0 && input_ids.shape()[1] > 0,
            "input IDs require a nonempty [batch,sequence] matrix");
    const int batch = input_ids.shape()[0], sequence = input_ids.shape()[1], previous = past ? past->length : 0;
    require(previous >= 0 && int64_t(previous) + sequence <= c.max_positions, "request exceeds context capacity");
    const int total = previous + sequence;
    const size_t host_bytes = checked_memory_add(checked_memory_mul(size_t(batch) * total, 32),
        checked_memory_mul(size_t(batch) * sequence, 32 + size_t(c.rotary_dimension) * 16));
    MemoryReservation host_workspace(Backend::CPU, host_bytes);
    require(!past || (previous ? past->layers.size() == size_t(c.layers) : past->layers.empty()), "cache layer count differs");
    std::vector<int32_t> ids = integers(input_ids);
    for (int id : ids) require(id >= 0 && id < c.vocabulary, "token ID is outside the vocabulary");
    std::vector<int32_t> mask(size_t(batch) * total, 1);
    if (attention_mask.defined()) {
        require(attention_mask.shape() == Shape{batch, total}, "attention mask geometry differs");
        mask = integers(attention_mask, true);
        for (int value : mask) require(value == 0 || value == 1, "attention mask requires zero or one");
    }
    std::vector<int32_t> positions(ids.size());
    if (position_ids.defined()) {
        require(position_ids.shape() == input_ids.shape(), "position ID geometry differs");
        positions = integers(position_ids);
    } else for (int b = 0; b < batch; ++b) {
        int position = 0;
        for (int t = 0; t < total; ++t) {
            position += mask[size_t(b) * total + t];
            if (t >= previous) positions[size_t(b) * sequence + t - previous] = mask[size_t(b) * total + t] ? position - 1 : 0;
        }
    }
    for (int position : positions) require(position >= 0 && position < c.max_positions, "position exceeds context capacity");
    graph::RecordingGuard inference;
    std::vector<float> cosines(ids.size() * c.rotary_dimension), sines(cosines.size());
    for (size_t row = 0; row < ids.size(); ++row) for (int d = 0; d < c.rotary_dimension / 2; ++d) {
        const float angle = positions[row] * inverse_frequencies_[d];
        cosines[row * c.rotary_dimension + d] = cosines[row * c.rotary_dimension + d + c.rotary_dimension / 2] = angle;
        sines[row * c.rotary_dimension + d] = sines[row * c.rotary_dimension + d + c.rotary_dimension / 2] = angle;
    }
    require(cosines.size() <= INT_MAX, "rotary table exceeds indexing limits");
    if (!try_unary_math_cpu(cosines.data(), cosines.data(), int(cosines.size()), CpuMathOperation::Cos))
        for (float& value : cosines) value = std::cos(value);
    if (!try_unary_math_cpu(sines.data(), sines.data(), int(sines.size()), CpuMathOperation::Sin))
        for (float& value : sines) value = std::sin(value);
    for (float& value : cosines) value *= rotary_magnitude_;
    for (float& value : sines) value *= rotary_magnitude_;
    Tensor cosine = Tensor::from_data({batch, 1, sequence, c.rotary_dimension}, cosines, backend_, false, dtype_);
    Tensor sine = Tensor::from_data({batch, 1, sequence, c.rotary_dimension}, sines, backend_, false, dtype_);
    const float minimum = dtype_ == DType::Float16 ? -65504.f : dtype_ == DType::BFloat16 ? -3.3895313892515355e38f : std::numeric_limits<float>::lowest();
    std::vector<float> padding_values(mask.size());
    for (size_t i = 0; i < mask.size(); ++i) padding_values[i] = mask[i] ? 0.f : minimum;
    Tensor padding = Tensor::from_data({batch, 1, 1, total}, padding_values, backend_, false, dtype_);
    Tensor hidden = reshape(embedding_.embedding(Tensor::from_int_data({int(ids.size())}, ids, backend_), dtype_), {batch, sequence, c.hidden});
    const uint64_t workspace_count = std::max(layer_workspace_count_, logits == DecoderLogits::None ? size_t(0) : head_workspace_count_);
    require(workspace_count <= INT_MAX, "weight workspace exceeds indexing limits");
    DeviceTensorCopies cosine_copies(cosine), sine_copies(sine), padding_copies(padding);
    std::map<int, Tensor> workspaces;
    LocalFunction workspace_for{[&](int ordinal, size_t count) {
        Tensor& value = workspaces[ordinal];
        if (count && (!value.defined() || value.numel() < count)) value = Tensor::zeros({int(count)}, backend_, false, dtype_);
        return value;
    }};
    DecoderOutput result;
    result.cache.length = total;
    LocalFunction feed_forward{[&](const Expert& weights, const Tensor& input, const Tensor& workspace) {
        Tensor gate = weights.gate.linear(input, {}, workspace), up = weights.up.linear(input, {}, workspace);
        return weights.down.linear(silu_mul(gate, up), {}, workspace);
    }};
    for (int index = 0; index < c.layers; ++index) {
        const int ordinal = placement_.layer_device(index);
        DeviceStage layer_device = placement_.enter_layer(index);
        hidden = placement_.transfer_to_layer(hidden, index);
        const Tensor& cosine = cosine_copies.get(ordinal);
        const Tensor& sine = sine_copies.get(ordinal);
        const Tensor& padding = padding_copies.get(ordinal);
        Tensor workspace = workspace_for(ordinal, placement_.distributed() ? layer_workspace_count_ : workspace_count);
        const DeepSeek::Layer& layer = layers_[index];
        LocalFunction emit{[&](const std::string& name, const Tensor& value) { if (trace) trace("layer." + std::to_string(index) + "." + name, value); }};
        if (capture_hidden) result.hidden_states.push_back(hidden);
        Tensor normalized = rms_norm(hidden, layer.input_norm, c.rms_epsilon, RmsNormRounding::Normalized);
        Tensor query = c.query_rank ? layer.query_b.linear(rms_norm(layer.query_a.linear(normalized, layer.query_bias, workspace),
            layer.query_norm, c.rms_epsilon, RmsNormRounding::Normalized), {}, workspace) : layer.query.linear(normalized, {}, workspace);
        query = transpose(reshape(query, {batch, sequence, c.heads, c.plain_dimension + c.rotary_dimension}), 1, 2);
        Tensor plain_query = slice(query, -1, 0, c.plain_dimension);
        Tensor rotary_query = rotate(slice(query, -1, c.plain_dimension, c.plain_dimension + c.rotary_dimension), cosine, sine);
        Tensor compressed = layer.latent_a.linear(normalized, layer.latent_bias, workspace);
        Tensor latent = rms_norm(slice(compressed, -1, 0, c.latent_cache_rank), layer.latent_norm, c.rms_epsilon, RmsNormRounding::Normalized);
        Tensor rotary_key = rotate(unsqueeze(slice(compressed, -1, c.latent_cache_rank, c.latent_cache_rank + c.rotary_dimension), 1), cosine, sine);
        LatentKVCache old;
        if (previous) {
            old = {past->layers[index].key, past->layers[index].value};
            require(old.latent.defined() && old.rotary.defined() && old.latent.device() == ordinal && old.rotary.device() == ordinal,
                    "compressed cache layer belongs to a different device");
            require(old.latent.defined() && old.latent.shape().size() == 3 && old.length() == previous, "compressed cache length differs");
        }
        LatentKVCache cached = append_latent_cache(previous ? &old : nullptr, latent, squeeze(rotary_key, 1));
        result.cache.layers.push_back({cached.latent, cached.rotary});
        emit("query_plain", plain_query); emit("query_rotary", rotary_query);
        emit("latent_cache", cached.latent); emit("rotary_cache", cached.rotary);
        Tensor attended;
        if (absorbed_attention_) {
            std::pair<Tensor, Tensor> weights = layer.absorbed_weights;
            if (!weights.first.defined()) weights = split_latent_projection(layer.latent_b.materialize(workspace), c);
            attended = latent_attention(plain_query, rotary_query, cached, weights.first, weights.second, padding, c.attention_scale(), previous);
        } else {
            Tensor expanded = transpose(reshape(layer.latent_b.linear(cached.latent, {}, workspace),
                {batch, total, c.heads, c.plain_dimension + c.value_dimension}), 1, 2);
            Tensor keys = concat({slice(expanded, -1, 0, c.plain_dimension),
                expand(unsqueeze(cached.rotary, 1), {batch, c.heads, total, c.rotary_dimension})}, -1);
            Tensor values = slice(expanded, -1, c.plain_dimension, c.plain_dimension + c.value_dimension);
            Tensor full_query = concat({plain_query, rotary_query}, -1);
            std::vector<Tensor> tiles;
            for (int start = 0; start < sequence; start += 128) {
                const int end = std::min(start + 128, sequence);
                Tensor scores = mul(matmul_transposed(slice(full_query, 2, start, end), keys), c.attention_scale());
                Tensor future = cast(triu(Tensor::ones({1, 1, end - start, total}, backend_), previous + start + 1), DType::Bool);
                Tensor excluded = add(Tensor::zeros({1, 1, end - start, total}, backend_, false, dtype_), minimum);
                tiles.push_back(matmul(attention_softmax(scores, where(future, excluded, padding)), values));
            }
            attended = tiles.size() == 1 ? tiles.front() : concat(tiles, 2);
        }
        emit("attention", attended);
        hidden = add(hidden, layer.output.linear(reshape(transpose(attended, 1, 2), {batch, sequence, c.heads * c.value_dimension}), layer.output_bias, workspace));
        normalized = rms_norm(hidden, layer.post_norm, c.rms_epsilon, RmsNormRounding::Normalized);
        Tensor feed;
        if (c.sparse_layer(index)) {
            Tensor flat = reshape(normalized, {batch * sequence, c.hidden});
            ExpertRoutes routes = route_experts(linear(cast(flat, DType::Float32), layer.router), c.routing, layer.correction);
            emit("expert_indices", routes.indices); emit("expert_weights", routes.weights);
            feed = evaluate_experts(flat, routes, [&](int id, const Tensor& selected) { return feed_forward(layer.experts[id], selected, workspace); });
            if (c.shared_experts) feed = add(feed, feed_forward(layer.shared, flat, workspace));
            feed = reshape(feed, hidden.shape());
        } else feed = feed_forward(layer.dense, normalized, workspace);
        hidden = add(hidden, feed);
        emit("output", hidden);
    }
    DeviceStage output_device = placement_.enter_output();
    hidden = hidden.to_device(placement_.output_device());
    Tensor workspace = workspace_for(placement_.output_device(), logits == DecoderLogits::None ? 0 : head_workspace_count_);
    if (capture_hidden || logits != DecoderLogits::None) hidden = rms_norm(hidden, norm_, c.rms_epsilon, RmsNormRounding::Normalized);
    if (capture_hidden) result.hidden_states.push_back(hidden);
    if (logits == DecoderLogits::Last && sequence > 1) hidden = slice(hidden, 1, sequence - 1, sequence);
    if (logits != DecoderLogits::None) result.logits = cast(head_.linear(hidden, {}, workspace), DType::Float32);
    if (trace && result.logits.defined()) trace("logits", result.logits);
    if (result.logits.defined()) result.logits = result.logits.to_device(placement_.input_device());
    return result;
}
} // namespace elman
