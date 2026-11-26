#include "Models/LLM/Qwen35.h"

#include "Core/Device.h"
#include "Core/Graph.h"
#include "Core/TensorDispatch.h"
#include "Inference/LLM/ProjectionWeight.h"
#include "Kernals/recurrent_ops.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace elman {
namespace {
void require(bool condition, const std::string& message) {
    if (!condition) throw std::invalid_argument("Qwen3.5: " + message);
}
Tensor stable_softplus(const Tensor& input) {
    const auto zero = Tensor::zeros({}, input.backend());
    return add(maximum(input, zero), log(add(exp(mul(abs(input), -1)), 1)));
}
Tensor split_heads(const Tensor& input, int heads, int width) {
    return transpose(reshape(input, {input.shape()[0], input.shape()[1], heads, width}), 1, 2);
}
}

Qwen35Config Qwen35Config::from_gguf(const GgufFile& file) {
    require(file.strings().at("general.architecture") == "qwen35", "expected qwen35 architecture");
    auto integer = [&](const std::string& key, int fallback = -1) {
        const auto item = file.numbers().find("qwen35." + key);
        if (item == file.numbers().end() && fallback >= 0) return fallback;
        require(item != file.numbers().end() && std::isfinite(item->second) && item->second >= 0 &&
            item->second <= std::numeric_limits<int>::max() && std::floor(item->second) == item->second, "invalid metadata " + key);
        return int(item->second);
    };
    Qwen35Config c;
    c.hidden = integer("embedding_length"); c.intermediate = integer("feed_forward_length");
    c.layers = integer("block_count") - integer("nextn_predict_layers", 0);
    c.heads = integer("attention.head_count"); c.kv_heads = integer("attention.head_count_kv");
    c.max_positions = integer("context_length"); c.rotary_dimensions = integer("rope.dimension_count");
    c.convolution_kernel = integer("ssm.conv_kernel"); c.key_heads = integer("ssm.group_count");
    c.value_heads = integer("ssm.time_step_rank"); c.key_width = integer("ssm.state_size");
    require(c.hidden > 0 && c.intermediate > 0 && c.layers > 0 && c.layers <= 256 && c.heads > 0 && c.kv_heads > 0 &&
        c.max_positions > 0 && c.key_heads > 0 && c.value_heads > 0 && c.key_width > 0, "dimensions must be positive");
    const int inner = integer("ssm.inner_size");
    require(inner > 0 && inner % c.value_heads == 0, "invalid recurrent value width"); c.value_width = inner / c.value_heads;
    require(c.hidden % c.heads == 0 && c.heads % c.kv_heads == 0 && c.value_heads % c.key_heads == 0 &&
        integer("attention.key_length", c.head_dim()) == c.head_dim() && integer("attention.value_length", c.head_dim()) == c.head_dim(),
        "unsupported attention head dimensions");
    require(c.rotary_dimensions > 0 && c.rotary_dimensions % 2 == 0 && c.rotary_dimensions <= c.head_dim() &&
        c.convolution_kernel > 0 && c.convolution_kernel <= 64, "invalid rotary or convolution dimensions");
    c.rope_theta = float(file.numbers().at("qwen35.rope.freq_base"));
    c.rms_epsilon = float(file.numbers().at("qwen35.attention.layer_norm_rms_epsilon"));
    require(std::isfinite(c.rope_theta) && c.rope_theta > 0 && std::isfinite(c.rms_epsilon) && c.rms_epsilon > 0,
        "invalid normalization or rotary constants");
    const auto& embedding = file.tensors().at("token_embd.weight").shape;
    require(embedding.size() == 2 && embedding[1] == c.hidden, "invalid token embedding"); c.vocabulary = embedding[0];
    c.tied_embeddings = file.tensors().count("output.weight") == 0;
    const int interval = integer("full_attention_interval", 4);
    require(interval > 0, "full attention interval must be positive");
    const auto kinds = file.number_arrays().find("qwen35.attention.recurrent_layers");
    if (kinds != file.number_arrays().end()) require(kinds->second.size() >= size_t(c.layers), "missing recurrent layer flags");
    for (int i = 0; i < c.layers; ++i) c.recurrent.push_back(kinds == file.number_arrays().end() ?
        (i + 1) % interval != 0 : kinds->second[i] != 0);
    c.recurrent_layers = int(std::count(c.recurrent.begin(), c.recurrent.end(), true));
    c.recurrent_state_elements = size_t(c.recurrent_layers) * c.value_heads * c.key_width * c.value_width;
    c.recurrent_history_elements = size_t(c.recurrent_layers) * (c.convolution_kernel - 1) *
        (2 * size_t(c.key_heads) * c.key_width + size_t(c.value_heads) * c.value_width);
    return c;
}

struct Qwen35::Impl {
    struct Layer {
        Tensor input_norm, post_norm, query_norm, key_norm, conv, decay, dt, recurrent_norm;
        ProjectionWeight query, key, value, gate, alpha, beta, output, up, ffn_gate, down;
    };
    Qwen35Config config;
    Backend backend;
    DType dtype;
    std::vector<int> devices;
    size_t parameters = 0, layer_workspace = 0, head_workspace = 0;
    bool quantized = false;
    ProjectionWeight embedding, head;
    Tensor norm;
    std::vector<Layer> layers;

    Impl(const std::filesystem::path& location, Backend target, DType precision)
        : backend(target), dtype(precision), devices{current_device(target)} {
        require(is_compute_float(dtype), "weights require a floating compute dtype");
        const auto path = find_gguf_model(location); require(!path.empty(), "GGUF model was not found");
        const GgufFile file(path); config = Qwen35Config::from_gguf(file); const auto& c = config;
        auto check_shape = [&](const std::string& name, const Shape& shape) {
            require(file.tensors().at(name).shape == shape, "invalid weight shape for " + name);
            parameters += shape_element_count(shape);
        };
        auto dense = [&](const std::string& name, const Shape& shape, DType type) {
            check_shape(name, shape); return cast(file.load(name, backend), type);
        };
        auto projection = [&](const std::string& name, int rows, int columns) {
            check_shape(name, {rows, columns});
            if (file.encoding_type(name) == 8) { quantized = true; return ProjectionWeight(file.load_quantized(name, backend)); }
            return ProjectionWeight(cast(file.load(name, backend), dtype));
        };
        embedding = projection("token_embd.weight", c.vocabulary, c.hidden);
        head = c.tied_embeddings ? embedding : projection("output.weight", c.vocabulary, c.hidden);
        head_workspace = head.workspace_count();
        norm = dense("output_norm.weight", {c.hidden}, DType::Float32);
        for (int i = 0; i < c.layers; ++i) {
            const std::string prefix = "blk." + std::to_string(i) + ".";
            Layer layer;
            layer.input_norm = dense(prefix + "attn_norm.weight", {c.hidden}, DType::Float32);
            layer.post_norm = dense(prefix + "post_attention_norm.weight", {c.hidden}, DType::Float32);
            if (c.recurrent[i]) {
                const int channels = 2 * c.key_heads * c.key_width + c.value_heads * c.value_width;
                layer.query = projection(prefix + "attn_qkv.weight", channels, c.hidden);
                layer.gate = projection(prefix + "attn_gate.weight", c.value_heads * c.value_width, c.hidden);
                layer.alpha = projection(prefix + "ssm_alpha.weight", c.value_heads, c.hidden);
                layer.beta = projection(prefix + "ssm_beta.weight", c.value_heads, c.hidden);
                layer.output = projection(prefix + "ssm_out.weight", c.hidden, c.value_heads * c.value_width);
                layer.conv = dense(prefix + "ssm_conv1d.weight", {channels, c.convolution_kernel}, dtype);
                layer.decay = dense(prefix + "ssm_a", {c.value_heads}, DType::Float32);
                layer.dt = dense(prefix + "ssm_dt.bias", {c.value_heads}, DType::Float32);
                layer.recurrent_norm = dense(prefix + "ssm_norm.weight", {c.value_width}, DType::Float32);
            } else {
                layer.query = projection(prefix + "attn_q.weight", 2 * c.hidden, c.hidden);
                layer.key = projection(prefix + "attn_k.weight", c.kv_heads * c.head_dim(), c.hidden);
                layer.value = projection(prefix + "attn_v.weight", c.kv_heads * c.head_dim(), c.hidden);
                layer.output = projection(prefix + "attn_output.weight", c.hidden, c.hidden);
                layer.query_norm = dense(prefix + "attn_q_norm.weight", {c.head_dim()}, DType::Float32);
                layer.key_norm = dense(prefix + "attn_k_norm.weight", {c.head_dim()}, DType::Float32);
            }
            layer.ffn_gate = projection(prefix + "ffn_gate.weight", c.intermediate, c.hidden);
            layer.up = projection(prefix + "ffn_up.weight", c.intermediate, c.hidden);
            layer.down = projection(prefix + "ffn_down.weight", c.hidden, c.intermediate);
            for (const auto* weight : {&layer.query, &layer.key, &layer.value, &layer.gate, &layer.alpha,
                &layer.beta, &layer.output, &layer.ffn_gate, &layer.up, &layer.down})
                layer_workspace = std::max(layer_workspace, weight->workspace_count());
            layers.push_back(std::move(layer));
        }
    }
};

Qwen35::Qwen35(const std::filesystem::path& path, Backend backend, DType dtype) : impl_(std::make_unique<Impl>(path, backend, dtype)) {}
Qwen35::~Qwen35() = default;
const Qwen35Config& Qwen35::config() const { return impl_->config; }
Backend Qwen35::backend() const { return impl_->backend; }
DType Qwen35::dtype() const { return impl_->dtype; }
int Qwen35::input_device() const { return impl_->devices[0]; }
const std::vector<int>& Qwen35::device_ordinals() const { return impl_->devices; }
size_t Qwen35::parameter_count() const { return impl_->parameters; }
bool Qwen35::quantized_weights() const { return impl_->quantized; }
size_t Qwen35::quantization_workspace_bytes(DecoderLogits logits) const {
    return dtype_storage_bytes(dtype(), std::max(impl_->layer_workspace, logits == DecoderLogits::None ? size_t(0) : impl_->head_workspace));
}

DecoderOutput Qwen35::forward(const Tensor& input_ids, const Tensor& attention_mask, const Tensor& position_ids,
    const DecoderCache* past, bool capture_hidden, DecoderLogits logits, const DecoderTrace& trace) const {
    const auto& c = config();
    require(input_ids.shape().size() == 2 && input_ids.shape()[0] > 0 && input_ids.shape()[1] > 0 &&
        (input_ids.dtype() == DType::Int32 || input_ids.dtype() == DType::Int64), "token IDs must have shape [batch, time]");
    const int batch = input_ids.shape()[0], time = input_ids.shape()[1], offset = past ? past->length : 0;
    require(offset >= 0 && offset <= c.max_positions && time <= c.max_positions - offset &&
        (!past || past->layers.size() == size_t(c.layers)), "invalid decoder context or cache");
    const auto token_values = cast(input_ids, DType::Int64).int64_data();
    for (int64_t id : token_values) require(id >= 0 && id < c.vocabulary, "token ID outside the vocabulary");
    if (attention_mask.defined()) {
        require(attention_mask.shape() == Shape{batch, offset + time}, "invalid attention mask shape");
        for (float value : cast(attention_mask, DType::Float32).data()) require(value == 1, "padded hybrid batches are unsupported");
    }
    if (position_ids.defined()) {
        require(position_ids.shape() == Shape{batch, time} && (position_ids.dtype() == DType::Int32 || position_ids.dtype() == DType::Int64),
            "invalid position IDs");
        const auto positions = cast(position_ids, DType::Int64).int64_data();
        for (size_t i = 0; i < positions.size(); ++i) require(positions[i] == offset + int(i % time), "hybrid positions must be contiguous");
    }
    DeviceScope owner(backend(), input_device()); graph::RecordingGuard no_grad;
    const size_t workspace_count = std::max(impl_->layer_workspace, logits == DecoderLogits::None ? size_t(0) : impl_->head_workspace);
    require(workspace_count <= size_t(std::numeric_limits<int>::max()), "weight workspace exceeds indexing range");
    const Tensor workspace = Tensor::zeros({int(workspace_count)}, backend(), false, dtype());
    const Tensor ids = reshape(cast(input_ids, DType::Int32).to(backend(), input_device()), {batch * time});
    Tensor hidden = reshape(impl_->embedding.embedding(ids, dtype()), {batch, time, c.hidden});
    const int rotary = c.rotary_dimensions;
    std::vector<float> cosines(size_t(time) * rotary), sines(cosines.size());
    for (int t = 0; t < time; ++t) for (int d = 0; d < rotary; ++d) {
        const double angle = (offset + t) * std::pow(c.rope_theta, -2. * (d % (rotary / 2)) / rotary);
        cosines[size_t(t) * rotary + d] = float(std::cos(angle)); sines[size_t(t) * rotary + d] = float(std::sin(angle));
    }
    const Tensor cosine = Tensor::from_data({1, 1, time, rotary}, cosines, backend(), false, dtype());
    const Tensor sine = Tensor::from_data({1, 1, time, rotary}, sines, backend(), false, dtype());
    auto rotate = [&](const Tensor& tensor) {
        const auto rotated = rotary_embedding(slice(tensor, 3, 0, rotary), cosine, sine);
        return rotary == c.head_dim() ? rotated : concat({rotated, slice(tensor, 3, rotary, c.head_dim())}, 3);
    };
    DecoderOutput result; result.cache.length = offset + time; result.cache.layers.resize(c.layers);
    for (int i = 0; i < c.layers; ++i) {
        const auto& layer = impl_->layers[i]; auto& next = result.cache.layers[i];
        const DecoderLayerCache* previous = past ? &past->layers[i] : nullptr;
        if (previous) for (const auto* item : {&previous->key, &previous->value})
            require(item->defined() && item->backend() == backend() && item->device() == input_device(), "cache is on a different device");
        if (capture_hidden) result.hidden_states.push_back(hidden);
        auto emit = [&](const char* key, const Tensor& tensor) { if (trace) trace("layer." + std::to_string(i) + "." + key, tensor); };
        const Tensor normalized = rms_norm(hidden, layer.input_norm, c.rms_epsilon);
        emit("input_norm", normalized);
        auto project = [&](const ProjectionWeight& weight) { return weight.linear(normalized, {}, workspace); };
        Tensor attended;
        if (c.recurrent[i]) {
            const auto convolution = causal_depthwise_conv1d(project(layer.query), layer.conv, previous ? previous->key : Tensor{});
            next.key = convolution.state;
            const Tensor mixed = swish(convolution.values);
            const int key_size = c.key_heads * c.key_width, value_size = c.value_heads * c.value_width;
            const Tensor q = reshape(slice(mixed, 2, 0, key_size), {batch, time, c.key_heads, c.key_width});
            const Tensor k = reshape(slice(mixed, 2, key_size, 2 * key_size), {batch, time, c.key_heads, c.key_width});
            const Tensor v = reshape(slice(mixed, 2, 2 * key_size, 2 * key_size + value_size), {batch, time, c.value_heads, c.value_width});
            const Tensor decay = mul(layer.decay, stable_softplus(add(cast(project(layer.alpha), DType::Float32), layer.dt)));
            const Tensor beta = sigmoid(project(layer.beta));
            const auto recurrent = gated_delta_rule(q, k, v, decay, beta, previous ? previous->value : Tensor{});
            next.value = recurrent.state; emit("recurrent", recurrent.values);
            const Tensor gate = reshape(cast(project(layer.gate), DType::Float32), {batch, time, c.value_heads, c.value_width});
            const Tensor gated = mul(rms_norm(cast(recurrent.values, DType::Float32), layer.recurrent_norm, c.rms_epsilon), swish(gate));
            attended = layer.output.linear(reshape(cast(gated, dtype()), {batch, time, value_size}), {}, workspace);
        } else {
            const Tensor projected = reshape(project(layer.query), {batch, time, c.heads, 2 * c.head_dim()});
            Tensor q = rms_norm(slice(projected, 3, 0, c.head_dim()), layer.query_norm, c.rms_epsilon);
            q = rotate(transpose(q, 1, 2));
            Tensor k = rotate(rms_norm(split_heads(project(layer.key), c.kv_heads, c.head_dim()), layer.key_norm, c.rms_epsilon));
            Tensor v = split_heads(project(layer.value), c.kv_heads, c.head_dim());
            if (previous) {
                const Shape expected{batch, c.kv_heads, offset, c.head_dim()};
                require(previous->key.shape() == expected && previous->value.shape() == expected &&
                    previous->key.dtype() == dtype() && previous->value.dtype() == dtype(), "invalid full attention cache");
                k = concat({previous->key, k}, 2); v = concat({previous->value, v}, 2);
            }
            next = {k, v};
            const Tensor attention = tiled_attention(cast(q, DType::Float32), cast(k, DType::Float32), cast(v, DType::Float32),
                Tensor::zeros({}, backend()), std::sqrt(c.head_dim()), 64, true, offset);
            const Tensor gate = reshape(slice(projected, 3, c.head_dim(), 2 * c.head_dim()), {batch, time, c.hidden});
            const Tensor joined = cast(reshape(transpose(attention, 1, 2), {batch, time, c.hidden}), dtype());
            attended = layer.output.linear(mul(joined, sigmoid(gate)), {}, workspace);
            emit("attention", attended);
        }
        hidden = add(hidden, attended);
        const Tensor post = rms_norm(hidden, layer.post_norm, c.rms_epsilon);
        const Tensor feed_forward = silu_mul(layer.ffn_gate.linear(post, {}, workspace), layer.up.linear(post, {}, workspace));
        hidden = add(hidden, layer.down.linear(feed_forward, {}, workspace));
        emit("output", hidden);
    }
    hidden = rms_norm(hidden, impl_->norm, c.rms_epsilon);
    if (capture_hidden) result.hidden_states.push_back(hidden);
    if (logits != DecoderLogits::None) result.logits = cast(impl_->head.linear(
        logits == DecoderLogits::Last ? slice(hidden, 1, time - 1, time) : hidden, {}, workspace), DType::Float32);
    return result;
}
} // namespace elman
