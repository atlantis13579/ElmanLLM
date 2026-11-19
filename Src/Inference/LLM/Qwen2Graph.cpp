#include "Inference/LLM/Qwen2Graph.h"
#include "Core/Memory.h"
#include <algorithm>
#include <limits>
#include <numeric>
#include <stdexcept>

namespace elman {
namespace {
void require(bool valid, const char* message) { if (!valid) throw std::invalid_argument(message); }
}
Qwen2Graph::Qwen2Graph(const Qwen2& model, int capacity, const DecoderCache& prefix)
    : model_(model), capacity_(capacity), length_(prefix.length) {
    require(model.backend() == Backend::CUDA && model.execution_stages() == 1 && model.tensor_parallel_size() == 1,
        "Whole-model CUDA Graph requires a single CUDA device");
    require(model.tiled_attention_enabled(), "Whole-model CUDA Graph requires tiled attention");
    require(current_device(Backend::CUDA) == model.input_device(), "Construct the graph on the model input device");
    const auto& c = model.config();
    require(capacity >= 2 && capacity <= c.max_positions && length_ > 0 && length_ < capacity,
        "Graph capacity must contain the nonempty prefill and at least one decode position");
    require(prefix.layers.size() == size_t(c.layers), "Graph prefix layer count differs");
    DeviceScope owner(Backend::CUDA, model.input_device());
    storage_.length = capacity - 1;
    for (const auto& layer : prefix.layers) {
        const Shape shape{1, c.kv_heads, length_, c.head_dim()};
        for (const auto* tensor : {&layer.key, &layer.value})
            require(tensor->defined() && tensor->shape() == shape && tensor->backend() == Backend::CUDA &&
                tensor->device() == model.input_device() && tensor->dtype() == model.dtype(), "Graph prefix cache differs from model");
        storage_.layers.push_back({Tensor::zeros({1, c.kv_heads, capacity - 1, c.head_dim()}, Backend::CUDA, false, model.dtype()),
                                  Tensor::zeros({1, c.kv_heads, capacity - 1, c.head_dim()}, Backend::CUDA, false, model.dtype())});
    }
    cache_index_ = Tensor::from_int_data({1}, {0}, Backend::CUDA);
    auto token = Tensor::from_int_data({1, 1}, {0});
    std::vector<int32_t> mask(size_t(capacity), 0); mask.back() = 1;
    auto attention = Tensor::from_int_data({1, capacity}, mask);
    auto position = Tensor::from_int_data({1, 1}, {0});
    graph_.record([&] {
        auto output = model.forward_impl(token, attention, position, &storage_, nullptr, true, DecoderLogits::Last, {}, &inputs_);
        logits_ = output.logits;
        output_cache_ = output.cache;
        hidden_states_ = output.hidden_states;
        for (int i = 0; i < c.layers; ++i) {
            auto key = slice(output.cache.layers[i].key, 2, capacity - 1, capacity);
            auto value = slice(output.cache.layers[i].value, 2, capacity - 1, capacity);
            storage_.layers[i].key.assign_indexed(2, cache_index_, key);
            storage_.layers[i].value.assign_indexed(2, cache_index_, value);
        }
    });
    graph_.compile();
    // Capture uses a dummy token. Populate the committed prefix only after all
    // warmup work has retired; masked future slots never contribute to attention.
    std::vector<int32_t> prefix_indices(size_t(length_), 0);
    std::iota(prefix_indices.begin(), prefix_indices.end(), 0);
    auto indices = Tensor::from_int_data({length_}, prefix_indices, Backend::CUDA);
    for (int i = 0; i < c.layers; ++i) {
        storage_.layers[i].key.assign_indexed(2, indices, prefix.layers[i].key);
        storage_.layers[i].value.assign_indexed(2, indices, prefix.layers[i].value);
    }
}
Tensor Qwen2Graph::decode(int32_t token) {
    require(token >= 0 && token < model_.config().vocabulary, "Graph token is outside the vocabulary");
    require(length_ < capacity_, "Graph context capacity is exhausted");
    DeviceScope owner(Backend::CUDA, model_.input_device());
    inputs_.indices.assign_int({token});
    const auto rotary = model_.rotary_values({length_});
    inputs_.cosine.assign(rotary.first); inputs_.sine.assign(rotary.second);
    const float minimum = model_.dtype() == DType::Float16 ? -65504.f : model_.dtype() == DType::BFloat16 ?
        -3.3895313892515355e38f : std::numeric_limits<float>::lowest();
    std::vector<float> mask(size_t(capacity_), minimum);
    std::fill(mask.begin(), mask.begin() + length_, 0.f); mask.back() = 0.f;
    if (inputs_.padding.defined()) inputs_.padding.assign(mask);
    if (inputs_.causal.defined()) inputs_.causal.assign(mask);
    // The final token does not need to be retained after capacity is exhausted.
    cache_index_.assign_int({std::min(length_, capacity_ - 2)});
    graph_.replay();
    ++length_;
    return logits_;
}
DecoderCache Qwen2Graph::cache_snapshot() const {
    DeviceScope owner(Backend::CUDA, model_.input_device());
    if (length_ == capacity_) return output_cache_;
    DecoderCache result;
    result.length = length_;
    for (const auto& layer : storage_.layers)
        result.layers.push_back({slice(layer.key, 2, 0, length_), slice(layer.value, 2, 0, length_)});
    return result;
}
size_t Qwen2Graph::memory_bound(const DecoderConfig& c, DType dtype, int capacity, size_t workspace) {
    require(capacity >= 2 && c.layers > 0 && c.heads > 0 && c.kv_heads > 0 && is_compute_float(dtype), "Invalid graph memory geometry");
    const auto mul = [](size_t a, size_t b) { return checked_memory_mul(a, b); };
    const auto add = [](size_t a, size_t b) { return checked_memory_add(a, b); };
    // Retain intermediates from every layer instead of relying on eager liveness.
    // Include old/new caches, attention scores, all fusion alternatives, full
    // precision casts and per-command allocation alignment/metadata headroom.
    // The fixed past and concatenated present each retain a K/V pair in the
    // model dtype. Tiled attention consumes these directly without FP32 copies.
    size_t layer = mul(mul(4, dtype_storage_bytes(dtype, 1)), mul(size_t(c.kv_heads) * c.head_dim(), size_t(capacity)));
    layer = add(layer, mul(32, mul(size_t(c.heads), size_t(capacity))));
    layer = add(layer, mul(128, add(size_t(c.hidden), size_t(c.intermediate))));
    size_t bytes = add(mul(size_t(c.layers), layer), mul(32, size_t(c.vocabulary)));
    bytes = add(bytes, workspace);
    bytes = add(bytes, mul(size_t(c.layers), 1024 * 1024));
    return add(bytes, 8 * 1024 * 1024);
}
} // namespace elman
