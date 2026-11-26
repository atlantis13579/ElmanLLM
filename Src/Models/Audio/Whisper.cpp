#include "Models/Audio/Whisper.h"

#include <cmath>
#include <stdexcept>

#include "Core/Device.h"
#include "Core/Graph.h"
#include "IO/CTranslate2.h"
#include "Kernals/audio_ops.h"
#include "Kernals/inference_ops.h"

namespace elman {
namespace {
void require(bool condition, const std::string& message) {
    if (!condition) throw std::invalid_argument("Whisper: " + message);
}
struct Norm {
    Tensor weight, bias;
    Tensor operator()(const Tensor& input) const { return layer_norm(input, weight, bias, 1e-5); }
};
struct Linear {
    Tensor weight, bias;
    Tensor operator()(const Tensor& input) const { return linear(input, weight, bias); }
};
struct Block {
    Norm self_norm, cross_norm, ffn_norm;
    Linear qkv, self_out, cross_q, cross_kv, cross_out, ffn_in, ffn_out;
};
Tensor heads(const Tensor& value, int count) {
    const auto& s = value.shape();
    return transpose(reshape(value, {s[0], s[1], count, s[2] / count}), 1, 2);
}
Tensor attend(const Tensor& query, const WhisperAttentionCache& memory, bool causal, int offset) {
    const auto& s = query.shape();
    const Tensor q = cast(query, DType::Float32), k = cast(memory.key, DType::Float32), v = cast(memory.value, DType::Float32);
    Tensor output = tiled_attention(q, k, v, Tensor::zeros({}, q.backend()), std::sqrt(s[3]),
                                    kAttentionQueryTile, causal, offset);
    return cast(reshape(transpose(output, 1, 2), {s[0], s[2], s[1] * s[3]}), query.dtype());
}
}

struct Whisper::Impl {
    WhisperConfig config;
    Backend backend;
    DType dtype;
    int device;
    Tensor first_conv, first_bias, second_conv, second_bias, audio_positions, text_positions, embedding;
    Norm audio_norm, text_norm;
    std::vector<Block> encoder, decoder;

    Impl(const std::filesystem::path& directory, Backend target, DType precision)
        : backend(target), dtype(precision), device(current_device(target)) {
        require(is_compute_float(dtype), "weights require a floating compute dtype");
        CTranslate2File file(directory / "model.bin");
        require(file.specification() == "WhisperSpec" && file.revision() == 3,
                "expected CTranslate2 WhisperSpec revision 3");
        auto shape = [&](const std::string& key, size_t rank) {
            const auto dimensions = file.info(key).shape;
            require(dimensions.size() == rank, "invalid weight rank for " + key);
            return dimensions;
        };
        auto weight = [&](const std::string& key, const Shape& dimensions) {
            require(file.info(key).shape == dimensions, "invalid weight shape for " + key);
            return file.load_float(key, backend, dtype);
        };
        auto norm = [&](const std::string& prefix, int width) {
            return Norm{weight(prefix + "/gamma", {width}), weight(prefix + "/beta", {width})};
        };
        auto projection = [&](const std::string& prefix, int input, int output) {
            return Linear{weight(prefix + "/weight", {output, input}), weight(prefix + "/bias", {output})};
        };
        auto count_layers = [&](const std::string& prefix) {
            int count = 0;
            while (file.tensors().count(prefix + "/layer_" + std::to_string(count) + "/self_attention/linear_0/weight")) ++count;
            require(count > 0 && count <= 128, "invalid layer count for " + prefix);
            return count;
        };
        const auto conv = shape("encoder/conv1/weight", 3);
        const auto audio_pos = shape("encoder/position_encodings/encodings", 2);
        const auto text_pos = shape("decoder/position_encodings/encodings", 2);
        const auto emb = shape("decoder/embeddings/weight", 2);
        config.mel_bins = conv[1]; config.audio_width = conv[0]; config.audio_context = audio_pos[0];
        config.text_width = emb[1]; config.vocabulary = emb[0]; config.text_context = text_pos[0];
        config.audio_heads = int(cast(file.load("encoder/num_heads"), DType::Float32).data().at(0));
        config.text_heads = int(cast(file.load("decoder/num_heads"), DType::Float32).data().at(0));
        config.audio_layers = count_layers("encoder"); config.text_layers = count_layers("decoder");
        require(config.audio_heads > 0 && config.audio_width % config.audio_heads == 0 &&
                config.text_heads > 0 && config.text_width % config.text_heads == 0,
                "attention widths must be divisible by their head counts");
        first_conv = weight("encoder/conv1/weight", {config.audio_width, config.mel_bins, 3});
        first_bias = weight("encoder/conv1/bias", {config.audio_width});
        second_conv = weight("encoder/conv2/weight", {config.audio_width, config.audio_width, 3});
        second_bias = weight("encoder/conv2/bias", {config.audio_width});
        audio_positions = weight("encoder/position_encodings/encodings", {config.audio_context, config.audio_width});
        text_positions = weight("decoder/position_encodings/encodings", {config.text_context, config.text_width});
        embedding = weight("decoder/embeddings/weight", {config.vocabulary, config.text_width});
        audio_norm = norm("encoder/layer_norm", config.audio_width);
        text_norm = norm("decoder/layer_norm", config.text_width);
        for (bool is_decoder : {false, true}) {
            auto& blocks = is_decoder ? decoder : encoder;
            const int width = is_decoder ? config.text_width : config.audio_width;
            const int count = is_decoder ? config.text_layers : config.audio_layers;
            for (int i = 0; i < count; ++i) {
                const std::string prefix = std::string(is_decoder ? "decoder" : "encoder") + "/layer_" + std::to_string(i);
                Block block;
                block.self_norm = norm(prefix + "/self_attention/layer_norm", width);
                block.qkv = projection(prefix + "/self_attention/linear_0", width, 3 * width);
                block.self_out = projection(prefix + "/self_attention/linear_1", width, width);
                block.ffn_norm = norm(prefix + "/ffn/layer_norm", width);
                block.ffn_in = projection(prefix + "/ffn/linear_0", width, 4 * width);
                block.ffn_out = projection(prefix + "/ffn/linear_1", 4 * width, width);
                if (is_decoder) {
                    block.cross_norm = norm(prefix + "/attention/layer_norm", width);
                    block.cross_q = projection(prefix + "/attention/linear_0", width, width);
                    block.cross_kv = projection(prefix + "/attention/linear_1", config.audio_width, 2 * width);
                    block.cross_out = projection(prefix + "/attention/linear_2", width, width);
                }
                blocks.push_back(std::move(block));
            }
        }
    }

    Tensor self_attention(const Tensor& input, const Block& block, int head_count,
                          const WhisperAttentionCache* past, WhisperAttentionCache& next, bool causal, int offset) const {
        const int width = input.shape().back();
        const Tensor mixed = block.qkv(block.self_norm(input));
        const Tensor query = heads(slice(mixed, 2, 0, width), head_count);
        next.key = heads(slice(mixed, 2, width, 2 * width), head_count);
        next.value = heads(slice(mixed, 2, 2 * width, 3 * width), head_count);
        if (past) {
            const Shape expected{input.shape()[0], head_count, offset, width / head_count};
            require(past->key.shape() == expected && past->value.shape() == expected, "invalid self-attention cache shape");
            next.key = concat({past->key, next.key}, 2);
            next.value = concat({past->value, next.value}, 2);
        }
        return add(input, block.self_out(attend(query, next, causal, offset)));
    }
};

Whisper::Whisper(const std::filesystem::path& directory, Backend backend, DType dtype)
    : impl_(std::make_unique<Impl>(directory, backend, dtype)) {}
Whisper::~Whisper() = default;
const WhisperConfig& Whisper::config() const { return impl_->config; }
Backend Whisper::backend() const { return impl_->backend; }
DType Whisper::dtype() const { return impl_->dtype; }
int Whisper::device() const { return impl_->device; }

Tensor Whisper::encode(const Tensor& mel, const std::function<bool(int, int)>& progress) const {
    const auto& c = config();
    require(mel.shape() == Shape{c.mel_bins, 2 * c.audio_context} && mel.backend() == backend() && mel.device() == device(),
            "encoder requires one complete mel window on the model device");
    DeviceScope owner(backend(), device());
    graph::RecordingGuard no_grad;
    Tensor value = cast(reshape(mel, {1, c.mel_bins, 2 * c.audio_context}), dtype());
    value = exact_gelu(conv1d(value, impl_->first_conv, impl_->first_bias, 1, 1));
    value = exact_gelu(conv1d(value, impl_->second_conv, impl_->second_bias, 2, 1));
    value = add(transpose(value, 1, 2), impl_->audio_positions);
    for (int i = 0; i < c.audio_layers; ++i) {
        if (progress && !progress(i, c.audio_layers)) throw std::runtime_error("Whisper encoding cancelled");
        const Block& block = impl_->encoder[i];
        WhisperAttentionCache cache;
        value = impl_->self_attention(value, block, c.audio_heads, nullptr, cache, false, 0);
        value = add(value, block.ffn_out(exact_gelu(block.ffn_in(block.ffn_norm(value)))));
    }
    return impl_->audio_norm(value);
}

WhisperMemory Whisper::prepare_memory(const Tensor& encoded) const {
    const auto& c = config();
    require(encoded.shape() == Shape{1, c.audio_context, c.audio_width} && encoded.backend() == backend() &&
                encoded.device() == device() && encoded.dtype() == dtype(), "invalid encoded memory");
    DeviceScope owner(backend(), device());
    graph::RecordingGuard no_grad;
    WhisperMemory memory;
    for (const auto& block : impl_->decoder) {
        const Tensor mixed = block.cross_kv(encoded);
        memory.layers.push_back({heads(slice(mixed, 2, 0, c.text_width), c.text_heads),
                                 heads(slice(mixed, 2, c.text_width, 2 * c.text_width), c.text_heads)});
    }
    return memory;
}

WhisperOutput Whisper::decode(const std::vector<int32_t>& tokens, const WhisperMemory& memory, const WhisperCache* past,
                              const std::vector<std::pair<int, int>>& alignment_heads) const {
    const auto& c = config();
    const int offset = past ? past->length : 0;
    require(offset >= 0 && offset < c.text_context && !tokens.empty() && tokens.size() <= size_t(c.text_context - offset),
            "decoder token count exceeds the context limit");
    require(memory.layers.size() == size_t(c.text_layers) && (!past || past->layers.size() == size_t(c.text_layers)),
            "decoder cache layer count does not match the model");
    for (int32_t token : tokens) require(token >= 0 && token < c.vocabulary, "token ID is outside the vocabulary");
    for (const auto& head : alignment_heads)
        require(head.first >= 0 && head.first < c.text_layers && head.second >= 0 && head.second < c.text_heads,
                "alignment head is outside the model");
    DeviceScope owner(backend(), device());
    graph::RecordingGuard no_grad;
    const int count = int(tokens.size());
    const Tensor ids = Tensor::from_int_data({count}, tokens, backend());
    Tensor value = reshape(index_select(impl_->embedding, 0, ids), {1, count, c.text_width});
    value = add(value, slice(impl_->text_positions, 0, offset, offset + count));
    WhisperOutput result;
    result.cache.length = offset + count;
    result.cache.layers.resize(c.text_layers);
    std::vector<Tensor> alignment(alignment_heads.size());
    for (int i = 0; i < c.text_layers; ++i) {
        const Block& block = impl_->decoder[i];
        value = impl_->self_attention(value, block, c.text_heads, past ? &past->layers[i] : nullptr,
                                     result.cache.layers[i], true, offset);
        const auto& cross = memory.layers[i];
        const Shape expected{1, c.text_heads, c.audio_context, c.text_width / c.text_heads};
        require(cross.key.shape() == expected && cross.value.shape() == expected, "invalid cross-attention cache shape");
        const Tensor query = heads(block.cross_q(block.cross_norm(value)), c.text_heads);
        for (size_t h = 0; h < alignment_heads.size(); ++h) if (alignment_heads[h].first == i) {
            const int head = alignment_heads[h].second;
            const auto q = cast(slice(query, 1, head, head + 1), DType::Float32);
            const auto k = cast(slice(cross.key, 1, head, head + 1), DType::Float32);
            alignment[h] = reshape(div(matmul(q, transpose(k, 2, 3)), std::sqrt(c.text_width / c.text_heads)),
                                   {1, count, c.audio_context});
        }
        value = add(value, block.cross_out(attend(query, cross, false, 0)));
        value = add(value, block.ffn_out(exact_gelu(block.ffn_in(block.ffn_norm(value)))));
    }
    result.logits = cast(linear(impl_->text_norm(slice(value, 1, count - 1, count)), impl_->embedding), DType::Float32);
    if (!alignment.empty()) result.alignment = concat(alignment, 0);
    return result;
}

} // namespace elman
